"""Redis-backed runtime coordination for durable PostgreSQL jobs.

PostgreSQL remains the source of truth for clients, messages, jobs and final
results. Redis owns only the execution queue, live progress, cancellation
signals and the active-job pointer for a chat session.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from contextlib import suppress
from typing import Any, AsyncIterator, Awaitable, Callable

import redis
import redis.asyncio as async_redis
from redis.exceptions import ResponseError


JobHandler = Callable[[str, int, dict[str, Any]], Awaitable[None]]

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0").strip()
STREAM_KEY = os.getenv("REDIS_JOB_STREAM", "luma:jobs").strip()
CONSUMER_GROUP = os.getenv("REDIS_JOB_GROUP", "luma-workers").strip()
STATE_TTL_SECONDS = max(int(os.getenv("REDIS_JOB_STATE_TTL_SECONDS", "3600")), 300)
ACTIVE_TTL_SECONDS = max(int(os.getenv("REDIS_SESSION_ACTIVE_TTL_SECONDS", "21600")), 600)
CANCEL_TTL_SECONDS = max(int(os.getenv("REDIS_JOB_CANCEL_TTL_SECONDS", "3600")), 300)
CLAIM_IDLE_MS = max(int(os.getenv("REDIS_JOB_CLAIM_IDLE_MS", "30000")), 10000)
HEARTBEAT_SECONDS = max(float(os.getenv("REDIS_JOB_HEARTBEAT_SECONDS", "10")), 2.0)
WORKER_CONCURRENCY = max(int(os.getenv("REDIS_JOB_WORKER_CONCURRENCY", "2")), 1)
STREAM_MAXLEN = max(int(os.getenv("REDIS_JOB_STREAM_MAXLEN", "10000")), 1000)
CANCEL_CHANNEL = "luma:jobs:cancel"
EVENT_CHANNEL_PREFIX = "luma:job:events:"
EVENT_HEARTBEAT_SECONDS = max(
    float(os.getenv("REDIS_JOB_EVENT_HEARTBEAT_SECONDS", "15")),
    5.0,
)

STATE_PREFIX = "luma:job:state:"
ACTIVE_PREFIX = "luma:session:active:"
CANCEL_PREFIX = "luma:job:cancel:"
TERMINAL_STATUSES = {"completed", "failed", "canceled"}


def _now() -> int:
    return int(time.time())


def _state_key(job_id: str) -> str:
    return f"{STATE_PREFIX}{job_id}"


def _active_key(client_id: int, session_id: str) -> str:
    return f"{ACTIVE_PREFIX}{client_id}:{session_id}"


def _cancel_key(job_id: str) -> str:
    return f"{CANCEL_PREFIX}{job_id}"


def _event_channel(job_id: str) -> str:
    return f"{EVENT_CHANNEL_PREFIX}{job_id}"


def _encode_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _state_mapping(values: dict[str, Any]) -> dict[str, str]:
    mapping = {
        str(key): _encode_value(value)
        for key, value in values.items()
        if value is not None
    }
    mapping["updated_at"] = str(_now())
    return mapping


def _decode_state(raw: dict[str, str] | None) -> dict[str, Any] | None:
    if not raw:
        return None
    result: dict[str, Any] = dict(raw)
    for key in (
        "client_id",
        "created_at",
        "started_at",
        "completed_at",
        "updated_at",
        "progress_current",
        "progress_total",
        "event_seq",
    ):
        value = result.get(key)
        if value not in (None, ""):
            try:
                result[key] = int(value)
            except (TypeError, ValueError):
                pass
    for key in ("usage",):
        value = result.get(key)
        if value:
            try:
                result[key] = json.loads(value)
            except (TypeError, ValueError):
                result[key] = None
    return result


class RedisJobQueue:
    def __init__(self) -> None:
        self.consumer_name = (
            f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        )
        self.client: async_redis.Redis | None = None
        self.sync_client: redis.Redis | None = None
        self.handlers: dict[str, JobHandler] = {}
        self.worker_tasks: list[asyncio.Task] = []
        self.listener_task: asyncio.Task | None = None
        self.active_tasks: dict[str, asyncio.Task] = {}
        self.started = False

    async def connect(self) -> None:
        if self.client is None:
            self.client = async_redis.from_url(REDIS_URL, decode_responses=True)
        await self.client.ping()
        try:
            await self.client.xgroup_create(
                STREAM_KEY,
                CONSUMER_GROUP,
                id="0-0",
                mkstream=True,
            )
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    def get_sync_client(self) -> redis.Redis:
        if self.sync_client is None:
            self.sync_client = redis.from_url(REDIS_URL, decode_responses=True)
        return self.sync_client

    async def start(self, handlers: dict[str, JobHandler]) -> None:
        if self.started:
            return
        self.handlers = dict(handlers)
        await self.connect()
        self.started = True
        self.worker_tasks = [
            asyncio.create_task(
                self._worker_loop(index),
                name=f"redis-job-worker-{index}",
            )
            for index in range(WORKER_CONCURRENCY)
        ]
        self.listener_task = asyncio.create_task(
            self._cancel_listener(),
            name="redis-job-cancel-listener",
        )
        print(
            "redis job queue started "
            f"stream={STREAM_KEY} group={CONSUMER_GROUP} "
            f"consumer={self.consumer_name} concurrency={WORKER_CONCURRENCY}",
            flush=True,
        )

    async def close(self) -> None:
        if not self.started:
            return
        self.started = False
        tasks = [
            *self.worker_tasks,
            *self.active_tasks.values(),
            *([self.listener_task] if self.listener_task else []),
        ]
        for task in tasks:
            if task and not task.done():
                task.cancel()
        for task in tasks:
            if task:
                with suppress(asyncio.CancelledError):
                    await task
        self.worker_tasks.clear()
        self.active_tasks.clear()
        self.listener_task = None
        if self.client is not None:
            await self.client.aclose()
            self.client = None
        if self.sync_client is not None:
            self.sync_client.close()
            self.sync_client = None

    async def reserve_chat_session(
        self,
        client_id: int,
        session_id: str,
        job_id: str,
    ) -> bool:
        await self.connect()
        return bool(
            await self.client.eval(
                """
                local current = redis.call('GET', KEYS[1])
                if not current then
                    redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
                    return 1
                end
                local status = redis.call('HGET', ARGV[3] .. current, 'status')
                if status == 'completed' or status == 'failed' or status == 'canceled' then
                    redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2])
                    return 1
                end
                return 0
                """,
                1,
                _active_key(client_id, session_id),
                job_id,
                ACTIVE_TTL_SECONDS,
                STATE_PREFIX,
            )
        )

    async def release_chat_session(
        self,
        client_id: int,
        session_id: str,
        job_id: str,
    ) -> None:
        await self.connect()
        await self.client.eval(
            """
            if redis.call('GET', KEYS[1]) == ARGV[1] then
                return redis.call('DEL', KEYS[1])
            end
            return 0
            """,
            1,
            _active_key(client_id, session_id),
            job_id,
        )

    async def enqueue(
        self,
        job_type: str,
        job_id: str,
        client_id: int,
        payload: dict[str, Any],
        *,
        session_id: str = "",
        model: str = "",
    ) -> str:
        await self.connect()
        state = _state_mapping(
            {
                "job_id": job_id,
                "job_type": job_type,
                "client_id": client_id,
                "session_id": session_id,
                "collection_id": payload.get("collection_id"),
                "document_id": payload.get("document_id"),
                "model": model,
                "status": "queued",
                "stage": "queued",
                "created_at": _now(),
                "event_seq": 1,
            }
        )
        pipeline = self.client.pipeline(transaction=True)
        pipeline.hset(_state_key(job_id), mapping=state)
        pipeline.expire(_state_key(job_id), STATE_TTL_SECONDS)
        pipeline.xadd(
            STREAM_KEY,
            {
                "job_id": job_id,
                "job_type": job_type,
                "client_id": str(client_id),
                "payload": json.dumps(payload, ensure_ascii=False, default=str),
            },
            maxlen=STREAM_MAXLEN,
            approximate=True,
        )
        pipeline.hgetall(_state_key(job_id))
        results = await pipeline.execute()
        stream_entry_id = str(results[-2])
        await self._publish_state(job_id, _decode_state(results[-1]))
        return stream_entry_id

    async def get_live_job(self, job_id: str) -> dict[str, Any] | None:
        await self.connect()
        return _decode_state(await self.client.hgetall(_state_key(job_id)))

    async def get_session_live_job(
        self,
        client_id: int,
        session_id: str,
    ) -> dict[str, Any] | None:
        await self.connect()
        key = _active_key(client_id, session_id)
        job_id = await self.client.get(key)
        if not job_id:
            return None
        state = await self.get_live_job(job_id)
        if not state:
            await self.release_chat_session(client_id, session_id, job_id)
            return None
        return state

    async def update_live_job(self, job_id: str, **changes: Any) -> None:
        await self.connect()
        pipeline = self.client.pipeline(transaction=True)
        pipeline.hset(
            _state_key(job_id),
            mapping=_state_mapping({"job_id": job_id, **changes}),
        )
        pipeline.hincrby(_state_key(job_id), "event_seq", 1)
        pipeline.expire(_state_key(job_id), STATE_TTL_SECONDS)
        pipeline.hgetall(_state_key(job_id))
        results = await pipeline.execute()
        await self._publish_state(job_id, _decode_state(results[-1]))

    def update_live_job_sync(self, job_id: str, **changes: Any) -> None:
        client = self.get_sync_client()
        pipeline = client.pipeline(transaction=True)
        pipeline.hset(
            _state_key(job_id),
            mapping=_state_mapping({"job_id": job_id, **changes}),
        )
        pipeline.hincrby(_state_key(job_id), "event_seq", 1)
        pipeline.expire(_state_key(job_id), STATE_TTL_SECONDS)
        pipeline.hgetall(_state_key(job_id))
        results = pipeline.execute()
        state = _decode_state(results[-1])
        if state:
            client.publish(
                _event_channel(job_id),
                json.dumps(state, ensure_ascii=False, default=str),
            )

    async def _publish_state(
        self,
        job_id: str,
        state: dict[str, Any] | None,
    ) -> None:
        if not state:
            return
        await self.client.publish(
            _event_channel(job_id),
            json.dumps(state, ensure_ascii=False, default=str),
        )

    async def iter_job_events(
        self,
        job_id: str,
    ) -> AsyncIterator[dict[str, Any] | None]:
        """Yield a current snapshot followed by Pub/Sub updates for one job.

        ``None`` is a heartbeat marker. The Redis hash is durable for the live
        state TTL; Pub/Sub is deliberately only the low-latency notification
        layer.
        """
        await self.connect()
        pubsub = self.client.pubsub()
        channel = _event_channel(job_id)
        await pubsub.subscribe(channel)
        last_sequence = -1
        last_emit_at = time.monotonic()
        try:
            snapshot = await self.get_live_job(job_id)
            if snapshot:
                try:
                    last_sequence = int(snapshot.get("event_seq") or 0)
                except (TypeError, ValueError):
                    last_sequence = 0
                yield snapshot
                last_emit_at = time.monotonic()
                if snapshot.get("status") in TERMINAL_STATUSES:
                    return

            while True:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if message:
                    try:
                        state = json.loads(str(message.get("data") or "{}"))
                    except (TypeError, ValueError):
                        state = {}
                    if not isinstance(state, dict) or not state:
                        continue
                    try:
                        sequence = int(state.get("event_seq") or 0)
                    except (TypeError, ValueError):
                        sequence = 0
                    if sequence and sequence <= last_sequence:
                        continue
                    last_sequence = max(last_sequence, sequence)
                    yield state
                    last_emit_at = time.monotonic()
                    if state.get("status") in TERMINAL_STATUSES:
                        return
                    continue

                if time.monotonic() - last_emit_at >= EVENT_HEARTBEAT_SECONDS:
                    yield None
                    last_emit_at = time.monotonic()
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    async def finish_live_job(
        self,
        job_id: str,
        *,
        client_id: int,
        session_id: str = "",
        status: str,
        error: str | None = None,
        usage: dict[str, Any] | None = None,
        **extra: Any,
    ) -> None:
        await self.update_live_job(
            job_id,
            status=status,
            stage=status,
            error=error or "",
            usage=usage,
            completed_at=_now(),
            **extra,
        )
        if session_id:
            await self.release_chat_session(client_id, session_id, job_id)

    async def request_cancel(self, job_id: str) -> None:
        await self.connect()
        pipeline = self.client.pipeline(transaction=True)
        pipeline.set(_cancel_key(job_id), "1", ex=CANCEL_TTL_SECONDS)
        pipeline.publish(CANCEL_CHANNEL, job_id)
        await pipeline.execute()

    async def is_cancel_requested(self, job_id: str) -> bool:
        await self.connect()
        return bool(await self.client.exists(_cancel_key(job_id)))

    async def _cancel_listener(self) -> None:
        pubsub = self.client.pubsub()
        await pubsub.subscribe(CANCEL_CHANNEL)
        try:
            while self.started:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )
                if not message:
                    continue
                job_id = str(message.get("data") or "")
                task = self.active_tasks.get(job_id)
                if task and not task.done():
                    task.cancel()
        finally:
            await pubsub.unsubscribe(CANCEL_CHANNEL)
            await pubsub.aclose()

    async def _heartbeat(self, entry_id: str, job_id: str) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_SECONDS)
            if await self.is_cancel_requested(job_id):
                task = self.active_tasks.get(job_id)
                if task and not task.done():
                    task.cancel()
                return
            await self.client.xclaim(
                STREAM_KEY,
                CONSUMER_GROUP,
                self.consumer_name,
                min_idle_time=0,
                message_ids=[entry_id],
                idle=0,
                justid=True,
            )

    async def _claim_stale(self) -> tuple[str, dict[str, str]] | None:
        result = await self.client.xautoclaim(
            STREAM_KEY,
            CONSUMER_GROUP,
            self.consumer_name,
            min_idle_time=CLAIM_IDLE_MS,
            start_id="0-0",
            count=1,
        )
        messages = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else []
        return messages[0] if messages else None

    async def _read_new(self) -> tuple[str, dict[str, str]] | None:
        result = await self.client.xreadgroup(
            CONSUMER_GROUP,
            self.consumer_name,
            streams={STREAM_KEY: ">"},
            count=1,
            block=5000,
        )
        if not result:
            return None
        return result[0][1][0]

    async def _process_entry(
        self,
        entry_id: str,
        fields: dict[str, str],
    ) -> None:
        job_id = str(fields.get("job_id") or "")
        job_type = str(fields.get("job_type") or "")
        try:
            client_id = int(fields.get("client_id") or 0)
            payload = json.loads(fields.get("payload") or "{}")
            handler = self.handlers[job_type]
        except Exception as exc:
            print(
                f"redis job discarded entry={entry_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            await self.client.xack(STREAM_KEY, CONSUMER_GROUP, entry_id)
            return

        if await self.is_cancel_requested(job_id):
            session_id = str(payload.get("session_id") or "")
            if session_id:
                await self.release_chat_session(client_id, session_id, job_id)
            await self.client.xack(STREAM_KEY, CONSUMER_GROUP, entry_id)
            return

        handler_task = asyncio.create_task(
            handler(job_id, client_id, payload),
            name=f"redis-job-{job_type}-{job_id}",
        )
        self.active_tasks[job_id] = handler_task
        if await self.is_cancel_requested(job_id):
            handler_task.cancel()
        heartbeat_task = asyncio.create_task(
            self._heartbeat(entry_id, job_id),
            name=f"redis-job-heartbeat-{job_id}",
        )
        should_ack = True
        try:
            await handler_task
        except asyncio.CancelledError:
            # A user cancellation has a Redis marker and may be acknowledged.
            # Shutdown cancellation must stay pending so another consumer can
            # reclaim the job after the service restarts.
            should_ack = await self.is_cancel_requested(job_id)
            if not should_ack:
                raise
        except Exception as exc:
            print(
                f"redis job handler failed job_id={job_id}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        finally:
            heartbeat_task.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat_task
            self.active_tasks.pop(job_id, None)
            if should_ack:
                await self.client.xack(STREAM_KEY, CONSUMER_GROUP, entry_id)

    async def _worker_loop(self, index: int) -> None:
        while self.started:
            try:
                item = await self._claim_stale()
                if item is None:
                    item = await self._read_new()
                if item is not None:
                    await self._process_entry(*item)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(
                    f"redis worker loop error worker={index}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
                await asyncio.sleep(1)


JOB_QUEUE = RedisJobQueue()


def update_live_job_sync(job_id: str, **changes: Any) -> None:
    """Synchronous bridge for LangGraph trace callbacks and RAG threads."""
    JOB_QUEUE.update_live_job_sync(job_id, **changes)
