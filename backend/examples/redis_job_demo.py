"""A small Redis job example that mirrors the project's chat_jobs flow.

This file is intentionally isolated from the production API. It demonstrates:
1. PostgreSQL-like durable job metadata represented by a simple job payload.
2. Redis as the short-lived queue and live status store.
3. A worker that consumes job IDs and updates progress.

Install: pip install redis
Run Redis locally: docker run --rm -p 6379:6379 redis:7-alpine
Run: python redis_job_demo.py
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any

import redis.asyncio as redis


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
QUEUE_KEY = "luma:demo:jobs"
JOB_KEY_PREFIX = "luma:demo:job:"
JOB_TTL_SECONDS = 15 * 60


def job_key(job_id: str) -> str:
    return f"{JOB_KEY_PREFIX}{job_id}"


async def create_job(client: redis.Redis, client_id: int, message: str) -> dict[str, Any]:
    """Create a job record, then put only its ID into the queue."""
    job_id = f"job_{uuid.uuid4().hex}"
    job = {
        "job_id": job_id,
        "client_id": str(client_id),
        "status": "queued",
        "stage": "queued",
        "message": message,
        "created_at": str(int(time.time())),
    }

    # Redis stores strings. In production, the durable equivalent is your
    # existing create_chat_job(...) INSERT in PostgreSQL.
    await client.set(job_key(job_id), json.dumps(job, ensure_ascii=False), ex=JOB_TTL_SECONDS)
    await client.rpush(QUEUE_KEY, job_id)
    return job


async def get_job(client: redis.Redis, job_id: str) -> dict[str, Any] | None:
    raw = await client.get(job_key(job_id))
    return json.loads(raw) if raw else None


async def update_job(client: redis.Redis, job_id: str, **changes: Any) -> None:
    job = await get_job(client, job_id)
    if not job:
        return
    job.update(changes)
    await client.set(job_key(job_id), json.dumps(job, ensure_ascii=False), ex=JOB_TTL_SECONDS)


async def worker(client: redis.Redis) -> None:
    """Consume jobs forever; one process can run multiple workers."""
    while True:
        item = await client.blpop(QUEUE_KEY, timeout=0)
        _, job_id = item
        job_id = job_id.decode() if isinstance(job_id, bytes) else job_id

        await update_job(client, job_id, status="running", stage="agent")
        await asyncio.sleep(1)
        await update_job(client, job_id, stage="tool")
        await asyncio.sleep(1)
        await update_job(
            client,
            job_id,
            status="completed",
            stage="completed",
            result="这是一个由后台 worker 完成的示例结果。",
            completed_at=str(int(time.time())),
        )


async def demo() -> None:
    client = redis.from_url(REDIS_URL, decode_responses=False)
    await client.ping()

    worker_task = asyncio.create_task(worker(client))
    job = await create_job(client, client_id=1, message="帮我测试一下 Redis job")
    print("提交后：", await get_job(client, job["job_id"]))

    while True:
        current = await get_job(client, job["job_id"])
        print("查询状态：", current)
        if current and current["status"] == "completed":
            break
        await asyncio.sleep(0.5)

    worker_task.cancel()
    await client.aclose()


if __name__ == "__main__":
    asyncio.run(demo())
