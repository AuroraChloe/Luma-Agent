"""Realtime DashScope speech recognition bridge for the web client.

The browser sends mono PCM frames over a WebSocket.  DashScope's realtime
SDK runs its callback on a worker thread, so callback events are marshalled
back into the FastAPI event loop before being sent to the browser.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("uvicorn.error")

try:
    from dashscope.audio.asr import RecognitionCallback as _DashScopeRecognitionCallback
except ImportError:
    class _DashScopeRecognitionCallback:
        pass


def _valid_api_key(websocket: WebSocket) -> bool:
    """Accept a service key from the WebSocket header or query string.

    The open-source core is intentionally stateless with respect to browser
    sessions. Deployments use their own gateway or API key layer instead.
    """
    expected = os.getenv("CORE_API_KEY", "").strip()
    if not expected:
        return True
    authorization = websocket.headers.get("authorization", "")
    provided = authorization.removeprefix("Bearer ").strip() or websocket.query_params.get("api_key", "")
    return provided == expected


def _put_threadsafe(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, event: dict[str, Any]) -> None:
    loop.call_soon_threadsafe(queue.put_nowait, event)


class _RecognitionCallback(_DashScopeRecognitionCallback):
    """Adapter for DashScope's callback interface."""

    def __init__(self, loop: asyncio.AbstractEventLoop, queue: asyncio.Queue):
        super().__init__()
        self._loop = loop
        self._queue = queue

    def on_open(self):
        _put_threadsafe(self._loop, self._queue, {"type": "upstream_ready"})

    def on_event(self, result):
        sentence = result.get_sentence() or {}
        if isinstance(sentence, dict):
            if sentence.get("heartbeat"):
                return
            text = str(sentence.get("text") or "").strip()
            sentence_id = sentence.get("sentence_id")
            begin_time = sentence.get("begin_time")
            end_time = sentence.get("end_time")
            try:
                from dashscope.audio.asr import RecognitionResult

                is_final = bool(RecognitionResult.is_sentence_end(sentence))
            except Exception:
                is_final = False
        else:
            text = str(sentence).strip()
            sentence_id = None
            begin_time = None
            end_time = None
            is_final = False
        if text:
            logger.debug(
                "asr transcript sentence_id=%s final=%s chars=%d begin_ms=%s end_ms=%s",
                sentence_id,
                is_final,
                len(text),
                begin_time,
                end_time,
            )
            _put_threadsafe(
                self._loop,
                self._queue,
                {
                    "type": "transcript",
                    "text": text,
                    "is_final": is_final,
                    "sentence_id": sentence_id,
                    "begin_time": begin_time,
                    "end_time": end_time,
                },
            )

    def on_complete(self):
        _put_threadsafe(self._loop, self._queue, {"type": "upstream_complete"})

    def on_error(self, message):
        if isinstance(message, str):
            error_message = message
        else:
            error_message = (
                getattr(message, "message", None)
                or getattr(message, "error_message", None)
                or "上游语音识别服务异常"
            )
        _put_threadsafe(
            self._loop,
            self._queue,
            {"type": "error", "message": str(error_message)},
        )

    def on_close(self):
        _put_threadsafe(self._loop, self._queue, {"type": "upstream_closed"})


def _configure_dashscope():
    import dashscope

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    dashscope.api_key = api_key
    if workspace_id:
        dashscope.base_websocket_api_url = (
            f"wss://{workspace_id}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
        )
    return dashscope


async def run_realtime_asr(websocket: WebSocket) -> None:
    await websocket.accept()

    if not _valid_api_key(websocket):
        await websocket.send_json({"type": "error", "message": "Invalid API key"})
        await websocket.close(code=1008)
        return

    try:
        dashscope = _configure_dashscope()
        from dashscope.audio.asr import Recognition

        model = os.getenv("DASHSCOPE_ASR_MODEL", "fun-asr-realtime").strip()
        sample_rate = int(os.getenv("DASHSCOPE_ASR_SAMPLE_RATE", "16000"))
        workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID", "").strip()
        loop = asyncio.get_running_loop()
        events: asyncio.Queue = asyncio.Queue()
        callback = _RecognitionCallback(loop, events)
        recognition = Recognition(
            model=model,
            format="pcm",
            sample_rate=sample_rate,
            workspace=workspace_id or None,
            language_hints=["zh", "en"],
            semantic_punctuation_enabled=False,
            callback=callback,
        )
        await asyncio.to_thread(recognition.start)
        await websocket.send_json({"type": "ready", "model": model, "sample_rate": sample_rate})
        logger.info("asr session ready model=%s sample_rate=%s", model, sample_rate)
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": f"语音服务初始化失败：{exc}"})
        await websocket.close(code=1011)
        return

    stopped = False
    session_started_at = time.monotonic()
    first_audio_at = None
    first_transcript_at = None
    audio_frames_received = 0
    audio_bytes_received = 0
    audio_queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    audio_sender_done = asyncio.Event()

    async def send_audio_frames():
        try:
            while True:
                frame = await audio_queue.get()
                if frame is None:
                    return
                # DashScope only enqueues the frame here; no blocking I/O occurs.
                recognition.send_audio_frame(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _put_threadsafe(loop, events, {"type": "error", "message": f"音频发送失败：{exc}"})
        finally:
            audio_sender_done.set()

    async def receive_audio():
        nonlocal stopped, first_audio_at, audio_frames_received, audio_bytes_received
        try:
            while True:
                packet = await websocket.receive()
                if packet.get("bytes") is not None:
                    if not stopped:
                        audio_frames_received += 1
                        audio_bytes_received += len(packet["bytes"])
                        if first_audio_at is None:
                            first_audio_at = time.monotonic()
                            logger.info(
                                "asr first audio delay_ms=%d",
                                (first_audio_at - session_started_at) * 1000,
                            )
                        await audio_queue.put(packet["bytes"])
                    continue
                text = packet.get("text") or ""
                if not text:
                    continue
                try:
                    command = json.loads(text)
                except json.JSONDecodeError:
                    command = {"type": text}
                if command.get("type") == "stop":
                    stopped = True
                    stop_requested_at = time.monotonic()
                    logger.info(
                        "asr stop requested audio_ms=%d frames=%d bytes=%d bridge_queue=%d",
                        (stop_requested_at - session_started_at) * 1000,
                        audio_frames_received,
                        audio_bytes_received,
                        audio_queue.qsize(),
                    )
                    await audio_queue.put(None)
                    await audio_sender_done.wait()
                    drained_at = time.monotonic()
                    logger.info(
                        "asr bridge drained drain_ms=%d",
                        (drained_at - stop_requested_at) * 1000,
                    )
                    await asyncio.to_thread(recognition.stop)
                    logger.info(
                        "asr sdk stopped stop_ms=%d last_package_delay_ms=%s",
                        (time.monotonic() - drained_at) * 1000,
                        recognition.get_last_package_delay(),
                    )
                    return
        except WebSocketDisconnect:
            stopped = True
        except Exception as exc:
            stopped = True
            _put_threadsafe(loop, events, {"type": "error", "message": f"音频接收失败：{exc}"})

    async def emit_events():
        nonlocal first_transcript_at
        while True:
            event = await events.get()
            event_type = event.get("type")
            if event_type in {"upstream_ready", "upstream_closed"}:
                continue
            if event_type == "transcript" and first_transcript_at is None:
                first_transcript_at = time.monotonic()
                logger.info(
                    "asr first transcript delay_ms=%d",
                    (first_transcript_at - session_started_at) * 1000,
                )
            if event_type == "upstream_complete":
                logger.info("asr completed total_ms=%d", (time.monotonic() - session_started_at) * 1000)
            try:
                await websocket.send_json(event)
            except (WebSocketDisconnect, RuntimeError):
                return "client_disconnected"
            if event_type in {"upstream_complete", "error"}:
                return event_type

    sender = asyncio.create_task(send_audio_frames())
    receiver = asyncio.create_task(receive_audio())
    emitter = asyncio.create_task(emit_events())
    try:
        done, _ = await asyncio.wait({receiver, emitter}, return_when=asyncio.FIRST_COMPLETED)
        if emitter in done:
            await receiver
        else:
            try:
                final_timeout = max(3.0, float(os.getenv("DASHSCOPE_ASR_FINAL_TIMEOUT", "8")))
                await asyncio.wait_for(emitter, timeout=final_timeout)
            except asyncio.TimeoutError:
                logger.warning("asr finalization timeout seconds=%s", final_timeout)
                await websocket.send_json({"type": "error", "message": "语音识别结束超时，请重试"})
    finally:
        for task in (sender, receiver, emitter):
            if not task.done():
                task.cancel()
        if not stopped:
            try:
                await asyncio.to_thread(recognition.stop)
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass
