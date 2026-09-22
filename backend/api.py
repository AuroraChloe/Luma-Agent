"""Service-key-protected HTTP transport for Luma."""

import asyncio
import base64
import json
import os
import re
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any

import requests
from fastapi import Body, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import media_store as media_storage
from asr_runtime import run_realtime_asr
from chat_protocol import latest_user_message, message_text_only
from chat_runtime import ChatRuntime
from coding_runtime import save_coding_workspace, workspace_path_for_client
from image import image_edit, image_generate
from provider import DEFAULT_LLM_BASE_URL, default_provider, llm_api_key, provider_capabilities
from rag import embedding, split
from redis_jobs import JOB_QUEUE, update_live_job_sync
from sql import (
    RAG_SCHEMA_READY,
    add_chat_interruption_message,
    add_chat_message,
    cancel_chat_job,
    create_chat_job,
    create_chat_session,
    create_rag_collection,
    create_rag_document,
    create_rag_job,
    delete_chat_session,
    delete_rag_chunks_for_document,
    ensure_core_client,
    ensure_core_project,
    get_chat_job,
    get_chat_session,
    get_or_create_video_project,
    get_rag_collection,
    get_rag_job,
    get_video_project,
    insert_rag_chunk,
    latest_chat_job_for_session,
    list_chat_messages,
    list_chat_sessions,
    list_stale_video_generation_projects,
    list_rag_collections,
    list_rag_documents,
    rag_schema_ready,
    regenerate_chat_message,
    search_rag_chunks,
    update_chat_job_status,
    update_chat_session,
    update_chat_session_title,
    update_rag_document_status,
    update_rag_job_status,
)
from token_utils import count_chat_tokens
from video_generation_runtime import reconcile_video_generation


app = FastAPI(title="Luma", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[item.strip() for item in os.getenv("LUMA_ALLOWED_ORIGINS", "*").split(",") if item.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_CHAT_MODEL = os.getenv("DEFAULT_CHAT_MODEL", "").strip()
LUMA_API_KEY = os.getenv("LUMA_API_KEY", "").strip()
LUMA_CLIENT_ID = int(os.getenv("LUMA_CLIENT_ID", "1"))
CONTEXT_MESSAGE_LIMIT = int(os.getenv("AGENT_CONTEXT_MESSAGE_LIMIT", "30"))
CONTEXT_TOKEN_BUDGET = int(os.getenv("AGENT_CONTEXT_TOKEN_BUDGET", "6000"))
RAG_DIR = Path(os.getenv("RAG_DIR", "./data/rag_files")).resolve()
RAG_MAX_UPLOAD_BYTES = int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
RAG_EMBEDDING_MAX_RETRIES = int(os.getenv("RAG_EMBEDDING_MAX_RETRIES", "2"))
RAG_DIR.mkdir(parents=True, exist_ok=True)


class ProjectContext(BaseModel):
    project_id: str = "default"


class RagScopeRequest(BaseModel):
    mode: str = "none"
    collection_id: str = ""


class SessionCreateRequest(ProjectContext):
    title: str = ""
    rag: RagScopeRequest | None = None
    chat_mode: str = "normal"
    coding_workspace_id: str = ""


class SessionUpdateRequest(BaseModel):
    title: str = ""
    rag: RagScopeRequest | None = None
    chat_mode: str | None = None
    coding_workspace_id: str | None = None


class AgentRunRequest(ProjectContext):
    session_id: str
    messages: list[dict[str, Any]]
    model: str = ""
    image_path: str | None = None
    tts_enabled: bool = False


class RegenerateRequest(BaseModel):
    content: str | None = None
    model: str = ""
    tts_enabled: bool = False


class RagSearchRequest(ProjectContext):
    query: str
    collection_id: str = ""
    top_k: int = Field(default=5, ge=1, le=20)


class TtsRequest(BaseModel):
    text: str


def _bearer_token(request: Request) -> str:
    value = request.headers.get("authorization", "")
    return value[7:].strip() if value.lower().startswith("bearer ") else ""


def _project_id(request: Request, requested: str | None = None) -> str:
    value = (requested or request.headers.get("X-Luma-Project-ID") or "default").strip()
    if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value):
        raise HTTPException(status_code=400, detail="Invalid project_id")
    return value


def require_context(request: Request, project_id: str | None = None) -> tuple[int, str]:
    if not LUMA_API_KEY or _bearer_token(request) != LUMA_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid service API key")
    client_id = LUMA_CLIENT_ID
    ensure_core_client(client_id, label=os.getenv("LUMA_CLIENT_LABEL", "default-client"))
    resolved_project = _project_id(request, project_id)
    ensure_core_project(client_id, resolved_project)
    return client_id, resolved_project


def model_items() -> list[dict[str, Any]]:
    if provider_capabilities().upstream_models:
        try:
            return default_provider().list_models()
        except Exception:
            pass
    try:
        source = Path(__file__).with_name("models.json")
        data = json.loads(source.read_text(encoding="utf-8"))
        return data.get("data", []) if isinstance(data, dict) else data
    except Exception:
        return []


def normalize_model(model: str | None) -> str:
    ids = [str(item.get("id")) for item in model_items() if isinstance(item, dict) and item.get("id")]
    if model and model in ids:
        return model
    return DEFAULT_CHAT_MODEL or (ids[0] if ids else (model or "default"))


def compact_context(messages: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    selected, used = [], 0
    for item in reversed(messages[-CONTEXT_MESSAGE_LIMIT:]):
        role = item.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            continue
        content = item.get("content", "")
        try:
            cost = count_chat_tokens([{"role": role, "content": content}], model=model) + 8
        except Exception:
            cost = len(str(content)) // 3 + 8
        if selected and used + cost > CONTEXT_TOKEN_BUDGET:
            break
        selected.append({"role": role, "content": content})
        used += cost
    return list(reversed(selected))


def build_agent_context(client_id: int, session_id: str, model: str | None = None) -> list[dict[str, Any]]:
    return compact_context(list_chat_messages(client_id, session_id, limit=CONTEXT_MESSAGE_LIMIT, ascending=False), model or "default")


def openai_error(message: str, status_code: int = 502, code: str = "upstream_error") -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": {"message": message, "type": code, "code": code}})


async def run_blocking(stage: str, func, *args, **kwargs):
    try:
        return await asyncio.to_thread(func, *args, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"{stage}: {exc}") from exc


def _upstream_url(path: str) -> str:
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL).rstrip("/")
    return f"{base_url}/{path.lstrip('/')}"


async def proxy_upstream(request: Request, path: str):
    """Forward the standard OpenAI surface without changing its payload."""
    payload = await request.body()
    wants_stream = False
    with suppress(Exception):
        wants_stream = bool(json.loads(payload.decode("utf-8")).get("stream"))
    headers = {
        "Authorization": f"Bearer {llm_api_key()}",
        "Content-Type": request.headers.get("content-type", "application/json"),
    }
    try:
        response = await asyncio.to_thread(
            requests.post,
            _upstream_url(path),
            data=payload,
            headers=headers,
            timeout=(10, int(os.getenv("OPENAI_REQUEST_TIMEOUT", "75"))),
            stream=wants_stream,
        )
    except requests.RequestException as exc:
        return openai_error(f"Upstream request failed: {exc}")
    if wants_stream:
        def content():
            try:
                yield from response.iter_content(chunk_size=8192)
            finally:
                response.close()
        headers_out = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        content_type = response.headers.get("content-type", "text/event-stream")
        return StreamingResponse(content(), status_code=response.status_code, media_type=content_type, headers=headers_out)
    try:
        return JSONResponse(status_code=response.status_code, content=response.json())
    except ValueError:
        return StreamingResponse(iter([response.content]), status_code=response.status_code, media_type=response.headers.get("content-type", "application/octet-stream"))


def image_response(result: dict[str, Any], response_format: str) -> dict[str, Any]:
    items = []
    for item in result.get("data", []) or []:
        url = item.get("url") if isinstance(item, dict) else None
        if not url:
            continue
        cached = media_storage.cache_remote_generated_image_file(url)
        output = {"url": cached["url"]}
        if response_format == "b64_json" and cached.get("path"):
            output["b64_json"] = base64.b64encode(Path(cached["path"]).read_bytes()).decode("ascii")
        if isinstance(item, dict) and item.get("revised_prompt"):
            output["revised_prompt"] = item["revised_prompt"]
        items.append(output)
    return {"created": int(time.time()), "data": items}


CHAT_RUNTIME = ChatRuntime(
    normalize_chat_model=normalize_model,
    run_blocking=run_blocking,
    build_web_session_context=build_agent_context,
    quota_exceeded_response=lambda _quota: openai_error("Usage limit exceeded", 429, "usage_limit"),
    api_token_limit_response=lambda: openai_error("Usage limit exceeded", 429, "usage_limit"),
    upstream_error_response=lambda exc, stage: openai_error(f"{stage}: {exc}"),
    openai_error_response=openai_error,
    is_rate_limit_error=lambda exc: "429" in str(exc) or "rate limit" in str(exc).lower(),
    error_stage=lambda exc: str(exc).split(":", 1)[0],
)


async def run_agent_job(job_id: str, client_id: int, payload: dict[str, Any]) -> None:
    session_id = str(payload.get("session_id") or "")
    try:
        if not update_chat_job_status(client_id, job_id, "running"):
            return
        await JOB_QUEUE.update_live_job(job_id, status="running", stage="agent", started_at=int(time.time()))
        response = await CHAT_RUNTIME.execute(
            {**payload, "_job_id": job_id}, client_id=client_id, web_chatbot=True,
            api_metered=False, persist_user_message=False,
        )
        status_code = getattr(response, "status_code", 200)
        if status_code >= 400:
            body = json.loads(getattr(response, "body", b"{}").decode("utf-8") or "{}")
            error = body.get("error", {}).get("message") or body.get("detail") or f"Request failed: {status_code}"
            add_chat_message(client_id, session_id, "assistant", str(error), model=normalize_model(payload.get("model")))
            update_chat_job_status(client_id, job_id, "failed", error=str(error))
            await JOB_QUEUE.finish_live_job(job_id, client_id=client_id, session_id=session_id, status="failed", error=str(error))
            return
        payload_body = json.loads(getattr(response, "body", b"{}").decode("utf-8") or "{}")
        update_chat_job_status(client_id, job_id, "completed", usage=payload_body.get("usage"))
        await JOB_QUEUE.finish_live_job(job_id, client_id=client_id, session_id=session_id, status="completed", usage=payload_body.get("usage"))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = f"Agent run failed: {exc}"
        update_chat_job_status(client_id, job_id, "failed", error=error)
        await JOB_QUEUE.finish_live_job(job_id, client_id=client_id, session_id=session_id, status="failed", error=error)


def read_rag_file(path: str, ext: str) -> str:
    if ext in {"txt", "md"}:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    if ext == "pdf":
        import fitz
        document = fitz.open(path)
        try:
            return "\n\n".join(page.get_text("text") for page in document)
        finally:
            document.close()
    if ext == "docx":
        from docx import Document
        document = Document(path)
        return "\n\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text.strip())
    raise ValueError("Unsupported file type")


async def run_rag_job(job_id: str, client_id: int, payload: dict[str, Any]) -> None:
    document_id, collection_id = payload["document_id"], payload["collection_id"]
    try:
        update_rag_job_status(client_id, job_id, "running", stage="parsing")
        await JOB_QUEUE.update_live_job(job_id, status="running", stage="parsing", started_at=int(time.time()))
        content = await asyncio.to_thread(read_rag_file, payload["path"], payload["ext"])
        chunks = [item for item in split(content) if str(item.get("content") if isinstance(item, dict) else item).strip()]
        if not chunks:
            raise ValueError("No indexable text found")
        delete_rag_chunks_for_document(client_id, document_id)
        for index, item in enumerate(chunks):
            text = str(item.get("content") if isinstance(item, dict) else item).strip()
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            vector = await asyncio.to_thread(embedding, text)
            insert_rag_chunk(client_id, collection_id, document_id, index, text, vector, metadata=metadata)
            if index == len(chunks) - 1 or index % max(len(chunks) // 50, 1) == 0:
                await JOB_QUEUE.update_live_job(job_id, status="running", stage="embedding", progress_current=index + 1, progress_total=len(chunks))
        update_rag_document_status(client_id, document_id, "completed", chunk_count=len(chunks))
        job = update_rag_job_status(client_id, job_id, "completed", stage="completed", progress_current=len(chunks), progress_total=len(chunks))
        await JOB_QUEUE.finish_live_job(job_id, client_id=client_id, status=job.get("status", "completed"), usage=None)
    except Exception as exc:
        error = f"RAG ingest failed: {exc}"
        update_rag_document_status(client_id, document_id, "failed", error=error, chunk_count=0)
        update_rag_job_status(client_id, job_id, "failed", stage="failed", error=error)
        await JOB_QUEUE.finish_live_job(job_id, client_id=client_id, status="failed", error=error)


def sse_event(name: str, data: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps({'object': name, 'data': data}, ensure_ascii=False, default=str)}\n\n"


def job_events(request: Request, job_id: str, name: str, durable: dict[str, Any]) -> StreamingResponse:
    async def stream():
        yield "retry: 2000\n\n"
        async for state in JOB_QUEUE.iter_job_events(job_id):
            if await request.is_disconnected():
                return
            if state is None:
                yield ": heartbeat\n\n"
            else:
                yield sse_event(name, state)
    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.on_event("startup")
async def startup() -> None:
    await JOB_QUEUE.start({"agent": run_agent_job, "rag": run_rag_job})
    app.state.video_reconcile_task = asyncio.create_task(video_reconcile_loop())


@app.on_event("shutdown")
async def shutdown() -> None:
    task = getattr(app.state, "video_reconcile_task", None)
    if task:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
    await JOB_QUEUE.close()


async def video_reconcile_loop() -> None:
    """Keep gateway renders progressing even after the originating run ends."""
    while True:
        try:
            projects = await asyncio.to_thread(list_stale_video_generation_projects, int(time.time()) - 2)
            for project in projects:
                outcome = await asyncio.to_thread(reconcile_video_generation, project)
                if outcome and outcome.get("state") in {"completed", "failed"}:
                    await asyncio.to_thread(
                        add_chat_message,
                        int(outcome["client_id"]),
                        str(outcome["session_id"]),
                        "assistant",
                        str(outcome["content"]),
                    )
        except Exception as exc:
            print(f"video reconcile loop error: {type(exc).__name__}: {exc}", flush=True)
        await asyncio.sleep(10)


@app.get("/healthz")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.websocket("/v1/audio/asr")
async def asr(websocket: WebSocket) -> None:
    await run_realtime_asr(websocket)


@app.get("/temp_file/{file_name}")
async def serve_media(file_name: str):
    path = Path(media_storage.UPLOAD_DIR, Path(file_name).name)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(path)


@app.post("/v1/assets")
async def upload_asset(request: Request, file: UploadFile = File(...), project_id: str = Form("default")):
    require_context(request, project_id)
    path = await media_storage.save_upload_image_file(file)
    return {"object": "asset", "path": path, "url": media_storage.uploaded_image_public_url(path)}


@app.post("/v1/coding/workspaces")
async def coding_workspace(request: Request, files: list[UploadFile] = File(...), relative_paths: str = Form(""), project_id: str = Form("default")):
    client_id, _ = require_context(request, project_id)
    paths = json.loads(relative_paths) if relative_paths else []
    return {"object": "coding.workspace", "data": await save_coding_workspace(client_id, files, paths if isinstance(paths, list) else [])}


@app.post("/v1/images/generations")
async def image_generation(request: Request):
    require_context(request)
    payload = await request.json()
    if not str(payload.get("prompt") or "").strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    result = await run_blocking("image_generation", image_generate, payload["prompt"], payload.get("model") or "gpt-image-2", int(payload.get("n") or 1))
    return image_response(result, payload.get("response_format", "url"))


@app.post("/v1/images/edits")
async def image_editing(request: Request, image: UploadFile = File(...), prompt: str = Form(...), model: str = Form("gpt-image-2"), n: int = Form(1), response_format: str = Form("url")):
    require_context(request)
    path = await media_storage.save_upload_image_file(image)
    result = await run_blocking("image_edit", image_edit, path, prompt, model, n)
    return image_response(result, response_format)


@app.get("/v1/models")
async def models(request: Request):
    require_context(request)
    return {"object": "list", "data": model_items()}


@app.post("/v1/chat/completions")
async def chat_completions_proxy(request: Request):
    require_context(request)
    return await proxy_upstream(request, "chat/completions")


@app.post("/v1/responses")
async def responses_proxy(request: Request):
    require_context(request)
    return await proxy_upstream(request, "responses")


@app.post("/v1/sessions")
async def create_session(request: Request, payload: SessionCreateRequest = Body(default=SessionCreateRequest())):
    client_id, project_id = require_context(request, payload.project_id)
    return create_chat_session(client_id, payload.title, rag_mode=(payload.rag.mode if payload.rag else "none"), rag_collection_id=(payload.rag.collection_id if payload.rag else None), chat_mode=payload.chat_mode, coding_workspace_id=payload.coding_workspace_id or None, project_id=project_id)


@app.get("/v1/sessions")
async def sessions(request: Request, project_id: str | None = None):
    client_id, project_id = require_context(request, project_id)
    return {"object": "list", "data": list_chat_sessions(client_id, project_id=project_id)}


@app.get("/v1/sessions/{session_id}/messages")
async def messages(session_id: str, request: Request):
    client_id, _ = require_context(request)
    if not get_chat_session(client_id, session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"object": "list", "data": list_chat_messages(client_id, session_id)}


@app.patch("/v1/sessions/{session_id}")
async def patch_session(session_id: str, request: Request, payload: SessionUpdateRequest):
    client_id, _ = require_context(request)
    updated = update_chat_session(client_id, session_id, title=payload.title, chat_mode=payload.chat_mode, coding_workspace_id=payload.coding_workspace_id) if payload.chat_mode is not None else update_chat_session_title(client_id, session_id, payload.title)
    if not updated:
        raise HTTPException(status_code=404, detail="Session not found")
    return updated


@app.delete("/v1/sessions/{session_id}")
async def remove_session(session_id: str, request: Request):
    client_id, _ = require_context(request)
    return {"success": delete_chat_session(client_id, session_id)}


@app.post("/v1/agent/runs", status_code=202)
async def create_run(request: Request, payload: AgentRunRequest):
    client_id, _ = require_context(request, payload.project_id)
    if not get_chat_session(client_id, payload.session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    if not latest_user_message(payload.messages):
        raise HTTPException(status_code=400, detail="messages must include a user turn")
    job_id = f"job_{uuid.uuid4().hex}"
    if not await JOB_QUEUE.reserve_chat_session(client_id, payload.session_id, job_id):
        raise HTTPException(status_code=409, detail="A run is already active for this session")
    model = normalize_model(payload.model)
    try:
        last = latest_user_message(payload.messages)
        persisted = add_chat_message(client_id, payload.session_id, "user", message_text_only(last.get("content")), model=model)
        job = create_chat_job(client_id, payload.session_id, model=model, job_id=job_id)
        await JOB_QUEUE.enqueue("agent", job_id, client_id, {"session_id": payload.session_id, "messages": payload.messages, "model": model, "image_path": payload.image_path, "tts_enabled": payload.tts_enabled}, session_id=payload.session_id, model=model)
    except Exception:
        await JOB_QUEUE.release_chat_session(client_id, payload.session_id, job_id)
        raise
    return {"object": "agent.run", "data": job, "user_message": persisted}


@app.get("/v1/agent/runs/{job_id}")
async def agent_run(job_id: str, request: Request):
    client_id, _ = require_context(request)
    job = await JOB_QUEUE.get_live_job(job_id)
    if job and int(job.get("client_id") or 0) == client_id:
        return {"object": "agent.run", "data": job}
    job = get_chat_job(client_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"object": "agent.run", "data": job}


@app.get("/v1/sessions/{session_id}/job")
async def session_job(session_id: str, request: Request):
    client_id, _ = require_context(request)
    live = await JOB_QUEUE.get_session_live_job(client_id, session_id)
    if live:
        return {"object": "agent.run", "data": live}
    job = latest_chat_job_for_session(client_id, session_id)
    return {"object": "agent.run", "data": job} if job else {"object": "agent.run", "data": None}


@app.get("/v1/agent/runs/{job_id}/events")
async def agent_run_events(job_id: str, request: Request):
    client_id, _ = require_context(request)
    job = get_chat_job(client_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Run not found")
    return job_events(request, job_id, "agent.run", job)


@app.get("/v1/chat/jobs/{job_id}")
async def chat_job_alias(job_id: str, request: Request):
    return await agent_run(job_id, request)


@app.get("/v1/chat/jobs/{job_id}/events")
async def chat_job_events_alias(job_id: str, request: Request):
    return await agent_run_events(job_id, request)


@app.post("/v1/agent/runs/{job_id}/cancel")
async def cancel_run(job_id: str, request: Request):
    client_id, _ = require_context(request)
    job = cancel_chat_job(client_id, job_id)
    if not job:
        raise HTTPException(status_code=409, detail="Run is already finished")
    await JOB_QUEUE.request_cancel(job_id)
    await JOB_QUEUE.update_live_job(job_id, status="canceled", stage="canceled", completed_at=int(time.time()))
    interruption = add_chat_interruption_message(client_id, job["session_id"], model=job.get("model"))
    return {"object": "agent.run", "data": job, "interruption_message": interruption}


@app.post("/v1/chat/jobs/{job_id}/cancel")
async def cancel_job_alias(job_id: str, request: Request):
    return await cancel_run(job_id, request)


@app.post("/v1/sessions/{session_id}/messages/{message_id}/regenerate", status_code=202)
async def regenerate(session_id: str, message_id: str, request: Request, payload: RegenerateRequest):
    client_id, _ = require_context(request)
    job_id = f"job_{uuid.uuid4().hex}"
    if not await JOB_QUEUE.reserve_chat_session(client_id, session_id, job_id):
        raise HTTPException(status_code=409, detail="A run is already active")
    result = regenerate_chat_message(client_id, session_id, message_id, model=normalize_model(payload.model), content=payload.content, job_id=job_id)
    if result.get("error"):
        await JOB_QUEUE.release_chat_session(client_id, session_id, job_id)
        raise HTTPException(status_code=400, detail=result["error"])
    message, job = result["message"], result["job"]
    await JOB_QUEUE.enqueue("agent", job_id, client_id, {"session_id": session_id, "messages": [{"role": "user", "content": message["content"]}], "model": job["model"], "tts_enabled": payload.tts_enabled}, session_id=session_id, model=job["model"])
    return {"object": "agent.run", "data": job, "user_message": message}


@app.get("/v1/sessions/{session_id}/video-generation")
async def video_generation(session_id: str, request: Request):
    client_id, _ = require_context(request)
    project = get_video_project(client_id, session_id) or {}
    task = dict(project.get("project_brief", {}).get("video_task") or {})
    return {"object": "video.generation", "data": {"status": task.get("status") or project.get("status") or "idle", "stage": project.get("stage"), "gateway_job_id": task.get("gateway_job_id"), "error": task.get("error") or ""}}


@app.post("/v1/rag/files", status_code=202)
async def rag_file(request: Request, file: UploadFile = File(...), collection_id: str = Form(""), collection_name: str = Form(""), project_id: str = Form("default")):
    if not (RAG_SCHEMA_READY or rag_schema_ready()):
        raise HTTPException(status_code=503, detail="RAG storage requires pgvector")
    client_id, project_id = require_context(request, project_id)
    ext = Path(file.filename or "").suffix.lower().lstrip(".")
    if ext not in {"pdf", "docx", "txt", "md"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    content = await file.read()
    if len(content) > RAG_MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="RAG file is too large")
    collection = get_rag_collection(client_id, collection_id) if collection_id else create_rag_collection(client_id, collection_name or Path(file.filename or "knowledge").stem, project_id=project_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    path = RAG_DIR / f"{uuid.uuid4().hex}.{ext}"
    path.write_bytes(content)
    document = create_rag_document(client_id, collection["collection_id"], file.filename or path.name, str(path), file.content_type, ext)
    job = create_rag_job(client_id, collection["collection_id"], document["document_id"])
    await JOB_QUEUE.enqueue("rag", job["job_id"], client_id, {"collection_id": collection["collection_id"], "document_id": document["document_id"], "path": str(path), "ext": ext}, session_id=None)
    return {"object": "rag.job", "job": job, "collection": collection, "document": document}


@app.get("/v1/rag/collections")
async def rag_collections(request: Request, project_id: str | None = None):
    client_id, _ = require_context(request, project_id)
    return {"object": "list", "data": list_rag_collections(client_id)}


@app.get("/v1/rag/collections/{collection_id}/documents")
async def rag_documents(collection_id: str, request: Request):
    client_id, _ = require_context(request)
    if not get_rag_collection(client_id, collection_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return {"object": "list", "data": list_rag_documents(client_id, collection_id)}


@app.get("/v1/rag/jobs/{job_id}")
async def rag_job(job_id: str, request: Request):
    client_id, _ = require_context(request)
    job = get_rag_job(client_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="RAG job not found")
    return {"object": "rag.job", "data": job}


@app.get("/v1/rag/jobs/{job_id}/events")
async def rag_events(job_id: str, request: Request):
    client_id, _ = require_context(request)
    job = get_rag_job(client_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="RAG job not found")
    return job_events(request, job_id, "rag.job", job)


@app.post("/v1/rag/search")
async def rag_search(request: Request, payload: RagSearchRequest):
    client_id, _ = require_context(request, payload.project_id)
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")
    if payload.collection_id and not get_rag_collection(client_id, payload.collection_id):
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    vector = await asyncio.to_thread(embedding, payload.query, input_type="query")
    return {"object": "list", "data": search_rag_chunks(client_id, vector, payload.collection_id or None, payload.top_k)}


@app.post("/v1/audio/tts")
async def tts(request: Request, payload: TtsRequest):
    require_context(request)
    from tts_runtime import synthesize_text_to_file
    return await asyncio.to_thread(synthesize_text_to_file, payload.text)
