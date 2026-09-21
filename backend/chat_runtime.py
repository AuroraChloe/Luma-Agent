import asyncio
import base64
import json
import mimetypes
import os
import re

from fastapi import HTTPException
from fastapi.responses import JSONResponse

from langchain_agent_runtime import run_langchain_agent
from coding_runtime import run_coding_analysis, workspace_path_for_client
from chat_protocol import (
    add_chat_usage,
    build_chat_completion_response,
    build_chat_completion_message_response,
    build_chat_completion_stream,
    chat_message_to_openai_dict,
    chat_messages_token_count,
    latest_user_message,
    message_text_only,
    text_token_count,
)
from llm import llm_chat
from media_store import uploaded_image_public_url
from rag_runtime import rag_running_usage, run_rag_chat
from redis_jobs import update_live_job_sync
from tts_runtime import synthesize_text_to_file
from video_creation_runtime import run_video_creation_chat
from sql import (
    add_chat_message,
    client_usage_available,
    ensure_chat_asset,
    get_chat_job,
    get_chat_session,
    list_chat_assets,
    record_client_token_usage,
    record_client_chat_usage,
    record_client_image_usage,
)


VISION_LLM_MODEL = os.getenv("VISION_LLM_MODEL", "moonshotai/kimi-k2.6")
VISION_LLM_BASE_URL = (os.getenv("VISION_LLM_BASE_URL") or os.getenv("NVIDIA_LLM_BASE_URL") or os.getenv("RAG_EMBEDDING_BASE_URL") or "").rstrip("/")
VISION_LLM_API_KEY = (
    os.getenv("VISION_LLM_API_KEY")
    or os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("NVIDIA_API_KEY")
)
def explicit_image_request(original_user_request=""):
    text = str(original_user_request or "")
    image_keywords = (
        r"(生成|创建|画|绘制|设计|做|制作|出|来|弄|编辑|修改|改成|换成).{0,30}"
        r"(图片|图像|照片|海报|头像|插画|壁纸|logo|图标|表情包|图)"
        r"|"
        r"(图片|图像|照片|海报|头像|插画|壁纸|logo|图标|表情包|图).{0,30}"
        r"(生成|创建|画|绘制|设计|做|制作|编辑|修改|改成|换成)"
        r"|"
        r"^\s*(画|绘制|帮我画|给我画|draw|paint)"
        r"|"
        r"^\s*(生成|创建|设计|做|制作|出一张|来一张|给我一张).{0,40}$"
        r"|"
        r"\b(generate|create|draw|paint|make|design|edit|modify|turn|convert)\b.{0,40}"
        r"\b(image|photo|picture|poster|illustration|avatar|wallpaper|logo|icon)\b"
    )
    return bool(re.search(image_keywords, text, re.IGNORECASE))

def likely_image_tool_request(original_user_request="", image_path=""):
    if explicit_image_request(original_user_request):
        return True
    if not image_path:
        return False
    return bool(re.search(
        r"(改|修改|编辑|修|修图|换成|改成|变成|去掉|去除|删除|加上|添加|调整|美化|润色|换风格|重绘)",
        str(original_user_request or ""),
        re.IGNORECASE,
    ))

def typed_user_content(user_text, image_path=None):
    text = user_text or ("请分析这张图片。" if image_path else "")
    content = [{"type": "text", "text": text}]
    if not image_path:
        return content
    image_url = uploaded_image_public_url(image_path)
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content


def local_image_data_url(image_path):
    if not image_path or not os.path.exists(image_path):
        return None
    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def vision_image_url(image_ref):
    image_ref = str(image_ref or "").strip()
    if not image_ref:
        return None
    if image_ref.startswith(("https://", "http://", "data:")):
        return image_ref
    base_url = (VISION_LLM_BASE_URL or "").lower()
    return local_image_data_url(image_ref) if "moonshot" in base_url else uploaded_image_public_url(image_ref)


def vision_user_content(user_text, image_ref=None):
    text = user_text or ("请分析这张图片。" if image_ref else "")
    content = [{"type": "text", "text": text}]
    image_url = vision_image_url(image_ref)
    if image_url:
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content


def attach_vision_image_to_messages(user_text, image_ref):
    return [{"role": "user", "content": vision_user_content(user_text, image_ref)}]


def planner_text_content(user_text, image_path=None):
    text = user_text or ""
    if not image_path:
        return text
    image_context = (
        f"当前用户上传了图片，服务器图片路径：{image_path}\n"
        "如果用户要修改、编辑、重绘这张图片，请选择 image_edit 工具；"
        "如果用户只是询问图片内容、识别图片、描述图片，请选择 none。"
    )
    return f"{text}\n\n{image_context}" if text else image_context


def attach_uploaded_image_to_messages(messages, user_text, image_path):
    content = typed_user_content(user_text, image_path)
    next_messages = [dict(item) for item in (messages or []) if isinstance(item, dict)]
    for index in range(len(next_messages) - 1, -1, -1):
        if next_messages[index].get("role") == "user":
            next_messages[index] = {**next_messages[index], "content": content}
            return next_messages
    return next_messages + [{"role": "user", "content": content}]


def attach_typed_text_to_messages(messages, user_text):
    return attach_uploaded_image_to_messages(messages, user_text, None)


def add_no_image_guard(messages):
    return [
        {
            "role": "system",
            "content": (
                "当前这次用户请求没有上传图片，也没有附带可见图片 URL。"
                "除非用户消息或历史记录明确提供了图片内容，不要使用“根据图片”“图中”“图片内容”等说法。"
            ),
        },
        *(messages or []),
    ]


def strip_uploaded_image_path(text, image_path=""):
    text = str(text or "")
    if image_path:
        text = text.replace(str(image_path), "")
    text = re.sub(r"/opt/key_college/temp_images/[^\s<>()]+?\.(?:png|jpg|jpeg|webp|gif)", "", text, flags=re.I)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def vision_chat_overrides():
    overrides = {}
    if VISION_LLM_BASE_URL:
        overrides["_provider_base_url"] = VISION_LLM_BASE_URL
    if VISION_LLM_API_KEY:
        overrides["_provider_api_key"] = VISION_LLM_API_KEY
    vision_timeout = os.getenv("VISION_LLM_TIMEOUT")
    if vision_timeout:
        try:
            overrides["_provider_timeout"] = float(vision_timeout)
        except ValueError:
            pass
    return VISION_LLM_MODEL, overrides


class ChatRuntime:
    def __init__(
        self,
        *,
        normalize_chat_model,
        run_blocking,
        build_web_session_context,
        quota_exceeded_response,
        api_token_limit_response,
        upstream_error_response,
        openai_error_response,
        is_rate_limit_error,
        error_stage,
    ):
        self.normalize_chat_model = normalize_chat_model
        self.run_blocking = run_blocking
        self.build_web_session_context = build_web_session_context
        self.quota_exceeded_response = quota_exceeded_response
        self.api_token_limit_response = api_token_limit_response
        self.upstream_error_response = upstream_error_response
        self.openai_error_response = openai_error_response
        self.is_rate_limit_error = is_rate_limit_error
        self.error_stage = error_stage

    async def execute(self, data, client_id, web_chatbot=False, api_metered=False, persist_user_message=True):
        messages = data.get("messages") or []
        model = self.normalize_chat_model(data.get("model"))
        image_path = data.get("image_path")
        client_ip = data.get("_client_ip") or ""
        session_id = data.get("session_id")
        job_id = data.get("_job_id")
        stream_requested = bool(data.get("stream", False))

        if api_metered and not client_usage_available(client_id):
            return self.api_token_limit_response()

        current_user_message = latest_user_message(messages)
        current_user_content = current_user_message.get("content", "") if current_user_message else ""
        original_user_request = strip_uploaded_image_path(message_text_only(current_user_content), image_path)
        session_enabled = bool(session_id)
        session = None
        session_assets = []

        if session_enabled:
            if client_id is None:
                raise HTTPException(status_code=401, detail="User key required for session history")
            session = get_chat_session(client_id, session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            if persist_user_message and current_user_message:
                add_chat_message(client_id, session_id, "user", original_user_request, model=model)
            if image_path:
                ensure_chat_asset(
                    client_id,
                    session_id,
                    asset_type="image_upload",
                    local_path=None,
                    public_url=uploaded_image_public_url(image_path),
                    asset_label="用户上传的原始视觉素材",
                    operation_prompt="用户上传的视觉素材",
                    make_active=True,
                )
            messages = self.build_web_session_context(client_id, session_id, model=model)
            session_assets = list_chat_assets(client_id, session_id)

        kwargs = {k: v for k, v in data.items() if k not in ("model", "messages", "image_path", "session_id", "_job_id", "_client_ip", "tts_enabled")}
        kwargs.pop("stream", None)
        kwargs.pop("stream_options", None)
        tts_enabled = bool(data.get("tts_enabled")) and web_chatbot

        rag_scope = (session or {}).get("rag") or {}
        chat_mode = (session or {}).get("chat_mode") or "normal"
        if web_chatbot and chat_mode == "coding":
            workspace_id = (session or {}).get("coding_workspace_id")
            if not workspace_id:
                raise HTTPException(status_code=400, detail="请先上传要分析的项目文件夹")
            try:
                workspace_path = workspace_path_for_client(client_id, workspace_id)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(status_code=404, detail="代码分析工作区不存在，请重新上传项目文件夹") from exc

            def coding_trace_callback(trace=None, used_tools=None, usage=None):
                if not job_id:
                    return
                update_live_job_sync(
                    job_id,
                    status="running",
                    stage="coding_agent",
                    usage={
                        **(usage or {}),
                        "trace": trace or [],
                        "used_tools": used_tools or [],
                    },
                )

            coding_result = await self.run_blocking(
                "coding_agent",
                run_coding_analysis,
                original_user_request,
                workspace_path=str(workspace_path),
                messages=messages,
                model=model,
                trace_callback=coding_trace_callback,
            )
            final_reply = coding_result.get("content") or "项目分析未生成，请稍后重试。"
            final_usage = {
                **(coding_result.get("usage") or {}),
                "trace": coding_result.get("trace") or [],
                "used_tools": coding_result.get("used_tools") or [],
            }
        elif web_chatbot and chat_mode == "video_creation":
            def video_subject_trace_callback(trace=None, used_tools=None, usage=None):
                if not job_id:
                    return
                update_live_job_sync(
                    job_id,
                    status="running",
                    stage="video_creation",
                    usage={
                        **(usage or {}),
                        "trace": trace or [],
                        "used_tools": used_tools or [],
                    },
                )

            video_result = await self.run_blocking(
                "video_creation",
                run_video_creation_chat,
                client_id,
                client_id=client_id,
                session_id=session_id,
                model=model,
                messages=messages,
                user_message=original_user_request,
                image_path=image_path,
                trace_callback=video_subject_trace_callback,
            )
            final_reply = video_result.get("content") or "视频创作流程没有生成有效回复，请重新发送一次。"
            final_usage = {
                **(video_result.get("usage") or {}),
                "trace": video_result.get("trace") or [],
                "used_tools": video_result.get("used_tools") or [],
            }
        elif web_chatbot and rag_scope.get("mode") in {"all", "collection"}:
            if job_id:
                update_live_job_sync(
                    job_id,
                    status="running",
                    stage="rag_retrieval",
                    usage=rag_running_usage(rag_scope, query=original_user_request),
                )
            rag_result = await self.run_blocking(
                "rag_chat",
                run_rag_chat,
                client_id,
                client_id=client_id,
                rag_scope=rag_scope,
                model=model,
                messages=messages,
                user_message=current_user_content,
                kwargs=kwargs,
            )
            final_reply = rag_result.get("content") or ""
            final_usage = rag_result.get("usage") or {}
        elif web_chatbot:
            final_reply, final_usage = await self._run_agent_or_plain_chat(
                model=model,
                client_id=client_id,
                session_id=session_id,
                original_user_request=original_user_request,
                messages=messages,
                kwargs=kwargs,
                image_path=image_path,
                client_ip=client_ip,
                job_id=job_id,
                assets=session_assets,
            )
        else:
            final_reply, final_usage = await self._run_plain_chat(
                model=model,
                client_id=client_id,
                messages=messages,
                kwargs=kwargs,
                preserve_tool_calls=api_metered,
            )

        if job_id and (get_chat_job(client_id, job_id) or {}).get("status") == "canceled":
            raise asyncio.CancelledError()

        if isinstance(final_reply, JSONResponse):
            return final_reply

        asset_events = []
        if isinstance(final_usage, dict):
            asset_events = list(final_usage.pop("_asset_events", []) or [])
        usage = add_chat_usage(None, final_usage)
        if isinstance(final_usage, dict):
            for key in ("trace", "used_tools"):
                if key in final_usage:
                    usage = usage or {}
                    usage[key] = final_usage[key]
        if web_chatbot:
            chat_quota = record_client_chat_usage(client_id, (usage or {}).get("total_tokens", 0))
            if not chat_quota.get("allowed"):
                return self.quota_exceeded_response(chat_quota)
        elif api_metered:
            api_quota = record_client_token_usage(client_id, (usage or {}).get("total_tokens", 0))
            print(f"api token usage client_id={client_id} model={model} usage={(usage or {}).get('total_tokens', 0)} quota={api_quota}", flush=True)

        openai_message_payload = final_reply if isinstance(final_reply, dict) and final_reply.get("_openai_message") else None
        if tts_enabled and not openai_message_payload:
            try:
                tts_result = await self.run_blocking(
                    "tts",
                    synthesize_text_to_file,
                    final_reply,
                    client_id=client_id,
                    kind="tts",
                )
                usage = usage or {}
                usage["tts_enabled"] = True
                if tts_result.get("status") == "completed":
                    usage["tts_url"] = tts_result.get("url")
                    usage["tts_model"] = tts_result.get("model")
                    usage["tts_voice"] = tts_result.get("voice")
                else:
                    usage["tts_status"] = tts_result.get("status") or "skipped"
            except Exception as exc:
                usage = usage or {}
                usage["tts_enabled"] = True
                usage["tts_status"] = "failed"
                print(f"tts post-process failed client_id={client_id}: {type(exc).__name__}: {exc}", flush=True)
        if job_id and (get_chat_job(client_id, job_id) or {}).get("status") == "canceled":
            raise asyncio.CancelledError()
        if session_enabled:
            for event in asset_events:
                if isinstance(event, dict):
                    ensure_chat_asset(
                        client_id,
                        session_id,
                        asset_type=event.get("asset_type") or "image",
                        local_path=event.get("local_path"),
                        public_url=event.get("public_url"),
                        source_asset_id=event.get("source_asset_id"),
                        asset_label=event.get("asset_label"),
                        operation_prompt=event.get("operation_prompt"),
                        make_active=True,
                    )
            if openai_message_payload:
                add_chat_message(
                    client_id,
                    session_id,
                    "assistant",
                    json.dumps(openai_message_payload.get("_openai_message") or {}, ensure_ascii=False),
                    model=model,
                )
            else:
                add_chat_message(client_id, session_id, "assistant", final_reply, model=model)
        if openai_message_payload:
            return JSONResponse(build_chat_completion_message_response(
                model,
                openai_message_payload.get("_openai_message") or {},
                usage,
                finish_reason=openai_message_payload.get("_finish_reason") or "tool_calls",
            ))
        if stream_requested:
            return build_chat_completion_stream(model, final_reply, usage)
        return JSONResponse(build_chat_completion_response(model, final_reply, usage))

    async def _run_agent_or_plain_chat(self, *, model, client_id, session_id=None, original_user_request, messages, kwargs, image_path=None, client_ip="", job_id=None, assets=None):
        agent_probe_usage = None
        needs_vision = False
        resolved_visual_assets = []
        asset_events = []
        if likely_image_tool_request(original_user_request, image_path):
            image_quota = record_client_image_usage(client_id, 1)
            if not image_quota.get("allowed"):
                return self.quota_exceeded_response(image_quota), {}

        try:
            def trace_callback(trace=None, used_tools=None, usage=None):
                if not job_id:
                    return
                update_live_job_sync(
                    job_id,
                    status="running",
                    stage="agent",
                    usage={
                        **(usage or {}),
                        "trace": trace or [],
                        "used_tools": used_tools or [],
                    },
                )

            agent_messages = messages
            if client_ip:
                ip_context = (
                    f"当前 HTTP 客户端公网 IP：{client_ip}\n"
                    "只有当用户询问当前/本地天气但没有提供地点时，weather_current 工具才可以用这个 IP 做定位。"
                )
                agent_messages = list(agent_messages or []) + [{"role": "user", "content": ip_context}]
            if image_path:
                image_context = (
                    f"当前用户上传了图片，服务器图片路径：{image_path}\n"
                    "如果用户要修改、编辑、重绘这张图片，请选择 image_edit 工具；"
                    "如果用户只是询问图片内容、识别图片、描述图片，请选择 none。"
                )
                agent_messages = list(messages or []) + [{"role": "user", "content": image_context}]

            agent_assets = list(assets or [])
            if image_path and not any(
                str(item.get("local_path") or "") == str(image_path)
                for item in agent_assets
                if isinstance(item, dict)
            ):
                agent_assets.append({
                    "asset_id": "current_upload",
                    "asset_type": "image_upload",
                    "local_path": image_path,
                    "public_url": uploaded_image_public_url(image_path),
                    "asset_label": "当前用户上传的视觉素材",
                    "operation_prompt": "用户上传的视觉素材",
                    "is_active": True,
                })

            planner_content = planner_text_content(original_user_request, image_path)
            if client_ip:
                planner_content = (
                    f"{planner_content}\n\n"
                    + (
                        f"系统上下文：当前 HTTP 客户端公网 IP 是 {client_ip}。"
                        "如果当前用户请求询问本地/当前位置/我这里/当前天气但没有地点，weather_current 可使用该 IP 做天气定位。"
                    )
                )

            # Keep web Agent execution on the native LangChain path while it
            # is being validated. Do not hide defects with the legacy runner.
            agent_result = await self.run_blocking(
                "agent",
                run_langchain_agent,
                original_user_request,
                messages=agent_messages,
                model=model,
                client_id=client_id,
                agent_client_id=client_id,
                chat_session_id=session_id,
                trace_callback=trace_callback,
                planner_content=planner_content,
                assets=agent_assets,
            )
            content = agent_result.get("content") or ""
            used_tools = agent_result.get("used_tools") or []
            agent_probe_usage = agent_result.get("usage") or None
            interpretation = agent_result.get("interpretation") or {}
            needs_vision = bool(interpretation.get("needs_vision"))
            resolved_visual_assets = agent_result.get("visual_assets") or []
            asset_events = agent_result.get("asset_events") or []
            vision_context_ready = bool(agent_result.get("vision_context_ready"))
            agent_answer_ready = bool(agent_result.get("agent_answer_ready"))
            if content and (used_tools or agent_answer_ready) and (
                not needs_vision or used_tools or vision_context_ready
            ):
                print(
                    f"agent used_tools={used_tools} text={original_user_request[:80]}",
                    flush=True,
                )
                usage = agent_result.get("usage") or {}
                if isinstance(usage, dict):
                    usage = {
                        **usage,
                        "trace": agent_result.get("trace") or [],
                        "used_tools": used_tools,
                        "_asset_events": asset_events,
                    }
                return content, usage
            print(f"agent no_tool plain_chat text={original_user_request[:80]}", flush=True)
        except Exception as exc:
            print(f"agent langchain failed without fallback: {type(exc).__name__}: {exc}", flush=True)
            raise

        vision_image_ref = image_path
        vision_source = "current_upload" if image_path else ""
        if not vision_image_ref and needs_vision and resolved_visual_assets:
            selected_asset = resolved_visual_assets[0]
            vision_image_ref = selected_asset.get("public_url") or selected_asset.get("local_path")
            vision_source = selected_asset.get("asset_id") or "session_asset"

        if vision_image_ref:
            # Keep vision calls one-shot so historical tool-call metadata never reaches the vision upstream.
            plain_messages = attach_vision_image_to_messages(original_user_request, vision_image_ref)
            plain_model, provider_overrides = vision_chat_overrides()
            kwargs = {**kwargs, **provider_overrides}
            print(
                f"vision plain_chat model={plain_model} source={vision_source} text={original_user_request[:80]}",
                flush=True,
            )
        else:
            plain_messages = attach_typed_text_to_messages(messages, original_user_request)
            plain_messages = add_no_image_guard(plain_messages)
            plain_model = model

        final_reply, final_usage = await self._run_plain_chat(
            model=plain_model,
            client_id=client_id,
            messages=plain_messages,
            kwargs=kwargs,
            initial_usage=agent_probe_usage,
        )
        if asset_events and isinstance(final_usage, dict):
            final_usage = {**final_usage, "_asset_events": asset_events}
        return final_reply, final_usage

    async def _run_plain_chat(self, *, model, client_id, messages, kwargs, initial_usage=None, preserve_tool_calls=False):
        try:
            final_output = await self.run_blocking(
                "chat",
                llm_chat,
                model=model,
                client_id=client_id,
                messages=messages,
                **kwargs,
            )
        except Exception as exc:
            if self.is_rate_limit_error(exc):
                raise HTTPException(status_code=429, detail=f"{model} 当前上游限流，请稍后重试。")
            return str(exc), {}
        choice = final_output.choices[0]
        final_message = choice.message
        if preserve_tool_calls and (
            getattr(final_message, "tool_calls", None)
            or getattr(final_message, "function_call", None)
        ):
            final_reply = {
                "_openai_message": chat_message_to_openai_dict(final_message),
                "_finish_reason": getattr(choice, "finish_reason", None) or "tool_calls",
            }
        else:
            final_reply = final_message.content
        final_usage = add_chat_usage(initial_usage, getattr(final_output, "usage", None))
        if not final_usage:
            prompt_tokens = chat_messages_token_count(messages, model=model)
            completion_tokens = text_token_count(
                final_reply.get("_openai_message") if isinstance(final_reply, dict) else final_reply,
                model=model,
            )
            final_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        return final_reply, final_usage
