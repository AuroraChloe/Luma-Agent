import json
import time
import uuid

import tiktoken
from fastapi.responses import StreamingResponse

from token_utils import count_chat_tokens


def token_encoder(model=None):
    try:
        return tiktoken.encoding_for_model(model or "gpt-4o")
    except Exception:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def text_token_count(value, model=None):
    if value is None:
        return 0
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, default=str)
    return len(token_encoder(model).encode(value))


def chat_messages_token_count(messages, model=None):
    try:
        return count_chat_tokens(messages or [], model=model or "gpt-4o")
    except Exception:
        return text_token_count(messages or [], model=model)


def usage_to_dict(usage):
    if not usage:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(mode="json", exclude_none=True)
    if isinstance(usage, dict):
        return usage
    return {
        key: getattr(usage, key)
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_tokens",
            "output_tokens",
        )
        if getattr(usage, key, None) is not None
    }


def add_chat_usage(*usage_items):
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    found = False
    for usage in usage_items:
        data = usage_to_dict(usage)
        if not data:
            continue
        found = True
        prompt_tokens = data.get("prompt_tokens", data.get("input_tokens", 0)) or 0
        completion_tokens = data.get("completion_tokens", data.get("output_tokens", 0)) or 0
        total_tokens = data.get("total_tokens") or prompt_tokens + completion_tokens
        total["prompt_tokens"] += prompt_tokens
        total["completion_tokens"] += completion_tokens
        total["total_tokens"] += total_tokens
    return total if found else None


def message_content_text(content):
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def message_text_only(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text"} and item.get("text"):
                    parts.append(str(item.get("text")))
                elif isinstance(item.get("content"), str):
                    parts.append(item.get("content"))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return message_content_text(content)


def latest_user_message(messages):
    for message in reversed(messages or []):
        if message.get("role") == "user":
            return message
    return None


def chat_message_to_openai_dict(message):
    """Normalize an SDK chat message while retaining tool-call metadata."""
    if hasattr(message, "model_dump"):
        data = message.model_dump(mode="json", exclude_none=True)
    elif isinstance(message, dict):
        data = dict(message)
    else:
        data = {
            key: getattr(message, key)
            for key in ("role", "content", "tool_calls", "function_call", "refusal")
            if getattr(message, key, None) is not None
        }
    data.setdefault("role", "assistant")
    return data


def build_chat_completion_response(model, content, usage):
    now = int(time.time())
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def build_chat_completion_message_response(model, message, usage, finish_reason="stop"):
    """Return a non-streaming OpenAI response that preserves tool-call fields."""
    now = int(time.time())
    normalized = dict(message) if isinstance(message, dict) else {"content": str(message or "")}
    normalized.setdefault("role", "assistant")
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": normalized,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def build_chat_completion_stream(model, content, usage):
    now = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    def chunk(delta, finish_reason=None, usage_data=None):
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": usage_data,
        }

    def generator():
        yield f"data: {json.dumps(chunk({'role': 'assistant'}), ensure_ascii=False)}\n\n"
        if content:
            yield f"data: {json.dumps(chunk({'content': content}), ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps(chunk({}, finish_reason='stop', usage_data=usage), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")
