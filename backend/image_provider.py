"""Provider adapter for OpenAI Images-compatible and Qwen Image services."""

from __future__ import annotations

import base64
import mimetypes
import os
from contextlib import ExitStack

from openai import OpenAI


OPENAI_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "420"))
IMAGE_GENERATION_TIMEOUT_SECONDS = int(os.getenv("IMAGE_GENERATION_TIMEOUT_SECONDS", "360"))
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "openai_images").strip().lower()
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "").strip()
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "").strip().rstrip("/")


def _client() -> OpenAI:
    if not IMAGE_API_KEY or not IMAGE_BASE_URL:
        raise RuntimeError("IMAGE_API_KEY and IMAGE_BASE_URL must be configured")
    return OpenAI(
        api_key=IMAGE_API_KEY,
        base_url=IMAGE_BASE_URL,
        timeout=OPENAI_REQUEST_TIMEOUT,
        max_retries=0,
    )


def _data_url(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_dict(response) -> dict:
    return response.model_dump() if hasattr(response, "model_dump") else response.dict()


def generate_image(prompt: str, *, model: str, n: int = 1, size: str | None = None) -> dict:
    request = {
        "model": model,
        "prompt": prompt,
        "n": max(int(n or 1), 1),
    }
    if size:
        request["size"] = str(size)
    if IMAGE_PROVIDER == "qwen_image":
        # Qwen Image 3 uses OpenAI-compatible /images/generations for T2I.
        request["extra_body"] = {"prompt_extend": True}
    else:
        request["response_format"] = "public_url"
        request["extra_body"] = {"timeout_seconds": IMAGE_GENERATION_TIMEOUT_SECONDS}
    return _response_dict(_client().images.generate(**request))


def edit_image(image_paths: list[str], prompt: str, *, model: str, n: int = 1, size: str | None = None) -> dict:
    if not image_paths:
        raise ValueError("image_paths is required")
    if IMAGE_PROVIDER == "qwen_image":
        # Qwen's OpenAI-compatible mode intentionally does not implement
        # /images/edits multipart. Image editing is /images/generations plus
        # the provider-specific image URL/Base64 field.
        request = {
            "model": model,
            "prompt": prompt,
            "n": max(int(n or 1), 1),
            "extra_body": {"image": [_data_url(path) for path in image_paths], "prompt_extend": True},
        }
        if size:
            request["size"] = str(size)
        return _response_dict(_client().images.generate(**request))

    with ExitStack() as stack:
        image_files = [stack.enter_context(open(path, "rb")) for path in image_paths]
        request = {
            "model": model,
            "image": image_files[0] if len(image_files) == 1 else image_files,
            "prompt": prompt,
            "response_format": "public_url",
            "n": max(int(n or 1), 1),
            "extra_body": {"timeout_seconds": IMAGE_GENERATION_TIMEOUT_SECONDS},
        }
        if size:
            request["size"] = str(size)
        return _response_dict(_client().images.edit(**request))
