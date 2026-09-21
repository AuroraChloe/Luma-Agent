import os
from urllib.parse import urlparse

from langchain_core.tools import tool
from media_store import UPLOAD_DIR, cache_remote_generated_image_file, ensure_local_media_file
from image_provider import edit_image as provider_edit_image

from .base import AgentTool


OPENAI_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "420"))
IMAGE_GENERATION_TIMEOUT_SECONDS = int(os.getenv("IMAGE_GENERATION_TIMEOUT_SECONDS", "360"))
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "")
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "http://127.0.0.1:3000/v1")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2")


def _normalize_image_paths(image_path):
    values = image_path if isinstance(image_path, (list, tuple)) else [image_path]
    local_paths = []
    for value in values:
        local_path = ensure_local_media_file(str(value or "").strip())
        if not local_path:
            return []
        local_paths.append(local_path)
    return local_paths


def _edit_image(image_path, prompt, model=IMAGE_MODEL, n=1, size=None):
    image_paths = _normalize_image_paths(image_path)
    if not image_paths:
        return {"status": "0", "info": "IMAGE_FILE_NOT_FOUND", "image_path": image_path}

    result = provider_edit_image(image_paths, prompt, model=model, n=n, size=size)
    remote_url = (result.get("data") or [{}])[0].get("url")
    if not remote_url:
        return {"status": "0", "info": "NO_IMAGE_URL", "image_path": image_paths, "upstream": result}
    cached = cache_remote_generated_image_file(remote_url)
    image_url = cached["url"]
    output_image_path = cached["path"]
    return {
        "status": "1",
        "info": "OK",
        "edit_prompt": prompt,
        "source_image_path": image_paths[0] if len(image_paths) == 1 else image_paths,
        "image_url": image_url,
        "image_path": output_image_path,
        "upstream": result,
    }


def edit_image_result(image_path, prompt, model=IMAGE_MODEL, n=1, size=None):
    """Edit one or more reference images through the configured provider."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"status": "0", "info": "MISSING_EDIT_PROMPT"}
    if isinstance(image_path, (list, tuple)):
        image_path = [str(item or "").strip() for item in image_path if str(item or "").strip()]
    else:
        image_path = str(image_path or "").strip()
    if not image_path:
        return {"status": "0", "info": "MISSING_IMAGE_PATH"}
    return _edit_image(image_path, prompt, model=model, n=n, size=size)


@tool
def edit_image(edit_prompt: str, image_path: str) -> dict:
    """根据用户提示编辑上传的图片。"""
    edit_prompt = (edit_prompt or "").strip()
    image_path = (image_path or "").strip()
    if not edit_prompt:
        return {"status": "0", "info": "MISSING_EDIT_PROMPT"}
    if not image_path:
        return {"status": "0", "info": "MISSING_IMAGE_PATH"}
    return edit_image_result(image_path, edit_prompt)


def _image_edit_fallback(result, user_message=""):
    if not isinstance(result, dict):
        return str(result)
    if result.get("status") == "1" and result.get("image_url"):
        return f"图片已修改：{result.get('image_url')}"
    info = result.get("info") or "IMAGE_EDIT_FAILED"
    if info == "MISSING_EDIT_PROMPT":
        return "我没有识别到要怎么修改图片，请补充修改要求。"
    if info == "MISSING_IMAGE_PATH":
        return "我没有拿到要修改的图片，请先上传一张图片。"
    if info == "IMAGE_FILE_NOT_FOUND":
        return "要修改的图片文件没有找到，请重新上传图片。"
    if info == "NO_IMAGE_URL":
        return "图片修改完成异常：上游没有返回图片地址。"
    return f"图片修改失败：{info}"


def _validate_image_edit_args(args, user_message="", context=""):
    # Keep the interpreter's normalized visual instruction. The conversational
    # query may contain asset-selection context and subjective commentary.
    edit_prompt = str(args.get("edit_prompt") or "").strip()
    if not edit_prompt:
        edit_prompt = (user_message or "").strip()
    image_path = str(args.get("image_path") or "").strip()

    if image_path:
        parsed = urlparse(image_path)
        if parsed.scheme in {"http", "https"}:
            image_path = image_path
        else:
            candidate = os.path.realpath(image_path)
            managed_dir = os.path.realpath(UPLOAD_DIR)
            if candidate != managed_dir and candidate.startswith(managed_dir + os.sep):
                image_path = candidate
            else:
                image_path = ""
    return {"edit_prompt": edit_prompt, "image_path": image_path}


IMAGE_EDIT_TOOL = AgentTool(
    name="image_edit",
    description=(
        "根据用户当前请求编辑、延续或生成基于已有图片素材的视觉结果。"
        "素材可以来自当前上传图片或历史对话中仍存在的生成图片；"
        "用于保持人物、风格、场景、构图等视觉连续性，不限于传统修图。"
        "如果用户是在连续对话中要求再次呈现、改变姿态、角度、场景或其他画面结果，"
        "应优先考虑复用已有图片素材，而不是把它当成一次无关的新生图。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "edit_prompt": {
                "type": "string",
                "description": "完整的图片修改要求，保留用户想要改成什么、保留什么、去掉什么。",
            },
            "image_path": {
                "type": "string",
                "description": "服务器上的图片路径。通常来自用户上传图片后的 image_path。",
            },
        },
        "required": ["edit_prompt", "image_path"],
    },
    handler=edit_image,
    required=("edit_prompt", "image_path"),
    missing_args_message="我没有拿到完整的修图要求或图片，请先上传图片并说明要怎么改。",
    validate_args=_validate_image_edit_args,
    fallback_formatter=_image_edit_fallback,
    argument_prompt=(
        "你只负责为 image_edit 工具准备参数。"
        "edit_prompt 只保留可执行的视觉修改指令，不要照抄整句聊天内容。"
        "保留明确的新增、删除、替换、颜色、姿态、构图、风格和必要的保持条件。"
        "不要把图片版本选择、代词、比较、情绪评价或口语填充写进 edit_prompt；"
        "图片选择通过 context_refs 表达，主观评价只有在用户明确转成可执行视觉变化时才保留。"
        "如果一句话同时包含评价、上下文引用和具体修改，只保留具体修改，并补充必要的‘其他内容保持不变’约束。"
        "不要自行补充用户没有提出的主体、风格、颜色、构图或其他视觉细节。"
        "如果当前请求依赖历史图片素材，使用系统提供的历史图片路径作为 image_path；"
        "如果当前有用户上传图片，则优先使用当前上传图片。不要编造不存在的 image_path。"
    ),
    answer_prompt=(
        "你会收到用户问题和 image_edit 工具返回的 JSON。"
        "如果 JSON 里有 image_url，必须把这个图片 URL 原样告诉用户。"
        "回答要简洁自然，不要编造额外图片地址，不要说还在修改。"
    ),
)
