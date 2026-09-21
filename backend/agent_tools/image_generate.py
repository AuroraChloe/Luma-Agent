import os

from langchain_core.tools import tool
from media_store import cache_remote_generated_image_file
from image_provider import generate_image as provider_generate_image

from .base import AgentTool


OPENAI_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "420"))
IMAGE_GENERATION_TIMEOUT_SECONDS = int(os.getenv("IMAGE_GENERATION_TIMEOUT_SECONDS", "360"))
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "")
IMAGE_BASE_URL = os.getenv("IMAGE_BASE_URL", "http://127.0.0.1:3000/v1")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-2")


def _generate_image(prompt, model=IMAGE_MODEL, n=1, size=None):
    result = provider_generate_image(prompt, model=model, n=n, size=size)
    remote_url = (result.get("data") or [{}])[0].get("url")
    if not remote_url:
        return {"status": "0", "info": "NO_IMAGE_URL", "upstream": result}
    cached = cache_remote_generated_image_file(remote_url)
    image_url = cached["url"]
    image_path = cached["path"]
    return {
        "status": "1",
        "info": "OK",
        "prompt": prompt,
        "image_url": image_url,
        "image_path": image_path,
        "upstream": result,
    }


def generate_image_result(prompt, model=IMAGE_MODEL, n=1, size=None):
    """Generate one image through the configured provider for domain workflows."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"status": "0", "info": "MISSING_IMAGE_PROMPT"}
    return _generate_image(prompt, model=model, n=n, size=size)


@tool
def generate_image(image_prompt: str) -> dict:
    """根据文字提示生成图片。"""
    image_prompt = (image_prompt or "").strip()
    if not image_prompt:
        return {"status": "0", "info": "MISSING_IMAGE_PROMPT"}
    return generate_image_result(image_prompt)


def _image_generate_fallback(result, user_message=""):
    if not isinstance(result, dict):
        return str(result)
    if result.get("status") == "1" and result.get("image_url"):
        return f"图片已生成：{result.get('image_url')}"
    info = result.get("info") or "IMAGE_GENERATION_FAILED"
    if info == "MISSING_IMAGE_PROMPT":
        return "我没有识别到要生成图片的提示词，请补充你想画什么。"
    if info == "NO_IMAGE_URL":
        return "图片生成完成异常：上游没有返回图片地址。"
    return f"图片生成失败：{info}"


def _validate_image_generate_args(args, user_message="", context=""):
    image_prompt = str(args.get("image_prompt") or "").strip()
    if not image_prompt:
        image_prompt = (user_message or "").strip()
    return {"image_prompt": image_prompt}


IMAGE_GENERATE_TOOL = AgentTool(
    name="generate_image",
    description=(
        "根据用户当前请求生成新的视觉结果，例如图片、插画、头像、海报、表情包或照片。"
        "当请求需要新的独立画面且不依赖已有图片素材时使用；"
        "如果当前请求需要基于已有或历史图片保持视觉连续性，应选择 image_edit，"
        "不要因为用户使用了“生成”或没有使用“编辑”一词就忽略已有素材。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "image_prompt": {
                "type": "string",
                "description": "完整的图片生成提示词，保留用户想要的主体、风格、颜色、构图和细节。",
            }
        },
        "required": ["image_prompt"],
    },
    handler=generate_image,
    required=("image_prompt",),
    missing_args_message="我没有识别到要生成图片的提示词，请补充你想画什么。",
    validate_args=_validate_image_generate_args,
    fallback_formatter=_image_generate_fallback,
    argument_prompt=(
        "你只负责为 generate_image 工具准备参数。"
        "从用户请求中提取完整图片生成提示词，保留主体、风格、颜色、构图、场景和限制。"
        "不要把“帮我生成/画一张”等命令壳当作重点，但不要丢失用户的画面要求。"
    ),
    answer_prompt=(
        "你会收到用户问题和 generate_image 工具返回的 JSON。"
        "如果 JSON 里有 image_url，必须把这个图片 URL 原样告诉用户。"
        "回答要简洁自然，不要编造额外图片地址，不要说还在生成。"
    ),
)
