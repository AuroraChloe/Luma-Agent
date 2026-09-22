"""Optional text-to-speech post-processing for the web chat job flow."""

import os
import re
import uuid

from media_store import UPLOAD_DIR, upload_media_file


TTS_MODEL = os.getenv("DASHSCOPE_TTS_MODEL", "qwen-audio-3.0-tts-flash").strip()
TTS_VOICE = os.getenv("DASHSCOPE_TTS_VOICE", "longanhuan_v3.6").strip()
TTS_MAX_CHARS = int(os.getenv("DASHSCOPE_TTS_MAX_CHARS", "8000"))

_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)", re.IGNORECASE)
_HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_URL_RE = re.compile(r"(?:https?://|data:image/)[^\s<>\]\)]+", re.IGNORECASE)
_LOCAL_IMAGE_RE = re.compile(r"(?:/app/backend/data/media/|data/media/|temp_images/)[^\s<>\]\)]+", re.IGNORECASE)


def text_for_tts(content) -> str:
    """Keep human-readable text while removing image/link payloads."""

    text = str(content or "")
    text = _MARKDOWN_IMAGE_RE.sub("", text)
    text = _HTML_IMAGE_RE.sub("", text)
    text = _URL_RE.sub("", text)
    text = _LOCAL_IMAGE_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def synthesize_text_to_file(content) -> dict:
    """Synthesize final assistant text and return a public audio asset."""

    text = text_for_tts(content)
    if not text:
        return {"status": "skipped", "reason": "empty_text"}
    if TTS_MAX_CHARS > 0:
        text = text[:TTS_MAX_CHARS]

    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer

    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not configured")
    dashscope.api_key = api_key
    dashscope.base_websocket_api_url = os.getenv(
        "DASHSCOPE_TTS_BASE_WEBSOCKET_URL",
        "wss://dashscope.aliyuncs.com/api-ws/v1/inference",
    )

    audio = SpeechSynthesizer(model=TTS_MODEL, voice=TTS_VOICE).call(text)
    if not isinstance(audio, (bytes, bytearray)) or not audio:
        raise RuntimeError(f"TTS returned no audio bytes: {type(audio).__name__}")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_name = f"tts_{uuid.uuid4().hex}.mp3"
    output_path = os.path.join(UPLOAD_DIR, file_name)
    with open(output_path, "wb") as audio_file:
        audio_file.write(audio)
    public_url = upload_media_file(
        output_path,
        file_name=file_name,
        content_type="audio/mpeg",
    )

    return {
        "status": "completed",
        "url": public_url,
        "path": output_path,
        "file_name": file_name,
        "model": TTS_MODEL,
        "voice": TTS_VOICE,
        "text_chars": len(text),
    }
