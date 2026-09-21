import base64
import mimetypes
import os
import time
import uuid
from functools import lru_cache
from urllib.parse import quote, unquote, urlparse

import requests
from fastapi import HTTPException


UPLOAD_DIR = os.path.abspath(os.getenv("UPLOAD_DIR", "./temp_images"))
PUBLIC_API_BASE = os.getenv("PUBLIC_API_BASE", "https://api.lumanova.icu").rstrip("/")
PUBLIC_IMAGE_BASE_URL = os.getenv("PUBLIC_IMAGE_BASE_URL", PUBLIC_API_BASE).rstrip("/")
MEDIA_S3_BUCKET = os.getenv("MEDIA_S3_BUCKET", "").strip()
MEDIA_S3_PREFIX = os.getenv("MEDIA_S3_PREFIX", "temp_file").strip().strip("/")
MEDIA_AWS_REGION = os.getenv("MEDIA_AWS_REGION", os.getenv("AWS_REGION", "us-east-1")).strip()
PUBLIC_MEDIA_BASE_URL = os.getenv(
    "PUBLIC_MEDIA_BASE_URL",
    f"{PUBLIC_IMAGE_BASE_URL}/{MEDIA_S3_PREFIX}" if MEDIA_S3_PREFIX else PUBLIC_IMAGE_BASE_URL,
).rstrip("/")
MEDIA_CACHE_CONTROL = os.getenv(
    "MEDIA_CACHE_CONTROL",
    "public, max-age=31536000, immutable",
).strip()
MEDIA_LOCAL_RETENTION_SECONDS = int(os.getenv("MEDIA_LOCAL_RETENTION_SECONDS", str(24 * 3600)))
IMAGE_DOWNLOAD_TIMEOUT = (10, 180)
IMAGE_DOWNLOAD_MAX_BYTES = int(os.getenv("IMAGE_DOWNLOAD_MAX_BYTES", str(32 * 1024 * 1024)))
VIDEO_DOWNLOAD_MAX_BYTES = int(os.getenv("VIDEO_DOWNLOAD_MAX_BYTES", str(1024 * 1024 * 1024)))
IMAGE_EXT_BY_CONTENT_TYPE = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


os.makedirs(UPLOAD_DIR, exist_ok=True)


def _safe_file_name(value):
    file_name = os.path.basename(unquote(str(value or ""))).strip()
    if not file_name or file_name in {".", ".."}:
        raise ValueError("Media file name is missing")
    return file_name


def _media_key(file_name):
    file_name = _safe_file_name(file_name)
    return f"{MEDIA_S3_PREFIX}/{file_name}" if MEDIA_S3_PREFIX else file_name


@lru_cache(maxsize=1)
def _s3_client():
    if not MEDIA_S3_BUCKET:
        raise RuntimeError("MEDIA_S3_BUCKET is not configured")
    import boto3

    return boto3.client("s3", region_name=MEDIA_AWS_REGION)


def media_storage_enabled():
    return bool(MEDIA_S3_BUCKET)


def generated_image_public_url(file_name):
    return f"{PUBLIC_MEDIA_BASE_URL}/{quote(_safe_file_name(file_name))}"


def media_content_type(path, fallback="application/octet-stream"):
    return mimetypes.guess_type(str(path or ""))[0] or fallback


def upload_media_file(path, file_name=None, content_type=None):
    path = os.path.abspath(str(path or ""))
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    file_name = _safe_file_name(file_name or path)
    if not media_storage_enabled():
        return generated_image_public_url(file_name)

    extra_args = {
        "ContentType": content_type or media_content_type(path),
        "CacheControl": MEDIA_CACHE_CONTROL,
    }
    _s3_client().upload_file(
        path,
        MEDIA_S3_BUCKET,
        _media_key(file_name),
        ExtraArgs=extra_args,
    )
    return generated_image_public_url(file_name)


def ensure_local_media_file(media_ref):
    media_ref = str(media_ref or "").strip()
    if not media_ref:
        return None

    parsed = urlparse(media_ref)
    if parsed.scheme in {"http", "https"}:
        file_name = _safe_file_name(parsed.path)
    else:
        candidate = os.path.abspath(media_ref)
        if os.path.isfile(candidate):
            return candidate
        file_name = _safe_file_name(candidate)

    local_path = os.path.join(UPLOAD_DIR, file_name)
    if os.path.isfile(local_path):
        return local_path
    if not media_storage_enabled():
        return None

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    temporary_path = f"{local_path}.{uuid.uuid4().hex}.part"
    try:
        _s3_client().download_file(MEDIA_S3_BUCKET, _media_key(file_name), temporary_path)
        os.replace(temporary_path, local_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
        raise
    return local_path


def cleanup_local_media_cache(max_age_seconds=None):
    max_age_seconds = (
        MEDIA_LOCAL_RETENTION_SECONDS
        if max_age_seconds is None
        else max(int(max_age_seconds), 0)
    )
    cutoff = time.time() - max_age_seconds
    removed = 0
    removed_bytes = 0
    for entry in os.scandir(UPLOAD_DIR):
        if not entry.is_file(follow_symlinks=False):
            continue
        try:
            stat = entry.stat(follow_symlinks=False)
            if stat.st_mtime > cutoff:
                continue
            os.remove(entry.path)
            removed += 1
            removed_bytes += stat.st_size
        except FileNotFoundError:
            continue
    return {"removed": removed, "bytes": removed_bytes}


def image_extension_from_response(url, content_type):
    parsed_ext = os.path.splitext(urlparse(url).path)[1].lower()
    if parsed_ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return ".jpg" if parsed_ext == ".jpeg" else parsed_ext
    content_type = (content_type or "").split(";", 1)[0].lower()
    return IMAGE_EXT_BY_CONTENT_TYPE.get(content_type, ".png")


def cache_remote_generated_image_file(image_url):
    parsed = urlparse(image_url or "")
    if parsed.scheme not in {"http", "https"}:
        return {"url": image_url, "path": None, "file_name": None}

    with requests.get(image_url, stream=True, timeout=IMAGE_DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if content_type and not content_type.lower().startswith("image/"):
            raise ValueError(f"Remote generated file is not an image: {content_type}")

        ext = image_extension_from_response(image_url, content_type)
        file_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_DIR, file_name)
        total_bytes = 0

        with open(save_path, "wb") as image_file:
            for chunk in response.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > IMAGE_DOWNLOAD_MAX_BYTES:
                    image_file.close()
                    os.remove(save_path)
                    raise ValueError("Remote generated image exceeds max download size")
                image_file.write(chunk)

    public_url = upload_media_file(save_path, file_name=file_name, content_type=content_type or None)
    return {
        "url": public_url,
        "path": save_path,
        "file_name": file_name,
    }


def cache_remote_generated_image(image_url):
    return cache_remote_generated_image_file(image_url)["url"]


def cache_remote_generated_video_file(video_url):
    """Persist an upstream clip so temporary provider URLs never become project state."""
    parsed = urlparse(video_url or "")
    if parsed.scheme not in {"http", "https"}:
        return {"url": video_url, "path": None, "file_name": None}

    with requests.get(video_url, stream=True, timeout=IMAGE_DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if content_type and not content_type.lower().startswith("video/"):
            raise ValueError(f"Remote generated file is not a video: {content_type}")
        ext = os.path.splitext(urlparse(video_url).path)[1].lower()
        if ext not in {".mp4", ".webm", ".mov", ".mkv"}:
            ext = ".mp4"
        file_name = f"{uuid.uuid4().hex}{ext}"
        save_path = os.path.join(UPLOAD_DIR, file_name)
        total_bytes = 0
        with open(save_path, "wb") as video_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total_bytes += len(chunk)
                if total_bytes > VIDEO_DOWNLOAD_MAX_BYTES:
                    video_file.close()
                    os.remove(save_path)
                    raise ValueError("Remote generated video exceeds max download size")
                video_file.write(chunk)

    public_url = upload_media_file(save_path, file_name=file_name, content_type=content_type or "video/mp4")
    return {"url": public_url, "path": save_path, "file_name": file_name}


def local_image_to_b64(path):
    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("ascii")


def uploaded_image_public_url(image_path):
    if not image_path:
        return None
    parsed = urlparse(str(image_path))
    if parsed.scheme in {"http", "https"}:
        return str(image_path)
    file_name = os.path.basename(str(image_path))
    if not file_name:
        return None
    return generated_image_public_url(file_name)


async def save_upload_image_file(file):
    if not file or not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail={"error": "Field `image` must be an image file."})
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in {".png", ".jpg", ".jpeg", ".webp"}:
        ext = IMAGE_EXT_BY_CONTENT_TYPE.get((file.content_type or "").split(";", 1)[0].lower(), ".png")
    unique_name = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)
    content = await file.read()
    if len(content) > IMAGE_DOWNLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail={"error": "Uploaded image exceeds max size."})
    with open(save_path, "wb") as output:
        output.write(content)
    try:
        upload_media_file(save_path, file_name=unique_name, content_type=file.content_type or None)
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise
    return save_path
