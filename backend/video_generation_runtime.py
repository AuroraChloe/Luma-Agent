"""Submit one continuous short-video task from a reviewed storyboard sheet."""

from __future__ import annotations

import os
import re
import time

import requests

from media_store import cache_remote_generated_video_file
from sql import (
    ensure_chat_asset,
    get_or_create_video_project,
    list_video_subject_assets,
    list_video_subjects,
    update_video_project,
)


VIDEO_GATEWAY_BASE_URL = os.getenv("VIDEO_GATEWAY_BASE_URL", "https://comfy.xianshi.icu/video-api").rstrip("/")
VIDEO_GATEWAY_API_KEY = os.getenv("VIDEO_GATEWAY_API_KEY", "").strip()
VIDEO_RESOLUTION = os.getenv("VIDEO_RESOLUTION", "480p")
VIDEO_ASPECT_RATIO = os.getenv("VIDEO_ASPECT_RATIO", "16:9")
VIDEO_INSTANCE_TYPE = os.getenv("VIDEO_INSTANCE_TYPE", "ultra")
VIDEO_POLL_INTERVAL_SECONDS = max(float(os.getenv("VIDEO_POLL_INTERVAL_SECONDS", "4")), 1.0)
VIDEO_POLL_TIMEOUT_SECONDS = max(int(os.getenv("VIDEO_POLL_TIMEOUT_SECONDS", "900")), 30)
PICTURE_TAG_RE = re.compile(r"\[Picture\s+(\d+)\]", re.I)


class _Trace:
    def __init__(self, callback=None):
        self.callback, self.items, self.used_tools, self.usage = callback, [], [], {}

    def call(self, name: str, content: dict):
        self.used_tools.append(name)
        self.items.append({"type": "ToolCall", "name": name, "content": content, "tool_calls": [{"name": name, "args": content, "type": "tool_call"}]})
        self.emit()

    def result(self, name: str, content: dict):
        self.items.append({"type": "ToolMessage", "name": name, "content": content, "tool_calls": []})
        self.emit()

    def answer(self, content: str):
        self.items.append({"type": "ModelAnswer", "name": None, "content": content, "tool_calls": []})
        self.emit()

    def emit(self):
        if self.callback:
            self.callback(trace=list(self.items), used_tools=list(dict.fromkeys(self.used_tools)), usage=self.usage)


def _active_asset(assets: list[dict], asset_id: str | None) -> dict | None:
    return next((item for item in assets if item.get("asset_id") == asset_id), None)


def _https_url(asset: dict | None) -> str:
    url = str((asset or {}).get("public_url") or "").strip()
    return url if url.startswith("https://") else ""


def _production_reference_urls(client_id: int, session_id: str, brief: dict) -> list[str]:
    """Keep URL ordering identical to the Vision review's Picture mapping."""
    sheet_url = str((brief.get("storyboard_sheet") or {}).get("public_url") or "").strip()
    if not sheet_url.startswith("https://"):
        return []
    urls = [sheet_url]
    all_subjects = list_video_subjects(client_id, session_id)
    subjects_by_name = {str(item.get("display_name") or "").strip().lower(): item for item in all_subjects}
    ordered_subjects = []
    for concept_subject in (brief.get("concept") or {}).get("subjects") or []:
        if not isinstance(concept_subject, dict):
            continue
        subject = subjects_by_name.get(str(concept_subject.get("display_name") or "").strip().lower())
        if subject and subject not in ordered_subjects:
            ordered_subjects.append(subject)
    for subject in ordered_subjects or all_subjects:
        asset = _active_asset(list_video_subject_assets(client_id, session_id, subject.get("subject_id")), subject.get("approved_asset_id"))
        if url := _https_url(asset):
            urls.append(url)
    for scene in brief.get("scene_materials") or []:
        if not isinstance(scene, dict):
            continue
        url = str(scene.get("public_url") or "").strip()
        if scene.get("approved") and url.startswith("https://"):
            urls.append(url)
    return urls[:10]


def _single_video_task(client_id: int, session_id: str, project: dict) -> dict | None:
    brief = project.get("project_brief") or {}
    report = ((brief.get("video_production") or {}).get("report") or {})
    prompt = str(report.get("production_prompt") or "").strip()
    references = _production_reference_urls(client_id, session_id, brief)
    if not prompt or not references:
        return None
    mentioned = {int(value) for value in PICTURE_TAG_RE.findall(prompt)}
    if not set(range(1, len(references) + 1)).issubset(mentioned):
        return None
    try:
        duration = int(report.get("duration_seconds") or 15)
    except (TypeError, ValueError):
        duration = 15
    return {
        "duration": max(1, min(duration, 15)),
        "prompt": prompt,
        "reference_urls": references,
        "storyboard_asset_id": (brief.get("storyboard_sheet") or {}).get("asset_id"),
    }


def _gateway_headers() -> dict[str, str]:
    return {"X-API-Key": VIDEO_GATEWAY_API_KEY}


def _submit_video(task: dict) -> str:
    fields = [
        ("text", (None, task["prompt"])),
        ("resolution", (None, VIDEO_RESOLUTION)),
        ("aspect_ratio", (None, VIDEO_ASPECT_RATIO)),
        ("duration", (None, str(task["duration"]))),
        ("instance_type", (None, VIDEO_INSTANCE_TYPE)),
    ]
    fields.extend(("image_urls", (None, url)) for url in task["reference_urls"])
    response = requests.post(f"{VIDEO_GATEWAY_BASE_URL}/v1/videos/reference-to-video", headers=_gateway_headers(), files=fields, timeout=(10, 60))
    response.raise_for_status()
    job_id = str((response.json() or {}).get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError("Video gateway did not return job_id")
    return job_id


def _query_video(job_id: str) -> dict:
    response = requests.get(f"{VIDEO_GATEWAY_BASE_URL}/v1/jobs/{job_id}", headers=_gateway_headers(), timeout=(10, 30))
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _save_project(client_id: int, session_id: str, project: dict, task: dict, *, stage: str, status: str, final_video_url: str | None = None):
    brief = dict(project.get("project_brief") or {})
    brief["video_task"] = task
    brief.pop("video_clips", None)
    if final_video_url:
        brief["final_video_url"] = final_video_url
    return update_video_project(client_id, session_id, stage=stage, status=status, project_brief=brief)


def reconcile_video_generation(project: dict) -> dict | None:
    """Finish a submitted gateway job after its foreground chat worker is gone."""
    if not project or project.get("stage") != "video_generating":
        return None
    client_id = int(project.get("client_id") or 0)
    session_id = str(project.get("session_id") or "")
    task = dict((project.get("project_brief") or {}).get("video_task") or {})
    gateway_job_id = str(task.get("gateway_job_id") or "").strip()
    if not client_id or not session_id or not gateway_job_id:
        return None
    try:
        payload = _query_video(gateway_job_id)
        gateway_status = str(payload.get("status") or "pending").lower()
        if gateway_status in {"pending", "queued", "running", "processing"}:
            return {"state": "running", "client_id": client_id, "session_id": session_id}
        if gateway_status in {"completed", "succeeded", "success"}:
            source_url = str((payload.get("videos") or [""])[0] or "").strip()
            cached = cache_remote_generated_video_file(source_url)
            if not cached.get("url"):
                raise RuntimeError("Video gateway completed without a usable video URL")
            task.update({"status": "completed", "video_url": cached["url"], "local_path": cached.get("path"), "error": ""})
            ensure_chat_asset(client_id, session_id, asset_type="video_final", local_path=cached.get("path"), public_url=cached["url"], asset_label="完整连续短视频", operation_prompt=task.get("prompt") or "", source_asset_id=task.get("storyboard_asset_id"), make_active=True)
            _save_project(client_id, session_id, project, task, stage="video_ready", status="completed", final_video_url=cached["url"])
            return {"state": "completed", "client_id": client_id, "session_id": session_id, "content": f"最终视频已生成：{cached['url']}"}
        if gateway_status in {"failed", "error", "cancelled", "canceled"}:
            task.update({"status": "failed", "error": str(payload.get("error") or "Video generation failed")})
            _save_project(client_id, session_id, project, task, stage="video_prompt_ready", status="video_failed")
            return {"state": "failed", "client_id": client_id, "session_id": session_id, "content": f"最终视频没有生成成功：{task['error']}。分镜图与总提示词已保留，可以直接重试。"}
    except Exception as exc:
        # A temporary gateway or storage failure must not discard a live render.
        print(f"video generation reconcile deferred session_id={session_id}: {type(exc).__name__}: {exc}", flush=True)
    return None


def run_video_generation_chat(client_id, *, session_id: str, model: str, messages: list[dict], user_message: str, image_path: str | None = None, trace_callback=None) -> dict:
    del model, messages, user_message, image_path
    trace = _Trace(trace_callback)
    if not VIDEO_GATEWAY_API_KEY:
        reply = "视频生成服务尚未配置，暂时不能提交视频任务。"
        trace.answer(reply)
        return {"content": reply, "usage": {}, "trace": trace.items, "used_tools": []}
    project = get_or_create_video_project(client_id, session_id)
    task = _single_video_task(client_id, session_id, project)
    if not task:
        reply = "完整分镜图或总视频提示词尚未准备完成，暂时不能提交最终视频。"
        trace.answer(reply)
        return {"content": reply, "usage": {}, "trace": trace.items, "used_tools": []}

    previous = dict((project.get("project_brief") or {}).get("video_task") or {})
    if previous.get("status") == "completed" and previous.get("video_url"):
        reply = f"最终视频已生成：{previous['video_url']}"
        trace.answer(reply)
        return {"content": reply, "usage": {}, "trace": trace.items, "used_tools": []}
    task = {**previous, **task}
    if not task.get("gateway_job_id") or task.get("status") == "failed":
        trace.call("submit_final_video", {"duration": task["duration"], "reference_count": len(task["reference_urls"])})
        try:
            task.update({"gateway_job_id": _submit_video(task), "status": "pending", "error": ""})
            trace.result("submit_final_video", {"status": "1", "job_id": task["gateway_job_id"]})
        except Exception as exc:
            task.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            _save_project(client_id, session_id, project, task, stage="video_prompt_ready", status="video_failed")
            reply = "最终视频任务提交失败，已保留分镜图与提示词，可以稍后重试。"
            trace.result("submit_final_video", {"status": "0"})
            trace.answer(reply)
            return {"content": reply, "usage": {}, "trace": trace.items, "used_tools": trace.used_tools}

    project = _save_project(client_id, session_id, project, task, stage="video_generating", status="video_running")
    deadline = time.monotonic() + VIDEO_POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        time.sleep(VIDEO_POLL_INTERVAL_SECONDS)
        try:
            payload = _query_video(task["gateway_job_id"])
            status = str(payload.get("status") or "pending").lower()
            if status in {"completed", "succeeded", "success"}:
                source_url = str((payload.get("videos") or [""])[0] or "").strip()
                cached = cache_remote_generated_video_file(source_url)
                if not cached.get("url"):
                    raise RuntimeError("Video gateway completed without a usable video URL")
                task.update({"status": "completed", "video_url": cached["url"], "local_path": cached.get("path"), "error": ""})
                ensure_chat_asset(client_id, session_id, asset_type="video_final", local_path=cached.get("path"), public_url=cached["url"], asset_label="完整连续短视频", operation_prompt=task["prompt"], source_asset_id=task.get("storyboard_asset_id"), make_active=True)
                _save_project(client_id, session_id, project, task, stage="video_ready", status="completed", final_video_url=cached["url"])
                reply = f"最终视频已生成：{cached['url']}"
                trace.result("final_video_completed", {"status": "1", "video_url": cached["url"]})
                trace.answer(reply)
                return {"content": reply, "usage": {}, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}
            if status in {"failed", "error", "cancelled"}:
                task.update({"status": "failed", "error": str(payload.get("error") or "Video generation failed")})
                _save_project(client_id, session_id, project, task, stage="video_prompt_ready", status="video_failed")
                reply = "最终视频没有生成成功，已保留分镜图与总提示词，可以直接重试。"
                trace.result("final_video_completed", {"status": "0"})
                trace.answer(reply)
                return {"content": reply, "usage": {}, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}
            task["status"] = "running"
            project = _save_project(client_id, session_id, project, task, stage="video_generating", status="video_running")
        except Exception as exc:
            task.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
            _save_project(client_id, session_id, project, task, stage="video_prompt_ready", status="video_failed")
            reply = "最终视频生成过程中出现异常，已保留分镜图与总提示词，可以稍后重试。"
            trace.result("final_video_completed", {"status": "0"})
            trace.answer(reply)
            return {"content": reply, "usage": {}, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}

    task.update({"status": "running", "error": ""})
    _save_project(client_id, session_id, project, task, stage="video_generating", status="video_running")
    reply = "最终视频仍在生成中，已保留任务状态，稍后继续查询即可。"
    trace.answer(reply)
    return {"content": reply, "usage": {}, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}
