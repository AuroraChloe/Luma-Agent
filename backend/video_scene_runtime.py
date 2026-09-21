"""Scene-reference material stage for the guided video workflow.

Scene records live in the project brief while their actual images are regular
chat assets.  It keeps the asset lifecycle and access controls shared with the
rest of the product without introducing a second media store.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_llm import build_agent_llm
from agent_tools.image_edit import edit_image_result
from agent_tools.image_generate import generate_image_result
from chat_protocol import add_chat_usage
from sql import ensure_chat_asset, get_or_create_video_project, record_client_image_usage, update_video_project


SKILL_PATH = Path(__file__).resolve().parent / "workflow_skills" / "video_scene_material" / "SKILL.md"
VIDEO_SCENE_SIZE = "1536x1024"


class SceneSpec(BaseModel):
    scene_ref: str = ""
    display_name: str = ""
    setting: str = ""
    visual_direction: str = ""
    continuity_notes: str = ""
    related_beat_indexes: list[int] = Field(default_factory=list)


class VideoSceneDecision(BaseModel):
    intent: Literal["generate_scenes", "revise_scene", "approve_scenes", "clarify", "reply"] = "clarify"
    scenes: list[SceneSpec] = Field(default_factory=list)
    target_scene_refs: list[str] = Field(default_factory=list)
    change_request: str = ""
    assistant_reply: str = ""


def _skill_body() -> str:
    try:
        content = SKILL_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""
    if not content.startswith("---"):
        return content
    parts = content.split("---", 2)
    return parts[2].strip() if len(parts) == 3 else content


def _usage(message) -> dict:
    value = getattr(message, "usage_metadata", None)
    if isinstance(value, dict):
        return {
            "prompt_tokens": value.get("input_tokens", 0) or 0,
            "completion_tokens": value.get("output_tokens", 0) or 0,
            "total_tokens": value.get("total_tokens", 0) or 0,
        }
    metadata = getattr(message, "response_metadata", None) or {}
    return dict(metadata.get("token_usage") or {}) if isinstance(metadata, dict) else {}


def _history(messages: list[dict], limit=12, max_chars=5500) -> str:
    rows = []
    for item in (messages or [])[-limit:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(block.get("text") or "") for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        value = re.sub(r"\s+", " ", str(content or "")).strip()
        if value:
            rows.append(f"{item['role']}: {value[:800]}")
    value = "\n".join(rows)
    return value[-max_chars:]


def _concept(project: dict) -> dict:
    brief = project.get("project_brief") or {}
    value = brief.get("concept")
    return value if isinstance(value, dict) else {}


def _scenes(project: dict) -> list[dict]:
    value = (project.get("project_brief") or {}).get("scene_materials")
    return [dict(item) for item in value] if isinstance(value, list) else []


def _scene_catalog(scenes: list[dict]) -> list[dict]:
    return [
        {
            "scene_id": item.get("scene_id"),
            "ordinal": item.get("ordinal"),
            "display_name": item.get("display_name"),
            "setting": item.get("setting"),
            "related_beat_indexes": item.get("related_beat_indexes") or [],
            "status": item.get("status"),
            "has_image": bool(item.get("asset_id")),
            "is_approved": bool(item.get("approved")),
        }
        for item in scenes
    ]


def _decide(*, model: str, current_request: str, messages: list[dict], project: dict, scenes: list[dict]):
    concept = _concept(project)
    prompt = (
        "你是 LumaNova 视频生成流水线的场景美术导演。当前只负责为已确认概念创建可复用的场景参考图。"
        "先根据完整概念脚本识别实际需要的地点或环境状态；同一连续地点只建一张场景基准图，环境发生关键变化时才拆成新场景。"
        "场景图必须是无人、无角色、无文字、无分格的横向环境基准图，供后续分镜图编辑时锁定空间、光线、道具和氛围。"
        "不要执行人物设计、分镜图、视频生成或逐镜头提示词。"
        "用户要求开始或继续时，在尚无场景素材时使用 generate_scenes；用户明确满意时使用 approve_scenes。"
        "scene_ref 只能引用已有 scene_id、显示名，或本轮 scenes 中的显示名。related_beat_indexes 使用从 1 开始的节拍序号。"
        "assistant_reply 仅用于自然中文回复，不暴露字段、JSON、工具或系统实现。\n\n" + _skill_body()
    )
    payload = {
        "current_request": str(current_request or "").strip(),
        "recent_conversation": _history(messages),
        "confirmed_script": {
            "title": concept.get("title"),
            "visual_direction": concept.get("visual_direction"),
            "beats": [{"beat_index": i, **item} for i, item in enumerate(concept.get("beats") or [], 1)],
        },
        "existing_scenes": _scene_catalog(scenes),
    }
    response = build_agent_llm(model=model).bind_tools([VideoSceneDecision]).invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False, default=str) + "\n\n请调用 VideoSceneDecision 返回本轮唯一决策。"),
    ])
    calls = getattr(response, "tool_calls", None) or []
    if not calls:
        raise RuntimeError("video scene director returned no structured decision")
    return VideoSceneDecision.model_validate(calls[0].get("args") or {}), _usage(response)


def _scene_prompt(scene: dict, concept: dict) -> str:
    return (
        "为 AI 视频制作创建一张正式场景基准图。"
        f"场景名称：{scene.get('display_name') or '未命名场景'}。"
        f"地点与空间：{scene.get('setting') or ''}。"
        f"视觉方向：{scene.get('visual_direction') or concept.get('visual_direction') or ''}。"
        f"连续性要求：{scene.get('continuity_notes') or ''}。"
        "这是一张供后续图片编辑和图生视频引用的环境资产：只呈现场景本身、固定空间布局、关键道具、材质、色彩、光源和大气。"
        "画面内严禁出现任何人物、角色、动物、可辨认面孔、文字、标题、镜头编号、水印、分镜格或设定板排版。"
        f"统一输出 {VIDEO_SCENE_SIZE} 横向电影画幅，构图完整且保留人物进入画面的可用空间。"
    )


def _scene_revision_prompt(scene: dict, change_request: str) -> str:
    return (
        "修正这张视频场景基准图。"
        f"本次只执行：{change_request.strip()}。"
        f"场景基线：{json.dumps({key: scene.get(key) for key in ('display_name', 'setting', 'visual_direction', 'continuity_notes')}, ensure_ascii=False)}。"
        "保留未要求改变的空间布局、光源、道具、材质、色彩与氛围；仍不得出现角色、文字、分格或水印。"
        f"输出 {VIDEO_SCENE_SIZE} 横向单幅环境画面。"
    )


def _resolve_scene(scenes: list[dict], reference: str) -> dict | None:
    value = str(reference or "").strip().lower()
    if not value:
        return None
    for scene in scenes:
        if value in {str(scene.get("scene_id") or "").lower(), str(scene.get("display_name") or "").lower()}:
            return scene
    match = re.search(r"(\d+)", value)
    if match:
        ordinal = int(match.group(1))
        return next((item for item in scenes if item.get("ordinal") == ordinal), None)
    return None


class _Trace:
    def __init__(self, callback=None):
        self.callback, self.items, self.used_tools, self.usage = callback, [], [], {}

    def add_usage(self, usage):
        self.usage = add_chat_usage(self.usage, usage) or self.usage

    def call(self, name, content):
        self.used_tools.append(name)
        self.items.append({"type": "ToolCall", "name": name, "content": content, "tool_calls": [{"name": name, "args": content, "type": "tool_call"}]})
        self.emit()

    def result(self, name, content):
        self.items.append({"type": "ToolMessage", "name": name, "content": content, "tool_calls": []})
        self.emit()

    def answer(self, content):
        self.items.append({"type": "ModelAnswer", "name": None, "content": content, "tool_calls": []})
        self.emit()

    def emit(self):
        if self.callback:
            self.callback(trace=list(self.items), used_tools=list(dict.fromkeys(self.used_tools)), usage=self.usage)


def _persist_scene_asset(client_id: int, session_id: str, scene: dict, result: dict) -> dict | None:
    if result.get("status") != "1" or not result.get("image_url"):
        return None
    asset = ensure_chat_asset(
        client_id, session_id,
        asset_type="video_scene_design",
        local_path=result.get("image_path"), public_url=result.get("image_url"),
        source_asset_id=scene.get("asset_id"),
        asset_label=f"{scene.get('display_name') or '场景'}的场景基准图",
        operation_prompt=result.get("compiled_prompt"), make_active=False,
    )
    if not asset:
        return None
    return {**scene, "asset_id": asset.get("asset_id"), "public_url": asset.get("public_url"), "local_path": asset.get("local_path"), "status": "awaiting_review"}


def run_video_scene_chat(client_id, *, session_id: str, model: str, messages: list[dict], user_message: str, image_path: str | None = None, trace_callback=None) -> dict:
    del image_path
    project = get_or_create_video_project(client_id, session_id)
    scenes = _scenes(project)
    trace = _Trace(trace_callback)
    decision, usage = _decide(model=model, current_request=user_message, messages=messages, project=project, scenes=scenes)
    trace.add_usage(usage)
    brief = dict(project.get("project_brief") or {})
    concept = _concept(project)

    if decision.intent == "generate_scenes":
        if scenes:
            pending = [item for item in scenes if not item.get("asset_id")]
            if not pending:
                reply = "场景基准图已经生成。请确认整组场景，或指出需要调整的场景。"
                trace.answer(reply)
                return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": []}
        else:
            pending = []
            for ordinal, item in enumerate(decision.scenes, 1):
                data = item.model_dump()
                data.update({"scene_id": f"scene_{uuid.uuid4().hex}", "ordinal": ordinal, "status": "planned", "approved": False})
                pending.append(data)
            scenes = list(pending)
        outputs = []
        by_id = {item.get("scene_id"): item for item in scenes}
        for scene in pending:
            if not record_client_image_usage(client_id, 1).get("allowed"):
                outputs.append("图片额度已达到当前时段上限，未生成的场景计划已保留。")
                break
            trace.call("generate_video_scene", {"ordinal": scene.get("ordinal"), "display_name": scene.get("display_name")})
            prompt = _scene_prompt(scene, concept)
            result = generate_image_result(prompt, size=VIDEO_SCENE_SIZE)
            result["compiled_prompt"] = prompt
            trace.result("generate_video_scene", {key: result.get(key) for key in ("status", "info", "image_url") if result.get(key) is not None})
            persisted = _persist_scene_asset(client_id, session_id, scene, result)
            if persisted:
                by_id[persisted["scene_id"]] = persisted
                outputs.append(f"**场景 {persisted.get('ordinal')} · {persisted.get('display_name')}**\n\n{persisted.get('public_url')}")
            else:
                outputs.append(f"**场景 {scene.get('ordinal')} · {scene.get('display_name')}**\n\n场景基准图没有成功生成，计划已保留。")
        scenes = [by_id[item.get("scene_id")] for item in scenes]
        brief["scene_materials"] = scenes
        update_video_project(client_id, session_id, stage="scene_material", status="awaiting_scene_review", project_brief=brief)
        reply = "\n\n".join(outputs) + "\n\n这些是后续分镜图统一引用的场景资产。确认后，我会用角色设计图和场景图一起构建完整分镜。"

    elif decision.intent == "revise_scene":
        target = _resolve_scene(scenes, (decision.target_scene_refs or [""])[0])
        if not target or not target.get("public_url"):
            reply = "还没有找到可以修改的场景基准图。"
        elif not decision.change_request.strip():
            reply = "请告诉我这个场景具体需要怎样调整。"
        elif not record_client_image_usage(client_id, 1).get("allowed"):
            reply = "本次图片使用已达到当前时段上限，修改要求还没有执行。"
        else:
            prompt = _scene_revision_prompt(target, decision.change_request)
            trace.call("revise_video_scene", {"ordinal": target.get("ordinal"), "change_request": decision.change_request})
            result = edit_image_result(target.get("public_url") or target.get("local_path"), prompt, size=VIDEO_SCENE_SIZE)
            result["compiled_prompt"] = prompt
            trace.result("revise_video_scene", {key: result.get(key) for key in ("status", "info", "image_url") if result.get(key) is not None})
            persisted = _persist_scene_asset(client_id, session_id, target, result)
            if persisted:
                scenes = [persisted if item.get("scene_id") == persisted.get("scene_id") else item for item in scenes]
                brief["scene_materials"] = scenes
                update_video_project(client_id, session_id, stage="scene_material", status="awaiting_scene_review", project_brief=brief)
                reply = f"**场景 {persisted.get('ordinal')} · {persisted.get('display_name')}** 已更新：\n\n{persisted.get('public_url')}"
            else:
                reply = "场景图这次没有成功更新，上一版仍然保留。"

    elif decision.intent == "approve_scenes":
        pending = [item for item in scenes if not item.get("asset_id")]
        if not scenes or pending:
            reply = "场景基准图尚未全部生成，暂时不能进入分镜构建。"
        else:
            scenes = [{**item, "approved": True, "status": "approved"} for item in scenes]
            brief["scene_materials"] = scenes
            update_video_project(client_id, session_id, stage="keyframe_material", status="storyboard_ready", project_brief=brief)
            reply = "场景素材已经锁定。下一步会让每张分镜图同时引用对应角色设计图与场景基准图，再按脚本逐镜头构建。"

    else:
        reply = decision.assistant_reply.strip() or "角色素材已经定稿。你可以让我开始生成正式场景基准图。"

    trace.answer(reply)
    return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}
