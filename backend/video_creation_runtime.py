"""Stage-aware runtime for the guided video creation workflow."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_llm import build_agent_llm
from chat_protocol import add_chat_usage
from sql import get_or_create_video_project, update_video_project
from video_keyframe_runtime import run_video_keyframe_chat
from video_generation_runtime import run_video_generation_chat
from video_scene_runtime import run_video_scene_chat
from video_subject_runtime import run_video_subject_chat


SKILL_PATH = (
    Path(__file__).resolve().parent
    / "workflow_skills"
    / "video_creation"
    / "SKILL.md"
)


class ConceptBeat(BaseModel):
    time_range: str = ""
    purpose: str = ""
    content: str = ""
    visual: str = ""


class ConceptSubject(BaseModel):
    display_name: str = ""
    subject_type: str = "other"
    role: str = ""
    visual_direction: str = ""


class ShortVideoConcept(BaseModel):
    title: str = ""
    format: str = "single_short_video"
    duration_seconds: int = 15
    theme: str = ""
    audience: str = ""
    tone: str = ""
    logline: str = ""
    synopsis: str = ""
    beats: list[ConceptBeat] = Field(default_factory=list)
    visual_direction: str = ""
    audio_direction: str = ""
    subjects: list[ConceptSubject] = Field(default_factory=list)
    ending: str = ""


class VideoConceptDecision(BaseModel):
    intent: Literal[
        "clarify",
        "draft_concept",
        "revise_concept",
        "approve_concept",
        "reply",
    ] = "clarify"
    concept: ShortVideoConcept | None = None
    missing_decision: str = ""
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


def _compact_history(messages, *, limit=14, max_chars=6000) -> str:
    rows = []
    for item in (messages or [])[-limit:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if isinstance(content, list):
            content = " ".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        text = re.sub(r"\s+", " ", str(content or "")).strip()
        if text:
            rows.append(f"{item['role']}: {text[:1000]}")
    history = "\n".join(rows)
    return history[-max_chars:] if len(history) > max_chars else history


def _message_usage(message) -> dict:
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("input_tokens", 0) or 0,
            "completion_tokens": usage.get("output_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
        }
    metadata = getattr(message, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    return dict(token_usage or {})


def _concept_decision(
    *,
    model: str,
    current_request: str,
    messages: list[dict],
    project: dict,
) -> tuple[VideoConceptDecision, dict]:
    system_prompt = (
        "你是 LumaNova 视频创作 Agent 的创意导演，当前只负责视频概念规划与确认。"
        "最终产品是一条独立完整、最长 15 秒的短视频，不是短剧、网剧、连续小说、电影或多集内容。"
        "用户提到网剧、短剧、电影或小说时，只能借用题材、节奏或视觉气质，必须收缩成 15 秒内闭环的单条短视频。"
        "在概念获得用户明确确认前，绝不能进入主体设计、生图、场景或其他素材阶段。"
        "用户说‘你来想想’或允许自由发挥时，应直接给出概念草案，而不是继续盘问，也不能直接生成素材。"
        "首次收到清晰想法时也要先整理成概念草案供确认。"
        "已有草案时，准确区分用户是在确认、要求修改，还是只询问相关问题。"
        "概念必须只有一个中心事件、一个完整转折或情绪落点，并尽量控制在一到三个重要主体。"
        "beats 是可直接指导后续制作的镜头级概念脚本。对于默认 15 秒视频，必须按时间顺序输出 9 到 15 个微节拍，并按动作复杂度决定数量，"
        "覆盖建立、动作起势、推进、反应、转折、高潮、结果与结尾；每个节拍必须写清可见动作、人物状态和画面变化，不能只写抽象剧情总结。"
        "assistant_reply 使用自然中文，不得暴露结构字段、工具、JSON、阶段机或系统实现。\n\n"
        + _skill_body()
    )
    brief = project.get("project_brief") or {}
    payload = {
        "current_request": str(current_request or "").strip(),
        "recent_conversation": _compact_history(messages),
        "project_status": project.get("status"),
        "current_concept": brief.get("concept"),
    }
    request_messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                json.dumps(payload, ensure_ascii=False, default=str)
                + "\n\n请调用 VideoConceptDecision 返回本轮唯一决策。"
            )
        ),
    ]
    llm = build_agent_llm(model=model)
    response = llm.bind_tools([VideoConceptDecision]).invoke(request_messages)
    calls = getattr(response, "tool_calls", None) or []
    if not calls:
        # Some OpenAI-compatible providers occasionally answer in prose despite
        # an available function schema. Retry this dedicated decision call with
        # an explicit required function instead of failing the workflow.
        response = llm.bind_tools([VideoConceptDecision], tool_choice="VideoConceptDecision").invoke(request_messages)
        calls = getattr(response, "tool_calls", None) or []
    if not calls:
        raise RuntimeError("video concept director returned no structured decision")
    try:
        decision = VideoConceptDecision.model_validate(calls[0].get("args") or {})
    except Exception as exc:
        raise RuntimeError(f"video concept director returned an invalid decision: {exc}") from exc
    return decision, _message_usage(response)


def _normalize_concept(concept: ShortVideoConcept) -> dict[str, Any]:
    data = concept.model_dump()
    try:
        duration = int(data.get("duration_seconds") or 15)
    except (TypeError, ValueError):
        duration = 15
    data["duration_seconds"] = max(5, min(duration, 15))
    data["format"] = "single_short_video"
    data["beats"] = [item for item in (data.get("beats") or [])[:15] if item.get("content")]
    data["subjects"] = [
        item for item in (data.get("subjects") or [])[:3]
        if item.get("display_name") or item.get("role")
    ]
    return data


def _render_concept(concept: dict, *, revised=False) -> str:
    title = concept.get("title") or "未命名短视频"
    lines = [
        f"**{'修改后的' if revised else ''}短视频概念 · {title}**",
        "",
        f"- **建议时长**：{concept.get('duration_seconds') or 15} 秒",
        f"- **主题**：{concept.get('theme') or '待确认'}",
        f"- **调性**：{concept.get('tone') or '待确认'}",
    ]
    if concept.get("audience"):
        lines.append(f"- **目标观众**：{concept['audience']}")
    lines.extend([
        "",
        f"**一句话创意**\n{concept.get('logline') or '待补充'}",
        "",
        f"**内容大纲**\n{concept.get('synopsis') or '待补充'}",
        "",
        "**概念脚本**",
    ])
    for index, beat in enumerate(concept.get("beats") or [], start=1):
        heading = beat.get("time_range") or f"段落 {index}"
        purpose = f" · {beat.get('purpose')}" if beat.get("purpose") else ""
        lines.append(f"{index}. **{heading}{purpose}**：{beat.get('content') or ''}")
        if beat.get("visual"):
            lines.append(f"   画面：{beat['visual']}")
    subjects = concept.get("subjects") or []
    if subjects:
        lines.extend(["", "**主要主体**"])
        for subject in subjects:
            name = subject.get("display_name") or "未命名主体"
            role = subject.get("role") or "核心主体"
            visual = subject.get("visual_direction") or "后续确认视觉设计"
            lines.append(f"- **{name}**：{role}；{visual}")
    if concept.get("visual_direction"):
        lines.extend(["", f"**视觉方向**\n{concept['visual_direction']}"])
    if concept.get("audio_direction"):
        lines.extend(["", f"**声音方向**\n{concept['audio_direction']}"])
    if concept.get("ending"):
        lines.extend(["", f"**结尾落点**\n{concept['ending']}"])
    lines.extend([
        "",
        "先确认这版视频概念。你可以直接说“就按这个”，也可以告诉我需要修改的主题、节奏或结尾；确认后再进入主体素材设计。",
    ])
    return "\n".join(lines)


class _Trace:
    def __init__(self, callback=None):
        self.callback = callback
        self.items = []
        self.used_tools = []
        self.usage = {}

    def add_usage(self, usage):
        self.usage = add_chat_usage(self.usage, usage) or self.usage

    def call(self, name, content):
        self.used_tools.append(name)
        self.items.append({
            "type": "ToolCall",
            "name": name,
            "content": content,
            "tool_calls": [{"name": name, "args": content, "type": "tool_call"}],
        })
        self.emit()

    def result(self, name, content):
        self.items.append({
            "type": "ToolMessage",
            "name": name,
            "content": content,
            "tool_calls": [],
        })
        self.emit()

    def answer(self, content):
        self.items.append({
            "type": "ModelAnswer",
            "name": None,
            "content": content,
            "tool_calls": [],
        })
        self.emit()

    def emit(self):
        if self.callback:
            self.callback(
                trace=list(self.items),
                used_tools=list(dict.fromkeys(self.used_tools)),
                usage=self.usage,
            )


def run_video_creation_chat(
    client_id,
    *,
    session_id: str,
    model: str,
    messages: list[dict],
    user_message: str,
    image_path: str | None = None,
    trace_callback=None,
) -> dict:
    project = get_or_create_video_project(client_id, session_id)
    if project.get("stage") == "keyframe_material" and not (project.get("project_brief") or {}).get("scene_materials"):
        # Sessions created before scene assets existed resume at the new stage
        # on their next turn instead of failing inside storyboard generation.
        project = update_video_project(
            client_id,
            session_id,
            stage="scene_material",
            status="scene_ready",
            project_brief=project.get("project_brief") or {},
        )
    if project.get("stage") == "keyframe_material":
        return run_video_keyframe_chat(
            client_id,
            session_id=session_id,
            model=model,
            messages=messages,
            user_message=user_message,
            image_path=image_path,
            trace_callback=trace_callback,
        )
    if project.get("stage") == "scene_material":
        return run_video_scene_chat(
            client_id,
            session_id=session_id,
            model=model,
            messages=messages,
            user_message=user_message,
            image_path=image_path,
            trace_callback=trace_callback,
        )
    if project.get("stage") == "video_prompt_ready":
        # A reviewed prompt remains editable until the user explicitly asks to
        # submit it. This keeps a casual follow-up from becoming a paid video job.
        request = str(user_message or "").strip().lower()
        submit_markers = ("开始生成视频", "生成最终视频", "开始出视频", "确认并生成", "确认生成", "开始制作视频", "生成啊开始")
        if any(marker in request for marker in submit_markers):
            return run_video_generation_chat(
                client_id,
                session_id=session_id,
                model=model,
                messages=messages,
                user_message=user_message,
                image_path=image_path,
                trace_callback=trace_callback,
            )
        return run_video_keyframe_chat(
            client_id,
            session_id=session_id,
            model=model,
            messages=messages,
            user_message=user_message,
            image_path=image_path,
            trace_callback=trace_callback,
        )
    if project.get("stage") == "video_generating":
        return run_video_generation_chat(
            client_id,
            session_id=session_id,
            model=model,
            messages=messages,
            user_message=user_message,
            image_path=image_path,
            trace_callback=trace_callback,
        )
    if project.get("stage") != "concept_planning":
        return run_video_subject_chat(
            client_id,
            session_id=session_id,
            model=model,
            messages=messages,
            user_message=user_message,
            image_path=image_path,
            trace_callback=trace_callback,
        )

    trace = _Trace(trace_callback)
    decision, usage = _concept_decision(
        model=model,
        current_request=user_message,
        messages=messages,
        project=project,
    )
    trace.add_usage(usage)
    brief = dict(project.get("project_brief") or {})
    existing_concept = brief.get("concept") if isinstance(brief.get("concept"), dict) else None

    if decision.intent == "approve_concept":
        if not existing_concept or project.get("status") != "awaiting_concept_review":
            reply = "目前还没有可确认的视频概念。先告诉我你想做的大致主题；如果没有想法，也可以让我直接设计一版。"
        else:
            trace.call("approve_video_concept", {
                "title": existing_concept.get("title") or "未命名短视频",
                "duration_seconds": existing_concept.get("duration_seconds") or 15,
            })
            project = update_video_project(
                client_id,
                session_id,
                stage="subject_material",
                status="concept_approved",
                project_brief=brief,
            )
            trace.result("approve_video_concept", {"status": "1", "next": "subject_material"})
            subject_names = [
                item.get("display_name")
                for item in (existing_concept.get("subjects") or [])
                if item.get("display_name")
            ]
            subject_text = "、".join(subject_names) or "核心主体"
            reply = (
                f"视频概念已经确认。接下来开始准备 **{subject_text}** 的主体素材，"
                "让后续画面能够保持形象一致。你可以先补充外观要求，也可以直接回复“开始生成主体素材”，由我根据已确认的概念完成设计。"
            )
    elif decision.intent in {"draft_concept", "revise_concept"}:
        if decision.concept is None:
            reply = decision.assistant_reply.strip() or "我还需要一个能决定视频方向的关键信息。你也可以让我自由设计一版。"
        else:
            concept = _normalize_concept(decision.concept)
            revision = int(brief.get("concept_revision") or 0) + 1
            brief.update({
                "concept": concept,
                "concept_revision": revision,
                "format_constraints": {
                    "format": "single_short_video",
                    "max_duration_seconds": 15,
                    "serial_content": False,
                },
            })
            trace.call("save_video_concept", {
                "title": concept.get("title"),
                "duration_seconds": concept.get("duration_seconds"),
                "revision": revision,
            })
            project = update_video_project(
                client_id,
                session_id,
                stage="concept_planning",
                status="awaiting_concept_review",
                project_brief=brief,
            )
            trace.result("save_video_concept", {
                "status": "1",
                "revision": revision,
                "awaiting_confirmation": True,
            })
            reply = _render_concept(
                concept,
                revised=decision.intent == "revise_concept" or revision > 1,
            )
    elif decision.intent == "clarify":
        reply = decision.assistant_reply.strip()
        if not reply:
            missing = decision.missing_decision.strip() or "你希望这条短视频主要表达什么"
            reply = f"在整理视频概念前，我想先确认：{missing}？如果你暂时没有想法，也可以直接让我自由设计。"
    else:
        reply = decision.assistant_reply.strip() or (
            "我负责先把你的想法整理成一条不超过 15 秒的完整视频概念，确认后再进入主体素材设计。"
        )

    trace.answer(reply)
    return {
        "content": reply,
        "usage": trace.usage,
        "trace": trace.items,
        "used_tools": list(dict.fromkeys(trace.used_tools)),
    }
