"""Script-bound keyframe planning and material generation runtime."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_llm import build_agent_llm
from chat_protocol import add_chat_usage
from sql import (
    get_or_create_video_project,
    list_video_keyframes,
    list_video_subjects,
    replace_video_keyframe_plan,
    update_video_keyframe_asset_inspection,
    update_video_project,
)
from video_keyframe_tools import VideoKeyframeToolbox, expand_storyboard_panels, public_keyframe_result
from vision_context_runtime import (
    run_video_storyboard_review,
    run_video_storyboard_sheet_quality_inspection,
    run_video_storyboard_sheet_review,
)


SKILL_PATH = (
    Path(__file__).resolve().parent
    / "workflow_skills"
    / "video_keyframe_material"
    / "SKILL.md"
)


class KeyframeDirection(BaseModel):
    source_beat_index: int
    subject_refs: list[str] = Field(default_factory=list)
    scene_refs: list[str] = Field(default_factory=list)
    frame_role: str = ""
    shot_size: str = ""
    camera_angle: str = ""
    composition: str = ""
    action_state: str = ""
    environment_state: str = ""
    lighting: str = ""
    continuity_notes: str = ""


class VideoKeyframeDecision(BaseModel):
    intent: Literal[
        "generate_keyframes",
        "revise_storyboard",
        "approve_storyboard",
        "approve_keyframes",
        "clarify",
        "reply",
    ] = "reply"
    keyframes: list[KeyframeDirection] = Field(default_factory=list)
    target_keyframe_refs: list[str] = Field(default_factory=list)
    start_video: bool = False
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


def _compact_history(messages, *, limit=12, max_chars=5000) -> str:
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
            rows.append(f"{item['role']}: {text[:800]}")
    history = "\n".join(rows)
    return history[-max_chars:] if len(history) > max_chars else history


def _concept(project: dict) -> dict:
    brief = project.get("project_brief") or {}
    value = brief.get("concept")
    return value if isinstance(value, dict) else {}


def _scenes(project: dict) -> list[dict]:
    value = (project.get("project_brief") or {}).get("scene_materials")
    return [dict(item) for item in value] if isinstance(value, list) else []


def _resolve_scene(scenes: list[dict], reference: str) -> dict | None:
    value = str(reference or "").strip().lower()
    if not value:
        return None
    for scene in scenes:
        if value in {str(scene.get("scene_id") or "").lower(), str(scene.get("display_name") or "").lower()}:
            return scene
    number = re.search(r"(\d+)", value)
    if number:
        return next((item for item in scenes if item.get("ordinal") == int(number.group(1))), None)
    return None


def _resolve_subject(subjects: list[dict], reference: str) -> dict | None:
    value = str(reference or "").strip()
    if not value:
        return None
    lowered = value.lower()
    for subject in subjects:
        if lowered in {
            str(subject.get("subject_id") or "").lower(),
            str(subject.get("display_name") or "").lower(),
        }:
            return subject
    number_match = re.fullmatch(r"(?:主体|人物|角色)?\s*(\d+)", value)
    if number_match:
        ordinal = int(number_match.group(1))
        return next((item for item in subjects if item.get("ordinal") == ordinal), None)
    return None


def _required_subjects(project: dict, subjects: list[dict]) -> tuple[list[dict], list[str]]:
    concept_subjects = _concept(project).get("subjects") or []
    if not concept_subjects:
        required = list(subjects)
        return required, []
    required = []
    missing = []
    seen = set()
    for source in concept_subjects:
        if not isinstance(source, dict):
            continue
        name = str(source.get("display_name") or "").strip()
        matched = _resolve_subject(subjects, name)
        if not matched:
            if name:
                missing.append(name)
            continue
        if matched.get("subject_id") not in seen:
            seen.add(matched.get("subject_id"))
            required.append(matched)
    return required, missing


def _subject_catalog(subjects: list[dict]) -> list[dict]:
    return [
        {
            "subject_id": item.get("subject_id"),
            "ordinal": item.get("ordinal"),
            "display_name": item.get("display_name"),
            "subject_type": item.get("subject_type"),
            "story_role": item.get("story_role"),
            "spec": item.get("spec") or {},
        }
        for item in subjects
    ]


def _frame_catalog(frames: list[dict]) -> list[dict]:
    return [
        {
            "keyframe_id": item.get("keyframe_id"),
            "ordinal": item.get("ordinal"),
            "source_beat_index": item.get("source_beat_index"),
            "time_range": item.get("time_range"),
            "status": item.get("status"),
            "has_image": bool(item.get("current_asset_id")),
            "is_approved": bool(item.get("approved_asset_id")),
        }
        for item in frames
    ]


def _director_decision(
    *,
    model: str,
    current_request: str,
    messages: list[dict],
    project: dict,
    subjects: list[dict],
    scenes: list[dict],
    frames: list[dict],
    repair_feedback: str = "",
) -> tuple[VideoKeyframeDecision, dict]:
    concept = _concept(project)
    system_prompt = (
        "你是 Luma 视频生成流水线的关键帧导演。"
        "当前输入中的 confirmed_script 是已经由用户确认的只读脚本事实，绝对不能改写剧情、替换事件、改变结局或添加新事件。"
        "你的任务是为每个脚本节拍规划一个分镜格；运行时会把所有分镜格一次性生成到一张连续的分镜图合集，而不是分别生成独立图片。"
        "每格只补充镜头景别、机位、构图、动作瞬间、环境状态、光线与上一格、下一格的连续性要求。"
        "source_beat_index 使用从 1 开始的脚本节拍序号；首次生成时每个节拍必须且只能输出一项，不能遗漏或重复。"
        "subject_refs 只能引用 approved_subjects 中的 subject_id 或 display_name，不得创造新主体。"
        "scene_refs 只能引用 approved_scenes 的 scene_id 或 display_name；每一个节拍必须引用至少一个对应的场景。"
        "脚本节拍明确提到的主体必须出现在对应 subject_refs 中；仅列出画面实际出现或必须作为环境参考的主体。"
        "脚本明确要求字幕或字样时，将它作为后期合成要求保留安全区，不得擅自删除，也不得自行新增其他文字。"
        "已有分镜计划时不要再次制定新计划，除非后续实现明确提供了重做能力。"
        "用户要求开始、继续或完成分镜时选择 generate_keyframes。分镜图合集生成后会由 Vision 结合完整初版脚本统一审查并写出唯一总视频提示词，不需要逐张人工审批。"
        "当 existing_keyframes 对应的完整分镜图和总视频提示词已经存在时，用户提出具体画面、镜头、动作、节奏或连续性修改，必须选择 revise_storyboard，"
        "change_request 只保留用户本轮对分镜图合集的明确修改；用户明确确认分镜与提示词时选择 approve_storyboard。"
        "只有用户明确同时表达‘确认并开始生成视频’时，start_video 才为 true；仅确认分镜或提示词时必须为 false。"
        "assistant_reply 必须自然简洁，不得暴露字段、JSON、数据库、工具或系统实现。\n\n"
        + _skill_body()
    )
    payload = {
        "current_request": str(current_request or "").strip(),
        "recent_conversation": _compact_history(messages),
        "project_status": project.get("status"),
        "confirmed_script": {
            "title": concept.get("title"),
            "duration_seconds": concept.get("duration_seconds"),
            "visual_direction": concept.get("visual_direction"),
            "beats": [
                {"source_beat_index": index, **beat}
                for index, beat in enumerate(concept.get("beats") or [], start=1)
                if isinstance(beat, dict)
            ],
        },
        "approved_subjects": _subject_catalog(subjects),
        "approved_scenes": [
            {
                "scene_id": item.get("scene_id"),
                "ordinal": item.get("ordinal"),
                "display_name": item.get("display_name"),
                "setting": item.get("setting"),
                "visual_direction": item.get("visual_direction"),
                "related_beat_indexes": item.get("related_beat_indexes") or [],
            }
            for item in scenes if item.get("approved") and item.get("asset_id")
        ],
        "existing_keyframes": _frame_catalog(frames),
        "validation_feedback": repair_feedback,
    }
    response = build_agent_llm(model=model).bind_tools([VideoKeyframeDecision]).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                json.dumps(payload, ensure_ascii=False, default=str)
                + "\n\n请调用 VideoKeyframeDecision 返回本轮唯一决策。"
            )
        ),
    ])
    calls = getattr(response, "tool_calls", None) or []
    if not calls:
        raise RuntimeError("video keyframe director returned no structured decision")
    try:
        decision = VideoKeyframeDecision.model_validate(calls[0].get("args") or {})
    except Exception as exc:
        raise RuntimeError(f"video keyframe director returned an invalid decision: {exc}") from exc
    return decision, _message_usage(response)


def _beat_fingerprint(beat: dict) -> str:
    payload = {
        "time_range": beat.get("time_range") or "",
        "purpose": beat.get("purpose") or "",
        "content": beat.get("content") or "",
        "visual": beat.get("visual") or "",
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compile_plan(
    project: dict,
    subjects: list[dict],
    directions: list[KeyframeDirection],
    scenes: list[dict] | None = None,
) -> list[dict]:
    beats = [item for item in (_concept(project).get("beats") or []) if isinstance(item, dict)]
    if not beats:
        raise ValueError("已确认脚本没有可用的节拍，不能建立关键帧计划")
    expected = list(range(1, len(beats) + 1))
    actual = [item.source_beat_index for item in directions]
    if sorted(actual) != expected or len(actual) != len(set(actual)):
        raise ValueError(f"关键帧计划必须完整覆盖脚本节拍 {expected}，实际为 {actual}")

    direction_by_index = {item.source_beat_index: item for item in directions}
    rows = []
    for source_beat_index, beat in enumerate(beats, start=1):
        direction = direction_by_index[source_beat_index]
        resolved = []
        seen = set()
        for reference in direction.subject_refs:
            subject = _resolve_subject(subjects, reference)
            if not subject:
                raise ValueError(f"脚本节拍 {source_beat_index} 引用了未知主体：{reference}")
            if subject.get("subject_id") not in seen:
                seen.add(subject.get("subject_id"))
                resolved.append(subject)

        beat_text = " ".join(str(beat.get(key) or "") for key in ("purpose", "content", "visual"))
        for subject in subjects:
            name = str(subject.get("display_name") or "").strip()
            if name and name in beat_text and subject.get("subject_id") not in seen:
                seen.add(subject.get("subject_id"))
                resolved.append(subject)

        resolved_scenes = []
        for reference in direction.scene_refs:
            scene = _resolve_scene(scenes or [], reference)
            if not scene:
                raise ValueError(f"脚本节拍 {source_beat_index} 引用了未知场景：{reference}")
            if scene.get("scene_id") not in {item.get("scene_id") for item in resolved_scenes}:
                resolved_scenes.append(scene)
        if scenes and not resolved_scenes:
            linked = [item for item in scenes if source_beat_index in (item.get("related_beat_indexes") or [])]
            resolved_scenes = linked or list(scenes[:1])
        if scenes and not resolved_scenes:
            raise ValueError(f"脚本节拍 {source_beat_index} 缺少场景引用")

        plan = direction.model_dump(exclude={"source_beat_index", "subject_refs"})
        plan["scene_ids"] = [item.get("scene_id") for item in resolved_scenes]
        rows.append({
            "source_beat_index": source_beat_index,
            "source_beat_fingerprint": _beat_fingerprint(beat),
            "time_range": beat.get("time_range") or "",
            "purpose": beat.get("purpose") or "",
            "script_content": beat.get("content") or "",
            "script_visual": beat.get("visual") or "",
            "subject_ids": [item.get("subject_id") for item in resolved],
            "plan": plan,
        })
    return rows


def _resolve_keyframes(frames: list[dict], references: list[str]) -> list[dict]:
    if not references:
        return list(frames)
    selected = []
    seen = set()
    for reference in references:
        value = str(reference or "").strip()
        matched = next(
            (
                item
                for item in frames
                if value.lower() == str(item.get("keyframe_id") or "").lower()
                or value == str(item.get("time_range") or "")
            ),
            None,
        )
        if not matched:
            number = re.search(r"(\d+)", value)
            if number:
                ordinal = int(number.group(1))
                matched = next((item for item in frames if item.get("ordinal") == ordinal), None)
        if matched and matched.get("keyframe_id") not in seen:
            seen.add(matched.get("keyframe_id"))
            selected.append(matched)
    return selected


def _segment_duration(time_range: str) -> int:
    numbers = [int(value) for value in re.findall(r"\d+", str(time_range or ""))]
    if len(numbers) >= 2:
        return max(1, min(numbers[-1] - numbers[0], 15))
    return 10


def _review_storyboards(
    *,
    client_id: int,
    session_id: str,
    project: dict,
    frames: list[dict],
    subjects: list[dict],
    scenes: list[dict],
    toolbox: VideoKeyframeToolbox,
    trace: _Trace,
) -> tuple[list[dict], bool]:
    concept = _concept(project)
    subject_by_id = {item.get("subject_id"): item for item in subjects}
    scene_by_id = {item.get("scene_id"): item for item in scenes}
    reviews = []
    for frame in frames:
        asset = toolbox.current_asset(frame)
        if not asset or not (asset.get("public_url") or asset.get("local_path")):
            return reviews, False
        frame_subjects = [subject_by_id[item] for item in frame.get("subject_ids") or [] if item in subject_by_id]
        frame_scenes = [scene_by_id[item] for item in (frame.get("plan") or {}).get("scene_ids") or [] if item in scene_by_id]
        references = []
        for item in frame_subjects:
            references.append({"picture": len(references) + 1, "kind": "角色", "name": item.get("display_name"), "purpose": "锁定人物形象与服装"})
        for item in frame_scenes:
            references.append({"picture": len(references) + 1, "kind": "场景", "name": item.get("display_name"), "purpose": "锁定场景空间、光源、道具与氛围"})
        context = {
            "original_video_concept": {
                "title": concept.get("title"),
                "duration_seconds": concept.get("duration_seconds"),
                "theme": concept.get("theme"),
                "tone": concept.get("tone"),
                "visual_direction": concept.get("visual_direction"),
                "audio_direction": concept.get("audio_direction"),
            },
            "current_script_beat": {
                "time_range": frame.get("time_range"),
                "purpose": frame.get("purpose"),
                "content": frame.get("script_content"),
                "visual": frame.get("script_visual"),
                "frame_plan": frame.get("plan") or {},
            },
            "segment_duration_seconds": _segment_duration(frame.get("time_range")),
            "reference_images": references,
            "continuity_boundary": "只描述当前节拍从分镜图起始状态到该节拍结尾状态的过程；不得借用前后节拍的事件。",
        }
        trace.call("review_video_storyboard", {"ordinal": frame.get("ordinal"), "time_range": frame.get("time_range")})
        result = run_video_storyboard_review(
            image_ref=asset.get("public_url") or asset.get("local_path"),
            production_context=context,
        )
        trace.add_usage(result.get("usage") or {})
        report = result.get("report") or {}
        trace.result("review_video_storyboard", {"status": "1" if report else "0", "ordinal": frame.get("ordinal"), "duration_seconds": report.get("duration_seconds") if report else None})
        if not report:
            return reviews, False
        inspection = {"storyboard_review": report, "production_context": context}
        update_video_keyframe_asset_inspection(client_id, session_id, frame.get("keyframe_id"), asset.get("asset_id"), inspection)
        reviews.append({"ordinal": frame.get("ordinal"), "time_range": frame.get("time_range"), "report": report})
    return reviews, True


def _review_storyboard_sheet(
    *,
    project: dict,
    frames: list[dict],
    subjects: list[dict],
    scenes: list[dict],
    storyboard_asset: dict,
    trace: _Trace,
) -> tuple[dict, bool]:
    concept = _concept(project)
    references = [{"picture": 1, "kind": "完整分镜图合集", "name": "整条视频的连续时序与镜头蓝图", "purpose": "锁定各格的构图、走位、动作和转场关系"}]
    for item in subjects:
        references.append({"picture": len(references) + 1, "kind": "角色", "name": item.get("display_name"), "purpose": "锁定角色身份、外观与服装"})
    for item in scenes:
        references.append({"picture": len(references) + 1, "kind": "场景", "name": item.get("display_name"), "purpose": "锁定空间、光源、道具与氛围"})
    context = {
        "original_video_concept": {
            "title": concept.get("title"), "duration_seconds": min(int(concept.get("duration_seconds") or 15), 15),
            "theme": concept.get("theme"), "tone": concept.get("tone"), "visual_direction": concept.get("visual_direction"),
            "audio_direction": concept.get("audio_direction"), "synopsis": concept.get("synopsis"),
            "beats": concept.get("beats") or [],
        },
        "storyboard_panels": [
            {"panel": item.get("ordinal"), "time_range": item.get("time_range"), "purpose": item.get("purpose"),
             "content": item.get("script_content"), "visual": item.get("script_visual"), "direction": item.get("plan") or {}}
            for item in frames
        ],
        "reference_images": references,
        "hard_constraints": ["输出一条完整连续视频", "总时长1到15秒", "不得拆分为多个视频任务", "每格结束状态必须衔接下一格起始状态"],
    }
    trace.call("review_storyboard_sheet", {"panel_count": len(frames), "target_duration_seconds": context["original_video_concept"]["duration_seconds"]})
    result = run_video_storyboard_sheet_review(
        image_ref=storyboard_asset.get("public_url") or storyboard_asset.get("local_path"),
        production_context=context,
    )
    trace.add_usage(result.get("usage") or {})
    report = result.get("report") or {}
    trace.result("review_storyboard_sheet", {"status": "1" if report else "0", "duration_seconds": report.get("duration_seconds") if report else None})
    if not report and result.get("error"):
        raise RuntimeError(f"storyboard Vision review failed: {result['error']}")
    return report, bool(report)


def _inspect_storyboard_sheet_quality(
    *,
    project: dict,
    frames: list[dict],
    storyboard_asset: dict,
    trace: _Trace,
) -> dict:
    """Keep visual QA separate from the later Vision production-prompt pass."""
    concept = _concept(project)
    expected_panels = expand_storyboard_panels(frames)
    context = {
        "concept_title": concept.get("title"),
        "target_duration_seconds": min(int(concept.get("duration_seconds") or 15), 15),
        "expected_panel_count": len(expected_panels),
        "expected_timeline": [
            {
                "panel": item.get("panel"),
                "source_beat_index": item.get("source_beat_index"),
                "camera_moment": item.get("camera_moment"),
            }
            for item in expected_panels
        ],
        "quality_scope": "仅评估合集中可见的绘制质量、镜头可读性与相邻格连续性；不改写剧情和视频提示词。",
    }
    trace.call("inspect_storyboard_sheet_quality", {"expected_panel_count": len(expected_panels)})
    result = run_video_storyboard_sheet_quality_inspection(
        image_ref=storyboard_asset.get("public_url") or storyboard_asset.get("local_path"),
        quality_context=context,
    )
    trace.add_usage(result.get("usage") or {})
    report = result.get("report") or {}
    trace.result("inspect_storyboard_sheet_quality", {
        "status": "1" if report else "0",
        "score": report.get("score") if report else None,
        "panel_count_estimate": report.get("panel_count_estimate") if report else None,
        "passed": report.get("passed") if report else None,
    })
    return report


def _render_storyboard_review(storyboard_asset: dict, report: dict, *, revised: bool = False) -> str:
    prompt = str(report.get("production_prompt") or "").strip()
    prefix = "分镜图已按你的意见更新" if revised else "完整连续分镜图已生成"
    return (
        f"{prefix}，请先审阅画面连续性和下面的最终视频生成提示词。\n\n"
        f"{storyboard_asset.get('public_url')}\n\n"
        "**最终视频生成提示词（审阅稿）**\n"
        f"```text\n{prompt}\n```\n\n"
        "可以直接告诉我需要修改哪些分镜、动作、镜头或节奏；确认无误后回复“确认分镜并开始生成视频”。"
    )


def _is_explicit_storyboard_submission_request(message: str) -> bool:
    """Approval is a product action, not an LLM interpretation problem."""
    value = re.sub(r"\s+", "", str(message or "").lower())
    if not value:
        return False
    revision_markers = ("修改", "改成", "调整", "替换", "不要", "删掉", "增加", "减少", "但是")
    if any(marker in value for marker in revision_markers):
        return False
    exact_confirmations = {"确认", "可以", "没问题", "就这样", "开始", "开始吧", "生成", "生成吧", "生成啊开始"}
    if value in exact_confirmations:
        return True
    return any(
        marker in value
        for marker in (
            "确认分镜并开始生成视频",
            "确认并生成",
            "确认生成",
            "开始生成视频",
            "开始制作视频",
            "生成最终视频",
            "开始出视频",
        )
    )


class _Trace:
    def __init__(self, callback=None):
        self.callback = callback
        self.items = []
        self.used_tools = []
        self.usage = {}

    def add_usage(self, usage):
        self.usage = add_chat_usage(self.usage, usage) or self.usage

    def call(self, name: str, arguments: dict):
        self.used_tools.append(name)
        self.items.append({
            "type": "ToolCall",
            "name": name,
            "content": arguments,
            "tool_calls": [{"name": name, "args": arguments, "type": "tool_call"}],
        })
        self.emit()

    def result(self, name: str, result: dict):
        self.items.append({"type": "ToolMessage", "name": name, "content": result, "tool_calls": []})
        self.emit()

    def answer(self, content: str):
        self.items.append({"type": "ModelAnswer", "name": None, "content": content, "tool_calls": []})
        self.emit()

    def emit(self):
        if self.callback:
            self.callback(
                trace=list(self.items),
                used_tools=list(dict.fromkeys(self.used_tools)),
                usage=self.usage,
            )


def run_video_keyframe_chat(
    client_id,
    *,
    session_id: str,
    model: str,
    messages: list[dict],
    user_message: str,
    image_path: str | None = None,
    trace_callback=None,
) -> dict:
    del image_path
    project = get_or_create_video_project(client_id, session_id)
    subjects = list_video_subjects(client_id, session_id)
    scenes = _scenes(project)
    required_subjects, missing_subjects = _required_subjects(project, subjects)
    unapproved = [item for item in required_subjects if not item.get("approved_asset_id")]
    unapproved_scenes = [item for item in scenes if not item.get("approved") or not item.get("asset_id")]
    trace = _Trace(trace_callback)

    if missing_subjects or unapproved:
        names = missing_subjects + [item.get("display_name") or "未命名主体" for item in unapproved]
        reply = f"还不能开始关键帧：{'、'.join(names)} 的正式主体素材尚未全部确认。先完成这些主体的定稿，避免后续画面形象漂移。"
        trace.answer(reply)
        return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": []}
    if not scenes or unapproved_scenes:
        reply = "还不能开始分镜构建：正式场景素材尚未全部确认。先完成场景基准图，后续每一张分镜才会同时锁定人物与空间。"
        trace.answer(reply)
        return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": []}

    frames = list_video_keyframes(client_id, session_id)
    brief = dict(project.get("project_brief") or {})
    existing_sheet = brief.get("storyboard_sheet") if isinstance(brief.get("storyboard_sheet"), dict) else {}
    existing_report = ((brief.get("video_production") or {}).get("report") or {})
    if (
        existing_sheet.get("public_url")
        and existing_report.get("production_prompt")
        and _is_explicit_storyboard_submission_request(user_message)
    ):
        update_video_project(
            client_id,
            session_id,
            stage="video_prompt_ready",
            status="production_prompt_approved",
            project_brief=brief,
        )
        from video_generation_runtime import run_video_generation_chat
        return run_video_generation_chat(
            client_id,
            session_id=session_id,
            model=model,
            messages=messages,
            user_message=user_message,
            trace_callback=trace_callback,
        )

    decision, usage = _director_decision(
        model=model,
        current_request=user_message,
        messages=messages,
        project=project,
        subjects=required_subjects,
        scenes=scenes,
        frames=frames,
    )
    trace.add_usage(usage)

    if decision.intent != "generate_keyframes":
        reply = decision.assistant_reply.strip() or "角色与场景素材已经准备好。你可以让我按已确认脚本生成整组分镜图。"
        trace.answer(reply)
        return {
            "content": reply,
            "usage": trace.usage,
            "trace": trace.items,
            "used_tools": list(dict.fromkeys(trace.used_tools)),
        }

    if not frames:
        try:
            plan_rows = _compile_plan(project, required_subjects, decision.keyframes, scenes)
        except ValueError as first_error:
            repaired, repair_usage = _director_decision(
                model=model,
                current_request=user_message,
                messages=messages,
                project=project,
                subjects=required_subjects,
                scenes=scenes,
                frames=[],
                repair_feedback=str(first_error),
            )
            trace.add_usage(repair_usage)
            if repaired.intent != "generate_keyframes":
                raise RuntimeError(f"video keyframe plan repair changed intent: {repaired.intent}")
            plan_rows = _compile_plan(project, required_subjects, repaired.keyframes, scenes)

        trace.call("plan_video_keyframes", {
            "script_revision": (project.get("project_brief") or {}).get("concept_revision"),
            "beat_count": len(plan_rows),
        })
        frames = replace_video_keyframe_plan(client_id, session_id, plan_rows)
        trace.result("plan_video_keyframes", {
            "status": "1",
            "keyframe_count": len(frames),
            "source_beats": [item.get("source_beat_index") for item in frames],
        })
    toolbox = VideoKeyframeToolbox(client_id=client_id, session_id=session_id)
    if existing_sheet.get("public_url"):
        if existing_report.get("production_prompt"):
            if decision.intent == "revise_storyboard":
                change_request = decision.change_request.strip()
                if not change_request:
                    reply = "请告诉我这次希望怎样调整分镜图，例如具体镜头、动作、节奏、画面关系或连续性。"
                    trace.answer(reply)
                    return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}
                if not toolbox.consume_image_quota():
                    reply = "当前图片额度不足，分镜修改还没有执行；现有分镜图和最终提示词已保留。"
                    trace.answer(reply)
                    return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}
                trace.call("revise_storyboard_sheet", {"change_request": change_request})
                revised_result = toolbox.revise_storyboard_sheet(
                    existing_sheet, frames, required_subjects, scenes, change_request,
                )
                trace.result("revise_storyboard_sheet", public_keyframe_result(revised_result))
                revised_asset = toolbox.persist_storyboard_sheet(revised_result)
                if not revised_asset:
                    reply = "这次分镜图修改没有成功，原来的分镜图和最终提示词都已保留。"
                    trace.answer(reply)
                    return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}
                brief.pop("video_production", None)
                brief["storyboard_sheet"] = {
                    "asset_id": revised_asset.get("asset_id"),
                    "public_url": revised_asset.get("public_url"),
                    "local_path": revised_asset.get("local_path"),
                    "panel_count": len(expand_storyboard_panels(frames)),
                }
                brief["storyboard_quality"] = _inspect_storyboard_sheet_quality(
                    project=project, frames=frames, storyboard_asset=revised_asset, trace=trace,
                )
                report, prompts_ready = _review_storyboard_sheet(
                    project=project,
                    frames=frames,
                    subjects=required_subjects,
                    scenes=scenes,
                    storyboard_asset=revised_asset,
                    trace=trace,
                )
                if not prompts_ready:
                    update_video_project(
                        client_id, session_id, stage="keyframe_material", status="storyboard_review_pending", project_brief=brief,
                    )
                    reply = "分镜图已更新，但最终视频生成提示词暂时没有生成成功；分镜图已保留，可以稍后继续。"
                else:
                    brief["video_production"] = {"report": report, "storyboard_asset_id": revised_asset.get("asset_id")}
                    update_video_project(
                        client_id, session_id, stage="keyframe_material", status="awaiting_storyboard_review", project_brief=brief,
                    )
                    reply = _render_storyboard_review(revised_asset, report, revised=True)
                trace.answer(reply)
                return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}

            if decision.intent == "approve_storyboard":
                update_video_project(
                    client_id, session_id, stage="video_prompt_ready", status="production_prompt_approved", project_brief=brief,
                )
                reply = "分镜图和最终视频提示词已确认。"
                if decision.start_video:
                    from video_generation_runtime import run_video_generation_chat
                    return run_video_generation_chat(
                        client_id,
                        session_id=session_id,
                        model=model,
                        messages=messages,
                        user_message=user_message,
                        trace_callback=trace_callback,
                    )
                reply += "需要生成时，回复“开始生成视频”即可。"
                trace.answer(reply)
                return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}

            update_video_project(
                client_id, session_id, stage="keyframe_material", status="awaiting_storyboard_review", project_brief=brief,
            )
            reply = _render_storyboard_review(existing_sheet, existing_report)
            trace.answer(reply)
            return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}

        # A previous Vision attempt may have timed out after the sheet was
        # persisted. Resume only the missing review instead of making users
        # regenerate an expensive storyboard image.
        report, prompts_ready = _review_storyboard_sheet(
            project=project,
            frames=frames,
            subjects=required_subjects,
            scenes=scenes,
            storyboard_asset=existing_sheet,
            trace=trace,
        )
        if prompts_ready:
            brief["video_production"] = {"report": report, "storyboard_asset_id": existing_sheet.get("asset_id")}
            update_video_project(
                client_id, session_id, stage="keyframe_material", status="awaiting_storyboard_review", project_brief=brief,
            )
            reply = _render_storyboard_review(existing_sheet, report)
        else:
            update_video_project(
                client_id, session_id, stage="keyframe_material", status="storyboard_review_pending", project_brief=brief,
            )
            reply = "完整分镜图已保留，但总视频提示词暂时没有生成成功；你可以稍后继续，无需重新生成分镜图。"
        trace.answer(reply)
        return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}

    if not toolbox.consume_image_quota():
        reply = "图片额度已达到当前时段上限，完整分镜计划已经保留，额度恢复后可以继续生成分镜图合集。"
        trace.answer(reply)
        return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}
    trace.call("generate_storyboard_sheet", {
        "panel_count": len(frames),
        "subjects": [item.get("display_name") for item in required_subjects],
        "scenes": [item.get("display_name") for item in scenes],
    })
    result = toolbox.generate_storyboard_sheet(frames, required_subjects, scenes)
    trace.result("generate_storyboard_sheet", public_keyframe_result(result))
    storyboard_asset = toolbox.persist_storyboard_sheet(result)
    if not storyboard_asset:
        reply = "完整分镜图合集这次没有成功生成，已保留分镜计划，可以稍后继续。"
        trace.answer(reply)
        return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}

    brief["storyboard_sheet"] = {
        "asset_id": storyboard_asset.get("asset_id"),
        "public_url": storyboard_asset.get("public_url"),
        "local_path": storyboard_asset.get("local_path"),
        "panel_count": len(expand_storyboard_panels(frames)),
    }
    # This QA is intentionally diagnostic-only. The next Vision call starts
    # from the original concept and rendered sheet, not from this report.
    quality_report = _inspect_storyboard_sheet_quality(
        project=project,
        frames=frames,
        storyboard_asset=storyboard_asset,
        trace=trace,
    )
    brief["storyboard_quality"] = quality_report
    report, prompts_ready = _review_storyboard_sheet(
        project=project,
        frames=frames,
        subjects=required_subjects,
        scenes=scenes,
        storyboard_asset=storyboard_asset,
        trace=trace,
    )
    if prompts_ready:
        brief["video_production"] = {"report": report, "storyboard_asset_id": storyboard_asset.get("asset_id")}
    update_video_project(
        client_id,
        session_id,
        stage="keyframe_material",
        status="awaiting_storyboard_review" if prompts_ready else "storyboard_review_pending",
        project_brief=brief,
    )
    if prompts_ready:
        reply = _render_storyboard_review(storyboard_asset, report)
    else:
        reply = f"完整连续分镜图已生成：\n\n{storyboard_asset.get('public_url')}\n\n但本轮视觉审查没有返回完整的总视频提示词，可以稍后继续。"
    trace.answer(reply)
    return {
        "content": reply,
        "usage": trace.usage,
        "trace": trace.items,
        "used_tools": list(dict.fromkeys(trace.used_tools)),
    }
