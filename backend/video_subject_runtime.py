"""Dedicated runtime for the video creation subject-material stage."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent_llm import build_agent_llm
from chat_protocol import add_chat_usage
from sql import (
    get_or_create_video_project,
    list_video_subjects,
    remove_video_subject_from_roster,
    replace_video_subject_in_roster,
    update_video_project,
)
from video_subject_tools import VideoSubjectToolbox, public_tool_result
from vision_context_runtime import run_vision_context


SKILL_PATH = (
    Path(__file__).resolve().parent
    / "workflow_skills"
    / "video_subject_material"
    / "SKILL.md"
)


class SubjectUpdate(BaseModel):
    subject_ref: str = ""
    display_name: str = ""
    subject_type: str = "other"
    story_role: str = ""
    spec: dict[str, Any] = Field(default_factory=dict)


class SubjectReplacement(BaseModel):
    from_subject_ref: str = ""
    to_subject_ref: str = ""


class ReadinessDecision(BaseModel):
    status: Literal["blocked", "ready", "awaiting_review", "approved"] = "blocked"
    missing_critical_fields: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    rationale: str = ""


class VideoSubjectDecision(BaseModel):
    intent: Literal[
        "clarify",
        "generate_design",
        "revise_design",
        "approve_design",
        "replace_subject",
        "remove_subject",
        "advance_pipeline",
        "offer_preview",
        "generate_preview",
        "reply",
    ] = "clarify"
    readiness: ReadinessDecision = Field(default_factory=ReadinessDecision)
    subject_updates: list[SubjectUpdate] = Field(default_factory=list)
    subject_replacements: list[SubjectReplacement] = Field(default_factory=list)
    target_scope: Literal["selected", "all_pending"] = "selected"
    target_subject_refs: list[str] = Field(default_factory=list)
    change_request: str = ""
    preview_scene: str = ""
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


def _compact_history(messages, *, limit=16, max_chars=7000) -> str:
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
            rows.append(f"{item['role']}: {text[:900]}")
    history = "\n".join(rows)
    return history[-max_chars:] if len(history) > max_chars else history


def _usage_from_message(message) -> dict:
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


def _subject_catalog(subjects: list[dict]) -> list[dict]:
    return [
        {
            "subject_id": item.get("subject_id"),
            "ordinal": item.get("ordinal"),
            "display_name": item.get("display_name"),
            "subject_type": item.get("subject_type"),
            "story_role": item.get("story_role"),
            "status": item.get("status"),
            "has_design": bool(item.get("current_asset_id")),
            "is_approved": bool(item.get("approved_asset_id")),
            "spec": item.get("spec") or {},
        }
        for item in subjects
    ]


def _resolve_subject(subjects: list[dict], reference: str, active_subject_id: str | None = None) -> dict | None:
    value = str(reference or "").strip()
    if value:
        lowered = value.lower()
        for subject in subjects:
            if lowered in {
                str(subject.get("subject_id") or "").lower(),
                str(subject.get("display_name") or "").lower(),
            }:
                return subject
        number_match = re.search(r"(?:主体|人物|角色)?\s*(\d+)", value)
        if number_match:
            ordinal = int(number_match.group(1))
            matched = next((item for item in subjects if item.get("ordinal") == ordinal), None)
            if matched:
                return matched
        # An explicit but unknown reference means a new/unresolved subject. It
        # must never fall through to the active or only existing subject.
        return None
    if active_subject_id:
        matched = next((item for item in subjects if item.get("subject_id") == active_subject_id), None)
        if matched:
            return matched
    return subjects[0] if len(subjects) == 1 else None


def _director_decision(
    *,
    model: str,
    current_request: str,
    messages: list[dict],
    project: dict,
    subjects: list[dict],
    has_uploaded_reference: bool,
    reference_visual_facts: str,
) -> tuple[VideoSubjectDecision, dict]:
    system_prompt = (
        "你是 LumaNova 视频创作流水线中负责主体素材准备的导演 Agent。"
        "你必须根据当前用户请求、有限历史和持久化主体状态，输出一个结构化决策。"
        "project.brief 中的 concept 是用户已经确认的只读上游视频概念，必须把它作为主体身份、角色和视觉方向的事实来源，不得改写或复制回传。"
        "不要执行最终视频、剧本、分镜、场景资产或音频任务；当前只准备主体素材。"
        "模糊需求应自然追问一个最关键的问题，明确需求应直接进入主体设计，不要机械确认。"
        "用户允许自由发挥时，可以补全非核心美术细节。"
        "subject_updates 必须包含本轮新获得或需要更新的主体设定；一个请求包含多个独立主体时分别列出。"
        "每个 subject_updates 项只能描述该项对应的一个主体，严禁复制其他人物、载具或物体的字段。"
        "spec 应是可复用视觉设定，不要放聊天口语、图片版本指代或操作命令。"
        "target_scope 必须明确表达执行范围：用户指定某一个或若干主体时使用 selected；"
        "用户要求开始、继续或完成主体素材但没有点名某一个主体时，使用 all_pending，一次处理所有尚无设计图的主体。"
        "all_pending 时 target_subject_refs 留空；selected 时 target_subject_refs 必须精确列出目标。"
        "target_subject_refs 只能引用已知 subject_id、显示名，或本轮 subject_updates 中的 display_name。"
        "revise_design 的 change_request 只保留本轮可执行视觉改动。"
        "用户明确说‘A 替换为 B’、‘不要 A，改用 B’时，必须使用 replace_subject，"
        "在 subject_replacements 写明 from_subject_ref=A、to_subject_ref=B；这不是普通回复。"
        "用户明确移除某主体且不替换时，使用 remove_subject，并在 target_subject_refs 写明该主体。"
        "用户要求推进、继续分镜、开始后续制作，而主体已全部定稿时，必须使用 advance_pipeline，不能只做口头交接。"
        "只有用户明确确认满意时才能 approve_design；只有已定稿且用户明确要求效果图时才能 generate_preview。"
        "assistant_reply 是准备发给用户的自然中文回复，不得暴露内部字段、工具、JSON 或系统实现。\n\n"
        + _skill_body()
    )
    payload = {
        "current_request": current_request,
        "recent_conversation": _compact_history(messages),
        "project": {
            "stage": project.get("stage"),
            "status": project.get("status"),
            "active_subject_id": project.get("active_subject_id"),
            "brief": project.get("project_brief") or {},
        },
        "known_subjects": _subject_catalog(subjects),
        "uploaded_reference_available": has_uploaded_reference,
        "uploaded_reference_visual_facts": reference_visual_facts,
    }
    llm = build_agent_llm(model=model)
    response = llm.bind_tools([VideoSubjectDecision]).invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                json.dumps(payload, ensure_ascii=False, default=str)
                + "\n\n请调用 VideoSubjectDecision 返回本轮唯一决策。"
            )
        ),
    ])
    calls = getattr(response, "tool_calls", None) or []
    if not calls:
        raise RuntimeError("video subject director returned no structured decision")
    try:
        parsed = VideoSubjectDecision.model_validate(calls[0].get("args") or {})
    except Exception as exc:
        raise RuntimeError(f"video subject director returned an invalid decision: {exc}") from exc
    return parsed, _usage_from_message(response)


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
        self.items.append({
            "type": "ToolMessage",
            "name": name,
            "content": result,
            "tool_calls": [],
        })
        self.emit()

    def answer(self, content: str):
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


def _save_subject_updates(
    toolbox: VideoSubjectToolbox,
    decision: VideoSubjectDecision,
    subjects: list[dict],
) -> tuple[list[dict], list[dict]]:
    saved = []
    known = list(subjects)
    status = "ready" if decision.readiness.status != "blocked" else "collecting"
    for update_model in decision.subject_updates:
        update = update_model.model_dump()
        subject_ref = str(update.get("subject_ref") or "").strip()
        existing = _resolve_subject(known, subject_ref, None) if subject_ref else None
        if existing:
            update["subject_id"] = existing["subject_id"]
        spec = dict(update.get("spec") or {})
        if update.get("display_name"):
            spec.setdefault("identity", update["display_name"])
        update["spec"] = spec
        subject = toolbox.save_subject_spec(update, status=status)
        if subject:
            saved.append(subject)
            known = [item for item in known if item.get("subject_id") != subject.get("subject_id")]
            known.append(subject)
    known.sort(key=lambda item: item.get("ordinal") or 0)
    return saved, known


def _concept_subject_updates(project: dict) -> list[dict]:
    brief = project.get("project_brief") or {}
    concept = brief.get("concept") if isinstance(brief.get("concept"), dict) else {}
    updates = []
    for item in concept.get("subjects") or []:
        if not isinstance(item, dict):
            continue
        display_name = str(item.get("display_name") or "").strip()
        if not display_name:
            continue
        spec = {"identity": display_name}
        visual_direction = str(item.get("visual_direction") or "").strip()
        if visual_direction:
            spec["visual_direction"] = visual_direction
        updates.append({
            "subject_ref": display_name,
            "display_name": display_name,
            "subject_type": str(item.get("subject_type") or "other").strip() or "other",
            "story_role": str(item.get("role") or "").strip(),
            "spec": spec,
        })
    return updates


def _seed_concept_subjects(
    toolbox: VideoSubjectToolbox,
    project: dict,
    subjects: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Persist the approved concept's subject roster before LLM enrichment."""
    known = list(subjects)
    created = []
    for update in _concept_subject_updates(project):
        if _resolve_subject(known, update["display_name"], None):
            continue
        subject = toolbox.save_subject_spec(update, status="ready")
        if subject:
            known.append(subject)
            created.append(subject)
    known.sort(key=lambda item: item.get("ordinal") or 0)
    return created, known


def _concept_subject_approval_state(project: dict, subjects: list[dict]) -> tuple[bool, list[str]]:
    brief = project.get("project_brief") or {}
    concept = brief.get("concept") if isinstance(brief.get("concept"), dict) else {}
    names = [
        str(item.get("display_name") or "").strip()
        for item in (concept.get("subjects") or [])
        if isinstance(item, dict) and str(item.get("display_name") or "").strip()
    ]
    required = []
    missing = []
    for name in names:
        subject = _resolve_subject(subjects, name, None)
        if not subject:
            missing.append(name)
        else:
            required.append(subject)
    if not names:
        required = list(subjects)
    pending = missing + [
        item.get("display_name") or "未命名主体"
        for item in required
        if not item.get("approved_asset_id")
    ]
    return bool(required) and not pending, pending


def _advance_to_scene_material(client_id, session_id: str, project: dict, subjects: list[dict]) -> tuple[bool, list[str]]:
    """Move only when the canonical concept roster has approved assets."""
    all_approved, pending = _concept_subject_approval_state(project, subjects)
    if all_approved:
        update_video_project(
            client_id,
            session_id,
            stage="scene_material",
            status="scene_ready",
            project_brief=dict(project.get("project_brief") or {}),
        )
    return all_approved, pending


def _target_subjects(
    decision: VideoSubjectDecision,
    subjects: list[dict],
    active_subject_id: str | None,
    saved_subjects: list[dict] | None = None,
) -> list[dict]:
    if decision.intent == "generate_design" and decision.target_scope == "all_pending":
        return [item for item in subjects if not item.get("current_asset_id")]
    if decision.intent == "approve_design" and decision.target_scope == "all_pending":
        return [
            item for item in subjects
            if item.get("current_asset_id") and not item.get("approved_asset_id")
        ]

    targets = []
    seen = set()
    for reference in decision.target_subject_refs:
        subject = _resolve_subject(subjects, reference, active_subject_id)
        if subject and subject["subject_id"] not in seen:
            seen.add(subject["subject_id"])
            targets.append(subject)
    if not targets and decision.intent == "generate_design":
        pending_saved = [item for item in (saved_subjects or []) if not item.get("current_asset_id")]
        if pending_saved:
            return pending_saved
    if not targets:
        subject = _resolve_subject(subjects, "", active_subject_id)
        if subject:
            targets.append(subject)
    return targets


def _image_failure_message() -> str:
    return "这次主体设计没有成功生成，已有设定和素材都没有被覆盖。可以稍后重试，或者先调整一下视觉要求。"


def run_video_subject_chat(
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
    subjects = list_video_subjects(client_id, session_id)
    trace = _Trace(trace_callback)
    toolbox = VideoSubjectToolbox(client_id=client_id, session_id=session_id)

    seeded_subjects, subjects = _seed_concept_subjects(toolbox, project, subjects)
    if seeded_subjects:
        trace.call("save_subject_spec", {
            "subjects": [
                {
                    "display_name": item.get("display_name"),
                    "subject_type": item.get("subject_type"),
                }
                for item in seeded_subjects
            ],
            "source": "approved_concept",
        })
        trace.result("save_subject_spec", {
            "status": "1",
            "subjects": [
                {
                    "ordinal": item.get("ordinal"),
                    "display_name": item.get("display_name"),
                    "status": item.get("status"),
                }
                for item in seeded_subjects
            ],
        })

    reference_visual_facts = ""
    if image_path:
        vision = run_vision_context(
            user_message=(
                "提取这张参考图中主体可用于后续视频一致性设计的可见身份特征，"
                "包括主体类型、轮廓、比例、面部或表面特征、服装或材质、配色和标志性元素。"
            ),
            image_ref=image_path,
            context=user_message,
        )
        reference_visual_facts = str(vision.get("content") or "").strip()
        trace.add_usage(vision.get("usage") or {})

    decision, decision_usage = _director_decision(
        model=model,
        current_request=str(user_message or "").strip(),
        messages=messages,
        project=project,
        subjects=subjects,
        has_uploaded_reference=bool(image_path),
        reference_visual_facts=reference_visual_facts,
    )
    trace.add_usage(decision_usage)
    saved_subjects = []
    if decision.subject_updates:
        trace.call("save_subject_spec", {
            "subjects": [
                {
                    "subject_ref": item.subject_ref,
                    "display_name": item.display_name,
                    "subject_type": item.subject_type,
                    "story_role": item.story_role,
                }
                for item in decision.subject_updates
            ]
        })
        saved_subjects, subjects = _save_subject_updates(toolbox, decision, subjects)
        trace.result("save_subject_spec", {
            "status": "1",
            "subjects": [
                {
                    "display_name": item.get("display_name"),
                    "status": item.get("status"),
                }
                for item in saved_subjects
            ],
        })

    brief = dict(project.get("project_brief") or {})
    project_status = "collecting" if decision.readiness.status == "blocked" else project.get("status") or "collecting"
    project = update_video_project(
        client_id,
        session_id,
        status=project_status,
        project_brief=brief,
    )

    if decision.intent == "replace_subject":
        replacements = []
        for replacement in decision.subject_replacements:
            outgoing = _resolve_subject(subjects, replacement.from_subject_ref, None)
            incoming = _resolve_subject(subjects, replacement.to_subject_ref, None)
            if not outgoing or not incoming or outgoing.get("subject_id") == incoming.get("subject_id"):
                continue
            result = replace_video_subject_in_roster(
                client_id,
                session_id,
                outgoing_subject_id=outgoing["subject_id"],
                incoming_subject_id=incoming["subject_id"],
            )
            if result:
                replacements.append((result["outgoing"].get("display_name"), result["incoming"].get("display_name")))
        project = get_or_create_video_project(client_id, session_id)
        subjects = list_video_subjects(client_id, session_id)
        all_approved, pending = _advance_to_scene_material(client_id, session_id, project, subjects)
        if replacements and all_approved:
            reply = f"已用{'、'.join(item[1] for item in replacements)}替换{'、'.join(item[0] for item in replacements)}，正式主体素材已全部定稿。现在进入场景基准图构建阶段。"
        elif replacements:
            reply = f"已完成主体替换：{'；'.join(f'{old} → {new}' for old, new in replacements)}。还需要确认：{'、'.join(pending)}。"
        else:
            reply = "没有找到可替换的主体，请明确告诉我原主体和替代主体。"
        trace.answer(reply)
        return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}

    if decision.intent == "remove_subject":
        removed = []
        for reference in decision.target_subject_refs:
            subject = _resolve_subject(subjects, reference, None)
            if not subject:
                continue
            result = remove_video_subject_from_roster(client_id, session_id, subject_id=subject["subject_id"])
            if result:
                removed.append(result["removed"].get("display_name"))
        project = get_or_create_video_project(client_id, session_id)
        subjects = list_video_subjects(client_id, session_id)
        all_approved, pending = _advance_to_scene_material(client_id, session_id, project, subjects)
        if removed and all_approved:
            reply = f"已移除{'、'.join(removed)}，其余正式主体均已定稿。现在进入场景基准图构建阶段。"
        elif removed:
            reply = f"已移除{'、'.join(removed)}。还需要确认：{'、'.join(pending)}。"
        else:
            reply = "没有找到需要移除的主体。"
        trace.answer(reply)
        return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}

    if decision.intent == "advance_pipeline":
        refreshed_subjects = list_video_subjects(client_id, session_id)
        all_approved, pending = _advance_to_scene_material(client_id, session_id, project, refreshed_subjects)
        if all_approved:
            # A direct request to advance is execution consent. Hand off to the
            # next runtime in the same turn instead of making the user ask twice.
            from video_scene_runtime import run_video_scene_chat

            return run_video_scene_chat(
                client_id,
                session_id=session_id,
                model=model,
                messages=messages,
                user_message="开始生成全部正式场景基准图",
                trace_callback=trace_callback,
            )
        else:
            reply = f"暂时还不能进入下一阶段，还需要确认：{'、'.join(pending)}。"
        trace.answer(reply)
        return {"content": reply, "usage": trace.usage, "trace": trace.items, "used_tools": list(dict.fromkeys(trace.used_tools))}

    if decision.readiness.status == "blocked" and decision.intent == "generate_design":
        decision.intent = "clarify"

    if decision.intent in {"clarify", "reply", "offer_preview"}:
        reply = decision.assistant_reply.strip()
        if not reply and decision.intent == "clarify":
            missing = "、".join(decision.readiness.missing_critical_fields[:2])
            reply = f"在开始设计主体前，我还需要确认：{missing or '你希望重点呈现的主体是什么'}。你也可以直接让我自由发挥。"
        elif not reply:
            reply = "可以继续告诉我你希望怎样调整或推进这个主体素材。"
        trace.answer(reply)
        return {
            "content": reply,
            "usage": trace.usage,
            "trace": trace.items,
            "used_tools": list(dict.fromkeys(trace.used_tools)),
        }

    targets = _target_subjects(
        decision,
        subjects,
        project.get("active_subject_id"),
        saved_subjects,
    )
    if not targets:
        reply = decision.assistant_reply.strip() or "我还没有找到需要处理的主体。请先描述主体是什么，或者告诉我新建一个主体。"
        trace.answer(reply)
        return {
            "content": reply,
            "usage": trace.usage,
            "trace": trace.items,
            "used_tools": list(dict.fromkeys(trace.used_tools)),
        }

    if decision.intent == "generate_design":
        outputs = []
        for index, subject in enumerate(targets):
            if not toolbox.consume_image_quota():
                outputs.append("本次图片使用已达到当前时段上限，主体设定已经保存，可以在额度恢复后继续生成设计图。")
                break
            trace.call("generate_subject_design", {
                "display_name": subject["display_name"],
            })
            result = toolbox.generate_subject_design(
                subject,
                reference_image=image_path if len(targets) == 1 else None,
            )
            trace.result("generate_subject_design", public_tool_result(result))
            if result.get("status") != "1" or not result.get("image_url"):
                outputs.append(
                    f"**主体 {subject.get('ordinal')} · {subject.get('display_name')}**\n\n"
                    + _image_failure_message()
                )
                continue

            trace.call("inspect_subject_design", {
                "display_name": subject["display_name"],
                "asset": "generated_design",
            })
            inspection_result = toolbox.inspect_subject_design(subject, result)
            trace.add_usage(inspection_result.get("usage") or {})
            report = inspection_result.get("report") or {}
            trace.result("inspect_subject_design", {
                "status": "1" if report else "0",
                "passed": report.get("passed") if report else None,
                "score": report.get("score") if report else None,
            })
            persisted = toolbox.persist_design(
                subject,
                result,
                inspection=report,
                source_asset_id=None,
            )
            final_result = result
            final_report = report
            final_persisted = persisted

            blocking = report.get("blocking_issues") or []
            correction = str(report.get("correction_prompt") or "").strip()
            if blocking and correction and persisted and toolbox.consume_image_quota():
                refreshed = toolbox.reload_subject(subject["subject_id"]) or subject
                trace.call("revise_subject_design", {
                    "display_name": subject["display_name"],
                    "reason": "automatic_quality_correction",
                })
                correction_result = toolbox.revise_subject_design(refreshed, correction)
                trace.result("revise_subject_design", public_tool_result(correction_result))
                if correction_result.get("status") == "1" and correction_result.get("image_url"):
                    trace.call("inspect_subject_design", {
                        "display_name": subject["display_name"],
                        "asset": "corrected_design",
                    })
                    corrected_inspection = toolbox.inspect_subject_design(refreshed, correction_result)
                    trace.add_usage(corrected_inspection.get("usage") or {})
                    corrected_report = corrected_inspection.get("report") or {}
                    trace.result("inspect_subject_design", {
                        "status": "1" if corrected_report else "0",
                        "passed": corrected_report.get("passed") if corrected_report else None,
                        "score": corrected_report.get("score") if corrected_report else None,
                    })
                    corrected_persisted = toolbox.persist_design(
                        refreshed,
                        correction_result,
                        inspection=corrected_report,
                        source_asset_id=correction_result.get("source_asset_id"),
                    )
                    if corrected_persisted:
                        final_result = correction_result
                        final_report = corrected_report
                        final_persisted = corrected_persisted

            if not final_persisted:
                outputs.append(
                    f"**主体 {subject.get('ordinal')} · {subject.get('display_name')}**\n\n"
                    + _image_failure_message()
                )
                continue
            link = final_persisted.get("link") or {}
            quality_note = (
                "自动检查未发现影响后续使用的明显问题。"
                if final_report.get("passed")
                else "设计图已经保留，请重点确认主体外观和你预期是否一致。"
            )
            outputs.append(
                f"**主体 {subject.get('ordinal')} · {subject.get('display_name')}**\n\n"
                f"这是第 {link.get('version') or 1} 版主体建模设计图：\n"
                f"{final_result.get('image_url')}\n\n{quality_note}"
            )
            update_video_project(
                client_id,
                session_id,
                status="awaiting_review",
                active_subject_id=subject["subject_id"],
                project_brief=brief,
            )
        suffix = "\n\n你可以直接告诉我需要修改的地方；满意的话，我会把对应版本定为后续视频制作使用的正式主体素材。"
        reply = "\n\n".join(outputs).strip() + suffix

    elif decision.intent == "revise_design":
        subject = targets[0]
        change_request = decision.change_request.strip()
        if not change_request:
            reply = decision.assistant_reply.strip() or "请告诉我这次具体要修改主体的哪些视觉特征。"
        elif not toolbox.consume_image_quota():
            reply = "本次图片使用已达到当前时段上限，修改要求还没有执行，可以在额度恢复后继续。"
        else:
            trace.call("revise_subject_design", {
                "display_name": subject["display_name"],
                "change_request": change_request,
            })
            result = toolbox.revise_subject_design(subject, change_request)
            trace.result("revise_subject_design", public_tool_result(result))
            if result.get("status") != "1" or not result.get("image_url"):
                reply = _image_failure_message()
            else:
                trace.call("inspect_subject_design", {
                    "display_name": subject["display_name"],
                    "asset": "revised_design",
                })
                inspection_result = toolbox.inspect_subject_design(subject, result)
                trace.add_usage(inspection_result.get("usage") or {})
                report = inspection_result.get("report") or {}
                trace.result("inspect_subject_design", {
                    "status": "1" if report else "0",
                    "passed": report.get("passed") if report else None,
                    "score": report.get("score") if report else None,
                })
                persisted = toolbox.persist_design(
                    subject,
                    result,
                    inspection=report,
                    source_asset_id=result.get("source_asset_id"),
                )
                if not persisted:
                    reply = _image_failure_message()
                else:
                    version = (persisted.get("link") or {}).get("version") or 1
                    reply = (
                        f"**主体 {subject.get('ordinal')} · {subject.get('display_name')}** 已更新为第 {version} 版：\n\n"
                        f"{result.get('image_url')}\n\n"
                        "请确认这一版是否可以定稿，也可以继续告诉我需要调整的地方。"
                    )

    elif decision.intent == "approve_design":
        approved = []
        for subject in targets:
            trace.call("approve_subject_design", {
                "display_name": subject["display_name"],
            })
            item = toolbox.approve_subject_design(subject)
            trace.result("approve_subject_design", {
                "status": "1" if item else "0",
                "display_name": item.get("display_name") if item else subject.get("display_name"),
            })
            if item:
                approved.append(item.get("display_name"))
        if approved:
            refreshed_subjects = list_video_subjects(client_id, session_id)
            all_approved, pending = _advance_to_scene_material(client_id, session_id, project, refreshed_subjects)
            if all_approved:
                reply = (
                    f"已将{'、'.join(approved)}的当前设计定为正式主体素材。"
                    "视频所需主体现在已经全部定稿，下一步先建立可复用的场景基准图，再用角色和场景素材共同构建整组分镜。"
                )
            else:
                reply = (
                    f"已将{'、'.join(approved)}的当前设计定为正式主体素材，后续画面会固定使用这个版本。"
                    f"还需要确认：{'、'.join(pending)}。"
                )
        else:
            reply = "当前还没有可以定稿的主体设计图，请先完成设计或修改。"

    elif decision.intent == "generate_preview":
        subject = targets[0]
        if not subject.get("approved_asset_id"):
            reply = "这个主体还没有定稿。先确认一版正式主体设计图，再生成场景效果参考会更稳。"
        elif not toolbox.consume_image_quota():
            reply = "本次图片使用已达到当前时段上限，正式主体素材不受影响，可以在额度恢复后再生成场景效果图。"
        else:
            trace.call("generate_subject_preview", {
                "display_name": subject["display_name"],
                "scene": decision.preview_scene or "由导演根据视频主题选择简洁场景",
            })
            result = toolbox.generate_subject_preview(subject, decision.preview_scene)
            trace.result("generate_subject_preview", public_tool_result(result))
            persisted = toolbox.persist_preview(subject, result)
            if persisted and result.get("image_url"):
                reply = (
                    f"这是 **{subject.get('display_name')}** 进入场景后的效果参考：\n\n"
                    f"{result.get('image_url')}\n\n"
                    "这张图只用于确认视觉效果，不会替代已经定稿的正式主体素材。"
                )
            else:
                reply = "场景效果图这次没有成功生成，已经定稿的主体素材不会受到影响。"
    else:
        reply = decision.assistant_reply.strip() or "可以继续告诉我你希望怎样推进这个主体素材。"

    trace.answer(reply)
    return {
        "content": reply,
        "usage": trace.usage,
        "trace": trace.items,
        "used_tools": list(dict.fromkeys(trace.used_tools)),
    }
