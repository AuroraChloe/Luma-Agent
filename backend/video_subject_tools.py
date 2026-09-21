"""Domain tools for the video creation subject-material stage."""

from __future__ import annotations

from typing import Any

from agent_tools.image_edit import edit_image_result
from agent_tools.image_generate import generate_image_result
from sql import (
    approve_video_subject_asset,
    attach_video_subject_asset,
    ensure_chat_asset,
    get_video_subject,
    list_video_subject_assets,
    record_client_image_usage,
    upsert_video_subject,
)
from vision_context_runtime import run_vision_inspection


VIDEO_SUBJECT_TOOL_NAMES = (
    "save_subject_spec",
    "generate_subject_design",
    "inspect_subject_design",
    "revise_subject_design",
    "approve_subject_design",
    "generate_subject_preview",
)


def _clean_spec(value: Any) -> Any:
    """Remove empty model output without imposing a subject-specific schema."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            normalized = _clean_spec(item)
            if normalized not in (None, "", [], {}):
                cleaned[str(key)] = normalized
        return cleaned
    if isinstance(value, list):
        return [item for item in (_clean_spec(item) for item in value) if item not in (None, "", [], {})]
    if isinstance(value, str):
        return value.strip()
    return value


def build_subject_design_prompt(subject: dict) -> str:
    spec = _clean_spec(subject.get("spec") or {})
    identity = str(spec.get("identity") or subject.get("display_name") or "未命名主体").strip()
    visual_direction = str(spec.get("visual_direction") or "").strip()
    declared_details = []
    for key, value in spec.items():
        if key in {"identity", "visual_direction"}:
            continue
        if isinstance(value, (str, int, float, bool)):
            declared_details.append(f"{key}：{value}")
        if len(declared_details) >= 5:
            break
    lines = [
        "制作一张供后续视频保持形象一致的简洁主体参考图。",
        f"主体：{identity}。",
        f"主体类型：{subject.get('subject_type') or 'other'}。",
    ]
    if subject.get("story_role"):
        lines.append(f"定位：{subject['story_role']}。")
    if visual_direction:
        lines.append(f"已确认的视觉要求：{visual_direction}。")
    if declared_details:
        lines.append("用户明确细节：" + "；".join(declared_details) + "。")
    lines.extend([
        "使用干净中性背景和统一柔和光线，只展示同一个主体。",
        "以完整主视图为核心，补充侧视图、背视图和 1 至 2 个能锁定身份的局部细节；不要加入故事场景、无关道具、文字或复杂设定板排版。",
        "若主体是用户明确指定的已有 IP、人物或品牌形象，严格保持其可识别身份、经典轮廓、服装与配色；不要重设世界观、年龄、性格、职业、服装体系或额外标志。",
        "只有当用户没有给出明确外观时，才为不影响主体身份的次要细节做克制补全。",
        "所有视图保持同一张脸或外观结构、比例、服装或材质和配色，作为后续视频的一致性参考。",
    ])
    return "".join(lines)


def build_subject_revision_prompt(subject: dict, change_request: str) -> str:
    return (
        f"这是视频主体“{subject.get('display_name') or '当前主体'}”的建模参考图。"
        f"本次只执行以下明确修改：{str(change_request or '').strip()}。"
        "保留用户未要求改变的主体身份、经典外观、轮廓、比例、视图布局、服装或材质、配色和标志性特征。"
        "修改后仍保持干净中性背景、统一光线和工程可用的主体设定板形式。"
    )


def build_subject_preview_prompt(subject: dict, scene_prompt: str) -> str:
    scene = str(scene_prompt or "").strip() or "选择一个简洁、具有叙事感且符合主体故事定位的场景"
    return (
        f"基于这张已定稿的主体设计图，生成一张用于视频创意确认的场景效果参考图。场景要求：{scene}。"
        "必须严格保持主体的身份、外观、轮廓、比例、面部或表面特征、服装、材质、配色和标志性元素一致。"
        "画面可以具有电影感和明确环境，但不要重新设计主体，不要加入重复主体。"
    )


class VideoSubjectToolbox:
    def __init__(self, *, client_id: int, session_id: str):
        self.client_id = client_id
        self.session_id = session_id

    def save_subject_spec(self, update: dict, *, status: str) -> dict:
        return upsert_video_subject(
            self.client_id,
            self.session_id,
            subject_id=update.get("subject_id") or None,
            display_name=update.get("display_name") or None,
            subject_type=update.get("subject_type") or "other",
            story_role=update.get("story_role") or None,
            spec=update.get("spec") or {},
            status=status,
        )

    def consume_image_quota(self) -> bool:
        quota = record_client_image_usage(self.client_id, 1)
        return bool(quota.get("allowed"))

    def generate_subject_design(self, subject: dict, *, reference_image: str | None = None) -> dict:
        prompt = build_subject_design_prompt(subject)
        if reference_image:
            result = edit_image_result(reference_image, prompt)
        else:
            result = generate_image_result(prompt)
        return {**result, "compiled_prompt": prompt}

    def revise_subject_design(self, subject: dict, change_request: str) -> dict:
        source = self.current_design_asset(subject)
        if not source:
            return {"status": "0", "info": "SUBJECT_DESIGN_NOT_FOUND"}
        prompt = build_subject_revision_prompt(subject, change_request)
        source_ref = source.get("public_url") or source.get("local_path")
        result = edit_image_result(source_ref, prompt)
        return {
            **result,
            "compiled_prompt": prompt,
            "source_asset_id": source.get("asset_id"),
        }

    def inspect_subject_design(self, subject: dict, result: dict) -> dict:
        image_ref = result.get("image_url") or result.get("image_path")
        return run_vision_inspection(
            image_ref=image_ref,
            subject_spec={
                "display_name": subject.get("display_name"),
                "subject_type": subject.get("subject_type"),
                "story_role": subject.get("story_role"),
                **(subject.get("spec") or {}),
            },
            purpose="design_sheet",
        )

    def persist_design(
        self,
        subject: dict,
        result: dict,
        *,
        inspection: dict,
        source_asset_id: str | None = None,
    ) -> dict | None:
        if result.get("status") != "1" or not result.get("image_url"):
            return None
        asset = ensure_chat_asset(
            self.client_id,
            self.session_id,
            asset_type="video_subject_design",
            local_path=result.get("image_path"),
            public_url=result.get("image_url"),
            source_asset_id=source_asset_id,
            asset_label=f"{subject.get('display_name') or '主体'}的建模设计图",
            operation_prompt=result.get("compiled_prompt") or result.get("prompt") or result.get("edit_prompt"),
            make_active=True,
        )
        if not asset:
            return None
        link = attach_video_subject_asset(
            self.client_id,
            self.session_id,
            subject["subject_id"],
            asset["asset_id"],
            asset_role="design_sheet",
            inspection=inspection,
            pipeline_eligible=False,
        )
        return {"asset": asset, "link": link}

    def current_design_asset(self, subject: dict) -> dict | None:
        assets = list_video_subject_assets(
            self.client_id,
            self.session_id,
            subject.get("subject_id"),
        )
        current_asset_id = subject.get("current_asset_id")
        if current_asset_id:
            matched = next((item for item in assets if item.get("asset_id") == current_asset_id), None)
            if matched:
                return matched
        designs = [item for item in assets if item.get("asset_role") == "design_sheet"]
        return designs[-1] if designs else None

    def approved_design_asset(self, subject: dict) -> dict | None:
        approved_asset_id = subject.get("approved_asset_id")
        if not approved_asset_id:
            return None
        return next(
            (
                item
                for item in list_video_subject_assets(
                    self.client_id,
                    self.session_id,
                    subject.get("subject_id"),
                )
                if item.get("asset_id") == approved_asset_id
            ),
            None,
        )

    def approve_subject_design(self, subject: dict) -> dict | None:
        return approve_video_subject_asset(
            self.client_id,
            self.session_id,
            subject.get("subject_id"),
            subject.get("current_asset_id"),
        )

    def generate_subject_preview(self, subject: dict, scene_prompt: str) -> dict:
        source = self.approved_design_asset(subject)
        if not source:
            return {"status": "0", "info": "APPROVED_SUBJECT_DESIGN_NOT_FOUND"}
        prompt = build_subject_preview_prompt(subject, scene_prompt)
        source_ref = source.get("public_url") or source.get("local_path")
        result = edit_image_result(source_ref, prompt)
        return {
            **result,
            "compiled_prompt": prompt,
            "source_asset_id": source.get("asset_id"),
        }

    def persist_preview(self, subject: dict, result: dict) -> dict | None:
        if result.get("status") != "1" or not result.get("image_url"):
            return None
        asset = ensure_chat_asset(
            self.client_id,
            self.session_id,
            asset_type="video_subject_preview",
            local_path=result.get("image_path"),
            public_url=result.get("image_url"),
            source_asset_id=result.get("source_asset_id"),
            asset_label=f"{subject.get('display_name') or '主体'}的场景效果参考图",
            operation_prompt=result.get("compiled_prompt") or result.get("edit_prompt"),
            make_active=False,
        )
        if not asset:
            return None
        link = attach_video_subject_asset(
            self.client_id,
            self.session_id,
            subject["subject_id"],
            asset["asset_id"],
            asset_role="scene_preview",
            inspection={},
            pipeline_eligible=False,
        )
        return {"asset": asset, "link": link}

    def reload_subject(self, subject_id: str) -> dict | None:
        return get_video_subject(self.client_id, self.session_id, subject_id)


def public_tool_result(result: Any) -> dict:
    """Keep traces useful without leaking provider payloads or server paths."""
    if not isinstance(result, dict):
        return {"status": "0"}
    visible = {
        key: result.get(key)
        for key in ("status", "info", "image_url")
        if result.get(key) is not None
    }
    return visible
