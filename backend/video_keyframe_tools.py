"""Production tools for script-bound video keyframe material."""

from __future__ import annotations

import json
import os
from typing import Any

from agent_tools.image_edit import edit_image_result
from agent_tools.image_generate import generate_image_result
from sql import (
    attach_video_keyframe_asset,
    ensure_chat_asset,
    list_video_keyframe_assets,
    list_video_subject_assets,
    record_client_image_usage,
)


VIDEO_KEYFRAME_TOOL_NAMES = (
    "plan_video_keyframes",
    "generate_video_keyframe",
)

VIDEO_KEYFRAME_SIZE = os.getenv("VIDEO_KEYFRAME_SIZE", "1536x1024")
STORYBOARD_MIN_PANELS = 9
STORYBOARD_MAX_PANELS = 15


def expand_storyboard_panels(frames: list[dict], *, target_panels: int | None = None) -> list[dict]:
    """Expand broad script beats into connected camera moments without adding events."""
    source = [item for item in frames if isinstance(item, dict)]
    if not source:
        return []
    inferred_target = max(STORYBOARD_MIN_PANELS, len(source) * 2)
    target = min(STORYBOARD_MAX_PANELS, max(int(target_panels or inferred_target), len(source)))
    counts = [1] * len(source)
    cursor = 0
    while sum(counts) < target:
        counts[cursor % len(counts)] += 1
        cursor += 1
    phase_sets = {
        1: ["核心瞬间"],
        2: ["动作起势与空间关系", "动作结果与人物反应"],
        3: ["动作起势与环境建立", "动作推进与镜头加压", "动作结果与下一镜衔接"],
    }
    panels = []
    panel_no = 1
    for frame, count in zip(source, counts):
        phases = phase_sets.get(count) or phase_sets[3] + ["连续反应镜头"] * (count - 3)
        for within_beat, phase in enumerate(phases, start=1):
            panels.append({
                "panel": panel_no,
                "source_beat_index": frame.get("source_beat_index"),
                "time_range": frame.get("time_range"),
                "story": frame.get("script_content"),
                "visual": frame.get("script_visual"),
                "direction": frame.get("plan") or {},
                "within_beat": within_beat,
                "within_beat_total": count,
                "camera_moment": phase,
            })
            panel_no += 1
    return panels


def build_keyframe_prompt(frame: dict, subjects: list[dict], scenes: list[dict] | None = None) -> str:
    plan = frame.get("plan") or {}
    reference_rows = []
    for index, subject in enumerate(subjects, start=1):
        reference_rows.append({
            "reference_image": index,
            "subject_id": subject.get("subject_id"),
            "display_name": subject.get("display_name"),
            "subject_type": subject.get("subject_type"),
            "story_role": subject.get("story_role"),
            "approved_spec": subject.get("spec") or {},
        })
    for index, scene in enumerate(scenes or [], start=len(reference_rows) + 1):
        reference_rows.append({
            "reference_image": index,
            "reference_kind": "scene",
            "scene_id": scene.get("scene_id"),
            "display_name": scene.get("display_name"),
            "setting": scene.get("setting"),
            "visual_direction": scene.get("visual_direction"),
            "continuity_notes": scene.get("continuity_notes"),
        })

    return (
        "为后续图生视频流水线生成一张单幅、可直接使用的剧情关键帧。"
        "这不是主体设定板、分镜拼贴、海报或带文字的说明图。"
        "必须忠实呈现下列已确认脚本节拍，不得增加、删除、替换剧情事件，不得改变人物关系或事件结果。"
        f"脚本时间段：{frame.get('time_range') or '未标注'}。"
        f"节拍目的：{frame.get('purpose') or '推进当前剧情'}。"
        f"脚本内容：{frame.get('script_content') or ''}。"
        f"原始画面要求：{frame.get('script_visual') or '依据脚本内容自然呈现'}。"
        "导演只允许在不改变脚本事实的前提下控制单帧镜头语言："
        + json.dumps(plan, ensure_ascii=False, default=str, separators=(",", ":"))
        + "。"
        "随请求附带的参考图片映射如下："
        + json.dumps(reference_rows, ensure_ascii=False, default=str, separators=(",", ":"))
        + "。"
        "角色参考图只用于锁定身份、面部或表面特征、轮廓比例、服装或材质、配色和标志性元素；"
        "场景参考图只用于锁定空间布局、道具、光源、色彩、材质和氛围。"
        "不要照搬角色设定板的白底、排版、多视图或文字，也不要把场景基准图中的空景误画成无人剧情。"
        "必须以当前节拍指定的角色、动作与空间关系重新构图，输出一张构图完整、动作可延续、具有连续视频起始状态的正式剧情分镜图。"
        f"统一输出规格为 {VIDEO_KEYFRAME_SIZE} 横向画幅；所有关键帧必须保持相同尺寸和画幅比例。"
        "不要擅自添加脚本未要求的边框、字幕、时间码、镜头编号、水印、说明文字或多个分格。"
        "如果原始脚本明确要求后期文字，只为该文字保留合适的构图安全区，不要改写文字内容，也不要让文字遮挡主体。"
    )


def build_storyboard_sheet_prompt(frames: list[dict], subjects: list[dict], scenes: list[dict] | None = None) -> str:
    """Describe one connected storyboard sheet rather than unrelated stills."""
    panel_rows = expand_storyboard_panels(frames)
    reference_rows = []
    for index, subject in enumerate(subjects, start=1):
        reference_rows.append({
            "reference_image": index,
            "kind": "character_or_subject",
            "name": subject.get("display_name"),
            "identity": subject.get("spec") or {},
        })
    for index, scene in enumerate(scenes or [], start=len(reference_rows) + 1):
        reference_rows.append({
            "reference_image": index,
            "kind": "scene",
            "name": scene.get("display_name"),
            "setting": scene.get("setting"),
            "continuity": scene.get("continuity_notes"),
        })
    return (
        "为一条不超过15秒的完整短视频制作一张正式、横向电影分镜图合集。"
        "必须只输出一张完整的多格 storyboard sheet，不得输出单张海报、单格电影剧照、人物设定板或多个独立图片。"
        f"画面按从左到右、从上到下的阅读顺序包含 {len(panel_rows)} 个清晰分镜格；根据格数自动采用 3x3、3x4、3x5 或同等紧凑的电影分镜布局。每格左上角仅保留小型阿拉伯序号，用于表示连续时序，禁止任何标题、字幕、说明文字、时间码和水印。"
        "每格都必须是独立可读的电影镜头，也是同一段连续动作中的一个可运动瞬间：主动变化景别、机位和构图，在建立镜头、视线反应、重心转移、动作起势、动作推进、碰撞或冲击、余波、结果和转场之间形成明确节奏。"
        "格与格之间使用细窄深色分隔线，布局紧凑但每格的画面、人物手脚和关键动作必须清晰可读，严禁把整组分镜画成一张模糊的远景或重复的静态站桩图。"
        "所有格子必须是同一条故事线的连续发展：角色身份、脸部、服装、道具、伤痕、天气、空间结构和光线保持连贯；"
        "每一格的结束姿态、人物朝向、屏幕左右位置、运动方向和环境变化，必须成为下一格的起始条件。"
        "当一个原始节拍被拆成多格时，只能把该节拍已有动作细化为起势、过程、反应或结果，不得新增事件、人物、能力、道具、对白或结局。"
        "不要让每格像互不相关的宣传海报；要像专业电影分镜，将开场、推进、转折和结局连成可直接拍摄的因果链。"
        "参考图片映射："
        + json.dumps(reference_rows, ensure_ascii=False, default=str, separators=(",", ":"))
        + "。完整分镜计划："
        + json.dumps(panel_rows, ensure_ascii=False, default=str, separators=(",", ":"))
        + "。参考图片仅用于锁定角色身份和场景连续性，严禁把白底设定板或空场景原样复制到分镜格中。"
        + f"统一输出 {VIDEO_KEYFRAME_SIZE} 横向画幅，完整多格电影分镜合集、镜头景别变化和连续动作表达。"
    )


def build_storyboard_sheet_revision_prompt(
    frames: list[dict],
    subjects: list[dict],
    scenes: list[dict],
    change_request: str,
) -> str:
    """Edit a sheet while keeping its approved narrative and visual references intact."""
    references = []
    for index, subject in enumerate(subjects, start=2):
        references.append({"reference_image": index, "kind": "角色", "name": subject.get("display_name")})
    for index, scene in enumerate(scenes, start=len(references) + 2):
        references.append({"reference_image": index, "kind": "场景", "name": scene.get("display_name")})
    return (
        "参考图1是一张已生成的完整连续视频分镜图合集。"
        f"本次只执行用户提出的分镜修改：{str(change_request or '').strip()}。"
        "保留用户没有要求修改的剧情、时间顺序、角色身份、角色外观、场景空间、关键动作、因果关系和结局。"
        "参考图2起仅用于重新锁定角色与场景的一致性，不得把这些参考图本身复制成分镜格。"
        "输出仍必须是一张完整横向多格电影分镜图合集，按从左到右、从上到下连续阅读；"
        "保留清晰的连续动作、镜头变化和紧凑分格，不要输出单张剧照、海报、文字说明或新的故事内容。"
        f"已确认的分镜计划：{json.dumps(expand_storyboard_panels(frames), ensure_ascii=False, default=str, separators=(',', ':'))}。"
        f"参考图映射：{json.dumps(references, ensure_ascii=False, default=str, separators=(',', ':'))}。"
        f"统一输出 {VIDEO_KEYFRAME_SIZE} 横向画幅。"
    )


def build_keyframe_revision_prompt(
    frame: dict,
    subjects: list[dict],
    report: dict,
    *,
    script_timeline: list[dict] | None = None,
) -> str:
    subject_rows = [
        {
            "reference_image": index + 2,
            "subject_id": subject.get("subject_id"),
            "display_name": subject.get("display_name"),
            "approved_spec": subject.get("spec") or {},
        }
        for index, subject in enumerate(subjects)
    ]
    current_index = int(frame.get("source_beat_index") or frame.get("ordinal") or 1)
    timeline = [item for item in (script_timeline or []) if isinstance(item, dict)]
    neighboring = [
        item for index, item in enumerate(timeline, start=1)
        if abs(index - current_index) <= 1
    ]
    return (
        "修正参考图片1中的剧情关键帧。参考图片1是待修正画面，参考图片2起是已确认主体素材。"
        "只修正质检指出的问题，并严格回到当前脚本节拍；不要重新设计剧情，不要把前后节拍的事件混进当前画面。"
        f"当前脚本节拍：{json.dumps({'time_range': frame.get('time_range'), 'purpose': frame.get('purpose'), 'content': frame.get('script_content'), 'visual': frame.get('script_visual')}, ensure_ascii=False, separators=(',', ':'))}。"
        f"当前镜头计划：{json.dumps(frame.get('plan') or {}, ensure_ascii=False, separators=(',', ':'))}。"
        f"相邻节拍仅用于识别并排除时间越界：{json.dumps(neighboring, ensure_ascii=False, default=str, separators=(',', ':'))}。"
        f"主体参考映射：{json.dumps(subject_rows, ensure_ascii=False, default=str, separators=(',', ':'))}。"
        f"必须修正：{json.dumps(report.get('blocking_issues') or [], ensure_ascii=False)}。"
        f"执行要求：{str(report.get('correction_prompt') or '').strip()}。"
        "严格保持主体身份、面部或表面特征、轮廓比例、服装材质、配色和标志性元素。"
        f"输出一张 {VIDEO_KEYFRAME_SIZE} 横向单幅电影画面，不要文字、边框、分格、设定板或说明标记。"
    )


class VideoKeyframeToolbox:
    def __init__(self, *, client_id: int, session_id: str):
        self.client_id = client_id
        self.session_id = session_id

    def consume_image_quota(self) -> bool:
        quota = record_client_image_usage(self.client_id, 1)
        return bool(quota.get("allowed"))

    def approved_subject_asset(self, subject: dict) -> dict | None:
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
                if item.get("asset_id") == approved_asset_id and item.get("pipeline_eligible")
            ),
            None,
        )

    @staticmethod
    def approved_scene_asset(scene: dict) -> dict | None:
        if not scene.get("approved"):
            return None
        reference = scene.get("public_url") or scene.get("local_path")
        if not reference:
            return None
        return {"asset_id": scene.get("asset_id"), "reference": reference}

    def generate_keyframe(self, frame: dict, subjects: list[dict], scenes: list[dict] | None = None) -> dict:
        prompt = build_keyframe_prompt(frame, subjects, scenes)
        references = []
        source_asset_ids = []
        for subject in subjects:
            asset = self.approved_subject_asset(subject)
            if not asset:
                return {
                    "status": "0",
                    "info": "APPROVED_SUBJECT_ASSET_NOT_FOUND",
                    "subject_id": subject.get("subject_id"),
                }
            reference = asset.get("public_url") or asset.get("local_path")
            if not reference:
                return {
                    "status": "0",
                    "info": "APPROVED_SUBJECT_ASSET_NOT_FOUND",
                    "subject_id": subject.get("subject_id"),
                }
            references.append(reference)
            source_asset_ids.append(asset.get("asset_id"))
        for scene in scenes or []:
            asset = self.approved_scene_asset(scene)
            if not asset:
                return {
                    "status": "0",
                    "info": "APPROVED_SCENE_ASSET_NOT_FOUND",
                    "scene_id": scene.get("scene_id"),
                }
            references.append(asset["reference"])
            if asset.get("asset_id"):
                source_asset_ids.append(asset["asset_id"])

        result = (
            edit_image_result(references, prompt, size=VIDEO_KEYFRAME_SIZE)
            if references
            else generate_image_result(prompt, size=VIDEO_KEYFRAME_SIZE)
        )
        return {
            **result,
            "compiled_prompt": prompt,
            "source_asset_ids": source_asset_ids,
        }

    def generate_storyboard_sheet(self, frames: list[dict], subjects: list[dict], scenes: list[dict]) -> dict:
        prompt = build_storyboard_sheet_prompt(frames, subjects, scenes)
        references = []
        source_asset_ids = []
        for subject in subjects:
            asset = self.approved_subject_asset(subject)
            if not asset:
                return {"status": "0", "info": "APPROVED_SUBJECT_ASSET_NOT_FOUND", "subject_id": subject.get("subject_id")}
            reference = asset.get("public_url") or asset.get("local_path")
            if not reference:
                return {"status": "0", "info": "APPROVED_SUBJECT_ASSET_NOT_FOUND", "subject_id": subject.get("subject_id")}
            references.append(reference)
            source_asset_ids.append(asset.get("asset_id"))
        for scene in scenes:
            asset = self.approved_scene_asset(scene)
            if not asset:
                return {"status": "0", "info": "APPROVED_SCENE_ASSET_NOT_FOUND", "scene_id": scene.get("scene_id")}
            references.append(asset["reference"])
            if asset.get("asset_id"):
                source_asset_ids.append(asset["asset_id"])
        if len(references) > 10:
            return {"status": "0", "info": "TOO_MANY_STORYBOARD_REFERENCES"}
        result = edit_image_result(references, prompt, size=VIDEO_KEYFRAME_SIZE)
        return {**result, "compiled_prompt": prompt, "source_asset_ids": source_asset_ids}

    def revise_storyboard_sheet(
        self,
        current_sheet: dict,
        frames: list[dict],
        subjects: list[dict],
        scenes: list[dict],
        change_request: str,
    ) -> dict:
        source = current_sheet.get("public_url") or current_sheet.get("local_path")
        if not source:
            return {"status": "0", "info": "STORYBOARD_SHEET_NOT_FOUND"}
        prompt = build_storyboard_sheet_revision_prompt(frames, subjects, scenes, change_request)
        references = [source]
        source_asset_ids = [current_sheet.get("asset_id")]
        for subject in subjects:
            asset = self.approved_subject_asset(subject)
            if not asset:
                return {"status": "0", "info": "APPROVED_SUBJECT_ASSET_NOT_FOUND", "subject_id": subject.get("subject_id")}
            reference = asset.get("public_url") or asset.get("local_path")
            if reference:
                references.append(reference)
                source_asset_ids.append(asset.get("asset_id"))
        for scene in scenes:
            asset = self.approved_scene_asset(scene)
            if not asset:
                return {"status": "0", "info": "APPROVED_SCENE_ASSET_NOT_FOUND", "scene_id": scene.get("scene_id")}
            references.append(asset["reference"])
            source_asset_ids.append(asset.get("asset_id"))
        if len(references) > 10:
            return {"status": "0", "info": "TOO_MANY_STORYBOARD_REFERENCES"}
        result = edit_image_result(references, prompt, size=VIDEO_KEYFRAME_SIZE)
        return {
            **result,
            "compiled_prompt": prompt,
            "source_asset_ids": [item for item in source_asset_ids if item],
        }

    def persist_storyboard_sheet(self, result: dict) -> dict | None:
        if result.get("status") != "1" or not result.get("image_url"):
            return None
        source_asset_ids = result.get("source_asset_ids") or []
        return ensure_chat_asset(
            self.client_id,
            self.session_id,
            asset_type="video_storyboard_sheet",
            local_path=result.get("image_path"),
            public_url=result.get("image_url"),
            source_asset_id=source_asset_ids[0] if len(source_asset_ids) == 1 else None,
            asset_label="完整连续视频分镜图合集",
            operation_prompt=result.get("compiled_prompt") or result.get("prompt") or result.get("edit_prompt"),
            make_active=True,
        )

    def inspect_keyframe(
        self,
        frame: dict,
        subjects: list[dict],
        result: dict,
        *,
        script_timeline: list[dict] | None = None,
    ) -> dict:
        image_ref = result.get("image_url") or result.get("image_path")
        return run_vision_inspection(
            image_ref=image_ref,
            subject_spec={
                "source_beat": {
                    "time_range": frame.get("time_range"),
                    "purpose": frame.get("purpose"),
                    "content": frame.get("script_content"),
                    "visual": frame.get("script_visual"),
                },
                "frame_plan": frame.get("plan") or {},
                "script_timeline": [
                    {"source_beat_index": index, **item}
                    for index, item in enumerate(script_timeline or [], start=1)
                    if isinstance(item, dict)
                ],
                "output_format": {
                    "size": VIDEO_KEYFRAME_SIZE,
                    "orientation": "landscape",
                },
                "approved_subjects": [
                    {
                        "display_name": subject.get("display_name"),
                        "subject_type": subject.get("subject_type"),
                        "story_role": subject.get("story_role"),
                        "spec": subject.get("spec") or {},
                    }
                    for subject in subjects
                ],
            },
            purpose="video_keyframe",
        )

    def revise_keyframe(
        self,
        frame: dict,
        subjects: list[dict],
        current_result: dict,
        report: dict,
        *,
        script_timeline: list[dict] | None = None,
    ) -> dict:
        current_image = current_result.get("image_url") or current_result.get("image_path")
        if not current_image:
            return {"status": "0", "info": "CURRENT_KEYFRAME_NOT_FOUND"}

        references = [current_image]
        source_asset_ids = []
        for subject in subjects:
            asset = self.approved_subject_asset(subject)
            if not asset:
                return {
                    "status": "0",
                    "info": "APPROVED_SUBJECT_ASSET_NOT_FOUND",
                    "subject_id": subject.get("subject_id"),
                }
            reference = asset.get("public_url") or asset.get("local_path")
            if not reference:
                return {
                    "status": "0",
                    "info": "APPROVED_SUBJECT_ASSET_NOT_FOUND",
                    "subject_id": subject.get("subject_id"),
                }
            references.append(reference)
            source_asset_ids.append(asset.get("asset_id"))

        prompt = build_keyframe_revision_prompt(
            frame,
            subjects,
            report,
            script_timeline=script_timeline,
        )
        result = edit_image_result(
            references,
            prompt,
            size=VIDEO_KEYFRAME_SIZE,
        )
        return {
            **result,
            "compiled_prompt": prompt,
            "source_asset_ids": source_asset_ids,
        }

    def persist_keyframe(self, frame: dict, result: dict, *, inspection: dict) -> dict | None:
        if result.get("status") != "1" or not result.get("image_url"):
            return None
        source_asset_ids = result.get("source_asset_ids") or []
        asset = ensure_chat_asset(
            self.client_id,
            self.session_id,
            asset_type="video_keyframe",
            local_path=result.get("image_path"),
            public_url=result.get("image_url"),
            source_asset_id=source_asset_ids[0] if len(source_asset_ids) == 1 else None,
            asset_label=f"关键帧 {frame.get('ordinal') or frame.get('source_beat_index')}",
            operation_prompt=result.get("compiled_prompt") or result.get("prompt") or result.get("edit_prompt"),
            make_active=False,
        )
        if not asset:
            return None
        link = attach_video_keyframe_asset(
            self.client_id,
            self.session_id,
            frame["keyframe_id"],
            asset["asset_id"],
            inspection=inspection,
            source_subject_ids=frame.get("subject_ids") or [],
        )
        return {"asset": asset, "link": link}

    def approve_keyframe(self, frame: dict) -> dict | None:
        return approve_video_keyframe_asset(
            self.client_id,
            self.session_id,
            frame.get("keyframe_id"),
            frame.get("current_asset_id"),
        )

    def current_asset(self, frame: dict) -> dict | None:
        current_asset_id = frame.get("current_asset_id")
        if not current_asset_id:
            return None
        return next(
            (
                item
                for item in list_video_keyframe_assets(
                    self.client_id,
                    self.session_id,
                    frame.get("keyframe_id"),
                )
                if item.get("asset_id") == current_asset_id
            ),
            None,
        )


def public_keyframe_result(result: Any) -> dict:
    if not isinstance(result, dict):
        return {"status": "0"}
    return {
        key: result.get(key)
        for key in ("status", "info", "image_url")
        if result.get(key) is not None
    }
