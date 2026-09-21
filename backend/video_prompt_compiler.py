"""Compile structured shot directions into strict H3 reference-video prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


PICTURE_TAG_RE = re.compile(r"<Picture\s+(\d+)>")


@dataclass(frozen=True)
class PictureReference:
    url: str
    purpose: str
    instruction: str


@dataclass(frozen=True)
class TimelineCue:
    start_second: float
    end_second: float
    shot: str
    camera: str
    blocking: str
    action: str
    environment: str = ""
    sound: str = ""


@dataclass(frozen=True)
class VideoPromptSpec:
    title: str
    duration: int
    references: tuple[PictureReference, ...]
    visual_baseline: str
    opening_state: str
    ending_state: str
    timeline: tuple[TimelineCue, ...]
    continuity: tuple[str, ...] = field(default_factory=tuple)
    negative_constraints: tuple[str, ...] = field(default_factory=tuple)


def picture_tag(index: int) -> str:
    return f"<Picture {index}>"


def _format_second(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def validate_video_prompt_spec(spec: VideoPromptSpec) -> None:
    if not 1 <= int(spec.duration) <= 15:
        raise ValueError("H3 clip duration must be between 1 and 15 seconds")
    if not spec.references:
        raise ValueError("at least one picture reference is required")
    if len(spec.references) > 10:
        raise ValueError("H3 accepts at most 10 reference materials")
    for reference in spec.references:
        if not str(reference.url or "").startswith("https://"):
            raise ValueError(f"reference URL must use public HTTPS: {reference.url}")

    cues = sorted(spec.timeline, key=lambda item: (item.start_second, item.end_second))
    if not cues:
        raise ValueError("at least one timeline cue is required")
    cursor = 0.0
    for cue in cues:
        if cue.start_second < 0 or cue.end_second <= cue.start_second:
            raise ValueError(f"invalid timeline cue: {cue.start_second}-{cue.end_second}")
        if abs(cue.start_second - cursor) > 0.001:
            raise ValueError(
                f"timeline must be continuous; expected {cursor:g}s, got {cue.start_second:g}s"
            )
        cursor = cue.end_second
    if abs(cursor - float(spec.duration)) > 0.001:
        raise ValueError(
            f"timeline must end at {spec.duration}s, got {cursor:g}s"
        )


def compile_h3_video_prompt(spec: VideoPromptSpec) -> str:
    """Return a deterministic prompt whose media tags match upload order."""

    validate_video_prompt_spec(spec)
    lines = [
        f"【片段】{spec.title}；总时长严格为 {spec.duration} 秒。",
        "【参考素材绑定】以下编号严格对应本次 multipart 请求中 image_urls 的提交顺序：",
    ]
    for index, reference in enumerate(spec.references, start=1):
        lines.append(
            f"- {picture_tag(index)}：{reference.purpose}。{reference.instruction}"
        )

    lines.extend([
        "【视觉与摄影基准】",
        spec.visual_baseline,
        "【首尾状态】",
        f"- 0 秒开场：{spec.opening_state}",
        f"- {spec.duration} 秒结束：{spec.ending_state}",
        "【逐秒镜头与动作时间轴】",
    ])
    for cue in sorted(spec.timeline, key=lambda item: item.start_second):
        interval = f"{_format_second(cue.start_second)}-{_format_second(cue.end_second)} 秒"
        lines.append(f"- {interval}｜景别：{cue.shot}。")
        lines.append(f"  运镜：{cue.camera}。")
        lines.append(f"  人物走位：{cue.blocking}。")
        lines.append(f"  动作表演：{cue.action}。")
        if cue.environment:
            lines.append(f"  环境与特效：{cue.environment}。")
        if cue.sound:
            lines.append(f"  声音节奏：{cue.sound}。")

    if spec.continuity:
        lines.append("【跨镜头连续性】")
        lines.extend(f"- {item}" for item in spec.continuity)
    if spec.negative_constraints:
        lines.append("【禁止项】")
        lines.extend(f"- {item}" for item in spec.negative_constraints)

    prompt = "\n".join(lines).strip()
    found = {int(value) for value in PICTURE_TAG_RE.findall(prompt)}
    expected = set(range(1, len(spec.references) + 1))
    if found != expected:
        raise ValueError(
            f"prompt picture tags {sorted(found)} do not match uploaded references {sorted(expected)}"
        )
    return prompt


def reference_urls(spec: VideoPromptSpec) -> list[str]:
    validate_video_prompt_spec(spec)
    return [item.url for item in spec.references]
