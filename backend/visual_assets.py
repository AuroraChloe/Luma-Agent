"""Pure helpers for resolving visual assets inside a chat session."""

import os
import re
from urllib.parse import urlparse


def _compact_text(value, limit=240):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def visual_asset_catalog(assets, limit=8):
    """Return a compact, semantic inventory. Paths and URLs stay code-only."""
    available = [
        item for item in (assets or [])
        if isinstance(item, dict) and item.get("asset_id")
    ]
    by_id = {item["asset_id"]: item for item in available}
    roots = {}
    lineage = {}

    for asset in available:
        current = asset
        seen = set()
        depth = 0
        while current.get("source_asset_id") in by_id and current["asset_id"] not in seen:
            seen.add(current["asset_id"])
            current = by_id[current["source_asset_id"]]
            depth += 1
        root_id = current["asset_id"]
        roots.setdefault(root_id, current)
        lineage[asset["asset_id"]] = (root_id, depth)

    root_order = sorted(
        roots,
        key=lambda asset_id: (roots[asset_id].get("created_at") or 0, asset_id),
    )
    root_numbers = {asset_id: index for index, asset_id in enumerate(root_order, start=1)}
    rows = []
    for asset in sorted(available, key=lambda item: (item.get("created_at") or 0, item["asset_id"]))[:limit]:
        root_id, depth = lineage[asset["asset_id"]]
        root_number = root_numbers[root_id]
        version = depth + 1
        if depth == 0:
            label = f"第 {root_number} 个原始视觉素材"
        else:
            label = f"第 {root_number} 个素材的第 {version} 版"
        semantic_label = _compact_text(asset.get("asset_label") or asset.get("operation_prompt"))
        operation_prompt = _compact_text(asset.get("operation_prompt"))
        rows.append({
            "asset_id": asset["asset_id"],
            "label": semantic_label or label,
            "operation_prompt": operation_prompt or None,
            "kind": asset.get("asset_type") or "image",
            "version": version,
            "root_asset_id": root_id,
            "source_asset_id": asset.get("source_asset_id") or None,
            "is_active": bool(asset.get("is_active")),
        })
    return rows


def resolve_visual_assets(assets, context_refs=None, fallback_active=False):
    """Resolve only known asset IDs; never trust an LLM-supplied file path."""
    available = [item for item in (assets or []) if isinstance(item, dict) and item.get("asset_id")]
    by_id = {item["asset_id"]: item for item in available}
    resolved = []
    seen = set()
    for ref in context_refs or []:
        asset = by_id.get(str(ref or "").strip())
        if asset and asset["asset_id"] not in seen:
            resolved.append(_with_local_path(asset))
            seen.add(asset["asset_id"])

    if not resolved and fallback_active:
        active = next((item for item in available if item.get("is_active")), None)
        if active:
            resolved.append(_with_local_path(active))
    return resolved


def _with_local_path(asset):
    normalized = dict(asset)
    if normalized.get("local_path") and os.path.isfile(normalized["local_path"]):
        return normalized
    normalized["local_path"] = None
    public_url = str(normalized.get("public_url") or "")
    parsed = urlparse(public_url)
    file_name = os.path.basename(parsed.path)
    if re.fullmatch(r"[^/\\]+\.(?:png|jpg|jpeg|webp|gif)", file_name, re.I):
        candidate = os.path.join(os.getenv("UPLOAD_DIR", "./temp_images"), file_name)
        if os.path.isfile(candidate):
            normalized["local_path"] = candidate
    return normalized


def tool_image_asset_event(tool_name, result, source_asset_id=None):
    """Extract a successful generated/edited image into a persistence event."""
    if tool_name not in {"generate_image", "image_edit"} or not isinstance(result, dict):
        return None
    image_url = str(result.get("image_url") or "").strip()
    if result.get("status") != "1" or not image_url:
        return None
    operation_prompt = _compact_text(
        result.get("edit_prompt") if tool_name == "image_edit" else result.get("prompt"),
        limit=500,
    ) or None
    action_label = "图片编辑结果" if tool_name == "image_edit" else "图片生成结果"
    asset_label = f"{action_label}：{operation_prompt}" if operation_prompt else action_label
    return {
        "asset_type": "image_edit" if tool_name == "image_edit" else "image_generate",
        "public_url": image_url,
        "local_path": None,
        "source_asset_id": source_asset_id or None,
        "asset_label": asset_label,
        "operation_prompt": operation_prompt,
    }
