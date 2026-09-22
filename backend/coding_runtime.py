"""Read-only coding agent runtime.

The first coding mode deliberately has no edit_file or shell tool.  It accepts
an uploaded project workspace, lets the model inspect it through bounded read
tools, and returns an engineering analysis with an observable trace.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from langchain.agents import create_agent
from langchain_core.tools import StructuredTool

from agent_llm import build_agent_llm


CODING_WORKSPACE_ROOT = Path(
    os.getenv("CODING_WORKSPACE_ROOT", "/opt/key_college/coding_workspaces")
).resolve()
CODING_MAX_FILES = int(os.getenv("CODING_MAX_FILES", "1600"))
CODING_MAX_FILE_BYTES = int(os.getenv("CODING_MAX_FILE_BYTES", str(2 * 1024 * 1024)))
CODING_MAX_TOTAL_BYTES = int(os.getenv("CODING_MAX_TOTAL_BYTES", str(80 * 1024 * 1024)))
CODING_MAX_READ_LINES = int(os.getenv("CODING_MAX_READ_LINES", "700"))
CODING_MAX_SEARCH_RESULTS = int(os.getenv("CODING_MAX_SEARCH_RESULTS", "80"))

IGNORED_PARTS = {
    ".git", ".svn", ".hg", "node_modules", "vendor", "venv", ".venv",
    "__pycache__", ".next", "dist", "build", "coverage", "target",
}
SENSITIVE_NAMES = {".env", ".env.local", ".env.production", ".env.development"}
SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".crt", ".token")


def _normalise_relative_path(value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    raw = raw.lstrip("/")
    path = PurePosixPath(raw)
    if not raw or path == PurePosixPath(".") or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("invalid relative path")
    return "/".join(path.parts)


def _is_ignored_path(relative_path: str) -> bool:
    parts = PurePosixPath(relative_path).parts
    name = parts[-1].lower()
    return any(part.lower() in IGNORED_PARTS for part in parts) or name in SENSITIVE_NAMES or name.endswith(SENSITIVE_SUFFIXES)


def workspace_path_for_client(client_id: int, workspace_id: str) -> Path:
    if not str(client_id).isdigit() or not re.fullmatch(r"codews_[a-f0-9]{32}", str(workspace_id or "")):
        raise ValueError("invalid coding workspace")
    candidate = (CODING_WORKSPACE_ROOT / str(client_id) / workspace_id).resolve()
    owner_root = (CODING_WORKSPACE_ROOT / str(client_id)).resolve()
    if candidate.parent != owner_root:
        raise ValueError("invalid coding workspace")
    if not candidate.is_dir():
        raise FileNotFoundError("coding workspace not found")
    return candidate


async def save_coding_workspace(client_id: int, files, relative_paths=None) -> dict:
    """Persist a browser directory upload into a user-owned workspace."""
    path_hints = list(relative_paths or [])
    workspace_id = f"codews_{uuid.uuid4().hex}"
    workspace = CODING_WORKSPACE_ROOT / str(client_id) / workspace_id
    workspace.mkdir(parents=True, exist_ok=False)
    accepted = []
    skipped = []
    total_bytes = 0

    try:
        for index, upload in enumerate(files or []):
            raw_name = path_hints[index] if index < len(path_hints) and path_hints[index] else upload.filename
            try:
                relative_path = _normalise_relative_path(raw_name)
            except ValueError:
                skipped.append({"path": str(raw_name or ""), "reason": "invalid_path"})
                continue
            if _is_ignored_path(relative_path):
                skipped.append({"path": relative_path, "reason": "ignored_or_sensitive"})
                continue
            if len(accepted) >= CODING_MAX_FILES:
                skipped.append({"path": relative_path, "reason": "file_limit"})
                continue

            destination = (workspace / relative_path).resolve()
            if workspace not in destination.parents:
                raise ValueError("workspace path escaped")
            destination.parent.mkdir(parents=True, exist_ok=True)
            file_bytes = 0
            with destination.open("wb") as output:
                while True:
                    chunk = await upload.read(1024 * 256)
                    if not chunk:
                        break
                    file_bytes += len(chunk)
                    total_bytes += len(chunk)
                    if file_bytes > CODING_MAX_FILE_BYTES or total_bytes > CODING_MAX_TOTAL_BYTES:
                        raise ValueError("coding workspace exceeds upload size limit")
                    output.write(chunk)
            accepted.append({"path": relative_path, "bytes": file_bytes})

        if not accepted:
            raise ValueError("no readable source files were uploaded")
        (workspace / ".luma_workspace.json").write_text(
            json.dumps({"workspace_id": workspace_id, "files": accepted}, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "workspace_id": workspace_id,
            "file_count": len(accepted),
            "total_bytes": total_bytes,
            "skipped": skipped,
        }
    except Exception:
        shutil.rmtree(workspace, ignore_errors=True)
        raise


def _safe_file(root: Path, relative_path: str) -> Path:
    clean = _normalise_relative_path(relative_path)
    if _is_ignored_path(clean):
        raise ValueError("file is not available for analysis")
    path = (root / clean).resolve()
    if root not in path.parents or not path.is_file():
        raise FileNotFoundError("file not found")
    return path


def _safe_directory(root: Path, relative_path: str) -> Path:
    clean = _normalise_relative_path(relative_path)
    if _is_ignored_path(clean):
        raise ValueError("directory is not available for analysis")
    path = (root / clean).resolve()
    if root not in path.parents or not path.is_dir():
        raise FileNotFoundError("directory not found")
    return path


def _iter_source_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == ".luma_workspace.json" or _is_ignored_path(relative):
            continue
        yield path, relative


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if b"\x00" in raw[:8192]:
        raise ValueError("binary file is not readable as source")
    return raw.decode("utf-8", errors="replace")


def _tree(root: Path, max_entries: int = 1200) -> str:
    rows = []
    for path, relative in sorted(_iter_source_files(root), key=lambda item: item[1].lower()):
        if len(rows) >= max_entries:
            rows.append("... (tree truncated)")
            break
        rows.append(relative)
    return "\n".join(rows) or "(no readable source files)"


def _tool_error(exc: Exception) -> str:
    return json.dumps({"error": str(exc)}, ensure_ascii=False)


def _make_tools(root: Path):
    def list_files(path: str = "", max_entries: int = 400) -> str:
        try:
            base = root if not path else _safe_directory(root, path)
            prefix = base.relative_to(root).as_posix() if base != root else ""
            rows = []
            for item in sorted(base.iterdir(), key=lambda value: (not value.is_dir(), value.name.lower())):
                relative = item.relative_to(root).as_posix()
                if _is_ignored_path(relative):
                    continue
                suffix = "/" if item.is_dir() else ""
                rows.append(relative + suffix)
                if len(rows) >= max(1, min(int(max_entries or 400), 800)):
                    rows.append("... (listing truncated)")
                    break
            return "\n".join(rows) or f"{prefix or '.'} is empty"
        except Exception as exc:
            return _tool_error(exc)

    def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        try:
            file_path = _safe_file(root, path)
            text = _read_text(file_path)
            lines = text.splitlines()
            start = max(int(start_line or 1), 1)
            end = int(end_line or (start + CODING_MAX_READ_LINES - 1))
            end = max(start, min(end, start + CODING_MAX_READ_LINES - 1, len(lines)))
            selected = [f"{number}: {lines[number - 1]}" for number in range(start, end + 1)]
            header = f"FILE {file_path.relative_to(root).as_posix()} lines {start}-{end}/{len(lines)}"
            return header + "\n" + "\n".join(selected)
        except Exception as exc:
            return _tool_error(exc)

    def search_code(query: str, path: str = "", max_results: int = 50) -> str:
        try:
            needle = str(query or "").strip()
            if not needle:
                return _tool_error(ValueError("query is required"))
            base = root if not path else _safe_directory(root, path)
            results = []
            pattern = None
            try:
                pattern = re.compile(needle, re.IGNORECASE)
            except re.error:
                pass
            for file_path, relative in _iter_source_files(root):
                if base != root and base not in file_path.parents:
                    continue
                try:
                    lines = _read_text(file_path).splitlines()
                except Exception:
                    continue
                for number, line in enumerate(lines, 1):
                    matched = bool(pattern.search(line)) if pattern else needle.casefold() in line.casefold()
                    if matched:
                        results.append(f"{relative}:{number}: {line.strip()[:500]}")
                        if len(results) >= max(1, min(int(max_results or 50), CODING_MAX_SEARCH_RESULTS)):
                            return "\n".join(results) + "\n... (search truncated)"
            return "\n".join(results) or "(no matches)"
        except Exception as exc:
            return _tool_error(exc)

    return [
        StructuredTool.from_function(
            list_files,
            name="list_project_files",
            description="List readable files and directories in the uploaded project. Use before reading files.",
        ),
        StructuredTool.from_function(
            read_file,
            name="read_project_file",
            description="Read a bounded line range from one readable source file. Never write files.",
        ),
        StructuredTool.from_function(
            search_code,
            name="search_project_code",
            description="Search text or a regular expression across readable project source files.",
        ),
    ]


def _content(message: Any) -> str:
    value = getattr(message, "content", "")
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(str(item.get("text") or "") for item in value if isinstance(item, dict)).strip()
    return str(value or "").strip()


def _trace_item(message: Any) -> list[dict]:
    calls = getattr(message, "tool_calls", None) or []
    if calls:
        return [{"type": "ToolCall", "name": call.get("name"), "content": call.get("args") or {}} for call in calls]
    name = getattr(message, "name", None)
    if name:
        return [{"type": "ToolMessage", "name": name, "content": _content(message)}]
    text = _content(message)
    return [{"type": "ModelAnswer", "name": None, "content": text}] if text else []


def _message_usage(message: Any) -> dict:
    metadata = getattr(message, "response_metadata", None) or {}
    usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    usage = usage if isinstance(usage, dict) else getattr(message, "usage_metadata", None)
    if not isinstance(usage, dict):
        return {}
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
    return {
        "prompt_tokens": int(input_tokens),
        "completion_tokens": int(output_tokens),
        "total_tokens": int(usage.get("total_tokens") or input_tokens + output_tokens),
    }


def _add_usage(total: dict, current: dict) -> dict:
    if not current:
        return total
    return {
        "prompt_tokens": total.get("prompt_tokens", 0) + current.get("prompt_tokens", 0),
        "completion_tokens": total.get("completion_tokens", 0) + current.get("completion_tokens", 0),
        "total_tokens": total.get("total_tokens", 0) + current.get("total_tokens", 0),
    }


def run_coding_analysis(
    user_message: str,
    *,
    workspace_path: str,
    messages=None,
    model=None,
    trace_callback=None,
):
    root = Path(workspace_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError("coding workspace not found")
    llm = build_agent_llm(model=model)
    tools = _make_tools(root)
    history = []
    for item in (messages or [])[-8:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = item.get("content")
        if isinstance(content, list):
            content = " ".join(str(block.get("text") or "") for block in content if isinstance(block, dict))
        if content:
            history.append(f"{item['role']}: {str(content)[-1200:]}")

    system_prompt = (
        "你是 Luma 的只读 Coding Agent。你的任务是理解用户上传的代码项目并给出工程分析。\n"
        "你只能使用 list_project_files、read_project_file、search_project_code 三个工具。\n"
        "禁止修改、创建、删除文件，禁止执行 shell，禁止假设没有读到的代码。\n"
        "先建立项目结构，再根据用户问题搜索并读取关键文件；必要时多次调用读取工具。\n"
        "最终用中文 Markdown 输出：项目概览、关键入口、请求/数据流、核心模块、发现的问题或风险、建议的下一步。\n"
        "必须区分‘代码中已确认’和‘根据结构推测’，不要声称已经改动任何代码。\n\n"
        f"上传项目文件树：\n{_tree(root)}\n\n"
        f"最近对话上下文：\n{chr(10).join(history) or '(无)'}"
    )
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        name="luma_coding_readonly_agent",
    )
    trace = []
    usage = {}
    max_iterations = max(int(os.getenv("CODING_ANALYSIS_MAX_ITERATIONS", "12")), 1)
    run_config = {"recursion_limit": max_iterations * 2 + 2}
    input_messages = [{"role": "user", "content": str(user_message or "请分析这个项目") }]
    for update in agent.stream({"messages": input_messages}, config=run_config, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for payload in update.values():
            for item in (payload or {}).get("messages", []) if isinstance(payload, dict) else []:
                for event in _trace_item(item):
                    if event.get("type") == "ModelAnswer" and not event.get("content"):
                        continue
                    trace.append(event)
                    if trace_callback:
                        trace_callback(trace=list(trace), used_tools=[], usage=usage)
                usage = _add_usage(usage, _message_usage(item))

    final_content = next(
        (item.get("content") for item in reversed(trace) if item.get("type") == "ModelAnswer" and item.get("content")),
        "项目分析未生成，请稍后重试。",
    )
    return {
        "content": final_content,
        "trace": trace,
        "used_tools": list(dict.fromkeys(item.get("name") for item in trace if item.get("type") == "ToolCall" and item.get("name"))),
        "usage": usage,
    }
