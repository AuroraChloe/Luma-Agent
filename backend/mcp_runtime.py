"""MCP server discovery and LangChain tool loading."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Literal

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from sql import get_chat_mcp_state, set_chat_mcp_state


_client: MultiServerMCPClient | None = None
_tools = None
_tool_groups = None
_tools_lock = asyncio.Lock()


def _exception_summary(exc: BaseException) -> str:
    nested = getattr(exc, "exceptions", None)
    if nested:
        details = "; ".join(_exception_summary(item) for item in nested)
        return f"{type(exc).__name__}: {details}"
    return f"{type(exc).__name__}: {exc}"


class StartMbtiTestArgs(BaseModel):
    testType: Literal["simplified", "cognitive"] = Field(
        default="simplified",
        description="测试类型：simplified 为简化版，cognitive 为认知功能版。",
    )


class AnswerMbtiQuestionArgs(BaseModel):
    score: int = Field(
        ...,
        ge=1,
        le=5,
        description="当前问题的评分：1强烈不同意，2不同意，3中立，4同意，5强烈同意。",
    )


class EmptyMcpArgs(BaseModel):
    pass


MBTI_TOOL_SPECS = {
    "start_mbti_test": {
        "description": (
            "开始一项新的 MBTI 测试并返回第一道题。"
            "只有用户明确要求开始、重做或进行 MBTI 测试时使用。"
        ),
        "args_schema": StartMbtiTestArgs,
    },
    "answer_question": {
        "description": (
            "提交用户对当前 MBTI 题目的 1-5 分回答，并返回下一道题。"
            "仅在当前聊天会话已经开始 MBTI 测试时使用。测试会话状态由系统自动维护。"
        ),
        "args_schema": AnswerMbtiQuestionArgs,
    },
    "get_progress": {
        "description": (
            "查询当前聊天会话中 MBTI 测试的完成进度。"
            "仅在已经开始测试且用户询问进度时使用。"
        ),
        "args_schema": EmptyMcpArgs,
    },
    "calculate_mbti_result": {
        "description": (
            "在 MBTI 测试题目全部完成后计算最终类型和分析结果。"
            "不要在题目尚未完成时提前调用。"
        ),
        "args_schema": EmptyMcpArgs,
    },
}

def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _server_connections() -> dict:
    connections = {}

    if _enabled(os.getenv("MCP_MBTI_ENABLED"), default=True):
        url = os.getenv("MCP_MBTI_URL", "").strip()
        api_key = os.getenv("MCP_MBTI_API_KEY", "").strip()
        if url and api_key:
            connections["mbti_test"] = {
                "transport": "http",
                "url": url,
                "headers": {"XBY-APIKEY": api_key},
            }

    if _enabled(os.getenv("MCP_MCD_ENABLED"), default=True):
        url = os.getenv("MCP_MCD_URL", "").strip()
        api_key = os.getenv("MCP_MCD_API_KEY", "").strip()
        if url and api_key:
            connections["mcd_mcp"] = {
                "transport": "http",
                "url": url,
                "headers": {"Authorization": f"Bearer {api_key}"},
            }

    return connections


def _mcp_client() -> MultiServerMCPClient | None:
    global _client
    if _client is None:
        connections = _server_connections()
        if not connections:
            return None
        _client = MultiServerMCPClient(connections)
    return _client


async def load_mcp_tool_groups(*, force_refresh: bool = False):
    """Discover tools per MCP server and retain their server boundaries."""
    global _tools, _tool_groups
    if _tool_groups is not None and not force_refresh:
        return {name: list(tools) for name, tools in _tool_groups.items()}

    async with _tools_lock:
        if _tool_groups is not None and not force_refresh:
            return {name: list(tools) for name, tools in _tool_groups.items()}

        client = _mcp_client()
        if client is None:
            _tools = []
            _tool_groups = {}
            return {}

        timeout = max(float(os.getenv("MCP_TOOL_LOAD_TIMEOUT_SECONDS", "20")), 1.0)
        server_names = list(client.connections)

        async def discover_server(server_name):
            try:
                return await asyncio.wait_for(
                    client.get_tools(server_name=server_name),
                    timeout=timeout,
                )
            except BaseException as exc:
                return exc

        discovered = await asyncio.gather(*[
            discover_server(server_name)
            for server_name in server_names
        ])
        next_groups = {}
        for server_name, result in zip(server_names, discovered):
            if isinstance(result, BaseException):
                print(
                    f"mcp tool discovery failed server={server_name}: "
                    f"{_exception_summary(result)}",
                    flush=True,
                )
                continue
            next_groups[server_name] = list(result or [])

        _tool_groups = next_groups
        _tools = [
            tool
            for tools in _tool_groups.values()
            for tool in tools
        ]
        print(
            "mcp tools loaded groups="
            f"{ {name: [tool.name for tool in tools] for name, tools in _tool_groups.items()} }",
            flush=True,
        )
        return {name: list(tools) for name, tools in _tool_groups.items()}


async def load_mcp_tools(*, force_refresh: bool = False):
    """Load all MCP tools while preserving compatibility with flat callers."""
    groups = await load_mcp_tool_groups(force_refresh=force_refresh)
    return [tool for tools in groups.values() for tool in tools]


def _structured_payload(result):
    blocks = result if isinstance(result, list) else [result]
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            value = block.get("text")
        elif isinstance(block, str):
            value = block
        else:
            continue
        try:
            parsed = json.loads(value)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return result if isinstance(result, dict) else {}


def _mbti_tool_adapter(tool, *, client_id, chat_session_id):
    spec = MBTI_TOOL_SPECS[tool.name]
    runtime_state = {"session": None, "loaded": False}

    async def load_state():
        if runtime_state["loaded"]:
            return runtime_state["session"]
        runtime_state["loaded"] = True
        if client_id is None or not chat_session_id:
            return None
        runtime_state["session"] = await asyncio.to_thread(
            get_chat_mcp_state,
            client_id,
            chat_session_id,
            "mbti_test",
            "test_session",
        )
        return runtime_state["session"]

    async def save_state(session):
        runtime_state["loaded"] = True
        runtime_state["session"] = session
        if client_id is None or not chat_session_id:
            return
        await asyncio.to_thread(
            set_chat_mcp_state,
            client_id,
            chat_session_id,
            "mbti_test",
            session,
            "test_session",
        )

    async def invoke(**raw_args):
        arguments = dict(raw_args or {})
        if tool.name != "start_mbti_test":
            session = await load_state()
            if not isinstance(session, dict):
                return {
                    "status": "0",
                    "info": "MBTI_TEST_NOT_STARTED",
                    "message": "当前对话还没有开始 MBTI 测试。",
                }
            arguments["session"] = session

        result = await tool.ainvoke(arguments)
        payload = _structured_payload(result)
        next_session = payload.get("session") if isinstance(payload, dict) else None
        if isinstance(next_session, dict):
            await save_state(next_session)
        return result

    return StructuredTool.from_function(
        coroutine=invoke,
        name=tool.name,
        description=spec["description"],
        args_schema=spec["args_schema"],
        infer_schema=False,
    )


async def load_agent_mcp_tool_groups(*, client_id=None, chat_session_id=None):
    """Build request-scoped tools and preserve the originating MCP server."""
    groups = await load_mcp_tool_groups()
    adapted_groups = {}
    for server_name, tools in groups.items():
        adapted = []
        for tool in tools:
            if tool.name in MBTI_TOOL_SPECS:
                adapted.append(_mbti_tool_adapter(
                    tool,
                    client_id=client_id,
                    chat_session_id=chat_session_id,
                ))
            else:
                adapted.append(tool)
        adapted_groups[server_name] = adapted
    return adapted_groups


async def load_agent_mcp_tools(*, client_id=None, chat_session_id=None):
    """Compatibility helper returning request-scoped MCP tools as a flat list."""
    groups = await load_agent_mcp_tool_groups(
        client_id=client_id,
        chat_session_id=chat_session_id,
    )
    return [tool for tools in groups.values() for tool in tools]
