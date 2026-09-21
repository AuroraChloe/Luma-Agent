"""LangChain chat-model construction for the active Web Agent runtime."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

from provider import DEFAULT_LLM_BASE_URL, llm_api_key


DEFAULT_AGENT_MODEL = os.getenv("AGENT_MODEL") or os.getenv("DEFAULT_CHAT_MODEL", "")


def build_agent_llm(model=DEFAULT_AGENT_MODEL):
    """Build the OpenAI-compatible LangChain model used by the Web Agent."""
    return ChatOpenAI(
        model=model,
        api_key=llm_api_key(),
        base_url=os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        temperature=0,
        timeout=float(os.getenv("AGENT_TIMEOUT", "75")),
        max_retries=0,
    )
