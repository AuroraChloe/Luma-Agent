import json
import re

from chat_protocol import add_chat_usage, message_text_only, usage_to_dict
from llm import llm_chat
from rag import embedding
from sql import search_rag_chunks


RAG_TOP_K = 5


def rag_trace_item(mode="none", collection_id=None, query="", retrieval_query="", sources=None, stage="rag_retrieval"):
    return {
        "stage": stage,
        "name": "rag_search",
        "content": {
            "mode": mode,
            "collection_id": collection_id,
            "query": query,
            "retrieval_query": retrieval_query,
            "sources": sources or [],
        },
        "tool_calls": [],
        "type": "RagRetrieval",
    }


def rag_running_usage(rag_scope=None, query=""):
    rag_scope = rag_scope or {"mode": "none"}
    mode = rag_scope.get("mode") or "none"
    collection_id = rag_scope.get("collection_id") if mode == "collection" else None
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "trace": [rag_trace_item(
            mode=mode,
            collection_id=collection_id,
            query=query,
            retrieval_query=query,
            stage="rag_retrieval",
        )],
        "used_tools": ["rag_search"],
    }


def _assistant_content(response):
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None)
    return (getattr(message, "content", "") or "").strip()


def _strip_reference_tail(text):
    text = str(text or "").strip()
    return re.sub(r"\n?\s*参考[:：]\s*(?:\[[^\]]+\]\s*)+$", "", text).strip()


def _source_lines(chunks):
    lines = []
    seen_parent_ids = set()
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        parent_id = metadata.get("parent_id")
        if parent_id:
            if parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(parent_id)
        filename = metadata.get("filename") or chunk.get("document_id")
        score = chunk.get("score")
        score_text = f", score={score:.3f}" if isinstance(score, (int, float)) else ""
        content = metadata.get("parent_content") or chunk.get("content") or ""
        context_type = "parent" if metadata.get("parent_content") else "chunk"
        lines.append(
            f"[{index}] source={filename}, chunk_index={chunk.get('chunk_index')}, context={context_type}{score_text}\n"
            f"{content}"
        )
    return "\n\n".join(lines)


def _public_rag_source(item):
    metadata = dict(item.get("metadata") or {})
    parent_content = metadata.pop("parent_content", None)
    if parent_content:
        metadata["parent_content_chars"] = len(str(parent_content))
        metadata["uses_parent_context"] = True
    return {
        "chunk_id": item.get("chunk_id"),
        "document_id": item.get("document_id"),
        "collection_id": item.get("collection_id"),
        "chunk_index": item.get("chunk_index"),
        "score": item.get("score"),
        "metadata": metadata,
    }


def _format_history_for_rewrite(messages, max_messages=8):
    rows = []
    for item in (messages or [])[-max_messages:]:
        role = item.get("role")
        if role not in ("user", "assistant"):
            continue
        content = message_text_only(item.get("content", "")).strip()
        if not content:
            continue
        label = "用户" if role == "user" else "助手"
        rows.append(f"{label}: {content}")
    return "\n".join(rows)


def _rewrite_query(model, messages, query):
    history_text = _format_history_for_rewrite((messages or [])[:-1])
    rewrite_messages = [
    {
        "role": "system",
        "content": (
            "你是一个知识库检索 Query Rewrite 模块。\n\n"
            "任务：根据历史对话和当前用户问题，将当前问题改写为适合向量检索的独立查询语句。\n\n"
            "## 判断逻辑（按顺序执行）\n"
            "1. 若历史对话为空或为「无」→ 首问句，原样输出，不做任何改写\n"
            "2. 若当前问题完整独立、无需上下文即可理解 → 原样输出\n"
            "3. 若当前问题存在指代词或省略，且历史对话中有明确对应信息 → 补全指代后输出\n"
            "4. 其他情况 → 原样输出当前问题，不引入历史\n\n"
            "## 改写规则\n"
            "- 只补全必要的指代或省略，不扩大问题范围\n"
            "- 不编造历史对话中没有的信息\n"
            "- 不引入助手回答中的无关补充内容\n"
            "- 改写结果为一句简洁的陈述或疑问句，不超过 30 字\n"
            "- 只输出改写后的查询语句，不输出任何解释、思考过程、JSON 或 Markdown\n\n"
            "## 示例\n"
            "历史：用户询问了商品退款流程\n"
            "当前问题：那超过30天还能退吗\n"
            "输出：超过30天的商品退款申请流程是什么\n\n"
            "历史：用户询问了A产品的价格\n"
            "当前问题：B产品呢\n"
            "输出：B产品的价格是多少\n\n"
            "历史：用户询问了天气情况\n"
            "当前问题：Python怎么读取CSV文件\n"
            "输出：Python怎么读取CSV文件\n\n"
            "历史：无\n"
            "当前问题：如何申请售后服务\n"
            "输出：如何申请售后服务"
            ),
        },
    {
        "role": "user",
        "content": (
            f"历史对话：\n{history_text or '无'}\n\n"
            f"当前用户问题：\n{query}\n\n"
            "改写后的检索查询："
        ),
    },
    ]
    response = llm_chat(model, rewrite_messages, temperature=0)
    rewritten_query = _assistant_content(response)
    return (rewritten_query or query).strip(), usage_to_dict(getattr(response, "usage", None))


def run_rag_chat(client_id, *, rag_scope, model, messages, user_message, kwargs=None):
    kwargs = dict(kwargs or {})
    rag_scope = rag_scope or {"mode": "none"}
    mode = rag_scope.get("mode") or "none"
    collection_id = rag_scope.get("collection_id") if mode == "collection" else None

    query = message_text_only(user_message).strip()
    if not query:
        query = "请根据知识库回答当前问题。"

    rewrite_usage = None
    retrieval_query = query
    try:
        retrieval_query, rewrite_usage = _rewrite_query(model, messages, query)
    except Exception as exc:
        print(f"rag query rewrite fallback: {type(exc).__name__}: {exc}", flush=True)

    query_vector = embedding(retrieval_query, input_type="query")
    chunks = search_rag_chunks(
        client_id,
        query_vector,
        collection_id=collection_id,
        limit=int(rag_scope.get("top_k") or RAG_TOP_K),
    )
    if not chunks:
        return {
            "content": "我没有在当前知识库里找到和这个问题相关的内容。",
            "usage": {
                **rag_running_usage(rag_scope, query=query),
                "rag_sources": [],
            },
        }

    source_text = _source_lines(chunks)
    rag_messages = [
        {
            "role": "system",
            "content": (
                "你是 LumaNova 知识库问答助手。当前任务只有一个：回答“用户问题”本身。"
                "只使用与用户问题直接相关的知识库片段；忽略同批召回中与问题不直接相关的片段。"
                "不要把知识库里的其他主题、其他对象、其他场景、其他规则、历史对话或相似问题扩展进答案。"
                "先判断用户明确询问的范围、对象和信息类型，只回答这些内容；不要主动补充未被询问的并列信息、背景信息或延伸建议。"
                "如果片段不足以支持结论，要明确说明知识库里没有足够依据。"
                "不要调用工具，不要处理生图、修图、天气等执行型请求；当前模式只做知识库问答。"
                "回答要自然、简洁，避免主动补充未被询问的内容，不要在答案末尾输出参考编号或来源列表。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户问题：\n{query}\n\n"
                f"知识库片段：\n{source_text}"
            ),
        },
    ]
    allowed_kwargs = {key: value for key, value in kwargs.items() if key not in {"tools", "tool_choice"}}
    response = llm_chat(model, rag_messages, **allowed_kwargs)
    content = _strip_reference_tail(_assistant_content(response))
    usage = add_chat_usage(rewrite_usage, usage_to_dict(getattr(response, "usage", None))) or {}
    usage["rag_sources"] = [_public_rag_source(item) for item in chunks]
    usage["trace"] = [rag_trace_item(
        mode=mode,
        collection_id=collection_id,
        query=query,
        retrieval_query=retrieval_query,
        sources=usage["rag_sources"],
    )]
    usage["used_tools"] = ["rag_search"]
    return {"content": content or "我没有在当前知识库里整理出可靠回答。", "usage": usage}
