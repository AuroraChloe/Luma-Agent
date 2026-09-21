import os
import requests
import json
import re
import hashlib
from pathlib import Path
from llm import llm_chat
from langchain_text_splitters import RecursiveCharacterTextSplitter

from provider import DEFAULT_LLM_BASE_URL, llm_api_key

def _load_dotenv(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))

_load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "nvidia/llama-nemotron-embed-1b-v2")
DEFAULT_EMBEDDING_TIMEOUT = float(os.getenv("RAG_EMBEDDING_TIMEOUT", "30"))
RAG_CHUNKER = os.getenv("RAG_CHUNKER", "router").strip().lower()
RAG_SEMANTIC_CHUNK_MODEL = os.getenv("RAG_SEMANTIC_CHUNK_MODEL") or os.getenv("DEFAULT_CHAT_MODEL", "")
RAG_DOC_TYPE_MODEL = os.getenv("RAG_DOC_TYPE_MODEL", RAG_SEMANTIC_CHUNK_MODEL)
RAG_SEMANTIC_CHUNK_TIMEOUT = float(os.getenv("RAG_SEMANTIC_CHUNK_TIMEOUT", "180"))
RAG_SEMANTIC_SOURCE_CHARS = int(os.getenv("RAG_SEMANTIC_SOURCE_CHARS", "2500"))
RAG_SEMANTIC_SOURCE_OVERLAP = int(os.getenv("RAG_SEMANTIC_SOURCE_OVERLAP", "120"))
RAG_SEMANTIC_MIN_CHARS = int(os.getenv("RAG_SEMANTIC_MIN_CHARS", "180"))
RAG_SEMANTIC_MAX_CHARS = int(os.getenv("RAG_SEMANTIC_MAX_CHARS", "1200"))
RAG_SEMANTIC_MAX_CHUNKS_PER_WINDOW = int(os.getenv("RAG_SEMANTIC_MAX_CHUNKS_PER_WINDOW", "6"))
RAG_SEMANTIC_FALLBACK = os.getenv("RAG_SEMANTIC_FALLBACK", "0").strip().lower() in {"1", "true", "yes", "on"}
RAG_SEMANTIC_JSON_MODE = os.getenv("RAG_SEMANTIC_JSON_MODE", "1").strip().lower() in {"1", "true", "yes", "on"}
RAG_SEMANTIC_REPAIR_JSON = os.getenv("RAG_SEMANTIC_REPAIR_JSON", "1").strip().lower() in {"1", "true", "yes", "on"}
RAG_DOC_TYPE_SAMPLE_CHARS = int(os.getenv("RAG_DOC_TYPE_SAMPLE_CHARS", "6000"))
RAG_NOVEL_PARENT_CHARS = int(os.getenv("RAG_NOVEL_PARENT_CHARS", "1800"))
RAG_NOVEL_CHILD_CHARS = int(os.getenv("RAG_NOVEL_CHILD_CHARS", "450"))
RAG_NOVEL_CHILD_OVERLAP = int(os.getenv("RAG_NOVEL_CHILD_OVERLAP", "60"))


def _assistant_text(response):
    choice = response.choices[0] if getattr(response, "choices", None) else None
    message = getattr(choice, "message", None)
    return (getattr(message, "content", "") or "").strip()


def file_checkouter(doc):
    sample = str(doc or "")[:RAG_DOC_TYPE_SAMPLE_CHARS]
    if not sample.strip():
        return "rules_documentation"
    file_check_messages = [
        {
            "role": "system",
            "content": (
                "你是一名文档处理专家，擅长进行文档内容的分析与归类。\n\n"
                "任务：根据文档内容，判断该文档是什么类型的内容.\n\n"
                "## 文档类型 \n"
                "1、literary_works\n"
                "2、rules_documentation\n"
                "## 输出规则\n"
                "只对该文档进行内容分析，只以字符串的形式返回命中‘文档类型’中字段，不要输出其他任何内容。\n"
                "## 示例\n"
                "当前文档内容：一段叙事文学、故事、小说、剧本或散文内容\n"
                "输出：literary_works"
            ),
        },
        {
            "role": "user",
            "content": (
                f"当前文档内容：{sample}\n\n"
            ),
        },
    ]
    try:
        response = llm_chat(RAG_DOC_TYPE_MODEL, file_check_messages, temperature=0)
        result = _assistant_text(response).strip().lower()
    except Exception as exc:
        print(f"rag doc type detector fallback: {type(exc).__name__}: {exc}", flush=True)
        result = ""
    allowed = {"literary_works", "rules_documentation"}
    if result in allowed:
        return result
    if _looks_like_literary_work(sample):
        return "literary_works"
    return "rules_documentation"


def _looks_like_literary_work(text):
    sample = str(text or "")[:RAG_DOC_TYPE_SAMPLE_CHARS]
    dialogue_hits = len(re.findall(r"[“\"].{2,80}[”\"]", sample))
    narrative_marks = sum(sample.count(mark) for mark in ("他说", "她说", "我说", "心想", "看着", "走进", "望着", "沉默"))
    paragraph_count = len([part for part in re.split(r"\n\s*\n", sample) if part.strip()])
    return dialogue_hits >= 5 or narrative_marks >= 6 or (paragraph_count >= 8 and narrative_marks >= 2)

def fallback_split(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)
    return chunks


def split(text):
    if RAG_CHUNKER in {"novel", "literary", "literary_works"}:
        return literary_parent_child_split(text)
    if RAG_CHUNKER in {"router", "auto", "adaptive", "llm"}:
        doc_type = file_checkouter(text)
        print(f"rag chunk router doc_type={doc_type}", flush=True)
        if doc_type == "literary_works":
            chunks = literary_parent_child_split(text)
            if chunks:
                print(f"rag literary parent-child split completed chunks={len(chunks)}", flush=True)
                return chunks
    elif RAG_CHUNKER not in {"semantic", "semantic_llm"}:
        return fallback_split(text)
    try:
        chunks = semantic_split(text)
        if chunks:
            print(f"rag semantic split completed chunks={len(chunks)}", flush=True)
            return chunks
    except Exception as exc:
        print(f"rag semantic split failed: {type(exc).__name__}: {exc}", flush=True)
        if not RAG_SEMANTIC_FALLBACK:
            raise
    return fallback_split(text)


def literary_parent_child_split(text):
    parent_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "，", ""],
        chunk_size=RAG_NOVEL_PARENT_CHARS,
        chunk_overlap=0,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "！", "？", "，", ""],
        chunk_size=RAG_NOVEL_CHILD_CHARS,
        chunk_overlap=RAG_NOVEL_CHILD_OVERLAP,
    )
    chunks = []
    parent_docs = parent_splitter.create_documents([str(text or "")])
    for parent_index, parent in enumerate(parent_docs):
        parent_content = parent.page_content.strip()
        if not parent_content:
            continue
        parent_hash = hashlib.sha1(parent_content[:1000].encode("utf-8", errors="ignore")).hexdigest()[:12]
        parent_id = f"literary_{parent_hash}_p{parent_index}"
        child_docs = child_splitter.create_documents([parent_content])
        for child_index, child in enumerate(child_docs):
            child_content = child.page_content.strip()
            if not child_content:
                continue
            chunks.append({
                "title": f"文学片段 {parent_index + 1}",
                "summary": "文学类父子分块：子 chunk 用于检索，父 chunk 用于回答上下文。",
                "keywords": [],
                "content": child_content,
                "metadata": {
                    "chunker": "literary_parent_child",
                    "doc_type": "literary_works",
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "parent_index": parent_index,
                    "child_index": child_index,
                    "parent_content": parent_content,
                },
            })
    return chunks


def novel_parent_child_split(text):
    return literary_parent_child_split(text)


def _coarse_windows(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_SEMANTIC_SOURCE_CHARS,
        chunk_overlap=RAG_SEMANTIC_SOURCE_OVERLAP,
    )
    return splitter.split_text(text)


def _extract_json_payload(text):
    raw = str(text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    object_start = raw.find("{")
    object_end = raw.rfind("}")
    if object_start >= 0 and object_end > object_start:
        return json.loads(raw[object_start:object_end + 1])
    array_start = raw.find("[")
    array_end = raw.rfind("]")
    if array_start >= 0 and array_end > array_start:
        return {"chunks": json.loads(raw[array_start:array_end + 1])}
    raise ValueError("semantic chunker returned non-json content")


def _post_chat_completion(url, api_key, body, timeout):
    payload = dict(body)
    if RAG_SEMANTIC_JSON_MODE:
        payload["response_format"] = {"type": "json_object"}
    response = requests.post(
        url,
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    if not response.ok and "response_format" in payload and response.status_code in {400, 422}:
        payload.pop("response_format", None)
        response = requests.post(
            url,
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
    return response


def _chat_content(payload):
    choices = payload.get("choices") or []
    message = (choices[0] or {}).get("message") if choices else {}
    return (message or {}).get("content") or ""


def _repair_chunk_json(raw_text, url, api_key):
    response = _post_chat_completion(
        url,
        api_key,
        {
            "model": RAG_SEMANTIC_CHUNK_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 JSON 修复器。把用户提供的内容修复成严格合法 JSON。"
                        "只能输出 JSON，不要 markdown，不要解释。"
                        "必须保留 chunks 数组，以及每个 chunk 的 title、summary、keywords、content 字段。"
                    ),
                },
                {
                    "role": "user",
                    "content": str(raw_text or "")[:20000],
                },
            ],
            "temperature": 0,
            "max_tokens": 4096,
        },
        RAG_SEMANTIC_CHUNK_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(f"Semantic chunk json repair failed: HTTP {response.status_code} {response.text[:500]}")
    return _chat_content(response.json())


def _normalize_semantic_chunks(payload, window_index):
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("chunks") or []
    else:
        items = []
    normalized = []
    for raw_index, item in enumerate(items):
        if isinstance(item, str):
            content = item.strip()
            title = ""
            summary = ""
            keywords = []
        elif isinstance(item, dict):
            content = str(item.get("content") or item.get("text") or "").strip()
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            keywords = item.get("keywords") or []
            if not isinstance(keywords, list):
                keywords = [str(keywords)]
            keywords = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
        else:
            continue
        if not content:
            continue
        normalized.append({
            "title": title,
            "summary": summary,
            "content": content,
            "keywords": keywords[:8],
            "metadata": {
                "chunker": "llm_semantic",
                "chunk_model": RAG_SEMANTIC_CHUNK_MODEL,
                "window_index": window_index,
                "raw_chunk_index": raw_index,
            },
        })
    return normalized


def _semantic_chunk_window(text, window_index):
    api_key = llm_api_key()
    base_url = os.getenv("RAG_SEMANTIC_CHUNK_BASE_URL") or os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    url = f"{base_url.rstrip('/')}/chat/completions"
    response = _post_chat_completion(
        url,
        api_key,
        {
            "model": RAG_SEMANTIC_CHUNK_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是 RAG 知识库的语义分块器。你的任务是把输入文本切成适合向量检索的语义 chunk。"
                        "每个 chunk 必须围绕一个完整语义单元，例如一个规则、一个问题答案、一个流程步骤、一个定义或一个案例。"
                        "不要总结替代原文，不要丢失关键约束、数字、实体、禁止话术、例外条件。"
                        "尽量保留原文表达；可以合并过短段落，也可以拆开过长段落。"
                        f"每个 chunk 建议 {RAG_SEMANTIC_MIN_CHARS}-{RAG_SEMANTIC_MAX_CHARS} 个中文字符。"
                        "只输出合法 JSON，不要 markdown，不要解释。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "请按下面 JSON Schema 输出：\n"
                        "{\n"
                        "  \"chunks\": [\n"
                        "    {\n"
                        "      \"title\": \"简短标题\",\n"
                        "      \"summary\": \"一句话概括这个 chunk 的语义用途\",\n"
                        "      \"keywords\": [\"关键词1\", \"关键词2\"],\n"
                        "      \"content\": \"适合向量化和问答引用的原文内容\"\n"
                        "    }\n"
                        "  ]\n"
                        "}\n\n"
                        f"最多输出 {RAG_SEMANTIC_MAX_CHUNKS_PER_WINDOW} 个 chunk。\n"
                        "待切分文本：\n"
                        f"{text}"
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 4096,
        },
        RAG_SEMANTIC_CHUNK_TIMEOUT,
    )
    if not response.ok:
        raise RuntimeError(f"Semantic chunk request failed: HTTP {response.status_code} {response.text[:500]}")
    payload = response.json()
    content = _chat_content(payload)
    try:
        parsed = _extract_json_payload(content)
    except Exception as exc:
        if not RAG_SEMANTIC_REPAIR_JSON:
            raise
        print(f"rag semantic json repair window={window_index}: {type(exc).__name__}: {exc}", flush=True)
        parsed = _extract_json_payload(_repair_chunk_json(content, url, api_key))
    return _normalize_semantic_chunks(parsed, window_index)


def semantic_split(text):
    chunks = []
    for window_index, window in enumerate(_coarse_windows(text)):
        chunks.extend(_semantic_chunk_window(window, window_index))
    return chunks

def embedding(text, input_type="passage"):
    return embedding_with_timeout(text, input_type=input_type)


def embedding_with_timeout(text, model=None, timeout=None, input_type="passage"):
    model = model or DEFAULT_EMBEDDING_MODEL
    timeout = float(timeout or DEFAULT_EMBEDDING_TIMEOUT)
    api_key = NVIDIA_API_KEY or llm_api_key()
    base_url = os.getenv("RAG_EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
    url = f"{base_url.rstrip('/')}/embeddings"
    response = requests.post(
        url,
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "input": text,
            "input_type": input_type,
            "encoding_format": "float",
        },
        timeout=timeout,
    )
    if not response.ok:
        raise RuntimeError(f"Embedding request failed: HTTP {response.status_code} {response.text[:500]}")
    payload = response.json()
    data = payload.get("data") or []
    if not data or "embedding" not in data[0]:
        raise RuntimeError("Embedding response missing data[0].embedding")
    vector = data[0]["embedding"]
    if not isinstance(vector, list) or not vector:
        raise RuntimeError("Embedding response returned empty vector")
    return vector
