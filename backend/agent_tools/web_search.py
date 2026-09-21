import os
import re
from tavily import TavilyClient

from .base import AgentTool


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?:\s*年)?(?!\d)")
_YEAR_DATE_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})"
    r"(?:\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"
    r"|\s*[-/.]\s*\d{1,2}(?:\s*[-/.]\s*\d{1,2})?)"
)
_MONTH_RE = re.compile(r"(?<!\d)(\d{1,2})\s*月")
_NUMERIC_DATE_MONTH_RE = re.compile(
    r"(?<!\d)(?:19|20)\d{2}\s*[-/.]\s*(\d{1,2})(?:\s*[-/.]\s*\d{1,2})?"
)
_QUARTER_RE = re.compile(r"(?:第\s*[一二三四1-4]\s*季度|Q[1-4])", re.IGNORECASE)


def _user_time_evidence(user_message="", context=""):
    """Return concrete time markers explicitly supplied by the user.

    Assistant/tool output can contain many historical dates. It must not be
    treated as permission to inject one into a new search query, so only the
    current request and user-labelled context lines are trusted here.
    """
    user_lines = []
    for line in str(context or "").splitlines():
        if line.strip().lower().startswith("user:"):
            user_lines.append(line.split(":", 1)[1])
    evidence = "\n".join([str(user_message or "")] + user_lines)
    return {
        "years": set(_YEAR_RE.findall(evidence)),
        "months": set(_MONTH_RE.findall(evidence))
        | set(_NUMERIC_DATE_MONTH_RE.findall(evidence)),
        "quarters": {item.lower().replace(" ", "") for item in _QUARTER_RE.findall(evidence)},
    }


def _remove_unanchored_time(query, user_message="", context=""):
    """Keep relative time words but remove invented concrete dates."""
    query = str(query or "").strip()
    evidence = _user_time_evidence(user_message=user_message, context=context)

    def remove_date(match):
        year = match.group(1)
        month_match = _MONTH_RE.search(match.group(0))
        numeric_month_match = _NUMERIC_DATE_MONTH_RE.search(match.group(0))
        month = (
            month_match.group(1)
            if month_match
            else numeric_month_match.group(1)
            if numeric_month_match
            else None
        )
        if year not in evidence["years"]:
            return ""
        if month and month not in evidence["months"]:
            return f"{year}年" if "年" in match.group(0) else year
        return match.group(0)

    query = _YEAR_DATE_RE.sub(remove_date, query)

    def remove_year(match):
        return match.group(0) if match.group(1) in evidence["years"] else ""

    query = _YEAR_RE.sub(remove_year, query)

    def remove_month(match):
        return match.group(0) if match.group(1) in evidence["months"] else ""

    query = _MONTH_RE.sub(remove_month, query)
    query = _QUARTER_RE.sub(
        lambda match: match.group(0)
        if match.group(0).lower().replace(" ", "") in evidence["quarters"]
        else "",
        query,
    )
    return re.sub(r"\s+", " ", query).strip(" ,，;；:：")

def web_search(query: str, search_depth: str = "advanced") -> dict:
    query = (query or "").strip()
    search_depth = (search_depth or "advanced").strip() or "advanced"

    if not query:
        return {"status": "0", "info": "MISSING_QUERY"}

    if not TAVILY_API_KEY:
        return {"status": "0", "info": "MISSING_TAVILY_API_KEY", "query": query}

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        data = client.search(
            query=query,
            search_depth=search_depth,
            include_answer=True,
        )
    except Exception as exc:
        return {
            "status": "0",
            "info": "SEARCH_API_ERROR",
            "query": query,
            "error": str(exc),
        }

    results = []
    for item in data.get("results") or []:
        results.append({
            "title": item.get("title") or "",
            "content": (item.get("content") or "")[:1800],
        })

    return {
        "status": "1",
        "info": "OK",
        "query": data.get("query") or query,
        "answer": data.get("answer") or "",
        "results": results,
        "response_time": data.get("response_time"),
        "request_id": data.get("request_id"),
    }

def _validate_web_args(args, user_message="", context=""):
    query = str(args.get("query") or "").strip()
    if not query:
        query = (user_message or "").strip()
    query = _remove_unanchored_time(query, user_message=user_message, context=context)
    if not query:
        query = (user_message or "").strip()

    search_depth = str(args.get("search_depth") or "advanced").strip()
    if search_depth not in {"basic", "advanced"}:
        search_depth = "advanced"

    return {
        "query": query,
        "search_depth": search_depth,
    }

def _web_fallback(result, user_message=""):
    if not isinstance(result, dict):
        return str(result)

    info = result.get("info")
    if info == "MISSING_QUERY":
        return "我没有识别到要搜索的内容，请补充一下你想查什么。"
    if info == "MISSING_TAVILY_API_KEY":
        return "网页搜索服务还没有配置 API Key。"
    if info == "SEARCH_API_ERROR":
        return "网页搜索服务暂时不可用，可以稍后再试。"

    answer = result.get("answer")
    if answer:
        return answer
    
    results = result.get("results") or []
    if not results:
        return "我没有搜到足够相关的网页结果。"

    lines = ["我找到了一些相关网页："]
    for i, item in enumerate(results[:5], start=1):
        title = item.get("title") or "未命名网页"
        content = item.get("content") or ""
        lines.append(f"{i}. {title}")
        if content:
            lines.append(f"   {content[:180]}")

    return "\n".join(lines)


WEB_SEARCH_TOOL = AgentTool(
    name="web_search",
    description=(
        "根据用户 query 查询互联网相关的在线信息。"
        "当前问题需要新的线上事实时调用；连续追问要结合上下文补全搜索主体，"
        "但如果只是总结、解释或改写上一轮搜索结果，不要重复搜索。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "用于线上搜索的查询内容，也就是完整的query",
            },
            "search_depth": {
                "type": "string",
                "description": "搜索的深度，旨在查询检索最相关的来源和内容片段，默认使用advanced",
            }
        },
        "required": ["query"],
    },
    handler=web_search,
    required=("query",),
    missing_args_message="",
    validate_args=_validate_web_args,
    fallback_formatter=_web_fallback,
    argument_prompt=(
    "你只负责为 web_search 工具准备参数。"
    "请根据最近对话上下文和当前用户请求，生成适合网页搜索的 query。"
    "如果当前请求是承接上文的追问，要把上文中的核心主体、限定条件补全到 query 中。"
    "如果当前请求本身已经完整，就尽量保持原意，不要过度扩写。"
    "时间条件必须忠实于用户：最新、最近、目前、当前等相对时间保持原样；"
    "除非用户当前问题或用户历史明确写出年份、月份、日期或季度，否则禁止自行补入具体时间。"
    "query 应该是自然搜索语句，不要写成回答。"
    "search_depth 默认使用 advanced。"
    ),
    answer_prompt=(
        "你会收到用户问题和 web_search 工具返回的 JSON。"
        "工具会返回搜索到的多条信息，需要结合各信息的内容综合分析，合理回答用户问题。不要编造没有出现的数据。"
        "如果工具 JSON 表示缺少搜索条件，配置错误或没有搜索结果，就自然地说明原因并提示用户补充。"
        "回答要亲切、简洁、像聊天，不要暴露 JSON、接口名、工具名或调用过程。"
        "如果搜索到的信息和用户问题出现巨大偏差或相关性过低，可以按自己的理解回复问题，但要明确说明线上相关的信息太少。"
    ),
)
