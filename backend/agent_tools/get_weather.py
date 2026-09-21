import os
from pathlib import Path

import requests
from langchain_core.tools import tool

from .base import AgentTool


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

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")


def resolve_city_adcode_raw(query: str) -> dict:
    if not WEATHER_API_KEY:
        return {"status": "0", "info": "MISSING_WEATHER_API_KEY"}
    result = requests.get(
        "https://restapi.amap.com/v3/config/district",
        params={
            "key": WEATHER_API_KEY,
            "keywords": query,
            "subdistrict": 0,
            "extensions": "base",
        },
        timeout=15,
    )
    data = result.json()
    districts = data.get("districts") or []
    if not districts:
        return {"status": data.get("status", "0"), "info": data.get("info") or "NO_DISTRICT_FOUND", "query": query}
    district = districts[0]
    return {
        "status": data.get("status", "1"),
        "info": data.get("info", "OK"),
        "query": query,
        "name": district.get("name"),
        "adcode": district.get("adcode"),
        "level": district.get("level"),
    }


def get_weather_raw(adcode: str) -> dict:
    if not WEATHER_API_KEY:
        return {"status": "0", "info": "MISSING_WEATHER_API_KEY"}
    result = requests.get(
        "https://restapi.amap.com/v3/weather/weatherInfo",
        params={"key": WEATHER_API_KEY, "city": adcode},
        timeout=15,
    )
    return result.json()


@tool
def weather_current(location: str) -> dict:
    """根据城市、区县、县、市或行政区名称查询当前实时天气。"""
    location = (location or "").strip()
    if not location:
        return {"status": "0", "info": "MISSING_LOCATION"}

    adcode_result = resolve_city_adcode_raw(location)
    adcode = adcode_result.get("adcode") if isinstance(adcode_result, dict) else None
    if not adcode:
        return {
            "status": "0",
            "info": adcode_result.get("info", "NO_DISTRICT_FOUND") if isinstance(adcode_result, dict) else "NO_DISTRICT_FOUND",
            "location": location,
            "adcode_result": adcode_result,
        }

    weather_result = get_weather_raw(adcode)
    live = None
    if isinstance(weather_result, dict):
        lives = weather_result.get("lives") or []
        if lives:
            live = lives[0]

    return {
        "status": weather_result.get("status", "0") if isinstance(weather_result, dict) else "0",
        "info": weather_result.get("info", "OK") if isinstance(weather_result, dict) else "INVALID_WEATHER_RESPONSE",
        "location": location,
        "adcode": adcode,
        "adcode_result": adcode_result,
        "weather_result": weather_result,
        "live": live,
    }


def _weather_fallback(result, user_message=""):
    data = result
    if not isinstance(data, dict):
        return str(data)

    live = data.get("live") if isinstance(data.get("live"), dict) else None
    if not isinstance(live, dict):
        info = data.get("info") or data.get("message") or "天气工具没有返回可读结果。"
        if info == "INVALID_USER_KEY":
            return "天气工具配置异常：WEATHER_API_KEY 无效或未开通，请检查天气服务配置。"
        if info == "MISSING_WEATHER_API_KEY":
            return "天气工具未配置 WEATHER_API_KEY。"
        if info == "MISSING_LOCATION":
            return "我没有识别到要查询天气的城市或地区，请补充一个地点。"
        if info == "NO_DISTRICT_FOUND":
            query = data.get("location") or "该地区"
            return f"没有查到“{query}”对应的行政区划码，暂时无法查询天气。"
        adcode_result = data.get("adcode_result")
        if isinstance(adcode_result, dict) and adcode_result.get("info") == "NO_DISTRICT_FOUND":
            query = adcode_result.get("query") or data.get("location") or "该地区"
            return f"没有查到“{query}”对应的行政区划码，暂时无法查询天气。"
        return str(info)

    city = live.get("city") or live.get("province") or "该地区"
    weather = live.get("weather") or "未知"
    temperature = live.get("temperature") or "未知"
    humidity = live.get("humidity") or "未知"
    winddirection = live.get("winddirection") or ""
    windpower = live.get("windpower") or ""
    wind = "".join(part for part in (winddirection, "风" if winddirection else "", windpower) if part)
    reply = f"{city}现在{weather}，气温{temperature}℃，湿度{humidity}%"
    if wind:
        reply += f"，{wind}"
    reply += "。"
    if any(keyword in (user_message or "") for keyword in ("伞", "下雨", "雨")):
        if "雨" in str(weather):
            reply += " 当前天气有雨，出门建议带伞。"
        else:
            reply += " 当前天气不是雨天，一般不用特意带伞。"
    return reply


def _location_is_grounded(location, user_message="", context=""):
    if not location:
        return False
    evidence = f"{user_message or ''}\n{context or ''}"
    suffixes = ("市", "区", "县", "自治县", "自治州", "州", "盟", "旗", "省")
    candidates = {location}
    for suffix in suffixes:
        if location.endswith(suffix) and len(location) > len(suffix):
            candidates.add(location[: -len(suffix)])
    return any(candidate and candidate in evidence for candidate in candidates)


def _validate_weather_args(args, user_message="", context=""):
    location = str(args.get("location") or "").strip()
    if not _location_is_grounded(location, user_message=user_message, context=context):
        location = ""
    return {"location": location}


WEATHER_CURRENT_TOOL = AgentTool(
    name="weather_current",
    description=(
        "查询当前、实时天气、气温、湿度、风力、是否需要带伞等天气信息。"
        "仅在当前问题需要新的天气事实时调用；连续追问可以从上下文继承地点，"
        "但如果只是根据已返回天气给穿搭、出行或生活建议，不要重复查询。"
    ),
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "城市、区县、县、市或行政区名称。不要包含天气、现在、几度、今天、吗、啊等词。",
            }
        },
        "required": ["location"],
    },
    handler=weather_current,
    required=("location",),
    missing_args_message="我没有识别到要查询天气的城市或地区，请补充一个地点。",
    validate_args=_validate_weather_args,
    fallback_formatter=_weather_fallback,
    argument_prompt=(
        "你只负责为 weather_current 工具准备参数。"
        "只提取城市、区县、县、市、行政区名称，不要包含“天气、现在、几度、今天、吗、啊”等词。"
        "地点只能来自当前用户问题或最近对话上下文，不要猜测默认城市。"
        "如果当前问题没有地点，可以继承上下文里上一轮天气查询的地点；仍没有地点则返回空字符串。"
    ),
    answer_prompt=(
        "你会收到用户问题和 weather_current 工具返回的 JSON。"
        "必须只基于工具 JSON 里的真实天气数据回答，不能编造没有出现的数据。"
        "如果工具 JSON 表示缺少地点、找不到城市、配置错误或没有天气结果，就自然地说明原因并提示用户补充。"
        "回答要亲切、简洁、像聊天，不要暴露 JSON、接口名、工具名或调用过程。"
    ),
)
