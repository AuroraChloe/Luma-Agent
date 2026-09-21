from .get_weather import WEATHER_CURRENT_TOOL
from .image_edit import IMAGE_EDIT_TOOL
from .image_generate import IMAGE_GENERATE_TOOL
from .web_search import WEB_SEARCH_TOOL

AGENT_TOOLS = [WEATHER_CURRENT_TOOL, IMAGE_GENERATE_TOOL, IMAGE_EDIT_TOOL,WEB_SEARCH_TOOL]
AGENT_TOOL_MAP = {tool.name: tool for tool in AGENT_TOOLS}

__all__ = ["AGENT_TOOLS", "AGENT_TOOL_MAP"]
