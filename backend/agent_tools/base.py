from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class AgentTool:
    name: str
    description: str
    parameters: dict
    handler: object
    argument_prompt: str
    answer_prompt: str
    fallback_formatter: Callable[[object, str], str]
    required: tuple[str, ...] = ()
    missing_args_message: str = "缺少工具所需参数。"
    validate_args: Callable[[dict, str, str], dict] | None = None

    def schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def normalize_args(self, args, user_message="", context=""):
        if not isinstance(args, dict):
            args = {}
        normalized = {key: value for key, value in args.items()}
        if self.validate_args:
            normalized = self.validate_args(normalized, user_message or "", context or "")
        return normalized

    def has_required_args(self, args):
        return all(str(args.get(key) or "").strip() for key in self.required)

    def validate(self, args, user_message="", context=""):
        return ToolValidator.validate(self, args, user_message=user_message, context=context)

    def invoke(self, args):
        if hasattr(self.handler, "invoke"):
            return self.handler.invoke(args)
        return self.handler(**args)


class ToolValidationResult:
    def __init__(self, arguments, valid, missing=()):
        self.arguments = arguments
        self.valid = valid
        self.missing = tuple(missing or ())


class ToolValidator:
    """Code-only validation and normalization after RequestInterpreter."""

    @staticmethod
    def validate(tool, args, user_message="", context=""):
        normalized = tool.normalize_args(args, user_message=user_message, context=context)
        if not isinstance(normalized, dict):
            normalized = {}
        missing = [key for key in tool.required if not str(normalized.get(key) or "").strip()]
        return ToolValidationResult(normalized, not missing, missing=missing)
