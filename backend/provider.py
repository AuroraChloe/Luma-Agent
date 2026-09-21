import os
import time
from pathlib import Path

from openai import OpenAI


def load_dotenv(path=".env"):
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


load_dotenv()


DEFAULT_PROVIDER_NAME = os.getenv("LLM_PROVIDER", "openai_compatible")
DEFAULT_LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
DEFAULT_REQUEST_TIMEOUT = float(os.getenv("OPENAI_REQUEST_TIMEOUT", "75"))
MODELS_CACHE_TTL_SECONDS = int(os.getenv("MODELS_CACHE_TTL_SECONDS", "300"))
_MODELS_CACHE = {"expires_at": 0, "data": None}


def llm_api_key():
    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("Missing LLM_API_KEY in environment")
    return key


def env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ProviderCapabilities:
    def __init__(self):
        self.tools = env_bool("LLM_SUPPORTS_TOOLS", True)
        self.vision = env_bool("LLM_SUPPORTS_VISION", True)
        self.images = env_bool("LLM_SUPPORTS_IMAGES", False)
        self.upstream_models = env_bool("LLM_MODELS_FROM_UPSTREAM", True)

    def as_dict(self):
        return {
            "tools": self.tools,
            "vision": self.vision,
            "images": self.images,
            "upstream_models": self.upstream_models,
        }


class OpenAICompatibleProvider:
    def __init__(self, api_key=None, base_url=None, timeout=None):
        self.name = DEFAULT_PROVIDER_NAME
        self.api_key = api_key or llm_api_key()
        self.base_url = (base_url or os.getenv("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL).rstrip("/")
        if not self.base_url:
            raise RuntimeError("Missing LLM_BASE_URL in environment")
        self.timeout = timeout or DEFAULT_REQUEST_TIMEOUT
        self.capabilities = ProviderCapabilities()

    def client(self):
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def chat(self, model, messages, **kwargs):
        return self.client().chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,
        )

    def list_models(self):
        global _MODELS_CACHE
        now = time.time()
        if _MODELS_CACHE["data"] is not None and _MODELS_CACHE["expires_at"] > now:
            return _MODELS_CACHE["data"]
        response = self.client().models.list()
        if hasattr(response, "model_dump"):
            payload = response.model_dump(mode="json", exclude_none=True)
        elif isinstance(response, dict):
            payload = response
        else:
            payload = {"object": "list", "data": [item.model_dump(mode="json", exclude_none=True) for item in response.data]}
        items = payload.get("data", []) if isinstance(payload, dict) else []
        normalized = [
            {
                "id": item.get("id"),
                "object": item.get("object", "model"),
                **{k: v for k, v in item.items() if k not in {"id", "object"}},
            }
            for item in items
            if isinstance(item, dict) and item.get("id")
        ]
        _MODELS_CACHE = {"expires_at": now + MODELS_CACHE_TTL_SECONDS, "data": normalized}
        return normalized


def default_provider():
    return OpenAICompatibleProvider()


def provider_capabilities():
    return ProviderCapabilities()
