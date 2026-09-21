from openai import OpenAI
import os

from provider import DEFAULT_LLM_BASE_URL, DEFAULT_REQUEST_TIMEOUT, llm_api_key


OPENAI_REQUEST_TIMEOUT = DEFAULT_REQUEST_TIMEOUT


# - 华北2（北京）: https://dashscope.aliyuncs.com/compatible-mode/v1
# - 美国（弗吉尼亚）: https://dashscope-us.aliyuncs.com/compatible-mode/v1
# - 新加坡: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
def llm_chat(model, messages, **kwargs):
    provider_api_key = kwargs.pop("_provider_api_key", None)
    provider_base_url = kwargs.pop("_provider_base_url", None)
    provider_timeout = kwargs.pop("_provider_timeout", None)
    provider_max_retries = kwargs.pop("_provider_max_retries", None)
    client_kwargs = {}
    if provider_max_retries is not None:
        client_kwargs["max_retries"] = int(provider_max_retries)
    client = OpenAI(
        api_key=provider_api_key or llm_api_key(),
        base_url=provider_base_url or os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
        timeout=provider_timeout or OPENAI_REQUEST_TIMEOUT,
        **client_kwargs,
    )
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return completion
