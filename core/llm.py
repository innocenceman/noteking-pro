"""LLM engine: unified interface supporting OpenAI-compatible providers."""

from __future__ import annotations

from .config import AppConfig


def _build_client(config: AppConfig):
    from openai import OpenAI
    import httpx

    base_url = config.llm.base_url or None
    api_key = config.llm.api_key

    if not api_key:
        raise ValueError(
            "No LLM API key configured. Set it via config file, "
            "environment variable NOTEKING_LLM_API_KEY, or --api-key flag."
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=httpx.Timeout(300.0, connect=60.0),
        max_retries=3,
    )


def chat(
    prompt: str,
    config: AppConfig,
    system: str = "",
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send a chat completion request and return the text response."""
    client = _build_client(config)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=config.llm.model,
        messages=messages,
        temperature=temperature if temperature is not None else config.llm.temperature,
        max_tokens=max_tokens or config.llm.max_tokens,
    )
    return resp.choices[0].message.content or ""


def chat_stream(
    prompt: str,
    config: AppConfig,
    system: str = "",
):
    """Stream a chat completion, yielding text chunks."""
    client = _build_client(config)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model=config.llm.model,
        messages=messages,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content
