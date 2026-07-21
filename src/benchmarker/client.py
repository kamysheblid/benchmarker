"""Async LLM client with streaming metrics (Phase 4).

Wraps an OpenAI-compatible ``/chat/completions`` endpoint (e.g. llama-server)
and records time-to-first-token (TTFT), total time, token counts and the full
response text while consuming a streamed (SSE) response.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

DEFAULT_BASE_URL = "http://localhost:8080/v1/chat/completions"


class LLMClientError(Exception):
    """Raised when the LLM endpoint returns an error or times out."""


class CompletionResult(BaseModel):
    """Metrics and text captured from a single completion request."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    response_text: str = ""
    ttft: float = 0.0
    total_time: float = 0.0
    tokens_per_sec: float = 0.0


class LLMClient:
    """Streaming client for an OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json", "accept": "text/event-stream"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        return headers

    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        **params: Any,
    ) -> CompletionResult:
        """Send a chat completion request and stream the response.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model name to query.
            **params: Sampling parameters (temperature, top_k, stop, ...) forwarded to
                the endpoint.

        Returns:
            A :class:`CompletionResult` with metrics and the full text.

        Raises:
            LLMClientError: on HTTP error, timeout, or streaming failure.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **params,
        }
        start = time.monotonic()
        ttft: float = 0.0
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    self.base_url,
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    if response.status_code >= 400:
                        body = await response.aread()
                        raise LLMClientError(
                            f"LLM endpoint returned {response.status_code}: "
                            f"{body.decode('utf-8', 'replace')[:200]}"
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:") :].strip()
                        if data == "[DONE]":
                            continue
                        try:
                            chunk = _loads(data)
                        except ValueError:
                            continue
                        if not isinstance(chunk, dict):
                            continue
                        usage = chunk.get("usage")
                        if isinstance(usage, dict):
                            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                        for choice in chunk.get("choices", []) or []:
                            delta = choice.get("delta") or {}
                            content = delta.get("content")
                            if content:
                                if ttft == 0.0:
                                    ttft = time.monotonic() - start
                                text_parts.append(content)
                            # Reasoning models (e.g. Qwen3) stream the thinking
                            # trace in `reasoning_content`. Capture it so the
                            # response is never blank when the answer `content`
                            # is empty or consumed by a small max_tokens budget.
                            reasoning = delta.get("reasoning_content")
                            if reasoning:
                                if ttft == 0.0:
                                    ttft = time.monotonic() - start
                                reasoning_parts.append(reasoning)
        except httpx.HTTPError as exc:  # timeout, connect error, etc.
            raise LLMClientError(f"Request to LLM endpoint failed: {exc}") from exc

        total_time = time.monotonic() - start
        if ttft == 0.0:  # no tokens streamed
            ttft = total_time
        tokens_per_sec = (completion_tokens / total_time) if total_time > 0 else 0.0

        response_text = "".join(text_parts)
        if not response_text and reasoning_parts:
            # Fall back to the reasoning trace so the record is not empty.
            response_text = "".join(reasoning_parts)

        return CompletionResult(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            response_text=response_text,
            ttft=ttft,
            total_time=total_time,
            tokens_per_sec=tokens_per_sec,
        )


def _loads(text: str) -> Any:
    import json

    return json.loads(text)
