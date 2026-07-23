"""Tests for the async LLM streaming client (Phase 4)."""

import asyncio
from pathlib import Path

import httpx
import pytest
import respx
from httpx import Response

from benchmarker.client import (
    ClientError,
    CompletionResult,
    LLMClient,
    LLMClientError,
    ServerError,
    TransientError,
)

BASE_URL = "http://localhost:8080/v1/chat/completions"


def _stream_body(text_parts: list[str], usage: dict) -> bytes:
    import json

    chunks = []
    for part in text_parts:
        chunks.append({"choices": [{"delta": {"content": part}}], "usage": None})
    chunks.append({"choices": [{"delta": {}}], "usage": usage})
    body = ""
    for c in chunks:
        body += f"data: {json.dumps(c)}\n\n"
    body += "data: [DONE]\n\n"
    return body.encode("utf-8")


@respx.mock
async def test_complete_basic_streaming() -> None:
    route = respx.post(BASE_URL).mock(
        return_value=Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_stream_body(
                ["Hello", " world"],
                {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            ),
        )
    )
    client = LLMClient(base_url=BASE_URL)
    result = await client.complete(
        messages=[{"role": "user", "content": "hi"}], model="m", temperature=0.7
    )
    assert isinstance(result, CompletionResult)
    assert result.response_text == "Hello world"
    assert result.prompt_tokens == 5
    assert result.completion_tokens == 2
    assert result.ttft > 0
    assert result.total_time > 0
    assert result.tokens_per_sec == pytest.approx(2 / result.total_time, rel=1e-6)
    assert route.called


@respx.mock
async def test_complete_ttft_smaller_than_total() -> None:
    route = respx.post(BASE_URL).mock(
        return_value=Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_stream_body(
                ["a", "b", "c"],
                {"prompt_tokens": 1, "completion_tokens": 3, "total_tokens": 4},
            ),
        )
    )
    client = LLMClient(base_url=BASE_URL)
    result = await client.complete(messages=[{"role": "user", "content": "x"}], model="m")
    assert result.ttft <= result.total_time
    assert result.completion_tokens == 3


@respx.mock
async def test_complete_http_500_raises() -> None:
    respx.post(BASE_URL).mock(return_value=Response(500, content="boom"))
    client = LLMClient(base_url=BASE_URL)
    with pytest.raises(LLMClientError):
        await client.complete(messages=[{"role": "user", "content": "x"}], model="m")


@respx.mock
async def test_complete_timeout_raises() -> None:
    import httpx as _httpx

    def _slow(request, **kwargs):
        raise _httpx.ConnectTimeout("timed out")

    respx.post(BASE_URL).mock(side_effect=_slow)
    client = LLMClient(base_url=BASE_URL, timeout=0.01)
    with pytest.raises(LLMClientError):
        await client.complete(messages=[{"role": "user", "content": "x"}], model="m")


@respx.mock
async def test_complete_passes_params_and_auth() -> None:
    route = respx.post(BASE_URL).mock(
        return_value=Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_stream_body(
                ["ok"], {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            ),
        )
    )
    client = LLMClient(base_url=BASE_URL, api_key="secret")
    await client.complete(
        messages=[{"role": "user", "content": "x"}], model="m", temperature=0.9, top_k=40
    )
    assert route.called
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer secret"
    import json

    payload = json.loads(request.content)
    assert payload["model"] == "m"
    assert payload["stream"] is True
    assert payload["temperature"] == 0.9
    assert payload["top_k"] == 40
    # ensure kwargs do not include reserved keys
    assert "messages" in payload and "model" in payload


@respx.mock
async def test_complete_http_400_raises_client_error() -> None:
    respx.post(BASE_URL).mock(return_value=Response(400, content="bad request"))
    client = LLMClient(base_url=BASE_URL)
    with pytest.raises(ClientError):
        await client.complete(messages=[{"role": "user", "content": "x"}], model="m")


@respx.mock
async def test_complete_http_500_raises_server_error() -> None:
    respx.post(BASE_URL).mock(return_value=Response(500, content="server error"))
    client = LLMClient(base_url=BASE_URL)
    with pytest.raises(ServerError):
        await client.complete(messages=[{"role": "user", "content": "x"}], model="m")


@respx.mock
async def test_complete_timeout_raises_transient_error() -> None:
    import httpx as _httpx

    def _slow(request, **kwargs):
        raise _httpx.ConnectTimeout("timed out")

    respx.post(BASE_URL).mock(side_effect=_slow)
    client = LLMClient(base_url=BASE_URL, timeout=0.01)
    with pytest.raises(TransientError):
        await client.complete(messages=[{"role": "user", "content": "x"}], model="m")
