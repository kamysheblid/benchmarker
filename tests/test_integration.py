"""End-to-end integration tests (Phase 11), using respx to mock llama-server."""

import asyncio
import json
from pathlib import Path

import pytest
import respx
from httpx import Response

from benchmarker.client import LLMClient
from benchmarker.config import (
    OptimizerConfig,
    ParameterSpec,
    ParameterType,
    TestCase,
    TestSuite,
)
from benchmarker.importer import import_scores
from benchmarker.optimizers import GridOptimizer
from benchmarker.runner import Runner, config_key


def _stream_body(text_parts: list[str], usage: dict) -> bytes:
    chunks = [{"choices": [{"delta": {"content": p}}], "usage": None} for p in text_parts]
    chunks.append({"choices": [{"delta": {}}], "usage": usage})
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode("utf-8")


BASE_URL = "http://localhost:8080/v1/chat/completions"


def _tiny_suite() -> TestSuite:
    return TestSuite(
        tests=[
            TestCase(id="t1", prompt="Hello", repeat=1),
            TestCase(id="t2", prompt="Count", repeat=1),
        ]
    )


def _grid() -> GridOptimizer:
    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=0.2, step=0.1)]
    return GridOptimizer(specs)  # 2 trials


@respx.mock
async def test_end_to_end_run_and_eval(tmp_path: Path) -> None:
    respx.post(BASE_URL).mock(
        return_value=Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_stream_body(
                ["Hi there", "One two three"],
                {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
            ),
        )
    )
    run_dir = tmp_path / "run"
    client = LLMClient(base_url=BASE_URL)
    runner = Runner(client, _tiny_suite(), _grid(), "m", run_dir)
    results = await runner.run()

    # 2 trials * 2 tests = 4 results
    assert len(results) == 4
    assert (run_dir / "raw_data.json").exists()
    assert (run_dir / "eval_output.md").exists()

    raw = json.loads((run_dir / "raw_data.json").read_text())
    assert len(raw) == 4
    assert all(r["response_text"] for r in raw)

    md = (run_dir / "eval_output.md").read_text()
    assert "Config:" in md
    assert "Hello" in md
    assert "Count" in md
    assert "RATING" in md.upper()


@respx.mock
async def test_end_to_end_import_flow(tmp_path: Path) -> None:
    respx.post(BASE_URL).mock(
        return_value=Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_stream_body(
                ["slow good", "fast ok"],
                {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            ),
        )
    )
    run_dir = tmp_path / "run"
    client = LLMClient(base_url=BASE_URL)
    await Runner(client, _tiny_suite(), _grid(), "m", run_dir).run()

    results = json.loads((run_dir / "raw_data.json").read_text())
    configs: list[dict] = []
    for r in results:
        if r["config"] not in configs:
            configs.append(r["config"])
    scores = [
        {"config": config_key(configs[0]), "test_id": "t1", "repetition": 1, "scores": {"overall": 9.0}},
        {"config": config_key(configs[0]), "test_id": "t2", "repetition": 1, "scores": {"overall": 8.0}},
        {"config": config_key(configs[1]), "test_id": "t1", "repetition": 1, "scores": {"overall": 4.0}},
        {"config": config_key(configs[1]), "test_id": "t2", "repetition": 1, "scores": {"overall": 3.0}},
    ]
    scores_path = run_dir / "scores.json"
    scores_path.write_text(json.dumps(scores))

    ranking = import_scores(run_dir, scores_path, weight_quality=0.5)
    assert len(ranking) == 2
    # configs[0] has higher quality -> should rank first at w=0.5
    assert ranking[0].config == config_key(configs[0])
    assert ranking[0].norm_quality == pytest.approx(1.0)


@respx.mock
async def test_end_to_end_server_error(tmp_path: Path) -> None:
    respx.post(BASE_URL).mock(return_value=Response(500, content="internal error"))
    run_dir = tmp_path / "run"
    client = LLMClient(base_url=BASE_URL)
    results = await Runner(client, _tiny_suite(), _grid(), "m", run_dir).run()
    # all attempts failed after retries -> recorded as failures
    assert len(results) == 4
    assert all(r.error for r in results)
    # raw data still written
    assert (run_dir / "raw_data.json").exists()
