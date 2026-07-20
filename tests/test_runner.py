"""Tests for the benchmark runner (Phase 6)."""

from pathlib import Path

import pytest

from benchmarker.client import CompletionResult, LLMClientError
from benchmarker.config import OptimizerConfig, ParameterSpec, ParameterType, TestCase, TestSuite
from benchmarker.optimizers import BayesianOptimizer, GridOptimizer, create_optimizer
from benchmarker.runner import RunResult, Runner


def _tiny_suite() -> TestSuite:
    return TestSuite(
        tests=[
            TestCase(id="t1", prompt="Hello"),
            TestCase(id="t2", prompt="World", repeat=2),
        ]
    )


def _specs() -> list[ParameterSpec]:
    return [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=0.2)]


class _FakeClient:
    """Records calls, returns a canned CompletionResult."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def complete(self, messages, model, **params):
        self.calls.append((messages, model, params))
        return CompletionResult(
            prompt_tokens=3,
            completion_tokens=5,
            response_text="resp",
            ttft=0.05,
            total_time=0.5,
            tokens_per_sec=10.0,
        )


async def test_runner_call_count_and_saving(tmp_path: Path) -> None:
    # grid of 2 temperature values, 2 tests (t2 repeated x2) => 2*3 = 6 calls
    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=0.2, step=0.1)]
    optimizer = GridOptimizer(specs)
    client = _FakeClient()
    runner = Runner(client, _tiny_suite(), optimizer, "m", tmp_path / "run")
    results = await runner.run()

    assert len(client.calls) == 6  # 2 trials * (1 + 2 reps)
    assert len(results) == 6
    raw = (tmp_path / "run" / "raw_data.json").read_text()
    assert "temperature" in raw
    assert all(isinstance(r, RunResult) for r in results)


async def test_runner_feeds_speed_to_bayesian(tmp_path: Path) -> None:
    optimizer = BayesianOptimizer(_specs(), budget=2)
    client = _FakeClient()
    runner = Runner(
        client, TestSuite(tests=[TestCase(id="t1", prompt="x")]), optimizer, "m", tmp_path / "run"
    )
    results = await runner.run()
    assert len(results) == 2
    told = [t for t in optimizer.study.trials if t.state.is_finished()]
    assert len(told) == 2


async def test_runner_records_failure_and_continues(tmp_path: Path) -> None:
    class _FlakyClient:
        def __init__(self) -> None:
            self.attempt = 0

        async def complete(self, messages, model, **params):
            self.attempt += 1
            if self.attempt == 1:
                raise LLMClientError("boom")
            return CompletionResult(
                prompt_tokens=1, completion_tokens=1, response_text="ok",
                ttft=0.01, total_time=0.1, tokens_per_sec=10.0,
            )

    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=0.1)]
    optimizer = GridOptimizer(specs)
    client = _FlakyClient()
    runner = Runner(
        client, TestSuite(tests=[TestCase(id="t1", prompt="x")]), optimizer, "m", tmp_path / "run"
    )
    results = await runner.run()
    assert len(results) == 1
    assert results[0].response_text == "ok"


async def test_runner_persistent_failure_records_error(tmp_path: Path) -> None:
    class _AlwaysFails:
        async def complete(self, messages, model, **params):
            raise LLMClientError("always")

    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=0.1)]
    optimizer = GridOptimizer(specs)
    client = _AlwaysFails()
    runner = Runner(
        client, TestSuite(tests=[TestCase(id="t1", prompt="x")]), optimizer, "m", tmp_path / "run"
    )
    results = await runner.run()
    assert len(results) == 1
    assert results[0].response_text == ""  # failure recorded


class _RecordingReporter:
    def __init__(self) -> None:
        self.started = 0
        self.advances = 0
        self.finished = 0

    def start(self, total: int) -> None:
        self.started = total

    def advance(self) -> None:
        self.advances += 1

    def finish(self) -> None:
        self.finished += 1


async def test_runner_progress_reporter_called(tmp_path: Path) -> None:
    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=0.2, step=0.1)]
    optimizer = GridOptimizer(specs)  # 2 trials
    client = _FakeClient()
    reporter = _RecordingReporter()
    runner = Runner(
        client,
        TestSuite(tests=[TestCase(id="t1", prompt="x"), TestCase(id="t2", prompt="y", repeat=2)]),
        optimizer,
        "m",
        tmp_path / "run",
        progress=reporter,
    )
    await runner.run()
    # 2 trials * (1 + 2 reps) = 6 work units
    assert reporter.started == 6
    assert reporter.advances == 6
    assert reporter.finished == 1


async def test_runner_merges_static_params(tmp_path: Path) -> None:
    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=0.2, step=0.1)]
    optimizer = GridOptimizer(specs)  # 2 trials
    client = _FakeClient()
    runner = Runner(
        client,
        TestSuite(tests=[TestCase(id="t1", prompt="x")]),
        optimizer,
        "m",
        tmp_path / "run",
        static_params={"chat_template_kwargs": {"enable_thinking": False}},
    )
    await runner.run()
    # static params must be present in every call, alongside sampled params.
    assert len(client.calls) == 2
    for _messages, _model, params in client.calls:
        assert params["chat_template_kwargs"] == {"enable_thinking": False}
        assert "temperature" in params
