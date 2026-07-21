"""Tests for the benchmark runner (Phase 6)."""

import json
from pathlib import Path
from typing import Any

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
    results, _ = await runner.run()

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
    results, _ = await runner.run()
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
    results, _ = await runner.run()
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
    results, _ = await runner.run()
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


async def test_runner_passes_stop_sequence(tmp_path: Path) -> None:
    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=0.1)]
    optimizer = GridOptimizer(specs)
    client = _FakeClient()
    runner = Runner(
        client,
        TestSuite(tests=[TestCase(id="t1", prompt="x", stop=["\n```", "\ndef "])]),
        optimizer,
        "m",
        tmp_path / "run",
    )
    await runner.run()
    assert len(client.calls) == 1
    _messages, _model, params = client.calls[0]
    assert params["stop"] == ["\n```", "\ndef "]


class _MockHistoryOptimizer:
    """Minimal optimizer that records tell calls and supports from_history."""

    def __init__(self, parameters: list[ParameterSpec], budget: int = 2, seed: int | None = None) -> None:
        self.parameters = parameters
        self.budget = budget
        self.seed = seed
        self._count = 0
        self.tell_calls: list[dict[str, Any]] = []

    def suggest(self) -> dict[str, Any]:
        if self._count >= self.budget:
            raise StopIteration
        self._count += 1
        return {"temperature": 0.5}

    def tell(self, metrics: dict[str, Any]) -> None:
        self.tell_calls.append(metrics)

    def estimated_steps(self) -> int:
        return self.budget

    @classmethod
    def from_history(
        cls,
        history_path: Path,
        parameters: list[ParameterSpec],
        budget: int = 2,
        seed: int | None = None,
    ):
        from benchmarker.optimizer_history import OptimizerHistory
        history = OptimizerHistory.load(history_path)
        optimizer = cls(parameters=parameters, budget=budget, seed=seed)
        for trial in history.trials:
            optimizer.tell({"tokens_per_sec": trial.tokens_per_sec, "quality": trial.quality})
        return optimizer


async def test_runner_saves_optimizer_history(tmp_path: Path) -> None:
    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=0.1)]
    optimizer = _MockHistoryOptimizer(specs, budget=1)
    client = _FakeClient()
    history_file = tmp_path / "optimizer_history.json"
    runner = Runner(
        client,
        TestSuite(tests=[TestCase(id="t1", prompt="x")]),
        optimizer,
        "m",
        tmp_path / "run",
        history_path=history_file,
    )
    await runner.run()

    assert history_file.exists()
    data = json.loads(history_file.read_text())
    assert len(data) == 1
    assert data[0]["params"] == {"temperature": 0.5}
    assert data[0]["tokens_per_sec"] == 10.0
    assert data[0]["quality"] is None


async def test_runner_replays_history_on_start(tmp_path: Path) -> None:
    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=0.1)]
    # Seed history with one trial
    history_file = tmp_path / "optimizer_history.json"
    history_file.write_text(
        json.dumps([{"params": {"temperature": 0.5}, "quality": None, "tokens_per_sec": 10.0}]),
        encoding="utf-8",
    )
    optimizer = _MockHistoryOptimizer(specs, budget=1)
    client = _FakeClient()
    runner = Runner(
        client,
        TestSuite(tests=[TestCase(id="t1", prompt="x")]),
        optimizer,
        "m",
        tmp_path / "run",
        history_path=history_file,
    )
    await runner.run()

    # After replay, runner.optimizer is a new instance produced by from_history.
    # The replayed historical trial should be recorded, plus the new trial.
    assert len(runner.optimizer.tell_calls) == 2
    assert runner.optimizer.tell_calls[0] == {"tokens_per_sec": 10.0, "quality": None}
    assert runner.optimizer.tell_calls[1]["tokens_per_sec"] == 10.0
