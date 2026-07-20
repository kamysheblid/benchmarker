"""Benchmark runner: orchestrates client, tests and optimizer (Phase 6)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from benchmarker.client import LLMClientError
from benchmarker.config import TestSuite
from benchmarker.eval_file import EVAL_FILE_NAME, generate_eval_md
from benchmarker.optimizers import BaseOptimizer

RAW_DATA_FILE = "raw_data.json"


class RunResult(BaseModel):
    """A single completed (or failed) completion measurement."""

    model_config = ConfigDict(extra="forbid")

    config: dict[str, Any]
    test_id: str
    repetition: int
    prompt: str
    response_text: str
    ttft: float
    total_time: float
    tokens_per_sec: float
    completion_tokens: int
    prompt_tokens: int
    error: str | None = None


def config_key(config: dict[str, Any]) -> str:
    """Stable, order-independent key for grouping results by config."""
    return json.dumps(config, sort_keys=True, default=str)


class ProgressReporter:
    """No-op progress reporter; safe default when no UI is wanted."""

    def start(self, total: int) -> None:
        pass

    def advance(self) -> None:
        pass

    def finish(self) -> None:
        pass


class Runner:
    """Drives the benchmark: for each optimizer suggestion, run the suite."""

    def __init__(
        self,
        client: Any,
        test_suite: TestSuite,
        optimizer: BaseOptimizer,
        model_name: str,
        run_dir: Path,
        max_retries: int = 1,
        progress: ProgressReporter | None = None,
        static_params: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.test_suite = test_suite
        self.optimizer = optimizer
        self.model_name = model_name
        self.run_dir = Path(run_dir)
        self.max_retries = max_retries
        self.progress = progress or ProgressReporter()
        # Fixed params merged into every request (e.g. enable_thinking:false).
        self.static_params = dict(static_params or {})

    async def run(self) -> list[RunResult]:
        """Execute the full benchmark, returning and persisting all results."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        results: list[RunResult] = []
        per_trial = sum(t.repeat for t in self.test_suite.tests) or 1
        total_steps = self.optimizer.estimated_steps() * per_trial
        self.progress.start(total_steps)

        trial_index = 0
        while True:
            try:
                config = self.optimizer.suggest()
            except StopIteration:
                break

            config_results: list[RunResult] = []
            for test in self.test_suite.tests:
                for rep in range(1, test.repeat + 1):
                    result = await self._run_one(config, test, rep)
                    results.append(result)
                    config_results.append(result)
                    self.progress.advance()

            avg_speed = self._avg_speed(config_results)
            self.optimizer.tell({"tokens_per_sec": avg_speed})
            trial_index += 1
            self._save(results)
        self.progress.finish()

        self._save(results)
        eval_path = self.run_dir / EVAL_FILE_NAME
        generate_eval_md(self.run_dir, results, out_path=eval_path)
        return results

    async def _run_one(self, config: dict[str, Any], test: Any, rep: int) -> RunResult:
        messages = []
        if test.system:
            messages.append({"role": "system", "content": test.system})
        messages.append({"role": "user", "content": test.prompt})

        params: dict[str, Any] = {**self.static_params, **config}
        if test.max_tokens is not None:
            params["max_tokens"] = test.max_tokens

        last_error: str | None = None
        for _ in range(self.max_retries + 1):
            try:
                completion = await self.client.complete(
                    messages=messages, model=self.model_name, **params
                )
                return RunResult(
                    config=config,
                    test_id=test.id,
                    repetition=rep,
                    prompt=test.prompt,
                    response_text=completion.response_text,
                    ttft=completion.ttft,
                    total_time=completion.total_time,
                    tokens_per_sec=completion.tokens_per_sec,
                    completion_tokens=completion.completion_tokens,
                    prompt_tokens=completion.prompt_tokens,
                    error=None,
                )
            except LLMClientError as exc:
                last_error = str(exc)
        # all retries exhausted -> record failure
        return RunResult(
            config=config,
            test_id=test.id,
            repetition=rep,
            prompt=test.prompt,
            response_text="",
            ttft=0.0,
            total_time=0.0,
            tokens_per_sec=0.0,
            completion_tokens=0,
            prompt_tokens=0,
            error=last_error,
        )

    @staticmethod
    def _avg_speed(results: list[RunResult]) -> float:
        ok = [r.tokens_per_sec for r in results if r.error is None]
        if not ok:
            return 0.0
        return sum(ok) / len(ok)

    def _save(self, results: list[RunResult]) -> None:
        path = self.run_dir / RAW_DATA_FILE
        path.write_text(
            json.dumps([r.model_dump() for r in results], indent=2, default=str),
            encoding="utf-8",
        )
