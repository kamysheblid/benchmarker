"""Benchmark runner: orchestrates client, tests and optimizer (Phase 6)."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from benchmarker.circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

from benchmarker.client import ClientError, LLMClientError, ServerError, TransientError
from benchmarker.config import TestSuite
from benchmarker.eval_file import JUDGE_PROMPT_FILE, generate_judge_prompt
from benchmarker.optimizer_history import OptimizerHistory, OptimizerTrial
from benchmarker.optimizers import AdaptiveOptimizer, BaseOptimizer, TwoPhaseOptimizer
from benchmarker.param_validation import validate_sampling_params

RAW_DATA_FILE = "raw_data.json"
AUTO_EVAL_FILE = "scores_auto.json"
OPTIMIZER_HISTORY_FILE = "optimizer_history.json"
CHECKPOINT_FILE = "checkpoint.json"
ERROR_REPORT_FILE = "error_report.json"


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
    category: str | None = None
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    config_aborted: bool = False
    success_rate: float | None = None
    coverage: float | None = None

    @property
    def cost_estimate(self) -> float:
        """Estimated USD cost for this single request."""
        input_cost = (self.prompt_tokens / 1_000_000) * self.cost_per_1m_input
        output_cost = (self.completion_tokens / 1_000_000) * self.cost_per_1m_output
        return round(input_cost + output_cost, 8)


def config_key(config: dict[str, Any]) -> str:
    """Stable, order-independent key for grouping results by config."""
    return json.dumps(config, sort_keys=True, default=str)


def _best_config_from_history(history: list[OptimizerTrial]) -> dict[str, Any]:
    """Return the best config by tokens_per_sec from trial history."""
    if not history:
        return {}
    best = max(history, key=lambda t: t.tokens_per_sec if t.tokens_per_sec is not None else 0.0)
    return dict(best.params)


def _build_refinement_hint(
    config: dict[str, Any], parameters: list[ParameterSpec], step: float = 1.0
) -> dict[str, list[float]]:
    """Build refinement hint by expanding each numeric param by ±step, clamped to original bounds."""
    hint: dict[str, list[float]] = {}
    specs = {spec.name: spec for spec in parameters}
    for name, value in config.items():
        if not isinstance(value, (int, float)):
            continue
        spec = specs.get(name)
        lo = float(value) - step
        hi = float(value) + step
        if spec is not None and spec.low is not None:
            lo = max(lo, float(spec.low))
        if spec is not None and spec.high is not None:
            hi = min(hi, float(spec.high))
        hint[name] = [lo, hi]
    return hint


def _build_refinement_hint_from_passing(
    results: list[RunResult], auto_scores: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Build refinement hints from configs that pass auto-eval quality thresholds."""
    # Group auto-eval scores by config key
    config_scores: dict[str, list[dict[str, float]]] = {}
    for r in results:
        ck = config_key(r.config)
        merge_key = f"{ck}::{r.test_id}::{r.repetition}"
        if merge_key in auto_scores:
            config_scores.setdefault(ck, []).append(auto_scores[merge_key])

    # Determine which configs are auto_rejected
    rejected_configs: set[str] = set()
    for ck, scores in config_scores.items():
        for s in scores:
            if s.get("exec_safety", 1.0) < 0.5 or (
                s.get("unit_pass_rate", 1.0) == 0 and s.get("static_quality", 1.0) < 0.3
            ):
                rejected_configs.add(ck)
                break

    passing_configs = [r.config for r in results if config_key(r.config) not in rejected_configs]
    if not passing_configs:
        return {}

    param_names = set(passing_configs[0].keys())

    hints: dict[str, dict[str, float]] = {}
    for name in param_names:
        values = [float(c[name]) for c in passing_configs if name in c and isinstance(c[name], (int, float))]
        if values:
            hints[name] = {"low": min(values), "high": max(values)}
    return hints


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
        max_retries: int = 2,
        progress: ProgressReporter | None = None,
        static_params: dict[str, Any] | None = None,
        auto_eval: bool = False,
        cost_per_1m_input: float = 0.0,
        cost_per_1m_output: float = 0.0,
        history_path: Path | None = None,
        params_path: Path | None = None,
        resume: bool = False,
        force: bool = False,
        circuit_breaker: CircuitBreaker | None = None,
        enable_health_check: bool = True,
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
        self.auto_eval = auto_eval
        self.cost_per_1m_input = cost_per_1m_input
        self.cost_per_1m_output = cost_per_1m_output
        self.history_path = history_path
        self.params_path = params_path
        self.resume = resume
        self.force = force
        self._history: list[OptimizerTrial] = []
        self._config_failures: dict[str, int] = {}
        self._config_attempts: dict[str, int] = {}
        self._circuit_breaker = circuit_breaker or CircuitBreaker()
        self._error_list: list[BaseException] = []
        self._completed_configs: set[str] = set()
        self._enable_health_check = enable_health_check

    async def run(self) -> tuple[list[RunResult], dict[str, Any] | None]:
        """Execute the full benchmark, returning and persisting all results.

        Returns:
            A tuple of (results, auto_eval_scores). ``auto_eval_scores`` is
            ``None`` when ``auto_eval`` is False.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Handle existing checkpoint
        existing_results = self._handle_checkpoint_on_start(self.resume, self.force)
        results: list[RunResult] = [RunResult.model_validate(r) for r in existing_results]
        self._completed_configs = {config_key(r.config) for r in results}
        trial_index = 0

        # Health check
        if self._enable_health_check and not await self._health_check():
            raise RuntimeError("Endpoint health check failed. Aborting.")

        per_trial = sum(t.repeat for t in self.test_suite.tests) or 1
        total_steps = self.optimizer.estimated_steps() * per_trial
        self.progress.start(total_steps)

        if self.history_path and self.history_path.exists():
            if hasattr(self.optimizer, "from_history"):
                self.optimizer = self.optimizer.from_history(
                    self.history_path,
                    self.optimizer.parameters,
                    getattr(self.optimizer, "budget", 0),
                    getattr(self.optimizer, "seed", None),
                )

        try:
            while True:
                try:
                    config = self.optimizer.suggest()
                except StopIteration:
                    break
                ckt = config_key(config)

                # Skip already-completed configs (e.g. from resumed checkpoint)
                if ckt in self._completed_configs:
                    logger.info("Skipping already-completed config %s", ckt)
                    trial_index += 1
                    continue

                config_results: list[RunResult] = []
                config_aborted = False
                for test in self.test_suite.tests:
                    if config_aborted:
                        break
                    for rep in range(1, test.repeat + 1):
                        try:
                            result = await self._run_one(config, test, rep)
                        except RuntimeError as exc:
                            if "Circuit breaker is open" in str(exc):
                                logger.error("Circuit breaker opened during run: %s", exc)
                                self._error_list.append(exc)
                                self._save_error_report(self._error_list)
                                raise
                            raise
                        results.append(result)
                        config_results.append(result)
                        self._config_attempts[ckt] = self._config_attempts.get(ckt, 0) + 1
                        if result.error is not None:
                            self._config_failures[ckt] = self._config_failures.get(ckt, 0) + 1
                            self._error_list.append(RuntimeError(result.error))
                        logger.info(
                            "repeat result: test=%s rep=%d status=%s error=%s",
                            test.id,
                            rep,
                            "success" if result.error is None else "failure",
                            result.error or "none",
                        )
                        if (
                            self._config_attempts[ckt] >= 2
                            and self._config_failures.get(ckt, 0) / self._config_attempts[ckt] > 0.8
                        ):
                            result.config_aborted = True
                            config_aborted = True
                            logger.info(
                                "circuit breaker tripped for config %s after %d attempts (%d failures)",
                                ckt,
                                self._config_attempts[ckt],
                                self._config_failures.get(ckt, 0),
                            )
                            break
                        self.progress.advance()

                avg_speed = self._avg_speed(config_results)
                successful_runs = sum(1 for r in config_results if r.error is None)
                total_runs = len(config_results)
                success_rate = successful_runs / total_runs if total_runs > 0 else 0.0
                successful_test_ids = {r.test_id for r in config_results if r.error is None}
                total_tests = len(self.test_suite.tests)
                coverage = len(successful_test_ids) / total_tests if total_tests > 0 else 0.0
                for r in config_results:
                    r.success_rate = success_rate
                    r.coverage = coverage
                penalized_speed = avg_speed * success_rate
                self.optimizer.tell({
                    "tokens_per_sec": penalized_speed,
                    "success_rate": success_rate,
                    "coverage": coverage,
                })
                self._history.append(
                    OptimizerTrial(
                        params=config,
                        tokens_per_sec=avg_speed,
                        metadata={"success_rate": success_rate, "coverage": coverage},
                    )
                )
                self._completed_configs.add(ckt)
                trial_index += 1
                self._save_checkpoint(results, trial_index)
                # Switch to phase2 when phase1 budget is exhausted
                if isinstance(self.optimizer, TwoPhaseOptimizer):
                    if (
                        trial_index >= self.optimizer._phase1_budget
                        and self.optimizer._active is self.optimizer._phase1
                    ):
                        if self.auto_eval:
                            from benchmarker.evaluator import evaluate_run
                            phase1_scores = evaluate_run(results)
                            hint = _build_refinement_hint_from_passing(results, phase1_scores)
                        else:
                            best_coarse = _best_config_from_history(self._history)
                            hint = _build_refinement_hint(best_coarse, self.optimizer.parameters, step=1.0)
                        phase2 = AdaptiveOptimizer(
                            parameters=self.optimizer.parameters,
                            resolution_factor=5,
                            refinement_hint=hint or None,
                        )
                        self.optimizer._phase2 = phase2
                        self.optimizer.switch_phase()
                self._save(results)
                self._save_history()
        finally:
            self.progress.finish()
            self._save(results)
            self._save_history()
            if self._error_list:
                self._save_error_report(self._error_list)

        judge_path = self.run_dir / JUDGE_PROMPT_FILE
        generate_judge_prompt(self.run_dir, results, out_path=judge_path)
        # config_map.json is written alongside the judge prompt by generate_judge_prompt

        # Auto-evaluation
        auto_scores: dict[str, Any] | None = None
        if self.auto_eval:
            from benchmarker.evaluator import evaluate_run

            auto_scores = evaluate_run(results)
            self._save_auto_eval(auto_scores)
            self._write_refinement_hints(results, auto_scores)

        return results, auto_scores

    async def _run_one(self, config: dict[str, Any], test: Any, rep: int) -> RunResult:
        messages = []
        if test.system:
            messages.append({"role": "system", "content": test.system})
        messages.append({"role": "user", "content": test.prompt})

        params: dict[str, Any] = {**self.static_params, **config}
        if test.max_tokens is not None:
            params["max_tokens"] = test.max_tokens
        if test.stop is not None:
            params["stop"] = test.stop

        validate_sampling_params(params)

        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = await self._circuit_breaker.call(
                    self.client.complete,
                    messages=messages,
                    model=self.model_name,
                    **params,
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
                    category=self.test_suite.categories.get(test.id),
                    cost_per_1m_input=self.cost_per_1m_input,
                    cost_per_1m_output=self.cost_per_1m_output,
                )
            except LLMClientError as exc:
                last_error = str(exc)
                if isinstance(exc, ClientError):
                    break  # fail fast
                if attempt < self.max_retries:
                    if isinstance(exc, ServerError):
                        await asyncio.sleep(2**attempt)  # exponential backoff
                    else:
                        await asyncio.sleep(attempt + 1)  # linear backoff
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
            category=self.test_suite.categories.get(test.id),
            cost_per_1m_input=self.cost_per_1m_input,
            cost_per_1m_output=self.cost_per_1m_output,
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

    def _save_auto_eval(self, scores: dict[str, dict[str, float]]) -> None:
        """Save auto-evaluation scores as a JSON scores file compatible with import-scores."""
        path = self.run_dir / AUTO_EVAL_FILE
        entries: list[dict[str, Any]] = []
        for merge_key, metrics in scores.items():
            # merge_key format: config::test_id::rep
            parts = merge_key.rsplit("::", 2)
            if len(parts) == 3:
                config_str, test_id, rep_str = parts
                entries.append({
                    "config": config_str,
                    "test_id": test_id,
                    "repetition": int(rep_str),
                    "scores": metrics,
                })
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def _write_refinement_hints(
        self, results: list[RunResult], auto_scores: dict[str, dict[str, float]]
    ) -> None:
        """Write refinement_hints to params.yaml from passing auto-eval configs."""
        if not self.params_path or not self.params_path.exists():
            return
        hints = _build_refinement_hint_from_passing(results, auto_scores)
        if not hints:
            return
        try:
            import yaml

            raw = yaml.safe_load(self.params_path.read_text(encoding="utf-8")) or {}
            raw["refinement_hints"] = hints
            self.params_path.write_text(
                yaml.safe_dump(raw, default_flow_style=False), encoding="utf-8"
            )
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("failed to write refinement hints: %s", exc)

    def _save_history(self, top_k: int = 20) -> None:
        """Persist optimizer trial history to JSON, keeping only the top-K trials."""
        if not self.history_path:
            return
        try:
            history = OptimizerHistory()
            for trial in self._history:
                history.add_trial(trial)
            history.to_json(self.history_path, top_k=top_k)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("failed to save optimizer history: %s", exc)

    def _generate_error_report(self, errors: list[BaseException]) -> dict[str, Any]:
        """Build a detailed error report from collected exceptions."""
        from collections import Counter

        error_types = Counter(e.__class__.__name__ for e in errors)
        configs_with_errors = []
        for e in errors:
            if hasattr(e, "context") and e.context:
                configs_with_errors.append({
                    "error": str(e),
                    "type": e.__class__.__name__,
                    "context": e.context,
                })
            else:
                configs_with_errors.append({"error": str(e), "type": e.__class__.__name__})

        return {
            "total_errors": len(errors),
            "error_types": dict(error_types),
            "first_error": str(errors[0]) if errors else None,
            "last_error": str(errors[-1]) if errors else None,
            "configs_with_errors": configs_with_errors,
            "recommendations": self._suggest_fixes(errors),
        }

    @staticmethod
    def _suggest_fixes(errors: list[BaseException]) -> list[str]:
        """Heuristic recommendations based on error patterns."""
        recs: list[str] = []
        transient_count = sum(1 for e in errors if isinstance(e, TransientError))
        server_count = sum(1 for e in errors if isinstance(e, ServerError))
        client_count = sum(1 for e in errors if isinstance(e, ClientError))

        if server_count > len(errors) * 0.5:
            recs.append("High server error rate (5xx). Check server health, logs, and resource limits.")
        if transient_count > len(errors) * 0.5:
            recs.append("Many transient timeouts. Check network stability and consider increasing client timeout.")
        if client_count > 0:
            recs.append("Client errors (4xx) detected. Verify request parameters and model name.")
        if any("Circuit breaker is open" in str(e) for e in errors):
            recs.append("Circuit breaker tripped. The endpoint may be down; wait before retrying.")

        if not recs:
            recs.append("No specific pattern detected. Review error_report.json for details.")

        return recs

    def _save_error_report(self, errors: list[BaseException]) -> None:
        """Persist error report to the run directory."""
        report = self._generate_error_report(errors)
        path = self.run_dir / ERROR_REPORT_FILE
        try:
            path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            logger.info("Error report saved to %s", path)
        except OSError as exc:
            logger.warning("failed to save error report: %s", exc)

    def _handle_checkpoint_on_start(self, resume: bool, force: bool) -> list[dict[str, Any]]:
        """Determine how to handle an existing checkpoint.

        Returns:
            List of existing results to prepend (empty when starting fresh).

        Raises:
            SystemExit: When non-interactive abort is required.
        """
        checkpoint_path = self.run_dir / CHECKPOINT_FILE
        if not checkpoint_path.exists():
            return []

        checkpoint = self._load_checkpoint(checkpoint_path)
        existing_results = checkpoint.get("results", [])

        if force:
            logger.info("--force given, ignoring checkpoint and starting fresh.")
            checkpoint_path.unlink(missing_ok=True)
            return []

        if resume:
            logger.info("--resume given, resuming from checkpoint with %d existing results.", len(existing_results))
            return existing_results

        # Interactive or non-interactive prompt
        last_index = checkpoint.get("trial_index", 0)
        timestamp = checkpoint.get("timestamp", "unknown")
        msg = (
            f"Checkpoint exists from {timestamp} "
            f"(last completed config index: {last_index}, results: {len(existing_results)}). "
            "Resume from checkpoint?"
        )

        if sys.stdin.isatty():
            try:
                answer = input(f"{msg} [y/N] ").strip().lower()
            except EOFError:
                answer = ""
            if answer in ("y", "yes"):
                logger.info("Resuming from checkpoint.")
                return existing_results
            else:
                logger.info("Starting fresh (checkpoint will be overwritten).")
                checkpoint_path.unlink(missing_ok=True)
                return []
        else:
            logger.error(
                "Checkpoint exists and not running interactively. Use --resume to resume or --force to overwrite."
            )
            raise SystemExit(1)

    def _load_checkpoint(self, path: Path) -> dict[str, Any]:
        """Read a checkpoint file from disk."""
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("checkpoint must be a JSON object")
            return raw
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("invalid checkpoint file, starting fresh: %s", exc)
            return {}

    def _save_checkpoint(self, results: list[RunResult], trial_index: int) -> None:
        """Write a checkpoint file to the run directory."""
        checkpoint = {
            "trial_index": trial_index,
            "results": [r.model_dump() for r in results],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = self.run_dir / CHECKPOINT_FILE
        try:
            path.write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")
        except OSError as exc:
            logger.warning("failed to save checkpoint: %s", exc)

    async def _health_check(self) -> bool:
        """Send a minimal request to verify endpoint health."""
        try:
            result = await self.client.complete(
                messages=[{"role": "user", "content": "ping"}],
                model=self.model_name,
                max_tokens=1,
                temperature=0.0,
            )
            return bool(result.response_text)
        except Exception as exc:
            logger.error("Health check failed: %s", exc)
            return False
