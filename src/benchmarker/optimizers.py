"""?Optimizer implementations for parameter search (Phase 5).

Provides a pluggable interface (:class:`BaseOptimizer`) with three strategies:
grid search, random search, and Bayesian optimization backed by Optuna.
"""

from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import optuna
from optuna import Study

from benchmarker.config import OptimizerConfig, ParameterSpec, ParameterType
from benchmarker.optimizer_history import OptimizerHistory


class BaseOptimizer(ABC):
    """Abstract base for parameter-space search strategies."""

    def __init__(self, parameters: list[ParameterSpec]) -> None:
        self.parameters = parameters

    @abstractmethod
    def suggest(self) -> dict[str, Any]:
        """Return the next parameter set to evaluate."""

    def tell(self, metrics: dict[str, Any]) -> None:
        """Report the outcome of the last ``suggest()`` (no-op by default)."""

    def estimated_steps(self) -> int:
        """Best-effort number of suggestions (0 if unknown)."""
        return 0

    def __iter__(self) -> "BaseOptimizer":
        return self

    def __next__(self) -> dict[str, Any]:
        return self.suggest()


class GridOptimizer(BaseOptimizer):
    """Enumerate every combination of parameter values via ``itertools.product``."""

    def __init__(self, parameters: list[ParameterSpec]) -> None:
        super().__init__(parameters)
        self._combos = self._build_combinations()
        self._index = 0

    def _values_for(self, spec: ParameterSpec) -> list[Any]:
        if spec.type is not ParameterType.CATEGORICAL and spec.low is not None and spec.low == spec.high:
            # degenerate range -> single point
            if spec.type is ParameterType.INT:
                return [int(spec.low)]
            return [float(spec.low)]
        if spec.type is ParameterType.CATEGORICAL:
            return list(spec.choices or [])
        if spec.step is not None:
            low, high, step = spec.low, spec.high, spec.step
            n = int(round((high - low) / step)) + 1
            seq = [low + i * step for i in range(n)]
            # ensure inclusive of high
            if seq[-1] < high:
                seq.append(high)
            if spec.type is ParameterType.INT:
                return [int(round(v)) for v in seq]
            return [float(v) for v in seq]
        # continuous range: sample a reasonable number of points
        low, high = spec.low, spec.high
        if spec.type is ParameterType.INT:
            return list(range(int(low), int(high) + 1))
        n = 10
        return [low + (high - low) * i / (n - 1) for i in range(n)]

    def _build_combinations(self) -> list[dict[str, Any]]:
        value_lists = [self._values_for(spec) for spec in self.parameters]
        combos = []
        for combo in itertools.product(*value_lists):
            combos.append({spec.name: val for spec, val in zip(self.parameters, combo)})
        return combos

    def suggest(self) -> dict[str, Any]:
        if self._index >= len(self._combos):
            raise StopIteration
        combo = self._combos[self._index]
        self._index += 1
        return dict(combo)

    def estimated_steps(self) -> int:
        """Number of suggestions this optimizer will yield."""
        return len(self._combos)


class RandomOptimizer(BaseOptimizer):
    """Sample random parameter combinations until a budget is exhausted."""

    def __init__(self, parameters: list[ParameterSpec], budget: int, seed: int | None = None) -> None:
        super().__init__(parameters)
        self.budget = budget
        self._count = 0
        self._rng = random.Random(seed)

    def _sample_one(self, spec: ParameterSpec) -> Any:
        if spec.type is ParameterType.CATEGORICAL:
            return self._rng.choice(list(spec.choices or []))
        if spec.type is ParameterType.INT:
            return self._rng.randint(int(spec.low), int(spec.high))
        if spec.step is not None:
            low, high, step = spec.low, spec.high, spec.step
            n = int(round((high - low) / step))
            n = max(n, 1)
            return float(low + self._rng.randint(0, n) * step)
        return self._rng.uniform(spec.low, spec.high)

    def suggest(self) -> dict[str, Any]:
        if self._count >= self.budget:
            raise StopIteration
        self._count += 1
        return {spec.name: self._sample_one(spec) for spec in self.parameters}

    def estimated_steps(self) -> int:
        return self.budget


class BayesianOptimizer(BaseOptimizer):
    """Bayesian optimization using Optuna's ``ask``/``tell`` API."""

    def __init__(
        self,
        parameters: list[ParameterSpec],
        budget: int = 20,
        study: Study | None = None,
        direction: str = "maximize",
        seed: int | None = None,
    ) -> None:
        super().__init__(parameters)
        self.budget = budget
        self._count = 0
        self._last_trial: optuna.trial.Trial | None = None
        self.seed = seed
        self.study = study or optuna.create_study(direction=direction, sampler=optuna.samplers.RandomSampler(seed=seed))

    def suggest(self) -> dict[str, Any]:
        if self._count >= self.budget:
            raise StopIteration
        trial = self.study.ask()
        self._last_trial = trial
        self._count += 1
        result: dict[str, Any] = {}
        for spec in self.parameters:
            if spec.type is ParameterType.CATEGORICAL:
                result[spec.name] = trial.suggest_categorical(spec.name, list(spec.choices or []))
            elif spec.type is ParameterType.INT:
                result[spec.name] = trial.suggest_int(spec.name, int(spec.low), int(spec.high))
            else:
                step = float(spec.step) if spec.step is not None else None
                result[spec.name] = trial.suggest_float(spec.name, float(spec.low), float(spec.high), step=step)
        return result

    def tell(self, metrics: dict[str, Any]) -> None:
        if self._last_trial is None:
            return
        tokens_per_sec = metrics.get("tokens_per_sec")
        reliability_score = metrics.get("reliability_score")
        success_rate = metrics.get("success_rate")
        if tokens_per_sec is None and reliability_score is None:
            self.study.tell(self._last_trial, state=optuna.trial.TrialState.FAIL)
        else:
            value = float(tokens_per_sec) if tokens_per_sec is not None else float(reliability_score)
            self.study.tell(self._last_trial, value)
        self._last_trial = None

    def estimated_steps(self) -> int:
        return self.budget

    @classmethod
    def from_history(
        cls,
        history_path: Path,
        parameters: list[ParameterSpec],
        budget: int = 20,
        seed: int | None = None,
    ) -> BayesianOptimizer:
        history = OptimizerHistory.from_json(history_path)
        optimizer = cls(parameters=parameters, budget=budget, seed=seed)
        history.replay_into(optimizer)
        return optimizer


class ControlledOptimizer(BaseOptimizer):
    """Ablation study: vary one parameter at a time while keeping others fixed.

    Starts with a baseline config, then for each parameter generates configs
    where only that parameter varies across its range. All other parameters
    are held at their baseline value. This isolates the effect of each
    parameter on performance.
    """

    def __init__(self, parameters: list[ParameterSpec], baseline: dict[str, Any] | None = None) -> None:
        super().__init__(parameters)
        self._baseline = dict(baseline or {})
        self._combos = self._build_ablations()
        self._index = 0

    def _values_for(self, spec: ParameterSpec) -> list[Any]:
        """Same value generation logic as GridOptimizer."""
        if spec.type is not ParameterType.CATEGORICAL and spec.low is not None and spec.low == spec.high:
            if spec.type is ParameterType.INT:
                return [int(spec.low)]
            return [float(spec.low)]
        if spec.type is ParameterType.CATEGORICAL:
            return list(spec.choices or [])
        if spec.step is not None:
            low, high, step = spec.low, spec.high, spec.step
            n = int(round((high - low) / step)) + 1
            seq = [low + i * step for i in range(n)]
            if seq[-1] < high:
                seq.append(high)
            if spec.type is ParameterType.INT:
                return [int(round(v)) for v in seq]
            return [float(v) for v in seq]
        low, high = spec.low, spec.high
        if spec.type is ParameterType.INT:
            return list(range(int(low), int(high) + 1))
        n = 10
        return [low + (high - low) * i / (n - 1) for i in range(n)]

    def _build_ablations(self) -> list[dict[str, Any]]:
        """Build ablation configs — one-parameter-at-a-time variations."""
        configs: list[dict[str, Any]] = []
        # Include the baseline itself as the reference point
        if self._baseline:
            configs.append(dict(self._baseline))

        for spec in self.parameters:
            values = self._values_for(spec)
            for val in values:
                cfg = dict(self._baseline)
                cfg[spec.name] = val
                configs.append(cfg)

        return configs

    def suggest(self) -> dict[str, Any]:
        if self._index >= len(self._combos):
            raise StopIteration
        combo = self._combos[self._index]
        self._index += 1
        return dict(combo)

    def estimated_steps(self) -> int:
        return len(self._combos)


class AdaptiveOptimizer(BaseOptimizer):
    """Narrowed refinement of a previous search based on a refinement hint.

    Accepts the original parameter spec and a refinement hint dict like
    ``{"temperature": [0.7, 0.9], "top_p": [0.8, 1.0]}``. Generates a grid
    over the hinted ranges with a finer step (controlled by ``resolution_factor``).

    When no hint is given, delegates to ``GridOptimizer`` with the original spec.
    """

    def __init__(
        self,
        parameters: list[ParameterSpec],
        refinement_hint: dict[str, list[float]] | None = None,
        resolution_factor: int = 5,
    ) -> None:
        super().__init__(parameters)
        self.refinement_hint = refinement_hint or {}
        self.resolution_factor = resolution_factor
        self._combos = self._build_refined()
        self._index = 0

    def _build_refined(self) -> list[dict[str, Any]]:
        """Build a refined grid over hinted ranges, or fall back to the original grid."""
        if not self.refinement_hint:
            # No hint → full original grid
            return GridOptimizer(self.parameters)._combos

        combos: list[dict[str, Any]] = []
        refined_specs: list[ParameterSpec] = []

        for spec in self.parameters:
            hint = self.refinement_hint.get(spec.name)
            if hint and len(hint) >= 2:
                lo, hi = float(hint[0]), float(hint[1])
                # Create a narrowed spec with finer resolution
                refined_specs.append(
                    ParameterSpec(
                        name=spec.name,
                        type=spec.type,
                        low=lo,
                        high=hi,
                        step=(hi - lo) / self.resolution_factor,
                        choices=spec.choices,
                    )
                )
            else:
                refined_specs.append(spec)

        value_lists: list[list[Any]] = []
        for spec in refined_specs:
            if spec.type is ParameterType.CATEGORICAL:
                vals = list(spec.choices or [])
            elif spec.step is not None:
                vals = []
                v = spec.low
                while v <= spec.high + 1e-9:
                    if spec.type is ParameterType.INT:
                        vals.append(int(round(v)))
                    else:
                        vals.append(float(v))
                    v += spec.step
            else:
                vals = [spec.low, spec.high]
            value_lists.append(vals)

        for combo in itertools.product(*value_lists):
            combos.append({spec.name: val for spec, val in zip(refined_specs, combo)})
        return combos

    def suggest(self) -> dict[str, Any]:
        if self._index >= len(self._combos):
            raise StopIteration
        combo = self._combos[self._index]
        self._index += 1
        return dict(combo)

    def estimated_steps(self) -> int:
        return len(self._combos)


class TwoPhaseOptimizer(BaseOptimizer):
    """Wrapper that runs phase1 then delegates to phase2 using a refinement hint.

    The runner is responsible for constructing the phase2 ``AdaptiveOptimizer``
    and calling ``switch_phase()`` when the phase1 budget is exhausted.
    """

    def __init__(
        self,
        phase1: BaseOptimizer,
        phase2: BaseOptimizer | None,
        phase1_budget: int,
    ) -> None:
        super().__init__(phase1.parameters)
        self._phase1 = phase1
        self._phase2 = phase2
        self._phase1_budget = phase1_budget
        self._active = phase1

    def suggest(self) -> dict[str, Any]:
        return self._active.suggest()

    def tell(self, metrics: dict[str, Any]) -> None:
        self._active.tell(metrics)

    def switch_phase(self) -> None:
        if self._phase2 is not None:
            self._active = self._phase2

    def estimated_steps(self) -> int:
        steps = self._phase1.estimated_steps()
        if self._phase2 is not None:
            steps += self._phase2.estimated_steps()
        return steps


def create_optimizer(
    config: OptimizerConfig,
    parameters: list[ParameterSpec],
    seed: int | None = None,
    refinement_hint: dict[str, list[float]] | None = None,
    resolution_factor: int = 5,
) -> BaseOptimizer:
    """Factory: build the optimizer described by ``config``.

    When ``refinement_hint`` is provided and the optimizer type is ``grid``,
    an :class:`AdaptiveOptimizer` is returned instead — narrowing the search
    space to the hinted ranges with finer granularity.
    """
    if refinement_hint and config.type in ("grid", "random"):
        return AdaptiveOptimizer(parameters, refinement_hint=refinement_hint, resolution_factor=resolution_factor)
    if config.type == "grid":
        return GridOptimizer(parameters)
    if config.type == "random":
        return RandomOptimizer(parameters, budget=config.budget, seed=seed)
    if config.type == "bayesian":
        return BayesianOptimizer(parameters, budget=config.budget, seed=seed)
    if config.type == "baseline_sweep":
        return ControlledOptimizer(parameters, baseline=config.baseline)
    raise ValueError(f"Unknown optimizer type: {config.type!r}")
