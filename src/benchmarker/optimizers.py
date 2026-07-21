"""Optimizer implementations for parameter search (Phase 5).

Provides a pluggable interface (:class:`BaseOptimizer`) with three strategies:
grid search, random search, and Bayesian optimization backed by Optuna.
"""

from __future__ import annotations

import itertools
import random
from abc import ABC, abstractmethod
from typing import Any

import optuna
from optuna import Study

from benchmarker.config import OptimizerConfig, ParameterSpec, ParameterType


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
        value = metrics.get("tokens_per_sec")
        if value is None:
            # mark as failed trial
            self.study.tell(self._last_trial, state=optuna.trial.TrialState.FAIL)
        else:
            self.study.tell(self._last_trial, float(value))
        self._last_trial = None

    def estimated_steps(self) -> int:
        return self.budget


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


def create_optimizer(config: OptimizerConfig, parameters: list[ParameterSpec]) -> BaseOptimizer:
    """Factory: build the optimizer described by ``config``."""
    if config.type == "grid":
        return GridOptimizer(parameters)
    if config.type == "random":
        return RandomOptimizer(parameters, budget=config.budget)
    if config.type == "bayesian":
        return BayesianOptimizer(parameters, budget=config.budget)
    if config.type == "baseline_sweep":
        return ControlledOptimizer(parameters, baseline=config.baseline)
    raise ValueError(f"Unknown optimizer type: {config.type!r}")
