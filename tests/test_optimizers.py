"""Tests for optimizer implementations (Phase 5)."""

from typing import Any

import pytest
from optuna import Study, Trial

from benchmarker.config import OptimizerConfig, ParameterSpec, ParameterType
from benchmarker.optimizers import (
    BaseOptimizer,
    BayesianOptimizer,
    GridOptimizer,
    RandomOptimizer,
    create_optimizer,
)


def _specs() -> list[ParameterSpec]:
    return [
        ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=1.0, step=0.1),
        ParameterSpec(name="top_k", type=ParameterType.INT, low=10, high=12),
        ParameterSpec(name="strategy", type=ParameterType.CATEGORICAL, choices=["a", "b"]),
    ]


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def test_grid_all_combinations() -> None:
    grid = GridOptimizer(_specs())
    combos = list(iter(grid))
    # temperature: 0.1,0.2,...,1.0 -> 10; top_k: 10,11,12 -> 3; strategy: a,b -> 2
    assert len(combos) == 10 * 3 * 2
    # every combo has all keys
    assert all(set(c.keys()) == {"temperature", "top_k", "strategy"} for c in combos)
    # unique
    assert len({tuple(c.items()) for c in combos}) == len(combos)


def test_grid_stop_iteration() -> None:
    grid = GridOptimizer(_specs())
    # drain it
    for _ in grid:
        pass
    with pytest.raises(StopIteration):
        grid.suggest()


# --------------------------------------------------------------------------- #
# Random
# --------------------------------------------------------------------------- #
def test_random_respects_budget() -> None:
    opt = RandomOptimizer(_specs(), budget=15)
    samples = [opt.suggest() for _ in range(15)]
    assert len(samples) == 15
    with pytest.raises(StopIteration):
        opt.suggest()


def test_random_values_in_bounds() -> None:
    opt = RandomOptimizer(_specs(), budget=200)
    for s in (opt.suggest() for _ in range(200)):
        assert 0.1 <= s["temperature"] <= 1.0
        assert s["top_k"] in (10, 11, 12)
        assert s["strategy"] in ("a", "b")


def test_random_int_step_alignment() -> None:
    spec = [ParameterSpec(name="top_k", type=ParameterType.INT, low=10, high=12)]
    opt = RandomOptimizer(spec, budget=50)
    for s in (opt.suggest() for _ in range(50)):
        assert isinstance(s["top_k"], int)


# --------------------------------------------------------------------------- #
# Bayesian
# --------------------------------------------------------------------------- #
class _FakeTrial(Trial):
    """Minimal stand-in capturing suggest_* calls."""

    def __init__(self, number: int, values: dict[str, Any]) -> None:
        # avoid calling super().__init__ which needs a study
        self._number = number
        self._values = values
        self.suggested: dict[str, Any] = {}

    @property
    def number(self) -> int:  # type: ignore[override]
        return self._number

    def suggest_float(self, name, low, high, **kwargs):  # type: ignore[override]
        self.suggested[name] = (low, high)
        return float(low)

    def suggest_int(self, name, low, high, **kwargs):  # type: ignore[override]
        self.suggested[name] = (low, high)
        return int(low)

    def suggest_categorical(self, name, choices, **kwargs):  # type: ignore[override]
        self.suggested[name] = list(choices)
        return choices[0]


class _FakeStudy(Study):
    def __init__(self) -> None:
        self.asked: list[_FakeTrial] = []
        self.told: list[tuple[int, float]] = []

    def ask(self, *, trial=None):  # type: ignore[override]
        t = _FakeTrial(len(self.asked), {})
        self.asked.append(t)
        return t

    def tell(self, trial, value=None, **kwargs):  # type: ignore[override]
        self.told.append((trial.number, value))


def test_bayesian_uses_ask_and_tell() -> None:
    study = _FakeStudy()
    opt = BayesianOptimizer(_specs(), budget=3, study=study)
    for i in range(3):
        trial = opt.suggest()
        # fill the trial with the values our fake expects by re-asking? Instead,
        # verify suggest returns a dict within bounds and tell records value.
        assert set(trial.keys()) == {"temperature", "top_k", "strategy"}
        opt.tell({"tokens_per_sec": 10.0 + i})
    assert len(study.asked) == 3
    assert len(study.told) == 3
    assert [v for _, v in study.told] == [10.0, 11.0, 12.0]


def test_bayesian_real_optuna_bounds() -> None:
    opt = BayesianOptimizer(_specs(), budget=20)
    for _ in range(20):
        s = opt.suggest()
        assert 0.1 <= s["temperature"] <= 1.0
        assert 10 <= s["top_k"] <= 12
        assert s["strategy"] in ("a", "b")
        opt.tell({"tokens_per_sec": 5.0})


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_create_optimizer_factory() -> None:
    assert isinstance(create_optimizer(OptimizerConfig(type="grid", budget=5), _specs()), GridOptimizer)
    assert isinstance(
        create_optimizer(OptimizerConfig(type="random", budget=5), _specs()), RandomOptimizer
    )
    assert isinstance(
        create_optimizer(OptimizerConfig(type="bayesian", budget=5), _specs()), BayesianOptimizer
    )
