"""Tests for optimizer implementations (Phase 5)."""

from pathlib import Path
from typing import Any

import pytest
from optuna import Study, Trial

from benchmarker.config import OptimizerConfig, ParameterSpec, ParameterType
from benchmarker.optimizers import (
    BaseOptimizer,
    BayesianOptimizer,
    ControlledOptimizer,
    GridOptimizer,
    RandomOptimizer,
    create_optimizer,
)
from benchmarker.optimizer_history import OptimizerHistory, OptimizerTrial


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
# Controlled / Ablation
# --------------------------------------------------------------------------- #
def test_controlled_requires_baseline() -> None:
    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.0, high=1.0, step=0.5)]
    opt = ControlledOptimizer(specs, baseline={"top_k": 40})
    combos = list(iter(opt))
    # baseline + 3 temperature values (0.0, 0.5, 1.0) = 4
    assert len(combos) == 4
    # All combos include the baseline param
    for c in combos:
        assert c["top_k"] == 40


def test_controlled_varies_one_param_at_a_time() -> None:
    baseline = {"temperature": 0.6, "top_p": 0.9}
    specs = [
        ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.0, high=1.0, step=0.5),
        ParameterSpec(name="top_p", type=ParameterType.FLOAT, low=0.5, high=1.0, step=0.25),
    ]
    opt = ControlledOptimizer(specs, baseline=baseline)
    combos = list(iter(opt))
    # baseline + 3 temp + 3 top_p = 7
    assert len(combos) == 7

    # Each non-baseline config differs in exactly one parameter
    for c in combos[1:]:
        diffs = sum(1 for k in baseline if c.get(k) != baseline[k])
        assert diffs == 1, f"{c} differs in {diffs} params, expected 1"


def test_controlled_estimated_steps() -> None:
    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.0, high=1.0, step=0.5)]
    opt = ControlledOptimizer(specs, baseline={"top_k": 40})
    assert opt.estimated_steps() == 4


def test_controlled_stop_iteration() -> None:
    opt = ControlledOptimizer([], baseline={"x": 1})
    list(iter(opt))  # drain
    with pytest.raises(StopIteration):
        opt.suggest()


def test_factory_creates_controlled() -> None:
    cfg = OptimizerConfig(type="baseline_sweep", baseline={"temperature": 0.6})
    specs = [ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.0, high=1.0, step=0.5)]
    opt = create_optimizer(cfg, specs)
    assert isinstance(opt, ControlledOptimizer)


def test_factory_invalid_type_raises() -> None:
    with pytest.raises((ValueError, Exception), match="baseline_sweep"):
        # Pydantic catches the invalid literal_type before the factory runs
        create_optimizer(OptimizerConfig(type="invalid"), [])


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


# --------------------------------------------------------------------------- #
# Optimizer history persistence
# --------------------------------------------------------------------------- #
def test_history_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = OptimizerHistory(
        trials=[
            OptimizerTrial(params={"temperature": 0.5}, quality=80.0, tokens_per_sec=100.0),
            OptimizerTrial(params={"temperature": 0.8}, quality=90.0, tokens_per_sec=120.0),
        ]
    )
    history.to_json(path)
    loaded = OptimizerHistory.from_json(path)
    assert len(loaded.trials) == 2
    assert loaded.trials[0].params == {"temperature": 0.5}
    assert loaded.trials[0].quality == 80.0
    assert loaded.trials[0].tokens_per_sec == 100.0
    assert loaded.trials[1].params == {"temperature": 0.8}
    assert loaded.trials[1].quality == 90.0
    assert loaded.trials[1].tokens_per_sec == 120.0


def test_bayesian_replay_history(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = OptimizerHistory(
        trials=[
            OptimizerTrial(params={"temperature": 0.3 + i * 0.1}, quality=50.0 + i * 10.0, tokens_per_sec=80.0 + i * 5.0)
            for i in range(5)
        ]
    )
    history.to_json(path)

    opt = BayesianOptimizer.from_history(
        path=path,
        parameters=_specs(),
        budget=10,
        seed=42,
    )
    assert isinstance(opt, BayesianOptimizer)
    # The study should now contain 5 completed trials
    assert len(opt.study.trials) == 5


def test_history_serialization_with_none_fields(tmp_path: Path) -> None:
    path = tmp_path / "history.json"
    history = OptimizerHistory(
        trials=[
            OptimizerTrial(params={"x": 1}, quality=None, tokens_per_sec=None),
            OptimizerTrial(params={"x": 2}, quality=75.0, tokens_per_sec=None),
            OptimizerTrial(params={"x": 3}, quality=None, tokens_per_sec=60.0),
        ]
    )
    history.to_json(path)
    loaded = OptimizerHistory.from_json(path)
    assert len(loaded.trials) == 3
    assert loaded.trials[0].quality is None
    assert loaded.trials[0].tokens_per_sec is None
    assert loaded.trials[1].quality == 75.0
    assert loaded.trials[1].tokens_per_sec is None
    assert loaded.trials[2].quality is None
    assert loaded.trials[2].tokens_per_sec == 60.0
