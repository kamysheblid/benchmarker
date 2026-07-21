"""Tests for configuration loading and Pydantic models (Phase 2)."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarker.config import (
    OptimizerConfig,
    ParameterSpec,
    ParamsConfig,
    ParameterType,
    TestCase,
    TestSuite,
    load_params,
    load_tests,
)


# --------------------------------------------------------------------------- #
# Pydantic model construction
# --------------------------------------------------------------------------- #
def test_parameter_spec_float() -> None:
    spec = ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=1.5)
    assert spec.name == "temperature"
    assert spec.low == 0.1
    assert spec.high == 1.5


def test_parameter_spec_categorical_requires_choices() -> None:
    with pytest.raises(ValidationError):
        ParameterSpec(name="stop", type=ParameterType.CATEGORICAL, low=0, high=1)
    spec = ParameterSpec(name="stop", type=ParameterType.CATEGORICAL, choices=["\n", "<|end|>"])
    assert spec.choices == ["\n", "<|end|>"]


def test_optimizer_config_default_budget() -> None:
    cfg = OptimizerConfig(type="bayesian")
    assert cfg.budget == 20


def test_test_case_default_repeat() -> None:
    tc = TestCase(id="t1", prompt="Hello")
    assert tc.repeat == 1
    assert tc.max_tokens is None
    assert tc.system is None
    assert tc.stop is None
    assert tc.reasoning is None


def test_test_case_with_stop_and_reasoning() -> None:
    tc = TestCase(id="t1", prompt="Hello", stop=["\n```", "\ndef "], reasoning=False)
    assert tc.stop == ["\n```", "\ndef "]
    assert tc.reasoning is False


# --------------------------------------------------------------------------- #
# load_params (YAML)
# --------------------------------------------------------------------------- #
def test_load_params_valid(tmp_path: Path) -> None:
    yaml_text = """
optimizer:
  type: bayesian
  budget: 10
parameters:
  - name: temperature
    type: float
    low: 0.1
    high: 1.5
  - name: top_k
    type: int
    low: 10
    high: 50
  - name: strategy
    type: categorical
    choices: [a, b, c]
"""
    path = tmp_path / "params.yaml"
    path.write_text(yaml_text)
    cfg = load_params(path)
    assert isinstance(cfg, ParamsConfig)
    assert cfg.optimizer.type == "bayesian"
    assert cfg.optimizer.budget == 10
    assert len(cfg.parameters) == 3
    assert cfg.parameters[0].name == "temperature"


def test_load_params_missing_optimizer_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("parameters:\n  - name: temperature\n    type: float\n    low: 0\n    high: 1\n")
    with pytest.raises(ValidationError):
        load_params(path)


def test_load_params_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "does_not_exist.yaml")


# --------------------------------------------------------------------------- #
# load_tests (JSON)
# --------------------------------------------------------------------------- #
def test_load_tests_valid(tmp_path: Path) -> None:
    data = [
        {"id": "t1", "prompt": "Say hi", "max_tokens": 50, "repeat": 2},
        {"id": "t2", "prompt": "Translate", "system": "You are helpful"},
    ]
    path = tmp_path / "tests.json"
    path.write_text(json.dumps(data))
    suite = load_tests(path)
    assert isinstance(suite, TestSuite)
    assert len(suite.tests) == 2
    assert suite.tests[0].repeat == 2
    assert suite.tests[1].system == "You are helpful"


def test_load_tests_default_repeat(tmp_path: Path) -> None:
    path = tmp_path / "tests.json"
    path.write_text(json.dumps([{"id": "t1", "prompt": "Hi"}]))
    suite = load_tests(path)
    assert suite.tests[0].repeat == 1


# --------------------------------------------------------------------------- #
# Phase 3: edge cases
# --------------------------------------------------------------------------- #
def test_load_tests_missing_file_raises_with_message(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError) as exc:
        load_tests(missing)
    assert "nope.json" in str(exc.value)


def test_load_tests_duplicate_ids_raises(tmp_path: Path) -> None:
    path = tmp_path / "tests.json"
    path.write_text(json.dumps([{"id": "t1", "prompt": "a"}, {"id": "t1", "prompt": "b"}]))
    with pytest.raises(ValidationError) as exc:
        load_tests(path)
    assert "duplicate test id" in str(exc.value)


def test_load_tests_missing_prompt_raises(tmp_path: Path) -> None:
    path = tmp_path / "tests.json"
    path.write_text(json.dumps([{"id": "t1"}]))
    with pytest.raises(ValidationError):
        load_tests(path)


def test_load_tests_empty_suite_ok(tmp_path: Path) -> None:
    path = tmp_path / "tests.json"
    path.write_text(json.dumps([]))
    suite = load_tests(path)
    assert suite.tests == []


def test_load_tests_negative_max_tokens_raises(tmp_path: Path) -> None:
    path = tmp_path / "tests.json"
    path.write_text(json.dumps([{"id": "t1", "prompt": "x", "max_tokens": -5}]))
    with pytest.raises(ValidationError):
        load_tests(path)


def test_load_tests_max_tokens_must_be_int(tmp_path: Path) -> None:
    path = tmp_path / "tests.json"
    path.write_text(json.dumps([{"id": "t1", "prompt": "x", "max_tokens": 1.5}]))
    with pytest.raises(ValidationError):
        load_tests(path)
