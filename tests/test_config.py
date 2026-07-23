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
    discover_benchmark_files,
    discover_categories,
    load_benchmark_file,
    load_params,
    load_tests,
    load_tests_from_dir,
    merge_params,
    validate_params,
    validate_params_match,
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
    assert tc.reasoning is None


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
    with pytest.raises(ValueError) as exc:
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


# --------------------------------------------------------------------------- #
# Directory-aware loaders
# --------------------------------------------------------------------------- #
def test_discover_categories_empty(tmp_path: Path) -> None:
    assert discover_categories(tmp_path) == []


def test_discover_categories_sorted(tmp_path: Path) -> None:
    (tmp_path / "zebra").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "mango").mkdir()
    assert discover_categories(tmp_path) == ["alpha", "mango", "zebra"]


def test_discover_categories_ignores_files(tmp_path: Path) -> None:
    (tmp_path / "readme.md").write_text("hello")
    (tmp_path / "cat-a").mkdir()
    assert discover_categories(tmp_path) == ["cat-a"]


def test_load_tests_from_dir_loads_all(tmp_path: Path) -> None:
    cat_a = tmp_path / "cat-a"
    cat_a.mkdir()
    (cat_a / "001-first.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))
    (cat_a / "002-second.json").write_text(json.dumps({"id": "t2", "prompt": "B"}))

    cat_b = tmp_path / "cat-b"
    cat_b.mkdir()
    (cat_b / "001-third.json").write_text(json.dumps({"id": "t3", "prompt": "C"}))

    suite = load_tests_from_dir(tmp_path)
    assert len(suite.tests) == 3
    assert suite.tests[0].id == "t1"
    assert suite.tests[1].id == "t2"
    assert suite.tests[2].id == "t3"


def test_load_tests_from_dir_filters_categories(tmp_path: Path) -> None:
    cat_a = tmp_path / "cat-a"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

    cat_b = tmp_path / "cat-b"
    cat_b.mkdir()
    (cat_b / "001-b.json").write_text(json.dumps({"id": "t2", "prompt": "B"}))

    suite = load_tests_from_dir(tmp_path, categories=["cat-a"])
    assert len(suite.tests) == 1
    assert suite.tests[0].id == "t1"


def test_load_tests_from_dir_skips_empty_category(tmp_path: Path) -> None:
    cat_a = tmp_path / "cat-a"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

    (tmp_path / "cat-b").mkdir()

    suite = load_tests_from_dir(tmp_path)
    assert len(suite.tests) == 1
    assert suite.tests[0].id == "t1"


def test_load_tests_from_dir_skips_non_json(tmp_path: Path) -> None:
    cat_a = tmp_path / "cat-a"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))
    (cat_a / "readme.txt").write_text("ignored")
    (cat_a / "data.yaml").write_text("ignored")

    suite = load_tests_from_dir(tmp_path)
    assert len(suite.tests) == 1
    assert suite.tests[0].id == "t1"


def test_load_tests_from_dir_duplicate_ids_raises_value_error(tmp_path: Path) -> None:
    cat_a = tmp_path / "cat-a"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

    cat_b = tmp_path / "cat-b"
    cat_b.mkdir()
    (cat_b / "001-b.json").write_text(json.dumps({"id": "t1", "prompt": "B"}))

    with pytest.raises(ValueError) as exc:
        load_tests_from_dir(tmp_path)
    assert "duplicate test id" in str(exc.value)


def test_load_tests_from_dir_case_sensitive_categories(tmp_path: Path) -> None:
    cat_a = tmp_path / "cat-a"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

    with pytest.raises(ValueError) as exc:
        load_tests_from_dir(tmp_path, categories=["Cat-A"])
    assert "Cat-A" in str(exc.value)


# --------------------------------------------------------------------------- #
# Directory-aware loader — additional coverage
# --------------------------------------------------------------------------- #
def test_load_tests_from_dir_loads_all(tmp_path: Path) -> None:
    cat_a = tmp_path / "code-generation"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

    cat_b = tmp_path / "bug-fixing"
    cat_b.mkdir()
    (cat_b / "001-b.json").write_text(json.dumps({"id": "t2", "prompt": "B"}))

    suite = load_tests_from_dir(tmp_path)
    assert isinstance(suite, TestSuite)
    assert len(suite.tests) == 2
    assert suite.tests[0].id == "t2"
    assert suite.tests[1].id == "t1"


def test_load_tests_from_dir_category_filter(tmp_path: Path) -> None:
    cat_a = tmp_path / "code-generation"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

    cat_b = tmp_path / "bug-fixing"
    cat_b.mkdir()
    (cat_b / "001-b.json").write_text(json.dumps({"id": "t2", "prompt": "B"}))

    cat_c = tmp_path / "refactoring"
    cat_c.mkdir()
    (cat_c / "001-c.json").write_text(json.dumps({"id": "t3", "prompt": "C"}))

    suite = load_tests_from_dir(tmp_path, categories=["bug-fixing"])
    assert len(suite.tests) == 1
    assert suite.tests[0].id == "t2"


def test_load_tests_from_dir_empty_category_skipped(tmp_path: Path) -> None:
    cat_a = tmp_path / "code-generation"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

    (tmp_path / "empty-category").mkdir()

    suite = load_tests_from_dir(tmp_path)
    assert len(suite.tests) == 1
    assert suite.tests[0].id == "t1"


def test_load_tests_from_dir_nonexistent_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError) as exc:
        load_tests_from_dir(missing)
    assert "does_not_exist" in str(exc.value)


def test_load_tests_from_dir_duplicate_ids_across_categories_raises(tmp_path: Path) -> None:
    cat_a = tmp_path / "code-generation"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

    cat_b = tmp_path / "bug-fixing"
    cat_b.mkdir()
    (cat_b / "001-b.json").write_text(json.dumps({"id": "t1", "prompt": "B"}))

    with pytest.raises(ValueError) as exc:
        load_tests_from_dir(tmp_path)
    assert "duplicate test id" in str(exc.value)


def test_discover_categories_lists_sorted(tmp_path: Path) -> None:
    (tmp_path / "zebra").mkdir()
    (tmp_path / "alpha").mkdir()
    (tmp_path / "mango").mkdir()
    assert discover_categories(tmp_path) == ["alpha", "mango", "zebra"]


def test_load_tests_from_dir_invalid_category_raises(tmp_path: Path) -> None:
    cat_a = tmp_path / "code-generation"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

    with pytest.raises(ValueError) as exc:
        load_tests_from_dir(tmp_path, categories=["nonexistent"])
    assert "nonexistent" in str(exc.value)


# --------------------------------------------------------------------------- #
# validate_params (1.1)
# --------------------------------------------------------------------------- #
def test_validate_params_valid() -> None:
    cfg = ParamsConfig(
        optimizer=OptimizerConfig(type="grid", budget=10),
        parameters=[
            ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=1.0, step=0.1)
        ],
    )
    validate_params(cfg)  # should not raise


def test_validate_params_low_gt_high_raises() -> None:
    cfg = ParamsConfig(
        optimizer=OptimizerConfig(type="grid", budget=10),
        parameters=[
            ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=1.0, high=0.1)
        ],
    )
    with pytest.raises(ValueError, match="low .* high"):
        validate_params(cfg)


def test_validate_params_step_zero_raises() -> None:
    cfg = ParamsConfig(
        optimizer=OptimizerConfig(type="grid", budget=10),
        parameters=[
            ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=1.0, step=0)
        ],
    )
    with pytest.raises(ValueError, match="step"):
        validate_params(cfg)


def test_validate_params_step_negative_raises() -> None:
    cfg = ParamsConfig(
        optimizer=OptimizerConfig(type="grid", budget=10),
        parameters=[
            ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=1.0, step=-0.1)
        ],
    )
    with pytest.raises(ValueError, match="step"):
        validate_params(cfg)


def test_validate_params_step_exceeds_range_raises() -> None:
    cfg = ParamsConfig(
        optimizer=OptimizerConfig(type="grid", budget=10),
        parameters=[
            ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.0, high=0.5, step=1.0)
        ],
    )
    with pytest.raises(ValueError, match="step"):
        validate_params(cfg)


def test_validate_params_step_within_range_ok() -> None:
    cfg = ParamsConfig(
        optimizer=OptimizerConfig(type="grid", budget=10),
        parameters=[
            ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.0, high=1.0, step=0.5)
        ],
    )
    validate_params(cfg)  # should not raise


def test_validate_params_budget_implicitly_valid() -> None:
    cfg = ParamsConfig(
        optimizer=OptimizerConfig(type="grid", budget=1),
        parameters=[],
    )
    validate_params(cfg)  # should not raise


def test_validate_params_categorical_skips_numeric_checks() -> None:
    cfg = ParamsConfig(
        optimizer=OptimizerConfig(type="grid", budget=5),
        parameters=[
            ParameterSpec(name="strategy", type=ParameterType.CATEGORICAL, choices=["a", "b"])
        ],
    )
    validate_params(cfg)  # should not raise


# --------------------------------------------------------------------------- #
# YAML benchmark file models
# --------------------------------------------------------------------------- #
def test_test_case_id_optional() -> None:
    tc = TestCase(prompt="Hello")
    assert tc.id is None
    assert tc.prompt == "Hello"


def test_load_benchmark_file_auto_ids(tmp_path: Path) -> None:
    yaml_text = """
tests:
  - prompt: "Hello"
  - prompt: "World"
"""
    path = tmp_path / "my-bench.yaml"
    path.write_text(yaml_text)
    suite, params = load_benchmark_file(path)
    assert suite.tests[0].id == "my-bench-test-1"
    assert suite.tests[1].id == "my-bench-test-2"
    assert params is None


def test_load_benchmark_file_explicit_ids_preserved(tmp_path: Path) -> None:
    yaml_text = """
optimizer:
  type: bayesian
  budget: 10
tests:
  - id: "custom-1"
    prompt: "Hello"
  - id: "custom-2"
    prompt: "World"
"""
    path = tmp_path / "my-bench.yaml"
    path.write_text(yaml_text)
    suite, params = load_benchmark_file(path)
    assert suite.tests[0].id == "custom-1"
    assert suite.tests[1].id == "custom-2"
    assert params is not None
    assert params.optimizer.type == "bayesian"


def test_load_benchmark_file_missing_optimizer_returns_none_params(tmp_path: Path) -> None:
    yaml_text = """
tests:
  - prompt: "Hello"
"""
    path = tmp_path / "my-bench.yaml"
    path.write_text(yaml_text)
    suite, params = load_benchmark_file(path)
    assert params is None
    assert len(suite.tests) == 1


def test_load_benchmark_file_missing_parameters_defaults_empty(tmp_path: Path) -> None:
    yaml_text = """
optimizer:
  type: bayesian
  budget: 10
tests:
  - prompt: "Hello"
"""
    path = tmp_path / "my-bench.yaml"
    path.write_text(yaml_text)
    suite, params = load_benchmark_file(path)
    assert params is not None
    assert params.parameters == []


def test_load_benchmark_file_duplicate_explicit_ids_raises(tmp_path: Path) -> None:
    yaml_text = """
tests:
  - id: "t1"
    prompt: "A"
  - id: "t1"
    prompt: "B"
"""
    path = tmp_path / "my-bench.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError) as exc:
        load_benchmark_file(path)
    assert "duplicate test id" in str(exc.value)


def test_load_benchmark_file_category_set_from_stem(tmp_path: Path) -> None:
    yaml_text = """
tests:
  - prompt: "Hello"
"""
    path = tmp_path / "my-category.yaml"
    path.write_text(yaml_text)
    suite, _ = load_benchmark_file(path)
    assert suite.categories == {"my-category-test-1": "my-category"}


def test_load_benchmark_file_validate_params_called(tmp_path: Path) -> None:
    yaml_text = """
optimizer:
  type: bayesian
  budget: 10
parameters:
  - name: temperature
    type: float
    low: 1.0
    high: 0.1
tests:
  - prompt: "Hello"
"""
    path = tmp_path / "my-bench.yaml"
    path.write_text(yaml_text)
    with pytest.raises(ValueError, match="low .* high"):
        load_benchmark_file(path)


def test_discover_benchmark_files_single_yaml(tmp_path: Path) -> None:
    path = tmp_path / "bench.yaml"
    path.write_text("")
    assert discover_benchmark_files(path) == [path]


def test_discover_benchmark_files_single_yml(tmp_path: Path) -> None:
    path = tmp_path / "bench.yml"
    path.write_text("")
    assert discover_benchmark_files(path) == [path]


def test_discover_benchmark_files_single_non_yaml_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "bench.txt"
    path.write_text("")
    assert discover_benchmark_files(path) == []


def test_discover_benchmark_files_directory(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("")
    (tmp_path / "b.yml").write_text("")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.yaml").write_text("")
    (tmp_path / "d.json").write_text("")
    result = discover_benchmark_files(tmp_path)
    assert result == [
        tmp_path / "a.yaml",
        tmp_path / "b.yml",
        sub / "c.yaml",
    ]


def test_discover_benchmark_files_sorted(tmp_path: Path) -> None:
    (tmp_path / "z.yaml").write_text("")
    (tmp_path / "a.yaml").write_text("")
    result = discover_benchmark_files(tmp_path)
    assert result == [tmp_path / "a.yaml", tmp_path / "z.yaml"]


def test_load_tests_missing_id_raises(tmp_path: Path) -> None:
    path = tmp_path / "tests.json"
    path.write_text(json.dumps([{"prompt": "Hello"}]))
    with pytest.raises(ValidationError) as exc:
        load_tests(path)
    assert "id" in str(exc.value).lower()


# --------------------------------------------------------------------------- #
# validate_params_match (2.1)
# --------------------------------------------------------------------------- #
def test_validate_params_match_both_none() -> None:
    validate_params_match(None, None)  # should not raise


def test_validate_params_match_one_none() -> None:
    cfg = ParamsConfig(
        optimizer=OptimizerConfig(type="bayesian"),
        parameters=[ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=1.0)],
    )
    validate_params_match(None, cfg)  # should not raise
    validate_params_match(cfg, None)  # should not raise


def test_validate_params_match_identical() -> None:
    cfg = ParamsConfig(
        optimizer=OptimizerConfig(type="bayesian", budget=20, baseline={"x": 1}),
        parameters=[ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=1.0, step=0.1, choices=None)],
        static_params={"a": 1},
    )
    validate_params_match(cfg, cfg)  # should not raise


def test_validate_params_match_optimizer_type_mismatch() -> None:
    a = ParamsConfig(optimizer=OptimizerConfig(type="bayesian"))
    b = ParamsConfig(optimizer=OptimizerConfig(type="grid"))
    with pytest.raises(ValueError, match="optimizer.type"):
        validate_params_match(a, b)


def test_validate_params_match_optimizer_budget_mismatch() -> None:
    a = ParamsConfig(optimizer=OptimizerConfig(budget=10))
    b = ParamsConfig(optimizer=OptimizerConfig(budget=20))
    with pytest.raises(ValueError, match="optimizer.budget"):
        validate_params_match(a, b)


def test_validate_params_match_optimizer_baseline_mismatch() -> None:
    a = ParamsConfig(optimizer=OptimizerConfig(baseline={"x": 1}))
    b = ParamsConfig(optimizer=OptimizerConfig(baseline={"x": 2}))
    with pytest.raises(ValueError, match="optimizer.baseline"):
        validate_params_match(a, b)


def test_validate_params_match_parameters_name_mismatch() -> None:
    a = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.FLOAT, low=0, high=1)],
    )
    b = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="b", type=ParameterType.FLOAT, low=0, high=1)],
    )
    with pytest.raises(ValueError, match="parameters\\[0\\].name"):
        validate_params_match(a, b)


def test_validate_params_match_parameters_type_mismatch() -> None:
    a = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.FLOAT, low=0, high=1)],
    )
    b = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.INT, low=0, high=1)],
    )
    with pytest.raises(ValueError, match="parameters\\[0\\].type"):
        validate_params_match(a, b)


def test_validate_params_match_parameters_low_mismatch() -> None:
    a = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.FLOAT, low=0, high=1)],
    )
    b = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.FLOAT, low=0.5, high=1)],
    )
    with pytest.raises(ValueError, match="parameters\\[0\\].low"):
        validate_params_match(a, b)


def test_validate_params_match_parameters_high_mismatch() -> None:
    a = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.FLOAT, low=0, high=1)],
    )
    b = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.FLOAT, low=0, high=2)],
    )
    with pytest.raises(ValueError, match="parameters\\[0\\].high"):
        validate_params_match(a, b)


def test_validate_params_match_parameters_step_mismatch() -> None:
    a = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.FLOAT, low=0, high=1, step=0.1)],
    )
    b = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.FLOAT, low=0, high=1, step=0.2)],
    )
    with pytest.raises(ValueError, match="parameters\\[0\\].step"):
        validate_params_match(a, b)


def test_validate_params_match_parameters_choices_mismatch() -> None:
    a = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.CATEGORICAL, choices=["x", "y"])],
    )
    b = ParamsConfig(
        optimizer=OptimizerConfig(),
        parameters=[ParameterSpec(name="a", type=ParameterType.CATEGORICAL, choices=["x", "z"])],
    )
    with pytest.raises(ValueError, match="parameters\\[0\\].choices"):
        validate_params_match(a, b)


def test_validate_params_match_static_params_mismatch() -> None:
    a = ParamsConfig(
        optimizer=OptimizerConfig(),
        static_params={"a": 1},
    )
    b = ParamsConfig(
        optimizer=OptimizerConfig(),
        static_params={"a": 2},
    )
    with pytest.raises(ValueError, match="static_params"):
        validate_params_match(a, b)


# --------------------------------------------------------------------------- #
# merge_params (2.2)
# --------------------------------------------------------------------------- #
def test_merge_params_overrides_field() -> None:
    base = ParamsConfig(
        optimizer=OptimizerConfig(type="bayesian", budget=20),
        parameters=[ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=1.0)],
        static_params={"a": 1},
    )
    override = ParamsConfig(
        optimizer=OptimizerConfig(type="grid", budget=10),
        parameters=[ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.2, high=1.0)],
        static_params={"a": 2, "b": 3},
    )
    merged = merge_params(base, override)
    assert merged.optimizer.type == "grid"
    assert merged.optimizer.budget == 10
    assert merged.parameters[0].low == 0.2
    assert merged.static_params == {"a": 2, "b": 3}

    # Base is unchanged
    assert base.optimizer.type == "bayesian"
    assert base.optimizer.budget == 20
    assert base.parameters[0].low == 0.1
    assert base.static_params == {"a": 1}


def test_merge_params_deepcopy_prevents_mutation_leak() -> None:
    base = ParamsConfig(
        optimizer=OptimizerConfig(type="bayesian"),
        parameters=[ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.1, high=1.0)],
        static_params={"a": [1, 2]},
    )
    override = ParamsConfig(
        optimizer=OptimizerConfig(type="grid"),
        parameters=[ParameterSpec(name="temperature", type=ParameterType.FLOAT, low=0.2, high=1.0)],
        static_params={"a": [3, 4]},
    )
    merged = merge_params(base, override)
    # Mutate merged nested fields
    merged.parameters[0].low = 99.0
    merged.static_params["a"].append(99)
    # Base must remain unchanged
    assert base.parameters[0].low == 0.1
    assert base.static_params["a"] == [1, 2]
