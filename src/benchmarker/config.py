"""Configuration models and loaders for benchmarker (Phase 2 + YAML benchmarks).

Defines Pydantic models for the parameter search space (YAML) and the test
suite (JSON / YAML), plus loader functions. Validation logic is centralized
here.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError

try:  # pydantic v2
    from pydantic import field_validator
except ImportError:  # pragma: no cover - only for very old versions
    field_validator = None  # type: ignore[assignment]


class ParameterType(str, Enum):
    """Supported sampling parameter types."""

    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"


class ParameterSpec(BaseModel):
    """Specification of a single searchable sampling parameter."""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: ParameterType
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: list[Any] | None = None

    @field_validator("choices", mode="before")
    @classmethod
    def _empty_choices_to_none(cls, v: Any) -> Any:
        if v is None:
            return None
        return v

    def model_post_init(self, _context: Any) -> None:
        if self.type is ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValidationError.from_exception_data(
                    self.__class__.__name__,
                    [
                        {
                            "type": "value_error",
                            "loc": ("choices",),
                            "input": self.choices,
                            "ctx": {"error": "categorical parameters require 'choices'"},
                        }
                    ],
                )
        else:
            if self.low is None or self.high is None:
                raise ValidationError.from_exception_data(
                    self.__class__.__name__,
                    [
                        {
                            "type": "value_error",
                            "loc": ("low", "high"),
                            "input": (self.low, self.high),
                            "ctx": {"error": "numeric parameters require 'low' and 'high'"},
                        }
                    ],
                )


class OptimizerConfig(BaseModel):
    """Configuration of the search strategy."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["bayesian", "grid", "random", "baseline_sweep"] = "bayesian"
    budget: PositiveInt = 20
    # Baseline parameter values for "baseline_sweep" (ablation) studies.
    # When set, the sweep varies one parameter at a time while holding the
    # others at these baseline values.
    baseline: dict[str, Any] = Field(default_factory=dict)


class ParamsConfig(BaseModel):
    """Top-level parameter search-space configuration."""

    model_config = ConfigDict(extra="forbid")

    optimizer: OptimizerConfig
    parameters: list[ParameterSpec] = Field(default_factory=list)
    # Fixed params sent with every request (merged under sampled params).
    # Useful for reasoning models, e.g. {"chat_template_kwargs":
    # {"enable_thinking": False}} so the token budget is spent on the
    # answer rather than the thinking trace.
    static_params: dict[str, Any] = Field(default_factory=dict)


class TestCase(BaseModel):
    """A single benchmark prompt."""

    model_config = ConfigDict(extra="forbid")
    __test__ = False  # prevent pytest from collecting this Pydantic model as a test case

    id: str | None = None
    prompt: str
    system: str | None = None
    max_tokens: PositiveInt | None = None
    repeat: int = 1
    reasoning: bool | None = None  # True = encourage CoT, False = discourage, None = default

    def model_post_init(self, _context: Any) -> None:
        if not self.prompt.strip():
            raise ValidationError.from_exception_data(
                self.__class__.__name__,
                [
                    {
                        "type": "value_error",
                        "loc": ("prompt",),
                        "input": self.prompt,
                        "ctx": {"error": "prompt must not be empty"},
                    }
                ],
            )


class TestSuite(BaseModel):
    """A collection of test cases."""

    model_config = ConfigDict(extra="forbid")
    __test__ = False  # prevent pytest from collecting this Pydantic model as a test suite

    tests: list[TestCase] = Field(default_factory=list)
    categories: dict[str, str] = Field(default_factory=dict)
    params: ParamsConfig | None = None


def validate_params(config: ParamsConfig) -> None:
    """Validate the search space before any benchmark starts.

    Checks:
    - ``low <= high`` for every numeric parameter.
    - ``step > 0`` when ``step`` is provided.
    - ``step <= high - low`` when ``step`` is provided.
    - ``budget >= 1``.

    Raises:
        ValueError: with an actionable message on the first violation found.
    """
    if config.optimizer.budget < 1:
        raise ValueError(f"budget must be >= 1, got {config.optimizer.budget}")
    for spec in config.parameters:
        if spec.type is ParameterType.CATEGORICAL:
            continue
        low = spec.low
        high = spec.high
        if low is None or high is None:
            continue
        if low > high:
            raise ValueError(
                f"parameter '{spec.name}': low ({low}) must be <= high ({high})"
            )
        if spec.step is not None:
            step = spec.step
            if step <= 0:
                raise ValueError(
                    f"parameter '{spec.name}': step must be > 0, got {step}"
                )
            range_size = high - low
            if step > range_size:
                raise ValueError(
                    f"parameter '{spec.name}': step ({step}) must be <= "
                    f"high - low ({range_size})"
                )


# --------------------------------------------------------------------------- #
# Legacy loaders (tests.json)
# --------------------------------------------------------------------------- #
def load_params(path: Path) -> ParamsConfig:
    """Load and validate a parameter search-space YAML file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValidationError: if the content is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parameter config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = ParamsConfig.model_validate(raw)
    validate_params(config)
    return config


def load_tests(path: Path) -> TestSuite:
    """Load and validate a test-suite JSON file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValidationError: if the content is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test suite not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("tests", [])
    # Legacy format requires explicit ids.
    seen: set[str] = set()
    for item in raw:
        tc_id = item.get("id")
        if tc_id is None:
            raise ValueError("missing test id in legacy JSON file")
        if tc_id in seen:
            raise ValueError(f"duplicate test id: {tc_id!r}")
        seen.add(tc_id)
    return TestSuite(tests=raw)


def discover_categories(path: Path) -> list[str]:
    """Return sorted category slugs found under *path*.

    A category is any immediate subdirectory of *path*.
    """
    path = Path(path)
    if not path.is_dir():
        return []
    return sorted(
        p.name for p in path.iterdir() if p.is_dir()
    )


def load_tests_from_dir(path: Path, categories: list[str] | None = None) -> TestSuite:
    """Load benchmark prompts from a legacy category directory structure.

    Each immediate subdirectory of *path* is treated as a category slug.
    JSON files are searched recursively within each category directory so
    subdirectories (e.g. ``forager/implementation/``) are supported.
    Files are processed in sorted order so numeric prefixes control
    sequencing.

    Empty category directories are skipped silently.

    Args:
        path: Root directory containing category subdirectories.
        categories: Optional allow-list of category slugs (exact, case-sensitive).

    Returns:
        A :class:`TestSuite` containing every loaded test case.

    Raises:
        ValueError: If duplicate test IDs are found across categories.
        ValidationError: If any JSON file is not a valid :class:`TestCase`.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Benchmark directory not found: {path}")

    if categories is not None:
        available = set(discover_categories(path))
        unknown = set(categories) - available
        if unknown:
            raise ValueError(f"unknown category(s): {', '.join(sorted(unknown))}")

    seen: set[str] = set()
    tests: list[dict[str, Any]] = []
    category_map: dict[str, str] = {}

    for category in discover_categories(path):
        if categories is not None and category not in categories:
            continue
        category_dir = path / category
        json_files = sorted(category_dir.rglob("*.json"))
        if not json_files:
            continue
        for json_file in json_files:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
            # Support wrapped {"tests": [...]} or bare test-case dict/list
            if isinstance(raw, dict):
                items = raw.get("tests", [raw])
            else:
                items = raw
            for item in items:
                tc_id = item.get("id")
                if tc_id is None:
                    raise ValueError("missing test id in legacy JSON file")
                if tc_id in seen:
                    raise ValueError(f"duplicate test id: {tc_id!r}")
                seen.add(tc_id)
                tests.append(item)
                category_map[tc_id] = category

    return TestSuite(tests=tests, categories=category_map)


# --------------------------------------------------------------------------- #
# YAML benchmark file helpers
# --------------------------------------------------------------------------- #
def _unique_ids(tests: list[TestCase]) -> list[TestCase]:
    """Validate that explicit ids are unique (skip None ids)."""
    seen: set[str] = set()
    for tc in tests:
        if tc.id is not None:
            if tc.id in seen:
                raise ValueError(f"duplicate test id: {tc.id!r}")
            seen.add(tc.id)
    return tests


def _autoid(tests: list[TestCase], stem: str) -> list[TestCase]:
    """Assign auto-generated ids to entries where id is None."""
    counter = 1
    for tc in tests:
        if tc.id is None:
            tc.id = f"{stem}-test-{counter}"
            counter += 1
    return tests


def load_benchmark_file(path: Path) -> tuple[TestSuite, ParamsConfig | None]:
    """Load a self-contained YAML benchmark file.

    Expected shape:

    ```yaml
    optimizer:
      type: bayesian
      budget: 20
    parameters:
      - name: temperature
        type: float
        low: 0.0
        high: 1.5
        step: 0.1
    static_params:
      key: value
    tests:
      - id: optional
        prompt: "..."
        # system, max_tokens, repeat, reasoning
    ```

    Returns a ``(TestSuite, ParamsConfig | None)`` tuple. ``params`` is ``None``
    when the ``optimizer`` key is absent entirely (signals fallback to bundled
    defaults).

    Raises:
        FileNotFoundError: if the file does not exist.
        ValidationError: if the content is invalid.
        ValueError: if duplicate or missing ids are detected.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # Extract top-level keys
    optimizer_data = raw.get("optimizer")
    parameters_data = raw.get("parameters", [])
    static_params = raw.get("static_params", {})
    tests_data = raw.get("tests", [])

    # Normalize tests to a list
    if isinstance(tests_data, dict):
        tests_data = tests_data.get("tests", [tests_data])

    # Build ParamsConfig if optimizer is present
    params: ParamsConfig | None = None
    if optimizer_data is not None:
        params_payload = {
            "optimizer": optimizer_data,
            "parameters": parameters_data,
            "static_params": static_params,
        }
        params = ParamsConfig.model_validate(params_payload)
        validate_params(params)

    # Build and validate tests
    tests = [TestCase.model_validate(item) for item in tests_data]
    tests = _unique_ids(tests)
    stem = path.stem
    tests = _autoid(tests, stem)

    # Final verification that all ids are unique after auto-id
    seen: set[str] = set()
    category_map: dict[str, str] = {}
    for tc in tests:
        if tc.id in seen:
            raise ValueError(f"duplicate test id: {tc.id!r}")
        seen.add(tc.id)
        category_map[tc.id] = stem

    return TestSuite(tests=tests, categories=category_map), params


def discover_benchmark_files(path: Path) -> list[Path]:
    """Return sorted benchmark YAML files under *path*.

    - If *path* is a file, return ``[path]`` when it ends in ``.yaml`` or
      ``.yml``, otherwise ``[]``.
    - If *path* is a directory, recursively collect ``**/*.yaml`` and
      ``**/*.yml`` files.
    """
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in (".yaml", ".yml"):
            return [path]
        return []
    if not path.is_dir():
        return []
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in (".yaml", ".yml")
    )


# --------------------------------------------------------------------------- #
# Bundled defaults
# --------------------------------------------------------------------------- #
def load_params_default() -> ParamsConfig:
    """Load the bundled default parameter search space."""
    config = _load_bundled("params.default.yaml", load_params)
    return config


def load_tests_default() -> TestSuite:
    """Load the bundled default test suite from self-contained YAML files."""
    from importlib import resources

    ref = resources.files("benchmarker.defaults").joinpath("benchmarks")
    with resources.as_file(ref) as path:
        files = discover_benchmark_files(path)
        merged_tests: list[TestCase] = []
        merged_categories: dict[str, str] = {}
        for f in files:
            suite, _ = load_benchmark_file(f)
            merged_tests.extend(suite.tests)
            merged_categories.update(suite.categories)
        # Final uniqueness check across all defaults
        seen: set[str] = set()
        for tc in merged_tests:
            if tc.id in seen:
                raise ValueError(f"duplicate test id: {tc.id!r}")
            seen.add(tc.id)
        return TestSuite(tests=merged_tests, categories=merged_categories)


def _load_bundled(name: str, loader: Callable[[Path], Any]) -> Any:
    from importlib import resources

    try:
        ref = resources.files("benchmarker.defaults").joinpath(name)
        with resources.as_file(ref) as path:
            return loader(path)
    except (ModuleNotFoundError, FileNotFoundError, ValidationError) as exc:
        raise FileNotFoundError(f"Bundled default {name!r} could not be loaded: {exc}") from exc
