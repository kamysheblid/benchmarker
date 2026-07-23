"""Configuration models and loaders for benchmarker (Phase 2).

Defines Pydantic models for the parameter search space (YAML) and the test
suite (JSON), plus loader functions. Validation logic is centralized here.
"""

from __future__ import annotations

import copy
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


# --------------------------------------------------------------------------- #
# Post-load helpers for YAML benchmark files
# --------------------------------------------------------------------------- #
def _unique_ids(tests: list[TestCase], source_name: str) -> list[TestCase]:
    """Validate that explicit IDs are unique, skipping ``None`` entries."""
    seen: set[str] = set()
    for tc in tests:
        if tc.id is None:
            continue
        if tc.id in seen:
            raise ValueError(f"duplicate test id: {tc.id!r} in {source_name}")
        seen.add(tc.id)
    return tests


def _autoid(tests: list[TestCase], stem: str) -> list[TestCase]:
    """Assign auto-generated IDs to ``None`` entries."""
    for i, tc in enumerate(tests, start=1):
        if tc.id is None:
            tc.id = f"{stem}-test-{i}"
    return tests


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
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


def validate_params_match(a: ParamsConfig | None, b: ParamsConfig | None) -> None:
    """Validate that two ``ParamsConfig`` objects are compatible for merging.

    Checks:
    - ``optimizer.type``
    - ``optimizer.budget``
    - ``optimizer.baseline``
    - ``parameters`` — compared by ``name``, ``type``, ``low``, ``high``,
      ``step``, and ``choices``
    - ``static_params`` — deep equality

    If either argument is ``None`` the check passes (None means "use
    defaults").

    Raises:
        ValueError: describing the first mismatched field found.
    """
    if a is None or b is None:
        return

    def _eq(left: Any, right: Any, path: str) -> None:
        if left != right:
            raise ValueError(f"{path}: {left!r} != {right!r}")

    _eq(a.optimizer.type, b.optimizer.type, "optimizer.type")
    _eq(a.optimizer.budget, b.optimizer.budget, "optimizer.budget")
    _eq(a.optimizer.baseline, b.optimizer.baseline, "optimizer.baseline")

    if len(a.parameters) != len(b.parameters):
        raise ValueError(
            f"parameters length mismatch: {len(a.parameters)} != {len(b.parameters)}"
        )
    for idx, (pa, pb) in enumerate(zip(a.parameters, b.parameters)):
        prefix = f"parameters[{idx}]"
        _eq(pa.name, pb.name, f"{prefix}.name")
        _eq(pa.type, pb.type, f"{prefix}.type")
        _eq(pa.low, pb.low, f"{prefix}.low")
        _eq(pa.high, pb.high, f"{prefix}.high")
        _eq(pa.step, pb.step, f"{prefix}.step")
        _eq(pa.choices, pb.choices, f"{prefix}.choices")

    _eq(a.static_params, b.static_params, "static_params")


def merge_params(base: ParamsConfig, override: ParamsConfig) -> ParamsConfig:
    """Merge *override* on top of *base*, returning a new ``ParamsConfig``.

    All top-level fields from *override* replace the corresponding fields
    in *base*. Nested dicts/lists are deep-copied so mutations on the
    result do not leak back to either input.

    Raises:
        ValueError: if the two configs are not compatible (use
            :func:`validate_params_match` first).
    """
    return ParamsConfig(
        optimizer=copy.deepcopy(override.optimizer),
        parameters=[copy.deepcopy(p) for p in override.parameters],
        static_params=copy.deepcopy(override.static_params),
    )


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


def load_benchmark_file(path: Path) -> tuple[TestSuite, ParamsConfig | None]:
    """Load a YAML benchmark file containing test cases and optional params.

    The expected YAML shape is::

        optimizer:
          type: bayesian
          budget: 20
        parameters: [...]
        static_params: {...}
        tests:
          - prompt: "..."
            # id optional, repeat, max_tokens, reasoning, system

    If the ``optimizer`` key is absent entirely, ``ParamsConfig`` is returned
    as ``None`` (signals fallback to bundled defaults).

    If ``optimizer`` is present but ``parameters`` is absent, it is treated as
    ``parameters: []``.

    After building ``ParamsConfig``, ``validate_params(config)`` is invoked to
    catch bad ranges before the optimizer sees them.

    Test cases without an explicit ``id`` are auto-numbered using the pattern
    ``<file-stem>-test-<N>``. The filename stem also becomes the default
    category for all tests in the file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValidationError: if the content is invalid or params fail validation.
        ValueError: if duplicate test IDs are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # Build params if optimizer is present
    params: ParamsConfig | None = None
    if "optimizer" in raw:
        params_data = {k: raw[k] for k in ("optimizer", "parameters", "static_params") if k in raw}
        if "parameters" not in params_data:
            params_data["parameters"] = []
        params = ParamsConfig.model_validate(params_data)
        validate_params(params)

    # Extract tests
    tests_raw = raw.get("tests", [])
    if not isinstance(tests_raw, list):
        tests_raw = [tests_raw]

    suite = TestSuite(tests=tests_raw)
    stem = path.stem

    # Pass 1: validate explicit IDs are unique, skip None
    _unique_ids(suite.tests, stem)
    # Pass 2: assign auto-ids to None entries
    _autoid(suite.tests, stem)
    # Pass 3: final uniqueness check (now all IDs should be present)
    _unique_ids(suite.tests, stem)

    # Default category mapping
    suite.categories = {tc.id: stem for tc in suite.tests}

    return suite, params


def discover_benchmark_files(path: Path) -> list[Path]:
    """Return sorted ``.yaml``/``.yml`` benchmark files under *path*.

    - If *path* is a file, return ``[path]`` when it ends in ``.yaml`` or
      ``.yml``, otherwise ``[]``.
    - If *path* is a directory, recursively collect all ``*.yaml`` and
      ``*.yml`` files and return them in sorted order.
    - For any other path, return ``[]``.
    """
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in {".yaml", ".yml"}:
            return [path]
        return []
    if not path.is_dir():
        return []
    results: list[Path] = []
    results.extend(path.rglob("*.yaml"))
    results.extend(path.rglob("*.yml"))
    return sorted(set(results))


def load_tests(path: Path) -> TestSuite:
    """Load and validate a test-suite JSON file.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValidationError: if the content is invalid.
        ValueError: if duplicate or missing test IDs are found.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test suite not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    # The on-disk format is a JSON array of test cases. If a wrapped object is
    # provided instead, normalize it to the array form.
    if isinstance(raw, dict):
        raw = raw.get("tests", [])

    tests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw:
        tc_id = item.get("id")
        if not isinstance(tc_id, str) or not tc_id.strip():
            raise ValidationError.from_exception_data(
                "TestCase",
                [
                    {
                        "type": "value_error",
                        "loc": ("id",),
                        "input": tc_id,
                        "ctx": {"error": "test case requires 'id'"},
                    }
                ],
            )
        if tc_id in seen:
            raise ValueError(f"duplicate test id: {tc_id!r}")
        seen.add(tc_id)
        tests.append(item)

    return TestSuite(tests=tests)


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
    """Load benchmark prompts from a category directory structure.

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
                if tc_id in seen:
                    raise ValueError(f"duplicate test id: {tc_id!r}")
                seen.add(tc_id)
                tests.append(item)
                category_map[tc_id] = category

    return TestSuite(tests=tests, categories=category_map)


def load_params_default() -> ParamsConfig:
    """Load the bundled default parameter search space."""
    config = _load_bundled("params.default.yaml", load_params)
    return config


def load_tests_default() -> TestSuite:
    """Load the bundled default test suite."""
    from importlib import resources

    ref = resources.files("benchmarker.defaults").joinpath("benchmarks")
    with resources.as_file(ref) as path:
        return load_tests_from_dir(path)


def _load_bundled(name: str, loader: Callable[[Path], Any]) -> Any:
    from importlib import resources

    try:
        ref = resources.files("benchmarker.defaults").joinpath(name)
        with resources.as_file(ref) as path:
            return loader(path)
    except (ModuleNotFoundError, FileNotFoundError, ValidationError) as exc:
        raise FileNotFoundError(f"Bundled default {name!r} could not be loaded: {exc}") from exc
