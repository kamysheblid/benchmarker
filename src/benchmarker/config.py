"""Configuration models and loaders for benchmarker (Phase 2).

Defines Pydantic models for the parameter search space (YAML) and the test
suite (JSON), plus loader functions. Validation logic is centralized here.
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

    type: Literal["bayesian", "grid", "random"] = "bayesian"
    budget: PositiveInt = 20


class ParamsConfig(BaseModel):
    """Top-level parameter search-space configuration."""

    model_config = ConfigDict(extra="forbid")

    optimizer: OptimizerConfig
    parameters: list[ParameterSpec] = Field(default_factory=list)


class TestCase(BaseModel):
    """A single benchmark prompt."""

    model_config = ConfigDict(extra="forbid")
    __test__ = False  # prevent pytest from collecting this Pydantic model as a test case

    id: str
    prompt: str
    system: str | None = None
    max_tokens: PositiveInt | None = None
    repeat: int = 1

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

    @field_validator("tests")
    @classmethod
    def _unique_ids(cls, tests: list[TestCase]) -> list[TestCase]:
        seen: set[str] = set()
        for tc in tests:
            if tc.id in seen:
                raise ValueError(f"duplicate test id: {tc.id!r}")
            seen.add(tc.id)
        return tests


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
    return ParamsConfig.model_validate(raw)


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
    # The on-disk format is a JSON array of test cases. If a wrapped object is
    # provided instead, normalize it to the array form.
    if isinstance(raw, dict):
        raw = raw.get("tests", [])
    return TestSuite(tests=raw)


def load_params_default() -> ParamsConfig:
    """Load the bundled default parameter search space."""
    return _load_bundled("params.default.yaml", load_params)


def load_tests_default() -> TestSuite:
    """Load the bundled default test suite."""
    return _load_bundled("tests.default.json", load_tests)


def _load_bundled(name: str, loader: Callable[[Path], Any]) -> Any:
    from importlib import resources

    try:
        ref = resources.files("benchmarker.defaults").joinpath(name)
        with resources.as_file(ref) as path:
            return loader(path)
    except (ModuleNotFoundError, FileNotFoundError, ValidationError) as exc:
        raise FileNotFoundError(f"Bundled default {name!r} could not be loaded: {exc}") from exc
