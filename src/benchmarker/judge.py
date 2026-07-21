"""Judge client interface and NullJudge implementation.

Provides a minimal abstraction for judge automation so future auto-judge
backends can be added without schema changes.

Default ``NullJudge`` preserves the existing manual workflow:
generate a prompt file, parse the human reply, and act on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarker.eval_file import generate_judge_prompt
from benchmarker.parse_judge import parse_and_act

if TYPE_CHECKING:
    from benchmarker.parse_judge import JudgeVerdict
    from benchmarker.runner import RunResult


class JudgeClient(ABC):
    """Minimal interface for judge automation backends."""

    @abstractmethod
    def generate_prompt(
        self,
        run_dir: Path | str,
        results: list[Any],
        out_path: Path | None = None,
    ) -> Path:
        """Produce a judge prompt file and return its path."""

    @abstractmethod
    def parse_reply(
        self,
        text: str,
        params_path: Path | str | None = None,
        run_dir: Path | str | None = None,
    ) -> None:
        """Parse a judge reply and apply any side-effects."""

    @abstractmethod
    def act(
        self,
        verdict: "JudgeVerdict",
        params_path: Path | str,
    ) -> None:
        """Apply a parsed verdict to params.yaml (or no-op for manual path)."""


class NullJudge(JudgeClient):
    """Default manual judge: delegates to existing CLI-style helpers."""

    def generate_prompt(
        self,
        run_dir: Path | str,
        results: list[Any],
        out_path: Path | None = None,
    ) -> Path:
        path, _ = generate_judge_prompt(run_dir, results, out_path=out_path)
        return path

    def parse_reply(
        self,
        text: str,
        params_path: Path | str | None = None,
        run_dir: Path | str | None = None,
    ) -> None:
        parse_and_act(text, params_path=params_path, run_dir=run_dir)

    def act(
        self,
        verdict: "JudgeVerdict",
        params_path: Path | str,
    ) -> None:
        """Manual path: do nothing. The user already reviewed the verdict."""
