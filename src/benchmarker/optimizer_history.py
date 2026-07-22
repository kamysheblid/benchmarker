"""Optimizer history persistence for JSON replay (Phase 6)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class OptimizerTrial:
    """Single recorded trial for an optimizer history file."""

    params: dict[str, Any]
    quality: float | None = None
    tokens_per_sec: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class OptimizerHistory:
    """In-memory collection of trials that can be serialized to/from JSON."""

    def __init__(self, trials: list[OptimizerTrial] | None = None) -> None:
        self.trials: list[OptimizerTrial] = list(trials or [])

    def add_trial(self, trial: OptimizerTrial) -> None:
        """Append a new trial to the history."""
        self.trials.append(trial)

    def to_json(self, path: str | Path, top_k: int | None = None) -> None:
        """Save the history to a JSON file, optionally keeping only the top-K trials.

        When ``top_k`` is provided, trials are sorted by quality/speed descending
        and only the best ``top_k`` are persisted. When ``top_k`` is ``None``
        (default), all trials are saved in insertion order for backward
        compatibility.
        """
        path = Path(path)
        if top_k is not None:
            sorted_trials = sorted(
                self.trials,
                key=lambda t: (
                    t.quality if t.quality is not None else float("-inf"),
                    t.tokens_per_sec if t.tokens_per_sec is not None else float("-inf"),
                ),
                reverse=True,
            )
            kept = sorted_trials[:top_k]
        else:
            kept = list(self.trials)
        payload = [
            {
                "params": trial.params,
                "quality": trial.quality,
                "tokens_per_sec": trial.tokens_per_sec,
                "metadata": trial.metadata,
            }
            for trial in kept
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> OptimizerHistory:
        """Load a history from a JSON file."""
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(
                f"Invalid optimizer history file: expected a JSON array, got {type(raw).__name__}"
            )
        trials = [
            OptimizerTrial(
                params=item.get("params", {}),
                quality=item.get("quality"),
                tokens_per_sec=item.get("tokens_per_sec"),
                metadata=item.get("metadata", {}),
            )
            for item in raw
        ]
        return cls(trials=trials)

    def replay_into(self, optimizer: Any) -> None:
        """Replay every trial by calling ``optimizer.tell()``."""
        for trial in self.trials:
            metrics = {
                "tokens_per_sec": trial.tokens_per_sec,
                "quality": trial.quality,
            }
            if hasattr(optimizer, "study"):
                try:
                    optimizer.suggest()
                except StopIteration:
                    logger.warning(
                        "history truncated: %d trials exceed optimizer budget %d",
                        len(self.trials),
                        getattr(optimizer, "budget", "?"),
                    )
                    break
            optimizer.tell(metrics)
