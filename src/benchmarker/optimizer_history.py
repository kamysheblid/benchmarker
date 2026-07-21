"""Optimizer history persistence for JSON replay (Phase 6)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class OptimizerTrial:
    """A single recorded optimizer trial."""

    params: dict[str, Any]
    quality: float | None = None
    tokens_per_sec: float = 0.0


class OptimizerHistory:
    """In-memory history that can be saved to / loaded from JSON."""

    def __init__(self) -> None:
        self.trials: list[OptimizerTrial] = []

    def add_trial(self, trial: OptimizerTrial) -> None:
        self.trials.append(trial)

    def save(self, path: Path) -> None:
        payload = [
            {"params": t.params, "quality": t.quality, "tokens_per_sec": t.tokens_per_sec}
            for t in self.trials
        ]
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> OptimizerHistory:
        history = cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        for entry in data:
            history.add_trial(
                OptimizerTrial(
                    params=entry.get("params", {}),
                    quality=entry.get("quality"),
                    tokens_per_sec=entry.get("tokens_per_sec", 0.0),
                )
            )
        return history
