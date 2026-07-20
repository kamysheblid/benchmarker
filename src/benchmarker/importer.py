"""Import judge scores and compute the final combined ranking (Phase 8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from benchmarker.runner import RAW_DATA_FILE, RunResult, config_key

RAW_DATA_FILE = RAW_DATA_FILE


class JudgeScoreEntry(BaseModel):
    """A single quality rating produced by the external judge."""

    model_config = ConfigDict(extra="forbid")

    config: str
    test_id: str
    repetition: int = 1
    scores: dict[str, float] = Field(default_factory=dict)


class FinalRankingItem(BaseModel):
    """Combined ranking row for one parameter configuration."""

    model_config = ConfigDict(extra="forbid")

    config: str
    avg_quality: float
    avg_speed: float
    norm_quality: float
    norm_speed: float
    combined: float


def _merge_key(config_str: str, test_id: str, rep: int) -> str:
    return f"{config_str}::{test_id}::{rep}"


def import_scores(
    run_dir: Path,
    scores_path: Path,
    weight_quality: float = 0.5,
) -> list[FinalRankingItem]:
    """Merge raw benchmark data with judge scores and rank configurations.

    Args:
        run_dir: Directory containing ``raw_data.json``.
        scores_path: Path to the judge's scores JSON array.
        weight_quality: Weight (0..1) for quality in the combined score.

    Returns:
        Configurations ranked by ``combined`` score, descending.

    Raises:
        FileNotFoundError: if either the raw data or scores file is missing.
    """
    run_dir = Path(run_dir)
    scores_path = Path(scores_path)

    raw_file = run_dir / RAW_DATA_FILE
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_file}")
    if not scores_path.exists():
        raise FileNotFoundError(f"Scores file not found: {scores_path}")

    raw_results = [RunResult(**r) for r in json.loads(raw_file.read_text(encoding="utf-8"))]
    score_entries = [JudgeScoreEntry(**s) for s in json.loads(scores_path.read_text(encoding="utf-8"))]

    # index scores by (config, test_id, rep)
    score_lookup: dict[str, list[float]] = {}
    for entry in score_entries:
        key = _merge_key(entry.config, entry.test_id, entry.repetition)
        score_lookup.setdefault(key, []).append(float(entry.scores.get("overall", 0.0)))

    # aggregate metrics per config
    per_config: dict[str, dict[str, list[float]]] = {}
    for r in raw_results:
        cfg_key = config_key(r.config)
        bucket = per_config.setdefault(cfg_key, {"quality": [], "speed": []})
        if r.error is None:
            bucket["speed"].append(r.tokens_per_sec)
        skey = _merge_key(cfg_key, r.test_id, r.repetition)
        if skey in score_lookup:
            bucket["quality"].extend(score_lookup[skey])

    rows: list[FinalRankingItem] = []
    for cfg_key, bucket in per_config.items():
        avg_quality = sum(bucket["quality"]) / len(bucket["quality"]) if bucket["quality"] else 0.0
        avg_speed = sum(bucket["speed"]) / len(bucket["speed"]) if bucket["speed"] else 0.0
        rows.append(
            FinalRankingItem(
                config=cfg_key,
                avg_quality=avg_quality,
                avg_speed=avg_speed,
                norm_quality=0.0,
                norm_speed=0.0,
                combined=0.0,
            )
        )

    _normalize(rows, weight_quality)
    rows.sort(key=lambda x: x.combined, reverse=True)
    return rows


def _normalize(rows: list[FinalRankingItem], weight_quality: float) -> None:
    qualities = [r.avg_quality for r in rows]
    speeds = [r.avg_speed for r in rows]
    q_min, q_max = (min(qualities), max(qualities)) if qualities else (0.0, 0.0)
    s_min, s_max = (min(speeds), max(speeds)) if speeds else (0.0, 0.0)

    for r in rows:
        r.norm_quality = _minmax(r.avg_quality, q_min, q_max)
        r.norm_speed = _minmax(r.avg_speed, s_min, s_max)
        r.combined = weight_quality * r.norm_quality + (1 - weight_quality) * r.norm_speed


def _minmax(value: float, lo: float, hi: float) -> float:
    if hi - lo == 0:
        return 1.0
    return (value - lo) / (hi - lo)
