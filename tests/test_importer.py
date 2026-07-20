"""Tests for score import and final ranking (Phase 8)."""

import json
from pathlib import Path

import pytest

from benchmarker.importer import FinalRankingItem, JudgeScoreEntry, import_scores
from benchmarker.runner import RunResult, config_key


def _write_raw(run_dir: Path) -> list[RunResult]:
    configs = [{"temperature": 0.1}, {"temperature": 0.9}]
    results = [
        RunResult(config=configs[0], test_id="t1", repetition=1, prompt="p",
                  response_text="slow but good", ttft=0.1, total_time=1.0,
                  tokens_per_sec=5.0, completion_tokens=5, prompt_tokens=2),
        RunResult(config=configs[1], test_id="t1", repetition=1, prompt="p",
                  response_text="fast but meh", ttft=0.05, total_time=0.4,
                  tokens_per_sec=20.0, completion_tokens=5, prompt_tokens=2),
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    run_dir.joinpath("raw_data.json").write_text(
        json.dumps([r.model_dump() for r in results], default=str)
    )
    return results


def _write_scores(run_dir: Path, configs: list[dict]) -> Path:
    scores = [
        JudgeScoreEntry(config=config_key(configs[0]), test_id="t1", repetition=1,
                        scores={"overall": 9.0}),
        JudgeScoreEntry(config=config_key(configs[1]), test_id="t1", repetition=1,
                        scores={"overall": 3.0}),
    ]
    path = run_dir / "scores.json"
    path.write_text(json.dumps([s.model_dump() for s in scores], default=str))
    return path


def test_import_scores_merge_and_rank(tmp_path: Path) -> None:
    configs = [{"temperature": 0.1}, {"temperature": 0.9}]
    run_dir = tmp_path / "run"
    _write_raw(run_dir)
    scores_path = _write_scores(run_dir, configs)

    ranking = import_scores(run_dir, scores_path, weight_quality=0.5)
    assert len(ranking) == 2
    assert all(isinstance(r, FinalRankingItem) for r in ranking)

    by_cfg = {r.config: r for r in ranking}
    slow = by_cfg[config_key(configs[0])]
    fast = by_cfg[config_key(configs[1])]
    # quality: slow=9 (best => norm 1.0), fast=3 (worst => norm 0.0)
    assert slow.norm_quality == pytest.approx(1.0)
    assert fast.norm_quality == pytest.approx(0.0)
    # speed: slow=5 (worst => 0.0), fast=20 (best => 1.0)
    assert slow.norm_speed == pytest.approx(0.0)
    assert fast.norm_speed == pytest.approx(1.0)
    # combined at w=0.5
    assert slow.combined == pytest.approx(0.5)
    assert fast.combined == pytest.approx(0.5)
    # ranking sorted by combined desc; tie -> order stable
    assert ranking[0].combined >= ranking[1].combined


def test_import_scores_weight_favors_quality(tmp_path: Path) -> None:
    configs = [{"temperature": 0.1}, {"temperature": 0.9}]
    run_dir = tmp_path / "run"
    _write_raw(run_dir)
    scores_path = _write_scores(run_dir, configs)

    ranking = import_scores(run_dir, scores_path, weight_quality=1.0)
    by_cfg = {r.config: r for r in ranking}
    # with quality-only weight, slow (quality 1.0) beats fast (0.0)
    assert by_cfg[config_key(configs[0])].combined == pytest.approx(1.0)
    assert ranking[0].config == config_key(configs[0])


def test_import_scores_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        import_scores(tmp_path / "run", tmp_path / "missing.json")


def test_import_scores_handles_unmatched_scores(tmp_path: Path) -> None:
    configs = [{"temperature": 0.1}, {"temperature": 0.9}]
    run_dir = tmp_path / "run"
    _write_raw(run_dir)
    # only score for config[0]; config[1] has no quality -> treated as 0
    scores = [JudgeScoreEntry(config=config_key(configs[0]), test_id="t1",
                              repetition=1, scores={"overall": 8.0})]
    path = run_dir / "scores.json"
    path.write_text(json.dumps([s.model_dump() for s in scores], default=str))

    ranking = import_scores(run_dir, path, weight_quality=0.5)
    by_cfg = {r.config: r for r in ranking}
    # config[1] has no score -> quality 0, speed best (1.0)
    assert by_cfg[config_key(configs[1])].norm_quality == pytest.approx(0.0)
    assert by_cfg[config_key(configs[1])].norm_speed == pytest.approx(1.0)
