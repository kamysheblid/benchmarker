"""Tests for eval-file generation (Phase 7)."""

from pathlib import Path

from benchmarker.eval_file import generate_eval_md
from benchmarker.runner import RunResult, config_key


def _results() -> list[RunResult]:
    return [
        RunResult(
            config={"temperature": 0.7}, test_id="t1", repetition=1, prompt="Say hi",
            response_text="Hello!", ttft=0.05, total_time=0.5, tokens_per_sec=10.0,
            completion_tokens=2, prompt_tokens=3,
        ),
        RunResult(
            config={"temperature": 0.7}, test_id="t2", repetition=1, prompt="Count to 3",
            response_text="1 2 3", ttft=0.06, total_time=0.6, tokens_per_sec=9.0,
            completion_tokens=3, prompt_tokens=4,
        ),
        RunResult(
            config={"temperature": 1.0}, test_id="t1", repetition=1, prompt="Say hi",
            response_text="Hi there.", ttft=0.04, total_time=0.4, tokens_per_sec=12.0,
            completion_tokens=2, prompt_tokens=3,
        ),
    ]


def test_generate_eval_md_structure(tmp_path: Path) -> None:
    results = _results()
    out = tmp_path / "eval_output.md"
    generate_eval_md(tmp_path, results, out_path=out)
    text = out.read_text()

    # one section per unique config
    assert "temperature=0.7" in text or "temperature\": 0.7" in text
    # config key appears as heading
    key07 = config_key({"temperature": 0.7})
    key10 = config_key({"temperature": 1.0})
    assert key07 in text
    assert key10 in text
    # prompts and responses present
    assert "Say hi" in text
    assert "Hello!" in text
    assert "Count to 3" in text
    assert "1 2 3" in text
    assert "Hi there." in text
    # rating template block present
    assert "RATING" in text.upper() or "rating" in text.lower()
    assert "overall" in text.lower()


def test_generate_eval_md_rating_instructions(tmp_path: Path) -> None:
    results = _results()
    out = tmp_path / "eval_output.md"
    generate_eval_md(tmp_path, results, out_path=out)
    text = out.read_text().lower()
    # The instructions should ask the judge to produce a scores JSON keyed by
    # config / test_id / repetition.
    assert "config" in text
    assert "test_id" in text
    assert "json" in text
    assert "scores" in text
