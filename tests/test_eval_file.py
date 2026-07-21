"""Tests for judge-prompt generation (Phase 7)."""

from pathlib import Path

from benchmarker.eval_file import generate_judge_prompt
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


def test_generate_judge_prompt_structure(tmp_path: Path) -> None:
    results = _results()
    out = tmp_path / "judge_prompt.md"
    generate_judge_prompt(tmp_path, results, out_path=out)
    text = out.read_text()

    # one section per unique config (now using short IDs like config_1)
    assert "temperature=0.7" in text or "temperature\": 0.7" in text
    # short IDs appear in headings and summary table
    assert "config_1" in text
    assert "config_2" in text
    # response previews present
    assert "Hello!" in text
    assert "1 2 3" in text
    assert "Hi there." in text
    # Config summary table present
    assert "## Config Summary" in text
    assert "Config ID" in text and "Avg Tok/s" in text
    # Category grouping present (defaults to general when no category set)
    assert "general" in text


def test_generate_judge_prompt_instructions(tmp_path: Path) -> None:
    results = _results()
    out = tmp_path / "judge_prompt.md"
    generate_judge_prompt(tmp_path, results, out_path=out)
    text = text = out.read_text().lower()
    # The instructions should ask the judge for a JSON response
    assert "recommendation" in text
    assert "scores" in text
    assert "conclude" in text
    assert "refine" in text
    assert "expand" in text
    assert "json" in text


def _category_results() -> list[RunResult]:
    return [
        RunResult(
            config={"temperature": 0.7}, test_id="t1", repetition=1, prompt="Say hi",
            response_text="Hello!", ttft=0.05, total_time=0.5, tokens_per_sec=10.0,
            completion_tokens=2, prompt_tokens=3, category="code-generation",
        ),
        RunResult(
            config={"temperature": 0.7}, test_id="t2", repetition=1, prompt="Fix bug",
            response_text="Fixed.", ttft=0.06, total_time=0.6, tokens_per_sec=9.0,
            completion_tokens=3, prompt_tokens=4, category="bug-fixing",
        ),
        RunResult(
            config={"temperature": 1.0}, test_id="t1", repetition=1, prompt="Say hi",
            response_text="Hi there.", ttft=0.04, total_time=0.4, tokens_per_sec=12.0,
            completion_tokens=2, prompt_tokens=3, category="code-generation",
        ),
    ]


def test_judge_prompt_groups_by_category(tmp_path: Path) -> None:
    results = _category_results()
    out = tmp_path / "judge_prompt.md"
    generate_judge_prompt(tmp_path, results, out_path=out)
    text = out.read_text()

    # Category headings should appear in the detailed section
    assert "code-generation" in text
    assert "bug-fixing" in text
    # Should not list every individual test in the detailed section;
    # instead it should summarize per category
    assert "### Test `t1`" not in text
    assert "### Test `t2`" not in text


def test_judge_prompt_excludes_errored_runs_from_details(tmp_path: Path) -> None:
    results = _category_results() + [
        RunResult(
            config={"temperature": 0.7}, test_id="t3", repetition=1, prompt="Fails",
            response_text="", ttft=0.0, total_time=0.0, tokens_per_sec=0.0,
            completion_tokens=0, prompt_tokens=0,
            error="Request to LLM endpoint failed", category="code-generation",
        ),
    ]
    out = tmp_path / "judge_prompt.md"
    generate_judge_prompt(tmp_path, results, out_path=out)
    text = out.read_text()

    # Error run should NOT appear in the detailed responses
    assert "Fails" not in text
    # But a summary line about errors should exist
    assert "failed with endpoint errors" in text or "Errors" in text
    # The config summary table should show 1 error
    assert "| 1 |" in text


def test_judge_instructions_contain_category_slugs(tmp_path: Path) -> None:
    results = _category_results()
    out = tmp_path / "judge_prompt.md"
    generate_judge_prompt(tmp_path, results, out_path=out)
    text = out.read_text()

    assert "code-generation" in text
    assert "bug-fixing" in text


def test_judge_instructions_contain_endpoint_error_clause(tmp_path: Path) -> None:
    results = _category_results()
    out = tmp_path / "judge_prompt.md"
    generate_judge_prompt(tmp_path, results, out_path=out)
    text = out.read_text()

    assert "endpoint error" in text.lower()
    assert "penalise" in text.lower() or "penalize" in text.lower()


def test_judge_instructions_use_per_category(tmp_path: Path) -> None:
    results = _category_results()
    out = tmp_path / "judge_prompt.md"
    generate_judge_prompt(tmp_path, results, out_path=out)
    text = out.read_text()

    assert "per_category" in text
    assert "per_task" not in text


def test_judge_instructions_generic_placeholder_when_no_categories(tmp_path: Path) -> None:
    results = [
        RunResult(
            config={"temperature": 0.7}, test_id="t1", repetition=1, prompt="hi",
            response_text="hello", ttft=0.1, total_time=0.2, tokens_per_sec=10.0,
            completion_tokens=1, prompt_tokens=1,
        ),
    ]
    out = tmp_path / "judge_prompt.md"
    generate_judge_prompt(tmp_path, results, out_path=out)
    text = out.read_text()

    assert "per-category" in text
