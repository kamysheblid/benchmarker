"""Tests for bundled defaults (Phase 10)."""

from pathlib import Path

from click.testing import CliRunner

from benchmarker.config import load_params_default, load_tests_default


def test_default_params_load() -> None:
    params = load_params_default()
    assert params.optimizer.type == "bayesian"
    assert len(params.parameters) >= 3
    names = {p.name for p in params.parameters}
    assert "temperature" in names


def test_default_tests_load() -> None:
    suite = load_tests_default()
    # bundled defaults contain exactly 13 prompts across 8 categories
    assert len(suite.tests) == 13
    ids = {t.id for t in suite.tests}
    assert ids == {
        "creative",
        "reasoning",
        "factual",
        "coding_chunk",
        "algorithmic_palindrome",
        "algorithmic_twosum",
        "algorithmic_bst",
        "bugfixing",
        "refactoring",
        "explanation_sql",
        "explanation_complexity",
        "integration_api",
        "test_generation",
    }
    # every default test has a non-empty prompt
    assert all(t.prompt.strip() for t in suite.tests)
    # every test has a repeat >= 5 for statistical significance
    assert all(t.repeat >= 5 for t in suite.tests)


def test_cli_falls_back_to_defaults(monkeypatch) -> None:
    from benchmarker import cli

    # avoid actually contacting a server
    class _NoopRunner:
        def __init__(self, *a, **k):
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    runner = CliRunner()
    # run in an isolated dir so tests.json/params.yaml are absent
    with runner.isolated_filesystem():
        result = runner.invoke(cli.main, ["run", "--model", "demo"])
    assert result.exit_code == 0
    assert "using bundled default" in result.output
