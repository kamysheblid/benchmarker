"""Tests for the benchmarker CLI (Phase 1 + config integration)."""

import json
from pathlib import Path

import yaml
from click.testing import CliRunner

from benchmarker.cli import main


def _write_configs(tmp_path: Path) -> tuple[Path, Path]:
    tests = tmp_path / "tests.json"
    tests.write_text(json.dumps([{"id": "t1", "prompt": "Hi"}]))
    params = tmp_path / "params.yaml"
    params.write_text(
        yaml.safe_dump(
            {
                "optimizer": {"type": "grid", "budget": 5},
                "parameters": [
                    {"name": "temperature", "type": "float", "low": 0.1, "high": 1.0}
                ],
            }
        )
    )
    return tests, params


def test_run_prints_model(tmp_path: Path) -> None:
    """`benchmarker run --model <name>` should echo the model name."""
    tests, params = _write_configs(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "test-model", "--tests", str(tests), "--params", str(params)]
    )
    assert result.exit_code == 0
    assert "test-model" in result.output


def test_run_default_executes(tmp_path: Path) -> None:
    """`benchmarker run` with valid files should load configs and exit 0."""
    tests, params = _write_configs(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--tests", str(tests), "--params", str(params)]
    )
    assert result.exit_code == 0
    assert "1 tests" in result.output
    assert "1 parameters" in result.output
