"""Tests for the benchmarker CLI (Phase 1 + config integration)."""

import json
from pathlib import Path

import yaml
from click.testing import CliRunner
from pydantic import ValidationError

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


def test_run_prints_model(tmp_path: Path, monkeypatch) -> None:
    """`benchmarker run --model <name>` should echo the model name."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.model_name = k.get("model_name")

        async def run(self):
            return [], None

    tests, params = _write_configs(tmp_path)
    monkeypatch.setattr(cli, "Runner", _NoopRunner)
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "test-model", "--tests", str(tests), "--params", str(params), "--force"]
    )
    assert result.exit_code == 0
    assert "test-model" in result.output


def test_run_default_executes(tmp_path: Path, monkeypatch) -> None:
    """`benchmarker run` with valid files should load configs and exit 0."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")

        async def run(self):
            return [], None

    tests, params = _write_configs(tmp_path)
    monkeypatch.setattr(cli, "Runner", _NoopRunner)
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--tests", str(tests), "--params", str(params), "--force"]
    )
    assert result.exit_code == 0
    assert "1 tests" in result.output
    assert "1 parameters" in result.output


# --------------------------------------------------------------------------- #
# Phase 3: directory-aware CLI                                                  #
# --------------------------------------------------------------------------- #
def test_init_creates_benchmarks_directory(tmp_path: Path) -> None:
    """`benchmarker init` should create benchmarks/ with category subdirs."""
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0

    benchmarks = tmp_path / "benchmarks"
    assert benchmarks.is_dir()

    subdirs = sorted(p.name for p in benchmarks.iterdir() if p.is_dir())
    expected_subdirs = [
        "api-integration",
        "bug-fixing",
        "code-generation",
        "comment-generation",
        "general",
        "refactoring",
        "security-vulnerability",
        "test-generation",
    ]
    assert subdirs == expected_subdirs

    json_files = sorted(benchmarks.glob("**/*.json"))
    assert len(json_files) == 13
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        assert "id" in data
        assert "prompt" in data


def test_init_does_not_create_tests_json(tmp_path: Path) -> None:
    """`benchmarker init` should NOT create tests.json."""
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert not (tmp_path / "tests.json").exists()


def test_init_creates_params_yaml(tmp_path: Path) -> None:
    """`benchmarker init` should still create params.yaml."""
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "params.yaml").exists()


def test_run_loads_directory_by_default(tmp_path: Path, monkeypatch) -> None:
    """Default --tests=benchmarks should load from directory when present."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    cat = benchmarks / "cat-a"
    cat.mkdir()
    (cat / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))
    (cat / "002-b.json").write_text(json.dumps({"id": "t2", "prompt": "B"}))

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

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "demo", "--tests", str(benchmarks), "--params", str(params)]
    )
    assert result.exit_code == 0
    assert "2 tests" in result.output


def test_run_loads_legacy_file_when_provided(tmp_path: Path, monkeypatch) -> None:
    """--tests legacy.json should still load flat file mode."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    tests = tmp_path / "legacy.json"
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

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "demo", "--tests", str(tests), "--params", str(params)]
    )
    assert result.exit_code == 0
    assert "1 tests" in result.output


def test_run_categories_filter(tmp_path: Path, monkeypatch) -> None:
    """--categories should filter directory-loaded tests."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    cat_a = benchmarks / "cat-a"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))
    cat_b = benchmarks / "cat-b"
    cat_b.mkdir()
    (cat_b / "001-b.json").write_text(json.dumps({"id": "t2", "prompt": "B"}))

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

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "demo",
            "--tests",
            str(benchmarks),
            "--categories",
            "cat-a",
            "--params",
            str(params),
        ],
    )
    assert result.exit_code == 0
    assert "1 tests" in result.output


def test_run_categories_invalid_slug_raises(tmp_path: Path, monkeypatch) -> None:
    """--categories with an invalid slug should raise click.BadParameter."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    cat_a = benchmarks / "cat-a"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

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

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "demo",
            "--tests",
            str(benchmarks),
            "--categories",
            "nonexistent",
            "--params",
            str(params),
        ],
    )
    assert result.exit_code != 0
    assert "Error:" in result.output
    assert "cat-a" in result.output


def test_run_categories_with_file_raises(tmp_path: Path, monkeypatch) -> None:
    """--categories with a flat --tests file should raise click.BadParameter."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    tests = tmp_path / "legacy.json"
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

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "demo",
            "--tests",
            str(tests),
            "--categories",
            "bug-fixing",
            "--params",
            str(params),
        ],
    )
    assert result.exit_code != 0
    assert "directory (benchmarks/)" in result.output


def test_run_categories_missing_dir_falls_back(tmp_path: Path, monkeypatch) -> None:
    """--categories with missing --tests should fall back to defaults."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

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

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "demo",
            "--tests",
            str(tmp_path / "nonexistent-benchmarks"),
            "--categories",
            "bug-fixing",
            "--params",
            str(params),
        ],
    )
    assert result.exit_code == 0
    assert "using bundled default" in result.output


# --------------------------------------------------------------------------- #
# Task 07: CLI integration tests for directory mode                           #
# --------------------------------------------------------------------------- #
def test_run_loads_benchmarks_dir(tmp_path: Path, monkeypatch) -> None:
    """Default benchmarks/ directory loads both category files."""
    import os

    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    bug_fixing = benchmarks / "bug-fixing"
    bug_fixing.mkdir()
    (bug_fixing / "001-typo.json").write_text(json.dumps({"id": "t1", "prompt": "Fix typo"}))
    code_gen = benchmarks / "code-generation"
    code_gen.mkdir()
    (code_gen / "001-chunk.json").write_text(json.dumps({"id": "t2", "prompt": "Generate code"}))

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

    runner = CliRunner()
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = runner.invoke(
            main,
            [
                "run",
                "--model",
                "demo",
                "--url",
                "http://example.com",
                "--params",
                str(params),
            ],
        )
    finally:
        os.chdir(old_cwd)
    assert result.exit_code == 0
    assert "2 tests" in result.output


def test_run_categories_filter(tmp_path: Path, monkeypatch) -> None:
    """--categories filters to only the selected category's tests."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    bug_fixing = benchmarks / "bug-fixing"
    bug_fixing.mkdir()
    (bug_fixing / "001-typo.json").write_text(json.dumps({"id": "t1", "prompt": "Fix typo"}))
    code_gen = benchmarks / "code-generation"
    code_gen.mkdir()
    (code_gen / "001-chunk.json").write_text(json.dumps({"id": "t2", "prompt": "Generate code"}))
    general = benchmarks / "general"
    general.mkdir()
    (general / "001-creative.json").write_text(json.dumps({"id": "t3", "prompt": "Be creative"}))

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

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "demo",
            "--tests",
            str(benchmarks),
            "--categories",
            "bug-fixing",
            "--params",
            str(params),
        ],
    )
    assert result.exit_code == 0
    assert "1 tests" in result.output


def test_run_invalid_categories_raises(tmp_path: Path, monkeypatch) -> None:
    """--categories with an invalid slug raises click.BadParameter."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    cat_a = benchmarks / "cat-a"
    cat_a.mkdir()
    (cat_a / "001-a.json").write_text(json.dumps({"id": "t1", "prompt": "A"}))

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

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "demo",
            "--tests",
            str(benchmarks),
            "--categories",
            "nonexistent",
            "--params",
            str(params),
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, SystemExit) or "BadParameter" in str(result.exception)


def test_run_categories_with_legacy_file_raises(tmp_path: Path, monkeypatch) -> None:
    """--categories with a flat legacy file raises click.BadParameter."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    tests = tmp_path / "legacy.json"
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

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "demo",
            "--tests",
            str(tests),
            "--categories",
            "bug-fixing",
            "--params",
            str(params),
        ],
    )
    assert result.exit_code != 0
    assert "directory (benchmarks/)" in result.output


def test_run_categories_fallback_to_defaults(tmp_path: Path, monkeypatch) -> None:
    """--categories with missing tests falls back to bundled defaults."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

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

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "demo",
            "--tests",
            str(tmp_path / "nonexistent-benchmarks"),
            "--categories",
            "bug-fixing",
            "--params",
            str(params),
        ],
    )
    assert result.exit_code == 0
    assert "using bundled default" in result.output


def test_init_creates_benchmarks_dir(tmp_path: Path) -> None:
    """`benchmarker init --dir` creates benchmarks/ with subdirs and prompt files."""
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0

    benchmarks = tmp_path / "benchmarks"
    assert benchmarks.is_dir()

    subdirs = sorted(p.name for p in benchmarks.iterdir() if p.is_dir())
    expected_subdirs = [
        "api-integration",
        "bug-fixing",
        "code-generation",
        "comment-generation",
        "general",
        "refactoring",
        "security-vulnerability",
        "test-generation",
    ]
    assert subdirs == expected_subdirs

    json_files = sorted(benchmarks.glob("**/*.json"))
    assert len(json_files) == 13
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        assert "id" in data
        assert "prompt" in data


def test_init_no_tests_json_created(tmp_path: Path) -> None:
    """`benchmarker init` does NOT create tests.json."""
    runner = CliRunner()
    result = runner.invoke(main, ["init", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert not (tmp_path / "tests.json").exists()


def test_init_force_removes_existing_benchmarks(tmp_path: Path) -> None:
    """`benchmarker init --force` recreates benchmarks/ from defaults."""
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    (benchmarks / "old-file.json").write_text(json.dumps({"id": "old", "prompt": "stale"}))

    runner = CliRunner()
    result = runner.invoke(main, ["init", "--dir", str(tmp_path), "--force"])
    assert result.exit_code == 0

    assert benchmarks.is_dir()
    assert not (benchmarks / "old-file.json").exists()

    json_files = sorted(benchmarks.glob("**/*.json"))
    assert len(json_files) == 13
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        assert "id" in data
        assert "prompt" in data
