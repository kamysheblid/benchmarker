"""Tests for the benchmarker CLI (Phase 1 + config integration)."""

import json
from pathlib import Path

import yaml
from click.testing import CliRunner
from pydantic import ValidationError

from benchmarker.cli import main


def _write_benchmark_yaml(path: Path) -> Path:
    b = path / "bench.yaml"
    b.write_text(
        yaml.safe_dump(
            {
                "tests": [{"id": "t1", "prompt": "Hi"}],
                "optimizer": {"type": "grid", "budget": 5},
                "parameters": [
                    {"name": "temperature", "type": "float", "low": 0.1, "high": 1.0}
                ],
            }
        )
    )
    return b


def test_run_prints_model(tmp_path: Path, monkeypatch) -> None:
    """`benchmarker run --model <name>` should echo the model name."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.model_name = k.get("model_name")

        async def run(self):
            return [], None

    bench = _write_benchmark_yaml(tmp_path)
    monkeypatch.setattr(cli, "Runner", _NoopRunner)
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "test-model", "--benchmarks", str(bench), "--force"]
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

    bench = _write_benchmark_yaml(tmp_path)
    monkeypatch.setattr(cli, "Runner", _NoopRunner)
    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--benchmarks", str(bench), "--force"]
    )
    assert result.exit_code == 0
    assert "1 tests" in result.output
    assert "1 parameters" in result.output


# --------------------------------------------------------------------------- #
# init                                                                        #
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


# --------------------------------------------------------------------------- #
# --benchmarks CLI                                                           #
# --------------------------------------------------------------------------- #
def test_run_loads_single_yaml(tmp_path: Path, monkeypatch) -> None:
    """--benchmarks path/to/bench.yaml should load 1 file."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    bench = _write_benchmark_yaml(tmp_path)

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "demo", "--benchmarks", str(bench), "--force"]
    )
    assert result.exit_code == 0
    assert "1 tests" in result.output


def test_run_loads_directory_of_yamls(tmp_path: Path, monkeypatch) -> None:
    """--benchmarks dir/ should load all **/*.yaml files and merge them."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    d = tmp_path / "benchmarks"
    d.mkdir()
    (d / "a.yaml").write_text(
        yaml.safe_dump({"tests": [{"id": "t1", "prompt": "A"}]})
    )
    (d / "b.yaml").write_text(
        yaml.safe_dump({"tests": [{"id": "t2", "prompt": "B"}]})
    )

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "demo", "--benchmarks", str(d), "--force"]
    )
    assert result.exit_code == 0
    assert "2 tests" in result.output


def test_run_legacy_json_backward_compat(tmp_path: Path, monkeypatch) -> None:
    """--benchmarks legacy.json should fall back to load_tests for backward compat."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps([{"id": "t1", "prompt": "Hi"}]))

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "demo", "--benchmarks", str(legacy), "--force"]
    )
    assert result.exit_code == 0
    assert "1 tests" in result.output


def test_run_empty_benchmarks_falls_back_to_defaults(tmp_path: Path, monkeypatch) -> None:
    """--benchmarks empty/ should fall back to bundled defaults."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    empty = tmp_path / "empty"
    empty.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "demo", "--benchmarks", str(empty), "--force"]
    )
    assert result.exit_code == 0
    assert "using bundled default" in result.output


def test_run_merges_params_from_first_file(tmp_path: Path, monkeypatch) -> None:
    """Params are taken from the first YAML file; subsequent files are validated."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.suite = k.get("suite")
            self.results = []

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    d = tmp_path / "benchmarks"
    d.mkdir()
    (d / "a.yaml").write_text(
        yaml.safe_dump(
            {
                "tests": [{"id": "t1", "prompt": "A"}],
                "optimizer": {"type": "grid", "budget": 5},
                "parameters": [
                    {"name": "temperature", "type": "float", "low": 0.1, "high": 1.0}
                ],
            }
        )
    )
    (d / "b.yaml").write_text(
        yaml.safe_dump(
            {
                "tests": [{"id": "t2", "prompt": "B"}],
                "optimizer": {"type": "grid", "budget": 5},
                "parameters": [
                    {"name": "temperature", "type": "float", "low": 0.1, "high": 1.0}
                ],
            }
        )
    )

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "demo", "--benchmarks", str(d), "--force"]
    )
    assert result.exit_code == 0
    assert "2 tests" in result.output


def test_run_param_mismatch_raises(tmp_path: Path, monkeypatch) -> None:
    """Params that differ across YAML files should raise ValueError."""
    from benchmarker import cli

    monkeypatch.setattr(cli, "Runner", lambda *a, **k: None)

    d = tmp_path / "benchmarks"
    d.mkdir()
    (d / "a.yaml").write_text(
        yaml.safe_dump(
            {
                "tests": [{"id": "t1", "prompt": "A"}],
                "optimizer": {"type": "grid", "budget": 5},
            }
        )
    )
    (d / "b.yaml").write_text(
        yaml.safe_dump(
            {
                "tests": [{"id": "t2", "prompt": "B"}],
                "optimizer": {"type": "bayesian", "budget": 5},
            }
        )
    )

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "demo", "--benchmarks", str(d), "--force"]
    )
    assert result.exit_code != 0


def test_run_writes_run_meta_json(tmp_path: Path, monkeypatch) -> None:
    """After run, run_meta.json should contain benchmark_files and params_source."""
    from benchmarker import cli

    class _NoopRunner:
        def __init__(self, *a, **k):
            self.run_dir = k.get("run_dir")

        async def run(self):
            return [], None

    monkeypatch.setattr(cli, "Runner", _NoopRunner)

    d = tmp_path / "benchmarks"
    d.mkdir()
    (d / "a.yaml").write_text(
        yaml.safe_dump(
            {
                "tests": [{"id": "t1", "prompt": "A"}],
                "optimizer": {"type": "grid", "budget": 5},
                "parameters": [
                    {"name": "temperature", "type": "float", "low": 0.1, "high": 1.0}
                ],
            }
        )
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--model",
            "demo",
            "--benchmarks",
            str(d),
            "--run-dir",
            str(run_dir),
            "--force",
        ],
    )
    assert result.exit_code == 0
    meta_path = run_dir / "run_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(meta["benchmark_files"]) == 1
    assert meta["benchmark_files"][0].endswith("a.yaml")
    assert meta["params_source"].endswith("a.yaml")


# --------------------------------------------------------------------------- #
# Legacy directory loading (existing behavior preserved)                       #
# --------------------------------------------------------------------------- #
def test_run_loads_directory_by_default(tmp_path: Path, monkeypatch) -> None:
    """Default --benchmarks=benchmarks should load from directory when present."""
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
    (cat / "001-a.yaml").write_text(
        yaml.safe_dump({"tests": [{"id": "t1", "prompt": "A"}], "optimizer": {"type": "grid", "budget": 5}})
    )
    (cat / "002-b.yaml").write_text(
        yaml.safe_dump({"tests": [{"id": "t2", "prompt": "B"}], "optimizer": {"type": "grid", "budget": 5}})
    )

    runner = CliRunner()
    result = runner.invoke(
        main, ["run", "--model", "demo", "--benchmarks", str(benchmarks), "--force"]
    )
    assert result.exit_code == 0
    assert "2 tests" in result.output


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
