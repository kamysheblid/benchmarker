"""Command-line interface for benchmarker."""

import asyncio
import csv
import logging
import shutil
from pathlib import Path

from typing import Any

import yaml

import click
import yaml
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from benchmarker.client import LLMClient
from benchmarker.config import (
    TestSuite,
    discover_benchmark_files,
    load_benchmark_file,
    load_params,
    load_params_default,
    load_tests,
    load_tests_default,
    validate_params_match,
)
from benchmarker.logging import setup_logging
from benchmarker.optimizers import create_optimizer
from benchmarker.parse_judge import parse_and_act
from benchmarker.runner import ProgressReporter, Runner, config_key

console = Console()
logger = logging.getLogger("benchmarker")


def _yaml_str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _yaml_str_representer, Dumper=yaml.SafeDumper)


@click.group()
@click.version_option(package_name="benchmarker")
def main() -> None:
    """benchmarker - find optimal LLM sampling parameters."""


# ------------------------------------------------------------------ #
#  init                                                               #
# ------------------------------------------------------------------ #
@main.command()
@click.option(
    "--dir",
    "target_dir",
    default=".",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory to initialise with default benchmark YAML files.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing files.",
)
def init(target_dir: Path, force: bool) -> None:
    """Create default self-contained benchmark files in the given directory."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    benchmarks_dest = target_dir / "benchmarks"
    copied = []

    if not benchmarks_dest.exists() or force:
        if force and benchmarks_dest.exists():
            shutil.rmtree(benchmarks_dest)
        benchmarks_dest.mkdir(parents=True, exist_ok=True)

        suite = load_tests_default()
        # Load params from bundled defaults for writing into each YAML file
        params = load_params_default()
        params_raw = {
            "optimizer": {
                "type": params.optimizer.type,
                "budget": params.optimizer.budget,
            },
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type.value,
                    "low": p.low,
                    "high": p.high,
                    "step": p.step,
                    "choices": p.choices,
                }
                for p in params.parameters
            ],
            "static_params": params.static_params,
        }

        # Group tests by category from suite metadata
        by_category: dict[str, list[dict[str, Any]]] = {}
        for tc in suite.tests:
            cat = suite.categories.get(tc.id, "general")
            by_category.setdefault(cat, []).append(tc.model_dump(exclude_none=True))

        for category, tests in sorted(by_category.items()):
            raw = dict(params_raw)
            raw["tests"] = tests
            filepath = benchmarks_dest / f"{category}.yaml"
            filepath.write_text(
                yaml.safe_dump(raw, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
            copied.append(filepath)
    else:
        click.echo(f"  (skip) {benchmarks_dest} already exists — use --force to overwrite.")

    if copied:
        for f in copied:
            click.echo(f"  created {f}")
    else:
        click.echo("  All files already exist. Nothing to do.")


# ------------------------------------------------------------------ #
#  run                                                                #
# ------------------------------------------------------------------ #
@main.command()
@click.option("--model", default="default", show_default=True, help="Model name to benchmark.")
@click.option(
    "--benchmarks",
    "benchmarks_path",
    default="benchmarks",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Path to a benchmark YAML file or a directory of benchmark YAML files.",
)
@click.option(
    "--url",
    "url",
    default=None,
    help="Base URL of the llama-server chat completions endpoint.",
)
@click.option(
    "--run-dir",
    "run_dir",
    default=Path("benchmark_runs/latest"),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory to store raw results and the judge prompt file.",
)
@click.option(
    "--csv",
    "csv_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional path to export the speed ranking as CSV.",
)
@click.option(
    "--auto-eval/--no-auto-eval",
    "auto_eval",
    default=False,
    help="Run automated code evaluation on responses (scoring rubric with unit tests, static analysis).",
)
@click.option(
    "--cost-per-1m-input",
    default=0.0,
    type=float,
    show_default=True,
    help="USD cost per 1M input tokens (for cost tracking in ranking).",
)
@click.option(
    "--cost-per-1m-output",
    default=0.0,
    type=float,
    show_default=True,
    help="USD cost per 1M output tokens (for cost tracking in ranking).",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for deterministic sampling (applied to optimizer).",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug-level console logging.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from an existing checkpoint if present.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Ignore any existing checkpoint and start fresh.",
)
def run(
    model: str,
    benchmarks_path: Path,
    url: str | None,
    run_dir: Path,
    csv_path: Path | None,
    auto_eval: bool,
    cost_per_1m_input: float,
    cost_per_1m_output: float,
    seed: int | None,
    verbose: bool,
    resume: bool,
    force: bool,
) -> None:
    """Run a benchmark for the given model."""
    setup_logging(run_dir=run_dir, verbose=verbose)

    # ------------------------------------------------------------------ #
    #  Resolve benchmark files                                            #
    # ------------------------------------------------------------------ #
    benchmark_files: list[Path] = []
    params_source: Path | None = None
    if benchmarks_path.exists() and benchmarks_path.is_file() and benchmarks_path.suffix == ".json":
        logger.info("Loading legacy JSON test suite: %s", benchmarks_path)
        suite = load_tests(benchmarks_path)
        params = None
        benchmark_files = [benchmarks_path]
    else:
        benchmark_files = discover_benchmark_files(benchmarks_path)
        if not benchmark_files:
            logger.info("(no %s found — using bundled default test suite)", benchmarks_path)
            suite = load_tests_default()
            params = None
        else:
            merged_suite = TestSuite()
            resolved_params: ParamsConfig | None = None
            params_source: Path | None = None
            for file in benchmark_files:
                file_suite, file_params = load_benchmark_file(file)
                merged_suite.tests.extend(file_suite.tests)
                merged_suite.categories.update(file_suite.categories)
                if file_params is not None:
                    if resolved_params is None:
                        resolved_params = file_params
                        params_source = file
                    else:
                        validate_params_match(resolved_params, file_params)
            # Final uniqueness check on merged tests
            seen: set[str] = set()
            for test in merged_suite.tests:
                if test.id in seen:
                    raise ValueError(f"duplicate test id across benchmark files: {test.id!r}")
                seen.add(test.id)
            suite = merged_suite
            if resolved_params is None:
                logger.info("(no params in benchmark files — using bundled default params)")
                params = load_params_default()
                params_source = None
            else:
                params = resolved_params
            logger.info(
                "Loaded %d benchmark file(s), %d tests total.",
                len(benchmark_files),
                len(suite.tests),
            )

    if params is None:
        params = load_params_default()

    logger.info(
        "Starting benchmark: model=%s, tests=%d, params=%d, run_dir=%s",
        model,
        len(suite.tests),
        len(params.parameters),
        run_dir,
    )
    logger.info(
        "Loaded %d tests and %d parameters (optimizer=%s, budget=%d).",
        len(suite.tests),
        len(params.parameters),
        params.optimizer.type,
        params.optimizer.budget,
    )

    client = LLMClient(base_url=url) if url else LLMClient()
    optimizer = create_optimizer(params.optimizer, params.parameters, seed=seed)
    reporter = _RichProgress()
    runner = Runner(
        client,
        suite,
        optimizer,
        model,
        run_dir,
        progress=reporter,
        static_params=params.static_params,
        auto_eval=auto_eval,
        cost_per_1m_input=cost_per_1m_input,
        cost_per_1m_output=cost_per_1m_output,
        resume=resume,
        force=force,
    )
    results, auto_scores = asyncio.run(runner.run())

    # Write benchmark metadata after the run completes.
    meta = {
        "benchmark_files": [str(p) for p in benchmark_files],
        "params_source": str(params_source) if params_source else None,
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    _print_quality_table(results)
    if csv_path:
        _write_quality_csv(csv_path, results)
        logger.info("Exported quality ranking CSV to %s", csv_path)
        click.echo(f"Exported quality ranking CSV to {csv_path}")

    if auto_scores:
        _print_auto_eval_summary(results, auto_scores)
        logger.info("Saved auto-evaluation scores to %s", run_dir / "scores_auto.json")
        click.echo(f"\nSaved auto-evaluation scores to {run_dir / 'scores_auto.json'}")

    logger.info("Saved raw results to %s", run_dir / "raw_data.json")
    logger.info("Saved judge prompt to %s", run_dir / "judge_prompt.md")
    click.echo(f"\nSaved raw results to {run_dir / 'raw_data.json'}")
    click.echo(f"Saved judge prompt to {run_dir / 'judge_prompt.md'}")
    click.echo(
        "\nNext step: copy the contents of judge_prompt.md to your judge LLM,\n"
        "save the reply as a text file, then run:\n"
        f"  benchmarker parse <reply_file> --run-dir {run_dir}"
    )


# ------------------------------------------------------------------ #
#  parse                                                              #
# ------------------------------------------------------------------ #
@main.command()
@click.argument("reply_file", type=click.Path(path_type=Path), required=False)
@click.option(
    "--benchmark-file",
    "benchmark_file",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to the benchmark YAML file to update. Inferred from --run-dir if omitted.",
)
@click.option(
    "--run-dir",
    "run_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Run directory to load config_map.json from for config lookup.",
)
def parse(reply_file: Path | None, benchmark_file: Path | None, run_dir: Path | None) -> None:
    """Parse the judge's reply and take action (conclude / refine / expand).

    If REPLY_FILE is provided, read it; otherwise read from stdin.
    """
    if reply_file:
        if not reply_file.exists():
            click.echo(f"Error: file not found: {reply_file}", err=True)
            raise SystemExit(1)
        text = reply_file.read_text(encoding="utf-8")
        click.echo(f"Read judge reply from {reply_file}")
    else:
        click.echo("Reading judge reply from stdin (paste the reply and press Ctrl+D)...")
        import sys

        text = sys.stdin.read()

    # Infer benchmark file from run_meta.json when --benchmark-file is omitted
    benchmark_source = benchmark_file
    if benchmark_source is None and run_dir is not None:
        meta_path = run_dir / "run_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                inferred = meta.get("benchmark_file")
                if inferred:
                    benchmark_source = Path(inferred)
            except (json.JSONDecodeError, OSError):
                pass

    try:
        parse_and_act(text, benchmark_source=benchmark_source, run_dir=run_dir)
    except ValueError as exc:
        click.echo(f"Error parsing judge reply: {exc}", err=True)
        click.echo("\nRaw text received:", err=True)
        click.echo(text[:1000], err=True)
        raise SystemExit(1) from exc


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #
class _RichProgress(ProgressReporter):
    def __init__(self) -> None:
        self._progress = Progress(transient=True)
        self._task = None

    def start(self, total: int) -> None:
        self._progress.start()
        self._task = self._progress.add_task("Benchmarking...", total=total or 1)

    def advance(self) -> None:
        if self._task is not None:
            self._progress.advance(self._task)

    def finish(self) -> None:
        self._progress.stop()


def _print_quality_table(results: list) -> None:
    per_config: dict[str, list[dict]] = {}
    for r in results:
        if r.error is None:
            k = config_key(r.config)
            per_config.setdefault(k, []).append({
                "success_rate": r.success_rate or 0.0,
                "coverage": r.coverage or 0.0,
                "quality": (r.success_rate or 0.0) * (r.coverage or 0.0),
                "tokens_per_sec": r.tokens_per_sec,
            })
    ranking = sorted(
        ((k, sum(v["quality"] for v in vals) / len(vals)) for k, vals in per_config.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    table = Table(title="Top Configs by Quality (success_rate × coverage)")
    table.add_column("Config", overflow="fold")
    table.add_column("Quality", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Coverage", justify="right")
    table.add_column("Avg Tok/s", justify="right")
    for cfg_key, quality in ranking:
        items = per_config[cfg_key]
        avg_success = sum(v["success_rate"] for v in items) / len(items)
        avg_coverage = sum(v["coverage"] for v in items) / len(items)
        avg_speed = sum(v["tokens_per_sec"] for v in items) / len(items)
        table.add_row(
            cfg_key,
            f"{quality:.3f}",
            f"{avg_success:.1%}",
            f"{avg_coverage:.1%}",
            f"{avg_speed:.2f}",
        )
    console.print(table)


def _write_quality_csv(path: Path, results: list) -> None:
    per_config: dict[str, list[float]] = {}
    for r in results:
        if r.error is None:
            quality = (r.success_rate or 0.0) * (r.coverage or 0.0)
            k = config_key(r.config)
            per_config.setdefault(k, []).append(quality)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", "quality"])
        for cfg_key, vals in per_config.items():
            writer.writerow([cfg_key, sum(vals) / len(vals)])


def _print_auto_eval_summary(results: list, auto_scores: dict) -> None:
    """Print a summary table of auto-evaluation scores grouped by config."""
    per_config: dict[str, list[float]] = {}
    for r in results:
        if r.error is not None:
            continue
        ck = config_key(r.config)
        key = f"{ck}::{r.test_id}::{r.repetition}"
        if key in auto_scores:
            overall = auto_scores[key].get("overall", 0.0)
            per_config.setdefault(ck, []).append(overall)

    table = Table(title="Auto-Evaluation: Top Configs by Code Quality")
    table.add_column("Config", overflow="fold")
    table.add_column("Avg Quality Score", justify="right")
    ranking = sorted(
        ((k, sum(v) / len(v)) for k, v in per_config.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    for cfg_key, avg_q in ranking:
        table.add_row(cfg_key, f"{avg_q:.3f}")
    console.print(table)


if __name__ == "__main__":
    main()
