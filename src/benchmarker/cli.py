"""Command-line interface for benchmarker."""

import asyncio
import csv
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from benchmarker.client import LLMClient
from benchmarker.config import load_params, load_params_default, load_tests, load_tests_default
from benchmarker.importer import import_scores
from benchmarker.optimizers import create_optimizer
from benchmarker.runner import ProgressReporter, Runner, config_key

console = Console()


@click.group()
@click.version_option(package_name="benchmarker")
def main() -> None:
    """benchmarker - find optimal LLM sampling parameters."""


@main.command()
@click.option("--model", default="default", show_default=True, help="Model name to benchmark.")
@click.option(
    "--tests",
    "tests_path",
    default="tests.json",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Path to the test suite JSON file (falls back to bundled default if missing).",
)
@click.option(
    "--params",
    "params_path",
    default="params.yaml",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Path to the parameter search-space YAML (falls back to bundled default if missing).",
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
    help="Directory to store raw results and the eval file.",
)
@click.option(
    "--csv",
    "csv_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional path to export the speed ranking as CSV.",
)
def run(
    model: str,
    tests_path: Path,
    params_path: Path,
    url: str | None,
    run_dir: Path,
    csv_path: Path | None,
) -> None:
    """Run a benchmark for the given model."""
    click.echo(f"Hello, world — benchmarking model: {model}")

    if tests_path.exists():
        suite = load_tests(tests_path)
    else:
        click.echo(f"(no {tests_path} found — using bundled default test suite)")
        suite = load_tests_default()
    if params_path.exists():
        params = load_params(params_path)
    else:
        click.echo(f"(no {params_path} found — using bundled default params)")
        params = load_params_default()

    click.echo(
        f"Loaded {len(suite.tests)} tests and {len(params.parameters)} parameters "
        f"(optimizer={params.optimizer.type}, budget={params.optimizer.budget})."
    )

    client = LLMClient(base_url=url) if url else LLMClient()
    optimizer = create_optimizer(params.optimizer, params.parameters)
    reporter = _RichProgress()
    runner = Runner(
        client,
        suite,
        optimizer,
        model,
        run_dir,
        progress=reporter,
        static_params=params.static_params,
    )
    results = asyncio.run(runner.run())

    _print_speed_table(results)
    if csv_path:
        _write_speed_csv(csv_path, results)
        click.echo(f"Exported speed ranking CSV to {csv_path}")

    click.echo(f"\nSaved raw results to {run_dir / 'raw_data.json'}")
    click.echo(f"Saved evaluation file to {run_dir / 'eval_output.md'}")
    click.echo("Copy that file's content to a judge LLM, then run `import-scores`.")


@main.command(name="import-scores")
@click.argument("scores_path", type=click.Path(path_type=Path, exists=True))
@click.option(
    "--run-dir",
    "run_dir",
    default=Path("benchmark_runs/latest"),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory containing raw_data.json from a previous run.",
)
@click.option(
    "--weight-quality",
    default=0.5,
    show_default=True,
    type=float,
    help="Weight (0..1) of quality vs speed in the combined score.",
)
@click.option(
    "--csv",
    "csv_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional path to export the ranking as CSV.",
)
def import_scores_cmd(
    scores_path: Path,
    run_dir: Path,
    weight_quality: float,
    csv_path: Path | None,
) -> None:
    """Compute the final ranking from a judge SCORES_PATH."""
    ranking = import_scores(run_dir, scores_path, weight_quality=weight_quality)

    table = Table(title="Final Benchmark Ranking")
    table.add_column("Config", overflow="fold")
    table.add_column("Avg Quality", justify="right")
    table.add_column("Avg Tok/s", justify="right")
    table.add_column("Combined", justify="right")
    for item in ranking:
        table.add_row(
            item.config,
            f"{item.avg_quality:.2f}",
            f"{item.avg_speed:.2f}",
            f"{item.combined:.3f}",
        )
    console.print(table)

    if csv_path:
        _write_csv(csv_path, ranking)
        click.echo(f"Exported ranking CSV to {csv_path}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
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


def _print_speed_table(results: list) -> None:
    speeds: dict[str, list[float]] = {}
    for r in results:
        if r.error is None:
            speeds.setdefault(config_key(r.config), []).append(r.tokens_per_sec)
    ranking = sorted(
        ((k, sum(v) / len(v)) for k, v in speeds.items()),
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    table = Table(title="Top Configs by tokens/s (speed only)")
    table.add_column("Config", overflow="fold")
    table.add_column("Avg Tok/s", justify="right")
    for cfg_key, speed in ranking:
        table.add_row(cfg_key, f"{speed:.2f}")
    console.print(table)


def _write_speed_csv(path: Path, results: list) -> None:
    speeds: dict[str, list[float]] = {}
    for r in results:
        if r.error is None:
            speeds.setdefault(config_key(r.config), []).append(r.tokens_per_sec)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", "avg_tokens_per_sec"])
        for cfg_key, vals in speeds.items():
            writer.writerow([cfg_key, sum(vals) / len(vals)])


def _write_csv(path: Path, ranking: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", "avg_quality", "avg_speed", "norm_quality", "norm_speed", "combined"])
        for item in ranking:
            writer.writerow(
                [
                    item.config,
                    item.avg_quality,
                    item.avg_speed,
                    item.norm_quality,
                    item.norm_speed,
                    item.combined,
                ]
            )


if __name__ == "__main__":
    main()
