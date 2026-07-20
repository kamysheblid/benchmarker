"""Generate the eval_output.md file for external LLM judging (Phase 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

EVAL_FILE_NAME = "eval_output.md"


def _render_result(r: RunResult) -> str:
    rep_label = f" (rep {r.repetition})" if r.repetition > 1 else ""
    err = f"\n\n> ERROR: {r.error}" if r.error else ""
    return (
        f"### Test `{r.test_id}`{rep_label}\n\n"
        f"**Prompt:**\n\n{r.prompt}\n\n"
        f"**Response:**\n\n{r.response_text}{err}\n\n"
        f"*metrics: ttft={r.ttft:.3f}s, total={r.total_time:.3f}s, "
        f"{r.tokens_per_sec:.2f} tok/s, prompt_tokens={r.prompt_tokens}, "
        f"completion_tokens={r.completion_tokens}*\n"
    )


_RATING_TEMPLATE = """\
---

## Rating Instructions

You are a quality judge. For **each** response above, assign a quality score.

For every config section and every test (including repetitions), produce one JSON
object with these fields:

- `config`: the exact config key shown in the section heading (copy it verbatim)
- `test_id`: the test identifier (e.g. `"t1"`)
- `repetition`: the repetition number (1 if not shown)
- `scores`: an object with at least an `"overall"` score from 1 (poor) to 10 (excellent).
  You may add additional sub-scores (e.g. `"coherence"`, `"correctness"`).

Return a **JSON array** like:

```json
[
  {
    "config": "<config key>",
    "test_id": "t1",
    "repetition": 1,
    "scores": { "overall": 8, "coherence": 7, "correctness": 9 }
  }
]
```

Save the array to a file named `scores.json` and run:
`benchmarker import-scores scores.json --run-dir <this directory>`.
"""


def generate_eval_md(
    run_dir: Path,
    run_results: list[RunResult],
    out_path: Path | None = None,
) -> Path:
    """Write ``eval_output.md`` grouping responses by config, with rating template.

    Args:
        run_dir: Directory containing the run (used as default output location).
        run_results: All run results from the benchmark.
        out_path: Optional explicit output path; defaults to ``run_dir/eval_output.md``.

    Returns:
        The path of the written file.
    """
    # Deferred import avoids a circular dependency with benchmarker.runner.
    from benchmarker.runner import RunResult, config_key

    run_dir = Path(run_dir)
    out_path = out_path or (run_dir / EVAL_FILE_NAME)

    grouped: dict[str, list[RunResult]] = {}
    for r in run_results:
        grouped.setdefault(config_key(r.config), []).append(r)

    sections: list[str] = ["# Benchmark Evaluation Output\n"]
    for key, items in grouped.items():
        sections.append(f"## Config: `{key}`\n")
        blocks = [_render_result(r) for r in items]
        sections.append("\n---\n\n".join(blocks))

    body = "\n".join(sections)
    content = body + "\n\n" + _RATING_TEMPLATE
    out_path.write_text(content, encoding="utf-8")
    return out_path
