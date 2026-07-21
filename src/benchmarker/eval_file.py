"""Generate the judge_prompt.md file for external LLM judging.

The judge receives a self-contained markdown file with all test results and
a strict prompt asking for a JSON response with scores and a recommendation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

JUDGE_PROMPT_FILE = "judge_prompt.md"


def _render_result(r: Any) -> str:
    """Render a single RunResult as a markdown block."""
    err = f"\n\n> **ERROR:** {r.error}" if r.error else ""
    return (
        f"**Prompt:**\n\n```\n{r.prompt}\n```\n\n"
        f"**Response:**\n\n```\n{r.response_text}\n```{err}\n\n"
        f"*metrics — ttft: {r.ttft:.3f}s, total: {r.total_time:.3f}s, "
        f"{r.tokens_per_sec:.2f} tok/s, prompt_tokens={r.prompt_tokens}, "
        f"completion_tokens={r.completion_tokens}*\n"
    )


# ------------------------------------------------------------------ #
#  Judge instruction template                                        #
# ------------------------------------------------------------------ #
_JUDGE_INSTRUCTIONS = """\
---

## Judge Instructions

You are a quality judge evaluating LLM sampling parameter configurations.

For every **Config** section above, assess the quality of each response.
Consider correctness, coherence, conciseness, and task completion.

### Required Output Format

At the **very end** of your reply, output a **single JSON object** with exactly these fields:

```json
{{
  "scores": {{
    "<config_key>": {{
      "overall": <1-10>,
      "reasoning": "<brief justification>"
    }}
  }},
  "recommendation": "conclude",
  "confidence": "high",
  "reasoning": "Overall assessment of the best configuration.",
  "refinement_hint": null
}}
```

### Field rules

- **`scores`** — object keyed by config key (copy it verbatim from the section heading).
  Each value has an `overall` score (1 = poor, 10 = excellent) and a `reasoning` string.
- **`recommendation`** — one of:
  - `"conclude"` — one config is clearly the best. Print final advice.
  - `"refine"` — a region of the parameter space shows promise; narrow down and re-run.
  - `"expand"` — results are inconclusive; broaden the search or add more tests.
- **`confidence`** — `"high"`, `"medium"`, or `"low"`.
- **`reasoning`** — brief justification for the recommendation.
- **`refinement_hint`** — when `recommendation` is `"refine"`, provide an object like
  `{{"temperature": [0.7, 0.9], "top_p": [0.85, 1.0]}}` with tighter ranges.
  Use `null` when not refining.

### Important

- Output **only** the JSON block at the very end — nothing after it.
- Do NOT wrap the JSON in markdown code fences in your actual reply
  (the example above shows fences only for clarity).
- Reply must be pure text with the JSON as the final block.
"""


def generate_judge_prompt(
    run_dir: Path,
    run_results: list[Any],
    out_path: Path | None = None,
) -> Path:
    """Write ``judge_prompt.md`` — a self-contained file for the judge LLM.

    Args:
        run_dir: Directory containing the run (used as default output location).
        run_results: All run results from the benchmark.
        out_path: Optional explicit output path; defaults to ``run_dir/judge_prompt.md``.

    Returns:
        The path of the written file.
    """
    from benchmarker.runner import config_key

    run_dir = Path(run_dir)
    out_path = out_path or (run_dir / JUDGE_PROMPT_FILE)

    grouped: dict[str, list[Any]] = {}
    for r in run_results:
        grouped.setdefault(config_key(r.config), []).append(r)

    sections: list[str] = [
        "# Benchmark Judge Prompt\n",
        f"_Run directory: `{run_dir}`_\n",
        f"_Configurations evaluated: {len(grouped)}_\n",
        f"_Total test runs: {len(run_results)}_\n",
    ]

    # Summary table of configs and their aggregate metrics
    sections.append("## Config Summary\n")
    sections.append("| Config | Avg Tok/s | Avg TTFT (s) | Avg Total (s) | Errors |")
    sections.append("|--------|-----------|--------------|---------------|--------|")
    for key, items in sorted(grouped.items()):
        speeds = [r.tokens_per_sec for r in items if r.error is None]
        ttfts = [r.ttft for r in items if r.error is None]
        totals = [r.total_time for r in items if r.error is None]
        errs = sum(1 for r in items if r.error is not None)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        avg_total = sum(totals) / len(totals) if totals else 0.0
        sections.append(
            f"| `{key}` | {avg_speed:.1f} | {avg_ttft:.3f} | {avg_total:.3f} | {errs} |"
        )
    sections.append("")

    # Detail sections per config
    for key, items in sorted(grouped.items()):
        rep_label = f" ({len(items)} runs)"
        sections.append(f"\n## Config: `{key}`{rep_label}\n")
        for r in items:
            rlabel = f"Repetition {r.repetition}" if r.repetition > 1 else "Run 1"
            sections.append(f"### Test `{r.test_id}` — {rlabel}\n")
            sections.append(_render_result(r))

    body = "\n".join(sections)
    content = body + "\n\n" + _JUDGE_INSTRUCTIONS
    out_path.write_text(content, encoding="utf-8")
    return out_path
