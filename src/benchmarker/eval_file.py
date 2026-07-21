"""Generate the judge_prompt.md file for external LLM judging.

The judge receives a self-contained markdown file with all test results and
a strict prompt asking for a JSON response with scores and a recommendation.

Configurations are assigned **short stable IDs** (``config_1``, ``config_2``, …)
to make the judge's JSON output clean and easy to parse. A companion mapping
file (``config_map.json``) is saved alongside the prompt so the ``parse``
command can translate short IDs back to real parameter values.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

JUDGE_PROMPT_FILE = "judge_prompt.md"
CONFIG_MAP_FILE = "config_map.json"


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


def _make_short_ids(
    group_keys: list[str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Assign stable short IDs to config keys.

    Returns:
        A tuple of:
        - ``key_to_short`` — maps full JSON config key → short ID (``config_1``).
        - ``short_to_key`` — reverse mapping.
    """
    key_to_short: dict[str, str] = {}
    short_to_key: dict[str, str] = {}
    for i, key in enumerate(sorted(group_keys), start=1):
        sid = f"config_{i}"
        key_to_short[key] = sid
        short_to_key[sid] = key
    return key_to_short, short_to_key


# ------------------------------------------------------------------ #
#  Judge instruction template (IMPROVED - per-task scoring, 0-100)   #
# ------------------------------------------------------------------ #
_JUDGE_INSTRUCTIONS = """\
---

## Judge Instructions

You are an expert evaluator of code-generation quality. For each **Config**
section above, assess the quality of **all responses** across all tests.

### Scoring (0–100)

- Assign an **overall score (0–100)** to each configuration, based on:
  - **Correctness** – does the code solve the problem?
  - **Completeness** – does it include docstrings, type hints, etc. as requested?
  - **Clarity** – is the code readable and well-structured?
  - **Failure rate** – if many tests failed (errors), penalise the config,
    but if the few successful ones are excellent, you may still give a moderate score.

Additionally, provide **per-task scores (0–100)** for each of the following
task categories (if present):
- `coding_chunk`
- `algorithmic` (includes palindrome, two_sum, BST)
- `bugfixing`
- `refactoring`
- `integration_api`
- `test_generation`

For categories where all runs failed, give a score of `0`.

### Recommendation

Based on the scores, decide on the **next experimental step**:

- `"conclude"` – one config is clearly the best across most tasks, and further
  refinement is unlikely to improve significantly.
  (Provide the best config ID and its parameters.)
- `"refine"` – a specific parameter region shows promise (e.g., low temperature,
  specific top_p), but you need a finer grid around it.
  (Provide a `refinement_hint` with tighter ranges, e.g.,
  `{"temperature": [0.1, 0.3]}`.)
- `"expand"` – results are inconclusive because most configs failed or performed
  poorly. You need to **broaden the search** (e.g., wider ranges for temperature,
  top_p, or add new parameters like `repeat_penalty`).
  (Provide a hint suggesting which parameters to expand, e.g.,
  `{"temperature": [0.0, 2.0], "top_p": [0.0, 1.0]}`.)

If some tasks have very different optimal parameters, mention that in the
`reasoning` field and optionally include a `best_config_per_task` mapping.

### Output Format

At the **very end** of your reply, output **only** a JSON object with these
fields:

```json
{
  "scores": {
    "config_1": { "overall": 45, "per_task": {"bugfixing": 70, "coding_chunk": 0, ...} },
    "config_2": { "overall": 12, "per_task": {...} },
    ...
  },
  "recommendation": "refine",
  "confidence": "medium",
  "reasoning": "Config 5 shows promise for bugfixing, but coding tasks failed. Need to narrow temperature and top_p.",
  "refinement_hint": { "temperature": [0.1, 0.3], "top_p": [0.3, 0.5] },
  "best_config_per_task": {
    "bugfixing": "config_5",
    "coding_chunk": null
  }
}
```

- **`scores`** – required, with `overall` and optionally `per_task`.
- **`recommendation`** – one of `"conclude"`, `"refine"`, `"expand"`.
- **`refinement_hint`** – only if `recommendation` is `"refine"` or `"expand"`.
  For `"expand"`, provide **wider** ranges than the current ones.
- **`best_config_per_task`** – optional, maps task name to the config ID that
  performed best for that task (or `null` if none).

**Important:** Output **only** the JSON block – nothing before or after it.
Do NOT wrap the JSON in markdown code fences in your actual reply
(the example above shows fences only for clarity).
Reply must be pure text with the JSON as the final block.
"""


def generate_judge_prompt(
    run_dir: Path,
    run_results: list[Any],
    out_path: Path | None = None,
) -> tuple[Path, dict[str, str]]:
    """Write ``judge_prompt.md`` and its companion ``config_map.json``.

    Args:
        run_dir: Directory containing the run (used as default output location).
        run_results: All run results from the benchmark.
        out_path: Optional explicit output path; defaults to ``run_dir/judge_prompt.md``.

    Returns:
        A tuple of ``(judge_prompt_path, short_to_key_map)`` where
        ``short_to_key_map`` maps short IDs (``config_1``) back to full JSON
        config keys.
    """
    from benchmarker.runner import config_key

    run_dir = Path(run_dir)
    out_path = out_path or (run_dir / JUDGE_PROMPT_FILE)

    # Group by config key
    grouped: dict[str, list[Any]] = {}
    for r in run_results:
        grouped.setdefault(config_key(r.config), []).append(r)

    # Filter out configs where ALL runs failed (100% errors)
    filtered_grouped = {}
    for key, items in grouped.items():
        total = len(items)
        errors = sum(1 for r in items if r.error is not None)
        if errors < total:
            filtered_grouped[key] = items
        else:
            # Log that we're skipping this config
            pass

    # If filtering removed everything, fall back to all configs
    if not filtered_grouped:
        filtered_grouped = grouped

    # Assign short IDs to the filtered configs
    key_to_short, short_to_key = _make_short_ids(list(filtered_grouped.keys()))

    # Build a description column: e.g. "temperature=0.7, top_p=0.9"
    def _short_desc(full_key: str) -> str:
        try:
            cfg = json.loads(full_key)
            return ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
        except (json.JSONDecodeError, TypeError):
            return full_key

    # ---------- Header ----------
    sections: list[str] = [
        "# Benchmark Judge Prompt\n",
        f"_Run directory: `{run_dir}`_\n",
        f"_Configurations evaluated: {len(filtered_grouped)}_\n",
        f"_Total test runs: {len(run_results)}_\n",
    ]

    # ---------- Config Summary table ----------
    sections.append("## Config Summary\n")
    sections.append(
        "| Config ID | Parameters | Avg Tok/s | Avg TTFT (s) | Avg Total (s) | Errors |"
    )
    sections.append(
        "|-----------|------------|-----------|--------------|---------------|--------|"
    )
    for key, items in sorted(filtered_grouped.items()):
        sid = key_to_short[key]
        desc = _short_desc(key)
        speeds = [r.tokens_per_sec for r in items if r.error is None]
        ttfts = [r.ttft for r in items if r.error is None]
        totals = [r.total_time for r in items if r.error is None]
        errs = sum(1 for r in items if r.error is not None)
        avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0.0
        avg_total = sum(totals) / len(totals) if totals else 0.0
        sections.append(
            f"| `{sid}` | `{desc}` | {avg_speed:.1f} | {avg_ttft:.3f} "
            f"| {avg_total:.3f} | {errs} |"
        )
    sections.append("")

    # ---------- Detailed Responses ----------
    for key, items in sorted(filtered_grouped.items()):
        sid = key_to_short[key]
        desc = _short_desc(key)
        sections.append(f"\n## Config `{sid}` — {desc} ({len(items)} runs)\n")
        for r in items:
            rlabel = f"Repetition {r.repetition}" if r.repetition > 1 else "Run 1"
            sections.append(f"### Test `{r.test_id}` — {rlabel}\n")
            sections.append(_render_result(r))

    body = "\n".join(sections)
    content = body + "\n\n" + _JUDGE_INSTRUCTIONS
    out_path.write_text(content, encoding="utf-8")

    # ---------- Save config map ----------
    map_path = run_dir / CONFIG_MAP_FILE
    map_path.write_text(
        json.dumps(short_to_key, indent=2, sort_keys=True), encoding="utf-8"
    )

    return out_path, short_to_key