"""Parse the judge's reply and act on the recommendation (conclude / refine / expand).

Replaces the old ``import_scores`` workflow with a simpler two-step loop:
1. ``benchmarker run`` → produces ``judge_prompt.md``.
2. User pastes judge reply → ``benchmarker parse reply.txt`` → this module.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from benchmarker.config import load_params
from benchmarker.optimizers import create_optimizer


# ------------------------------------------------------------------ #
#  JSON extraction                                                   #
# ------------------------------------------------------------------ #
def extract_json(text: str) -> dict[str, Any]:
    """Extract the outermost ``{ ... }`` JSON object from *text*.

    Strips markdown code fences, leading/trailing whitespace, then parses
    the first valid JSON object found.

    Raises:
        ValueError: if no valid JSON object can be extracted.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # Try to find outermost { ... }
    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No JSON object found in the judge's reply — no opening '{' detected.")

    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                raw = cleaned[start : i + 1]
                try:
                    return json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Found JSON boundaries but failed to parse: {exc}\n"
                        f"Raw snippet:\n{raw[:500]}"
                    ) from exc
    raise ValueError(
        "JSON object was opened but never closed — unbalanced braces.\n"
        f"Text around start:\n{cleaned[start:start + 200]}"
    )


# ------------------------------------------------------------------ #
#  Validation                                                        #
# ------------------------------------------------------------------ #
_REQUIRED_TOP = {"scores", "recommendation", "confidence", "reasoning"}
_VALID_RECOMMENDATIONS = {"conclude", "refine", "expand"}
_VALID_CONFIDENCE = {"high", "medium", "low"}


def validate_judge_response(data: dict[str, Any]) -> None:
    """Validate that *data* has all required fields and legal values.

    Raises:
        ValueError: describing the first validation failure.
    """
    missing = _REQUIRED_TOP - set(data.keys())
    if missing:
        raise ValueError(f"Missing required top-level fields: {missing}")

    rec = data.get("recommendation", "")
    if rec not in _VALID_RECOMMENDATIONS:
        raise ValueError(
            f"Invalid recommendation {rec!r}. Must be one of {_VALID_RECOMMENDATIONS}"
        )

    conf = data.get("confidence", "")
    if conf not in _VALID_CONFIDENCE:
        raise ValueError(
            f"Invalid confidence {conf!r}. Must be one of {_VALID_CONFIDENCE}"
        )

    scores = data.get("scores", {})
    if not isinstance(scores, dict) or not scores:
        raise ValueError("'scores' must be a non-empty dict of config_key → score objects.")

    for cfg_key, score_entry in scores.items():
        if not isinstance(score_entry, dict):
            raise ValueError(f"Score entry for {cfg_key!r} must be an object, got {type(score_entry).__name__}")
        if "overall" not in score_entry:
            raise ValueError(f"Score entry for {cfg_key!r} is missing 'overall' field.")
        try:
            val = float(score_entry["overall"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Score entry for {cfg_key!r} has non-numeric 'overall': {exc}") from exc
        if not (1 <= val <= 10):
            raise ValueError(f"Score entry for {cfg_key!r} has 'overall' out of range [1,10]: {val}")


# ------------------------------------------------------------------ #
#  Action implementations                                            #
# ------------------------------------------------------------------ #
def _action_conclude(data: dict[str, Any]) -> None:
    """Print final summary with the best configuration."""
    scores = data["scores"]
    best_cfg = max(scores, key=lambda k: float(scores[k].get("overall", 0)))
    best_score = float(scores[best_cfg]["overall"])
    print("=" * 60)
    print("  BENCHMARK COMPLETE — RECOMMENDATION: CONCLUDE")
    print("=" * 60)
    print()
    print(f"  Best configuration:  {best_cfg}")
    print(f"  Judge score:         {best_score:.1f} / 10")
    print(f"  Confidence:          {data.get('confidence', 'N/A')}")
    print(f"  Judge reasoning:     {data.get('reasoning', 'N/A')}")
    if scores[best_cfg].get("reasoning"):
        print(f"  Why this config:    {scores[best_cfg]['reasoning']}")
    print()
    print("  All scores:")
    for ck, se in sorted(scores.items(), key=lambda x: float(x[1].get("overall", 0)), reverse=True):
        print(f"    {ck}: {float(se.get('overall', 0)):.1f}/10 — {se.get('reasoning', '')}")
    print()
    print("  Suggested next step: use these parameters in production.")
    print("=" * 60)


def _action_refine(data: dict[str, Any], params_path: Path) -> None:
    """Narrow the parameter space using the refinement hint and update params.yaml."""
    scores = data["scores"]
    hint = data.get("refinement_hint")
    if not hint:
        # Infer from top-scoring config: take the best config's values and expand
        # slightly around them.
        best_cfg_key = max(scores, key=lambda k: float(scores[k].get("overall", 0)))
        hint = _infer_hint_from_config(best_cfg_key)

    print("=" * 60)
    print("  BENCHMARK REFINING — RECOMMENDATION: REFINE")
    print("=" * 60)
    print()
    print(f"  Refinement hint: {hint}")
    print()

    # Load current params, build AdaptiveOptimizer, write back
    params = load_params(params_path)
    opt = create_optimizer(
        params.optimizer,
        params.parameters,
        refinement_hint=hint,
        resolution_factor=5,
    )
    refined_combos = list(opt)

    # Update params.yaml with narrowed ranges
    for p in params.parameters:
        h = hint.get(p.name)
        if h and len(h) >= 2:
            p.low = float(h[0])
            p.high = float(h[1])
            step = (float(h[1]) - float(h[0])) / 5
            p.step = step

    params.optimizer.type = "grid"
    params.optimizer.budget = len(refined_combos)

    raw = {
        "optimizer": {"type": params.optimizer.type, "budget": params.optimizer.budget},
        "parameters": [
            {
                "name": p.name,
                "type": p.type.value,
                "low": float(p.low) if p.low is not None else None,
                "high": float(p.high) if p.high is not None else None,
                "step": float(p.step) if p.step is not None else None,
                "choices": p.choices,
            }
            for p in params.parameters
        ],
        "static_params": params.static_params,
    }
    params_path.write_text(yaml.safe_dump(raw, default_flow_style=False), encoding="utf-8")
    print(f"  Updated {params_path} with narrowed parameter ranges.")
    print(f"  New search space: {len(refined_combos)} combinations.")
    print()
    print("  Run `benchmarker run` again to continue.")
    print("=" * 60)


def _action_expand(data: dict[str, Any]) -> None:
    """Print a message suggesting to broaden the search."""
    print("=" * 60)
    print("  BENCHMARK INCONCLUSIVE — RECOMMENDATION: EXPAND")
    print("=" * 60)
    print()
    print(f"  Judge reasoning: {data.get('reasoning', 'N/A')}")
    print()
    print("  Suggestions:")
    print("    1. Broaden the parameter ranges in params.yaml and re-run.")
    print("    2. Add more diverse test prompts to tests.json.")
    print("    3. Increase the number of repetitions for statistical significance.")
    print()
    print("  Then run `benchmarker run` again.")
    print("=" * 60)


def _infer_hint_from_config(config_key_str: str) -> dict[str, list[float]]:
    """Parse a config key like ``{\"temperature\": 0.7, \"top_p\": 0.9}`` and
    produce a refinement hint with a ±20% window around each value."""
    hint: dict[str, list[float]] = {}
    try:
        cfg = json.loads(config_key_str)
    except json.JSONDecodeError:
        return hint
    for name, val in cfg.items():
        if isinstance(val, (int, float)):
            margin = abs(val) * 0.2 if val != 0 else 0.1
            hint[name] = [round(val - margin, 4), round(val + margin, 4)]
    return hint


# ------------------------------------------------------------------ #
#  Public API                                                        #
# ------------------------------------------------------------------ #
def parse_and_act(
    judge_response_text: str,
    params_path: Path = Path("params.yaml"),
) -> None:
    """Parse the judge's reply and take action (conclude / refine / expand).

    Args:
        judge_response_text: Raw text of the judge's reply (may contain
            markdown, prose explanation, etc. — JSON is extracted automatically).
        params_path: Path to the parameter YAML to update when refining.
    """
    data = extract_json(judge_response_text)
    validate_judge_response(data)

    rec = data["recommendation"]

    if rec == "conclude":
        _action_conclude(data)
    elif rec == "refine":
        _action_refine(data, params_path)
    elif rec == "expand":
        _action_expand(data)


# ------------------------------------------------------------------ #
#  CLI entry point                                                   #
# ------------------------------------------------------------------ #
def main() -> None:
    """Entry point for ``benchmarker parse [reply_file]``."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"Error: file not found: {path}", file=sys.stderr)
            sys.exit(1)
        text = path.read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    try:
        parse_and_act(text)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
