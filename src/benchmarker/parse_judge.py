"""Parse a judge's response and produce a structured recommendation.

Usage from CLI:

.. code-block:: bash

    benchmarker parse <judge_response_file> [--run-dir <path>]

The judge's reply must end with a JSON block (the last JSON-like object in
the text).  If ``--run-dir`` is provided the tool also loads the companion
``config_map.json`` so config short IDs (``config_1``) are translated back
to their real parameter values.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class JudgeScore:
    """Score for a single configuration."""

    overall: int
    reasoning: str
    per_category: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.overall <= 100:
            msg = f"Score must be between 0 and 100, got {self.overall}"
            raise ValueError(msg)

    @property
    def per_task(self) -> dict[str, int] | None:
        """Backward-compat alias for per_category."""
        return self.per_category


@dataclass
class JudgeVerdict:
    """Parsed verdict from the judge response."""

    scores: dict[str, JudgeScore]  # short ID or full-key -> score
    recommendation: str  # "conclude" | "refine" | "expand"
    confidence: str  # "high" | "medium" | "low"
    reasoning: str
    refinement_hint: dict[str, list[float]] | None = None
    best_config_per_category: dict[str, str | None] | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> JudgeVerdict:
        scores = {}
        for sid, sdata in data.get("scores", {}).items():
            per_category = sdata.get("per_category") or sdata.get("per_task")
            scores[sid] = JudgeScore(
                overall=sdata["overall"],
                reasoning=sdata.get("reasoning", ""),
                per_category=per_category,
            )
        return cls(
            scores=scores,
            recommendation=data["recommendation"],
            confidence=data.get("confidence", "medium"),
            reasoning=data.get("reasoning", ""),
            refinement_hint=data.get("refinement_hint"),
            best_config_per_category=data.get("best_config_per_category") or data.get("best_config_per_task"),
        )

    @property
    def best_config_per_task(self) -> dict[str, str | None] | None:
        """Backward-compat alias for best_config_per_category."""
        return self.best_config_per_category


def validate_judge_response(data: dict[str, Any]) -> None:
    """Validate a judge response dict (public API for backward compat).

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    required = ["scores", "recommendation", "confidence", "reasoning"]
    for field in required:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    if not data["scores"]:
        raise ValueError("scores must be non-empty")

    valid_recs = {"conclude", "refine", "expand"}
    if data["recommendation"] not in valid_recs:
        raise ValueError(f"Invalid recommendation: {data['recommendation']}")

    valid_conf = {"high", "medium", "low"}
    if data["confidence"] not in valid_conf:
        raise ValueError(f"Invalid confidence: {data['confidence']}")

    for sid, sdata in data["scores"].items():
        if "overall" not in sdata:
            raise ValueError(f"Score for {sid} missing 'overall'")
        overall = sdata["overall"]
        if not isinstance(overall, int) or not 0 <= overall <= 100:
            raise ValueError(f"Score overall out of range (0-100): {overall}")

        per_cat = sdata.get("per_category") or sdata.get("per_task")
        if per_cat is not None:
            if not isinstance(per_cat, dict):
                raise ValueError(f"Score per_category for {sid} must be a dict")
            for cat_name, cat_score in per_cat.items():
                if not isinstance(cat_score, int) or not 0 <= cat_score <= 100:
                    raise ValueError(
                        f"Score per_category['{cat_name}'] out of range (0-100): {cat_score}"
                    )


def extract_json(text: str) -> dict[str, Any]:
    """Extract the last JSON-object from *text* (public API for backward compat)."""
    return _extract_json(text)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the last JSON-object from *text* (greedy)."""
    # Find everything that looks like a JSON object (balanced braces)
    # Strategy: try json.loads on increasing suffixes, then regex-fallback.
    # Walk backwards from the end looking for the first '{'.
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}":
            if not stack:
                continue
            start = stack.pop()
            if not stack:  # outermost brace pair
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    # If we can't parse the JSON, we need to differentiate between
                    # "no JSON found" and "malformed JSON"
                    msg = f"Failed to parse JSON inside braces: {e}"
                    raise ValueError(msg) from e

    # Check if we found any valid JSON object
    if not stack:
        # No balanced braces found
        msg = "No JSON object found in judge response"
        raise ValueError(msg)
    
    # If we get here, we have unbalanced braces
    msg = "Unbalanced braces: never closed"
    raise ValueError(msg)


def _resolve_config(
    short_id: str,
    config_map: dict[str, str] | None,
) -> str:
    """Translate a short ID to a human-readable config description."""
    if config_map is None:
        return short_id
    full_key = config_map.get(short_id)
    if full_key is None:
        return short_id
    try:
        cfg = json.loads(full_key)
        return ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
    except (json.JSONDecodeError, TypeError):
        return full_key


def parse_judge_response(
    response_text: str,
    config_map: dict[str, str] | None = None,
) -> JudgeVerdict:
    """Parse a judge's free-text response into a structured verdict.

    Args:
        response_text: The full text of the judge's reply.
        config_map: Optional mapping from short ID to full JSON config key.
            When provided, scores are re-keyed to show the description.

    Returns:
        A :class:`JudgeVerdict` instance.
    """
    raw = _extract_json(response_text)
    verdict = JudgeVerdict.from_json(raw)

    # Resolve config short IDs if a mapping is available
    if config_map:
        resolved: dict[str, JudgeScore] = {}
        for sid, score in verdict.scores.items():
            desc = _resolve_config(sid, config_map)
            resolved[desc] = score
        verdict.scores = resolved

    return verdict


def load_config_map(run_dir: str | Path) -> dict[str, str] | None:
    """Load ``config_map.json`` from *run_dir*, returning ``None`` if absent."""
    p = Path(run_dir) / "config_map.json"
    if not p.exists():
        return None
    try:
        with p.open("r") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def format_verdict(verdict: JudgeVerdict) -> str:
    """Pretty-print a :class:`JudgeVerdict` for console output."""
    lines = [
        "=" * 55,
        "  Judge Verdict",
        "=" * 55,
        "",
        f"Recommendation: {verdict.recommendation.upper()}",
        f"Confidence:     {verdict.confidence}",
        f"Reasoning:      {verdict.reasoning}",
        "",
        "Scores:",
    ]
    for sid, score in verdict.scores.items():
        per_cat_str = ""
        if score.per_category:
            per_cat_str = "  (per-category: " + ", ".join(f"{k}={v}" for k, v in score.per_category.items()) + ")"
        lines.append(f"  {sid:>36}  {score.overall}/100  — {score.reasoning}{per_cat_str}")
    if verdict.refinement_hint:
        lines.append("")
        lines.append("Refinement hint:")
        for param, values in verdict.refinement_hint.items():
            formatted = ", ".join(str(v) for v in values)
            lines.append(f"  {param}: [{formatted}]")
    if verdict.best_config_per_category:
        lines.append("")
        lines.append("Best config per category:")
        for cat, cfg in verdict.best_config_per_category.items():
            lines.append(f"  {cat}: {cfg or 'none'}")
    lines.append("")
    lines.append("=" * 55)
    return "\n".join(lines)


# ------------------------------------------------------------------ #
#  Params YAML helpers                                               #
# ------------------------------------------------------------------ #
def _load_params_yaml(path: Path) -> dict[str, Any]:
    """Load params.yaml as a raw dict (preserving structure)."""
    if not path.exists():
        return {"optimizer": {"type": "bayesian", "budget": 20}, "parameters": [], "static_params": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_params_yaml(path: Path, data: dict[str, Any]) -> None:
    """Save params.yaml, preserving key order as much as possible."""
    path.write_text(yaml.safe_dump(data, default_flow_style=False, sort_keys=False), encoding="utf-8")


def _find_param(params: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    """Find a parameter spec by name."""
    for p in params:
        if p.get("name") == name:
            return p
    return None


def _update_params_for_refine(
    params_data: dict[str, Any],
    verdict: JudgeVerdict,
    config_map: dict[str, str] | None,
) -> dict[str, Any]:
    """Narrow parameter ranges based on refinement hint or best config."""
    params = params_data.get("parameters", [])
    hint = verdict.refinement_hint or {}

    if hint:
        # Apply specific hint ranges
        for param_name, new_range in hint.items():
            p = _find_param(params, param_name)
            if not p:
                continue
            if len(new_range) >= 2:
                new_low, new_high = new_range[0], new_range[1]
                p["low"] = new_low
                p["high"] = new_high
                # Reduce step size for finer grid (halve it)
                if "step" in p and p["step"] is not None:
                    p["step"] = p["step"] / 2
    else:
        # No hint: infer from best overall config
        if verdict.scores:
            best_desc = max(verdict.scores.items(), key=lambda x: x[1].overall)[0]
            # Try to find the config in config_map to get actual values
            if config_map:
                # Reverse lookup: find short_id for this description
                short_id = None
                for sid, full_key in config_map.items():
                    try:
                        cfg = json.loads(full_key)
                        desc = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items()))
                        if desc == best_desc:
                            short_id = sid
                            break
                    except (json.JSONDecodeError, TypeError):
                        continue
                if short_id and short_id in config_map:
                    full_key = config_map[short_id]
                    try:
                        best_cfg = json.loads(full_key)
                        for param_name, value in best_cfg.items():
                            p = _find_param(params, param_name)
                            if not p:
                                continue
                            # Narrow to ±20% around best value
                            span = p["high"] - p["low"]
                            margin = max(span * 0.2, 0.1 * abs(value) if value != 0 else 0.1)
                            p["low"] = max(p["low"], value - margin)
                            p["high"] = min(p["high"], value + margin)
                            if "step" in p and p["step"] is not None:
                                p["step"] = p["step"] / 2
                    except (json.JSONDecodeError, TypeError):
                        pass

    params_data["parameters"] = params
    return params_data


def _update_params_for_expand(
    params_data: dict[str, Any],
    verdict: JudgeVerdict,
) -> dict[str, Any]:
    """Widen parameter ranges based on expansion hint or default broadening."""
    params = params_data.get("parameters", [])
    hint = verdict.refinement_hint or {}

    if hint:
        # Apply specific hint ranges (wider than current)
        for param_name, new_range in hint.items():
            p = _find_param(params, param_name)
            if not p:
                continue
            if len(new_range) >= 2:
                new_low, new_high = new_range[0], new_range[1]
                p["low"] = new_low
                p["high"] = new_high
                # Increase step size (coarser grid)
                if "step" in p and p["step"] is not None:
                    p["step"] = p["step"] * 2
    else:
        # Default: broaden all numeric parameters by 50% on each side
        for p in params:
            if p.get("type") in ("float", "int") and p.get("low") is not None and p.get("high") is not None:
                low = p["low"]
                high = p["high"]
                span = high - low
                # Expand by 50% on each side
                if p["type"] == "float":
                    p["low"] = max(0, low - 0.5 * span)
                    p["high"] = high + 0.5 * span
                else:
                    p["low"] = max(0, low - int(0.5 * span))
                    p["high"] = high + int(0.5 * span)
                # Increase step size
                if "step" in p and p["step"] is not None:
                    p["step"] = p["step"] * 2

    params_data["parameters"] = params
    return params_data


def handle_judge_reply(
    text: str,
    params_path: str | Path = "params.yaml",
    run_dir: str | Path | None = None,
    benchmark_source: Path | None = None,
) -> JudgeVerdict:
    """Parse judge reply, print verdict, and update params.yaml or benchmark YAML if needed.

    Args:
        text: Raw judge reply text.
        params_path: Path to params.yaml to update (for refine/expand).
        run_dir: Optional run directory to load config_map.json from.
        benchmark_source: Optional benchmark YAML file to update in place.
            If provided, takes precedence over params_path for refine/expand.

    Returns:
        The parsed :class:`JudgeVerdict`.
    """
    config_map = load_config_map(run_dir) if run_dir else None
    verdict = parse_judge_response(text, config_map=config_map)

    # Print the verdict
    print(format_verdict(verdict))

    # Act based on recommendation
    if verdict.recommendation == "conclude":
        # Find and print the best config
        if verdict.scores:
            best_desc = max(verdict.scores.items(), key=lambda x: x[1].overall)[0]
            best_score = verdict.scores[best_desc].overall
            print(f"\n✅ Concluding: Best config is '{best_desc}' with score {best_score}/100")
            print("No changes made to params.yaml. Use these parameters for production.")
        else:
            print("\n⚠️  No scores available to determine best config.")

    elif verdict.recommendation in ("refine", "expand"):
        if benchmark_source is not None:
            print(f"\n🔧 Updating benchmark file {benchmark_source}...")
            benchmark_data = yaml.safe_load(benchmark_source.read_text(encoding="utf-8")) or {}
            if not isinstance(benchmark_data, dict):
                benchmark_data = {}
            if verdict.recommendation == "refine":
                updated = _update_params_for_refine(benchmark_data, verdict, config_map)
            else:
                updated = _update_params_for_expand(benchmark_data, verdict)
            benchmark_source.write_text(
                yaml.safe_dump(updated, default_flow_style=False, sort_keys=False),
                encoding="utf-8",
            )
            print(f"Updated {benchmark_source} with {'narrowed' if verdict.recommendation == 'refine' else 'widened'} ranges.")
            print("Run 'benchmarker run' again to continue.")
        else:
            params_path = Path(params_path)
            if verdict.recommendation == "refine":
                print("\n🔧 Refining parameter ranges...")
                params_data = _load_params_yaml(params_path)
                updated = _update_params_for_refine(params_data, verdict, config_map)
                _save_params_yaml(params_path, updated)
                print(f"Updated {params_path} with narrowed ranges and finer steps.")
                print("Run 'benchmarker run' again to continue.")
            else:
                print("\n📈 Expanding parameter ranges...")
                params_data = _load_params_yaml(params_path)
                updated = _update_params_for_expand(params_data, verdict)
                _save_params_yaml(params_path, updated)
                print(f"Updated {params_path} with widened ranges.")
                print("Run 'benchmarker run' again to continue.")

    else:
        print(f"\n⚠️  Unknown recommendation: {verdict.recommendation}")

    return verdict


# Backward compatibility
def do_parse(input_path: str, run_dir: str | None = None) -> JudgeVerdict:
    """CLI runner for ``benchmarker parse <input> [--run-dir <dir>]``."""
    text = Path(input_path).read_text(encoding="utf-8")
    config_map = load_config_map(run_dir) if run_dir else None
    verdict = parse_judge_response(text, config_map=config_map)
    print(format_verdict(verdict))
    return verdict


def parse_and_act(
    text: str,
    params_path: str | Path | None = None,
    run_dir: str | Path | None = None,
    benchmark_source: Path | None = None,
) -> JudgeVerdict:
    """Parse judge reply and print verdict (standalone / CLI entry).

    Args:
        text: Raw judge reply text.
        params_path: Path to params.yaml to update when refining/expanding.
        run_dir: Optional run directory to load ``config_map.json`` from.
        benchmark_source: Optional benchmark YAML file to update when refining/expanding.

    Returns:
        The parsed :class:`JudgeVerdict`.
    """
    return handle_judge_reply(
        text,
        params_path=params_path or "params.yaml",
        run_dir=run_dir,
        benchmark_source=benchmark_source,
    )