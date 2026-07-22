"""Tests for judge reply parsing and action dispatch."""

from pathlib import Path

import pytest
import yaml

from benchmarker.parse_judge import (
    extract_json,
    format_verdict,
    parse_and_act,
    validate_judge_response,
    JudgeScore,
    JudgeVerdict,
)


# ------------------------------------------------------------------ #
#  extract_json                                                       #
# ------------------------------------------------------------------ #
class TestExtractJson:
    def test_plain_json(self) -> None:
        text = '{"scores": {}, "recommendation": "conclude", "confidence": "high", "reasoning": "ok"}'
        result = extract_json(text)
        assert result["recommendation"] == "conclude"

    def test_with_markdown_fences(self) -> None:
        text = """Some prose here.

```json
{"scores": {}, "recommendation": "refine", "confidence": "medium", "reasoning": "adjusted"}
```

More text after."""
        result = extract_json(text)
        assert result["recommendation"] == "refine"

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON object"):
            extract_json("this has no braces at all")

    def test_unbalanced_raises(self) -> None:
        with pytest.raises(ValueError, match="never closed"):
            extract_json('{"scores": {}')

    def test_malformed_json_inside_braces(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            extract_json('{"scores": broken, "recommendation": "conclude"}')


# ------------------------------------------------------------------ #
#  validate_judge_response                                            #
# ------------------------------------------------------------------ #
class TestValidate:
    def test_valid_full(self) -> None:
        data = {
            "scores": {"cfg1": {"overall": 8}},
            "recommendation": "conclude",
            "confidence": "high",
            "reasoning": "All good.",
        }
        validate_judge_response(data)  # no raise

    def test_missing_fields(self) -> None:
        with pytest.raises(ValueError, match="Missing required"):
            validate_judge_response({"scores": {}})

    def test_bad_recommendation(self) -> None:
        with pytest.raises(ValueError, match="Invalid recommendation"):
            validate_judge_response({
                "scores": {"cfg1": {"overall": 5}},
                "recommendation": "unknown",
                "confidence": "high",
                "reasoning": "bad",
            })

    def test_bad_confidence(self) -> None:
        with pytest.raises(ValueError, match="Invalid confidence"):
            validate_judge_response({
                "scores": {"cfg1": {"overall": 5}},
                "recommendation": "conclude",
                "confidence": "very",
                "reasoning": "bad",
            })

    def test_out_of_range_score(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            validate_judge_response({
                "scores": {"cfg1": {"overall": 101}},
                "recommendation": "conclude",
                "confidence": "high",
                "reasoning": "too high",
            })

    def test_empty_scores(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            validate_judge_response({
                "scores": {},
                "recommendation": "conclude",
                "confidence": "high",
                "reasoning": "empty",
            })


# ------------------------------------------------------------------ #
#  parse_and_act (integration-style)                                  #
# ------------------------------------------------------------------ #
class TestParseAndAct:
    def test_conclude_output(self, capsys: pytest.CaptureFixture) -> None:
        text = """{
            "scores": {"cfg_a": {"overall": 9, "reasoning": "best"}, "cfg_b": {"overall": 5}},
            "recommendation": "conclude",
            "confidence": "high",
            "reasoning": "Config A is clearly better.",
            "refinement_hint": null
        }"""
        parse_and_act(text)
        captured = capsys.readouterr().out
        assert "CONCLUDE" in captured
        assert "cfg_a" in captured
        assert "Best config" in captured

    def test_expand_output(self, capsys: pytest.CaptureFixture) -> None:
        text = """{
            "scores": {"cfg_x": {"overall": 4}, "cfg_y": {"overall": 5}},
            "recommendation": "expand",
            "confidence": "low",
            "reasoning": "Results are too close.",
            "refinement_hint": null
        }"""
        parse_and_act(text)
        captured = capsys.readouterr().out
        assert "EXPAND" in captured
        assert "widened" in captured.lower()

    def test_refine_updates_params(self, tmp_path: Path) -> None:
        params_path = tmp_path / "params.yaml"
        params_path.write_text(yaml.safe_dump({
            "optimizer": {"type": "grid", "budget": 10},
            "parameters": [
                {"name": "temperature", "type": "float", "low": 0.0, "high": 1.5, "step": 0.1},
            ],
        }))
        text = """{
            "scores": {"cfg_z": {"overall": 6}},
            "recommendation": "refine",
            "confidence": "medium",
            "reasoning": "Promising region near 0.7-0.9.",
            "refinement_hint": {"temperature": [0.7, 0.9]}
        }"""
        parse_and_act(text, params_path=params_path)
        captured = _stdout_last(capsys=None) or "REFINE"
        # The params.yaml should have been updated
        updated = yaml.safe_load(params_path.read_text())
        assert updated["parameters"][0]["low"] == 0.7
        assert updated["parameters"][0]["high"] == 0.9

    def test_refine_infers_hint_when_missing(self, tmp_path: Path) -> None:
        params_path = tmp_path / "params.yaml"
        params_path.write_text(yaml.safe_dump({
            "optimizer": {"type": "grid", "budget": 10},
            "parameters": [
                {"name": "temperature", "type": "float", "low": 0.0, "high": 1.5, "step": 0.1},
            ],
        }))
        # Write a config_map so the parser can resolve config_1
        import json
        config_map_path = tmp_path / "config_map.json"
        config_map_path.write_text(json.dumps({
            "config_1": '{"temperature": 0.8}'
        }))
        # No refinement_hint provided — should infer from best config key
        text = """{
            "scores": {"config_1": {"overall": 8}},
            "recommendation": "refine",
            "confidence": "medium",
            "reasoning": "Narrow down.",
            "refinement_hint": null
        }"""
        parse_and_act(text, params_path=params_path, run_dir=tmp_path)
        updated = yaml.safe_load(params_path.read_text())
        # Should have narrowed around 0.8 (span 1.5, 20% = 0.3, so 0.5-1.1)
        assert updated["parameters"][0]["low"] >= 0.5
        assert updated["parameters"][0]["high"] <= 1.1

    def test_malformed_judge_reply(self) -> None:
        with pytest.raises(ValueError):
            parse_and_act("this is not json at all")


# ------------------------------------------------------------------ #
#  per_category support                                                #
# ------------------------------------------------------------------ #
class TestPerCategory:
    def test_judge_score_accepts_per_category(self) -> None:
        score = JudgeScore(overall=80, reasoning="ok", per_category={"clarity": 90, "accuracy": 70})
        assert score.per_category == {"clarity": 90, "accuracy": 70}

    def test_judge_score_backward_compat_per_task(self) -> None:
        score = JudgeScore(overall=80, reasoning="ok", per_category={"clarity": 90})
        assert score.per_task == {"clarity": 90}

    def test_judge_verdict_from_json_per_category(self) -> None:
        data = {
            "scores": {"cfg1": {"overall": 85, "reasoning": "good", "per_category": {"clarity": 90}}},
            "recommendation": "conclude",
            "confidence": "high",
            "reasoning": "ok",
        }
        verdict = JudgeVerdict.from_json(data)
        assert verdict.scores["cfg1"].per_category == {"clarity": 90}

    def test_judge_verdict_from_json_backward_compat_per_task(self) -> None:
        data = {
            "scores": {"cfg1": {"overall": 85, "reasoning": "good", "per_task": {"clarity": 90}}},
            "recommendation": "conclude",
            "confidence": "high",
            "reasoning": "ok",
        }
        verdict = JudgeVerdict.from_json(data)
        assert verdict.scores["cfg1"].per_category == {"clarity": 90}

    def test_validate_per_category_values(self) -> None:
        data = {
            "scores": {"cfg1": {"overall": 85, "per_category": {"clarity": 90, "accuracy": 70}}},
            "recommendation": "conclude",
            "confidence": "high",
            "reasoning": "ok",
        }
        validate_judge_response(data)  # should not raise

    def test_validate_per_category_out_of_range(self) -> None:
        data = {
            "scores": {"cfg1": {"overall": 85, "per_category": {"clarity": 101}}},
            "recommendation": "conclude",
            "confidence": "high",
            "reasoning": "ok",
        }
        with pytest.raises(ValueError, match="per_category"):
            validate_judge_response(data)

    def test_format_verdict_displays_per_category(self) -> None:
        verdict = JudgeVerdict(
            scores={"cfg1": JudgeScore(overall=85, reasoning="good", per_category={"clarity": 90})},
            recommendation="conclude",
            confidence="high",
            reasoning="ok",
        )
        out = format_verdict(verdict)
        assert "per-category" in out or "per_category" in out
        assert "clarity=90" in out

    def test_best_config_per_category(self) -> None:
        verdict = JudgeVerdict(
            scores={"cfg1": JudgeScore(overall=85, reasoning="good")},
            recommendation="conclude",
            confidence="high",
            reasoning="ok",
            best_config_per_category={"clarity": "cfg1", "accuracy": "cfg2"},
        )
        assert verdict.best_config_per_category == {"clarity": "cfg1", "accuracy": "cfg2"}

    def test_best_config_per_category_backward_compat(self) -> None:
        verdict = JudgeVerdict(
            scores={"cfg1": JudgeScore(overall=85, reasoning="good")},
            recommendation="conclude",
            confidence="high",
            reasoning="ok",
            best_config_per_category={"clarity": "cfg1"},
        )
        assert verdict.best_config_per_task == {"clarity": "cfg1"}


def _stdout_last(capsys: pytest.CaptureFixture | None) -> str:
    """Helper to capture stdout if capsys is available."""
    if capsys:
        return capsys.readouterr().out
    return ""
