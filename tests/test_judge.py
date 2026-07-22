"""Tests for JudgeClient interface and NullJudge implementation."""

from pathlib import Path
from unittest.mock import patch

import pytest

from benchmarker.judge import JudgeClient, NullJudge


# ------------------------------------------------------------------ #
#  JudgeClient interface                                               #
# ------------------------------------------------------------------ #
class TestJudgeClientInterface:
    def test_cannot_instantiate_abstract(self) -> None:
        """JudgeClient should be abstract and refuse instantiation."""
        with pytest.raises(TypeError):
            JudgeClient()  # type: ignore[abstract]

    def test_null_judge_implements_interface(self) -> None:
        """NullJudge should be a concrete subclass of JudgeClient."""
        judge = NullJudge()
        assert isinstance(judge, JudgeClient)


# ------------------------------------------------------------------ #
#  NullJudge.generate_prompt                                           #
# ------------------------------------------------------------------ #
class TestNullJudgeGeneratePrompt:
    def test_returns_out_path(self, tmp_path: Path) -> None:
        results = [
            _make_result(config={"temperature": 0.7}, test_id="t1", repetition=1),
        ]
        out = tmp_path / "judge_prompt.md"
        judge = NullJudge()
        returned = judge.generate_prompt(tmp_path, results, out_path=out)
        assert returned == out
        assert returned.is_file()

    def test_delegates_to_generate_judge_prompt(self, tmp_path: Path) -> None:
        results = [
            _make_result(config={"temperature": 0.7}, test_id="t1", repetition=1),
        ]
        out = tmp_path / "judge_prompt.md"
        judge = NullJudge()
        with patch("benchmarker.judge.generate_judge_prompt") as mock_gen:
            mock_gen.return_value = (out, {"config_1": '{"temperature": 0.7}'})
            returned = judge.generate_prompt(tmp_path, results, out_path=out)
        mock_gen.assert_called_once_with(tmp_path, results, out_path=out)
        assert returned == out


# ------------------------------------------------------------------ #
#  NullJudge.parse_reply                                               #
# ------------------------------------------------------------------ #
class TestNullJudgeParseReply:
    def test_delegates_to_parse_and_act(self, tmp_path: Path) -> None:
        text = '{"scores": {"c1": {"overall": 8}}, "recommendation": "conclude", "confidence": "high", "reasoning": "ok"}'
        params_path = tmp_path / "params.yaml"
        params_path.write_text("optimizer: {type: grid, budget: 10}\nparameters: []\n")
        judge = NullJudge()
        with patch("benchmarker.judge.parse_and_act") as mock_parse:
            result = judge.parse_reply(text, params_path=params_path, run_dir=tmp_path)
        mock_parse.assert_called_once_with(text, params_path=params_path, run_dir=tmp_path)
        assert result is None

    def test_returns_none(self, tmp_path: Path) -> None:
        text = '{"scores": {"c1": {"overall": 8}}, "recommendation": "conclude", "confidence": "high", "reasoning": "ok"}'
        params_path = tmp_path / "params.yaml"
        params_path.write_text("optimizer: {type: grid, budget: 10}\nparameters: []\n")
        judge = NullJudge()
        result = judge.parse_reply(text, params_path=params_path, run_dir=tmp_path)
        assert result is None


# ------------------------------------------------------------------ #
#  NullJudge.act                                                       #
# ------------------------------------------------------------------ #
class TestNullJudgeAct:
    def test_noop_does_not_raise(self) -> None:
        verdict = _make_verdict()
        params_path = Path("params.yaml")
        judge = NullJudge()
        result = judge.act(verdict, params_path)
        assert result is None

    def test_noop_ignores_verdict(self) -> None:
        """Act should not modify params_path even with a refine verdict."""
        import os
        import tempfile

        verdict = _make_verdict(recommendation="refine")
        with tempfile.TemporaryDirectory() as d:
            params_path = Path(d) / "params.yaml"
            params_path.write_text("parameters:\n  - name: temperature\n")
            judge = NullJudge()
            judge.act(verdict, params_path)
            # File should be unchanged because NullJudge.act is a no-op
            content = params_path.read_text()
            assert content == "parameters:\n  - name: temperature\n"


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #
def _make_result(**kwargs) -> "benchmarker.runner.RunResult":  # type: ignore[name-defined]
    from benchmarker.runner import RunResult

    defaults = dict(
        config={},
        test_id="t1",
        repetition=1,
        prompt="hi",
        response_text="hello",
        ttft=0.1,
        total_time=0.2,
        tokens_per_sec=10.0,
        completion_tokens=1,
        prompt_tokens=1,
        error=None,
        category=None,
        cost_per_1m_input=0.0,
        cost_per_1m_output=0.0,
        config_aborted=False,
        success_rate=None,
        coverage=None,
    )
    defaults.update(kwargs)
    return RunResult(**defaults)


def _make_verdict(**kwargs) -> "benchmarker.parse_judge.JudgeVerdict":  # type: ignore[name-defined]
    from benchmarker.parse_judge import JudgeScore, JudgeVerdict

    defaults = dict(
        scores={"c1": JudgeScore(overall=8, reasoning="ok")},
        recommendation="conclude",
        confidence="high",
        reasoning="ok",
        refinement_hint=None,
        best_config_per_category=None,
    )
    defaults.update(kwargs)
    return JudgeVerdict(**defaults)
