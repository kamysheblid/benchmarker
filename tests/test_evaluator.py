"""Tests for automated code evaluation (Phase 4)."""

from benchmarker.evaluator import (
    CodeExtractor,
    EvalScoreSet,
    HIDDEN_TEST_SUITES,
    StaticAnalysisResult,
    analyze_static,
    check_execution_safety,
    evaluate_response,
    run_unit_tests,
)
from benchmarker.runner import RunResult


# --------------------------------------------------------------------------- #
# CodeExtractor
# --------------------------------------------------------------------------- #

def test_extract_markdown_python_block() -> None:
    text = """Some text

```python
def foo():
    return 42
```

More text."""
    code = CodeExtractor.extract_python(text)
    assert code == "def foo():\n    return 42"


def test_extract_markdown_py_block() -> None:
    text = """```py
x = 1
```"""
    code = CodeExtractor.extract_python(text)
    assert "x = 1" in code


def test_extract_raw_python() -> None:
    text = "def bar(x):\n    return x + 1"
    code = CodeExtractor.extract_python(text)
    assert "def bar" in code
    assert "return x + 1" in code


def test_extract_no_code() -> None:
    assert CodeExtractor.extract_python("Just some plain text.") is None
    assert CodeExtractor.extract_python("") is None


def test_extract_non_python_raw() -> None:
    # Response that looks like prose should not be extracted
    text = "The answer is 42. I think this is correct."
    assert CodeExtractor.extract_python(text) is None


# --------------------------------------------------------------------------- #
# Execution safety
# --------------------------------------------------------------------------- #

def test_check_execution_safety_valid() -> None:
    code = "def add(a, b):\n    return a + b"
    result = check_execution_safety(code)
    assert result.compile_ok is True
    assert result.runtime_ok is True


def test_check_execution_safety_syntax_error() -> None:
    code = "def broken(:"  # invalid syntax
    result = check_execution_safety(code)
    assert result.compile_ok is False
    assert result.runtime_ok is False
    assert result.error is not None


def test_check_execution_safety_empty() -> None:
    result = check_execution_safety("")
    assert result.compile_ok is False
    assert result.error == "empty code"


def test_check_execution_safety_top_level_error() -> None:
    code = "x = undefined_var + 1"  # runs but would fail at import
    result = check_execution_safety(code)
    # compile is fine, but py_compile might not catch NameError at top level
    # Since py_compile only checks syntax, this should pass
    assert result.compile_ok is True


# --------------------------------------------------------------------------- #
# Static analysis
# --------------------------------------------------------------------------- #

def test_analyze_static_function_count() -> None:
    code = "def f1():\n    pass\ndef f2():\n    pass"
    result = analyze_static(code)
    assert result.num_functions == 2
    assert result.num_classes == 0


def test_analyze_static_docstring_detected() -> None:
    code = 'def f():\n    """Docstring."""\n    pass'
    result = analyze_static(code)
    assert result.has_docstring is True


def test_analyze_static_type_hints() -> None:
    code = "def add(a: int, b: int) -> int:\n    return a + b"
    result = analyze_static(code)
    assert result.has_type_hints is True


def test_analyze_static_bare_except() -> None:
    code = "try:\n    x = 1\nexcept:\n    pass"
    result = analyze_static(code)
    assert result.bare_excepts == 1


def test_analyze_static_quality_score_no_issues() -> None:
    code = 'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b'
    result = analyze_static(code)
    assert result.quality_score() >= 0.9


def test_analyze_static_quality_score_bare_except() -> None:
    code = 'def f():\n    """Do something."""\n    try:\n        pass\n    except:\n        pass'
    result = analyze_static(code)
    assert result.quality_score() < 1.0


# --------------------------------------------------------------------------- #
# Unit test runner
# --------------------------------------------------------------------------- #

def test_run_unit_tests_all_pass() -> None:
    code = "def add(a, b):\n    return a + b"
    tests = [
        {"id": "t1", "func": "add", "args": (1, 2), "expected": 3},
        {"id": "t2", "func": "add", "args": (-1, 1), "expected": 0},
    ]
    result = run_unit_tests(code, tests)
    assert result.passed == 2
    assert result.total == 2
    assert result.pass_rate() == 1.0


def test_run_unit_tests_some_fail() -> None:
    code = "def add(a, b):\n    return a - b"  # bug: subtract instead of add
    tests = [
        {"id": "t1", "func": "add", "args": (1, 2), "expected": 3},
        {"id": "t2", "func": "add", "args": (2, 2), "expected": 4},
    ]
    result = run_unit_tests(code, tests)
    assert result.passed < result.total


def test_run_unit_tests_function_missing() -> None:
    code = "def foo():\n    return 1"
    tests = [{"id": "t1", "func": "bar", "args": (), "expected": 1}]
    result = run_unit_tests(code, tests)
    assert result.passed == 0
    assert result.total == 1
    assert len(result.errors) == 1
    assert "not defined" in result.errors[0]


def test_run_unit_tests_empty_code() -> None:
    result = run_unit_tests("", [{"func": "x", "expected": 1}])
    assert result.passed == 0
    assert result.pass_rate() == 0.0


# --------------------------------------------------------------------------- #
# Full evaluation pipeline
# --------------------------------------------------------------------------- #

def test_evaluate_response_no_code() -> None:
    result = RunResult(
        config={},
        test_id="t1",
        repetition=1,
        prompt="",
        response_text="I don't know how to code.",
        ttft=0.0,
        total_time=0.0,
        tokens_per_sec=0.0,
        completion_tokens=0,
        prompt_tokens=0,
    )
    score = evaluate_response(result)
    assert score.code_extracted is False
    assert score.overall == 0.0
    assert len(score.issues) == 1


def test_evaluate_response_good_chunk_code() -> None:
    code = """\
def chunk(seq, n):
    '''Split seq into chunks of size n.'''
    return [seq[i:i + n] for i in range(0, len(seq), n)]\
"""
    result = RunResult(
        config={"temperature": 0.5},
        test_id="coding_chunk",
        repetition=1,
        prompt="",
        response_text=code,
        ttft=0.0,
        total_time=0.0,
        tokens_per_sec=0.0,
        completion_tokens=0,
        prompt_tokens=0,
    )
    score = evaluate_response(result)
    assert score.code_extracted is True
    assert score.exec_safety > 0
    assert score.static_quality > 0
    # Should pass most chunk unit tests
    assert score.unit_pass_rate > 0.5


def test_evaluate_response_no_hidden_tests() -> None:
    result = RunResult(
        config={},
        test_id="creative",
        repetition=1,
        prompt="",
        response_text="A poem about winter.",
        ttft=0.0,
        total_time=0.0,
        tokens_per_sec=0.0,
        completion_tokens=0,
        prompt_tokens=0,
    )
    score = evaluate_response(result)
    # No Python code extracted -> no evaluation
    assert score.code_extracted is False


def test_hidden_test_suites_defined() -> None:
    """Verify that hidden test suites for coding prompts exist."""
    assert "coding_chunk" in HIDDEN_TEST_SUITES
    assert "algorithmic_twosum" in HIDDEN_TEST_SUITES
    assert "bugfixing" in HIDDEN_TEST_SUITES
    assert len(HIDDEN_TEST_SUITES["coding_chunk"]) > 0
