"""Automated code evaluation for benchmark responses (Phase 4).

Extracts Python code from model responses and scores it on:
- Compilation & execution safety
- Unit test correctness
- Lightweight static-analysis metrics
"""

from __future__ import annotations

import ast
import py_compile
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any

from benchmarker.runner import RunResult


# --------------------------------------------------------------------------- #
# Code extraction
# --------------------------------------------------------------------------- #

class CodeExtractionError(ValueError):
    """Raised when Python code cannot be extracted from a response."""


class CodeExtractor:
    """Extract the first Python code block from a response string."""

    MARKER_PATTERNS = [
        ("```python\n", "\n```"),
        ("```py\n", "\n```"),
        ("```\n", "\n```"),
    ]

    @staticmethod
    def extract_python(text: str) -> str | None:
        """Return the first Python code block found, or ``None``."""
        if not text:
            return None

        # Try markdown code blocks first
        for start_marker, end_marker in CodeExtractor.MARKER_PATTERNS:
            start = text.find(start_marker)
            if start >= 0:
                content_start = start + len(start_marker)
                end = text.find(end_marker, content_start)
                if end >= 0:
                    code = text[content_start:end].strip()
                    if code:
                        return code

        # Fallback: treat the entire response as code if it looks like Python:
        # starts with import/from/def/class and has no obvious markdown
        stripped = text.strip()
        if stripped and not stripped.startswith("```"):
            lines = stripped.splitlines()
            # Heuristic: first non-empty line starts a Python keyword
            first_line = next((l for l in lines if l.strip()), "")
            py_keywords = ("import ", "from ", "def ", "class ", "@", "#")
            if first_line.lstrip().startswith(py_keywords):
                return stripped

        return None


# --------------------------------------------------------------------------- #
# Compilation & execution safety
# --------------------------------------------------------------------------- #

class ExecSafetyResult:
    """Result of executing a code snippet safely."""

    def __init__(
        self,
        compile_ok: bool = False,
        runtime_ok: bool = False,
        error: str | None = None,
    ) -> None:
        self.compile_ok = compile_ok
        self.runtime_ok = runtime_ok
        self.error = error

    def score(self) -> float:
        """0.0 (failure) to 1.0 (perfect safety)."""
        if self.compile_ok and self.runtime_ok:
            return 1.0
        if self.compile_ok:
            return 0.5
        return 0.0


def check_execution_safety(code: str) -> ExecSafetyResult:
    """Check that *code* compiles and can be imported without crashing.

    Uses a tempfile + subprocess import to catch runtime errors from top-level
    code (e.g. syntax errors, NameError on undefined variables at module scope).
    """
    if not code.strip():
        return ExecSafetyResult(compile_ok=False, runtime_ok=False, error="empty code")

    # compile check
    try:
        compile(code, "<eval>", "exec")
    except SyntaxError as exc:
        return ExecSafetyResult(compile_ok=False, runtime_ok=False, error=str(exc))

    compile_ok = True

    # runtime check — import in a fresh tempfile
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mod_path = Path(tmp) / "_eval_mod.py"
            mod_path.write_text(code, encoding="utf-8")
            py_compile.compile(str(mod_path), doraise=True)
    except py_compile.PyCompileError as exc:
        return ExecSafetyResult(compile_ok=True, runtime_ok=False, error=str(exc))

    return ExecSafetyResult(compile_ok=True, runtime_ok=True)


# --------------------------------------------------------------------------- #
# Static analysis (lightweight, no pylint dependency)
# --------------------------------------------------------------------------- #

class StaticAnalysisResult:
    """Lightweight static analysis metrics for a code snippet."""

    def __init__(
        self,
        num_functions: int = 0,
        num_classes: int = 0,
        num_lines: int = 0,
        has_docstring: bool = False,
        has_type_hints: bool = False,
        bare_excepts: int = 0,
        issues: list[str] | None = None,
    ) -> None:
        self.num_functions = num_functions
        self.num_classes = num_classes
        self.num_lines = num_lines
        self.has_docstring = has_docstring
        self.has_type_hints = has_type_hints
        self.bare_excepts = bare_excepts
        self.issues = issues or []

    def quality_score(self) -> float:
        """Normalised quality score (0..1) based on metrics."""
        score = 1.0
        # Penalise bare excepts
        score -= 0.2 * min(self.bare_excepts, 5)
        # Reward docstrings
        if not self.has_docstring and self.num_functions > 0:
            score -= 0.1
        # Reward type hints
        if self.has_type_hints:
            score += 0.05
        # Penalise issues
        score -= 0.1 * min(len(self.issues), 10)
        return max(0.0, min(1.0, score))


def analyze_static(code: str) -> StaticAnalysisResult:
    """Run lightweight static analysis on *code* using the ``ast`` module."""
    result = StaticAnalysisResult()

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return result  # will be caught by execution-safety check

    lines = code.splitlines()
    result.num_lines = len(lines)

    for node in ast.walk(tree):
        # function / method count
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result.num_functions += 1
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant, ast.Str)):
                result.has_docstring = True
            # check for type hints on parameters
            for arg in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
                if arg.annotation:
                    result.has_type_hints = True
            if node.returns:
                result.has_type_hints = True
        # class count + docstring
        if isinstance(node, ast.ClassDef):
            result.num_classes += 1
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, (ast.Constant, ast.Str)):
                result.has_docstring = True
        # bare except
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            result.bare_excepts += 1

    return result


# --------------------------------------------------------------------------- #
# Unit-test runner
# --------------------------------------------------------------------------- #

class UnitTestResult:
    """Outcome of running unit tests against generated code."""

    def __init__(
        self,
        passed: int = 0,
        total: int = 0,
        errors: list[str] | None = None,
    ) -> None:
        self.passed = passed
        self.total = total
        self.errors = errors or []

    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total


def run_unit_tests(code: str, test_cases: list[dict[str, Any]]) -> UnitTestResult:
    """Execute *code* against a list of test cases and count passes.

    Each test case is a dict with:
        ``input``: value passed as the argument (or ``args`` / ``kwargs``)
        ``expected``: expected return value (compared via ``==``)
        ``id``: optional test label (default "test_{i}")

    Returns:
        A :class:`UnitTestResult` with pass/fail counts.
    """
    if not code.strip():
        return UnitTestResult(passed=0, total=len(test_cases), errors=["empty code"])

    errors: list[str] = []
    passed = 0

    # Build a safe namespace
    namespace: dict[str, Any] = {}

    try:
        exec(compile(code, "<eval>", "exec"), namespace)
    except Exception as exc:
        return UnitTestResult(
            passed=0,
            total=len(test_cases),
            errors=[f"code execution error: {exc}"],
        )

    for i, tc in enumerate(test_cases):
        tc_id = tc.get("id", f"test_{i}")
        func_name = tc.get("func")
        args = tc.get("args", ())
        kwargs = tc.get("kwargs", {})
        expected = tc.get("expected")

        if func_name not in namespace:
            errors.append(f"{tc_id}: function '{func_name}' not defined")
            continue

        try:
            result = namespace[func_name](*args, **kwargs)
            if result == expected:
                passed += 1
            else:
                errors.append(
                    f"{tc_id}: expected {expected!r}, got {result!r}"
                )
        except Exception as exc:
            errors.append(f"{tc_id}: raised {type(exc).__name__}: {exc}")

    return UnitTestResult(passed=passed, total=len(test_cases), errors=errors)


# --------------------------------------------------------------------------- #
# Hidden test suites for built-in prompts
# --------------------------------------------------------------------------- #

# Maps test IDs to their hidden test cases for automated scoring.
HIDDEN_TEST_SUITES: dict[str, list[dict[str, Any]]] = {
    "coding_chunk": [
        {"id": "chunk_basic", "func": "chunk", "args": ([1, 2, 3, 4, 5], 2), "expected": [[1, 2], [3, 4], [5]]},
        {"id": "chunk_exact", "func": "chunk", "args": ([1, 2, 3, 4], 2), "expected": [[1, 2], [3, 4]]},
        {"id": "chunk_single", "func": "chunk", "args": ([1], 3), "expected": [[1]]},
        {"id": "chunk_empty", "func": "chunk", "args": ([], 2), "expected": []},
        {"id": "chunk_string", "func": "chunk", "args": ("abcde", 2), "expected": ["ab", "cd", "e"]},
        {"id": "chunk_large_n", "func": "chunk", "args": ([1, 2], 10), "expected": [[1, 2]]},
    ],
    "algorithmic_twosum": [
        {"id": "twosum_basic", "func": "two_sum", "args": ([2, 7, 11, 15], 9), "expected": [0, 1]},
        {"id": "twosum_middle", "func": "two_sum", "args": ([3, 2, 4], 6), "expected": [1, 2]},
        {"id": "twosum_negatives", "func": "two_sum", "args": ([-1, -2, -3, -4], -5), "expected": [0, 3]},
        {"id": "twosum_duplicates", "func": "two_sum", "args": ([3, 3], 6), "expected": [0, 1]},
    ],
    "algorithmic_bst": [
        {"id": "bst_insert_traverse", "func": "bst_insert_traverse", "args": ([5, 3, 7, 2, 4, 6, 8],), "expected": [2, 3, 4, 5, 6, 7, 8]},
    ],
    "bugfixing": [
        {"id": "bugfix_average", "func": "calculate_average", "args": ([1, 2, 3, 4],), "expected": 2.5},
        {"id": "bugfix_single", "func": "calculate_average", "args": ([5],), "expected": 5.0},
        {"id": "bugfix_empty", "func": "calculate_average", "args": ([],), "expected": 0.0},
        {"id": "bugfix_negatives", "func": "calculate_average", "args": ([-1, 0, 1],), "expected": 0.0},
    ],
    "test_generation": [
        # The test_generation prompt asks to generate tests. We evaluate
        # whether the generated test file defines the expected test functions.
    ],
}


def _make_bst_helper() -> str | None:
    """Build helper to evaluate BST output by wrapping generated code."""
    return textwrap.dedent("""\
def bst_insert_traverse(values):
    bst = BinarySearchTree()
    for v in values:
        bst.insert(v)
    return bst.in_order_traversal()
""")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #

class EvalScoreSet:
    """Aggregated automated scores for one response."""

    def __init__(
        self,
        code_extracted: bool = False,
        exec_safety: float = 0.0,
        static_quality: float = 0.0,
        unit_pass_rate: float = 0.0,
        overall: float = 0.0,
        issues: list[str] | None = None,
    ) -> None:
        self.code_extracted = code_extracted
        self.exec_safety = exec_safety
        self.static_quality = static_quality
        self.unit_pass_rate = unit_pass_rate
        self.overall = overall
        self.issues = issues or []


AUTO_EVAL_METRICS = [
    "exec_safety",
    "static_quality",
    "unit_pass_rate",
    "overall",
]


def evaluate_response(
    result: RunResult,
    hidden_tests: dict[str, list[dict[str, Any]]] | None = None,
) -> EvalScoreSet:
    """Run the full automated evaluation pipeline on a single response.

    Args:
        result: A single run result.
        hidden_tests: Optional override for hidden test suites.

    Returns:
        An :class:`EvalScoreSet` with all computed scores.
    """
    tests = hidden_tests or HIDDEN_TEST_SUITES
    issues: list[str] = []
    code = CodeExtractor.extract_python(result.response_text)

    if not code:
        return EvalScoreSet(
            code_extracted=False,
            exec_safety=0.0,
            static_quality=0.0,
            unit_pass_rate=0.0,
            overall=0.0,
            issues=["no Python code extracted"],
        )

    # 1. Execution safety
    safety = check_execution_safety(code)
    if not safety.compile_ok:
        issues.append(f"compile error: {safety.error}" if safety.error else "compile error")
    if not safety.runtime_ok and safety.compile_ok:
        issues.append(f"runtime error: {safety.error}" if safety.error else "runtime error")

    # 2. Static analysis
    static = analyze_static(code)
    if static.bare_excepts > 0:
        issues.append(f"bare except: {static.bare_excepts} occurrence(s)")
    if not static.has_docstring and static.num_functions > 0:
        issues.append("missing function/class docstring")

    # 3. Unit tests (if hidden tests exist for this test ID)
    unit_result: UnitTestResult | None = None
    test_id = result.test_id

    if test_id in tests:
        test_cases = list(tests[test_id])
        # For BST, inject the helper wrapper
        if test_id == "algorithmic_bst":
            helper = _make_bst_helper()
            if helper:
                code_with_helper = code + "\n\n" + helper
                unit_result = run_unit_tests(code_with_helper, test_cases)
            else:
                unit_result = UnitTestResult(passed=0, total=len(test_cases))
        else:
            unit_result = run_unit_tests(code, test_cases)

        if unit_result and unit_result.errors:
            issues.extend(unit_result.errors[:5])  # cap at 5

    # Compute scores
    exec_score = safety.score()
    static_score = static.quality_score()
    unit_rate = unit_result.pass_rate() if unit_result else 0.0

    # Overall: weighted average
    overall = 0.3 * exec_score + 0.2 * static_score + 0.5 * unit_rate

    return EvalScoreSet(
        code_extracted=True,
        exec_safety=exec_score,
        static_quality=static_score,
        unit_pass_rate=unit_rate,
        overall=overall,
        issues=issues,
    )


def evaluate_run(
    results: list[RunResult],
    hidden_tests: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate all results in a run, returning a lookup keyed by merge key.

    Returns:
        ``{config::test_id::rep: {"overall": ..., "exec_safety": ..., ...}}``
    """
    from benchmarker.runner import config_key

    output: dict[str, dict[str, float]] = {}
    for r in results:
        key = f"{config_key(r.config)}::{r.test_id}::{r.repetition}"
        score_set = evaluate_response(r, hidden_tests=hidden_tests)
        output[key] = {
            "overall": round(score_set.overall, 4),
            "exec_safety": round(score_set.exec_safety, 4),
            "static_quality": round(score_set.static_quality, 4),
            "unit_pass_rate": round(score_set.unit_pass_rate, 4),
        }
    return output
