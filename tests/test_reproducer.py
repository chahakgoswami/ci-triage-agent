"""Tests for the error reproduction engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ci_triage_agent.parser import FailureType, ParsedFailure
from ci_triage_agent.reproducer import (
    ReproductionEngine,
    ReproductionResult,
    _confirm_output,
    _make_import_error_code,
    _make_lint_error_code,
    _make_syntax_error_code,
    _make_test_failure_code,
    reproduce_failure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_failure(
    failure_type: FailureType,
    error_message: str = "SyntaxError: invalid syntax",
    error_file: str = "src/app/utils.py",
    error_line: int = 42,
    test_name: str | None = None,
    stack_trace: str | None = None,
    snippet: str | None = None,
    raw_metadata: dict | None = None,
) -> ParsedFailure:
    return ParsedFailure(
        run_id="test-run-001",
        source_file="/tmp/test.json",
        failure_type=failure_type,
        error_file=error_file,
        error_line=error_line,
        error_message=error_message,
        test_name=test_name,
        stack_trace=stack_trace,
        snippet=snippet,
        raw_metadata=raw_metadata or {},
    )


# ---------------------------------------------------------------------------
# ReproductionResult
# ---------------------------------------------------------------------------

class TestReproductionResult:
    def test_combined_output(self) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        result = ReproductionResult(
            parsed_failure=failure,
            process_exited_nonzero=True,
            stdout="out",
            stderr="err",
            confirmed=True,
            confirmation_reason="matched",
        )
        assert "out" in result.combined_output
        assert "err" in result.combined_output

    def test_defaults(self) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        result = ReproductionResult(
            parsed_failure=failure,
            process_exited_nonzero=False,
            stdout="",
            stderr="",
            confirmed=False,
            confirmation_reason="",
        )
        assert result.returncode == 0
        assert result.sandbox_dir == ""
        assert result.reproduction_error is None


# ---------------------------------------------------------------------------
# Code generators
# ---------------------------------------------------------------------------

class TestCodeGenerators:
    def test_syntax_error_code_is_valid_python(self) -> None:
        failure = _make_failure(
            FailureType.SYNTAX_ERROR,
            snippet="def broken(",
        )
        code = _make_syntax_error_code(failure)
        # The generated wrapper must itself be syntactically valid
        compile(code, "<test>", "exec")

    def test_syntax_error_code_contains_compile(self) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        code = _make_syntax_error_code(failure)
        assert "compile(" in code

    def test_import_error_code_contains_import(self) -> None:
        failure = _make_failure(
            FailureType.IMPORT_ERROR,
            error_message="ModuleNotFoundError: No module named 'dotenv'",
            snippet="from dotenv import load_dotenv",
        )
        code = _make_import_error_code(failure)
        assert "dotenv" in code

    def test_import_error_fallback_uses_module_name(self) -> None:
        failure = _make_failure(
            FailureType.IMPORT_ERROR,
            error_message="ModuleNotFoundError: No module named 'foobar_pkg'",
        )
        code = _make_import_error_code(failure)
        assert "foobar_pkg" in code

    def test_test_failure_code_contains_def(self) -> None:
        failure = _make_failure(
            FailureType.TEST_FAILURE,
            error_message="AssertionError: assert 0 == 1",
            test_name="test_compute_total",
        )
        code = _make_test_failure_code(failure)
        assert "def test_compute_total" in code

    def test_test_failure_code_contains_assertion(self) -> None:
        failure = _make_failure(
            FailureType.TEST_FAILURE,
            error_message="AssertionError: assert 0 == 1",
            test_name="test_foo",
        )
        code = _make_test_failure_code(failure)
        assert "assert" in code or "AssertionError" in code

    def test_lint_error_e501_code_has_long_line(self) -> None:
        failure = _make_failure(
            FailureType.LINT_ERROR,
            error_message="E501 line too long (102 > 88 characters)",
        )
        code = _make_lint_error_code(failure)
        # At least one line should be longer than 88 chars
        assert any(len(line) > 88 for line in code.splitlines())

    def test_lint_error_f401_code_has_import(self) -> None:
        failure = _make_failure(
            FailureType.LINT_ERROR,
            error_message="F401 'os' imported but unused",
        )
        code = _make_lint_error_code(failure)
        assert "import os" in code


# ---------------------------------------------------------------------------
# _confirm_output
# ---------------------------------------------------------------------------

class TestConfirmOutput:
    def test_syntax_error_confirmed_by_keyword(self) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        confirmed, reason = _confirm_output(failure, 1, "", "SyntaxError: invalid syntax")
        assert confirmed is True
        assert "SyntaxError" in reason

    def test_import_error_confirmed_by_module_not_found(self) -> None:
        failure = _make_failure(
            FailureType.IMPORT_ERROR,
            error_message="ModuleNotFoundError: No module named 'dotenv'",
        )
        confirmed, reason = _confirm_output(
            failure, 1, "", "ModuleNotFoundError: No module named 'dotenv'"
        )
        assert confirmed is True

    def test_exit_zero_not_confirmed(self) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        confirmed, reason = _confirm_output(failure, 0, "", "some output")
        assert confirmed is False
        assert "successfully" in reason

    def test_nonzero_exit_confirmed_as_fallback(self) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        confirmed, reason = _confirm_output(failure, 1, "", "some unrelated output")
        # Should still be confirmed because returncode != 0
        assert confirmed is True

    def test_lint_error_confirmed_by_code(self) -> None:
        failure = _make_failure(
            FailureType.LINT_ERROR,
            error_message="E501 line too long",
        )
        confirmed, reason = _confirm_output(
            failure, 1, "reproduce.py:2:89: E501 line too long\n", ""
        )
        assert confirmed is True

    def test_test_failure_confirmed_by_failed_keyword(self) -> None:
        failure = _make_failure(
            FailureType.TEST_FAILURE,
            error_message="AssertionError: assert 0 == 1",
        )
        confirmed, reason = _confirm_output(
            failure, 1, "1 failed", ""
        )
        assert confirmed is True


# ---------------------------------------------------------------------------
# ReproductionEngine – integration tests (actual subprocess)
# ---------------------------------------------------------------------------

class TestReproductionEngineIntegration:
    """These tests actually run subprocesses."""

    def test_syntax_error_confirmed(self) -> None:
        failure = _make_failure(
            FailureType.SYNTAX_ERROR,
            error_message="SyntaxError: invalid syntax",
            snippet="def broken(",
        )
        engine = ReproductionEngine(timeout=30)
        result = engine.reproduce(failure)
        assert isinstance(result, ReproductionResult)
        assert result.process_exited_nonzero is True
        assert result.confirmed is True
        assert result.reproduction_error is None

    def test_import_error_confirmed(self) -> None:
        failure = _make_failure(
            FailureType.IMPORT_ERROR,
            error_message="ModuleNotFoundError: No module named '_nonexistent_xyz_module'",
        )
        engine = ReproductionEngine(timeout=30)
        result = engine.reproduce(failure)
        assert result.process_exited_nonzero is True
        assert result.confirmed is True

    def test_test_failure_confirmed(self) -> None:
        failure = _make_failure(
            FailureType.TEST_FAILURE,
            error_message="AssertionError: assert 0 == 1",
            test_name="test_always_fails",
        )
        engine = ReproductionEngine(timeout=30)
        result = engine.reproduce(failure)
        assert result.process_exited_nonzero is True
        assert result.confirmed is True

    def test_lint_error_e501_confirmed(self) -> None:
        failure = _make_failure(
            FailureType.LINT_ERROR,
            error_message="E501 line too long (102 > 88 characters)",
        )
        engine = ReproductionEngine(timeout=30)
        result = engine.reproduce(failure)
        # flake8 may not be installed in all environments; skip if not found
        if "No module named flake8" in result.combined_output or result.reproduction_error:
            pytest.skip("flake8 not available in this environment")
        assert result.process_exited_nonzero is True
        assert result.confirmed is True

    def test_unknown_failure_not_confirmed(self) -> None:
        failure = _make_failure(
            FailureType.UNKNOWN,
            error_message="Something weird happened",
        )
        engine = ReproductionEngine(timeout=30)
        result = engine.reproduce(failure)
        assert result.confirmed is False
        assert "UNKNOWN" in result.confirmation_reason

    def test_result_has_sandbox_dir(self) -> None:
        failure = _make_failure(
            FailureType.SYNTAX_ERROR,
            snippet="def broken(",
        )
        engine = ReproductionEngine(timeout=30)
        result = engine.reproduce(failure)
        # sandbox_dir exists during the call but is cleaned up after;
        # we just confirm it was set
        assert result.sandbox_dir != ""

    def test_result_has_stdout_or_stderr(self) -> None:
        failure = _make_failure(
            FailureType.SYNTAX_ERROR,
            error_message="SyntaxError: invalid syntax",
            snippet="def broken(",
        )
        engine = ReproductionEngine(timeout=30)
        result = engine.reproduce(failure)
        assert result.stdout != "" or result.stderr != ""

    def test_returncode_recorded(self) -> None:
        failure = _make_failure(
            FailureType.IMPORT_ERROR,
            error_message="ModuleNotFoundError: No module named '_nonexistent_xyz_module'",
        )
        result = reproduce_failure(failure)
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# ReproductionEngine – error handling
# ---------------------------------------------------------------------------

class TestReproductionEngineErrorHandling:
    def test_exception_in_runner_captured(self) -> None:
        """If the runner raises (e.g. timeout), the result captures the error."""
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        engine = ReproductionEngine(timeout=30)

        with patch(
            "ci_triage_agent.reproducer._run_python",
            side_effect=RuntimeError("subprocess exploded"),
        ):
            result = engine.reproduce(failure)

        assert result.confirmed is False
        assert result.reproduction_error is not None
        assert "subprocess exploded" in result.reproduction_error


# ---------------------------------------------------------------------------
# reproduce_failure convenience function
# ---------------------------------------------------------------------------

class TestReproduceFailureFunction:
    def test_returns_reproduction_result(self) -> None:
        failure = _make_failure(
            FailureType.SYNTAX_ERROR,
            snippet="def broken(",
        )
        result = reproduce_failure(failure)
        assert isinstance(result, ReproductionResult)

    def test_accepts_custom_timeout(self) -> None:
        failure = _make_failure(
            FailureType.SYNTAX_ERROR,
            snippet="def broken(",
        )
        result = reproduce_failure(failure, timeout=60)
        assert isinstance(result, ReproductionResult)
