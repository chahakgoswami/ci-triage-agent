"""Tests for the LLM-backed fix-suggestion agent."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ci_triage_agent.agent import (
    FixSuggestion,
    FixSuggestionAgent,
    _MockLLMClient,
    _build_user_message,
    _parse_llm_response,
    suggest_fix,
)
from ci_triage_agent.parser import FailureType, ParsedFailure
from ci_triage_agent.reproducer import ReproductionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_failure(
    failure_type: FailureType = FailureType.SYNTAX_ERROR,
    error_message: str = "SyntaxError: invalid syntax",
    error_file: str = "src/app/utils.py",
    error_line: int = 42,
    test_name: str | None = None,
    snippet: str | None = None,
    stack_trace: str | None = None,
    details: str | None = None,
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
        snippet=snippet,
        stack_trace=stack_trace,
        details=details,
        raw_metadata=raw_metadata or {},
    )


def _make_reproduction(
    failure: ParsedFailure,
    confirmed: bool = True,
    returncode: int = 1,
    stdout: str = "",
    stderr: str = "SyntaxError: invalid syntax",
) -> ReproductionResult:
    return ReproductionResult(
        parsed_failure=failure,
        process_exited_nonzero=returncode != 0,
        stdout=stdout,
        stderr=stderr,
        confirmed=confirmed,
        confirmation_reason="matched keyword",
        returncode=returncode,
    )


# ---------------------------------------------------------------------------
# FixSuggestion dataclass
# ---------------------------------------------------------------------------

class TestFixSuggestion:
    def test_has_patch_true_when_nonempty(self) -> None:
        failure = _make_failure()
        sug = FixSuggestion(
            parsed_failure=failure,
            used_mock=True,
            explanation="fix this",
            patch="--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old\n+new",
            model="mock",
        )
        assert sug.has_patch is True

    def test_has_patch_false_when_empty(self) -> None:
        failure = _make_failure()
        sug = FixSuggestion(
            parsed_failure=failure,
            used_mock=True,
            explanation="cannot fix",
            patch="",
            model="mock",
        )
        assert sug.has_patch is False

    def test_defaults(self) -> None:
        failure = _make_failure()
        sug = FixSuggestion(
            parsed_failure=failure,
            used_mock=True,
            explanation="",
            patch="",
            model="mock",
        )
        assert sug.target_file == ""
        assert sug.raw_response == ""
        assert sug.agent_error is None


# ---------------------------------------------------------------------------
# _parse_llm_response
# ---------------------------------------------------------------------------

class TestParseLLMResponse:
    def test_extracts_explanation_and_patch(self) -> None:
        text = (
            "EXPLANATION:\nThe bug is X.\n\n"
            "PATCH:\n--- a/f.py\n+++ b/f.py"
        )
        exp, patch = _parse_llm_response(text)
        assert exp == "The bug is X."
        assert "--- a/f.py" in patch

    def test_empty_patch_section(self) -> None:
        text = "EXPLANATION:\nNo fix available.\n\nPATCH:"
        exp, patch = _parse_llm_response(text)
        assert exp == "No fix available."
        assert patch == ""

    def test_missing_explanation_returns_empty(self) -> None:
        text = "PATCH:\n--- a/f.py\n+++ b/f.py"
        exp, patch = _parse_llm_response(text)
        assert exp == ""
        assert "--- a/f.py" in patch

    def test_case_insensitive_headers(self) -> None:
        text = "explanation:\nSome text.\n\npatch:\ndiff content"
        exp, patch = _parse_llm_response(text)
        assert exp == "Some text."
        assert "diff content" in patch

    def test_multiline_explanation(self) -> None:
        text = "EXPLANATION:\nLine 1.\nLine 2.\nLine 3.\n\nPATCH:\ndiff"
        exp, patch = _parse_llm_response(text)
        assert "Line 1." in exp
        assert "Line 3." in exp


# ---------------------------------------------------------------------------
# _build_user_message
# ---------------------------------------------------------------------------

class TestBuildUserMessage:
    def test_contains_failure_type(self) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        reproduction = _make_reproduction(failure)
        msg = _build_user_message(failure, reproduction)
        assert "syntax_error" in msg

    def test_contains_error_message(self) -> None:
        failure = _make_failure(error_message="SyntaxError: invalid syntax")
        reproduction = _make_reproduction(failure)
        msg = _build_user_message(failure, reproduction)
        assert "SyntaxError: invalid syntax" in msg

    def test_contains_file_and_line(self) -> None:
        failure = _make_failure(error_file="src/app/utils.py", error_line=42)
        reproduction = _make_reproduction(failure)
        msg = _build_user_message(failure, reproduction)
        assert "src/app/utils.py" in msg
        assert "42" in msg

    def test_contains_snippet_when_present(self) -> None:
        failure = _make_failure(snippet="def broken(")
        reproduction = _make_reproduction(failure)
        msg = _build_user_message(failure, reproduction)
        assert "def broken(" in msg

    def test_contains_test_name_when_present(self) -> None:
        failure = _make_failure(
            failure_type=FailureType.TEST_FAILURE,
            test_name="test_compute_total",
        )
        reproduction = _make_reproduction(failure)
        msg = _build_user_message(failure, reproduction)
        assert "test_compute_total" in msg

    def test_contains_reproduction_output(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure, stderr="SyntaxError: invalid syntax")
        msg = _build_user_message(failure, reproduction)
        assert "SyntaxError" in msg

    def test_omits_test_name_when_absent(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure)
        msg = _build_user_message(failure, reproduction)
        assert "Test name" not in msg

    def test_omits_snippet_when_absent(self) -> None:
        failure = _make_failure(snippet=None)
        reproduction = _make_reproduction(failure)
        msg = _build_user_message(failure, reproduction)
        assert "Code snippet" not in msg


# ---------------------------------------------------------------------------
# _MockLLMClient
# ---------------------------------------------------------------------------

class TestMockLLMClient:
    def test_model_attribute(self) -> None:
        client = _MockLLMClient()
        assert client.model == "mock"

    def test_syntax_error_response_has_sections(self) -> None:
        client = _MockLLMClient()
        response = client.complete(
            system="",
            user="**Failure type**: syntax_error\n**File**: src/app/utils.py:42\nSyntaxError",
        )
        assert "EXPLANATION" in response
        assert "PATCH" in response

    def test_import_error_response_contains_requirements(self) -> None:
        client = _MockLLMClient()
        response = client.complete(
            system="",
            user="import_error\nNo module named 'dotenv'\n**File**: src/app/config.py:5",
        )
        assert "PATCH" in response
        assert "requirements" in response.lower() or "dotenv" in response

    def test_test_failure_response_has_patch(self) -> None:
        client = _MockLLMClient()
        response = client.complete(
            system="",
            user="test_failure\nAssertionError: assert 0 == 1\n**Test name**: test_foo\n**File**: tests/test_utils.py:34",
        )
        exp, patch = _parse_llm_response(response)
        assert exp != ""
        # patch may or may not be empty depending on template
        assert "PATCH" in response

    def test_lint_e501_response_has_explanation(self) -> None:
        client = _MockLLMClient()
        response = client.complete(
            system="",
            user="lint_error\nE501 line too long\n**File**: src/app/helpers.py:7",
        )
        exp, _ = _parse_llm_response(response)
        assert "E501" in exp or "length" in exp.lower()

    def test_lint_f401_response_mentions_import(self) -> None:
        client = _MockLLMClient()
        response = client.complete(
            system="",
            user="lint_error\nF401 'os' imported but unused\n**File**: src/app/utils.py:1",
        )
        assert "os" in response or "import" in response.lower()

    def test_unknown_failure_response_has_no_patch(self) -> None:
        client = _MockLLMClient()
        response = client.complete(
            system="",
            user="Something completely unrelated and unclassifiable.",
        )
        _, patch = _parse_llm_response(response)
        assert patch == ""


# ---------------------------------------------------------------------------
# FixSuggestionAgent – mock mode (no OPENAI_API_KEY)
# ---------------------------------------------------------------------------

class TestFixSuggestionAgentMockMode:
    def test_used_mock_true_without_api_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            agent = FixSuggestionAgent()
        assert agent.used_mock is True

    def test_model_name_is_mock(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            agent = FixSuggestionAgent()
        assert agent.model_name == "mock"

    def test_force_mock_overrides_api_key(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake"}):
            agent = FixSuggestionAgent(force_mock=True)
        assert agent.used_mock is True

    def test_suggest_fix_returns_fix_suggestion(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure)
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        assert isinstance(result, FixSuggestion)

    def test_suggest_fix_syntax_error_has_explanation(self) -> None:
        failure = _make_failure(
            FailureType.SYNTAX_ERROR,
            error_message="SyntaxError: invalid syntax",
            snippet="def broken(",
        )
        reproduction = _make_reproduction(failure)
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        assert result.explanation != ""

    def test_suggest_fix_syntax_error_has_patch(self) -> None:
        failure = _make_failure(
            FailureType.SYNTAX_ERROR,
            error_message="SyntaxError: invalid syntax",
        )
        reproduction = _make_reproduction(failure)
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        assert result.has_patch is True

    def test_suggest_fix_import_error(self) -> None:
        failure = _make_failure(
            FailureType.IMPORT_ERROR,
            error_message="ModuleNotFoundError: No module named 'dotenv'",
        )
        reproduction = _make_reproduction(failure, stderr="ModuleNotFoundError: No module named 'dotenv'")
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        assert result.has_patch is True
        assert "dotenv" in result.explanation or "dotenv" in result.patch

    def test_suggest_fix_test_failure(self) -> None:
        failure = _make_failure(
            FailureType.TEST_FAILURE,
            error_message="AssertionError: assert 0 == 1",
            test_name="test_compute_total",
        )
        reproduction = _make_reproduction(failure, stderr="AssertionError: assert 0 == 1")
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        assert result.explanation != ""

    def test_suggest_fix_lint_e501(self) -> None:
        failure = _make_failure(
            FailureType.LINT_ERROR,
            error_message="E501 line too long (102 > 88 characters)",
        )
        reproduction = _make_reproduction(failure)
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        assert result.explanation != ""
        assert result.has_patch is True

    def test_suggest_fix_lint_f401(self) -> None:
        failure = _make_failure(
            FailureType.LINT_ERROR,
            error_message="F401 'os' imported but unused",
        )
        reproduction = _make_reproduction(failure)
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        assert result.has_patch is True

    def test_suggest_fix_unknown_has_no_patch(self) -> None:
        failure = _make_failure(
            FailureType.UNKNOWN,
            error_message="Something weird",
        )
        reproduction = _make_reproduction(failure, confirmed=False, returncode=1)
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        # Mock client returns empty patch for unknown
        assert result.has_patch is False

    def test_suggest_fix_target_file_matches_failure(self) -> None:
        failure = _make_failure(error_file="src/app/utils.py")
        reproduction = _make_reproduction(failure)
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        assert result.target_file == "src/app/utils.py"

    def test_suggest_fix_used_mock_propagated(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure)
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        assert result.used_mock is True

    def test_suggest_fix_raw_response_non_empty(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure)
        agent = FixSuggestionAgent(force_mock=True)
        result = agent.suggest_fix(failure, reproduction)
        assert result.raw_response != ""


# ---------------------------------------------------------------------------
# FixSuggestionAgent – real OpenAI client detection
# ---------------------------------------------------------------------------

class TestFixSuggestionAgentRealMode:
    def test_used_mock_false_with_fake_key(self) -> None:
        """When OPENAI_API_KEY is set and force_mock=False, the real client is chosen.
        We don't actually call the API; we just verify client selection.
        """
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-key-for-testing"}):
            # _OpenAIClient will import openai; if not installed, skip
            try:
                agent = FixSuggestionAgent(model="gpt-4o", force_mock=False)
                assert agent.used_mock is False
            except ImportError:
                pytest.skip("openai package not installed")

    def test_force_mock_ignores_api_key(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-fake-key"}):
            agent = FixSuggestionAgent(force_mock=True)
        assert agent.used_mock is True


# ---------------------------------------------------------------------------
# FixSuggestionAgent – error handling
# ---------------------------------------------------------------------------

class TestFixSuggestionAgentErrorHandling:
    def test_llm_exception_captured_in_agent_error(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure)
        agent = FixSuggestionAgent(force_mock=True)

        # Patch the internal client to raise
        agent._client.complete = MagicMock(side_effect=RuntimeError("network timeout"))
        result = agent.suggest_fix(failure, reproduction)

        assert result.agent_error is not None
        assert "network timeout" in result.agent_error
        assert result.has_patch is False
        assert result.explanation == ""

    def test_result_still_has_failure_reference(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure)
        agent = FixSuggestionAgent(force_mock=True)
        agent._client.complete = MagicMock(side_effect=RuntimeError("boom"))
        result = agent.suggest_fix(failure, reproduction)
        assert result.parsed_failure is failure


# ---------------------------------------------------------------------------
# suggest_fix convenience function
# ---------------------------------------------------------------------------

class TestSuggestFixFunction:
    def test_returns_fix_suggestion(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure)
        result = suggest_fix(failure, reproduction, force_mock=True)
        assert isinstance(result, FixSuggestion)

    def test_force_mock_propagated(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure)
        result = suggest_fix(failure, reproduction, force_mock=True)
        assert result.used_mock is True

    def test_patch_is_string(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure)
        result = suggest_fix(failure, reproduction, force_mock=True)
        assert isinstance(result.patch, str)

    def test_explanation_is_string(self) -> None:
        failure = _make_failure()
        reproduction = _make_reproduction(failure)
        result = suggest_fix(failure, reproduction, force_mock=True)
        assert isinstance(result.explanation, str)
