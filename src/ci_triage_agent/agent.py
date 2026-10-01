"""LLM-backed fix-suggestion agent.

Receives a ParsedFailure and a ReproductionResult, then proposes a code
patch in unified diff format.  When OPENAI_API_KEY is absent (or the
``force_mock`` flag is set) a deterministic stub client is used instead
of the real OpenAI API so that the module works offline / in CI.
"""

from __future__ import annotations

import dataclasses
import difflib
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Optional

from ci_triage_agent.parser import FailureType, ParsedFailure
from ci_triage_agent.reproducer import ReproductionResult


# ---------------------------------------------------------------------------
# Patch / suggestion model
# ---------------------------------------------------------------------------

@dataclass
class FixSuggestion:
    """A proposed fix produced by the LLM agent."""

    # The failure that was analysed
    parsed_failure: ParsedFailure

    # Whether a real LLM was used or the stub fallback
    used_mock: bool

    # The LLM's free-text explanation / reasoning
    explanation: str

    # Unified-diff patch (may be empty if no fix was found)
    patch: str

    # Original file path that the patch targets (may be empty for UNKNOWN)
    target_file: str = ""

    # Model name / identifier used (e.g. "gpt-4o" or "mock")
    model: str = "mock"

    # Raw response text from the LLM (for debugging)
    raw_response: str = ""

    # Any error that prevented a proper suggestion
    agent_error: Optional[str] = None

    @property
    def has_patch(self) -> bool:
        """Return True when a non-empty patch was produced."""
        return bool(self.patch.strip())

    def summary(self) -> str:  # pragma: no cover
        lines = [
            "FixSuggestion",
            f"  model       : {self.model}",
            f"  used_mock   : {self.used_mock}",
            f"  target_file : {self.target_file}",
            f"  has_patch   : {self.has_patch}",
            f"  explanation : {self.explanation[:120]}",
        ]
        if self.agent_error:
            lines.append(f"  agent_error : {self.agent_error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert software engineer and CI triage assistant.
    You will receive a structured description of a failing CI build,
    including the error type, location, message, and a reproduction
    transcript.  Your job is to:

    1. Briefly explain the root cause of the failure (2-4 sentences).
    2. Propose a minimal fix as a unified diff (--- / +++ format).
       - Use 'a/<file>' and 'b/<file>' as the file paths.
       - Include 3 lines of context around each change.
       - Only change what is strictly necessary to fix the error.

    Respond with EXACTLY two sections, separated by a blank line:

    EXPLANATION:
    <your explanation here>

    PATCH:
    <unified diff here>

    If you cannot propose a fix, output an empty PATCH section.
""")


def _build_user_message(
    failure: ParsedFailure,
    reproduction: ReproductionResult,
) -> str:
    """Format the user turn sent to the LLM."""
    parts: list[str] = [
        f"## CI Failure Report",
        f"",
        f"- **Run ID**: {failure.run_id}",
        f"- **Failure type**: {failure.failure_type.value}",
        f"- **File**: {failure.error_file}:{failure.error_line}",
        f"- **Error message**: {failure.error_message}",
    ]
    if failure.test_name:
        parts.append(f"- **Test name**: {failure.test_name}")
    if failure.snippet:
        parts.append(f"\n### Code snippet\n```python\n{failure.snippet}\n```")
    if failure.stack_trace:
        parts.append(f"\n### Stack trace\n```\n{failure.stack_trace}\n```")
    if failure.details:
        parts.append(f"\n### Additional details\n{failure.details}")

    parts.append(f"\n## Reproduction transcript")
    parts.append(f"- **Confirmed**: {reproduction.confirmed}")
    parts.append(f"- **Exit code**: {reproduction.returncode}")
    if reproduction.combined_output:
        trimmed = reproduction.combined_output[:2000]
        parts.append(f"\n```\n{trimmed}\n```")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_llm_response(response_text: str) -> tuple[str, str]:
    """Extract (explanation, patch) from the LLM response text."""
    explanation = ""
    patch = ""

    # Split on the section headers (case-insensitive)
    exp_match = re.search(
        r"EXPLANATION:\s*\n(.*?)(?=\nPATCH:|$)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )
    patch_match = re.search(
        r"PATCH:\s*\n(.*)",
        response_text,
        re.DOTALL | re.IGNORECASE,
    )

    if exp_match:
        explanation = exp_match.group(1).strip()
    if patch_match:
        patch = patch_match.group(1).strip()

    return explanation, patch


# ---------------------------------------------------------------------------
# Mock / stub LLM client
# ---------------------------------------------------------------------------

class _MockLLMClient:
    """Deterministic stub that produces plausible patches without a network call."""

    model: str = "mock"

    def complete(self, system: str, user: str) -> str:  # noqa: ARG002
        """Return a canned response based on keywords in *user*."""
        return self._generate_response(user)

    # ------------------------------------------------------------------
    # Internal response templates
    # ------------------------------------------------------------------

    def _generate_response(self, user_message: str) -> str:
        if "syntax_error" in user_message or "SyntaxError" in user_message:
            return self._syntax_error_response(user_message)
        if "import_error" in user_message or "ModuleNotFoundError" in user_message or "ImportError" in user_message:
            return self._import_error_response(user_message)
        if "test_failure" in user_message or "AssertionError" in user_message:
            return self._test_failure_response(user_message)
        if "lint_error" in user_message or "E501" in user_message or "F401" in user_message:
            return self._lint_error_response(user_message)
        return self._generic_response()

    def _syntax_error_response(self, user_message: str) -> str:
        # Try to extract the file path from the user message
        file_match = re.search(r"\*\*File\*\*: ([^:]+):(\d+)", user_message)
        filepath = file_match.group(1) if file_match else "src/app/utils.py"
        lineno = int(file_match.group(2)) if file_match else 42

        snippet_match = re.search(r"```python\n(.*?)\n```", user_message, re.DOTALL)
        broken_line = snippet_match.group(1).strip() if snippet_match else "    def compute_total(items)"

        # Attempt to fix the most common syntax error: missing colon
        fixed_line = broken_line
        if not broken_line.rstrip().endswith(":"):
            fixed_line = broken_line.rstrip() + ":"

        explanation = (
            "The function definition is missing a colon at the end of the "
            "signature, which is required Python syntax. "
            "This causes a SyntaxError at parse time, preventing the module "
            "from being imported or executed. "
            "The fix is to append ':' to the offending line."
        )
        patch = textwrap.dedent(f"""\
            --- a/{filepath}
            +++ b/{filepath}
            @@ -{lineno},1 +{lineno},1 @@
            -{broken_line}
            +{fixed_line}
        """)
        return f"EXPLANATION:\n{explanation}\n\nPATCH:\n{patch}"

    def _import_error_response(self, user_message: str) -> str:
        module_match = re.search(r"No module named '([^']+)'", user_message)
        module = module_match.group(1) if module_match else "missing_module"
        # Map common modules to their pip package names
        pip_map: dict[str, str] = {
            "dotenv": "python-dotenv",
            "pydantic_settings": "pydantic-settings",
            "celery": "celery",
        }
        pip_name = pip_map.get(module, module.replace("_", "-"))

        file_match = re.search(r"\*\*File\*\*: ([^:]+):(\d+)", user_message)
        filepath = file_match.group(1) if file_match else "requirements.txt"

        explanation = (
            f"The module '{module}' is not installed in the current environment. "
            f"The fix is to add '{pip_name}' to requirements.txt so that it is "
            "installed as part of the CI dependency step. "
            "Alternatively, update the import to use an already-available module "
            "that provides the same functionality."
        )
        patch = textwrap.dedent(f"""\
            --- a/requirements.txt
            +++ b/requirements.txt
            @@ -1,3 +1,4 @@
             click>=8.1
             rich>=13.0
            +{pip_name}
        """)
        return f"EXPLANATION:\n{explanation}\n\nPATCH:\n{patch}"

    def _test_failure_response(self, user_message: str) -> str:
        test_match = re.search(r"\*\*Test name\*\*: (\w+)", user_message)
        test_name = test_match.group(1) if test_match else "test_failing"

        file_match = re.search(r"\*\*File\*\*: ([^:]+):(\d+)", user_message)
        filepath = file_match.group(1) if file_match else "tests/test_utils.py"
        lineno = int(file_match.group(2)) if file_match else 34

        assert_match = re.search(r"assert (\S+) == (\S+)", user_message)
        if assert_match:
            lhs = assert_match.group(1)
            rhs = assert_match.group(2)
            old_assert = f"    assert {lhs} == {rhs}"
            # Swap lhs/rhs to make test pass (simple illustrative fix)
            new_assert = f"    assert {rhs} == {rhs}  # fixed: was comparing {lhs}"
        else:
            old_assert = f"    assert False"
            new_assert = f"    assert True  # fixed"

        explanation = (
            f"The test '{test_name}' fails because the assertion compares an "
            "actual return value that differs from the expected value. "
            "The underlying implementation likely has a bug where the default "
            "or computed value is wrong. "
            "The fix shown updates the expected value in the test to match the "
            "intended behaviour (a real fix would update the implementation)."
        )
        patch = textwrap.dedent(f"""\
            --- a/{filepath}
            +++ b/{filepath}
            @@ -{lineno},1 +{lineno},1 @@
            -{old_assert}
            +{new_assert}
        """)
        return f"EXPLANATION:\n{explanation}\n\nPATCH:\n{patch}"

    def _lint_error_response(self, user_message: str) -> str:
        file_match = re.search(r"\*\*File\*\*: ([^:]+):(\d+)", user_message)
        filepath = file_match.group(1) if file_match else "src/app/helpers.py"
        lineno = int(file_match.group(2)) if file_match else 7

        if "E501" in user_message:
            explanation = (
                "The line exceeds the maximum allowed length of 88 characters "
                "(PEP 8 / flake8 E501). "
                "The fix is to break the long expression across multiple lines "
                "using Python's implicit line continuation inside parentheses, "
                "or by introducing an intermediate variable."
            )
            old_line = "    result = some_function_with_very_long_name(argument_one, argument_two, argument_three)"
            new_lines = (
                "    result = some_function_with_very_long_name(\n"
                "        argument_one, argument_two, argument_three"
                "\n    )"
            )
            patch = textwrap.dedent(f"""\
                --- a/{filepath}
                +++ b/{filepath}
                @@ -{lineno},1 +{lineno},3 @@
                -{old_line}
                +    result = some_function_with_very_long_name(
                +        argument_one, argument_two, argument_three
                +    )
            """)
        elif "F401" in user_message:
            mod_match = re.search(r"'([^']+)' imported but unused", user_message)
            module = mod_match.group(1) if mod_match else "os"
            explanation = (
                f"The module '{module}' is imported but never used in the file "
                "(flake8 F401). "
                "The fix is to remove the unused import statement entirely. "
                "If the import is needed in the future, it can be re-added at that time."
            )
            patch = textwrap.dedent(f"""\
                --- a/{filepath}
                +++ b/{filepath}
                @@ -{lineno},1 +{lineno},0 @@
                -import {module}
            """)
        else:
            explanation = (
                "A lint rule violation was detected in the file. "
                "Review the offending line and apply the appropriate style fix "
                "as indicated by the error code."
            )
            patch = ""

        return f"EXPLANATION:\n{explanation}\n\nPATCH:\n{patch}"

    def _generic_response(self) -> str:
        explanation = (
            "The failure type could not be automatically classified. "
            "Manual investigation is required to determine the root cause. "
            "No automated patch can be proposed."
        )
        return f"EXPLANATION:\n{explanation}\n\nPATCH:"


# ---------------------------------------------------------------------------
# Real OpenAI client wrapper
# ---------------------------------------------------------------------------

class _OpenAIClient:
    """Thin wrapper around the openai library."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        try:
            import openai  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required to use the real LLM client. "
                "Install it with: pip install openai"
            ) from exc
        self._client = openai.OpenAI(api_key=api_key)
        self.model = model

    def complete(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FixSuggestionAgent:
    """LLM-backed agent that proposes unified-diff patches for CI failures.

    If ``OPENAI_API_KEY`` is set in the environment (and ``force_mock`` is
    False), the real OpenAI API is used.  Otherwise a deterministic mock
    client is used so that the agent works without credentials.

    Parameters
    ----------
    model:
        OpenAI model name to use when the real client is active.
    force_mock:
        If True, always use the stub client regardless of the environment.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        force_mock: bool = False,
    ) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if api_key and not force_mock:
            self._client: Any = _OpenAIClient(api_key=api_key, model=model)
            self._used_mock = False
        else:
            self._client = _MockLLMClient()
            self._used_mock = True

    @property
    def used_mock(self) -> bool:
        """True when the stub client is active."""
        return self._used_mock

    @property
    def model_name(self) -> str:
        """Identifier for the active model."""
        return self._client.model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def suggest_fix(
        self,
        failure: ParsedFailure,
        reproduction: ReproductionResult,
    ) -> FixSuggestion:
        """Produce a FixSuggestion for the given failure and reproduction result.

        Parameters
        ----------
        failure:
            Structured failure metadata from the parsing layer.
        reproduction:
            Output from the reproduction engine.

        Returns
        -------
        FixSuggestion
            Always returns a result; errors are captured in ``agent_error``.
        """
        user_message = _build_user_message(failure, reproduction)
        raw_response = ""
        try:
            raw_response = self._client.complete(
                system=_SYSTEM_PROMPT,
                user=user_message,
            )
            explanation, patch = _parse_llm_response(raw_response)
            return FixSuggestion(
                parsed_failure=failure,
                used_mock=self._used_mock,
                explanation=explanation,
                patch=patch,
                target_file=failure.error_file,
                model=self.model_name,
                raw_response=raw_response,
                agent_error=None,
            )
        except Exception as exc:  # noqa: BLE001
            return FixSuggestion(
                parsed_failure=failure,
                used_mock=self._used_mock,
                explanation="",
                patch="",
                target_file=failure.error_file,
                model=self.model_name,
                raw_response=raw_response,
                agent_error=str(exc),
            )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def suggest_fix(
    failure: ParsedFailure,
    reproduction: ReproductionResult,
    *,
    model: str = "gpt-4o",
    force_mock: bool = False,
) -> FixSuggestion:
    """Convenience wrapper around :class:`FixSuggestionAgent`."""
    agent = FixSuggestionAgent(model=model, force_mock=force_mock)
    return agent.suggest_fix(failure, reproduction)
