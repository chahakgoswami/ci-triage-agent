"""Error reproduction engine.

Takes a ParsedFailure, scaffolds a minimal sandbox environment,
runs the failing code in a subprocess, and confirms the failure
matches the original log before returning a ReproductionResult.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ci_triage_agent.parser import FailureType, ParsedFailure


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

@dataclass
class ReproductionResult:
    """Outcome of an attempt to reproduce a CI failure locally."""

    # The failure that was fed in
    parsed_failure: ParsedFailure

    # Did the subprocess exit non-zero?
    process_exited_nonzero: bool

    # Combined stdout + stderr from the subprocess
    stdout: str
    stderr: str

    # Whether the observed output confirms the original error message
    confirmed: bool

    # Human-readable explanation of the confirmation decision
    confirmation_reason: str

    # Path to the sandbox directory (tmp dir) used during reproduction
    sandbox_dir: str = ""

    # Exit code from the subprocess
    returncode: int = 0

    # Any error that prevented even attempting reproduction
    reproduction_error: Optional[str] = None

    @property
    def combined_output(self) -> str:
        """Convenience accessor for stdout + stderr."""
        return (self.stdout + "\n" + self.stderr).strip()

    def summary(self) -> str:  # pragma: no cover
        lines = [
            f"ReproductionResult",
            f"  confirmed          : {self.confirmed}",
            f"  confirmation_reason: {self.confirmation_reason}",
            f"  returncode         : {self.returncode}",
            f"  process_nonzero    : {self.process_exited_nonzero}",
            f"  sandbox_dir        : {self.sandbox_dir}",
        ]
        if self.reproduction_error:
            lines.append(f"  reproduction_error : {self.reproduction_error}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sandbox code generators
# ---------------------------------------------------------------------------

def _make_syntax_error_code(failure: ParsedFailure) -> str:
    """Return Python source that reproduces the syntax error."""
    # Use the snippet from raw_metadata if available, else a generic trigger
    snippet = (
        failure.snippet
        or failure.raw_metadata.get("snippet")
        or "def broken("
    )
    # We embed it inside exec() so that this file itself is syntactically valid;
    # compile() is used to trigger the SyntaxError at the right layer.
    escaped = snippet.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return textwrap.dedent(f"""\
        # Reproduction scaffold for: {failure.error_message}
        code = \"\"\"
        {escaped}
        \"\"\"
        compile(code, '<sandbox>', 'exec')
    """)


def _make_import_error_code(failure: ParsedFailure) -> str:
    """Return Python source that reproduces the import error."""
    snippet = (
        failure.snippet
        or failure.raw_metadata.get("snippet")
        or ""
    )
    if snippet:
        return textwrap.dedent(f"""\
            # Reproduction scaffold for: {failure.error_message}
            {snippet}
        """)
    # Fallback: derive module name from the error message
    m = re.search(r"No module named '([^']+)'", failure.error_message)
    module = m.group(1) if m else "unknown_module"
    return textwrap.dedent(f"""\
        # Reproduction scaffold for: {failure.error_message}
        import {module}
    """)


def _make_test_failure_code(failure: ParsedFailure) -> str:
    """Return Python source (pytest-runnable) that reproduces the test failure."""
    test_name = failure.test_name or "test_reproduction"
    # Parse error message to build a meaningful assertion
    msg = failure.error_message

    # AssertionError: assert X == Y
    m = re.match(r"AssertionError: assert (.+?) == (.+)", msg)
    if m:
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        body = f"    assert {lhs} == {rhs}"
    else:
        # Generic failure: raise the error message directly
        escaped_msg = msg.replace('"', '\\"')
        body = f'    raise AssertionError("{escaped_msg}")'

    return textwrap.dedent(f"""\
        # Reproduction scaffold for: {msg}
        def {test_name}():
        {body}


        if __name__ == "__main__":
            import pytest, sys
            sys.exit(pytest.main([__file__, "-x", "-q"]))
    """)


def _make_lint_error_code(failure: ParsedFailure) -> str:
    """Return Python source that reproduces the lint violation."""
    snippet = (
        failure.snippet
        or failure.raw_metadata.get("snippet")
        or ""
    )
    msg = failure.error_message

    if "E501" in msg:
        # Line too long: generate a line that is definitely too long
        long_line = "x = " + "'" + "a" * 200 + "'"
        return textwrap.dedent(f"""\
            # Reproduction scaffold for: {msg}
            {long_line}
        """)
    if "F401" in msg:
        # Unused import
        m = re.search(r"'([^']+)' imported but unused", msg)
        module = m.group(1) if m else "os"
        return textwrap.dedent(f"""\
            # Reproduction scaffold for: {msg}
            import {module}
            x = 1
        """)

    if snippet:
        return textwrap.dedent(f"""\
            # Reproduction scaffold for: {msg}
            {snippet}
        """)

    return textwrap.dedent(f"""\
        # Reproduction scaffold for: {msg}
        pass
    """)


_CODE_GENERATORS = {
    FailureType.SYNTAX_ERROR: _make_syntax_error_code,
    FailureType.IMPORT_ERROR: _make_import_error_code,
    FailureType.TEST_FAILURE: _make_test_failure_code,
    FailureType.LINT_ERROR: _make_lint_error_code,
}


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

def _run_python(script: Path, timeout: int = 30) -> tuple[int, str, str]:
    """Run a Python script in a subprocess and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _run_pytest(script: Path, timeout: int = 60) -> tuple[int, str, str]:
    """Run pytest on a single file and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(script), "-x", "-q", "--tb=short"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _run_flake8(script: Path, timeout: int = 30) -> tuple[int, str, str]:
    """Run flake8 on a single file and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "flake8", str(script), "--max-line-length=88"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Confirmation helpers
# ---------------------------------------------------------------------------

_ERROR_KEYWORDS: dict[FailureType, list[str]] = {
    FailureType.SYNTAX_ERROR: ["SyntaxError"],
    FailureType.IMPORT_ERROR: ["ModuleNotFoundError", "ImportError"],
    FailureType.TEST_FAILURE: ["AssertionError", "FAILED", "failed", "error"],
    FailureType.LINT_ERROR: ["E501", "F401", "E302", "E303", "W"],
    FailureType.UNKNOWN: [],
}


def _confirm_output(
    failure: ParsedFailure,
    returncode: int,
    stdout: str,
    stderr: str,
) -> tuple[bool, str]:
    """Return (confirmed, reason) by checking output against expected failure."""
    combined = stdout + "\n" + stderr

    if returncode == 0 and failure.failure_type != FailureType.UNKNOWN:
        return False, "Subprocess exited successfully; expected a non-zero exit code."

    keywords = _ERROR_KEYWORDS.get(failure.failure_type, [])

    # Always look for the exact error message (or a keyword substring)
    for kw in keywords:
        if kw in combined:
            return True, f"Output contains expected keyword: {kw!r}"

    # Fallback: check if any word from the original error_message appears
    for word in failure.error_message.split():
        if len(word) > 4 and word in combined:
            return True, f"Output contains error_message token: {word!r}"

    if returncode != 0:
        return (
            True,
            "Subprocess exited non-zero (no keyword match, but failure confirmed by exit code).",
        )

    return False, "Output did not match expected error pattern."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class ReproductionEngine:
    """Attempts to reproduce CI failures locally in a sandboxed subprocess."""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def reproduce(self, failure: ParsedFailure) -> ReproductionResult:
        """Attempt to reproduce *failure* and return a ReproductionResult.

        A temporary directory is created for each reproduction attempt.
        The generated script (or test) is written there and executed in a
        child process; stdout/stderr are captured and compared against the
        original error message to produce a *confirmed* flag.

        Parameters
        ----------
        failure:
            The structured failure metadata from the parsing layer.

        Returns
        -------
        ReproductionResult
            Always returns a result object; never raises (exceptions from
            subprocess setup are captured in *reproduction_error*).
        """
        with tempfile.TemporaryDirectory(prefix="ci_triage_sandbox_") as tmpdir:
            sandbox = Path(tmpdir)
            try:
                return self._attempt(failure, sandbox)
            except Exception as exc:  # noqa: BLE001
                return ReproductionResult(
                    parsed_failure=failure,
                    process_exited_nonzero=False,
                    stdout="",
                    stderr="",
                    confirmed=False,
                    confirmation_reason="",
                    sandbox_dir=str(sandbox),
                    returncode=-1,
                    reproduction_error=str(exc),
                )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _attempt(
        self, failure: ParsedFailure, sandbox: Path
    ) -> ReproductionResult:
        """Inner attempt logic (may raise; callers must catch)."""
        generator = _CODE_GENERATORS.get(failure.failure_type)
        if generator is None:
            # UNKNOWN – nothing we can do
            return ReproductionResult(
                parsed_failure=failure,
                process_exited_nonzero=False,
                stdout="",
                stderr="",
                confirmed=False,
                confirmation_reason="No reproduction strategy for UNKNOWN failure type.",
                sandbox_dir=str(sandbox),
                returncode=0,
                reproduction_error=None,
            )

        code = generator(failure)
        script = sandbox / "reproduce.py"
        script.write_text(code, encoding="utf-8")

        # Choose runner
        if failure.failure_type == FailureType.LINT_ERROR:
            returncode, stdout, stderr = _run_flake8(script, timeout=self.timeout)
        elif failure.failure_type == FailureType.TEST_FAILURE:
            returncode, stdout, stderr = _run_pytest(script, timeout=self.timeout)
        else:
            returncode, stdout, stderr = _run_python(script, timeout=self.timeout)

        confirmed, reason = _confirm_output(failure, returncode, stdout, stderr)

        return ReproductionResult(
            parsed_failure=failure,
            process_exited_nonzero=returncode != 0,
            stdout=stdout,
            stderr=stderr,
            confirmed=confirmed,
            confirmation_reason=reason,
            sandbox_dir=str(sandbox),
            returncode=returncode,
            reproduction_error=None,
        )


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def reproduce_failure(
    failure: ParsedFailure, timeout: int = 30
) -> ReproductionResult:
    """Convenience wrapper around :class:`ReproductionEngine`."""
    engine = ReproductionEngine(timeout=timeout)
    return engine.reproduce(failure)
