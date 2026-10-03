"""Tests for the human-in-the-loop approval workflow."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from ci_triage_agent.agent import FixSuggestion
from ci_triage_agent.approval import (
    AUDIT_LOG_FILENAME,
    ApprovalDecision,
    _append_audit_entry,
    _write_patch,
    load_audit_trail,
    print_pr_summary,
    run_approval_workflow,
)
from ci_triage_agent.git_pr import MockPullRequest, TestRunResult
from ci_triage_agent.parser import FailureType, ParsedFailure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_failure(
    failure_type: FailureType = FailureType.SYNTAX_ERROR,
    error_message: str = "SyntaxError: invalid syntax",
    error_file: str = "src/app/utils.py",
    error_line: int = 42,
    test_name: str | None = None,
) -> ParsedFailure:
    return ParsedFailure(
        run_id="test-run-001",
        source_file="/tmp/test.json",
        failure_type=failure_type,
        error_file=error_file,
        error_line=error_line,
        error_message=error_message,
        test_name=test_name,
        raw_metadata={},
    )


def _make_suggestion(failure: ParsedFailure, patch: str = "") -> FixSuggestion:
    return FixSuggestion(
        parsed_failure=failure,
        used_mock=True,
        explanation="The bug is a missing colon on the function definition.",
        patch=patch,
        target_file=failure.error_file,
        model="mock",
        raw_response="EXPLANATION:\nFix.\n\nPATCH:\n" + patch,
    )


def _make_pr(
    failure: ParsedFailure | None = None,
    patch: str = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old\n+new\n",
    test_passed: bool = True,
    error: str | None = None,
    patch_applied: bool = True,
) -> MockPullRequest:
    if failure is None:
        failure = _make_failure()
    suggestion = _make_suggestion(failure, patch=patch)
    test_result = TestRunResult(
        passed=test_passed,
        returncode=0 if test_passed else 1,
        stdout="1 passed" if test_passed else "1 failed",
        stderr="",
        tests_passed=1 if test_passed else 0,
        tests_failed=0 if test_passed else 1,
    )
    return MockPullRequest(
        pr_id="abcd-1234-efgh-5678",
        title="fix(Syntax Error): SyntaxError in src/app/utils.py",
        branch_name="ci-triage/syntax_error/src-app-utils-py/abcd1234",
        diff=patch,
        changed_files=["src/app/utils.py"],
        test_result=test_result,
        parsed_failure=failure,
        fix_suggestion=suggestion,
        patch_applied=patch_applied,
        error=error,
    )


def _null_console() -> Console:
    """Console that discards all output (no-op for tests)."""
    return Console(quiet=True)


# ---------------------------------------------------------------------------
# ApprovalDecision
# ---------------------------------------------------------------------------

class TestApprovalDecision:
    def test_to_dict_has_required_keys(self) -> None:
        d = ApprovalDecision(pr_id="abc", decision="approve", patch_written_to="/tmp/x.diff")
        result = d.to_dict()
        assert {"pr_id", "decision", "timestamp", "patch_written_to", "edited_patch", "note"}.issubset(result.keys())

    def test_to_dict_values(self) -> None:
        d = ApprovalDecision(pr_id="x", decision="reject", note="not good")
        result = d.to_dict()
        assert result["decision"] == "reject"
        assert result["note"] == "not good"
        assert result["pr_id"] == "x"

    def test_timestamp_is_iso(self) -> None:
        d = ApprovalDecision(pr_id="y", decision="approve")
        # Should be parseable as ISO datetime
        from datetime import datetime
        dt = datetime.fromisoformat(d.timestamp)
        assert dt is not None


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

class TestAuditTrail:
    def test_append_creates_file(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        d = ApprovalDecision(pr_id="p1", decision="approve")
        _append_audit_entry(log, d)
        assert log.exists()

    def test_append_valid_jsonl(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        d = ApprovalDecision(pr_id="p1", decision="reject", note="bad patch")
        _append_audit_entry(log, d)
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["decision"] == "reject"
        assert entry["pr_id"] == "p1"

    def test_append_multiple_entries(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        for i in range(3):
            d = ApprovalDecision(pr_id=f"pr-{i}", decision="approve")
            _append_audit_entry(log, d)
        entries = load_audit_trail(log)
        assert len(entries) == 3
        assert entries[0]["pr_id"] == "pr-0"
        assert entries[2]["pr_id"] == "pr-2"

    def test_load_nonexistent_returns_empty(self, tmp_path: Path) -> None:
        result = load_audit_trail(tmp_path / "missing.jsonl")
        assert result == []

    def test_append_creates_parent_dirs(self, tmp_path: Path) -> None:
        log = tmp_path / "subdir" / "nested" / "audit.jsonl"
        d = ApprovalDecision(pr_id="x", decision="approve")
        _append_audit_entry(log, d)
        assert log.exists()

    def test_load_returns_list_of_dicts(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.jsonl"
        d = ApprovalDecision(pr_id="a", decision="edit")
        _append_audit_entry(log, d)
        entries = load_audit_trail(log)
        assert isinstance(entries, list)
        assert isinstance(entries[0], dict)


# ---------------------------------------------------------------------------
# print_pr_summary  (smoke tests – just ensure no exceptions)
# ---------------------------------------------------------------------------

class TestPrintPrSummary:
    def test_no_exception_tests_pass(self) -> None:
        pr = _make_pr(test_passed=True)
        print_pr_summary(pr, console=_null_console())

    def test_no_exception_tests_fail(self) -> None:
        pr = _make_pr(test_passed=False)
        print_pr_summary(pr, console=_null_console())

    def test_no_exception_no_patch(self) -> None:
        pr = _make_pr(patch="", patch_applied=False)
        print_pr_summary(pr, console=_null_console())

    def test_no_exception_with_test_name(self) -> None:
        failure = _make_failure(
            FailureType.TEST_FAILURE,
            error_message="AssertionError: assert 0 == 1",
            test_name="test_foo",
        )
        pr = _make_pr(failure=failure)
        print_pr_summary(pr, console=_null_console())

    def test_no_exception_error_status(self) -> None:
        pr = _make_pr(error="disk full", patch_applied=False)
        print_pr_summary(pr, console=_null_console())


# ---------------------------------------------------------------------------
# _write_patch
# ---------------------------------------------------------------------------

class TestWritePatch:
    def test_creates_file(self, tmp_path: Path) -> None:
        pr = _make_pr()
        dest = _write_patch(pr, tmp_path)
        assert dest.exists()

    def test_file_contains_diff(self, tmp_path: Path) -> None:
        patch_text = "--- a/f.py\n+++ b/f.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        pr = _make_pr(patch=patch_text)
        dest = _write_patch(pr, tmp_path)
        assert dest.read_text() == patch_text

    def test_filename_contains_pr_id(self, tmp_path: Path) -> None:
        pr = _make_pr()
        dest = _write_patch(pr, tmp_path)
        assert pr.pr_id[:8] in dest.name

    def test_custom_patch_overrides_diff(self, tmp_path: Path) -> None:
        pr = _make_pr(patch="original")
        dest = _write_patch(pr, tmp_path, patch_text="edited patch")
        assert dest.read_text() == "edited patch"

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        pr = _make_pr()
        out = tmp_path / "patches"
        _write_patch(pr, out)
        assert out.is_dir()

    def test_suffix_is_diff(self, tmp_path: Path) -> None:
        pr = _make_pr()
        dest = _write_patch(pr, tmp_path)
        assert dest.suffix == ".diff"


# ---------------------------------------------------------------------------
# run_approval_workflow – approve
# ---------------------------------------------------------------------------

class TestApprovalWorkflowApprove:
    def _run(self, pr: MockPullRequest, inputs: list[str], tmp_path: Path) -> ApprovalDecision:
        """Helper: run the workflow with mocked input."""
        input_iter = iter(inputs)
        audit = tmp_path / AUDIT_LOG_FILENAME
        out_dir = tmp_path / "patches"
        return run_approval_workflow(
            pr,
            output_dir=out_dir,
            audit_log=audit,
            console=_null_console(),
            _input_fn=lambda _prompt="": next(input_iter),
        )

    def test_approve_returns_decision(self, tmp_path: Path) -> None:
        pr = _make_pr()
        decision = self._run(pr, ["approve"], tmp_path)
        assert isinstance(decision, ApprovalDecision)

    def test_approve_decision_value(self, tmp_path: Path) -> None:
        pr = _make_pr()
        decision = self._run(pr, ["approve"], tmp_path)
        assert decision.decision == "approve"

    def test_approve_writes_patch_file(self, tmp_path: Path) -> None:
        pr = _make_pr()
        decision = self._run(pr, ["approve"], tmp_path)
        assert decision.patch_written_to != ""
        assert Path(decision.patch_written_to).exists()

    def test_approve_patch_content_correct(self, tmp_path: Path) -> None:
        patch_text = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-x\n+y\n"
        pr = _make_pr(patch=patch_text)
        decision = self._run(pr, ["approve"], tmp_path)
        assert Path(decision.patch_written_to).read_text() == patch_text

    def test_approve_logs_to_audit(self, tmp_path: Path) -> None:
        pr = _make_pr()
        self._run(pr, ["approve"], tmp_path)
        entries = load_audit_trail(tmp_path / AUDIT_LOG_FILENAME)
        assert len(entries) == 1
        assert entries[0]["decision"] == "approve"
        assert entries[0]["pr_id"] == pr.pr_id

    def test_approve_audit_has_patch_written_to(self, tmp_path: Path) -> None:
        pr = _make_pr()
        self._run(pr, ["approve"], tmp_path)
        entries = load_audit_trail(tmp_path / AUDIT_LOG_FILENAME)
        assert entries[0]["patch_written_to"] != ""


# ---------------------------------------------------------------------------
# run_approval_workflow – reject
# ---------------------------------------------------------------------------

class TestApprovalWorkflowReject:
    def _run(self, pr: MockPullRequest, inputs: list[str], tmp_path: Path) -> ApprovalDecision:
        input_iter = iter(inputs)
        audit = tmp_path / AUDIT_LOG_FILENAME
        out_dir = tmp_path / "patches"
        return run_approval_workflow(
            pr,
            output_dir=out_dir,
            audit_log=audit,
            console=_null_console(),
            _input_fn=lambda _prompt="": next(input_iter),
        )

    def test_reject_returns_decision(self, tmp_path: Path) -> None:
        pr = _make_pr()
        # reject + optional note
        decision = self._run(pr, ["reject", "not correct"], tmp_path)
        assert decision.decision == "reject"

    def test_reject_does_not_write_patch(self, tmp_path: Path) -> None:
        pr = _make_pr()
        decision = self._run(pr, ["reject", ""], tmp_path)
        assert decision.patch_written_to == ""

    def test_reject_records_note(self, tmp_path: Path) -> None:
        pr = _make_pr()
        decision = self._run(pr, ["reject", "wrong fix"], tmp_path)
        assert decision.note == "wrong fix"

    def test_reject_logged_to_audit(self, tmp_path: Path) -> None:
        pr = _make_pr()
        self._run(pr, ["reject", ""], tmp_path)
        entries = load_audit_trail(tmp_path / AUDIT_LOG_FILENAME)
        assert entries[0]["decision"] == "reject"

    def test_reject_no_patch_dir_created(self, tmp_path: Path) -> None:
        """Rejecting should not create the patch file."""
        pr = _make_pr()
        self._run(pr, ["reject", ""], tmp_path)
        patch_files = list((tmp_path / "patches").glob("*.diff")) if (tmp_path / "patches").exists() else []
        assert patch_files == []


# ---------------------------------------------------------------------------
# run_approval_workflow – edit then approve
# ---------------------------------------------------------------------------

class TestApprovalWorkflowEdit:
    def _run(self, pr: MockPullRequest, inputs: list[str], tmp_path: Path) -> ApprovalDecision:
        input_iter = iter(inputs)
        audit = tmp_path / AUDIT_LOG_FILENAME
        out_dir = tmp_path / "patches"
        return run_approval_workflow(
            pr,
            output_dir=out_dir,
            audit_log=audit,
            console=_null_console(),
            _input_fn=lambda _prompt="": next(input_iter),
        )

    def test_edit_then_approve_decision(self, tmp_path: Path) -> None:
        pr = _make_pr()
        # edit → type new patch lines → END → approve
        inputs = ["edit", "--- a/f.py", "+++ b/f.py", "+fixed", "END", "approve"]
        decision = self._run(pr, inputs, tmp_path)
        assert decision.decision == "approve"

    def test_edit_then_approve_writes_edited_patch(self, tmp_path: Path) -> None:
        pr = _make_pr(patch="original patch")
        inputs = ["edit", "edited patch content", "END", "approve"]
        decision = self._run(pr, inputs, tmp_path)
        written = Path(decision.patch_written_to).read_text()
        assert "edited patch content" in written

    def test_edit_then_reject(self, tmp_path: Path) -> None:
        pr = _make_pr()
        inputs = ["edit", "some patch", "END", "reject", "changed my mind"]
        decision = self._run(pr, inputs, tmp_path)
        assert decision.decision == "reject"
        assert decision.patch_written_to == ""


# ---------------------------------------------------------------------------
# run_approval_workflow – invalid command retry
# ---------------------------------------------------------------------------

class TestApprovalWorkflowInvalidCommand:
    def _run(self, pr: MockPullRequest, inputs: list[str], tmp_path: Path) -> ApprovalDecision:
        input_iter = iter(inputs)
        audit = tmp_path / AUDIT_LOG_FILENAME
        out_dir = tmp_path / "patches"
        return run_approval_workflow(
            pr,
            output_dir=out_dir,
            audit_log=audit,
            console=_null_console(),
            _input_fn=lambda _prompt="": next(input_iter),
        )

    def test_invalid_then_approve(self, tmp_path: Path) -> None:
        pr = _make_pr()
        inputs = ["maybe", "yes", "approve"]
        decision = self._run(pr, inputs, tmp_path)
        assert decision.decision == "approve"

    def test_invalid_then_reject(self, tmp_path: Path) -> None:
        pr = _make_pr()
        inputs = ["oops", "reject", ""]
        decision = self._run(pr, inputs, tmp_path)
        assert decision.decision == "reject"


# ---------------------------------------------------------------------------
# run_approval_workflow – audit trail accumulation
# ---------------------------------------------------------------------------

class TestApprovalWorkflowAuditAccumulation:
    def test_multiple_prs_accumulate_in_audit(self, tmp_path: Path) -> None:
        audit = tmp_path / AUDIT_LOG_FILENAME
        out_dir = tmp_path / "patches"

        for i in range(3):
            pr = _make_pr()
            # Give each PR a distinct ID
            pr.pr_id = f"pr-id-000{i}-xxxx-yyyy"
            input_seq = iter(["approve"])
            run_approval_workflow(
                pr,
                output_dir=out_dir,
                audit_log=audit,
                console=_null_console(),
                _input_fn=lambda _p="": next(input_seq),
            )

        entries = load_audit_trail(audit)
        assert len(entries) == 3
        decisions = {e["decision"] for e in entries}
        assert decisions == {"approve"}


# ---------------------------------------------------------------------------
# CLI integration – 'approve' command
# ---------------------------------------------------------------------------

class TestCLIApproveCommand:
    """Verify the CLI wires up the approval command correctly."""

    def test_approve_command_exists(self) -> None:
        from click.testing import CliRunner
        from ci_triage_agent.cli import main
        runner = CliRunner()
        result = runner.invoke(main, ["approve", "--help"])
        assert result.exit_code == 0
        assert "PR artifact" in result.output or "json" in result.output.lower() or "approve" in result.output.lower()
