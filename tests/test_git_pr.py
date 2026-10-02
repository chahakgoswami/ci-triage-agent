"""Tests for the mock Git/PR layer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ci_triage_agent.agent import FixSuggestion
from ci_triage_agent.git_pr import (
    GitPRLayer,
    MockPullRequest,
    TestRunResult,
    _apply_patch_python,
    _build_broken_source,
    _extract_changed_files,
    _make_branch_name,
    _make_pr_title,
    _parse_unified_diff,
    _run_tests,
    _scaffold_test_repo,
    create_pull_request,
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
        raw_metadata=raw_metadata or {},
    )


def _make_suggestion(
    failure: ParsedFailure | None = None,
    patch: str = "",
    explanation: str = "Fix applied.",
    has_patch_override: bool | None = None,
) -> FixSuggestion:
    if failure is None:
        failure = _make_failure()
    return FixSuggestion(
        parsed_failure=failure,
        used_mock=True,
        explanation=explanation,
        patch=patch,
        target_file=failure.error_file,
        model="mock",
        raw_response="EXPLANATION:\nFix applied.\n\nPATCH:\n" + patch,
    )


def _minimal_patch(filepath: str = "src/app/utils.py") -> str:
    return (
        f"--- a/{filepath}\n"
        f"+++ b/{filepath}\n"
        "@@ -1,1 +1,1 @@\n"
        "-old_line\n"
        "+new_line\n"
    )


# ---------------------------------------------------------------------------
# TestRunResult
# ---------------------------------------------------------------------------

class TestTestRunResult:
    def test_combined_output(self) -> None:
        r = TestRunResult(
            passed=True, returncode=0, stdout="ok\n", stderr=""
        )
        assert "ok" in r.combined_output

    def test_defaults_zero_counts(self) -> None:
        r = TestRunResult(
            passed=False, returncode=1, stdout="", stderr="err"
        )
        assert r.tests_collected == 0
        assert r.tests_passed == 0
        assert r.tests_failed == 0


# ---------------------------------------------------------------------------
# MockPullRequest
# ---------------------------------------------------------------------------

class TestMockPullRequest:
    def _make_pr(self, patch_applied: bool = True, test_passed: bool = True, error: str | None = None) -> MockPullRequest:
        failure = _make_failure()
        suggestion = _make_suggestion(failure)
        test_result = TestRunResult(
            passed=test_passed,
            returncode=0 if test_passed else 1,
            stdout="1 passed" if test_passed else "1 failed",
            stderr="",
        )
        return MockPullRequest(
            pr_id="abcd-1234-efgh-5678",
            title="fix: something",
            branch_name="ci-triage/syntax_error/src-app-utils-py/abcd1234",
            diff="--- a/f.py\n+++ b/f.py\n",
            changed_files=["src/app/utils.py"],
            test_result=test_result,
            parsed_failure=failure,
            fix_suggestion=suggestion,
            patch_applied=patch_applied,
            error=error,
        )

    def test_status_tests_pass(self) -> None:
        pr = self._make_pr(patch_applied=True, test_passed=True)
        assert pr.status == "tests_pass"

    def test_status_tests_fail(self) -> None:
        pr = self._make_pr(patch_applied=True, test_passed=False)
        assert pr.status == "tests_fail"

    def test_status_patch_failed(self) -> None:
        pr = self._make_pr(patch_applied=False, test_passed=False)
        assert pr.status == "patch_failed"

    def test_status_error(self) -> None:
        pr = self._make_pr(patch_applied=False, error="boom")
        assert pr.status == "error"

    def test_to_dict_has_required_keys(self) -> None:
        pr = self._make_pr()
        d = pr.to_dict()
        required = {
            "schema_version", "pr_id", "title", "branch_name", "diff",
            "changed_files", "status", "patch_applied", "created_at",
            "labels", "artifact_path", "error", "test_result",
            "fix_suggestion", "parsed_failure",
        }
        assert required.issubset(d.keys())

    def test_to_dict_schema_version(self) -> None:
        pr = self._make_pr()
        assert pr.to_dict()["schema_version"] == "1.0"

    def test_save_creates_json_file(self, tmp_path: Path) -> None:
        pr = self._make_pr()
        dest = pr.save(tmp_path)
        assert dest.exists()
        assert dest.suffix == ".json"

    def test_save_json_is_valid(self, tmp_path: Path) -> None:
        pr = self._make_pr()
        dest = pr.save(tmp_path)
        data = json.loads(dest.read_text(encoding="utf-8"))
        assert isinstance(data, dict)
        assert data["pr_id"] == pr.pr_id

    def test_save_sets_artifact_path(self, tmp_path: Path) -> None:
        pr = self._make_pr()
        dest = pr.save(tmp_path)
        assert pr.artifact_path == str(dest)

    def test_save_creates_output_dir(self, tmp_path: Path) -> None:
        pr = self._make_pr()
        out = tmp_path / "artifacts"
        pr.save(out)
        assert out.is_dir()


# ---------------------------------------------------------------------------
# _make_branch_name
# ---------------------------------------------------------------------------

class TestMakeBranchName:
    def test_contains_failure_type(self) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR, error_file="src/app/utils.py")
        name = _make_branch_name(failure, "abcd-1234")
        assert "syntax_error" in name

    def test_contains_short_id(self) -> None:
        failure = _make_failure()
        name = _make_branch_name(failure, "abcd-1234-efgh")
        assert "abcd-123" in name

    def test_no_slashes_in_file_segment(self) -> None:
        failure = _make_failure(error_file="src/app/utils.py")
        name = _make_branch_name(failure, "1234")
        # Only the ci-triage/ prefix slashes should exist
        parts = name.split("/")
        assert len(parts) >= 3

    def test_branch_safe_characters(self) -> None:
        failure = _make_failure(error_file="src/app/some.module.py")
        name = _make_branch_name(failure, "abc")
        # No spaces, no special shell chars
        for ch in " @#$%^&*()[]{}|;'\"<>":
            assert ch not in name


# ---------------------------------------------------------------------------
# _make_pr_title
# ---------------------------------------------------------------------------

class TestMakePrTitle:
    def test_contains_failure_type(self) -> None:
        failure = _make_failure(FailureType.LINT_ERROR, error_message="E501 line too long")
        title = _make_pr_title(failure)
        assert "Lint Error" in title or "lint_error" in title.lower() or "Lint" in title

    def test_contains_error_file(self) -> None:
        failure = _make_failure(error_file="src/app/utils.py")
        title = _make_pr_title(failure)
        assert "src/app/utils.py" in title

    def test_title_is_string(self) -> None:
        failure = _make_failure()
        assert isinstance(_make_pr_title(failure), str)


# ---------------------------------------------------------------------------
# _extract_changed_files
# ---------------------------------------------------------------------------

class TestExtractChangedFiles:
    def test_single_file(self) -> None:
        patch = "--- a/src/app/utils.py\n+++ b/src/app/utils.py\n@@ -1,1 +1,1 @@\n-old\n+new\n"
        files = _extract_changed_files(patch)
        assert files == ["src/app/utils.py"]

    def test_multiple_files(self) -> None:
        patch = (
            "--- a/file1.py\n+++ b/file1.py\n@@ -1 +1 @@\n-a\n+b\n"
            "--- a/file2.py\n+++ b/file2.py\n@@ -1 +1 @@\n-c\n+d\n"
        )
        files = _extract_changed_files(patch)
        assert "file1.py" in files
        assert "file2.py" in files

    def test_empty_patch(self) -> None:
        assert _extract_changed_files("") == []

    def test_strips_b_prefix(self) -> None:
        patch = "+++ b/requirements.txt\n"
        files = _extract_changed_files(patch)
        assert files == ["requirements.txt"]

    def test_deduplicates(self) -> None:
        patch = "+++ b/f.py\n+++ b/f.py\n"
        files = _extract_changed_files(patch)
        assert files.count("f.py") == 1


# ---------------------------------------------------------------------------
# _parse_unified_diff
# ---------------------------------------------------------------------------

class TestParseUnifiedDiff:
    def test_single_hunk(self) -> None:
        patch = (
            "--- a/src/app/utils.py\n"
            "+++ b/src/app/utils.py\n"
            "@@ -42,1 +42,1 @@\n"
            "-old line\n"
            "+new line\n"
        )
        hunks = _parse_unified_diff(patch)
        assert "src/app/utils.py" in hunks
        assert len(hunks["src/app/utils.py"]) == 1

    def test_multiple_hunks(self) -> None:
        patch = (
            "--- a/f.py\n"
            "+++ b/f.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-a\n"
            "+b\n"
            "@@ -10,1 +10,1 @@\n"
            "-c\n"
            "+d\n"
        )
        hunks = _parse_unified_diff(patch)
        assert len(hunks["f.py"]) == 2

    def test_empty_patch(self) -> None:
        hunks = _parse_unified_diff("")
        assert hunks == {}


# ---------------------------------------------------------------------------
# _apply_patch_python
# ---------------------------------------------------------------------------

class TestApplyPatchPython:
    def test_adds_new_line(self, tmp_path: Path) -> None:
        target = tmp_path / "src" / "app" / "utils.py"
        target.parent.mkdir(parents=True)
        target.write_text("x = 1\n", encoding="utf-8")

        patch = (
            "--- a/src/app/utils.py\n"
            "+++ b/src/app/utils.py\n"
            "@@ -1,1 +1,2 @@\n"
            " x = 1\n"
            "+y = 2\n"
        )
        ok, err = _apply_patch_python(tmp_path, patch)
        assert ok is True
        assert err == ""
        content = target.read_text(encoding="utf-8")
        assert "y = 2" in content

    def test_empty_patch_fails(self, tmp_path: Path) -> None:
        ok, err = _apply_patch_python(tmp_path, "")
        assert ok is False

    def test_creates_new_file(self, tmp_path: Path) -> None:
        patch = (
            "--- a/requirements.txt\n"
            "+++ b/requirements.txt\n"
            "@@ -0,0 +1,1 @@\n"
            "+click>=8.1\n"
        )
        ok, err = _apply_patch_python(tmp_path, patch)
        assert ok is True
        dest = tmp_path / "requirements.txt"
        assert dest.exists()
        assert "click" in dest.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# _scaffold_test_repo
# ---------------------------------------------------------------------------

class TestScaffoldTestRepo:
    def test_creates_error_file(self, tmp_path: Path) -> None:
        failure = _make_failure(error_file="src/app/utils.py")
        _scaffold_test_repo(tmp_path, failure)
        assert (tmp_path / "src" / "app" / "utils.py").exists()

    def test_creates_tests_dir(self, tmp_path: Path) -> None:
        failure = _make_failure()
        _scaffold_test_repo(tmp_path, failure)
        assert (tmp_path / "tests").is_dir()

    def test_creates_smoke_test(self, tmp_path: Path) -> None:
        failure = _make_failure()
        _scaffold_test_repo(tmp_path, failure)
        assert (tmp_path / "tests" / "test_smoke.py").exists()

    def test_creates_pyproject_toml(self, tmp_path: Path) -> None:
        failure = _make_failure()
        _scaffold_test_repo(tmp_path, failure)
        assert (tmp_path / "pyproject.toml").exists()


# ---------------------------------------------------------------------------
# _build_broken_source
# ---------------------------------------------------------------------------

class TestBuildBrokenSource:
    def test_syntax_error_source_parseable(self) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR, snippet="def foo(")
        src = _build_broken_source(failure, "def foo(")
        # The stub source file should itself be valid Python (broken code is in a comment)
        try:
            compile(src, "<test>", "exec")
        except SyntaxError:
            pytest.fail("_build_broken_source produced unparseable Python for SYNTAX_ERROR")

    def test_import_error_source_parseable(self) -> None:
        failure = _make_failure(
            FailureType.IMPORT_ERROR,
            error_message="ModuleNotFoundError: No module named 'dotenv'",
        )
        src = _build_broken_source(failure, "")
        compile(src, "<test>", "exec")

    def test_test_failure_source_has_function(self) -> None:
        failure = _make_failure(
            FailureType.TEST_FAILURE,
            error_message="AssertionError: assert 0 == 1",
            test_name="test_my_func",
        )
        src = _build_broken_source(failure, "")
        assert "def test_my_func" in src

    def test_lint_e501_source_has_long_line(self) -> None:
        failure = _make_failure(
            FailureType.LINT_ERROR,
            error_message="E501 line too long (102 > 88 characters)",
        )
        src = _build_broken_source(failure, "")
        assert any(len(line) > 88 for line in src.splitlines())

    def test_lint_f401_source_has_import(self) -> None:
        failure = _make_failure(
            FailureType.LINT_ERROR,
            error_message="F401 'os' imported but unused",
        )
        src = _build_broken_source(failure, "")
        assert "import os" in src


# ---------------------------------------------------------------------------
# _run_tests – integration (real subprocess)
# ---------------------------------------------------------------------------

class TestRunTests:
    def test_smoke_test_passes(self, tmp_path: Path) -> None:
        failure = _make_failure()
        _scaffold_test_repo(tmp_path, failure)
        result = _run_tests(tmp_path, timeout=60)
        assert result.passed is True
        assert result.returncode == 0

    def test_result_has_stdout(self, tmp_path: Path) -> None:
        failure = _make_failure()
        _scaffold_test_repo(tmp_path, failure)
        result = _run_tests(tmp_path, timeout=60)
        assert result.stdout != "" or result.stderr != ""

    def test_passed_count_populated(self, tmp_path: Path) -> None:
        failure = _make_failure()
        _scaffold_test_repo(tmp_path, failure)
        result = _run_tests(tmp_path, timeout=60)
        # At least 1 test (the smoke test) should have passed
        assert result.tests_passed >= 1


# ---------------------------------------------------------------------------
# GitPRLayer – full integration
# ---------------------------------------------------------------------------

class TestGitPRLayerIntegration:
    def test_create_pr_returns_mock_pull_request(self, tmp_path: Path) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR, error_file="src/app/utils.py")
        suggestion = _make_suggestion(failure, patch=_minimal_patch("src/app/utils.py"))
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        assert isinstance(pr, MockPullRequest)

    def test_create_pr_persists_json(self, tmp_path: Path) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR, error_file="src/app/utils.py")
        suggestion = _make_suggestion(failure, patch=_minimal_patch("src/app/utils.py"))
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        assert pr.artifact_path != ""
        assert Path(pr.artifact_path).exists()

    def test_create_pr_json_is_valid(self, tmp_path: Path) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR, error_file="src/app/utils.py")
        suggestion = _make_suggestion(failure, patch=_minimal_patch("src/app/utils.py"))
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        data = json.loads(Path(pr.artifact_path).read_text(encoding="utf-8"))
        assert data["pr_id"] == pr.pr_id

    def test_create_pr_branch_name_set(self, tmp_path: Path) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        suggestion = _make_suggestion(failure)
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        assert "syntax_error" in pr.branch_name

    def test_create_pr_has_test_result(self, tmp_path: Path) -> None:
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        suggestion = _make_suggestion(failure)
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        assert isinstance(pr.test_result, TestRunResult)

    def test_create_pr_no_patch_marks_no_apply(self, tmp_path: Path) -> None:
        failure = _make_failure(FailureType.UNKNOWN)
        suggestion = _make_suggestion(failure, patch="")
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        assert pr.patch_applied is False

    def test_create_pr_labels_contain_failure_type(self, tmp_path: Path) -> None:
        failure = _make_failure(FailureType.LINT_ERROR, error_message="E501 line too long")
        suggestion = _make_suggestion(failure)
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        assert "lint_error" in pr.labels

    def test_create_pr_labels_contain_mock_llm(self, tmp_path: Path) -> None:
        failure = _make_failure()
        suggestion = _make_suggestion(failure)
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        assert "mock-llm" in pr.labels

    def test_create_pr_diff_in_artifact(self, tmp_path: Path) -> None:
        failure = _make_failure(error_file="src/app/utils.py")
        patch_text = _minimal_patch("src/app/utils.py")
        suggestion = _make_suggestion(failure, patch=patch_text)
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        data = json.loads(Path(pr.artifact_path).read_text(encoding="utf-8"))
        assert data["diff"] == patch_text

    def test_create_pr_test_smoke_passes(self, tmp_path: Path) -> None:
        """Even without a real patch, the smoke test should pass."""
        failure = _make_failure(FailureType.SYNTAX_ERROR)
        suggestion = _make_suggestion(failure, patch="")
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        # Smoke test always passes regardless of patch status
        assert pr.test_result.passed is True

    def test_create_pr_import_error(self, tmp_path: Path) -> None:
        failure = _make_failure(
            FailureType.IMPORT_ERROR,
            error_message="ModuleNotFoundError: No module named 'dotenv'",
            error_file="src/app/config.py",
        )
        patch_text = (
            "--- a/requirements.txt\n"
            "+++ b/requirements.txt\n"
            "@@ -0,0 +1,1 @@\n"
            "+python-dotenv\n"
        )
        suggestion = _make_suggestion(failure, patch=patch_text)
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        pr = layer.create_pull_request(suggestion)
        assert isinstance(pr, MockPullRequest)
        data = json.loads(Path(pr.artifact_path).read_text(encoding="utf-8"))
        assert "python-dotenv" in data["diff"]

    def test_create_pr_exception_handled(self, tmp_path: Path) -> None:
        """If the internal build raises, we still get a PR with error set."""
        failure = _make_failure()
        suggestion = _make_suggestion(failure)
        layer = GitPRLayer(output_dir=tmp_path / "prs", test_timeout=60)
        with patch("ci_triage_agent.git_pr._scaffold_test_repo", side_effect=RuntimeError("disk full")):
            pr = layer.create_pull_request(suggestion)
        assert pr.error is not None
        assert "disk full" in pr.error
        # Artifact should still be persisted
        assert Path(pr.artifact_path).exists()


# ---------------------------------------------------------------------------
# create_pull_request convenience function
# ---------------------------------------------------------------------------

class TestCreatePullRequestFunction:
    def test_returns_mock_pull_request(self, tmp_path: Path) -> None:
        failure = _make_failure()
        suggestion = _make_suggestion(failure)
        pr = create_pull_request(suggestion, output_dir=tmp_path)
        assert isinstance(pr, MockPullRequest)

    def test_artifact_created(self, tmp_path: Path) -> None:
        failure = _make_failure()
        suggestion = _make_suggestion(failure)
        pr = create_pull_request(suggestion, output_dir=tmp_path)
        assert Path(pr.artifact_path).exists()
