"""Tests for the log ingestion and parsing layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ci_triage_agent.parser import (
    FailureType,
    ParsedFailure,
    ingest_directory,
    parse_log_file,
)
from ci_triage_agent.simulator import CIPipelineSimulator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_json_log(tmp_path: Path, category: str, metadata: dict) -> Path:
    record = {
        "schema_version": "1.0",
        "run_id": "test-run-0001",
        "timestamp": "2024-06-01T12:00:00+00:00",
        "category": category,
        "status": "failed",
        "metadata": metadata,
    }
    p = tmp_path / f"{category}_test.json"
    p.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# ParsedFailure dataclass
# ---------------------------------------------------------------------------

class TestParsedFailure:
    def test_is_dataclass(self) -> None:
        pf = ParsedFailure(
            run_id="abc",
            source_file="/tmp/foo.json",
            failure_type=FailureType.SYNTAX_ERROR,
            error_file="src/app/utils.py",
            error_line=42,
            error_message="SyntaxError: invalid syntax",
        )
        assert pf.run_id == "abc"
        assert pf.failure_type == FailureType.SYNTAX_ERROR
        assert pf.error_line == 42
        assert pf.test_name is None
        assert pf.stack_trace is None

    def test_raw_metadata_defaults_to_empty_dict(self) -> None:
        pf = ParsedFailure(
            run_id="r",
            source_file="f",
            failure_type=FailureType.UNKNOWN,
            error_file="",
            error_line=0,
            error_message="",
        )
        assert pf.raw_metadata == {}


# ---------------------------------------------------------------------------
# FailureType enum
# ---------------------------------------------------------------------------

class TestFailureType:
    def test_values(self) -> None:
        assert FailureType.SYNTAX_ERROR.value == "syntax_error"
        assert FailureType.IMPORT_ERROR.value == "import_error"
        assert FailureType.TEST_FAILURE.value == "test_failure"
        assert FailureType.LINT_ERROR.value == "lint_error"
        assert FailureType.UNKNOWN.value == "unknown"

    def test_is_str(self) -> None:
        # FailureType extends str
        assert isinstance(FailureType.LINT_ERROR, str)


# ---------------------------------------------------------------------------
# parse_log_file – JSON path
# ---------------------------------------------------------------------------

class TestParseLogFileJSON:
    def test_syntax_error(self, tmp_path: Path) -> None:
        p = _make_json_log(
            tmp_path,
            "syntax_error",
            {
                "file": "src/app/utils.py",
                "line": 42,
                "message": "SyntaxError: invalid syntax",
                "snippet": "    def foo()",
                "details": "Missing colon.",
            },
        )
        result = parse_log_file(p)
        assert result.failure_type == FailureType.SYNTAX_ERROR
        assert result.error_file == "src/app/utils.py"
        assert result.error_line == 42
        assert "SyntaxError" in result.error_message
        assert result.snippet == "    def foo()"
        assert result.details == "Missing colon."
        assert result.run_id == "test-run-0001"

    def test_import_error(self, tmp_path: Path) -> None:
        p = _make_json_log(
            tmp_path,
            "import_error",
            {
                "file": "src/app/config.py",
                "line": 5,
                "message": "ModuleNotFoundError: No module named 'dotenv'",
                "snippet": "from dotenv import load_dotenv",
                "details": "Missing package.",
            },
        )
        result = parse_log_file(p)
        assert result.failure_type == FailureType.IMPORT_ERROR
        assert result.error_file == "src/app/config.py"
        assert result.error_line == 5
        assert "ModuleNotFoundError" in result.error_message

    def test_test_failure(self, tmp_path: Path) -> None:
        p = _make_json_log(
            tmp_path,
            "test_failure",
            {
                "file": "tests/test_utils.py",
                "line": 34,
                "test_name": "test_compute_total_empty_list",
                "message": "AssertionError: assert 0 == 1",
                "stack_trace": "Traceback (most recent call last):\n  ...",
                "details": "Wrong return value.",
            },
        )
        result = parse_log_file(p)
        assert result.failure_type == FailureType.TEST_FAILURE
        assert result.test_name == "test_compute_total_empty_list"
        assert result.stack_trace == "Traceback (most recent call last):\n  ..."
        assert result.error_line == 34

    def test_lint_error(self, tmp_path: Path) -> None:
        p = _make_json_log(
            tmp_path,
            "lint_error",
            {
                "file": "src/app/helpers.py",
                "line": 7,
                "message": "E501 line too long (102 > 88 characters)",
                "snippet": "    very_long_line...",
                "details": "Exceeds max length.",
            },
        )
        result = parse_log_file(p)
        assert result.failure_type == FailureType.LINT_ERROR
        assert result.error_file == "src/app/helpers.py"
        assert result.error_line == 7
        assert "E501" in result.error_message

    def test_unknown_category(self, tmp_path: Path) -> None:
        record = {
            "schema_version": "1.0",
            "run_id": "xyz",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "category": "mystery_error",
            "status": "failed",
            "metadata": {"file": "x.py", "line": 1, "message": "Oops"},
        }
        p = tmp_path / "mystery.json"
        p.write_text(json.dumps(record), encoding="utf-8")
        result = parse_log_file(p)
        assert result.failure_type == FailureType.UNKNOWN

    def test_raw_metadata_preserved(self, tmp_path: Path) -> None:
        meta = {
            "file": "src/app/utils.py",
            "line": 10,
            "message": "SyntaxError: invalid syntax",
            "custom_key": "custom_value",
        }
        p = _make_json_log(tmp_path, "syntax_error", meta)
        result = parse_log_file(p)
        assert result.raw_metadata["custom_key"] == "custom_value"

    def test_source_file_recorded(self, tmp_path: Path) -> None:
        p = _make_json_log(
            tmp_path,
            "lint_error",
            {"file": "f.py", "line": 1, "message": "E001"},
        )
        result = parse_log_file(p)
        assert result.source_file == str(p)

    def test_timestamp_recorded(self, tmp_path: Path) -> None:
        p = _make_json_log(
            tmp_path,
            "lint_error",
            {"file": "f.py", "line": 1, "message": "E001"},
        )
        result = parse_log_file(p)
        assert result.timestamp == "2024-06-01T12:00:00+00:00"


# ---------------------------------------------------------------------------
# parse_log_file – TXT path
# ---------------------------------------------------------------------------

class TestParseLogFileTXT:
    """Tests for parsing plain-text log files."""

    def _write_txt(self, tmp_path: Path, name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_lint_error_txt(self, tmp_path: Path) -> None:
        text = (
            "=== CI Build Log ===\n"
            "Run ID  : abc123\n"
            "Started : 2024-01-01T00:00:00+00:00\n"
            "[PASS] checkout\n"
            "[FAIL] lint\n"
            "src/app/helpers.py:7: E501 line too long\n"
            "=== Build FAILED ===\n"
        )
        p = self._write_txt(tmp_path, "lint.txt", text)
        result = parse_log_file(p)
        assert result.failure_type == FailureType.LINT_ERROR
        assert result.error_file == "src/app/helpers.py"
        assert result.error_line == 7
        assert "E501" in result.error_message

    def test_syntax_error_txt(self, tmp_path: Path) -> None:
        text = (
            "=== CI Build Log ===\n"
            "Run ID  : def456\n"
            "[PASS] checkout\n"
            "[FAIL] build\n"
            '  File "src/app/utils.py", line 42\n'
            "SyntaxError: invalid syntax\n"
            "=== Build FAILED ===\n"
        )
        p = self._write_txt(tmp_path, "syntax.txt", text)
        result = parse_log_file(p)
        assert result.failure_type == FailureType.SYNTAX_ERROR
        assert result.error_file == "src/app/utils.py"
        assert result.error_line == 42
        assert "SyntaxError" in result.error_message

    def test_import_error_txt(self, tmp_path: Path) -> None:
        text = (
            "=== CI Build Log ===\n"
            "Run ID  : ghi789\n"
            "[FAIL] build\n"
            '  File "src/app/config.py", line 5\n'
            "ModuleNotFoundError: No module named 'dotenv'\n"
            "=== Build FAILED ===\n"
        )
        p = self._write_txt(tmp_path, "import.txt", text)
        result = parse_log_file(p)
        assert result.failure_type == FailureType.IMPORT_ERROR
        assert "ModuleNotFoundError" in result.error_message

    def test_test_failure_txt(self, tmp_path: Path) -> None:
        text = (
            "=== CI Build Log ===\n"
            "Run ID  : jkl000\n"
            "[PASS] build\n"
            "[FAIL] test\n"
            "FAILED tests/test_utils.py::test_compute_total_empty_list\n"
            "Traceback (most recent call last):\n"
            "  File \"tests/test_utils.py\", line 34, in test_compute_total_empty_list\n"
            "    assert compute_total([]) == 1\n"
            "AssertionError: assert 0 == 1\n"
            "=== Build FAILED ===\n"
        )
        p = self._write_txt(tmp_path, "test_fail.txt", text)
        result = parse_log_file(p)
        assert result.failure_type == FailureType.TEST_FAILURE
        assert result.test_name == "test_compute_total_empty_list"
        assert result.stack_trace is not None
        assert "AssertionError" in result.error_message

    def test_run_id_extracted_from_txt(self, tmp_path: Path) -> None:
        text = (
            "Run ID  : myrunid-123\n"
            "[FAIL] lint\n"
            "f.py:1: E001 msg\n"
            "=== Build FAILED ===\n"
        )
        p = self._write_txt(tmp_path, "x.txt", text)
        result = parse_log_file(p)
        assert result.run_id == "myrunid-123"

    def test_unknown_txt(self, tmp_path: Path) -> None:
        text = "Some random log without known markers.\n=== Build FAILED ===\n"
        p = self._write_txt(tmp_path, "unknown.txt", text)
        result = parse_log_file(p)
        assert result.failure_type == FailureType.UNKNOWN


# ---------------------------------------------------------------------------
# parse_log_file – error cases
# ---------------------------------------------------------------------------

class TestParseLogFileErrors:
    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            parse_log_file(tmp_path / "nonexistent.json")

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "log.csv"
        p.write_text("data", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported"):
            parse_log_file(p)


# ---------------------------------------------------------------------------
# ingest_directory
# ---------------------------------------------------------------------------

class TestIngestDirectory:
    def test_ingest_simulator_output(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=10)
        sim.generate(count=4)
        failures = ingest_directory(tmp_path)
        # 4 JSON files → 4 ParsedFailure objects (txt siblings skipped)
        assert len(failures) == 4

    def test_all_results_are_parsed_failures(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=11)
        sim.generate(count=2)
        failures = ingest_directory(tmp_path)
        for f in failures:
            assert isinstance(f, ParsedFailure)

    def test_failure_types_are_valid(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=12)
        sim.generate(count=8)
        failures = ingest_directory(tmp_path)
        valid_types = set(FailureType)
        for f in failures:
            assert f.failure_type in valid_types

    def test_error_files_non_empty(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=13)
        sim.generate(count=4)
        failures = ingest_directory(tmp_path)
        for f in failures:
            assert f.error_file != ""

    def test_error_lines_positive(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=14)
        sim.generate(count=4)
        failures = ingest_directory(tmp_path)
        for f in failures:
            assert f.error_line > 0

    def test_txt_only_files_ingested_without_json_sibling(self, tmp_path: Path) -> None:
        # Write a lone .txt file (no matching .json)
        text = (
            "Run ID  : lone-txt-1\n"
            "[FAIL] lint\n"
            "src/foo.py:3: E302 expected 2 blank lines\n"
            "=== Build FAILED ===\n"
        )
        (tmp_path / "lone_lint.txt").write_text(text, encoding="utf-8")
        failures = ingest_directory(tmp_path)
        assert len(failures) == 1
        assert failures[0].failure_type == FailureType.LINT_ERROR

    def test_json_preferred_over_sibling_txt(self, tmp_path: Path) -> None:
        # Create a JSON+TXT pair: only the JSON should be ingested
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=15)
        sim.generate(count=1)
        failures = ingest_directory(tmp_path)
        assert len(failures) == 1
        assert failures[0].source_file.endswith(".json")

    def test_empty_directory(self, tmp_path: Path) -> None:
        failures = ingest_directory(tmp_path)
        assert failures == []

    def test_not_a_directory_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "file.json"
        p.write_text("{}", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            ingest_directory(p)

    def test_run_ids_match_json_records(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=20)
        sim.generate(count=3)
        failures = ingest_directory(tmp_path)
        for f in failures:
            json_path = Path(f.source_file)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            assert f.run_id == data["run_id"]
