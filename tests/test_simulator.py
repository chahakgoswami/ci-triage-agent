"""Tests for the CI pipeline simulator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ci_triage_agent.simulator import (
    CIPipelineSimulator,
    _FAILURE_CATEGORIES,
    _build_log_text,
)


class TestCIPipelineSimulator:
    """Unit tests for CIPipelineSimulator."""

    def test_generate_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "logs"
        sim = CIPipelineSimulator(output_dir=str(out), seed=0)
        sim.generate(count=1)
        assert out.is_dir()

    def test_generate_returns_correct_number_of_paths(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=1)
        paths = sim.generate(count=3)
        # Each log produces a .json and a .txt file → 3 * 2 = 6 paths
        assert len(paths) == 6

    def test_generate_files_exist_on_disk(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=2)
        paths = sim.generate(count=2)
        for p in paths:
            assert p.exists(), f"Expected file not found: {p}"

    def test_json_files_are_valid_json(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=3)
        paths = sim.generate(count=4)
        json_paths = [p for p in paths if p.suffix == ".json"]
        for jp in json_paths:
            data = json.loads(jp.read_text(encoding="utf-8"))
            assert isinstance(data, dict)

    def test_json_schema_fields_present(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=4)
        paths = sim.generate(count=4)
        json_paths = [p for p in paths if p.suffix == ".json"]
        required_fields = {"schema_version", "run_id", "timestamp", "category", "status", "metadata"}
        for jp in json_paths:
            data = json.loads(jp.read_text(encoding="utf-8"))
            assert required_fields.issubset(data.keys()), f"Missing fields in {jp.name}: {required_fields - data.keys()}"

    def test_json_category_is_valid(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=5)
        paths = sim.generate(count=8)
        json_paths = [p for p in paths if p.suffix == ".json"]
        for jp in json_paths:
            data = json.loads(jp.read_text(encoding="utf-8"))
            assert data["category"] in _FAILURE_CATEGORIES

    def test_json_status_is_failed(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=6)
        paths = sim.generate(count=4)
        json_paths = [p for p in paths if p.suffix == ".json"]
        for jp in json_paths:
            data = json.loads(jp.read_text(encoding="utf-8"))
            assert data["status"] == "failed"

    def test_txt_files_contain_build_failed(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=7)
        paths = sim.generate(count=4)
        txt_paths = [p for p in paths if p.suffix == ".txt"]
        for tp in txt_paths:
            content = tp.read_text(encoding="utf-8")
            assert "Build FAILED" in content

    def test_txt_files_contain_run_id(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=8)
        paths = sim.generate(count=2)
        json_paths = [p for p in paths if p.suffix == ".json"]
        txt_paths = [p for p in paths if p.suffix == ".txt"]
        # Each txt filename shares the run_id prefix with its json counterpart
        for jp in json_paths:
            data = json.loads(jp.read_text(encoding="utf-8"))
            run_id = data["run_id"]
            matching_txt = jp.with_suffix(".txt")
            assert matching_txt in txt_paths
            assert run_id in matching_txt.read_text(encoding="utf-8")

    def test_seeded_generation_is_reproducible(self, tmp_path: Path) -> None:
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        sim1 = CIPipelineSimulator(output_dir=str(out1), seed=42)
        sim2 = CIPipelineSimulator(output_dir=str(out2), seed=42)
        paths1 = sim1.generate(count=3)
        paths2 = sim2.generate(count=3)
        json_paths1 = sorted(p for p in paths1 if p.suffix == ".json")
        json_paths2 = sorted(p for p in paths2 if p.suffix == ".json")
        for jp1, jp2 in zip(json_paths1, json_paths2):
            d1 = json.loads(jp1.read_text())
            d2 = json.loads(jp2.read_text())
            # Category and metadata should be identical; timestamps may differ slightly
            assert d1["category"] == d2["category"]
            assert d1["metadata"] == d2["metadata"]

    def test_generate_zero_count(self, tmp_path: Path) -> None:
        sim = CIPipelineSimulator(output_dir=str(tmp_path), seed=0)
        paths = sim.generate(count=0)
        assert paths == []


class TestBuildLogText:
    """Unit tests for the _build_log_text helper."""

    def _make_record(self, category: str, metadata: dict) -> dict:
        return {
            "schema_version": "1.0",
            "run_id": "abcd-1234",
            "timestamp": "2024-01-01T00:00:00+00:00",
            "category": category,
            "status": "failed",
            "metadata": metadata,
        }

    def test_syntax_error_log_contains_syntax_error(self) -> None:
        record = self._make_record(
            "syntax_error",
            {"file": "src/app/utils.py", "line": 42, "message": "SyntaxError: invalid syntax",
             "snippet": "    def foo()\n        ^", "details": "Missing colon."},
        )
        text = _build_log_text(record)
        assert "SyntaxError" in text
        assert "[FAIL] build" in text

    def test_import_error_log_contains_module_not_found(self) -> None:
        record = self._make_record(
            "import_error",
            {"file": "src/app/config.py", "line": 5,
             "message": "ModuleNotFoundError: No module named 'dotenv'",
             "snippet": "from dotenv import load_dotenv", "details": "Missing package."},
        )
        text = _build_log_text(record)
        assert "ModuleNotFoundError" in text
        assert "[FAIL] build" in text

    def test_test_failure_log_contains_failed_marker(self) -> None:
        record = self._make_record(
            "test_failure",
            {"file": "tests/test_utils.py", "line": 10, "test_name": "test_foo",
             "message": "AssertionError: assert 0 == 1",
             "stack_trace": "Traceback...", "details": "Wrong value."},
        )
        text = _build_log_text(record)
        assert "FAILED" in text
        assert "[FAIL] test" in text
        assert "[PASS] build" in text

    def test_lint_error_log_contains_lint_marker(self) -> None:
        record = self._make_record(
            "lint_error",
            {"file": "src/app/helpers.py", "line": 7,
             "message": "E501 line too long",
             "snippet": "    very_long_line...", "details": "Exceeds max length."},
        )
        text = _build_log_text(record)
        assert "[FAIL] lint" in text
        assert "E501" in text
