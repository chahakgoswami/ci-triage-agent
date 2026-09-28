"""Mock CI pipeline simulator.

Generates realistic failing build logs (syntax errors, test failures,
import errors) and persists them as JSON + plain-text files.
"""

from __future__ import annotations

import json
import os
import random
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Templates for each failure category
# ---------------------------------------------------------------------------

_SYNTAX_ERROR_TEMPLATES: list[dict[str, Any]] = [
    {
        "file": "src/app/utils.py",
        "line": 42,
        "message": "SyntaxError: invalid syntax",
        "snippet": "    def compute_total(items)\n                              ^",
        "details": "Missing colon at end of function definition.",
    },
    {
        "file": "src/app/models.py",
        "line": 17,
        "message": "SyntaxError: unexpected EOF while parsing",
        "snippet": "class UserProfile(\n                 ^",
        "details": "Unclosed parenthesis in class definition.",
    },
    {
        "file": "src/app/views.py",
        "line": 88,
        "message": "SyntaxError: invalid syntax",
        "snippet": "    return render(request 'home.html')\n                         ^",
        "details": "Missing comma between function arguments.",
    },
]

_IMPORT_ERROR_TEMPLATES: list[dict[str, Any]] = [
    {
        "file": "src/app/services.py",
        "line": 3,
        "message": "ModuleNotFoundError: No module named 'pydantic_settings'",
        "snippet": "from pydantic_settings import BaseSettings",
        "details": "Package 'pydantic-settings' is not listed in requirements.txt.",
    },
    {
        "file": "src/app/tasks.py",
        "line": 1,
        "message": "ImportError: cannot import name 'celery_app' from 'app.celery'",
        "snippet": "from app.celery import celery_app",
        "details": "'celery_app' was renamed to 'app' in a recent refactor.",
    },
    {
        "file": "src/app/config.py",
        "line": 5,
        "message": "ModuleNotFoundError: No module named 'dotenv'",
        "snippet": "from dotenv import load_dotenv",
        "details": "Package 'python-dotenv' is missing from the virtual environment.",
    },
]

_TEST_FAILURE_TEMPLATES: list[dict[str, Any]] = [
    {
        "file": "tests/test_utils.py",
        "line": 34,
        "test_name": "test_compute_total_empty_list",
        "message": "AssertionError: assert 0 == 1",
        "stack_trace": textwrap.dedent(
            """\
            Traceback (most recent call last):
              File \"tests/test_utils.py\", line 34, in test_compute_total_empty_list
                assert compute_total([]) == 1
            AssertionError: assert 0 == 1
            """
        ),
        "details": "Function returns 0 for empty list; expected default value of 1.",
    },
    {
        "file": "tests/test_models.py",
        "line": 58,
        "test_name": "test_user_profile_creation",
        "message": "TypeError: __init__() missing 1 required positional argument: 'email'",
        "stack_trace": textwrap.dedent(
            """\
            Traceback (most recent call last):
              File \"tests/test_models.py\", line 58, in test_user_profile_creation
                profile = UserProfile(name='Alice')
            TypeError: __init__() missing 1 required positional argument: 'email'
            """
        ),
        "details": "'email' field added to UserProfile but test not updated.",
    },
    {
        "file": "tests/test_views.py",
        "line": 12,
        "test_name": "test_home_view_status_code",
        "message": "AssertionError: assert 404 == 200",
        "stack_trace": textwrap.dedent(
            """\
            Traceback (most recent call last):
              File \"tests/test_views.py\", line 12, in test_home_view_status_code
                assert response.status_code == 200
            AssertionError: assert 404 == 200
            """
        ),
        "details": "Home URL pattern changed; test client hits old route.",
    },
]

_LINT_ERROR_TEMPLATES: list[dict[str, Any]] = [
    {
        "file": "src/app/helpers.py",
        "line": 7,
        "message": "E501 line too long (102 > 88 characters)",
        "snippet": "    result = some_function_with_very_long_name(argument_one, argument_two, argument_three, argument_four)",
        "details": "Flake8 E501: line exceeds maximum length.",
    },
    {
        "file": "src/app/utils.py",
        "line": 1,
        "message": "F401 'os' imported but unused",
        "snippet": "import os",
        "details": "Flake8 F401: unused import should be removed.",
    },
]

_FAILURE_CATEGORIES: list[str] = [
    "syntax_error",
    "import_error",
    "test_failure",
    "lint_error",
]

_TEMPLATES_BY_CATEGORY: dict[str, list[dict[str, Any]]] = {
    "syntax_error": _SYNTAX_ERROR_TEMPLATES,
    "import_error": _IMPORT_ERROR_TEMPLATES,
    "test_failure": _TEST_FAILURE_TEMPLATES,
    "lint_error": _LINT_ERROR_TEMPLATES,
}


# ---------------------------------------------------------------------------
# Log text builders
# ---------------------------------------------------------------------------

def _build_log_text(record: dict[str, Any]) -> str:
    """Render a human-readable CI log string from a structured record."""
    category = record["category"]
    meta = record["metadata"]
    ts = record["timestamp"]
    run_id = record["run_id"]

    lines: list[str] = [
        f"=== CI Build Log ===",
        f"Run ID  : {run_id}",
        f"Started : {ts}",
        f"Category: {category}",
        "",
        "--- Pipeline Steps ---",
        "[PASS] checkout",
        "[PASS] install-dependencies",
    ]

    if category == "lint_error":
        lines += [
            "[FAIL] lint",
            "",
            "--- Lint Output ---",
            f"{meta['file']}:{meta['line']}: {meta['message']}",
            f"  {meta.get('snippet', '')}",
            "",
            f"Details: {meta['details']}",
        ]
    elif category == "syntax_error":
        lines += [
            "[FAIL] build",
            "",
            "--- Build Output ---",
            f"  File \"{meta['file']}\", line {meta['line']}",
            f"  {meta.get('snippet', '')}",
            f"{meta['message']}",
            "",
            f"Details: {meta['details']}",
        ]
    elif category == "import_error":
        lines += [
            "[FAIL] build",
            "",
            "--- Build Output ---",
            f"  File \"{meta['file']}\", line {meta['line']}",
            f"    {meta.get('snippet', '')}",
            f"{meta['message']}",
            "",
            f"Details: {meta['details']}",
        ]
    elif category == "test_failure":
        lines += [
            "[PASS] build",
            "[FAIL] test",
            "",
            "--- Test Output ---",
            f"FAILED {meta['file']}::{meta['test_name']}",
            "",
            meta.get("stack_trace", ""),
            f"{meta['message']}",
            "",
            f"Details: {meta['details']}",
        ]

    lines += [
        "",
        "=== Build FAILED ===",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class CIPipelineSimulator:
    """Generates mock failing CI pipeline logs and persists them to disk."""

    def __init__(self, output_dir: str = "ci_logs", seed: int | None = None) -> None:
        self.output_dir = Path(output_dir)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, count: int = 3) -> list[Path]:
        """Generate *count* failing CI log pairs (JSON + .txt) and return their paths."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for _ in range(count):
            record = self._build_record()
            json_path, txt_path = self._persist(record)
            paths.extend([json_path, txt_path])
        return paths

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_category(self) -> str:
        return self._rng.choice(_FAILURE_CATEGORIES)

    def _build_record(self) -> dict[str, Any]:
        category = self._pick_category()
        templates = _TEMPLATES_BY_CATEGORY[category]
        metadata = dict(self._rng.choice(templates))  # shallow copy

        run_id = str(uuid.UUID(int=self._rng.getrandbits(128)))
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        return {
            "schema_version": "1.0",
            "run_id": run_id,
            "timestamp": timestamp,
            "category": category,
            "status": "failed",
            "metadata": metadata,
        }

    def _persist(self, record: dict[str, Any]) -> tuple[Path, Path]:
        run_id_short = record["run_id"].split("-")[0]
        category = record["category"]
        base_name = f"{category}_{run_id_short}"

        json_path = self.output_dir / f"{base_name}.json"
        txt_path = self.output_dir / f"{base_name}.txt"

        json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        txt_path.write_text(_build_log_text(record), encoding="utf-8")

        return json_path, txt_path
