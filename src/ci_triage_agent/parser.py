"""Log ingestion and parsing layer.

Reads mock CI log files (JSON and/or plain-text), classifies failure
types, and extracts structured error metadata into dataclass models.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class FailureType(str, Enum):
    """High-level classification of a CI failure."""

    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    TEST_FAILURE = "test_failure"
    LINT_ERROR = "lint_error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ParsedFailure:
    """Structured representation of a parsed CI failure."""

    # Source identifiers
    run_id: str
    source_file: str  # path to the log file that was parsed

    # Classification
    failure_type: FailureType

    # Core error location
    error_file: str  # source file mentioned in the error
    error_line: int
    error_message: str

    # Optional rich context
    test_name: Optional[str] = None
    stack_trace: Optional[str] = None
    snippet: Optional[str] = None
    details: Optional[str] = None

    # Raw metadata dict preserved for downstream use
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    # Timestamp from the log record
    timestamp: str = ""

    def __str__(self) -> str:  # pragma: no cover
        return (
            f"ParsedFailure(run_id={self.run_id!r}, "
            f"failure_type={self.failure_type.value!r}, "
            f"error_file={self.error_file!r}, "
            f"error_line={self.error_line}, "
            f"error_message={self.error_message!r})"
        )


# ---------------------------------------------------------------------------
# Classifier helpers
# ---------------------------------------------------------------------------

# Regexes used for plain-text log classification (fallback path)
_LINT_PATTERN = re.compile(r"\[FAIL\]\s+lint")
_BUILD_FAIL_PATTERN = re.compile(r"\[FAIL\]\s+build")
_TEST_FAIL_PATTERN = re.compile(r"\[FAIL\]\s+test")
_SYNTAX_PATTERN = re.compile(r"SyntaxError")
_IMPORT_PATTERN = re.compile(r"(?:ModuleNotFoundError|ImportError)")

# Plain-text extraction patterns
_FILE_LINE_PATTERN = re.compile(
    r'File\s+"(?P<file>[^"]+)",\s+line\s+(?P<line>\d+)'
)
_LINT_FILE_LINE_PATTERN = re.compile(
    r"(?P<file>[\w./\\-]+):(?P<line>\d+):\s+(?P<msg>.+)"
)
_TEST_FAILED_PATTERN = re.compile(
    r"FAILED\s+(?P<file>[\w./\\-]+)::(?P<test>[\w_]+)"
)
_ERROR_MSG_PATTERN = re.compile(
    r"(?P<msg>(?:SyntaxError|ModuleNotFoundError|ImportError|AssertionError|TypeError)[^\n]+)"
)


def _classify_from_category(category: str) -> FailureType:
    """Map a JSON 'category' string to a FailureType."""
    mapping: dict[str, FailureType] = {
        "syntax_error": FailureType.SYNTAX_ERROR,
        "import_error": FailureType.IMPORT_ERROR,
        "test_failure": FailureType.TEST_FAILURE,
        "lint_error": FailureType.LINT_ERROR,
    }
    return mapping.get(category, FailureType.UNKNOWN)


def _classify_from_text(text: str) -> FailureType:
    """Infer FailureType from plain-text log content."""
    if _LINT_PATTERN.search(text):
        return FailureType.LINT_ERROR
    if _TEST_FAIL_PATTERN.search(text):
        return FailureType.TEST_FAILURE
    if _BUILD_FAIL_PATTERN.search(text):
        # Distinguish syntax vs import within build failures
        if _IMPORT_PATTERN.search(text):
            return FailureType.IMPORT_ERROR
        if _SYNTAX_PATTERN.search(text):
            return FailureType.SYNTAX_ERROR
        return FailureType.SYNTAX_ERROR  # default build failure
    return FailureType.UNKNOWN


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_from_json(path: Path) -> ParsedFailure:
    """Parse a structured JSON log file produced by the simulator."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    meta: dict[str, Any] = data.get("metadata", {})

    failure_type = _classify_from_category(data.get("category", ""))

    return ParsedFailure(
        run_id=data.get("run_id", ""),
        source_file=str(path),
        failure_type=failure_type,
        error_file=meta.get("file", ""),
        error_line=int(meta.get("line", 0)),
        error_message=meta.get("message", ""),
        test_name=meta.get("test_name"),
        stack_trace=meta.get("stack_trace"),
        snippet=meta.get("snippet"),
        details=meta.get("details"),
        raw_metadata=meta,
        timestamp=data.get("timestamp", ""),
    )


def _parse_from_text(path: Path) -> ParsedFailure:
    """Best-effort parse of a plain-text CI log file."""
    text = path.read_text(encoding="utf-8")
    failure_type = _classify_from_text(text)

    # Extract run_id (appears after "Run ID  :")
    run_id = ""
    run_id_match = re.search(r"Run ID\s*:\s*(\S+)", text)
    if run_id_match:
        run_id = run_id_match.group(1)

    # Extract timestamp
    timestamp = ""
    ts_match = re.search(r"Started\s*:\s*(\S+)", text)
    if ts_match:
        timestamp = ts_match.group(1)

    error_file = ""
    error_line = 0
    error_message = ""
    test_name: Optional[str] = None
    stack_trace: Optional[str] = None
    snippet: Optional[str] = None

    if failure_type == FailureType.LINT_ERROR:
        m = _LINT_FILE_LINE_PATTERN.search(text)
        if m:
            error_file = m.group("file")
            error_line = int(m.group("line"))
            error_message = m.group("msg").strip()

    elif failure_type == FailureType.TEST_FAILURE:
        m = _TEST_FAILED_PATTERN.search(text)
        if m:
            error_file = m.group("file")
            test_name = m.group("test")
        # Grab stack trace block
        trace_m = re.search(
            r"(Traceback \(most recent call last\):.*?)(?=\n\n|\Z)",
            text,
            re.DOTALL,
        )
        if trace_m:
            stack_trace = trace_m.group(1)
        msg_m = _ERROR_MSG_PATTERN.search(text)
        if msg_m:
            error_message = msg_m.group("msg").strip()
        # Extract line number from stack trace
        if stack_trace:
            line_m = re.search(r", line (\d+),", stack_trace)
            if line_m:
                error_line = int(line_m.group(1))

    else:  # SYNTAX_ERROR, IMPORT_ERROR, UNKNOWN
        m = _FILE_LINE_PATTERN.search(text)
        if m:
            error_file = m.group("file")
            error_line = int(m.group("line"))
        msg_m = _ERROR_MSG_PATTERN.search(text)
        if msg_m:
            error_message = msg_m.group("msg").strip()

    return ParsedFailure(
        run_id=run_id,
        source_file=str(path),
        failure_type=failure_type,
        error_file=error_file,
        error_line=error_line,
        error_message=error_message,
        test_name=test_name,
        stack_trace=stack_trace,
        snippet=snippet,
        details=None,
        raw_metadata={},
        timestamp=timestamp,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_log_file(path: str | Path) -> ParsedFailure:
    """Parse a single CI log file (JSON or plain-text) into a ParsedFailure.

    JSON files produced by the simulator are parsed with full fidelity.
    Plain-text ``.txt`` files are parsed with best-effort regex extraction.

    Parameters
    ----------
    path:
        Path to the log file to parse.

    Returns
    -------
    ParsedFailure
        Structured failure metadata.

    Raises
    ------
    ValueError
        If the file extension is not ``.json`` or ``.txt``.
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    if path.suffix == ".json":
        return _parse_from_json(path)
    elif path.suffix == ".txt":
        return _parse_from_text(path)
    else:
        raise ValueError(
            f"Unsupported log file extension '{path.suffix}'. "
            "Expected '.json' or '.txt'."
        )


def ingest_directory(directory: str | Path) -> list[ParsedFailure]:
    """Ingest all CI log files from *directory*.

    Prefers JSON files when both ``.json`` and ``.txt`` files share the
    same stem (i.e. simulator output pairs); processes lone ``.txt``
    files independently.

    Parameters
    ----------
    directory:
        Directory containing CI log files.

    Returns
    -------
    list[ParsedFailure]
        One ParsedFailure per discovered log, sorted by filename.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    json_stems: set[str] = {p.stem for p in directory.glob("*.json")}
    results: list[ParsedFailure] = []

    # Collect files: JSON first, then .txt files without a sibling .json
    candidates: list[Path] = sorted(directory.glob("*.json"))
    for txt in sorted(directory.glob("*.txt")):
        if txt.stem not in json_stems:
            candidates.append(txt)

    for log_path in candidates:
        results.append(parse_log_file(log_path))

    return results
