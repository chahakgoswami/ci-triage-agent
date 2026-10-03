"""Human-in-the-loop approval workflow.

Prints a rich summary of a proposed MockPullRequest, prompts the user
for an explicit decision (approve / reject / edit), writes the final
patch to disk on approval, and logs every decision to an audit trail.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ci_triage_agent.git_pr import MockPullRequest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIT_LOG_FILENAME = "audit_trail.jsonl"

# Valid command tokens entered by the user
_CMD_APPROVE = "approve"
_CMD_REJECT = "reject"
_CMD_EDIT = "edit"
_VALID_COMMANDS = {_CMD_APPROVE, _CMD_REJECT, _CMD_EDIT}


# ---------------------------------------------------------------------------
# Decision model
# ---------------------------------------------------------------------------

@dataclass
class ApprovalDecision:
    """Records the outcome of a single human review."""

    pr_id: str
    decision: str  # 'approve' | 'reject' | 'edit'
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    patch_written_to: str = ""  # set when decision == 'approve'
    edited_patch: str = ""      # set when decision == 'edit'
    note: str = ""              # optional human comment

    def to_dict(self) -> dict:
        return {
            "pr_id": self.pr_id,
            "decision": self.decision,
            "timestamp": self.timestamp,
            "patch_written_to": self.patch_written_to,
            "edited_patch": self.edited_patch,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

def _append_audit_entry(
    audit_log_path: Path,
    decision: ApprovalDecision,
) -> None:
    """Append a JSON-lines entry to the audit trail."""
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(decision.to_dict()) + "\n")


def load_audit_trail(audit_log_path: str | Path) -> list[dict]:
    """Load all entries from an audit trail file.

    Returns an empty list if the file does not exist.
    """
    path = Path(audit_log_path)
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Rich summary printer
# ---------------------------------------------------------------------------

def print_pr_summary(
    pr: MockPullRequest,
    console: Optional[Console] = None,
) -> None:
    """Print a rich, human-readable summary of *pr* to the console."""
    if console is None:
        console = Console()

    failure = pr.parsed_failure
    suggestion = pr.fix_suggestion
    test = pr.test_result

    # ── Header panel ──────────────────────────────────────────────────────
    status_colour = {
        "tests_pass": "green",
        "tests_fail": "yellow",
        "patch_failed": "red",
        "error": "red",
    }.get(pr.status, "white")
    header = Text.from_markup(
        f"[bold]PR ID  :[/bold] [cyan]{pr.pr_id}[/cyan]\n"
        f"[bold]Title  :[/bold] {pr.title}\n"
        f"[bold]Branch :[/bold] [magenta]{pr.branch_name}[/magenta]\n"
        f"[bold]Status :[/bold] [{status_colour}]{pr.status}[/{status_colour}]"
    )
    console.print(Panel(header, title="[bold white]Proposed Pull Request[/bold white]", expand=False))

    # ── Failure details ───────────────────────────────────────────────────
    fail_table = Table(show_header=False, box=None, padding=(0, 1))
    fail_table.add_column(style="bold", min_width=18)
    fail_table.add_column()
    fail_table.add_row("Failure type", failure.failure_type.value)
    fail_table.add_row("Run ID", failure.run_id)
    fail_table.add_row("Error file", f"{failure.error_file}:{failure.error_line}")
    fail_table.add_row("Error message", failure.error_message[:120])
    if failure.test_name:
        fail_table.add_row("Test name", failure.test_name)
    console.print(Panel(fail_table, title="[bold]Failure Details[/bold]", expand=False))

    # ── Agent explanation ─────────────────────────────────────────────────
    model_tag = "[yellow]mock[/yellow]" if suggestion.used_mock else f"[green]{suggestion.model}[/green]"
    exp_text = suggestion.explanation or "(none)"
    console.print(
        Panel(
            f"[bold]Model:[/bold] {model_tag}\n\n{exp_text}",
            title="[bold]Agent Explanation[/bold]",
            expand=False,
        )
    )

    # ── Diff ──────────────────────────────────────────────────────────────
    if pr.diff.strip():
        syntax = Syntax(
            pr.diff,
            "diff",
            theme="ansi_dark",
            line_numbers=False,
        )
        console.print(Panel(syntax, title="[bold]Proposed Patch[/bold]", expand=False))
    else:
        console.print(Panel("[yellow]No patch provided.[/yellow]", title="[bold]Proposed Patch[/bold]", expand=False))

    # ── Test results ──────────────────────────────────────────────────────
    test_colour = "green" if test.passed else "red"
    test_summary = (
        f"[{test_colour}]{'PASS' if test.passed else 'FAIL'}[/{test_colour}] "
        f"(exit {test.returncode})  "
        f"{test.tests_passed} passed / {test.tests_failed} failed"
    )
    if test.combined_output:
        trimmed = test.combined_output[:800]
        test_body = test_summary + "\n\n" + trimmed
    else:
        test_body = test_summary
    console.print(Panel(test_body, title="[bold]Test Results[/bold]", expand=False))

    # ── Changed files ─────────────────────────────────────────────────────
    if pr.changed_files:
        files_text = "\n".join(f"  • {f}" for f in pr.changed_files)
        console.print(Panel(files_text, title="[bold]Changed Files[/bold]", expand=False))


# ---------------------------------------------------------------------------
# Patch writer
# ---------------------------------------------------------------------------

def _write_patch(
    pr: MockPullRequest,
    output_dir: Path,
    patch_text: str | None = None,
) -> Path:
    """Write the (optionally edited) patch to *output_dir* and return the path."""
    patch_text = patch_text if patch_text is not None else pr.diff
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"patch_{pr.pr_id[:8]}.diff"
    dest.write_text(patch_text, encoding="utf-8")
    return dest


# ---------------------------------------------------------------------------
# Core approval loop
# ---------------------------------------------------------------------------

def run_approval_workflow(
    pr: MockPullRequest,
    *,
    output_dir: str | Path = "approved_patches",
    audit_log: str | Path = AUDIT_LOG_FILENAME,
    console: Optional[Console] = None,
    # Injection point for tests: replace with a callable that returns a string.
    _input_fn: Callable[[str], str] = input,
) -> ApprovalDecision:
    """Interactive approval workflow for a proposed pull request.

    Prints a rich summary of *pr*, then enters a prompt loop accepting:

    * ``approve`` – write the patch to disk and mark as merged.
    * ``reject``  – discard the patch and record the rejection.
    * ``edit``    – open a simple line-editor for the patch text, then
                    re-prompt for approve/reject.

    Every decision is appended to *audit_log* (JSON-Lines).

    Parameters
    ----------
    pr:
        The MockPullRequest produced by the Git/PR layer.
    output_dir:
        Directory where approved patches are written.
    audit_log:
        Path to the JSON-Lines audit trail file.
    console:
        Rich Console instance (created if not supplied).
    _input_fn:
        Callable used to read user input; replaced by tests.

    Returns
    -------
    ApprovalDecision
        The decision (approve / reject / edit-then-approve/reject).
    """
    if console is None:
        console = Console()

    output_dir = Path(output_dir)
    audit_log = Path(audit_log)

    # 1. Print summary
    print_pr_summary(pr, console=console)

    # 2. Prompt loop
    current_patch = pr.diff
    decision: Optional[ApprovalDecision] = None

    while decision is None:
        console.print(
            "\n[bold]Commands:[/bold] "
            "[green]approve[/green] | [red]reject[/red] | [yellow]edit[/yellow]  "
            "(type your choice and press Enter)"
        )
        raw = _input_fn("Decision> ").strip().lower()

        if raw not in _VALID_COMMANDS:
            console.print(
                f"[red]Unknown command '{raw}'. "
                f"Please enter one of: {', '.join(sorted(_VALID_COMMANDS))}[/red]"
            )
            continue

        if raw == _CMD_APPROVE:
            dest = _write_patch(pr, output_dir, current_patch)
            console.print(f"[green]✓ Patch approved and written to:[/green] {dest}")
            decision = ApprovalDecision(
                pr_id=pr.pr_id,
                decision="approve",
                patch_written_to=str(dest),
            )

        elif raw == _CMD_REJECT:
            note_raw = _input_fn("Optional rejection note (press Enter to skip): ").strip()
            console.print("[red]✗ Pull request rejected.[/red]")
            decision = ApprovalDecision(
                pr_id=pr.pr_id,
                decision="reject",
                note=note_raw,
            )

        elif raw == _CMD_EDIT:
            console.print(
                "[yellow]Enter your edited patch below.[/yellow]  "
                "Type [bold]END[/bold] on its own line when done:"
            )
            lines: list[str] = []
            while True:
                line = _input_fn("")
                if line == "END":
                    break
                lines.append(line)
            current_patch = "\n".join(lines)
            console.print("[yellow]Patch updated. Re-enter a command.[/yellow]")
            # Show the updated patch
            if current_patch.strip():
                syntax = Syntax(current_patch, "diff", theme="ansi_dark")
                console.print(Panel(syntax, title="[bold]Edited Patch[/bold]", expand=False))

    # 3. Append to audit trail
    _append_audit_entry(audit_log, decision)
    console.print(
        f"[dim]Decision logged to audit trail:[/dim] {audit_log}"
    )

    return decision
