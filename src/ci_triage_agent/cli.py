"""CLI entry point for the CI Triage Agent."""

import json
from pathlib import Path

import click
from rich.console import Console

from ci_triage_agent.agent import FixSuggestionAgent
from ci_triage_agent.approval import (
    AUDIT_LOG_FILENAME,
    load_audit_trail,
    run_approval_workflow,
)
from ci_triage_agent.git_pr import MockPullRequest, TestRunResult
from ci_triage_agent.parser import FailureType, ParsedFailure, ingest_directory
from ci_triage_agent.reproducer import reproduce_failure
from ci_triage_agent.simulator import CIPipelineSimulator

console = Console()


@click.group()
@click.version_option()
def main() -> None:
    """CI Triage Agent – agentic CI triage bot."""


@main.command("simulate")
@click.option(
    "--output-dir",
    default="ci_logs",
    show_default=True,
    help="Directory where mock CI log files are written.",
)
@click.option(
    "--count",
    default=3,
    show_default=True,
    type=int,
    help="Number of failing build logs to generate.",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for reproducible log generation.",
)
def simulate(output_dir: str, count: int, seed: int | None) -> None:
    """Generate mock failing CI build logs."""
    simulator = CIPipelineSimulator(output_dir=output_dir, seed=seed)
    paths = simulator.generate(count=count)
    console.print(f"[green]Generated {len(paths)} CI log(s) in '[bold]{output_dir}[/bold]':[/green]")
    for p in paths:
        console.print(f"  \u2022 {p}")


@main.command("ingest")
@click.option(
    "--log-dir",
    default="ci_logs",
    show_default=True,
    help="Directory containing CI log files to ingest.",
)
def ingest(log_dir: str) -> None:
    """Ingest and parse CI log files, printing structured failure summaries."""
    failures = ingest_directory(log_dir)
    if not failures:
        console.print("[yellow]No CI log files found.[/yellow]")
        return
    console.print(f"[green]Parsed {len(failures)} failure(s):[/green]")
    for f in failures:
        console.print(
            f"  [bold]{f.failure_type.value}[/bold] "
            f"| run=[cyan]{f.run_id}[/cyan] "
            f"| file=[magenta]{f.error_file}:{f.error_line}[/magenta] "
            f"| {f.error_message}"
        )


@main.command("reproduce")
@click.option(
    "--log-dir",
    default="ci_logs",
    show_default=True,
    help="Directory containing CI log files to ingest and reproduce.",
)
@click.option(
    "--timeout",
    default=30,
    show_default=True,
    type=int,
    help="Subprocess timeout in seconds.",
)
def reproduce(log_dir: str, timeout: int) -> None:
    """Ingest CI log files and attempt to reproduce each failure locally."""
    failures = ingest_directory(log_dir)
    if not failures:
        console.print("[yellow]No CI log files found.[/yellow]")
        return
    console.print(f"[green]Reproducing {len(failures)} failure(s):[/green]")
    for f in failures:
        console.print(
            f"\n  [bold]{f.failure_type.value}[/bold] "
            f"| run=[cyan]{f.run_id}[/cyan] "
            f"| file=[magenta]{f.error_file}:{f.error_line}[/magenta]"
        )
        result = reproduce_failure(f, timeout=timeout)
        status = "[green]CONFIRMED[/green]" if result.confirmed else "[red]NOT CONFIRMED[/red]"
        console.print(f"    Reproduction: {status}")
        console.print(f"    Reason      : {result.confirmation_reason}")
        if result.reproduction_error:
            console.print(f"    [red]Error: {result.reproduction_error}[/red]")


@main.command("suggest")
@click.option(
    "--log-dir",
    default="ci_logs",
    show_default=True,
    help="Directory containing CI log files to ingest and fix.",
)
@click.option(
    "--timeout",
    default=30,
    show_default=True,
    type=int,
    help="Subprocess timeout in seconds.",
)
@click.option(
    "--model",
    default="gpt-4o",
    show_default=True,
    help="OpenAI model to use (ignored when OPENAI_API_KEY is absent).",
)
def suggest(log_dir: str, timeout: int, model: str) -> None:
    """Ingest CI logs, reproduce failures, and suggest fixes via the LLM agent."""
    failures = ingest_directory(log_dir)
    if not failures:
        console.print("[yellow]No CI log files found.[/yellow]")
        return

    agent = FixSuggestionAgent(model=model)
    mode = "[yellow]mock[/yellow]" if agent.used_mock else f"[green]{agent.model_name}[/green]"
    console.print(f"[green]LLM agent mode:[/green] {mode}")
    console.print(f"[green]Processing {len(failures)} failure(s)...[/green]")

    for f in failures:
        console.print(
            f"\n  [bold]{f.failure_type.value}[/bold] "
            f"| run=[cyan]{f.run_id}[/cyan] "
            f"| file=[magenta]{f.error_file}:{f.error_line}[/magenta]"
        )
        repro = reproduce_failure(f, timeout=timeout)
        repro_status = "[green]CONFIRMED[/green]" if repro.confirmed else "[red]NOT CONFIRMED[/red]"
        console.print(f"    Reproduction : {repro_status}")

        suggestion = agent.suggest_fix(f, repro)
        if suggestion.agent_error:
            console.print(f"    [red]Agent error: {suggestion.agent_error}[/red]")
            continue

        console.print(f"    Explanation  : {suggestion.explanation[:200]}")
        if suggestion.has_patch:
            console.print("    [green]Patch proposed:[/green]")
            console.print(suggestion.patch)
        else:
            console.print("    [yellow]No patch proposed.[/yellow]")


@main.command("approve")
@click.argument(
    "pr_artifact",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    metavar="PR_ARTIFACT_JSON",
)
@click.option(
    "--output-dir",
    default="approved_patches",
    show_default=True,
    help="Directory where approved patch files are written.",
)
@click.option(
    "--audit-log",
    default=AUDIT_LOG_FILENAME,
    show_default=True,
    help="Path to the JSON-Lines audit trail file.",
)
def approve_cmd(
    pr_artifact: str,
    output_dir: str,
    audit_log: str,
) -> None:
    """Review and approve/reject a PR artifact JSON file.

    PR_ARTIFACT_JSON is the path to a JSON file produced by the git-pr layer.
    """
    artifact_path = Path(pr_artifact)
    data = json.loads(artifact_path.read_text(encoding="utf-8"))

    # Reconstruct a MockPullRequest from the JSON artifact.
    # We build lightweight stubs from the serialised data so that
    # the approval workflow can present all relevant details.
    failure = ParsedFailure(
        run_id=data["parsed_failure"]["run_id"],
        source_file=data["parsed_failure"].get("source_file", ""),
        failure_type=FailureType(data["parsed_failure"]["failure_type"]),
        error_file=data["parsed_failure"]["error_file"],
        error_line=data["parsed_failure"]["error_line"],
        error_message=data["parsed_failure"]["error_message"],
        test_name=data["parsed_failure"].get("test_name"),
    )

    from ci_triage_agent.agent import FixSuggestion
    fs_data = data.get("fix_suggestion", {})
    suggestion = FixSuggestion(
        parsed_failure=failure,
        used_mock=fs_data.get("used_mock", True),
        explanation=fs_data.get("explanation", ""),
        patch=fs_data.get("patch", ""),
        target_file=fs_data.get("target_file", ""),
        model=fs_data.get("model", "mock"),
    )

    tr_data = data.get("test_result", {})
    test_result = TestRunResult(
        passed=tr_data.get("passed", False),
        returncode=tr_data.get("returncode", -1),
        stdout=tr_data.get("stdout", ""),
        stderr=tr_data.get("stderr", ""),
        tests_passed=tr_data.get("tests_passed", 0),
        tests_failed=tr_data.get("tests_failed", 0),
        tests_collected=tr_data.get("tests_collected", 0),
    )

    pr = MockPullRequest(
        pr_id=data["pr_id"],
        title=data["title"],
        branch_name=data["branch_name"],
        diff=data["diff"],
        changed_files=data.get("changed_files", []),
        test_result=test_result,
        parsed_failure=failure,
        fix_suggestion=suggestion,
        patch_applied=data.get("patch_applied", False),
        error=data.get("error"),
        labels=data.get("labels", []),
        created_at=data.get("created_at", ""),
        artifact_path=str(artifact_path),
    )

    decision = run_approval_workflow(
        pr,
        output_dir=output_dir,
        audit_log=audit_log,
        console=console,
    )

    if decision.decision == "approve":
        console.print(
            f"[bold green]Approved.[/bold green] "
            f"Patch written to: {decision.patch_written_to}"
        )
    else:
        console.print(f"[bold red]Rejected.[/bold red] Note: {decision.note or '(none)'}")


@main.command("run")
@click.option("--log-dir", default="ci_logs", show_default=True,
              help="Directory containing CI log files to triage.")
@click.option("--pr-output-dir", default="pr_artifacts", show_default=True)
@click.option("--approved-dir", default="approved_patches", show_default=True)
@click.option("--audit-log", default=AUDIT_LOG_FILENAME, show_default=True)
@click.option("--model", default="gpt-4o", show_default=True)
@click.option("--force-mock", is_flag=True, help="Always use the stub LLM client.")
@click.option("--auto-approve", is_flag=True,
              help="Approve every proposed PR without prompting (non-interactive).")
@click.option("--simulate", default=0, type=int, metavar="N",
              help="First generate N mock failing logs into --log-dir.")
@click.option("--seed", default=None, type=int, help="Seed for --simulate.")
def run_cmd(
    log_dir: str,
    pr_output_dir: str,
    approved_dir: str,
    audit_log: str,
    model: str,
    force_mock: bool,
    auto_approve: bool,
    simulate: int,
    seed: int | None,
) -> None:
    """Run the full end-to-end triage pipeline (ingest → reproduce → fix → PR → approve)."""
    from ci_triage_agent.orchestrator import OrchestratorConfig, run_pipeline

    if simulate:
        CIPipelineSimulator(output_dir=log_dir, seed=seed).generate(count=simulate)

    config = OrchestratorConfig(
        log_dir=log_dir,
        pr_output_dir=pr_output_dir,
        approved_dir=approved_dir,
        audit_log=audit_log,
        model=model,
        force_mock=force_mock,
        auto_approve=auto_approve,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
