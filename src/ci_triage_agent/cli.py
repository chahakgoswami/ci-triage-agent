"""CLI entry point for the CI Triage Agent."""

import click
from rich.console import Console

from ci_triage_agent.agent import FixSuggestionAgent
from ci_triage_agent.parser import ingest_directory
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


if __name__ == "__main__":
    main()
