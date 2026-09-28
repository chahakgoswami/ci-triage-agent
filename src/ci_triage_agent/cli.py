"""CLI entry point for the CI Triage Agent."""

import click
from rich.console import Console

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
        console.print(f"  • {p}")


if __name__ == "__main__":
    main()
