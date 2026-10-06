"""End-to-end orchestrator that wires the CI Triage Agent pipeline together.

Pipeline, per discovered failure:

    ingest  ->  reproduce  ->  suggest fix  ->  open mock PR  ->  human approval

The orchestrator is configurable (paths, timeouts, model, auto-approve for
non-interactive runs) and resilient: each failure is processed independently and
any exception is captured as a per-item error rather than aborting the whole run.

Run it as a module:

    python -m ci_triage_agent.orchestrator --log-dir ci_logs --auto-approve
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rich.console import Console

from ci_triage_agent.agent import FixSuggestion, FixSuggestionAgent
from ci_triage_agent.approval import ApprovalDecision, run_approval_workflow
from ci_triage_agent.git_pr import GitPRLayer, MockPullRequest
from ci_triage_agent.parser import ParsedFailure, ingest_directory
from ci_triage_agent.reproducer import ReproductionResult, reproduce_failure
from ci_triage_agent.simulator import CIPipelineSimulator


# Per-failure outcome statuses.
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_UNCONFIRMED = "unconfirmed"   # reproduction did not confirm the failure
STATUS_UNFIXABLE = "unfixable"       # agent produced no patch
STATUS_AGENT_ERROR = "agent_error"   # agent raised / returned an error
STATUS_ERROR = "error"               # unexpected exception in the pipeline


@dataclass
class OrchestratorConfig:
    """Tunable settings for an end-to-end run."""

    log_dir: str = "ci_logs"
    pr_output_dir: str = "pr_artifacts"
    approved_dir: str = "approved_patches"
    audit_log: str = "audit_trail.jsonl"

    reproduce_timeout: int = 30
    test_timeout: int = 60

    model: str = "gpt-4o"
    force_mock: bool = False

    # When True, every proposed PR is approved automatically (no prompt) — useful
    # for non-interactive / scheduled runs. When False, the interactive approval
    # workflow is used (see `input_fn`).
    auto_approve: bool = False

    # When reproduction cannot confirm the failure, skip the fix step if True.
    skip_unconfirmed: bool = True

    # Input callable for the interactive approval workflow (injectable for tests).
    input_fn: Callable[[str], str] = input


@dataclass
class PipelineItemResult:
    """Outcome of processing a single failure through the pipeline."""

    failure: ParsedFailure
    status: str
    reproduction: Optional[ReproductionResult] = None
    suggestion: Optional[FixSuggestion] = None
    pull_request: Optional[MockPullRequest] = None
    decision: Optional[ApprovalDecision] = None
    message: str = ""


@dataclass
class PipelineRunResult:
    """Aggregate result of a full orchestrator run."""

    items: list[PipelineItemResult] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for item in self.items:
            out[item.status] = out.get(item.status, 0) + 1
        return out


class Orchestrator:
    """Coordinates the full triage pipeline over a directory of CI logs.

    Components can be injected (agent / pr_layer) for testing; by default the
    real implementations are used.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        *,
        console: Optional[Console] = None,
        agent: Optional[FixSuggestionAgent] = None,
        pr_layer: Optional[GitPRLayer] = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.console = console or Console()
        self.agent = agent or FixSuggestionAgent(
            model=self.config.model, force_mock=self.config.force_mock
        )
        self.pr_layer = pr_layer or GitPRLayer(
            output_dir=self.config.pr_output_dir,
            test_timeout=self.config.test_timeout,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> PipelineRunResult:
        """Process every failure found under ``config.log_dir``."""
        failures = ingest_directory(self.config.log_dir)
        result = PipelineRunResult()
        if not failures:
            self.console.print("[yellow]No CI failures found to triage.[/yellow]")
            return result

        self.console.print(
            f"[bold]Triaging {len(failures)} failure(s) "
            f"(LLM mock: {self.agent.used_mock})[/bold]"
        )
        for failure in failures:
            result.items.append(self._process_one(failure))
        self._print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _process_one(self, failure: ParsedFailure) -> PipelineItemResult:
        try:
            # 1. Reproduce
            reproduction = reproduce_failure(
                failure, timeout=self.config.reproduce_timeout
            )
            if self.config.skip_unconfirmed and not reproduction.confirmed:
                return PipelineItemResult(
                    failure=failure,
                    status=STATUS_UNCONFIRMED,
                    reproduction=reproduction,
                    message=reproduction.confirmation_reason,
                )

            # 2. Suggest a fix
            suggestion = self.agent.suggest_fix(failure, reproduction)
            if suggestion.agent_error:
                return PipelineItemResult(
                    failure=failure,
                    status=STATUS_AGENT_ERROR,
                    reproduction=reproduction,
                    suggestion=suggestion,
                    message=suggestion.agent_error,
                )
            if not suggestion.has_patch:
                return PipelineItemResult(
                    failure=failure,
                    status=STATUS_UNFIXABLE,
                    reproduction=reproduction,
                    suggestion=suggestion,
                    message="Agent produced no patch for this failure.",
                )

            # 3. Open a mock PR (apply patch + run tests)
            pr = self.pr_layer.create_pull_request(suggestion)

            # 4. Human (or auto) approval
            decision = self._approve(pr)
            status = (
                STATUS_APPROVED if decision.decision == "approve" else STATUS_REJECTED
            )
            return PipelineItemResult(
                failure=failure,
                status=status,
                reproduction=reproduction,
                suggestion=suggestion,
                pull_request=pr,
                decision=decision,
                message=f"PR {pr.pr_id} -> {decision.decision}",
            )
        except Exception as exc:  # noqa: BLE001 - keep the loop alive per item
            return PipelineItemResult(
                failure=failure,
                status=STATUS_ERROR,
                message=str(exc),
            )

    def _approve(self, pr: MockPullRequest) -> ApprovalDecision:
        # Non-interactive auto-approve feeds a canned "approve" to the workflow,
        # so the same audit-trail / patch-writing path is exercised either way.
        input_fn = (lambda _prompt: "approve") if self.config.auto_approve else self.config.input_fn
        return run_approval_workflow(
            pr,
            output_dir=self.config.approved_dir,
            audit_log=self.config.audit_log,
            console=self.console,
            _input_fn=input_fn,
        )

    def _print_summary(self, result: PipelineRunResult) -> None:
        self.console.print("\n[bold]Triage summary[/bold]")
        for status, n in sorted(result.counts.items()):
            self.console.print(f"  {status:12s}: {n}")


# ---------------------------------------------------------------------------
# Module-level convenience + CLI
# ---------------------------------------------------------------------------

def run_pipeline(config: Optional[OrchestratorConfig] = None) -> PipelineRunResult:
    """Convenience wrapper: build an Orchestrator and run it."""
    return Orchestrator(config=config).run()


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the CI Triage Agent end to end.")
    p.add_argument("--log-dir", default="ci_logs", help="Directory of CI log files.")
    p.add_argument("--pr-output-dir", default="pr_artifacts")
    p.add_argument("--approved-dir", default="approved_patches")
    p.add_argument("--audit-log", default="audit_trail.jsonl")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--force-mock", action="store_true", help="Always use the stub LLM.")
    p.add_argument(
        "--auto-approve",
        action="store_true",
        help="Approve every proposed PR without prompting (non-interactive).",
    )
    p.add_argument(
        "--simulate",
        type=int,
        default=0,
        metavar="N",
        help="First generate N mock failing logs into --log-dir.",
    )
    p.add_argument("--seed", type=int, default=None, help="Seed for --simulate.")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.simulate:
        CIPipelineSimulator(output_dir=args.log_dir, seed=args.seed).generate(
            count=args.simulate
        )

    config = OrchestratorConfig(
        log_dir=args.log_dir,
        pr_output_dir=args.pr_output_dir,
        approved_dir=args.approved_dir,
        audit_log=args.audit_log,
        model=args.model,
        force_mock=args.force_mock,
        auto_approve=args.auto_approve,
    )
    result = run_pipeline(config)
    # Non-zero exit if nothing was approved and there were items to act on.
    approved = result.counts.get(STATUS_APPROVED, 0)
    return 0 if (not result.items or approved) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
