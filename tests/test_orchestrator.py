"""Integration tests for the end-to-end orchestrator (Day 7).

Covers the happy path, user rejection, unfixable errors, and the LLM mock
fallback. All tests force the stub LLM so they run offline and deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ci_triage_agent.agent import FixSuggestion
from ci_triage_agent.orchestrator import (
    Orchestrator,
    OrchestratorConfig,
    STATUS_APPROVED,
    STATUS_REJECTED,
    STATUS_UNFIXABLE,
)
from ci_triage_agent.parser import FailureType, ParsedFailure
from ci_triage_agent.reproducer import ReproductionResult
from ci_triage_agent.simulator import CIPipelineSimulator


def _make_config(tmp_path, **overrides) -> OrchestratorConfig:
    base = dict(
        log_dir=str(tmp_path / "ci_logs"),
        pr_output_dir=str(tmp_path / "pr_artifacts"),
        approved_dir=str(tmp_path / "approved_patches"),
        audit_log=str(tmp_path / "audit_trail.jsonl"),
        force_mock=True,
        reproduce_timeout=30,
        test_timeout=60,
    )
    base.update(overrides)
    return OrchestratorConfig(**base)


def _seed_logs(tmp_path, count=3, seed=7):
    sim = CIPipelineSimulator(output_dir=str(tmp_path / "ci_logs"), seed=seed)
    return sim.generate(count=count)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_happy_path_auto_approve(tmp_path):
    """Full pipeline with auto-approve: items are processed and at least one approved."""
    _seed_logs(tmp_path)
    config = _make_config(tmp_path, auto_approve=True)
    result = Orchestrator(config=config).run()

    assert result.items, "expected at least one failure to be processed"
    # Auto-approve means any item that reached the PR stage is approved.
    assert result.counts.get(STATUS_APPROVED, 0) >= 1
    # The audit trail file should have been written for approvals.
    assert (tmp_path / "audit_trail.jsonl").exists()


def test_mock_fallback_is_used(tmp_path):
    """With force_mock, the agent must report it used the stub (no network/credentials)."""
    _seed_logs(tmp_path)
    orch = Orchestrator(config=_make_config(tmp_path, auto_approve=True))
    assert orch.agent.used_mock is True
    result = orch.run()
    for item in result.items:
        if item.suggestion is not None:
            assert item.suggestion.used_mock is True


# ---------------------------------------------------------------------------
# User rejection
# ---------------------------------------------------------------------------

def test_user_rejection(tmp_path):
    """When the human rejects, items are marked rejected and nothing is approved."""
    _seed_logs(tmp_path)
    # Interactive workflow driven by an input stub that always rejects
    # (second prompt is the optional rejection note).
    replies = iter(["reject", ""] * 20)
    config = _make_config(tmp_path, auto_approve=False, input_fn=lambda _p: next(replies))
    result = Orchestrator(config=config).run()

    assert result.counts.get(STATUS_APPROVED, 0) == 0
    acted = [i for i in result.items if i.pull_request is not None]
    assert acted, "expected at least one PR to reach the approval step"
    assert all(i.status == STATUS_REJECTED for i in acted)


# ---------------------------------------------------------------------------
# Unfixable error (agent returns no patch) — via an injected stub agent
# ---------------------------------------------------------------------------

class _NoPatchAgent:
    """Stub agent whose suggestions never contain a patch."""

    used_mock = True
    model_name = "stub"

    def suggest_fix(self, failure, reproduction) -> FixSuggestion:
        return FixSuggestion(
            parsed_failure=failure,
            used_mock=True,
            explanation="Could not determine a safe fix.",
            patch="",  # no patch -> unfixable
            target_file=failure.error_file,
            model="stub",
        )


class _ConfirmingReproducer:
    pass


def test_unfixable_error(tmp_path, monkeypatch):
    """A failure the agent cannot patch is classified unfixable, not approved."""
    _seed_logs(tmp_path, count=2)

    # Force reproduction to always confirm, so we reach the agent step.
    import ci_triage_agent.orchestrator as orch_mod

    def _always_confirmed(failure, timeout=30):
        return ReproductionResult(
            parsed_failure=failure,
            process_exited_nonzero=True,
            stdout="",
            stderr="boom",
            confirmed=True,
            confirmation_reason="forced-confirm (test)",
        )

    monkeypatch.setattr(orch_mod, "reproduce_failure", _always_confirmed)

    config = _make_config(tmp_path, auto_approve=True)
    orch = Orchestrator(config=config, agent=_NoPatchAgent())
    result = orch.run()

    assert result.items
    assert all(i.status == STATUS_UNFIXABLE for i in result.items)
    assert result.counts.get(STATUS_APPROVED, 0) == 0


# ---------------------------------------------------------------------------
# Empty input
# ---------------------------------------------------------------------------

def test_no_logs_returns_empty(tmp_path):
    """With no logs, the run completes cleanly with no items."""
    (tmp_path / "ci_logs").mkdir()
    result = Orchestrator(config=_make_config(tmp_path)).run()
    assert result.items == []
