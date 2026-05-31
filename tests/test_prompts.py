"""Tests for prompt construction — focused on the cross-step history block that
keeps every call from being single-shot, and the per-persona memory in it."""

from datetime import date

from market_sim import prompts
from market_sim.models import (
    IndexReading,
    Persona,
    PersonaDecision,
    SimulationConfig,
    StepResult,
)


def _config(**kw) -> SimulationConfig:
    return SimulationConfig(model="m", **kw)


def _step(persona_id: str) -> StepResult:
    return StepResult(
        step_number=1,
        step_date=date(2026, 1, 1),
        persona_decisions=[
            PersonaDecision(persona_id=persona_id, persona_name="HH", decision="I hoarded dollars."),
            PersonaDecision(persona_id="other", persona_name="Firm", decision="I raised prices."),
        ],
        index_readings=[IndexReading(index_name="CPI", direction="up", magnitude="moderate")],
        market_summary="Inflation accelerated.",
    )


def test_persona_user_includes_own_prior_decision_as_memory():
    p = Persona(name="HH")
    out = prompts.persona_user(
        config=_config(),
        market_state="now",
        persona=p,
        step_number=2,
        step_date="2026-01-08",
        injection="",
        prior_decisions=[],
        history=[_step(p.id)],
    )
    assert "HISTORY OF PRIOR STEPS" in out
    assert "your decision: I hoarded dollars." in out
    # The persona sees the market/index trajectory, but not another actor's
    # decision dressed up as its own memory.
    assert "Inflation accelerated." in out
    assert "I raised prices." not in out


def test_history_block_empty_when_no_steps():
    assert prompts._history_block([]) == ""
