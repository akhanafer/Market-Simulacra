"""Simulation step logic.

The engine builds prompts, calls the LLM, and parses responses. It deliberately
does *not* own the step loop — app.py drives the loop one step per button press
so Streamlit can stream each persona's response live and pause between steps.
"""

from collections.abc import Iterator
from datetime import date, timedelta

from . import prompts
from .llm import LLMClient
from .models import (
    INTERVAL_DAYS,
    EconomicIndex,
    IndexAssessment,
    IndexAssessmentBatch,
    IndexReading,
    Persona,
    PersonaDecision,
    SimulationConfig,
)


def step_date(config: SimulationConfig, step_number: int) -> date:
    """Date for a 1-based step number, advancing by the configured interval."""
    offset = INTERVAL_DAYS[config.step_interval] * (step_number - 1)
    return config.start_date + timedelta(days=offset)


def persona_stream(
    client: LLMClient,
    config: SimulationConfig,
    market_state: str,
    persona: Persona,
    step_number: int,
    the_date: date,
    injection: str,
    prior_decisions: list[PersonaDecision],
) -> Iterator[str]:
    """Stream one persona's decision text. Consume with ``st.write_stream``."""
    system = prompts.persona_system(persona)
    user = prompts.persona_user(
        config=config,
        market_state=market_state,
        step_number=step_number,
        step_date=the_date.isoformat(),
        injection=injection,
        prior_decisions=prior_decisions,
    )
    return client.stream_text(system, user, max_tokens=600)


def summarize_market(
    client: LLMClient,
    previous_state: str,
    decisions: list[PersonaDecision],
    injection: str,
    the_date: date,
) -> Iterator[str]:
    """Stream the updated market-state narrative."""
    system = prompts.market_system()
    user = prompts.market_prompt(
        previous_state=previous_state,
        decisions=decisions,
        injection=injection,
        step_date=the_date.isoformat(),
    )
    return client.stream_text(system, user, max_tokens=500)


def assess_indices(
    client: LLMClient,
    indices: list[EconomicIndex],
    market_state: str,
    decisions: list[PersonaDecision],
    the_date: date,
) -> list[IndexReading]:
    """Ask the analyst model for a direction reading per index via structured output."""
    if not indices:
        return []
    system = prompts.index_system()
    user = prompts.index_prompt(indices, market_state, decisions, the_date.isoformat())
    batch = client.structured_output(IndexAssessmentBatch, system, user, max_tokens=900, temperature=0.3)
    return _align_readings(batch.readings, indices)


def _align_readings(returned: list[IndexAssessment], indices: list[EconomicIndex]) -> list[IndexReading]:
    """One reading per configured index, in order. Indices the model omitted are
    marked "undetermined" rather than silently treated as flat."""
    by_name = {a.index_name.strip().lower(): a for a in returned}
    readings: list[IndexReading] = []
    for ix in indices:
        match = by_name.get(ix.name.strip().lower())
        if match is not None:
            readings.append(
                IndexReading(
                    index_name=ix.name,  # canonical name from config, not the model's echo
                    direction=match.direction,
                    magnitude=match.magnitude,
                    rationale=match.rationale,
                )
            )
        else:
            readings.append(
                IndexReading(
                    index_name=ix.name,
                    direction="undetermined",
                    magnitude="undetermined",
                    rationale="No reading returned.",
                )
            )
    return readings
