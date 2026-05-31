"""Tests for the Pydantic data models and their derived helpers."""

from datetime import date
from typing import get_args

from market_sim.models import (
    DIRECTION_ARROW,
    Direction,
    EconomicIndex,
    IndexReading,
    Persona,
    SimulationConfig,
    SimulationRun,
    StepResult,
)


def _config(**kw) -> SimulationConfig:
    return SimulationConfig(model="claude-sonnet-4-6", **kw)


def test_total_steps_by_interval():
    assert _config(duration_days=30, step_interval="day").total_steps() == 30
    assert _config(duration_days=30, step_interval="week").total_steps() == 4
    assert _config(duration_days=30, step_interval="month").total_steps() == 1
    assert _config(duration_days=365, step_interval="year").total_steps() == 1


def test_total_steps_is_at_least_one():
    # Duration shorter than a single interval still yields one step.
    assert _config(duration_days=3, step_interval="week").total_steps() == 1


def test_config_defaults_are_deterministic_and_cheap():
    cfg = _config()
    assert cfg.temperature == 0.0
    assert cfg.reasoning_effort == "minimal"


def test_direction_arrow_covers_every_direction():
    # Every value the Direction literal allows must have a glyph, or the UI KeyErrors.
    assert set(get_args(Direction)) == set(DIRECTION_ARROW)


def test_index_reading_label():
    r = IndexReading(index_name="CPI", direction="up", magnitude="slight", rationale="x")
    assert r.label() == "▲ CPI (slight)"


def test_simulation_run_json_round_trip():
    run = SimulationRun(
        config=_config(personas=[Persona(name="HH")], indices=[EconomicIndex(name="CPI")]),
        market_state="start",
        steps=[
            StepResult(
                step_number=1,
                step_date=date(2026, 1, 1),
                index_readings=[IndexReading(index_name="CPI", direction="up")],
                market_summary="after step 1",
            )
        ],
    )
    restored = SimulationRun.model_validate_json(run.model_dump_json())
    assert restored == run
    assert restored.steps[0].index_readings[0].direction == "up"
