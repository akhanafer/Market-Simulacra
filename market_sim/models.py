"""Pydantic data models for Market Simulacra simulations.

These models define both the *configuration* of a simulation (which is what a
template stores) and the *results* of running one (a SimulationRun, which is what
gets auto-saved to data/results/).
"""

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


StepInterval = Literal["day", "week", "month", "year"]

# Days advanced per step for each interval. Months/years are approximate; the
# simulation cares about ordering and rough cadence, not calendar exactness.
INTERVAL_DAYS: dict[StepInterval, int] = {
    "day": 1,
    "week": 7,
    "month": 30,
    "year": 365,
}

# Direction of an economic index for a single step. We deliberately avoid
# absolute values: the model reports which way the index moved, not a level.
# "undetermined" is never chosen by the model — we use it when the analyst call
# omits an index entirely, so it reads clearly instead of masquerading as "flat".
Direction = Literal["up", "down", "flat", "undetermined"]
Magnitude = Literal["slight", "moderate", "strong", "undetermined"]

DIRECTION_ARROW: dict[Direction, str] = {"up": "▲", "down": "▼", "flat": "■", "undetermined": "?"}


class Persona(BaseModel):
    """An economic actor in the simulation."""

    id: str = Field(default_factory=_new_id)
    name: str
    description: str = ""


class EconomicIndex(BaseModel):
    """A custom economic indicator the user wants tracked (CPI, GDP, ...)."""

    id: str = Field(default_factory=_new_id)
    name: str
    description: str = ""


class Policy(BaseModel):
    """The economic policy under test plus what it is trying to achieve."""

    description: str = ""
    objectives: str = ""


class SimulationConfig(BaseModel):
    """Everything needed to (re)start a simulation. This is what a template stores."""

    model: str
    environment_description: str = ""
    personas: list[Persona] = Field(default_factory=list)
    indices: list[EconomicIndex] = Field(default_factory=list)
    policy: Policy = Field(default_factory=Policy)

    start_date: date = Field(default_factory=date.today)
    duration_days: int = 30
    step_interval: StepInterval = "week"
    shared_decisions: bool = False
    # Sampling temperature for every LLM call this run. Defaults to 0.0 for
    # reproducibility (the same config + history should reproduce a step); raise
    # it to study behavioural variance.
    temperature: float = 0.0

    def total_steps(self) -> int:
        per = INTERVAL_DAYS[self.step_interval]
        return max(1, self.duration_days // per)


class PersonaDecision(BaseModel):
    """One persona's action during one step."""

    persona_id: str
    persona_name: str
    decision: str = ""


class IndexReading(BaseModel):
    """How a single index moved during a single step."""

    index_name: str
    direction: Direction = "flat"
    magnitude: Magnitude = "moderate"
    rationale: str = ""

    def label(self) -> str:
        return f"{DIRECTION_ARROW[self.direction]} {self.index_name} ({self.magnitude})"


class IndexAssessment(BaseModel):
    """One index reading as returned by the analyst call (schema for structured output).

    The model may report "undetermined" when the evidence doesn't support a call;
    we also use that value as the fill-in for any index the model omits entirely.
    """

    index_name: str
    direction: Direction
    magnitude: Magnitude
    rationale: str


class IndexAssessmentBatch(BaseModel):
    """Top-level structured-output schema: one assessment per tracked index."""

    readings: list[IndexAssessment] = Field(default_factory=list)


class StepResult(BaseModel):
    """The full outcome of one simulation step."""

    step_number: int
    step_date: date
    environment_injection: str = ""
    persona_decisions: list[PersonaDecision] = Field(default_factory=list)
    index_readings: list[IndexReading] = Field(default_factory=list)
    market_summary: str = ""


class SimulationRun(BaseModel):
    """A config plus all steps executed against it. Auto-saved after every step."""

    id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=datetime.now)
    config: SimulationConfig
    steps: list[StepResult] = Field(default_factory=list)
    # Running narrative of the market, seeded from the environment description and
    # updated each step. Fed into the next step's prompts as "current state".
    market_state: str = ""

    def title(self) -> str:
        when = self.created_at.strftime("%Y-%m-%d %H:%M")
        n = len(self.steps)
        return f"{when} — {self.config.step_interval}×{n} ({self.config.model})"
