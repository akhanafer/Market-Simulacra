"""Tests for the simulation engine: date math, reading alignment, and the
structured-output index assessment (with the LLM client faked)."""

from datetime import date

from market_sim import engine, llm
from market_sim.models import EconomicIndex, IndexAssessment, IndexAssessmentBatch, SimulationConfig


def _config(**kw) -> SimulationConfig:
    return SimulationConfig(model="m", **kw)


def test_step_date_advances_by_interval():
    cfg = _config(start_date=date(2026, 1, 1), step_interval="week")
    assert engine.step_date(cfg, 1) == date(2026, 1, 1)  # first step is the start date
    assert engine.step_date(cfg, 3) == date(2026, 1, 15)  # +14 days


def test_align_readings_matches_and_preserves_config_order():
    indices = [EconomicIndex(name="CPI"), EconomicIndex(name="GDP")]
    returned = [  # deliberately out of order vs config
        IndexAssessment(index_name="GDP", direction="down", magnitude="strong", rationale="b"),
        IndexAssessment(index_name="CPI", direction="up", magnitude="slight", rationale="a"),
    ]
    out = engine._align_readings(returned, indices)
    assert [r.index_name for r in out] == ["CPI", "GDP"]
    assert out[0].direction == "up"
    assert out[1].direction == "down"


def test_align_readings_is_case_insensitive_and_uses_canonical_name():
    indices = [EconomicIndex(name="CPI")]
    returned = [IndexAssessment(index_name="  cpi ", direction="up", magnitude="slight", rationale="a")]
    out = engine._align_readings(returned, indices)
    assert out[0].index_name == "CPI"  # config's spelling, not the model's echo


def test_align_readings_marks_missing_index_undetermined():
    indices = [EconomicIndex(name="CPI"), EconomicIndex(name="GDP")]
    returned = [IndexAssessment(index_name="CPI", direction="up", magnitude="slight", rationale="a")]
    gdp = engine._align_readings(returned, indices)[1]
    assert gdp.direction == "undetermined"
    assert gdp.magnitude == "undetermined"
    assert gdp.rationale == "No reading returned."


def test_align_readings_ignores_indices_not_in_config():
    indices = [EconomicIndex(name="CPI")]
    returned = [
        IndexAssessment(index_name="CPI", direction="up", magnitude="slight", rationale="a"),
        IndexAssessment(index_name="Unemployment", direction="down", magnitude="strong", rationale="z"),
    ]
    out = engine._align_readings(returned, indices)
    assert [r.index_name for r in out] == ["CPI"]


class _FakeClient(llm.LLMClient):
    """An LLMClient whose structured_output returns a canned batch, no network."""

    def __init__(self, batch: IndexAssessmentBatch):
        super().__init__(api_key="x", model="m")
        self._batch = batch
        self.calls = 0

    def stream_text(self, system, user, max_tokens=1024, temperature=1.0):
        yield ""

    def complete(self, system, user, max_tokens=1024, temperature=1.0):
        return ""

    def structured_output(self, schema, system, user, max_tokens=1024, temperature=1.0):
        self.calls += 1
        assert schema is IndexAssessmentBatch
        return self._batch


def test_assess_indices_skips_call_when_no_indices():
    client = _FakeClient(IndexAssessmentBatch(readings=[]))
    assert engine.assess_indices(client, [], "state", [], date(2026, 1, 1)) == []
    assert client.calls == 0


def test_assess_indices_aligns_structured_output():
    batch = IndexAssessmentBatch(
        readings=[IndexAssessment(index_name="CPI", direction="up", magnitude="slight", rationale="a")]
    )
    client = _FakeClient(batch)
    out = engine.assess_indices(client, [EconomicIndex(name="CPI")], "state", [], date(2026, 1, 1))
    assert client.calls == 1
    assert out[0].direction == "up"
