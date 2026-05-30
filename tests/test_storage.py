"""Tests for JSON persistence of templates and runs. The storage dirs are
redirected to a tmp path so tests never touch the real data/ directory."""

from datetime import datetime

import pytest

from market_sim import storage
from market_sim.models import EconomicIndex, Persona, SimulationConfig, SimulationRun


@pytest.fixture
def tmp_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "TEMPLATES_DIR", tmp_path / "templates")
    monkeypatch.setattr(storage, "RESULTS_DIR", tmp_path / "results")
    return tmp_path


def _config() -> SimulationConfig:
    return SimulationConfig(
        model="claude-sonnet-4-6",
        personas=[Persona(name="HH")],
        indices=[EconomicIndex(name="CPI")],
    )


def test_slug():
    assert storage._slug("UK Carbon Levy!") == "uk-carbon-levy"
    assert storage._slug("   ") == "untitled"


def test_template_round_trip(tmp_storage):
    storage.save_template("My Template", _config())
    assert storage.list_templates() == ["my-template"]
    loaded = storage.load_template("My Template")
    assert loaded.personas[0].name == "HH"
    storage.delete_template("My Template")
    assert storage.list_templates() == []


def test_run_round_trip_newest_first(tmp_storage):
    older = SimulationRun(config=_config(), created_at=datetime(2026, 1, 1))
    newer = SimulationRun(config=_config(), created_at=datetime(2026, 2, 1))
    storage.save_run(older)
    storage.save_run(newer)

    assert [r.id for r in storage.list_runs()] == [newer.id, older.id]  # newest first
    assert storage.load_run(newer.id).id == newer.id

    storage.delete_run(newer.id)
    assert [r.id for r in storage.list_runs()] == [older.id]


def test_list_runs_skips_unparseable_files(tmp_storage):
    storage.save_run(SimulationRun(config=_config()))
    (storage.RESULTS_DIR / "broken.json").write_text("{ not json", encoding="utf-8")
    assert len(storage.list_runs()) == 1  # the bad file is skipped, not raised
