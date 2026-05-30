"""JSON persistence for templates (configs) and results (runs).

Everything lives under data/ as plain JSON so runs are portable and diffable.
Templates store a SimulationConfig; results store a full SimulationRun and are
auto-saved after every step.
"""

import re
from pathlib import Path

from .models import SimulationConfig, SimulationRun

_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = _ROOT / "data" / "templates"
RESULTS_DIR = _ROOT / "data" / "results"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-").lower()
    return s or "untitled"


def _ensure_dirs() -> None:
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --- Templates -------------------------------------------------------------

def save_template(name: str, config: SimulationConfig) -> Path:
    _ensure_dirs()
    path = TEMPLATES_DIR / f"{_slug(name)}.json"
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")
    return path


def list_templates() -> list[str]:
    _ensure_dirs()
    return sorted(p.stem for p in TEMPLATES_DIR.glob("*.json"))


def load_template(name: str) -> SimulationConfig:
    path = TEMPLATES_DIR / f"{_slug(name)}.json"
    return SimulationConfig.model_validate_json(path.read_text(encoding="utf-8"))


def delete_template(name: str) -> None:
    path = TEMPLATES_DIR / f"{_slug(name)}.json"
    path.unlink(missing_ok=True)


# --- Results ---------------------------------------------------------------

def result_path(run: SimulationRun) -> Path:
    return RESULTS_DIR / f"{run.id}.json"


def save_run(run: SimulationRun) -> Path:
    """Auto-save a run; overwrites the same file each step (keyed by run id)."""
    _ensure_dirs()
    path = result_path(run)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def list_runs() -> list[SimulationRun]:
    """Newest first. Skips any unparseable file rather than crashing."""
    _ensure_dirs()
    runs: list[SimulationRun] = []
    for p in RESULTS_DIR.glob("*.json"):
        try:
            runs.append(SimulationRun.model_validate_json(p.read_text(encoding="utf-8")))
        except Exception:  # noqa: BLE001 - tolerate hand-edited / partial files
            continue
    return sorted(runs, key=lambda r: r.created_at, reverse=True)


def load_run(run_id: str) -> SimulationRun:
    path = RESULTS_DIR / f"{run_id}.json"
    return SimulationRun.model_validate_json(path.read_text(encoding="utf-8"))


def delete_run(run_id: str) -> None:
    (RESULTS_DIR / f"{run_id}.json").unlink(missing_ok=True)
