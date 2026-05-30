# CLAUDE.md

Market Simulacra — a Streamlit app for LLM-driven economic agent-based modelling.
Define a market, personas (economic actors), a policy, and indices to track; the
app runs the simulation one step per click, asking each persona how it reacts,
judging how each index moved, and synthesizing the next market state.

## Commands (always via `uv run`)

```bash
uv run streamlit run app.py     # launch the app
uv run pytest                   # tests (fast, no network — LLM calls are faked)
uv run ruff check . --fix       # lint + autofix
uv run ruff format .            # format
uv run pyright                  # type-check
```

Run the full check before considering a change done: `ruff check .`, `ruff format --check .`, `pyright`, `pytest`.

## Environment & tooling rules

- **Use `uv` only.** Never `pip install` and never `source .venv/bin/activate` —
  prefix everything with `uv run`. Install deps with `uv add <pkg>` (or
  `uv add --dev <pkg>`), not by editing `pyproject.toml` by hand.
- **Don't move/copy `.venv`.** It is not relocatable; after relocating the
  project, run `uv sync` to rebuild it.
- Python ≥ 3.10. Built-in generics (`list[...]`, `X | None`) work natively, so do
  **not** add `from __future__ import annotations`.

## Architecture

`app.py` is the only UI; it drives the loop one step per button press (Streamlit
reruns top-to-bottom, state lives in `st.session_state`). The engine never owns
the loop.

| File | Role |
|---|---|
| `market_sim/models.py` | Pydantic models (config, run, readings) + `Direction`/`Magnitude` enums |
| `market_sim/llm.py` | Provider-agnostic LLM clients + `PROVIDERS` registry |
| `market_sim/prompts.py` | The 3 per-step prompt builders |
| `market_sim/engine.py` | Step date math, prompt assembly, reading alignment |
| `market_sim/storage.py` | Template + result JSON persistence under `data/` |

## Key conventions (don't break these)

- **Methodological guardrail:** every prompt instructs the model to *reason
  forward* from the current market state, never to recall the real-world
  historical outcome of a similar policy. See `_GUARDRAIL` in `prompts.py`. Keep
  this in any new prompt.
- **Indices are directional, not absolute.** A reading is a `Direction`
  (up/down/flat/undetermined) + `Magnitude`, never a numeric level. The model
  picks `undetermined` only when it genuinely can't call it; the engine also uses
  `undetermined` to fill any index the model omits (`engine._align_readings`).
- **Adding an LLM provider:** subclass `LLMClient` (implement `stream_text`,
  `complete`, `structured_output`), then add an entry to `PROVIDERS` in `llm.py`.
  A provider auto-appears in the UI once its SDK is importable — no `app.py` or
  `engine.py` changes. OpenAI/Gemini adapters exist but are untested live, and
  their model IDs are placeholders to verify.
- **Structured output** goes through `LLMClient.structured_output(schema, ...)`
  with a Pydantic schema — never hand-parse JSON from a text completion.
- **The market summary is generated every step** (it feeds the next step's
  personas) but only **displayed** once the run completes.

## Code style

- Match the surrounding style. Type hints on signatures. Comments explain *why*,
  not *what*, and are sparse.
- Tests cover pure logic (engine, models, storage, llm registry); LLM calls are
  faked with a small `LLMClient` subclass or a stub SDK client — tests must not
  hit the network.
