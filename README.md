# Market Simulacra

Economic agent-based modelling using Large Language Models — a Streamlit app for
running the experiments described in *Market Simulacra: Economic Agent Based
Modelling Using Large Language Models*.

You define a market environment, a cast of economic actors (personas), a policy
under test, and a set of indicators to track. The app then runs the simulation
**one step at a time**, asking each persona how it reacts, judging which way each
indicator moved, and synthesizing an updated market state — pausing between steps
so you can inject events (disasters, geopolitical shocks, etc.).

## Features

- **Model selection** — pick from a short-list of models. Anthropic and OpenAI
  work out of the box; Gemini is wired up and appears automatically once you
  install its SDK (see *Adding a provider* below). Edit the `PROVIDERS`
  registry in `market_sim/llm.py` to change the model lists.
- **Reasoning effort** — for OpenAI reasoning models (e.g. `gpt-5.4-mini`), a
  sidebar control sets `reasoning_effort` (`none`/`low`/`medium`/`high`/`xhigh`,
  default `low`) to cap reasoning-token spend. The control is hidden for
  non-reasoning models.
- **Environment definition** — free-text initial market state.
- **Dynamic personas** — add, reorder (↑/↓), and remove actors with a name and
  description.
- **Policy simulation** — describe the policy and its objectives.
- **Index tracking** — custom indicators reported per step as direction
  (▲ up / ▼ down / ■ flat) + magnitude + rationale, not absolute values.
- **Step-by-step execution** — runs one step per click; inject events between steps.
- **Scripted events** — optionally pre-define an event per step ahead of the run,
  so the simulation injects them automatically instead of you typing one in live
  each step (step 1 is always the baseline). Toggle it in the Setup tab.
- **Shared decisions** — optionally let later personas see earlier personas'
  decisions within the same step.
- **Configurable timeframes** — start date, duration in days, step interval
  (day/week/month/year).
- **Sampling temperature** — set the temperature applied to every call this run
  (default `0.0` for reproducibility; raise it to study behavioural variance).
- **Real-time output** — live token streaming; tabbed views for decisions,
  indices, and market state.
- **Templates** — save/load simulation configs (`data/templates/`).
- **Past results** — browse, load, and review previous runs (`data/results/`).
- **Export** — runs auto-save to JSON after every step; download from the History tab.

## Setup

Requires Python ≥ 3.10. Using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/akhanafer/Market-Simulacra.git
cd Market-Simulacra
uv sync
cp .env.example .env        # then put your key in .env (optional — you can also set it in the UI)
uv run streamlit run app.py
```

> **Note:** run commands with `uv run` rather than activating the venv. Don't move
> or copy `.venv` — if you relocate the project, re-run `uv sync` at the new path
> to rebuild it (venv scripts hardcode absolute paths and won't survive a move).

Or with plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
streamlit run app.py
```

The API key defaults to the selected model's provider env var from `.env`
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `GEMINI_API_KEY`), and can be
overridden per provider in the sidebar at runtime (kept in memory only).

## Adding a provider

The LLM layer is provider-agnostic. Anthropic and OpenAI ship enabled; the Gemini
adapter is already written but stays hidden until its SDK is installed. To turn it
on:

```bash
uv add google-genai
```

The provider's models then appear in the sidebar automatically. Set the matching
key (`OPENAI_API_KEY` / `GEMINI_API_KEY`) in `.env` or paste it in the sidebar.
For Gemini, **verify the model IDs** in the `PROVIDERS` registry in
`market_sim/llm.py` against its current model list (those defaults are placeholders).

To add a brand-new provider, write an `LLMClient` subclass implementing
`stream_text`, `complete`, and `structured_output`, then add an entry to
`PROVIDERS`. No changes to `app.py` or `engine.py` are needed.

## Project layout

```
app.py                  Streamlit UI
market_sim/
  models.py             Pydantic data models
  llm.py                Provider-agnostic LLM clients + model registry
  prompts.py            Prompt builders for the 3 per-step calls
  engine.py             Step date math, prompt assembly, response parsing
  storage.py            Template + result JSON persistence
data/
  templates/            Saved configs
  results/              Auto-saved runs
```

## How a step works

For each step the app:

1. Asks **each persona** (in order) for its decision, optionally feeding earlier
   personas' decisions from the same step if *shared decisions* is on.
2. Asks an **analyst** call to rate the direction/magnitude of every tracked index.
3. Synthesizes an **updated market state** for the next step.
4. Advances the date by the step interval, saves the run, and waits.

All three prompts instruct the model to reason *forward* from the current market
state rather than recall the real-world outcome of any similar historical policy —
the methodological guardrail from the proposal.
