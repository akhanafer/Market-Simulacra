"""Market Simulacra — Streamlit front end.

A "step" is a button press, not a background loop: Streamlit reruns top-to-bottom
on every interaction, so we keep all state in st.session_state and run exactly one
step per click. That makes the pause-between-steps and inject-event behaviour from
the proposal fall out for free — the app simply waits for the next click.
"""

import copy
from typing import cast

import streamlit as st
from dotenv import load_dotenv

from market_sim import engine, llm, storage
from market_sim.models import (
    DIRECTION_ARROW,
    EconomicIndex,
    Persona,
    PersonaDecision,
    Policy,
    ReasoningEffort,
    SimulationConfig,
    SimulationRun,
    StepInterval,
    StepResult,
)

load_dotenv()
st.set_page_config(page_title="Market Simulacra", page_icon="📈", layout="wide")


# --------------------------------------------------------------------------- state


def _default_config() -> SimulationConfig:
    return SimulationConfig(
        model=llm.DEFAULT_MODEL,
        personas=[Persona(name="Household consumer", description="")],
        indices=[EconomicIndex(name="CPI", description="Consumer Price Index")],
        policy=Policy(),
    )


def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("config", _default_config())
    ss.setdefault("run", None)  # active SimulationRun
    ss.setdefault("api_keys", {})  # provider name -> override key (in-memory only)


def _current_key() -> str:
    """Override key for the selected model's provider, falling back to its env var."""
    provider = llm.provider_for_model(cfg.model)
    if provider is None:
        return ""
    override = st.session_state.api_keys.get(provider.name, "")
    return override or llm.resolve_default_key(cfg.model)


_init_state()
cfg: SimulationConfig = st.session_state.config


# --------------------------------------------------------------------------- sidebar


def render_sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        st.header("Model & key")

        models = llm.available_models()  # "Provider · Model" -> model id
        labels = list(models)
        current_label = next((lbl for lbl, mid in models.items() if mid == cfg.model), labels[0])
        cfg.model = models[current_label]  # snap to an available model if the old one is gone
        chosen = st.selectbox("Model", labels, index=labels.index(current_label))
        cfg.model = models[chosen]

        if llm.is_reasoning_model(cfg.model):
            # "minimal" is intentionally omitted: gpt-5.4-mini rejects it. A config
            # loaded with "minimal" still validates (it's a legal enum value), so
            # snap it to a supported default rather than crashing on .index().
            efforts: list[ReasoningEffort] = ["none", "low", "medium", "high", "xhigh"]
            current = cfg.reasoning_effort if cfg.reasoning_effort in efforts else "low"
            cfg.reasoning_effort = cast(
                ReasoningEffort,
                st.selectbox(
                    "Reasoning effort",
                    efforts,
                    index=efforts.index(current),
                    help="How hard the model thinks before answering. Lower = fewer "
                    "(billed) reasoning tokens; 'none' turns reasoning off.",
                ),
            )

        provider = llm.provider_for_model(cfg.model)
        if provider is None:
            st.error(f"No provider registered for model '{cfg.model}'.")
            return
        env_key = llm.resolve_default_key(cfg.model)
        placeholder = (
            f"loaded from {provider.key_env}" if env_key else f"set {provider.key_env} or paste here"
        )
        entered = st.text_input(
            f"{provider.label} API key (overrides env)",
            value=ss.api_keys.get(provider.name, ""),
            type="password",
            placeholder=placeholder,
            key=f"key_input_{provider.name}",
        )
        ss.api_keys[provider.name] = entered

        if st.button("Test key", use_container_width=True):
            ok, msg = llm.validate_key(cfg.model, _current_key())
            (st.success if ok else st.error)(msg)

        st.divider()
        st.header("Templates")
        _render_templates()


def _render_templates() -> None:
    ss = st.session_state
    name = st.text_input("Template name", placeholder="e.g. uk-carbon-levy")
    if st.button("💾 Save current config", use_container_width=True, disabled=not name):
        storage.save_template(name, cfg)
        st.success(f"Saved '{name}'.")

    existing = storage.list_templates()
    if existing:
        pick = st.selectbox("Load template", existing)
        c1, c2 = st.columns(2)
        if c1.button("📂 Load", use_container_width=True):
            ss.config = storage.load_template(pick)
            st.rerun()
        if c2.button("🗑️ Delete", use_container_width=True):
            storage.delete_template(pick)
            st.rerun()


# --------------------------------------------------------------------------- setup tab


def render_setup() -> None:
    st.subheader("Environment")
    cfg.environment_description = st.text_area(
        "Initial market state (country, economic conditions, etc.)",
        value=cfg.environment_description,
        height=140,
        placeholder="e.g. A mid-sized export-driven economy with 4% unemployment, "
        "cooling inflation, and a large public-sector workforce...",
    )

    st.subheader("Timeframe")
    c1, c2, c3, c4 = st.columns(4)
    cfg.start_date = c1.date_input("Start date", value=cfg.start_date)
    cfg.duration_days = c2.number_input("Duration (days)", min_value=1, value=cfg.duration_days, step=1)
    intervals = ["day", "week", "month", "year"]
    cfg.step_interval = cast(
        StepInterval,
        c3.selectbox("Step interval", intervals, index=intervals.index(cfg.step_interval)),
    )
    cfg.shared_decisions = c4.checkbox(
        "Shared decisions",
        value=cfg.shared_decisions,
        help="Let later personas see earlier personas' decisions within the same step.",
    )
    st.caption(f"→ {cfg.total_steps()} step(s) total.")

    cfg.auto_inject = st.checkbox(
        "Auto-inject scripted events",
        value=cfg.auto_inject,
        help="Pre-define an event for each step now, instead of typing one in live "
        "during the run. Step 1 is always the baseline (no event).",
    )
    if cfg.auto_inject:
        _render_scripted_events()

    st.divider()
    _render_personas()
    st.divider()
    _render_indices()
    st.divider()
    _render_policy()

    st.divider()
    started = st.session_state.run is not None
    btn = "🔄 Restart simulation" if started else "🚀 Start simulation"
    if st.button(btn, type="primary", use_container_width=True):
        _start_run()


def _render_personas() -> None:
    st.subheader("Personas (economic actors)")
    st.caption(
        "Personas decide in the order listed — use ↑/↓ to reorder. With *shared "
        "decisions* on, later personas can see the choices of earlier ones this step."
    )
    personas = cfg.personas
    for i, p in enumerate(personas):
        with st.container(border=True):
            top = st.columns([6, 1, 1, 1])
            p.name = top[0].text_input("Name", value=p.name, key=f"pn_{p.id}", label_visibility="collapsed")
            if top[1].button("↑", key=f"pu_{p.id}", disabled=i == 0):
                personas[i - 1], personas[i] = personas[i], personas[i - 1]
                st.rerun()
            if top[2].button("↓", key=f"pd_{p.id}", disabled=i == len(personas) - 1):
                personas[i + 1], personas[i] = personas[i], personas[i + 1]
                st.rerun()
            if top[3].button("🗑️", key=f"px_{p.id}"):
                personas.pop(i)
                st.rerun()
            p.description = st.text_area(
                "Description",
                value=p.description,
                key=f"pdesc_{p.id}",
                height=80,
                label_visibility="collapsed",
                placeholder="Role, incentives, constraints...",
            )
    if st.button("➕ Add persona"):
        personas.append(Persona(name=f"Actor {len(personas) + 1}"))
        st.rerun()


def _render_indices() -> None:
    st.subheader("Economic indices to track")
    st.caption("Tracked as direction (▲ up / ▼ down / ■ flat) per step — not absolute values.")
    indices = cfg.indices
    for i, ix in enumerate(indices):
        with st.container(border=True):
            row = st.columns([4, 6, 1])
            ix.name = row[0].text_input(
                "Name", value=ix.name, key=f"in_{ix.id}", label_visibility="collapsed", placeholder="GDP"
            )
            ix.description = row[1].text_input(
                "Description",
                value=ix.description,
                key=f"id_{ix.id}",
                label_visibility="collapsed",
                placeholder="what it measures",
            )
            if row[2].button("🗑️", key=f"ix_{ix.id}"):
                indices.pop(i)
                st.rerun()
    if st.button("➕ Add index"):
        indices.append(EconomicIndex(name=""))
        st.rerun()


def _render_policy() -> None:
    st.subheader("Policy under test")
    cfg.policy.description = st.text_area(
        "Policy description",
        value=cfg.policy.description,
        height=100,
        placeholder="Describe the specific economic policy being introduced...",
    )
    cfg.policy.objectives = st.text_area(
        "Objectives",
        value=cfg.policy.objectives,
        height=80,
        placeholder="What is this policy intended to achieve?",
    )


def _render_scripted_events() -> None:
    st.subheader("Scripted events")
    st.caption(
        "One event per step, injected automatically before that step runs. Leave a "
        "step blank to inject nothing. Step 1 establishes the baseline and takes no event."
    )
    total = cfg.total_steps()
    if total < 2:
        st.info("Only one step — no events to script. Increase the duration to schedule events.")
        return
    for n in range(2, total + 1):
        the_date = engine.step_date(cfg, n).isoformat()
        cfg.scripted_events[n] = st.text_area(
            f"Step {n} — {the_date}",
            value=cfg.scripted_events.get(n, ""),
            key=f"scripted_evt_{n}",
            height=70,
            placeholder="e.g. A major earthquake disrupts the southern industrial region.",
        )
    # Drop entries for steps that no longer exist (duration/interval shrank), so a
    # saved template doesn't carry stale events past the end of the run.
    for stale in [n for n in cfg.scripted_events if n > total]:
        del cfg.scripted_events[stale]


def _start_run() -> None:
    run = SimulationRun(
        config=copy.deepcopy(cfg),
        market_state=cfg.environment_description,
    )
    storage.save_run(run)
    st.session_state.run = run
    st.success("Simulation started — go to the **Simulate** tab.")


# --------------------------------------------------------------------------- simulate tab


def _client() -> llm.LLMClient | None:
    try:
        return llm.build_client(cfg.model, _current_key(), cfg.reasoning_effort)
    except ValueError as exc:
        st.error(str(exc))
        return None


def render_simulate() -> None:
    run: SimulationRun | None = st.session_state.run
    if run is None:
        st.info("Configure a simulation in the **Setup** tab, then start it.")
        return

    rcfg = run.config
    done_steps = len(run.steps)
    total = rcfg.total_steps()
    is_done = done_steps >= total

    st.progress(done_steps / total, text=f"Step {done_steps} / {total}")

    # No injection on the very first step: it establishes the baseline from the
    # initial market state. Events are perturbations injected between steps.
    injection = ""
    if done_steps == 0:
        st.caption("The first step runs from the initial market state — inject events from step 2 onward.")
    elif rcfg.auto_inject:
        injection = rcfg.scripted_events.get(done_steps + 1, "")
        if injection:
            st.info(f"Scripted event for the next step:\n\n{injection}")
        else:
            st.caption("Auto-inject is on, but no event is scripted for the next step.")
    else:
        injection = st.text_area(
            "Inject an event before the next step (optional)",
            key=f"injection_{done_steps}",
            placeholder="e.g. A major earthquake disrupts the southern industrial region.",
            height=70,
        )

    cols = st.columns([1, 4])
    run_clicked = cols[0].button(
        "▶️ Run next step", type="primary", disabled=is_done, use_container_width=True
    )
    if is_done:
        cols[1].success("Simulation complete.")

    live, dec, idx, mkt = st.tabs(["🔴 Live step", "🧑 Decisions", "📊 Indices", "🏦 Market state"])

    with live:
        if run_clicked:
            client = _client()
            if client is not None:
                _execute_step(client, run, injection)
                st.rerun()
        elif run.steps:
            _render_step_detail(run.steps[-1], is_done=is_done)
        else:
            st.caption("No steps run yet. Press **Run next step**.")

    with dec:
        _render_decisions(run)
    with idx:
        _render_indices_matrix(run)
    with mkt:
        _render_market(run)


def _execute_step(client: llm.LLMClient, run: SimulationRun, injection: str) -> None:
    rcfg = run.config
    n = len(run.steps) + 1
    the_date = engine.step_date(rcfg, n)
    st.caption(f"Step {n} — {the_date.isoformat()}")

    decisions: list[PersonaDecision] = []
    for p in rcfg.personas:
        st.markdown(f"**{p.name}**")
        try:
            text = st.write_stream(
                engine.persona_stream(
                    client, rcfg, run.market_state, p, n, the_date, injection, decisions, run.steps
                )
            )
        except Exception as exc:  # noqa: BLE001 - surface API errors to the UI
            st.error(f"Persona '{p.name}' failed: {exc}")
            return
        decisions.append(PersonaDecision(persona_id=p.id, persona_name=p.name, decision=str(text)))

    readings = []
    if rcfg.indices:
        st.markdown("**Index readings**")
        with st.spinner("Assessing indices..."):
            readings = engine.assess_indices(
                client, rcfg, rcfg.indices, run.market_state, decisions, the_date, run.steps
            )
        for r in readings:
            st.write(f"{DIRECTION_ARROW[r.direction]} **{r.index_name}** ({r.magnitude}) — {r.rationale}")

    # The market summary still updates every step (it feeds the next step's
    # personas), but it's only shown once the run finishes: stream it live on the
    # final step, generate it quietly otherwise.
    market_stream = engine.summarize_market(
        client, rcfg, run.market_state, decisions, injection, the_date, run.steps
    )
    if n >= rcfg.total_steps():
        st.markdown("**Final market state**")
        summary = str(st.write_stream(market_stream))
    else:
        with st.spinner("Updating market state..."):
            summary = "".join(market_stream)

    run.steps.append(
        StepResult(
            step_number=n,
            step_date=the_date,
            environment_injection=injection,
            persona_decisions=decisions,
            index_readings=readings,
            market_summary=summary,
        )
    )
    run.market_state = summary or run.market_state
    storage.save_run(run)


# --------------------------------------------------------------------------- output renderers


def _render_step_detail(step: StepResult, is_done: bool = False) -> None:
    st.caption(f"Step {step.step_number} — {step.step_date.isoformat()}")
    if step.environment_injection:
        st.info(f"Injected: {step.environment_injection}")
    for d in step.persona_decisions:
        st.markdown(f"**{d.persona_name}**")
        st.write(d.decision)
    if step.index_readings:
        st.markdown("**Index readings**")
        for r in step.index_readings:
            st.write(f"{DIRECTION_ARROW[r.direction]} **{r.index_name}** ({r.magnitude}) — {r.rationale}")
    if is_done:
        st.markdown("**Final market state**")
        st.write(step.market_summary)


def _render_decisions(run: SimulationRun) -> None:
    if not run.steps:
        st.caption("No decisions yet.")
        return
    for step in run.steps:
        with st.expander(
            f"Step {step.step_number} — {step.step_date.isoformat()}", expanded=step is run.steps[-1]
        ):
            if step.environment_injection:
                st.info(f"Injected: {step.environment_injection}")
            for d in step.persona_decisions:
                st.markdown(f"**{d.persona_name}**")
                st.write(d.decision)


def _render_indices_matrix(run: SimulationRun) -> None:
    if not run.config.indices:
        st.caption("No indices configured.")
        return
    if not run.steps:
        st.caption("No readings yet.")
        return

    names = [ix.name for ix in run.config.indices]
    header = "| Step | Date | " + " | ".join(names) + " |"
    sep = "|" + "---|" * (len(names) + 2)
    rows = [header, sep]
    for step in run.steps:
        by_name = {r.index_name: r for r in step.index_readings}
        cells = []
        for name in names:
            r = by_name.get(name)
            cells.append(DIRECTION_ARROW[r.direction] if r else "·")
        rows.append(f"| {step.step_number} | {step.step_date.isoformat()} | " + " | ".join(cells) + " |")
    st.markdown("\n".join(rows))

    st.caption("Per-step rationales:")
    for step in run.steps:
        with st.expander(f"Step {step.step_number} — {step.step_date.isoformat()}"):
            for r in step.index_readings:
                st.write(f"{DIRECTION_ARROW[r.direction]} **{r.index_name}** ({r.magnitude}) — {r.rationale}")


def _render_market(run: SimulationRun) -> None:
    st.markdown("**Initial state**")
    st.write(run.config.environment_description or "_(none)_")
    if len(run.steps) < run.config.total_steps():
        st.caption("The market-state summary appears once the simulation completes.")
        return
    final = run.steps[-1]
    st.markdown(f"**Final state — after step {final.step_number} ({final.step_date.isoformat()})**")
    st.write(final.market_summary)


# --------------------------------------------------------------------------- history tab


def render_history() -> None:
    runs = storage.list_runs()
    if not runs:
        st.info("No saved runs yet. Runs auto-save after each step.")
        return

    labels = {r.title(): r for r in runs}
    pick = st.selectbox("Saved runs", list(labels))
    run = labels[pick]

    c1, c2, c3 = st.columns(3)
    if c1.button("📂 Load config into Setup", use_container_width=True):
        st.session_state.config = copy.deepcopy(run.config)
        st.success("Config loaded into Setup tab.")
    c2.download_button(
        "⬇️ Download JSON",
        data=run.model_dump_json(indent=2),
        file_name=f"{run.id}.json",
        mime="application/json",
        use_container_width=True,
    )
    if c3.button("🗑️ Delete run", use_container_width=True):
        storage.delete_run(run.id)
        st.rerun()

    st.divider()
    st.caption(
        f"Model: {run.config.model} · {len(run.steps)} steps · "
        f"{run.config.step_interval} interval · policy: "
        f"{run.config.policy.description[:80] or '(none)'}"
    )

    v_dec, v_idx, v_mkt = st.tabs(["🧑 Decisions", "📊 Indices", "🏦 Market state"])
    with v_dec:
        _render_decisions(run)
    with v_idx:
        _render_indices_matrix(run)
    with v_mkt:
        _render_market(run)


# --------------------------------------------------------------------------- main

st.title("📈 Market Simulacra")
st.caption("Economic agent-based modelling with LLMs")

render_sidebar()
tab_setup, tab_sim, tab_hist = st.tabs(["⚙️ Setup", "▶️ Simulate", "🗂️ History"])
with tab_setup:
    render_setup()
with tab_sim:
    render_simulate()
with tab_hist:
    render_history()
