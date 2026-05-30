"""Prompt construction for the three model calls that make up one step:

1. ``persona_prompt``  -> each actor decides what to do this step.
2. ``index_prompt``    -> an analyst rates how each tracked index moved.
3. ``market_prompt``   -> synthesize the new market state for the next step.

A recurring instruction across all three: reason *forward* from the current
market state, and do NOT reach for the historically known outcome of any
real-world policy. This is the core methodological guardrail from the proposal
(agents shouldn't just recreate outcomes memorized during training).
"""

from .models import EconomicIndex, Persona, PersonaDecision, Policy, SimulationConfig

_GUARDRAIL = (
    "Reason forward from the market state described below. Do not assume the "
    "real-world historical outcome of any similar policy; treat this scenario as "
    "novel and decide based only on the conditions presented."
)


def _indices_block(indices: list[EconomicIndex]) -> str:
    if not indices:
        return "(none specified)"
    return "\n".join(f"- {ix.name}: {ix.description or 'no description'}" for ix in indices)


def _shared_block(prior: list[PersonaDecision]) -> str:
    if not prior:
        return ""
    lines = "\n".join(f"- {d.persona_name}: {d.decision}" for d in prior)
    return (
        "\nEarlier this step, other actors have already decided:\n"
        f"{lines}\n"
        "You may react to their decisions.\n"
    )


def persona_system(persona: Persona) -> str:
    return (
        f"You are role-playing an economic actor in a market simulation.\n"
        f"ACTOR: {persona.name}\n"
        f"PROFILE: {persona.description or 'No additional profile.'}\n\n"
        "Stay fully in character. Make concrete, self-interested economic "
        "decisions consistent with this actor's situation and incentives. "
        f"{_GUARDRAIL}"
    )


def persona_user(
    config: SimulationConfig,
    market_state: str,
    step_number: int,
    step_date: str,
    injection: str,
    prior_decisions: list[PersonaDecision],
) -> str:
    policy = config.policy
    parts = [
        f"DATE: {step_date} (step {step_number} of {config.total_steps()})",
        f"\nCURRENT MARKET STATE:\n{market_state or config.environment_description}",
        f"\nPOLICY IN EFFECT:\n{policy.description or '(no specific policy)'}",
    ]
    if policy.objectives:
        parts.append(f"\nThe policy's stated objectives are:\n{policy.objectives}")
    if injection:
        parts.append(f"\nNEW DEVELOPMENT THIS STEP:\n{injection}")
    if config.shared_decisions:
        parts.append(_shared_block(prior_decisions))
    parts.append(
        "\nIn 2-4 sentences, state the concrete action(s) you take this step and "
        "your reasoning. Speak in the first person."
    )
    return "\n".join(parts)


def index_system() -> str:
    return (
        "You are a neutral macroeconomic analyst. Given a set of actor decisions "
        "and the market state, judge the DIRECTION each tracked indicator moves "
        "this step. Report direction and rough magnitude only — never invent "
        "absolute numbers. Use \"undetermined\" only when the evidence genuinely "
        f"does not support an up/down/flat call. {_GUARDRAIL}"
    )


def index_prompt(
    indices: list[EconomicIndex],
    market_state: str,
    decisions: list[PersonaDecision],
    step_date: str,
) -> str:
    decision_lines = "\n".join(f"- {d.persona_name}: {d.decision}" for d in decisions) or "(no decisions)"
    schema = (
        '{"readings": [{"index_name": "<exact name>", '
        '"direction": "up|down|flat|undetermined", '
        '"magnitude": "slight|moderate|strong|undetermined", '
        '"rationale": "<one sentence>"}]}'
    )
    return (
        f"DATE: {step_date}\n\n"
        f"MARKET STATE:\n{market_state}\n\n"
        f"ACTOR DECISIONS THIS STEP:\n{decision_lines}\n\n"
        f"INDICATORS TO ASSESS:\n{_indices_block(indices)}\n\n"
        "For EVERY indicator above, decide whether it moved up, down, stayed "
        "flat, or is undetermined this step, with a magnitude and a one-sentence "
        'rationale. If the direction is "undetermined", set the magnitude to '
        '"undetermined" as well.\n'
        "Respond with ONLY a JSON object, no prose, matching exactly:\n"
        f"{schema}"
    )


def market_system() -> str:
    return (
        "You are the simulation's market narrator. Produce a concise, updated "
        "description of the market state after this step's events. Be concrete "
        f"about conditions; do not editorialize about long-run outcomes. {_GUARDRAIL}"
    )


def market_prompt(
    previous_state: str,
    decisions: list[PersonaDecision],
    injection: str,
    step_date: str,
) -> str:
    decision_lines = "\n".join(f"- {d.persona_name}: {d.decision}" for d in decisions) or "(no decisions)"
    extra = f"\nEXTERNAL DEVELOPMENT THIS STEP:\n{injection}\n" if injection else ""
    return (
        f"DATE: {step_date}\n\n"
        f"PREVIOUS MARKET STATE:\n{previous_state}\n\n"
        f"ACTOR DECISIONS THIS STEP:\n{decision_lines}\n"
        f"{extra}\n"
        "Write the updated market state in 3-5 sentences, reflecting how these "
        "decisions and developments changed conditions."
    )
