---
description: Add a new LLM provider to the provider-agnostic layer
argument-hint: <provider, e.g. openai or gemini>
---

Add support for the LLM provider: **$ARGUMENTS**

Follow the project's provider pattern (see `market_sim/llm.py` and `CLAUDE.md`).
Do NOT modify `app.py` or `engine.py` — providers surface automatically through the
`PROVIDERS` registry once their SDK is importable.

1. If the SDK isn't a dependency yet, add it with `uv add <sdk-package>`. Import it
   **lazily** inside the client's `__init__` so `llm.py` still imports without it.
2. Write an `LLMClient` subclass implementing `stream_text`, `complete`, and
   `structured_output`.
3. Add a `Provider` entry to `PROVIDERS`: `name`, `label`, `key_env`, `sdk_module`,
   `client_cls`, and a `models` dict (UI label -> model id). **Verify the model IDs
   against the provider's current model list** — do not trust placeholder IDs.
4. Add the provider's API key env var to `.env.example` (commented).
5. Extend `tests/test_llm.py`: cover the new entry via `available_models` /
   `provider_for_model`, and a faked-SDK `structured_output` test if practical.
6. Run the verification loop: `uv run ruff check . && uv run pyright && uv run pytest`.
