"""Provider-agnostic LLM client.

The rest of the app only ever calls two methods on a client:

* ``stream_text``  — a generator of text deltas, for live ``st.write_stream`` UI.
* ``complete``     — the full text response, for structured (JSON) calls.

Each provider (Anthropic, OpenAI, Gemini, ...) supplies a concrete ``LLMClient``
subclass implementing those two methods. Providers are listed in ``PROVIDERS``;
a provider only shows up in the UI if its SDK is importable, so adding one later
is essentially: ``uv add <sdk>`` + a small block in ``PROVIDERS``.

SDK imports are done lazily inside each client's ``__init__`` so this module
imports fine even when only ``anthropic`` is installed.
"""

import abc
import importlib.util
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

from pydantic import BaseModel

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam

# A Pydantic model type used as a structured-output schema, returned as an instance.
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClient(abc.ABC):
    """Common interface every provider adapter implements."""

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("No API key provided.")
        self.api_key = api_key
        self.model = model

    @abc.abstractmethod
    def stream_text(
        self, system: str, user: str, max_tokens: int = 1024, temperature: float = 1.0
    ) -> Iterator[str]:
        """Yield text deltas. Suitable for ``st.write_stream``."""

    @abc.abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024, temperature: float = 1.0) -> str:
        """Return the full text response (non-streaming)."""

    @abc.abstractmethod
    def structured_output(
        self,
        schema: type[SchemaT],
        system: str,
        user: str,
        max_tokens: int = 1024,
        temperature: float = 1.0,
    ) -> SchemaT:
        """Call the model with structured output enabled and return a parsed,
        schema-validated instance of ``schema`` (a Pydantic model)."""


# --------------------------------------------------------------------------- adapters


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)

    def stream_text(self, system, user, max_tokens=1024, temperature=1.0):
        with self._client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        ) as stream:
            yield from stream.text_stream

    def complete(self, system, user, max_tokens=1024, temperature=1.0):
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in msg.content if block.type == "text")

    def structured_output(self, schema, system, user, max_tokens=1024, temperature=1.0):
        resp = self._client.messages.parse(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=schema,
        )
        if resp.parsed_output is None:
            raise RuntimeError(f"Model returned no structured output (stop reason: {resp.stop_reason}).")
        return resp.parsed_output


class OpenAIClient(LLMClient):
    """Adapter for the OpenAI Python SDK (``uv add openai``), on the Chat
    Completions API.

    ``max_tokens`` is the cross-provider interface name; OpenAI deprecated it for
    chat completions in favour of ``max_completion_tokens``, so we map to that.
    Structured output uses ``chat.completions.parse`` with a Pydantic
    ``response_format`` (OpenAI's native JSON-schema mode).
    """

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)

    @staticmethod
    def _messages(system: str, user: str) -> "list[ChatCompletionMessageParam]":
        return cast(
            "list[ChatCompletionMessageParam]",
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

    def stream_text(self, system, user, max_tokens=1024, temperature=1.0):
        stream = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            messages=self._messages(system, user),
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def complete(self, system, user, max_tokens=1024, temperature=1.0):
        resp = self._client.chat.completions.create(
            model=self.model,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            messages=self._messages(system, user),
        )
        return resp.choices[0].message.content or ""

    def structured_output(self, schema, system, user, max_tokens=1024, temperature=1.0):
        resp = self._client.chat.completions.parse(
            model=self.model,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            messages=self._messages(system, user),
            response_format=schema,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            raise RuntimeError("Model returned no structured output.")
        return parsed


class GeminiClient(LLMClient):
    """Adapter for the Google Gen AI SDK (``uv add google-genai``).

    Written to the ``google-genai`` (``from google import genai``) API; not
    live-tested in this repo. Verify the model IDs in ``PROVIDERS`` before use.
    """

    def __init__(self, api_key: str, model: str):
        super().__init__(api_key, model)
        from google import genai  # pyright: ignore[reportAttributeAccessIssue]

        self._client = genai.Client(api_key=api_key)

    def _config(self, system: str, max_tokens: int, temperature: float):
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

    def stream_text(self, system, user, max_tokens=1024, temperature=1.0):
        stream = self._client.models.generate_content_stream(
            model=self.model,
            contents=user,
            config=self._config(system, max_tokens, temperature),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def complete(self, system, user, max_tokens=1024, temperature=1.0):
        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=self._config(system, max_tokens, temperature),
        )
        return resp.text or ""

    def structured_output(self, schema, system, user, max_tokens=1024, temperature=1.0):
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        if resp.parsed is None:
            raise RuntimeError("Model returned no structured output.")
        return resp.parsed


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class Provider:
    name: str  # internal key, e.g. "anthropic"
    label: str  # UI label, e.g. "Anthropic"
    key_env: str  # env var holding the API key
    sdk_module: str  # import name used to check SDK availability
    client_cls: type[LLMClient]
    models: dict[str, str]  # UI label -> API model id


# To add a provider: install its SDK, write an LLMClient subclass above, and add
# an entry here. It appears in the UI automatically once its SDK is importable.
# Anthropic and OpenAI are implemented and enabled; Gemini's adapter is wired but
# its model IDs are still placeholders to verify against the current model list.
PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        name="anthropic",
        label="Anthropic",
        key_env="ANTHROPIC_API_KEY",
        sdk_module="anthropic",
        client_cls=AnthropicClient,
        models={
            "Claude Opus 4.8": "claude-opus-4-8",
            "Claude Sonnet 4.6": "claude-sonnet-4-6",
            "Claude Haiku 4.5": "claude-haiku-4-5-20251001",
        },
    ),
    "openai": Provider(
        name="openai",
        label="OpenAI",
        key_env="OPENAI_API_KEY",
        sdk_module="openai",
        client_cls=OpenAIClient,
        models={
            "GPT-4.1": "gpt-4.1",
            "GPT-4.1 mini": "gpt-4.1-mini",
            "GPT-4o": "gpt-4o",
        },
    ),
    "gemini": Provider(
        name="gemini",
        label="Gemini",
        key_env="GEMINI_API_KEY",
        sdk_module="google.genai",
        client_cls=GeminiClient,
        models={
            "Gemini 2.5 Pro": "gemini-2.5-pro",
            "Gemini 2.5 Flash": "gemini-2.5-flash",
        },
    ),
}

# The model selected by default when a new config is created.
DEFAULT_MODEL = "claude-sonnet-4-6"


def _sdk_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


def available_models() -> dict[str, str]:
    """``{"Anthropic · Claude Sonnet 4.6": "claude-sonnet-4-6", ...}`` for the UI.

    Only includes providers whose SDK is installed, so hidden providers can stay
    registered without breaking model selection.
    """
    out: dict[str, str] = {}
    for p in PROVIDERS.values():
        if not _sdk_available(p.sdk_module):
            continue
        for label, model_id in p.models.items():
            out[f"{p.label} · {label}"] = model_id
    return out


def provider_for_model(model_id: str) -> Provider | None:
    """Which provider owns a given model id (model ids are unique across providers)."""
    for p in PROVIDERS.values():
        if model_id in p.models.values():
            return p
    return None


def resolve_default_key(model_id: str) -> str:
    """API key from the selected model's provider env var (.env loaded by app.py)."""
    provider = provider_for_model(model_id)
    return os.environ.get(provider.key_env, "") if provider else ""


def build_client(model_id: str, api_key: str) -> LLMClient:
    """Construct the right provider client for ``model_id``."""
    provider = provider_for_model(model_id)
    if provider is None:
        raise ValueError(f"Unknown model '{model_id}'.")
    return provider.client_cls(api_key=api_key, model=model_id)


def validate_key(model_id: str, api_key: str) -> tuple[bool, str]:
    """Cheap check that a key works for the selected model. Returns (ok, message)."""
    if not api_key:
        return False, "No API key set."
    try:
        build_client(model_id, api_key).complete(
            system="You are a connectivity test.",
            user="Reply with: ok",
            max_tokens=8,
            temperature=0,
        )
        return True, "Key OK."
    except Exception as exc:  # noqa: BLE001 - surface any auth/transport error to UI
        return False, f"{type(exc).__name__}: {exc}"
