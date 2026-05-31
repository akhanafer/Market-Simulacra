"""Tests for the provider-agnostic LLM layer: the registry helpers and the
Anthropic adapter's structured_output (with the SDK client faked, no network)."""

import pytest

from market_sim import llm
from market_sim.models import IndexAssessment, IndexAssessmentBatch


def test_available_models_lists_only_installed_providers():
    models = llm.available_models()
    # Anthropic and OpenAI SDKs are installed -> their models appear, provider-labelled.
    assert any(label.startswith("Anthropic ·") for label in models)
    assert "claude-sonnet-4-6" in models.values()
    assert any(label.startswith("OpenAI ·") for label in models)
    assert "gpt-4.1" in models.values()
    # google-genai is not installed here -> Gemini stays hidden.
    assert not any(label.startswith("Gemini ·") for label in models)


def test_provider_for_model():
    provider = llm.provider_for_model("claude-opus-4-8")
    assert provider is not None
    assert provider.name == "anthropic"
    assert llm.provider_for_model("nope-404") is None


def test_resolve_default_key_reads_provider_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert llm.resolve_default_key("claude-opus-4-8") == "sk-test"
    assert llm.resolve_default_key("unknown-model") == ""


def test_build_client_unknown_model_raises():
    with pytest.raises(ValueError):
        llm.build_client("nope-404", "sk-x")


def test_build_client_empty_key_raises():
    with pytest.raises(ValueError):
        llm.build_client("claude-opus-4-8", "")


def test_validate_key_with_no_key():
    ok, msg = llm.validate_key("claude-opus-4-8", "")
    assert ok is False
    assert "No API key" in msg


class _FakeResp:
    def __init__(self, parsed_output):
        self.parsed_output = parsed_output
        self.stop_reason = "end_turn"


class _FakeMessages:
    def __init__(self, resp):
        self._resp = resp

    def parse(self, **kwargs):
        return self._resp


def _anthropic_client_with(resp) -> llm.AnthropicClient:
    client = llm.build_client("claude-opus-4-8", "sk-test")
    assert isinstance(client, llm.AnthropicClient)
    # Swap the real SDK client for a fake exposing just .messages.parse.
    client._client = type("FakeSDK", (), {"messages": _FakeMessages(resp)})()  # type: ignore[assignment]
    return client


def test_anthropic_structured_output_returns_parsed():
    batch = IndexAssessmentBatch(
        readings=[IndexAssessment(index_name="CPI", direction="up", magnitude="slight", rationale="a")]
    )
    client = _anthropic_client_with(_FakeResp(batch))
    assert client.structured_output(IndexAssessmentBatch, "sys", "user") is batch


def test_anthropic_structured_output_raises_when_empty():
    client = _anthropic_client_with(_FakeResp(None))
    with pytest.raises(RuntimeError):
        client.structured_output(IndexAssessmentBatch, "sys", "user")


# --- OpenAI adapter (SDK client faked, no network) -------------------------


class _FakeChatResp:
    """Mimics chat.completions.parse: resp.choices[0].message.parsed."""

    def __init__(self, parsed):
        message = type("Msg", (), {"parsed": parsed, "content": "ok"})()
        self.choices = [type("Choice", (), {"message": message})()]


class _FakeCompletions:
    def __init__(self, resp):
        self._resp = resp
        self.kwargs: dict = {}

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return self._resp


def _openai_client_with(resp) -> llm.OpenAIClient:
    client = llm.build_client("gpt-4.1", "sk-test")
    assert isinstance(client, llm.OpenAIClient)
    completions = _FakeCompletions(resp)
    # Swap the real SDK client for a fake exposing chat.completions.parse.
    chat = type("Chat", (), {"completions": completions})()
    client._client = type("FakeSDK", (), {"chat": chat})()  # type: ignore[assignment]
    return client


def test_openai_structured_output_returns_parsed():
    batch = IndexAssessmentBatch(
        readings=[IndexAssessment(index_name="CPI", direction="down", magnitude="moderate", rationale="b")]
    )
    client = _openai_client_with(_FakeChatResp(batch))
    assert client.structured_output(IndexAssessmentBatch, "sys", "user", temperature=0.0) is batch


def test_openai_structured_output_raises_when_empty():
    client = _openai_client_with(_FakeChatResp(None))
    with pytest.raises(RuntimeError):
        client.structured_output(IndexAssessmentBatch, "sys", "user")


def test_openai_maps_max_tokens_to_max_completion_tokens():
    client = _openai_client_with(_FakeChatResp(IndexAssessmentBatch(readings=[])))
    client.structured_output(IndexAssessmentBatch, "sys", "user", max_tokens=42)
    # OpenAI deprecated max_tokens for chat completions; we must send the new name.
    sent = client._client.chat.completions.kwargs  # type: ignore[attr-defined]
    assert sent["max_completion_tokens"] == 42
    assert "max_tokens" not in sent
