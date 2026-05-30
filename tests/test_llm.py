"""Tests for the provider-agnostic LLM layer: the registry helpers and the
Anthropic adapter's structured_output (with the SDK client faked, no network)."""

import pytest

from market_sim import llm
from market_sim.models import IndexAssessment, IndexAssessmentBatch


def test_available_models_lists_only_installed_providers():
    models = llm.available_models()
    # Anthropic SDK is installed -> its models appear, labelled by provider.
    assert any(label.startswith("Anthropic ·") for label in models)
    assert "claude-sonnet-4-6" in models.values()
    # openai / google-genai are not installed here -> those providers stay hidden.
    assert not any(label.startswith(("OpenAI ·", "Gemini ·")) for label in models)


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
