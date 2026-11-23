"""The Anthropic provider builds the right request and reads the response safely (no network)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from evalgate.prompts import BOT_SCHEMA, JUDGE_SCHEMA
from evalgate.providers import (
    AnthropicStructured,
    ClaudeCLIStructured,
    CodexCLIStructured,
    DEFAULT_CODEX_MODEL,
    ProviderError,
    make_provider,
)


def _response(stop_reason="end_turn", data=None, model="claude-opus-5-5", iterations=None):
    content = [SimpleNamespace(type="thinking", thinking="")]
    if data is not None:
        content.append(SimpleNamespace(type="text", text=json.dumps(data)))
    usage = SimpleNamespace(input_tokens=500, output_tokens=60, iterations=iterations)
    return SimpleNamespace(stop_reason=stop_reason, content=content, model=model, usage=usage)


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _client(response):
    return SimpleNamespace(messages=FakeMessages(response), beta=SimpleNamespace(messages=FakeMessages(response)))


DATA = {"answer": "$49", "citations": ["pricing-1"], "declined": False}


def test_opus_uses_server_side_fallback_and_structured_output():
    client = _client(_response(data=DATA))
    bot = AnthropicStructured("claude-opus-5-5", effort="low", fallbacks="default", client=client)
    out = bot.call(bot.payload("sys", "user", BOT_SCHEMA))
    sent = client.beta.messages.kwargs
    assert sent["betas"] == ["server-side-fallback-2026-07-01"] and sent["fallbacks"] == "default"
    assert sent["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": BOT_SCHEMA}}
    assert "temperature" not in sent and "thinking" not in sent
    assert out["data"] == DATA and out["fallback_ran"] is False
    assert client.messages.kwargs is None


def test_haiku_has_no_fallback_and_uses_plain_endpoint():
    client = _client(_response(data=DATA, model="claude-haiku-5-5"))
    bot = AnthropicStructured("claude-haiku-5-5", fallbacks="default", client=client)
    payload = bot.payload("sys", "user", BOT_SCHEMA)
    assert payload["fallbacks"] == ""
    bot.call(payload)
    assert client.messages.kwargs is not None and client.beta.messages.kwargs is None


def test_refusal_is_recorded_not_parsed():
    client = _client(_response(stop_reason="refusal"))
    bot = AnthropicStructured("claude-opus-5-5", client=client)
    out = bot.call(bot.payload("sys", "user", BOT_SCHEMA))
    assert out["refused"] is True and out["data"] is None


def test_fallback_answers_are_flagged():
    iterations = [SimpleNamespace(type="message"), SimpleNamespace(type="fallback_message")]
    client = _client(_response(data=DATA, model="claude-opus-5", iterations=iterations))
    bot = AnthropicStructured("claude-opus-5-5", fallbacks="default", client=client)
    out = bot.call(bot.payload("sys", "user", BOT_SCHEMA))
    assert out["fallback_ran"] is True and out["served_model"] == "claude-opus-5"


def test_claude_cli_reads_structured_output_and_model_usage(monkeypatch):
    seen_cmd = []

    def fake_run(cmd, **kwargs):
        seen_cmd[:] = cmd
        envelope = {
            "is_error": False,
            "subtype": "success",
            "structured_output": DATA,
            "usage": {"input_tokens": 2, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 18, "output_tokens": 30},
            "modelUsage": {"claude-opus-5-5-20260701": {"inputTokens": 120}},
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr("evalgate.providers.subprocess.run", fake_run)
    bot = ClaudeCLIStructured("claude-opus-5-5", effort="low")
    out = bot.call(bot.payload("sys", "user", BOT_SCHEMA))
    assert "--output-format" in seen_cmd and "--json-schema" in seen_cmd
    assert "--setting-sources" in seen_cmd
    assert out["data"] == DATA
    assert out["served_model"] == "claude-opus-5-5-20260701"
    assert out["usage"] == {"input_tokens": 120, "output_tokens": 30}


def test_claude_cli_falls_back_to_fenced_result_and_surfaces_errors(monkeypatch):
    def fake_fenced(cmd, **kwargs):
        envelope = {"is_error": False, "result": f"```json\n{json.dumps(DATA)}\n```"}
        return SimpleNamespace(returncode=0, stdout=json.dumps(envelope), stderr="")

    monkeypatch.setattr("evalgate.providers.subprocess.run", fake_fenced)
    bot = ClaudeCLIStructured("claude-opus-5-5")
    assert bot.call(bot.payload("sys", "user", BOT_SCHEMA))["data"] == DATA

    def fake_err(cmd, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps({"is_error": True, "result": "quota exceeded"}), stderr="")

    monkeypatch.setattr("evalgate.providers.subprocess.run", fake_err)
    with pytest.raises(ProviderError, match="quota exceeded"):
        bot.call(bot.payload("sys", "user", BOT_SCHEMA))

    def fake_nonzero_stdout(cmd, **kwargs):
        return SimpleNamespace(returncode=1, stdout=json.dumps({"is_error": True, "result": "Not logged in"}), stderr="")

    monkeypatch.setattr("evalgate.providers.subprocess.run", fake_nonzero_stdout)
    with pytest.raises(ProviderError, match="Not logged in"):
        bot.call(bot.payload("sys", "user", BOT_SCHEMA))


def test_codex_cli_ignores_user_config_pins_model_and_reads_output(monkeypatch):
    seen_cmd = []

    def fake_run(cmd, **kwargs):
        seen_cmd[:] = cmd
        out_path = Path(cmd[cmd.index("-o") + 1])
        out_path.write_text('```json\n{"verdict": "correct", "reason": "Matches."}\n```', encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("evalgate.providers.subprocess.run", fake_run)
    judge = make_provider({"provider": "codex-cli", "model": "", "effort": "medium"})
    assert isinstance(judge, CodexCLIStructured)
    assert judge.model == DEFAULT_CODEX_MODEL
    out = judge.call(judge.payload("", "grade this", JUDGE_SCHEMA))
    assert "--ignore-user-config" in seen_cmd
    assert seen_cmd[seen_cmd.index("-m") + 1] == DEFAULT_CODEX_MODEL
    assert out["served_model"] == DEFAULT_CODEX_MODEL
    assert out["data"] == {"verdict": "correct", "reason": "Matches."}


def test_codex_cli_raises_when_no_output_file(monkeypatch):
    monkeypatch.setattr(
        "evalgate.providers.subprocess.run",
        lambda cmd, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    judge = CodexCLIStructured(model="gpt-6.2-sol")
    with pytest.raises(ProviderError, match="no final message"):
        judge.call(judge.payload("", "grade this", JUDGE_SCHEMA))
