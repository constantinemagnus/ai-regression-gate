"""Model providers for the bot and the judge.

Every provider has the same two-step shape so the cache can sit in between:
``payload(...)`` builds a JSON-serialisable request (the cache key) and
``call(payload)`` performs it and returns a plain dict.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

# Models that accept the server-side refusal fallback (beta). Claude Haiku 5.5 has none.
FALLBACK_CAPABLE = frozenset(
    {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1"}
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"

FAMILY = {"anthropic": "anthropic", "claude-cli": "anthropic", "codex-cli": "openai"}
DEFAULT_CODEX_MODEL = "gpt-6.2-sol"


class ProviderError(RuntimeError):
    pass


def _parse_json_text(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"model returned non-JSON text: {text[:300]!r}") from exc
    if not isinstance(parsed, dict):
        raise ProviderError(f"expected JSON object, got {type(parsed).__name__}")
    return parsed


# --------------------------------------------------------------------------- Anthropic API


class AnthropicStructured:
    """One structured-output call to the Claude API through the Anthropic SDK."""

    name = "anthropic"

    def __init__(self, model: str, effort: str = "low", fallbacks: str = "", max_tokens: int = 4000, client: Any = None):
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks if model in FALLBACK_CAPABLE else ""
        self.max_tokens = max_tokens
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import anthropic  # imported lazily so offline replays need no SDK

            self._client = anthropic.Anthropic()
        return self._client

    def payload(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            "fallbacks": self.fallbacks,
        }

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        kwargs = {k: payload[k] for k in ("model", "max_tokens", "system", "messages", "output_config")}
        if payload["fallbacks"]:
            response = self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks=payload["fallbacks"], **kwargs
            )
        else:
            response = self.client.messages.create(**kwargs)

        iterations = getattr(response.usage, "iterations", None) or []
        result: dict[str, Any] = {
            "served_model": response.model,
            "fallback_ran": any(getattr(it, "type", None) == "fallback_message" for it in iterations),
            "stop_reason": response.stop_reason,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            "refused": response.stop_reason == "refusal",
            "data": None,
        }
        if result["refused"] or response.stop_reason == "max_tokens":
            return result
        text = next((block.text for block in response.content if block.type == "text"), "")
        result["data"] = _parse_json_text(text)
        return result


# --------------------------------------------------------------------------- Claude Code CLI


class ClaudeCLIStructured:
    """Runs `claude -p` on the user's Claude subscription with no tools."""

    name = "claude-cli"

    def __init__(self, model: str, effort: str = "low", timeout: int = 300, binary: str = "claude", **_: Any):
        self.model = model
        self.effort = effort
        self.timeout = timeout
        self.binary = binary

    def payload(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "effort": self.effort,
                "setting_sources": "", "system": system, "user": user, "schema": schema}

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        cmd = [
            self.binary, "-p",
            "--output-format", "json",
            "--json-schema", json.dumps(payload["schema"]),
            "--model", payload["model"],
            "--effort", payload["effort"],
            "--system-prompt", payload["system"],
            "--tools", "",
            "--setting-sources", "",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--disable-slash-commands",
        ]
        with tempfile.TemporaryDirectory() as workdir:
            proc = subprocess.run(cmd, input=payload["user"], capture_output=True, text=True,
                                  timeout=self.timeout, cwd=workdir)
        if proc.returncode != 0:
            detail = proc.stderr.strip()
            if not detail and proc.stdout.strip():
                try:
                    err_env = json.loads(proc.stdout)
                    detail = str(err_env.get("result") or proc.stdout.strip())
                except json.JSONDecodeError:
                    detail = proc.stdout.strip()
            raise ProviderError(f"claude exited {proc.returncode}: {detail[:500]}")
        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"claude returned non-JSON output: {proc.stdout[:300]!r}") from exc
        if envelope.get("is_error"):
            raise ProviderError(f"claude reported an error: {str(envelope.get('result'))[:500]}")
        data = envelope.get("structured_output")
        if data is None:
            data = _parse_json_text(envelope.get("result", ""))
        elif not isinstance(data, dict):
            raise ProviderError(f"claude structured_output must be an object, got {type(data).__name__}")
        usage = envelope.get("usage") or {}
        total_input_tokens = (
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
        )
        model_usage = envelope.get("modelUsage")
        served_model = payload["model"]
        if isinstance(model_usage, dict) and model_usage:
            served_model = next(iter(model_usage))
        elif isinstance(envelope.get("model"), str) and envelope["model"]:
            served_model = envelope["model"]
        return {
            "served_model": served_model,
            "fallback_ran": False,
            "stop_reason": envelope.get("subtype", "success"),
            "usage": {"input_tokens": total_input_tokens, "output_tokens": int(usage.get("output_tokens", 0))},
            "refused": False,
            "data": data,
        }


# --------------------------------------------------------------------------- Codex CLI


class CodexCLIStructured:
    """Runs `codex exec` read-only in an empty folder; used as the cross-family judge."""

    name = "codex-cli"

    def __init__(self, model: str = DEFAULT_CODEX_MODEL, effort: str = "", timeout: int = 600, binary: str = "codex", **_: Any):
        self.model = model or DEFAULT_CODEX_MODEL
        self.effort = effort
        self.timeout = timeout
        self.binary = binary

    def payload(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        prompt = f"{system}\n\n{user}" if system else user
        return {"provider": self.name, "model": self.model, "effort": self.effort,
                "ignore_user_config": True, "prompt": prompt, "schema": schema}

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as workdir:
            schema_path = Path(workdir) / "schema.json"
            out_path = Path(workdir) / "answer.json"
            schema_path.write_text(json.dumps(payload["schema"]), encoding="utf-8")
            cmd = [self.binary, "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
                   "-s", "read-only", "--output-schema", str(schema_path), "-o", str(out_path)]
            if payload["model"]:
                cmd += ["-m", payload["model"]]
            if payload["effort"]:
                cmd += ["-c", f'model_reasoning_effort="{payload["effort"]}"']
            cmd.append("-")
            proc = subprocess.run(cmd, input=payload["prompt"], capture_output=True, text=True,
                                  timeout=self.timeout, cwd=workdir)
            if proc.returncode != 0:
                raise ProviderError(f"codex exited {proc.returncode}: {proc.stderr.strip()[-500:]}")
            if not out_path.exists():
                raise ProviderError("codex produced no final message")
            data = _parse_json_text(out_path.read_text(encoding="utf-8"))
        return {"served_model": payload["model"], "fallback_ran": False,
                "stop_reason": "end_turn", "usage": {}, "refused": False, "data": data}


def make_provider(section: dict[str, Any]) -> Any:
    provider = section.get("provider", "anthropic")
    options = {k: v for k, v in section.items() if k != "provider"}
    if provider == "anthropic":
        return AnthropicStructured(model=options.get("model") or "claude-opus-5-5",
                                   effort=options.get("effort", "low"),
                                   fallbacks=options.get("fallbacks", ""),
                                   max_tokens=int(options.get("max_tokens", 4000)))
    if provider == "claude-cli":
        return ClaudeCLIStructured(model=options.get("model") or "claude-opus-5-5", effort=options.get("effort", "low"))
    if provider == "codex-cli":
        return CodexCLIStructured(model=options.get("model") or DEFAULT_CODEX_MODEL, effort=options.get("effort", ""))
    raise ValueError(f"unknown provider {provider!r}")
