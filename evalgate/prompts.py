"""Prompt templates and output schemas for the bot and the judge."""

from __future__ import annotations

from pathlib import Path

from .cases import Case
from .corpus import Passage

BOT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "declined": {"type": "boolean"},
    },
    "required": ["answer", "citations", "declined"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "incorrect"]},
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}


def load_prompt(prompts_dir: Path, name: str) -> str:
    return (Path(prompts_dir) / name).read_text(encoding="utf-8").strip()


def bot_user_message(question: str, passages: list[Passage]) -> str:
    docs = "\n\n".join(p.render() for p in passages)
    return f"Documentation passages:\n\n{docs}\n\nCustomer question: {question}"


def judge_message(template: str, case: Case, passages: list[Passage], answer: str) -> str:
    return template.format(
        question=case.question,
        passages="\n\n".join(p.render() for p in passages),
        reference=case.reference,
        answer=answer,
    )
