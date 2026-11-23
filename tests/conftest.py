"""Fake providers so the whole pipeline can be tested without any model calls."""

from __future__ import annotations

import re
from typing import Any

import pytest

from evalgate import common
from evalgate.cases import load_cases
from evalgate.checks import normalize
from evalgate.corpus import load_passages
from evalgate.prompts import load_prompt
from evalgate.runner import Suite

QUESTION = re.compile(r"Customer question: (.*)$", re.S)


@pytest.fixture(scope="session")
def passages():
    return load_passages(common.DOCS_DIR)


@pytest.fixture(scope="session")
def cases(passages):
    return load_cases(common.CASES_PATH, {p.id for p in passages})


@pytest.fixture
def suite(cases, passages):
    return Suite(cases, passages, load_prompt(common.PROMPTS_DIR, "bot_system.md"),
                 load_prompt(common.PROMPTS_DIR, "judge.md"))


class FakeBot:
    """Answers with the reference answer and gold citations; declines what the docs don't cover.

    `bluff`, `fake_cite` and `wrong` make it misbehave on chosen case ids, the way a
    cheaper model might.
    """

    name = "fake"

    def __init__(self, cases, model="fake-strong", bluff=(), fake_cite=(), wrong=()):
        self.by_question = {c.question: c for c in cases}
        self.model = model
        self.bluff, self.fake_cite, self.wrong = set(bluff), set(fake_cite), set(wrong)
        self.calls = 0

    def payload(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "system": system, "user": user}

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        case = self.by_question[QUESTION.search(payload["user"]).group(1).strip()]
        if case.id in self.bluff:
            data = {"answer": "Yes, that costs $5 per month.", "citations": ["pricing-1"], "declined": False}
        elif case.expects_decline:
            data = {"answer": "The documentation doesn't cover that.", "citations": [], "declined": True}
        elif case.id in self.fake_cite:
            data = {"answer": case.reference, "citations": ["pricing-99"], "declined": False}
        elif case.id in self.wrong:
            data = {"answer": "It is $1,000.", "citations": list(case.gold), "declined": False}
        else:
            data = {"answer": case.reference, "citations": list(case.gold), "declined": False}
        return {"served_model": self.model, "fallback_ran": False, "stop_reason": "end_turn",
                "usage": {"input_tokens": 100, "output_tokens": 20}, "refused": False, "data": data}


class FakeJudge:
    """Correct when the answer contains every expected fact of the case it is judging."""

    name = "fake-judge"

    def __init__(self, cases):
        self.cases = cases
        self.calls = 0

    def payload(self, system: str, user: str, schema: dict[str, Any]) -> dict[str, Any]:
        return {"provider": self.name, "model": "fake-judge", "prompt": user}

    def call(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        prompt = payload["prompt"]
        case = next(c for c in self.cases if f"Question:\n{c.question}\n" in prompt)
        answer = normalize(prompt.split("Bot answer:\n", 1)[1])
        ok = all(any(normalize(v) in answer for v in group) for group in case.facts) and "$1,000" not in answer
        return {"served_model": "fake-judge", "fallback_ran": False, "stop_reason": "end_turn", "usage": {},
                "refused": False, "data": {"verdict": "correct" if ok else "incorrect", "reason": "fake"}}
