"""Code checks that need no model: declining, citations and key facts.

Each check returns {"name", "passed", "detail", "tags"}. Tags are the short
failure labels used in reports, e.g. "made-up citation".
"""

from __future__ import annotations

import re
from typing import Any

from .cases import Case

DASHES = str.maketrans({"–": "-", "—": "-", "−": "-", " ": " ", "’": "'", "‘": "'"})


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(DASHES).lower()).strip()


def _fact_pattern(variant: str) -> re.Pattern[str]:
    v = normalize(variant)
    left = ""
    if v[:1].isdigit():
        left = r"(?<![0-9a-z])(?<![0-9][.,])"
    elif v[:1].isalnum():
        left = r"(?<![0-9a-z])"
    right = ""
    if v[-1:].isdigit():
        right = r"(?![0-9a-z]|,[0-9]{3}|\.(?:0*[1-9]))"
    elif v[-1:].isalnum():
        right = r"(?![0-9a-z])"
    return re.compile(left + re.escape(v) + right)


DECLINE_CUES = (
    "not covered",
    "doesn't cover",
    "does not cover",
    "don't cover",
    "do not cover",
    "doesn't mention",
    "does not mention",
    "don't mention",
    "do not mention",
    "no mention",
    "doesn't say",
    "does not say",
    "doesn't list",
    "does not list",
    "doesn't include",
    "does not include",
    "doesn't state",
    "does not state",
    "doesn't specify",
    "does not specify",
    "doesn't give",
    "does not give",
    "doesn't provide",
    "does not provide",
    "can't find",
    "cannot find",
    "don't have",
    "do not have",
    "no information",
    "not mentioned",
    "not specified",
    "not included",
    "not listed",
    "not in the documentation",
    "not in the provided",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "documentation doesn't",
    "documentation does not",
    "passages don't",
    "passages do not",
)


def _result(name: str, passed: bool, detail: str = "", tags: list[str] | None = None) -> dict[str, Any]:
    return {"name": name, "passed": passed, "detail": detail, "tags": tags or []}


def check_decline(case: Case, declined: bool) -> dict[str, Any]:
    if case.expects_decline and not declined:
        return _result("decline", False, "answered a question the docs don't cover", ["bluffed instead of declining"])
    if not case.expects_decline and declined:
        return _result("decline", False, "declined a question the docs answer", ["declined an answerable question"])
    return _result("decline", True)


def check_citations(case: Case, citations: list[str], retrieved_ids: list[str], corpus_ids: set[str]) -> dict[str, Any]:
    citations = [c.strip().strip("[]") for c in citations]
    if not citations:
        return _result("citations", False, "no citation given", ["missing citation"])
    made_up = [c for c in citations if c not in corpus_ids]
    not_shown = [c for c in citations if c in corpus_ids and c not in retrieved_ids]
    if made_up or not_shown:
        bad = made_up + not_shown
        return _result("citations", False, f"cited passages it was never shown: {bad}", ["made-up citation"])
    if not set(citations) & set(case.gold):
        return _result("citations", False, f"cited {citations}, expected one of {list(case.gold)}", ["wrong source cited"])
    if len(set(citations)) >= len(set(retrieved_ids)) and len(set(retrieved_ids)) > len(set(case.gold)):
        return _result("citations", False, f"cited every shown passage ({citations})", ["blanket citation"])
    return _result("citations", True)


def check_declined_citations(citations: list[str]) -> dict[str, Any]:
    cleaned = [c.strip().strip("[]") for c in citations if c.strip().strip("[]")]
    if cleaned:
        return _result("citations", False, f"declined but still cited {cleaned}", ["cited while declining"])
    return _result("citations", True)


def check_declined_answer(answer: str) -> dict[str, Any]:
    text = normalize(answer)
    if not text:
        return _result("decline_text", False, "declined with empty answer text", ["bad decline text"])
    if not any(cue in text for cue in DECLINE_CUES):
        return _result("decline_text", False, "declined flag set but answer text does not state it is not covered", ["bluffed while declining"])
    return _result("decline_text", True)


def check_facts(case: Case, answer: str) -> dict[str, Any] | None:
    if not case.facts:
        return None
    text = normalize(answer)
    missing = [group[0] for group in case.facts if not any(_fact_pattern(v).search(text) for v in group)]
    if missing:
        return _result("facts", False, f"missing {missing}", ["wrong or missing fact"])
    return _result("facts", True)
