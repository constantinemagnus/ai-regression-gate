"""The test set: questions, expected behaviour and reference answers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import read_jsonl

KINDS = ("answerable", "unanswerable", "tricky")


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    question: str
    gold: tuple[str, ...]
    reference: str
    facts: tuple[tuple[str, ...], ...]

    @property
    def expects_decline(self) -> bool:
        return self.kind == "unanswerable"


def load_cases(path: Path, passage_ids: set[str] | None = None) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        case = Case(
            id=row["id"],
            kind=row["kind"],
            question=row["question"],
            gold=tuple(row.get("gold", [])),
            reference=row["reference"],
            facts=tuple(tuple(group) for group in row.get("facts", [])),
        )
        if case.kind not in KINDS:
            raise ValueError(f"{case.id}: unknown kind {case.kind!r}")
        if case.id in seen:
            raise ValueError(f"duplicate case id {case.id!r}")
        seen.add(case.id)
        if case.expects_decline and case.gold:
            raise ValueError(f"{case.id}: unanswerable cases must have no gold passages")
        if not case.expects_decline and not case.gold:
            raise ValueError(f"{case.id}: answerable cases need at least one gold passage")
        if passage_ids is not None:
            unknown = set(case.gold) - passage_ids
            if unknown:
                raise ValueError(f"{case.id}: unknown gold passages {sorted(unknown)}")
        if any(not group for group in case.facts):
            raise ValueError(f"{case.id}: empty fact group")
        cases.append(case)
    return cases
