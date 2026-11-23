"""Run the test set against one configuration and score every answer."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .cache import OfflineMiss, ResponseCache
from .cases import Case
from .checks import check_citations, check_decline, check_declined_answer, check_declined_citations, check_facts
from .corpus import BM25Index, Passage
from .prompts import BOT_SCHEMA, JUDGE_SCHEMA, bot_user_message, judge_message
from .providers import FAMILY


@dataclass
class Suite:
    cases: list[Case]
    passages: list[Passage]
    bot_system: str
    judge_template: str
    top_k: int = 4
    index: BM25Index = field(init=False)
    by_id: dict[str, Passage] = field(init=False)

    def __post_init__(self) -> None:
        self.index = BM25Index(self.passages)
        self.by_id = {p.id: p for p in self.passages}


def _judge_passages(suite: Suite, case: Case, retrieved: list[Passage]) -> list[Passage]:
    seen, out = set(), []
    for p in [suite.by_id[g] for g in case.gold] + retrieved:
        if p.id not in seen:
            seen.add(p.id)
            out.append(p)
    return out


def _valid_bot_data(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("answer"), str)
        and isinstance(data.get("citations"), list)
        and all(isinstance(c, str) for c in data["citations"])
        and isinstance(data.get("declined"), bool)
    )


def _valid_bot_reply(reply: dict[str, Any]) -> bool:
    if reply.get("refused"):
        return True
    return _valid_bot_data(reply.get("data"))


def _valid_judge_reply(reply: dict[str, Any]) -> bool:
    verdict = reply.get("data")
    return (
        isinstance(verdict, dict)
        and verdict.get("verdict") in ("correct", "incorrect")
        and isinstance(verdict.get("reason"), str)
    )


def score_case(suite: Suite, case: Case, bot: Any, judge: Any, cache: ResponseCache) -> dict[str, Any]:
    retrieved = suite.index.search(case.question, suite.top_k)
    retrieved_ids = [p.id for p in retrieved]
    row: dict[str, Any] = {
        "case_id": case.id, "kind": case.kind, "question": case.question, "reference": case.reference,
        "retrieved": retrieved_ids, "checks": [], "judge": None, "error": None,
    }
    try:
        reply = cache.fetch(
            bot.payload(suite.bot_system, bot_user_message(case.question, retrieved), BOT_SCHEMA),
            bot.call,
            validate=_valid_bot_reply,
        )
    except OfflineMiss:
        raise
    except Exception as exc:  # a failed call is a failed case, not a crashed run
        row.update(error=f"bot: {type(exc).__name__}: {exc}", passed=False, tags=["error"])
        return row

    row.update(served_model=reply["served_model"], fallback_ran=reply["fallback_ran"],
               usage=reply.get("usage", {}), refused=reply["refused"], stop_reason=reply["stop_reason"])
    data = reply.get("data")
    if data is None:
        if reply["refused"]:
            row.update(answer="", citations=[], declined=False, passed=False, tags=["safety refusal"])
        else:
            err = f"bot: incomplete response ({reply['stop_reason']})"
            row.update(answer="", citations=[], declined=False, error=err, passed=False, tags=["error", f"no answer ({reply['stop_reason']})"])
        return row

    if not _valid_bot_data(data):
        row.update(answer="", citations=[], declined=False,
                   error=f"bot: malformed structured payload: {data!r}", passed=False, tags=["error"])
        return row
    answer = data["answer"]
    citations = list(data["citations"])
    declined = data["declined"]
    row.update(answer=answer, citations=citations, declined=declined)
    checks = [check_decline(case, declined)]
    if case.expects_decline and declined:
        checks.append(check_declined_citations(citations))
        checks.append(check_declined_answer(answer))
    if not case.expects_decline and not declined:
        checks.append(check_citations(case, citations, retrieved_ids, set(suite.by_id)))
        fact_check = check_facts(case, answer)
        if fact_check:
            checks.append(fact_check)
        try:
            prompt = judge_message(suite.judge_template, case, _judge_passages(suite, case, retrieved), answer)
            judge_reply = cache.fetch(judge.payload("", prompt, JUDGE_SCHEMA), judge.call, validate=_valid_judge_reply)
            if not _valid_judge_reply(judge_reply):
                raise ValueError(f"malformed judge payload (stop_reason={judge_reply.get('stop_reason')!r}): {judge_reply.get('data')!r}")
            verdict = judge_reply["data"]
            row["judge"] = verdict
            checks.append({"name": "judge", "passed": verdict["verdict"] == "correct",
                           "detail": verdict["reason"],
                           "tags": [] if verdict["verdict"] == "correct" else ["judge: incorrect"]})
        except OfflineMiss:
            raise
        except Exception as exc:
            row["error"] = f"judge: {type(exc).__name__}: {exc}"
            checks.append({"name": "judge", "passed": False, "detail": row["error"], "tags": ["error"]})

    row["checks"] = checks
    row["tags"] = sorted({t for c in checks for t in c["tags"]})
    row["passed"] = all(c["passed"] for c in checks)
    return row


def run_suite(suite: Suite, bot: Any, judge: Any, cache: ResponseCache, workers: int = 4,
              progress: Callable[[dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
    def work(case: Case) -> dict[str, Any]:
        row = score_case(suite, case, bot, judge, cache)
        if progress:
            progress(row)
        return row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(work, suite.cases))


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    tag_counts: dict[str, int] = {}
    for row in rows:
        for tag in row.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    by_kind: dict[str, dict[str, int]] = {}
    for row in rows:
        k = by_kind.setdefault(row["kind"], {"total": 0, "passed": 0})
        k["total"] += 1
        k["passed"] += int(row["passed"])
    usage = {"input_tokens": 0, "output_tokens": 0}
    for row in rows:
        for key in usage:
            usage[key] += int((row.get("usage") or {}).get(key, 0))
    bot_family = FAMILY.get(config["bot"]["provider"], "unknown")
    judge_family = FAMILY.get(config["judge"]["provider"], "unknown")
    return {
        "name": config["name"],
        "bot": config["bot"],
        "judge": config["judge"],
        "judge_same_family": bot_family == judge_family,
        "total": len(rows),
        "passed": sum(int(r["passed"]) for r in rows),
        "by_kind": by_kind,
        "failure_tags": dict(sorted(tag_counts.items())),
        "fallback_answers": sum(int(bool(r.get("fallback_ran"))) for r in rows),
        "errors": sum(int(bool(r.get("error"))) for r in rows),
        "usage": usage,
    }
