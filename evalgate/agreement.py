"""How far to trust the LLM judge: compare its verdicts with answers labelled by a person."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("label_id") or row["case_id"])


def answer_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "question": row.get("question", ""),
        "reference": row.get("reference", ""),
        "answer": row.get("answer", ""),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def judged_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("judge")]


def sample_for_labeling(rows: list[dict[str, Any]], n: int, seed: int = 7) -> list[dict[str, Any]]:
    """Pick up to n judged answers, keeping both verdicts represented when possible."""
    pool = judged_rows(rows)
    rng = random.Random(seed)
    correct = [r for r in pool if r["judge"]["verdict"] == "correct"]
    incorrect = [r for r in pool if r["judge"]["verdict"] != "correct"]
    rng.shuffle(correct)
    rng.shuffle(incorrect)
    take_incorrect = min(len(incorrect), max(n // 2, n - len(correct)))
    picked = incorrect[:take_incorrect] + correct[: n - take_incorrect]
    return sorted(picked, key=row_key)


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float:
    """Agreement beyond chance between two binary raters. 1.0 = perfect, 0 = chance."""
    n = len(pairs)
    if n == 0:
        raise ValueError("no labelled pairs")
    observed = sum(a == b for a, b in pairs) / n
    p_a = sum(a for a, _ in pairs) / n
    p_b = sum(b for _, b in pairs) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


def agreement_report(
    rows: list[dict[str, Any]],
    labels: dict[str, bool],
    label_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    by_id = {row_key(r): r for r in judged_rows(rows)}
    pairs, disagreements = [], []
    stale: list[str] = []
    for item_id, human_correct in sorted(labels.items()):
        row = by_id.get(item_id)
        if row is None:
            continue
        if label_hashes is not None:
            expected_hash = answer_fingerprint(row)
            if label_hashes.get(item_id) != expected_hash:
                stale.append(item_id)
                continue
        judge_correct = row["judge"]["verdict"] == "correct"
        pairs.append((human_correct, judge_correct))
        if human_correct != judge_correct:
            disagreements.append({"case_id": item_id, "human": "correct" if human_correct else "incorrect",
                                  "judge": row["judge"]["verdict"], "judge_reason": row["judge"]["reason"]})
    if stale:
        raise ValueError(f"stale labels do not match current answers (re-run label): {stale}")
    if not pairs:
        raise ValueError("none of the labelled cases were judged in this run")
    confusion = {
        "both_correct": sum(1 for h, j in pairs if h and j),
        "both_incorrect": sum(1 for h, j in pairs if not h and not j),
        "judge_too_lenient": sum(1 for h, j in pairs if not h and j),
        "judge_too_strict": sum(1 for h, j in pairs if h and not j),
    }
    human_pos = sum(1 for h, _ in pairs if h) / len(pairs)
    judge_pos = sum(1 for _, j in pairs if j) / len(pairs)
    imbalanced = max(human_pos, 1 - human_pos) >= 0.9 or max(judge_pos, 1 - judge_pos) >= 0.9
    return {
        "labelled": len(pairs),
        "agreement": sum(h == j for h, j in pairs) / len(pairs),
        "kappa": cohen_kappa(pairs),
        "imbalanced": imbalanced,
        "confusion": confusion,
        "disagreements": disagreements,
    }
