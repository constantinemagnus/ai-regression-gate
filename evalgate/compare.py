"""Compare a candidate run with the baseline and decide whether the change may ship."""

from __future__ import annotations

from typing import Any


def _index_rows(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    if not rows:
        raise ValueError(f"{label} run contains no cases")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in indexed:
            raise ValueError(f"{label} run has duplicate case_id {case_id!r}")
        indexed[case_id] = row
    return indexed


def compare_runs(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    base = _index_rows(baseline, "baseline")
    cand = _index_rows(candidate, "candidate")
    if set(base) != set(cand):
        missing = sorted(set(base) ^ set(cand))
        raise ValueError(f"runs cover different cases: {missing}")

    regressions, fixes, still_failing = [], [], []
    for case_id in base:
        b, c = base[case_id], cand[case_id]
        if b.get("question") != c.get("question") or b.get("kind") != c.get("kind"):
            raise ValueError(f"case {case_id!r} differs between baseline and candidate")
        if "reference" in b and "reference" in c and b["reference"] != c["reference"]:
            raise ValueError(f"case {case_id!r} has different reference answers between runs")
        entry = {
            "case_id": case_id,
            "kind": c["kind"],
            "question": c["question"],
            "baseline_answer": b.get("answer", ""),
            "candidate_answer": c.get("answer", ""),
            "candidate_tags": c.get("tags", []),
            "candidate_detail": [ch["detail"] for ch in c.get("checks", []) if not ch["passed"]] or ([c["error"]] if c.get("error") else []),
        }
        if b["passed"] and not c["passed"]:
            regressions.append(entry)
        elif not b["passed"] and c["passed"]:
            fixes.append(entry)
        elif not b["passed"] and not c["passed"]:
            still_failing.append(entry)

    tag_counts: dict[str, int] = {}
    for entry in regressions:
        for tag in entry["candidate_tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    n = len(base)
    return {
        "total": n,
        "baseline_passed": sum(int(r["passed"]) for r in baseline),
        "candidate_passed": sum(int(r["passed"]) for r in candidate),
        "baseline_errors": sum(int(bool(r.get("error"))) for r in baseline),
        "candidate_errors": sum(int(bool(r.get("error"))) for r in candidate),
        "regressions": regressions,
        "fixes": fixes,
        "still_failing": still_failing,
        "regression_tags": dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def gate_passes(
    comparison: dict[str, Any],
    max_regressions: int = 0,
    baseline_errors: int = 0,
    candidate_errors: int = 0,
) -> bool:
    if (
        baseline_errors > 0
        or candidate_errors > 0
        or comparison.get("baseline_errors", 0) > 0
        or comparison.get("candidate_errors", 0) > 0
    ):
        return False
    return len(comparison["regressions"]) <= max_regressions
