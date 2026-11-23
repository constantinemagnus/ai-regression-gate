"""Markdown reports for a single run and for a baseline-vs-candidate comparison."""

from __future__ import annotations

from typing import Any


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "n/a"


def run_report(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        f"# Run: {summary['name']}",
        "",
        f"**{summary['passed']} of {summary['total']} cases passed ({_pct(summary['passed'], summary['total'])}).**",
        "",
        f"- Bot: {summary['bot']['provider']} / {summary['bot']['model']} (effort {summary['bot'].get('effort', '-')})",
        f"- Judge: {summary['judge']['provider']} / {summary['judge'].get('model') or 'default model'}"
        + (" (same model family as the bot: treat its verdicts with extra caution)" if summary["judge_same_family"] else ""),
        f"- Tokens: {summary['usage']['input_tokens']:,} in, {summary['usage']['output_tokens']:,} out",
    ]
    if summary["fallback_answers"]:
        lines.append(f"- {summary['fallback_answers']} answers came from a fallback model after a refusal")
    if summary["errors"]:
        lines.append(f"- {summary['errors']} cases hit an error")
    lines += ["", "| Kind | Passed |", "| --- | --- |"]
    for kind, k in summary["by_kind"].items():
        lines.append(f"| {kind} | {k['passed']} / {k['total']} |")
    if summary["failure_tags"]:
        lines += ["", "| Failure | Cases |", "| --- | --- |"]
        for tag, count in summary["failure_tags"].items():
            lines.append(f"| {tag} | {count} |")
    failed = [r for r in rows if not r["passed"]]
    if failed:
        lines += ["", "## Failed cases", ""]
        for r in failed:
            lines.append(f"- **{r['case_id']}** ({r['kind']}): {r['question']}")
            lines.append(f"  - Answer: {r.get('answer') or '(none)'}")
            for ch in r.get("checks", []):
                if not ch["passed"]:
                    lines.append(f"  - {ch['name']}: {ch['detail']}")
            if r.get("error"):
                lines.append(f"  - error: {r['error']}")
    return "\n".join(lines) + "\n"


def comparison_report(cmp: dict[str, Any], base_summary: dict[str, Any], cand_summary: dict[str, Any],
                      passed_gate: bool, max_regressions: int) -> str:
    verdict = "PASS" if passed_gate else "FAIL"
    lines = [
        f"# Regression gate: {verdict}",
        "",
        f"Baseline `{base_summary['bot']['model']}` passed {cmp['baseline_passed']} of {cmp['total']}; "
        f"candidate `{cand_summary['bot']['model']}` passed {cmp['candidate_passed']} of {cmp['total']}.",
        "",
        f"**{len(cmp['regressions'])} regressions** (allowed: {max_regressions}), "
        f"{len(cmp['fixes'])} fixes, {len(cmp['still_failing'])} cases failing in both.",
        "",
        "| | Baseline | Candidate |",
        "| --- | --- | --- |",
        f"| Passed | {cmp['baseline_passed']} / {cmp['total']} | {cmp['candidate_passed']} / {cmp['total']} |",
        f"| Input tokens | {base_summary['usage']['input_tokens']:,} | {cand_summary['usage']['input_tokens']:,} |",
        f"| Output tokens | {base_summary['usage']['output_tokens']:,} | {cand_summary['usage']['output_tokens']:,} |",
    ]
    if base_summary.get("errors") or cand_summary.get("errors"):
        lines += [
            "",
            f"**Errors blocked the gate:** baseline had {base_summary.get('errors', 0)} errors, "
            f"candidate had {cand_summary.get('errors', 0)} errors.",
        ]
    if cmp["regression_tags"]:
        lines += ["", "| What broke | Cases |", "| --- | --- |"]
        for tag, count in cmp["regression_tags"].items():
            lines.append(f"| {tag} | {count} |")
    if cmp["regressions"]:
        lines += ["", "## Regressions", ""]
        for e in cmp["regressions"]:
            lines.append(f"- **{e['case_id']}** ({e['kind']}): {e['question']}")
            lines.append(f"  - Baseline: {e['baseline_answer'] or '(none)'}")
            lines.append(f"  - Candidate: {e['candidate_answer'] or '(none)'}")
            for d in e["candidate_detail"]:
                lines.append(f"  - Why it failed: {d}")
    if cmp["fixes"]:
        lines += ["", "## Fixed by the candidate", ""]
        lines += [f"- **{e['case_id']}**: {e['question']}" for e in cmp["fixes"]]
    return "\n".join(lines) + "\n"


def agreement_text(rep: dict[str, Any], min_kappa: float, min_labels: int = 10) -> str:
    too_few = rep["labelled"] < min_labels
    imbalanced = bool(rep.get("imbalanced", False))
    ok = rep["kappa"] >= min_kappa and not imbalanced and not too_few
    if ok:
        status = "TRUSTED"
    elif too_few:
        status = f"INCONCLUSIVE (fewer than {min_labels} labels)"
    elif imbalanced:
        status = "INCONCLUSIVE (imbalanced sample)"
    else:
        status = "NOT TRUSTED"
    lines = [
        f"# Judge agreement: {status}",
        "",
        f"{rep['labelled']} answers labelled by hand (minimum {min_labels}). Agreement {rep['agreement']:.0%}, "
        f"Cohen's kappa {rep['kappa']:.2f} (threshold {min_kappa}).",
    ]
    if imbalanced:
        lines += [
            "",
            "Warning: >=90% of the labelled answers fell into one verdict class. Cohen's kappa is "
            "unreliable under extreme class imbalance; label a combined pool (e.g. `runs/baseline runs/candidate`) "
            "so both correct and incorrect answers are tested.",
        ]
    lines += [
        "",
        "| | Count |",
        "| --- | --- |",
    ]
    for key, value in rep["confusion"].items():
        lines.append(f"| {key.replace('_', ' ')} | {value} |")
    if rep["disagreements"]:
        lines += ["", "## Disagreements", ""]
        for d in rep["disagreements"]:
            lines.append(f"- **{d['case_id']}**: you said {d['human']}, judge said {d['judge']} ({d['judge_reason']})")
    return "\n".join(lines) + "\n"
