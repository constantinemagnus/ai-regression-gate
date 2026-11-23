"""Command line: run, compare, label, agreement."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from . import common
from .agreement import agreement_report, answer_fingerprint, row_key, sample_for_labeling
from .cache import OfflineMiss, ResponseCache
from .cases import load_cases
from .compare import compare_runs, gate_passes
from .corpus import load_passages
from .prompts import load_prompt
from .providers import make_provider
from .report import agreement_text, comparison_report, run_report
from .runner import Suite, run_suite, summarize


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        config = tomllib.load(f)
    config.setdefault("name", Path(path).stem)
    config.setdefault("retrieval", {}).setdefault("top_k", 4)
    return config


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.bot_provider:
        prev_bot_provider = config["bot"].get("provider")
        config["bot"]["provider"] = args.bot_provider
        if args.bot_model:
            config["bot"]["model"] = args.bot_model
        elif (args.bot_provider == "codex-cli") != (prev_bot_provider == "codex-cli"):
            config["bot"]["model"] = ""
    elif args.bot_model:
        config["bot"]["model"] = args.bot_model
    if args.judge_provider:
        prev_judge_provider = config["judge"].get("provider")
        config["judge"]["provider"] = args.judge_provider
        if args.judge_model:
            config["judge"]["model"] = args.judge_model
        elif (args.judge_provider == "codex-cli") != (prev_judge_provider == "codex-cli"):
            config["judge"]["model"] = ""
    elif args.judge_model:
        config["judge"]["model"] = args.judge_model
    passages = load_passages(common.DOCS_DIR)
    cases = load_cases(args.cases, {p.id for p in passages})
    if args.case_ids is not None:
        wanted = [cid.strip() for cid in args.case_ids.split(",") if cid.strip()]
        if not wanted:
            raise ValueError("--case-ids must contain at least one case id")
        if len(wanted) != len(set(wanted)):
            raise ValueError(f"duplicate case ids in --case-ids: {wanted}")
        by_id = {c.id: c for c in cases}
        unknown = [cid for cid in wanted if cid not in by_id]
        if unknown:
            raise ValueError(f"unknown case ids: {unknown}")
        cases = [by_id[cid] for cid in wanted]
    if args.limit:
        cases = cases[: args.limit]
    suite = Suite(cases, passages, load_prompt(common.PROMPTS_DIR, "bot_system.md"),
                  load_prompt(common.PROMPTS_DIR, "judge.md"), top_k=int(config["retrieval"]["top_k"]))
    mode = "offline" if args.offline else "refresh" if args.refresh else "live"
    cache = ResponseCache(common.CACHE_DIR, mode)
    bot, judge = make_provider(config["bot"]), make_provider(config["judge"])

    done = 0

    def progress(row: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        mark = "pass" if row["passed"] else "FAIL " + ", ".join(row.get("tags", []))
        print(f"[{done}/{len(cases)}] {row['case_id']}: {mark}", file=sys.stderr)

    try:
        rows = run_suite(suite, bot, judge, cache, workers=args.workers, progress=progress)
    except OfflineMiss as exc:
        print(f"offline replay failed: {exc}\nRun without --offline to call the models.", file=sys.stderr)
        return 2
    summary = summarize(rows, config)
    out = Path(args.out or common.RUNS_DIR / config["name"])
    common.write_jsonl(out / "results.jsonl", rows)
    common.write_json(out / "summary.json", summary)
    (out / "report.md").write_text(run_report(summary, rows), encoding="utf-8")
    print(f"{summary['passed']}/{summary['total']} passed. Cache: {cache.hits} replayed, {cache.misses} live. Report: {out / 'report.md'}")
    return 0


def _load_run(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run_dir = Path(run_dir)
    return common.read_jsonl(run_dir / "results.jsonl"), json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _load_runs_for_labeling(run_dirs: list[Path]) -> tuple[list[dict[str, Any]], str, Path]:
    if len(run_dirs) == 1:
        rows, summary = _load_run(run_dirs[0])
        return rows, summary["name"], Path(run_dirs[0])
    pooled: list[dict[str, Any]] = []
    names: list[str] = []
    seen_names: set[str] = set()
    judge_sig: tuple[Any, ...] | None = None
    for run_dir in run_dirs:
        rows, summary = _load_run(run_dir)
        run_name = summary["name"]
        if run_name in seen_names:
            raise ValueError(f"cannot pool multiple runs with the same name {run_name!r}")
        seen_names.add(run_name)
        j = summary.get("judge") or {}
        sig = (j.get("provider"), j.get("model"), j.get("effort"))
        if judge_sig is None:
            judge_sig = sig
        elif sig != judge_sig:
            raise ValueError(f"cannot pool runs evaluated by different judges: {judge_sig} vs {sig}")
        names.append(run_name)
        for row in rows:
            tagged = dict(row)
            tagged["label_id"] = f"{run_name}:{row['case_id']}"
            pooled.append(tagged)
    return pooled, "+".join(names), Path(run_dirs[-1])


def cmd_compare(args: argparse.Namespace) -> int:
    base_rows, base_summary = _load_run(args.baseline)
    cand_rows, cand_summary = _load_run(args.candidate)
    cmp = compare_runs(base_rows, cand_rows)
    ok = gate_passes(
        cmp,
        args.max_regressions,
        baseline_errors=base_summary.get("errors", 0),
        candidate_errors=cand_summary.get("errors", 0),
    )
    text = comparison_report(cmp, base_summary, cand_summary, ok, args.max_regressions)
    out = Path(args.out or Path(args.candidate) / "comparison.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok else 1


def cmd_label(args: argparse.Namespace) -> int:
    rows, label_name, _ = _load_runs_for_labeling(args.runs)
    picked = sample_for_labeling(rows, args.n, args.seed)
    labels_path = Path(args.labels or common.LABELS_DIR / f"{label_name}.jsonl")
    existing = {r["case_id"]: r for r in common.read_jsonl(labels_path)} if labels_path.exists() else {}
    cases = {c.id: c for c in load_cases(common.CASES_PATH)}
    print(f"Label {len(picked)} answers. Judge verdicts are hidden. Type y (correct), n (incorrect) or q (quit).\n")
    for i, row in enumerate(picked, 1):
        item_id = row_key(row)
        fp = answer_fingerprint(row)
        if item_id in existing and existing[item_id].get("answer_hash") == fp:
            continue
        reference = row.get("reference") or (cases[row["case_id"]].reference if row["case_id"] in cases else "(none)")
        print(f"--- {i}/{len(picked)}  {item_id}\nQuestion:  {row['question']}\nReference: {reference}\nAnswer:    {row['answer']}")
        while True:
            reply = input("Correct? [y/n/q] ").strip().lower()
            if reply in ("y", "n", "q"):
                break
        if reply == "q":
            break
        existing[item_id] = {"case_id": item_id, "correct": reply == "y", "answer_hash": fp}
        common.write_jsonl(labels_path, existing.values())
    print(f"\n{len(existing)} labels saved to {labels_path}")
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    rows, label_name, out_dir = _load_runs_for_labeling(args.runs)
    labels_path = Path(args.labels or common.LABELS_DIR / f"{label_name}.jsonl")
    label_rows = common.read_jsonl(labels_path)
    labels = {r["case_id"]: bool(r["correct"]) for r in label_rows}
    label_hashes = {r["case_id"]: r.get("answer_hash", "") for r in label_rows}
    rep = agreement_report(rows, labels, label_hashes=label_hashes)
    text = agreement_text(rep, args.min_kappa, min_labels=args.min_labels)
    out = Path(args.out or out_dir / "agreement.md")
    out.write_text(text, encoding="utf-8")
    print(text)
    ok = rep["kappa"] >= args.min_kappa and not rep["imbalanced"] and rep["labelled"] >= args.min_labels
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalgate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the test set against one config")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--cases", type=Path, default=common.CASES_PATH)
    run.add_argument("--out", type=Path)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--limit", type=int, help="only the first N cases (for a cheap smoke test)")
    run.add_argument("--case-ids", help="comma-separated case IDs to run (e.g. a01,u02,t03)")
    run.add_argument("--bot-provider", choices=["anthropic", "claude-cli", "codex-cli"])
    run.add_argument("--bot-model")
    run.add_argument("--judge-provider", choices=["anthropic", "claude-cli", "codex-cli"])
    run.add_argument("--judge-model")
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--offline", action="store_true", help="replay cached responses only; fail on a miss")
    mode.add_argument("--refresh", action="store_true", help="ignore the cache and call every model again")
    run.set_defaults(func=cmd_run)

    cmp = sub.add_parser("compare", help="gate a candidate run against the baseline run")
    cmp.add_argument("baseline", type=Path)
    cmp.add_argument("candidate", type=Path)
    cmp.add_argument("--max-regressions", type=int, default=0)
    cmp.add_argument("--out", type=Path)
    cmp.set_defaults(func=cmd_compare)

    label = sub.add_parser("label", help="label answers by hand to calibrate the judge")
    label.add_argument("runs", type=Path, nargs="+")
    label.add_argument("--n", type=int, default=20)
    label.add_argument("--seed", type=int, default=7)
    label.add_argument("--labels", type=Path)
    label.set_defaults(func=cmd_label)

    agree = sub.add_parser("agreement", help="compare the judge with your labels")
    agree.add_argument("runs", type=Path, nargs="+")
    agree.add_argument("--labels", type=Path)
    agree.add_argument("--out", type=Path)
    agree.add_argument("--min-kappa", type=float, default=0.6)
    agree.add_argument("--min-labels", type=int, default=10)
    agree.set_defaults(func=cmd_agreement)

    args = parser.parse_args(argv)
    return args.func(args)
