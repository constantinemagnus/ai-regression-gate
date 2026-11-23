import pytest

from evalgate.agreement import agreement_report, answer_fingerprint, cohen_kappa, sample_for_labeling
from evalgate.cache import OfflineMiss, ResponseCache, request_key
from evalgate.compare import compare_runs, gate_passes
from evalgate.report import agreement_text, comparison_report
from evalgate.runner import run_suite, score_case, summarize

from .conftest import FakeBot, FakeJudge

CONFIG = {"name": "t", "bot": {"provider": "anthropic", "model": "m"}, "judge": {"provider": "codex-cli"}}


def test_strong_bot_passes_everything(suite, cases, tmp_path):
    rows = run_suite(suite, FakeBot(cases), FakeJudge(cases), ResponseCache(tmp_path))
    summary = summarize(rows, CONFIG)
    assert summary["passed"] == 40, [r for r in rows if not r["passed"]]
    assert summary["judge_same_family"] is False


def test_cheap_bot_regressions_are_caught_and_explained(suite, cases, tmp_path):
    base = run_suite(suite, FakeBot(cases), FakeJudge(cases), ResponseCache(tmp_path / "a"))
    cheap = FakeBot(cases, model="fake-cheap", bluff={"u02", "u09"}, fake_cite={"a04"}, wrong={"t03"})
    cand = run_suite(suite, cheap, FakeJudge(cases), ResponseCache(tmp_path / "b"))
    cmp = compare_runs(base, cand)
    assert [e["case_id"] for e in cmp["regressions"]] == ["a04", "u02", "u09", "t03"]
    assert cmp["regression_tags"]["bluffed instead of declining"] == 2
    assert cmp["regression_tags"]["made-up citation"] == 1
    assert cmp["regression_tags"]["wrong or missing fact"] == 1
    assert cmp["regression_tags"]["judge: incorrect"] == 1
    assert not gate_passes(cmp, 0)
    assert gate_passes(cmp, 4)
    report = comparison_report(cmp, summarize(base, CONFIG), summarize(cand, CONFIG), False, 0)
    assert report.startswith("# Regression gate: FAIL")


def test_offline_replay_reproduces_a_run_without_calls(suite, cases, tmp_path):
    first = run_suite(suite, FakeBot(cases), FakeJudge(cases), ResponseCache(tmp_path, "live"))
    bot, judge = FakeBot(cases), FakeJudge(cases)
    replay = run_suite(suite, bot, judge, ResponseCache(tmp_path, "offline"))
    assert replay == first
    assert bot.calls == 0 and judge.calls == 0


def test_offline_mode_fails_loudly_on_a_changed_prompt(suite, cases, tmp_path):
    run_suite(suite, FakeBot(cases), FakeJudge(cases), ResponseCache(tmp_path, "live"))
    suite.bot_system += "\nBe brief."
    with pytest.raises(OfflineMiss):
        run_suite(suite, FakeBot(cases), FakeJudge(cases), ResponseCache(tmp_path, "offline"))


def test_cache_modes(tmp_path):
    calls = []
    cache = ResponseCache(tmp_path)
    fn = lambda p: calls.append(p) or {"x": 1}
    assert cache.fetch({"a": 1}, fn) == {"x": 1}
    assert cache.fetch({"a": 1}, fn) == {"x": 1}
    assert len(calls) == 1
    ResponseCache(tmp_path, "refresh").fetch({"a": 1}, fn)
    assert len(calls) == 2
    assert request_key({"a": 1, "b": 2}) == request_key({"b": 2, "a": 1})
    assert request_key({"a": 1}) != request_key({"a": 2})


def test_kappa():
    assert cohen_kappa([(True, True), (False, False)]) == 1.0
    assert cohen_kappa([(True, True), (True, False), (False, False), (False, True)]) == 0.0
    assert round(cohen_kappa([(True, True)] * 8 + [(False, False)] * 1 + [(True, False)]), 2) == 0.62


def test_labeling_sample_and_agreement(suite, cases, tmp_path):
    cheap = FakeBot(cases, wrong={"a01", "a02", "a03", "t03"})
    rows = run_suite(suite, cheap, FakeJudge(cases), ResponseCache(tmp_path))
    picked = sample_for_labeling(rows, 10)
    assert len(picked) == 10
    assert sum(r["judge"]["verdict"] == "incorrect" for r in picked) == 4
    labels = {r["case_id"]: r["judge"]["verdict"] == "correct" for r in picked}
    labels["a01"] = True  # a person disagrees with the judge once
    rep = agreement_report(rows, labels)
    assert rep["labelled"] == 10 and rep["agreement"] == 0.9
    assert rep["confusion"]["judge_too_strict"] == 1
    assert rep["imbalanced"] is False


def test_agreement_flags_imbalanced_sample(suite, cases, tmp_path):
    rows = run_suite(suite, FakeBot(cases), FakeJudge(cases), ResponseCache(tmp_path))
    picked = sample_for_labeling(rows, 10)
    labels = {r["case_id"]: True for r in picked}
    rep = agreement_report(rows, labels)
    assert rep["imbalanced"] is True
    assert "INCONCLUSIVE" in agreement_text(rep, 0.6)


def test_malformed_bot_payload_and_cited_decline_do_not_crash(suite, cases, tmp_path):
    class WeirdBot(FakeBot):
        def call(self, payload):
            out = super().call(payload)
            if "How much does the Team plan cost" in payload["user"]:
                out["data"] = {"unexpected": "shape"}
            elif "lifetime deal" in payload["user"]:
                out["data"] = {"answer": "Not covered.", "citations": ["pricing-1"], "declined": True}
            elif "Enterprise plan" in payload["user"]:
                out["data"] = {"answer": "Yes, it costs $499.", "citations": [], "declined": True}
            elif "additional user" in payload["user"]:
                out["data"] = None
                out["stop_reason"] = "max_tokens"
            return out

    cache = ResponseCache(tmp_path)
    judge = FakeJudge(cases)
    bad = score_case(suite, cases[0], WeirdBot(cases), judge, cache)
    assert bad["passed"] is False and bad["tags"] == ["error"]
    # Malformed bot payload must NOT be persisted in cache
    assert list(tmp_path.rglob("*.json")) == []

    declined_with_cite = score_case(suite, next(c for c in cases if c.id == "u01"), WeirdBot(cases), judge, cache)
    assert declined_with_cite["passed"] is False and declined_with_cite["tags"] == ["cited while declining"]

    bluffed_decline = score_case(suite, next(c for c in cases if c.id == "u02"), WeirdBot(cases), judge, cache)
    assert bluffed_decline["passed"] is False and bluffed_decline["tags"] == ["bluffed while declining"]

    truncated = score_case(suite, cases[1], WeirdBot(cases), judge, cache)
    assert truncated["passed"] is False and truncated["error"] and "max_tokens" in truncated["error"]


def test_compare_rejects_empty_duplicates_mismatches_and_errors(suite, cases, tmp_path):
    base = run_suite(suite, FakeBot(cases), FakeJudge(cases), ResponseCache(tmp_path / "a"))
    cand = run_suite(suite, FakeBot(cases), FakeJudge(cases), ResponseCache(tmp_path / "b"))
    cmp = compare_runs(base, cand)
    assert gate_passes(cmp, 0) is True
    assert gate_passes(cmp, 0, candidate_errors=1) is False
    assert gate_passes(cmp, 0, baseline_errors=1) is False

    with pytest.raises(ValueError, match="no cases"):
        compare_runs([], [])
    with pytest.raises(ValueError, match="duplicate"):
        compare_runs([base[0], base[0]], [cand[0], cand[0]])
    altered = dict(cand[0], question="Different question?")
    with pytest.raises(ValueError, match="differs"):
        compare_runs([base[0]], [altered])
    errored = [dict(r, error="bot: truncated (max_tokens)", passed=False) if r["case_id"] == "a01" else r for r in cand]
    cmp_err = compare_runs(base, errored)
    assert cmp_err["candidate_errors"] == 1
    assert gate_passes(cmp_err, max_regressions=5) is False


def test_agreement_rejects_stale_label_hashes(suite, cases, tmp_path):
    cheap = FakeBot(cases, wrong={"a01", "a02"})
    rows = run_suite(suite, cheap, FakeJudge(cases), ResponseCache(tmp_path))
    picked = sample_for_labeling(rows, 4)
    labels = {r["case_id"]: r["judge"]["verdict"] == "correct" for r in picked}
    hashes = {r["case_id"]: answer_fingerprint(r) for r in picked}
    assert agreement_report(rows, labels, label_hashes=hashes)["labelled"] == 4
    hashes[picked[0]["case_id"]] = "stale00000000000"
    with pytest.raises(ValueError, match="stale labels"):
        agreement_report(rows, labels, label_hashes=hashes)
