"""End to end through the command line, with fake models."""

import pytest

from evalgate import cli, common
from evalgate.agreement import answer_fingerprint

from .conftest import FakeBot, FakeJudge


def test_run_and_compare(cases, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(common, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(common, "RUNS_DIR", tmp_path / "runs")

    def fake_provider(section):
        if section["provider"] == "codex-cli":
            return FakeJudge(cases)
        if section["model"] == "claude-haiku-5-5":
            return FakeBot(cases, model="cheap", bluff={"u09"})
        return FakeBot(cases)

    monkeypatch.setattr(cli, "make_provider", fake_provider)
    root = common.ROOT
    assert cli.main(["run", "--config", str(root / "configs/baseline.toml")]) == 0
    assert cli.main(["run", "--config", str(root / "configs/candidate.toml")]) == 0
    code = cli.main(["compare", str(tmp_path / "runs/baseline"), str(tmp_path / "runs/candidate")])
    out = capsys.readouterr().out
    assert code == 1
    assert "1 regressions" in out and "u09" in out
    # second run replays from the cache with no model calls
    assert cli.main(["run", "--config", str(root / "configs/baseline.toml"), "--offline"]) == 0


def test_cli_overrides_case_ids_and_pooled_agreement(cases, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(common, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(common, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(common, "LABELS_DIR", tmp_path / "labels")
    seen_sections = []

    def fake_provider(section):
        seen_sections.append(dict(section))
        if section["provider"] == "anthropic" and section.get("effort") == "medium":
            return FakeJudge(cases)
        if section["provider"] == "codex-cli":
            return FakeJudge(cases)
        if section["model"] == "claude-haiku-5-5":
            return FakeBot(cases, model="cheap", wrong={"a01", "t03"})
        return FakeBot(cases)

    monkeypatch.setattr(cli, "make_provider", fake_provider)
    root = common.ROOT

    # Empty or duplicate --case-ids rejected
    with pytest.raises(ValueError, match="at least one"):
        cli.main(["run", "--config", str(root / "configs/baseline.toml"), "--case-ids", ","])
    with pytest.raises(ValueError, match="duplicate"):
        cli.main(["run", "--config", str(root / "configs/baseline.toml"), "--case-ids", "a01,a01"])

    # Verify --bot-provider and --judge-provider preserve effort and reset cross-family model
    assert cli.main([
        "run", "--config", str(root / "configs/baseline.toml"),
        "--case-ids", "a01,u02,t03",
        "--bot-provider", "claude-cli",
        "--judge-provider", "anthropic",
        "--out", str(tmp_path / "runs/override_check"),
    ]) == 0
    assert seen_sections[0]["provider"] == "claude-cli"
    assert seen_sections[1]["provider"] == "anthropic" and seen_sections[1]["effort"] == "medium"
    assert seen_sections[1]["model"] == ""

    # Run baseline and candidate with the SAME judge config, then pool for calibration
    assert cli.main(["run", "--config", str(root / "configs/baseline.toml"), "--case-ids", "a01,u02,t03"]) == 0
    assert cli.main(["run", "--config", str(root / "configs/candidate.toml"), "--case-ids", "a01,u02,t03"]) == 0
    base_rows = {r["case_id"]: r for r in common.read_jsonl(tmp_path / "runs/baseline/results.jsonl")}
    cand_rows = {r["case_id"]: r for r in common.read_jsonl(tmp_path / "runs/candidate/results.jsonl")}
    assert list(base_rows) == ["a01", "u02", "t03"]

    labels_path = tmp_path / "labels/baseline+candidate.jsonl"
    common.write_jsonl(labels_path, [
        {"case_id": "baseline:a01", "correct": True, "answer_hash": answer_fingerprint(base_rows["a01"])},
        {"case_id": "baseline:t03", "correct": True, "answer_hash": answer_fingerprint(base_rows["t03"])},
        {"case_id": "candidate:a01", "correct": False, "answer_hash": answer_fingerprint(cand_rows["a01"])},
        {"case_id": "candidate:t03", "correct": False, "answer_hash": answer_fingerprint(cand_rows["t03"])},
    ])
    # Default min_labels=10 rejects a tiny 4-label sample
    assert cli.main(["agreement", str(tmp_path / "runs/baseline"), str(tmp_path / "runs/candidate")]) == 1
    assert "INCONCLUSIVE" in capsys.readouterr().out
    # With --min-labels 4, balanced 4-label sample with matching hashes passes
    assert cli.main(["agreement", str(tmp_path / "runs/baseline"), str(tmp_path / "runs/candidate"), "--min-labels", "4"]) == 0
    out = capsys.readouterr().out
    assert "TRUSTED" in out and "Cohen's kappa 1.00" in out

    # Pooling runs with different judge configs is rejected
    with pytest.raises(ValueError, match="different judges"):
        cli.main(["agreement", str(tmp_path / "runs/override_check"), str(tmp_path / "runs/candidate"), "--labels", str(labels_path)])
