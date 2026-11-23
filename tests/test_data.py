import pytest

from evalgate.cases import load_cases
from evalgate.corpus import BM25Index, tokenize


def test_corpus_has_unique_ids(passages):
    ids = [p.id for p in passages]
    assert len(ids) == len(set(ids)) == 27


def test_case_mix(cases):
    kinds = [c.kind for c in cases]
    assert len(cases) == 40
    assert (kinds.count("answerable"), kinds.count("unanswerable"), kinds.count("tricky")) == (25, 10, 5)


def test_retrieval_finds_a_gold_passage_for_every_answerable_case(cases, passages):
    index = BM25Index(passages)
    for case in cases:
        if case.expects_decline:
            continue
        top = [p.id for p in index.search(case.question, 4)]
        assert set(top) & set(case.gold), f"{case.id}: gold {case.gold} not in top 4 {top}"


def test_tokenizer_keeps_numbers():
    assert "99.9" in tokenize("uptime of 99.9% per month")
    assert "2,000,000" in tokenize("2,000,000 calls")


def test_case_validation(tmp_path):
    bad = tmp_path / "cases.jsonl"
    bad.write_text('{"id": "x", "kind": "answerable", "question": "q", "gold": ["nope-1"], "reference": "r"}\n')
    with pytest.raises(ValueError, match="unknown gold"):
        load_cases(bad, {"pricing-1"})
    bad.write_text('{"id": "x", "kind": "unanswerable", "question": "q", "gold": ["pricing-1"], "reference": "r"}\n')
    with pytest.raises(ValueError, match="no gold"):
        load_cases(bad)
