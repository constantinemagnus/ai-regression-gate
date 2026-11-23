from evalgate.cases import Case
from evalgate.checks import (
    check_citations,
    check_decline,
    check_declined_answer,
    check_declined_citations,
    check_facts,
)

ANSWERABLE = Case("a", "answerable", "q", ("pricing-1",), "r", (("$6",),))
UNANSWERABLE = Case("u", "unanswerable", "q", (), "r", ())
CORPUS = {"pricing-1", "pricing-2", "sla-1"}


def test_decline_rules():
    assert check_decline(UNANSWERABLE, True)["passed"]
    assert check_decline(UNANSWERABLE, False)["tags"] == ["bluffed instead of declining"]
    assert check_decline(ANSWERABLE, True)["tags"] == ["declined an answerable question"]
    assert check_decline(ANSWERABLE, False)["passed"]


def test_citation_rules():
    shown = ["pricing-1", "pricing-2"]
    assert check_citations(ANSWERABLE, ["pricing-1"], shown, CORPUS)["passed"]
    assert check_citations(ANSWERABLE, ["[pricing-1]"], shown, CORPUS)["passed"]
    assert check_citations(ANSWERABLE, [], shown, CORPUS)["tags"] == ["missing citation"]
    assert check_citations(ANSWERABLE, ["pricing-9"], shown, CORPUS)["tags"] == ["made-up citation"]
    # exists in the docs but was never shown to the bot: also made up
    assert check_citations(ANSWERABLE, ["sla-1"], shown, CORPUS)["tags"] == ["made-up citation"]
    assert check_citations(ANSWERABLE, ["pricing-2"], shown, CORPUS)["tags"] == ["wrong source cited"]
    # citing every shown passage to game citations fails
    assert check_citations(ANSWERABLE, ["pricing-1", "pricing-2"], shown, CORPUS)["tags"] == ["blanket citation"]
    assert check_declined_citations([])["passed"]
    assert check_declined_citations(["pricing-1"])["tags"] == ["cited while declining"]
    assert check_declined_answer("The documentation doesn't cover that.")["passed"]
    assert check_declined_answer(
        "The documentation doesn't cover Enterprise plan pricing. It only describes Starter ($19/month)."
    )["passed"]
    assert check_declined_answer("I can't find an Enterprise plan in our documentation.")["passed"]
    assert check_declined_answer("The documentation I have doesn't list a phone number.")["passed"]
    assert check_declined_answer("Yes, it costs $499.")["tags"] == ["bluffed while declining"]
    assert check_declined_answer("")["tags"] == ["bad decline text"]


def test_fact_matching_respects_number_boundaries():
    assert check_facts(ANSWERABLE, "It costs $6 per user.")["passed"]
    assert check_facts(ANSWERABLE, "It costs $6.00 per user.")["passed"]
    assert check_facts(ANSWERABLE, "It costs $6.")["passed"]
    assert check_facts(ANSWERABLE, "It costs $6, plus tax.")["passed"]
    assert not check_facts(ANSWERABLE, "It costs $60 per user.")["passed"]
    assert not check_facts(ANSWERABLE, "It costs $6,000 per user.")["passed"]
    assert not check_facts(ANSWERABLE, "It costs $6.50 per user.")["passed"]
    assert not check_facts(ANSWERABLE, "It costs $6.05 per user.")["passed"]
    num_case = Case("n", "answerable", "q", ("pricing-1",), "r", (("25",),))
    assert check_facts(num_case, "Includes 25 users.")["passed"]
    assert not check_facts(num_case, "Includes 1.25 users.")["passed"]
    assert not check_facts(num_case, "Includes 25.5 users.")["passed"]
    rate_case = Case("r", "answerable", "q", ("api-1",), "r", (("600",),))
    assert check_facts(rate_case, "Limit is 600 requests per minute.")["passed"]
    assert not check_facts(rate_case, "Limit is 1,600 requests per minute.")["passed"]
    assert check_facts(UNANSWERABLE, "anything") is None


def test_fact_matching_normalizes_dashes_and_case():
    case = Case("b", "answerable", "q", ("refunds-3",), "r", (("10 business days",), ("Request refund",)))
    assert check_facts(case, "Use request refund; it takes 5–10\u00a0Business Days.")["passed"]
