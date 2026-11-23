# AI regression gate

Before you switch a support bot to a much cheaper model, this project measures whether the switch is safe. It runs the same 40 questions through both models and checks each answer for made-up citations, confident answers to questions the docs don't cover, and wrong facts. If the cheaper model does worse on any question, CI blocks the deploy, and the report shows which questions it got wrong and why.

> Brightwell Cloud and its documentation are fictional, written for this demo.

## What it checks

The bot answers 40 customer questions from 27 documentation passages (pricing, refunds, SLA, API limits, data retention). It must return an answer, the passage IDs it used, and whether it declined.

| Questions | Count | What must happen |
| --- | --- | --- |
| Answerable | 25 | Correct answer, citing the right passage |
| Not covered by the docs | 10 | Decline instead of guessing |
| Tricky (conditions, maths) | 5 | Work out the right result, such as which refund window applies |

Every answer goes through four checks:

1. **Declining** (code): declines exactly the questions the docs don't cover.
2. **Citations** (code): cites a passage it was actually shown, and the right one. Citing anything else counts as a made-up citation.
3. **Key facts** (code): the answer contains the expected numbers and terms; `$6` does not match `$60`.
4. **LLM judge**: a model from a different family (OpenAI's, through the Codex CLI) compares the answer with a reference answer written by a person.

The judge is only trusted after a calibration check: you label 20 answers by hand, and `evalgate agreement` reports agreement and Cohen's kappa. Below 0.6, the judge isn't trusted.

## Results

First live comparison (`claude-opus-5-5` baseline vs. `claude-haiku-5-5` candidate, both at `low` effort, judged by OpenAI `gpt-6.2-sol` via Codex CLI):

| | Baseline (`claude-opus-5-5`) | Candidate (`claude-haiku-5-5`) |
| --- | --- | --- |
| Passed | **40 / 40 (100%)** | **38 / 40 (95%)** |
| Bot input tokens | 66,585 | 67,602 |
| Bot output tokens | 7,613 | 7,533 |
| Gate verdict | — | **FAIL (2 regressions, exit code `1`)** |

What the gate flagged (full output in `runs/candidate/comparison.md`):

- **`u09`** *(code check — `cited while declining`)*: Haiku declined the unanswerable SMS-pricing question accurately in text, but still attached `citations: ["refunds-4"]`, breaking the empty-citations-on-decline contract.
- **`t04`** *(cross-family judge — `gpt-6.2-sol`)*: Haiku gave the right header and shutdown date, but added a countdown using the live system date (`2026-11-24`) even though the docs never establish today's date; Opus stayed strictly grounded in the passages.

**Honest takeaway:** Haiku matched Opus on core factual accuracy (0 made-up citations, 0 math errors, 0 bluffed answers)—so manual inspection of the two flagged cases tells a team whether to relax those strict contract/grounding rules and take the cost cut, or tighten the candidate prompt before shipping. Note that because 59 of 60 judged answers across both runs passed, Cohen's kappa calibration is flagged `INCONCLUSIVE (imbalanced sample)` until tested against a more balanced mix of passing and failing answers.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q                     # 30 tests, no model calls
```

Choose how the bot reaches Claude in `configs/*.toml` (default: `claude-cli`):

- `provider = "claude-cli"`: `claude -p --setting-sources ""` on a Claude subscription, no API key.
- `provider = "anthropic"`: the Claude API through the Anthropic SDK. Needs `ANTHROPIC_API_KEY` or an `ant auth login` profile.

The judge uses `codex exec --ignore-user-config` pinned to `gpt-6.2-sol` on a ChatGPT/Codex login. Without one, add `--judge-provider anthropic`; the report then warns that judge and bot share a model family.

```bash
.venv/bin/python -m evalgate run --config configs/baseline.toml --case-ids a01,u02,t03   # 3-case smoke test (1 of each kind)
.venv/bin/python -m evalgate run --config configs/baseline.toml
.venv/bin/python -m evalgate run --config configs/candidate.toml
.venv/bin/python -m evalgate compare runs/baseline runs/candidate                        # exit code 1 = blocked
.venv/bin/python -m evalgate label runs/baseline runs/candidate                          # label 20 answers across both runs
.venv/bin/python -m evalgate agreement runs/baseline runs/candidate
```

To override the provider without editing `configs/*.toml`, pass `--bot-provider anthropic` or `--bot-provider claude-cli`.

Estimated cost for both full runs, by token arithmetic rather than measurement: under $1 in API fees (Claude Opus 5.5 baseline, Claude Haiku 5.5 candidate, low effort), plus about 60 judge calls on the Codex subscription.

## How CI uses it

Every model response is cached under `cache/`, keyed by the exact request. Commit the cache after a live run, and the GitHub Actions workflow replays both runs on every pull request with no API keys and no cost, then fails the PR if the candidate regresses on any case.

A pull request that changes a prompt, model or document has no cached responses, so the replay stops and asks for a live run. It never reuses an old answer for a new prompt.

## Design notes

- **No silent model mixing.** The baseline uses the API's server-side refusal fallback. If a fallback model answers instead, the result records it and the report counts it.
- **Retrieval is deterministic** (a small BM25 index, no embeddings), and a test asserts that the right passage is in the top 4 for every answerable question. A failure is therefore the model's, not the retriever's.
- **A failed call is a failed case,** never a crashed run, and offline mode never guesses.
