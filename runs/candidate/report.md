# Run: candidate

**38 of 40 cases passed (95%).**

- Bot: claude-cli / claude-haiku-5-5 (effort low)
- Judge: codex-cli / gpt-6.2-sol
- Tokens: 67,602 in, 7,533 out

| Kind | Passed |
| --- | --- |
| answerable | 25 / 25 |
| unanswerable | 9 / 10 |
| tricky | 4 / 5 |

| Failure | Cases |
| --- | --- |
| cited while declining | 1 |
| judge: incorrect | 1 |

## Failed cases

- **u09** (unanswerable): How much does one SMS credit cost?
  - Answer: The documentation doesn't say how much one SMS credit costs. It only mentions that usage-based add-ons such as SMS credits are not refunded once consumed, and it does not list add-on prices.
  - citations: declined but still cited ['refunds-4']
- **t04** (tricky): We still call version 1 of the API. What do we change to move to version 2?
  - Answer: To move to version 2, have your client send the header "Brightwell-Version: 2" with its requests. Version 1 is deprecated and will be shut down on 2027-03-31, so the switch should be made before then (about four months from today, 2026-11-24).
  - judge: The header and shutdown date match the reference, but the passages do not establish the current date used for the four-month countdown.
