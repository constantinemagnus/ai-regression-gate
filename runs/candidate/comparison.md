# Regression gate: FAIL

Baseline `claude-opus-5-5` passed 40 of 40; candidate `claude-haiku-5-5` passed 38 of 40.

**2 regressions** (allowed: 0), 0 fixes, 0 cases failing in both.

| | Baseline | Candidate |
| --- | --- | --- |
| Passed | 40 / 40 | 38 / 40 |
| Input tokens | 66,585 | 67,602 |
| Output tokens | 7,613 | 7,533 |

| What broke | Cases |
| --- | --- |
| cited while declining | 1 |
| judge: incorrect | 1 |

## Regressions

- **u09** (unanswerable): How much does one SMS credit cost?
  - Baseline: I can't find the price of an SMS credit in the documentation. It only says SMS credits are a usage-based add-on and aren't refunded once you've used them. It doesn't say what they cost.
  - Candidate: The documentation doesn't say how much one SMS credit costs. It only mentions that usage-based add-ons such as SMS credits are not refunded once consumed, and it does not list add-on prices.
  - Why it failed: declined but still cited ['refunds-4']
- **t04** (tricky): We still call version 1 of the API. What do we change to move to version 2?
  - Baseline: To move to version 2, change the Brightwell-Version header on your API requests to "Brightwell-Version: 2". Version 2 is the current API version. Version 1 is deprecated and will be shut down on 2027-03-31, so switch before that date. The documentation doesn't describe any other changes needed for the migration.
  - Candidate: To move to version 2, have your client send the header "Brightwell-Version: 2" with its requests. Version 1 is deprecated and will be shut down on 2027-03-31, so the switch should be made before then (about four months from today, 2026-11-24).
  - Why it failed: The header and shutdown date match the reference, but the passages do not establish the current date used for the four-month countdown.
