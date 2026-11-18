# Decision log

Record decisions that change scope, contracts or experiment design: what was decided, why, and who decided. Add the newest entry at the top; put longer rationale in a section below the table.

| Date | Decision | Why | Decided by |
| --- | --- | --- | --- |
| 2026-11-18 | Run the main GPU matrix in four preregistered tiers, and cut by the date the integrated pilot first passes ([details](#2026-11-18-gpu-matrix-tiers-and-cut-rule)) | The 11/15 integrated-pilot gate was missed. The matrix itself needs only about 10 GPU hours, so the integration date, not compute, decides what fits before the 11/25 data freeze | Shirley (proposed; Lordphone and Victoria confirm in the PR) |

## 2026-11-18: GPU matrix tiers and cut rule

### Status when decided

- The CPU protocol path (#7), the recovering runner (#11, green CI on Linux) and the cost model (#12) are in place. #11 has not yet run on the RTX 5090, so the 11/15 gate "integrated GPU pilot" is not met.
- The data freeze is 11/25. Plan section 9 already says that an incomplete matrix is reported as "the completed preregistered subset". This entry defines that subset and its order before any GPU data exists, so the cut cannot be chosen after seeing results.

### Estimated cost

Estimates come from the cost model (`analysis/costmodel`, calibrated on the 11/03 spike). They assume a fresh inference worker per run, 20 steps, APC off, the assumed task shape, and 30 s of fixed overhead per run (tool service start, verifier, teardown).

| Prefix | Fault-free run | One-fault run, OutcomeServe / restart, at 10% · 50% · 90% |
| --- | --- | --- |
| 2K | 27 s | 48 / 49 s · 48 / 51 s · 48 / 54 s |
| 8K | 35 s | 56 / 57 s · 56 / 63 s · 56 / 69 s |
| 16K | 49 s | 71 / 72 s · 71 / 83 s · 71 / 95 s |

All four tiers together take about 10.3 GPU hours, far below the plan's 90–120 hour cap.

### Tiers, in execution order

Every tier uses the primary configuration from `infra/experiment-plan.json`: inference-process kill, single-GPU process restart, APC off, 20 steps, and both methods (full restart and OutcomeServe) on the same paired seeds.

| Tier | Cells | Runs | Est. GPU hours | Answers |
| --- | --- | --- | --- | --- |
| 1 | 2K / 8K / 16K × 50% progress, one fault | 3 × 2 × 20 = 120 | 3.1 | RQ2 headline: resume vs restart at each prefix length |
| 1H | 2K / 8K / 16K, no fault | 3 × 2 × 10 = 60 | 1.1 | RQ3: failure-free overhead H for the break-even curve |
| 2 | 2K / 8K / 16K × 90% progress, one fault | 120 | 3.2 | Largest predicted saving S |
| 3 | 2K / 8K / 16K × 10% progress, one fault | 120 | 3.0 | Smallest S; where resume may not pay off |

Tier 1H uses 10 paired seeds instead of 20 because H is a whole-task difference with little variance. If its confidence interval is wider than a quarter of the smallest Tier 1 saving, run 10 more seeds before starting Tier 2.

**Order within a tier.** The outer loop is seeds and the inner loop is prefixes. Each seed runs restart and OutcomeServe back to back on the same fixture. A tier stopped partway therefore still has complete pairs at every prefix length, and never a full 2K column with an empty 16K one.

### Cut rule

"Integrated pilot passes" means one fault-free run and one inference-kill run at 8K / 50% complete on the RTX 5090 using `outcomeserve.runner`. Both must pass `verify_business` with zero duplicate effects, and the committed prefix of the fault run must equal that of the fault-free run.

| Integrated pilot first passes by | Run | Report |
| --- | --- | --- |
| end of 11/22 | All tiers (≈ 10 h), finishing by 11/24; 11/25 is kept for reruns of invalid cells | Full matrix |
| end of 11/24 | Tiers 1 and 1H (≈ 4 h) | Measured 50% cells and H. The 10% and 90% cells appear only as cost-model predictions, labelled as such |
| not by 11/24 | No main GPU matrix | Spike measurements, the CPU correctness matrix and cost-model predictions. The missing GPU matrix is stated as a limitation and is not replaced by smoke runs |

Whatever tier is reached, these rules hold:

- Cells are only ever dropped as a whole tier, in the order above. Retained cells keep 20 paired seeds (plan section 9).
- Invalid or failed runs are reported with counts and reasons, never dropped to make a comparison look better (plan section 9).
- Cost-model predictions are never pooled with measurements. Wherever a cell was measured, the report compares prediction and measurement as a check on the model.
- No new scope is added when time is short: no warm replica, no concurrency, no new baselines (plan section 9).

### Roles

- **Lordphone:** runs the tiers and keeps raw data, as in plan section 6. He decides the exact batch schedule within these rules.
- **Victoria:** pairing, exclusions and confidence intervals.
- **Shirley:** method configuration, and the prediction-vs-measurement comparison for RQ3.
