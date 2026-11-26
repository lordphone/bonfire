# RTX 5090 handoff: production and verification of GPU results

Victoria owns the artifact contract, independent verification, result assembly,
statistics and figures. Shirley owns runner/recovery semantics. Lordphone owns
the supervisor, real worker lifecycle, fault injection, telemetry and scheduling.
This guide is the interface between those parts; it does not introduce a GPU executor.

## 1. Before collecting data

- Use one agreed source revision including the C6/model-verifier changes. Record
  the commit, any uncommitted diff and a dependency/environment lock in the batch.
- Integrate Shirley's recovering runner into Lordphone's supervisor. The current
  `scripts/run_gpu_pilot.py` is the older 1.1.0 path, not a ready-made executor for
  this handoff. Do not rename its files and assume it has exercised the new runner.
- Keep the RTX 5090 environment, image, model/tokenizer revision, template hash,
  thinking/APC settings and driver/cache policy explicit. Follow
  [environment notes](infrastructure/environment.md); do not pool other hardware.
- Check real `/tokenize` and streaming `token_ids` support. Detect truncation as
  service loss and distinguish model errors from recovery violations. The result
  interface expects exit 0 = completed, 3 = abstained, 5 = model error. Other exits
  stay failed with their reason; they are not successful samples or exclusions.
- Pass one unique per-run `clock_id` from supervisor to runner. Never rewrite
  timestamps or clock IDs during import. Raw runner events are accepted directly.
- Write `identity.json` before launching the attempt, including `task_id`, so a
  failed run can still be retained and assigned to its planned cell.

This handoff produces C6 **1.3.0** results. It tightens the previously published
1.2.0 format instead of silently changing that version. Legacy files remain
readable but are excluded from the formal matrix/figure inputs.

The analysis-only environment needs Python >=3.12, the repository dependencies
and `matplotlib==3.10.7`. In a dedicated analysis environment:

```bash
python -m pip install -e '.[dev]' matplotlib==3.10.7
python -m pytest -q tests/test_gpu_handoff.py tests/test_model_token_verification.py tests/test_report_data.py
```

These tests use synthetic SQLite/token/timing fixtures and no vLLM, GPU, Docker,
worker kills or CPU correctness sweep. Their figures are layout/interface tests,
not experimental observations.

## 2. Integrated pilot gate

Collect four runs at the same 8K target / 50% fault position: OutcomeServe and
restart, each fault-free and with inference loss during generation. Use
`tier: "pilot"` for all four (including the no-fault pilot), pair the workload,
seed, model/template, GPU and cache/start policy, and return the pilot first.

For each run, independent business verification must pass with zero duplicate
effects. OutcomeServe must also pass the **completed model-mode** protocol check.
That check already covers historical replay: after an inference loss, the faulted
step's `STEP_PREPARED` record holds the prompt actually sent to the replacement
worker, and `PROMPT_TOKENS` requires it to extend exactly the durable committed
prefix.

*Optional extra evidence.* If the supervisor also keeps a consistent pre-fault
SQLite backup (`pre-fault.sqlite`) and the first replacement request
(`recovery-request.json`: `task_id`, `context_spec_id`, `prefix_token_ids`,
`prompt_token_ids`, recorded at the request boundary), `verify_run.py
--replay-snapshot/--replay-request` checks the request against the pre-fault
prefix. A failed replay check is a recovery violation; a missing one is not
required.

The supervisor additionally retains old/replacement worker identities, kill and
readiness evidence, explicit `reusable_kv: false`, actual fault progress/prefix
length and GPU identity. Log self-consistency cannot establish real KV loss.
For restart, verify the isolated fixture reset, include reset time in completion
windows, and finish with the same business oracle. No durable-log protocol report
is expected for this baseline.

**Do not require the two runs' final token sequences to match.** Future generation
may differ, as the proposal allows. `--same-prefix-as` is only an optional
deterministic-fixture diagnostic, not the real-model gate. Pilot samples are
counted separately and excluded from H, S and matrix figures.

## 3. Batch plan and retained files

Before measurement, freeze a `batch-plan.json` describing every run directory,
seed/repetition, method, tier, prefix target, no-fault flag/progress and execution
order. Include incomplete and failed planned attempts when returning the batch;
explain directories without usable identities rather than omitting them.

| Tier | Conditions | Repetitions |
| --- | --- | --- |
| 1 | 2K/8K/16K, 50%, both methods | 20 paired seeds per prefix |
| 1H | Same prefixes, no fault, both methods | Declare the agreed sample count; proposed rule is 10 pairs plus 10 if the precision check recommends it |
| 2 | Same prefixes, 90%, both methods | 20 paired seeds per prefix, if retained by the agreed cut rule |
| 3 | Same prefixes, 10%, both methods | 20 paired seeds per prefix, if retained by the agreed cut rule |

Preserve the team's declared tier/cut rule. Do not select cells after observing
their advantage. Alternate or pre-randomize method order by seed; each pair uses
positions 1 and 2 back to back. Pair-local order checks do not prove randomization
across seeds. The supervisor controls actual order and prefix calibration.

Each run directory contains these files, which Victoria's tools read:

| File | Required content |
| --- | --- |
| `identity.json` | All C6 identity fields in [run-result.md](run-result.md), including task ID and both requested and actual progress |
| `runtime.sqlite` | Final consistent database, exported with SQLite backup or after all connections close |
| `workload.json` | Exact independent business requests and initial fixture, not model-generated expectations |
| `measurements.jsonl` | Schema-valid supervisor events with run/configuration/clock IDs and real fault, readiness, reset and completion boundaries |
| `runner-events.jsonl` | Complete raw runner trace across attempts, on the supervisor's clock |
| `resources.json` | Resource summary with `measurement_valid` |
| `runner-exit.json` | Last attempt's exit code and failure reason |
| `verification.json` | Business report, plus the completed model-mode protocol report for OutcomeServe; business only for restart |

The plan (work distribution §6, §10) also has the supervisor keep its own raw
evidence: `context.json`, `configuration.json` (launch, GPU UUID, APC/thinking,
cache and readiness policies), raw resource samples, per-attempt stdout/stderr,
fault and worker logs (kill, readiness, no reusable KV, reset for restart) and
per-step actual token lengths for the cost model. The analysis does not read these
files, but they belong in the returned batch.

Return the whole batch directory unedited, including failures, invalid attempts,
plan, source/dependency provenance and checksums. An `outcome` in a supplied result
does not replace the original evidence. Send pilot first, then Tier 1/1H before
expanding. Victoria should confirm file completeness on receipt before expensive
collection continues. This is a data-interface check, not GPU execution ownership.

## 4. Verification and result production

Run inside the agreed environment from the repository root. For a completed
OutcomeServe run (`RUN` is its directory):

```bash
python scripts/verify_run.py --db "$RUN/runtime.sqlite" --workload "$RUN/workload.json" \
  --token-source model --with-business --output "$RUN/verification-rechecked.json"
```

If the optional replay evidence was kept, add it:

```bash
python scripts/verify_run.py --db "$RUN/runtime.sqlite" --workload "$RUN/workload.json" \
  --token-source model --with-business \
  --replay-snapshot "$RUN/pre-fault.sqlite" --replay-request "$RUN/recovery-request.json" \
  --output "$RUN/verification-rechecked.json"
```

For restart:

```bash
python scripts/verify_run.py --db "$RUN/runtime.sqlite" --workload "$RUN/workload.json" \
  --business-only --output "$RUN/verification-rechecked.json"
```

A verifier return code 1 means a recorded failed check, not permission to drop the
run. Preserve it and build its failed result. Unreadable evidence/CLI errors need
their stderr retained; never replace them with fabricated passing reports. An
abstained run uses `--checkpoint halted` on its protocol check and remains a failed
main-matrix task. Model errors can have an incomplete checkpoint; they retain
exit 5 and their reason. The supervisor emits `VERIFIED_COMPLETE` only after both
required checks pass, with real timestamps (never retroactively edited).

After verification, once per run:

```bash
python scripts/build_run_result.py "$RUN" --verification-file verification-rechecked.json \
  --events measurements.jsonl runner-events.jsonl
```

Both commands refuse to overwrite outputs. Keep the producer's verification and
append an independent recheck. If a `result.json` already exists, use the builder's
`--output` to write into a fresh derived dataset containing only the selected
rechecked result per run. Do not recursively analyze both old and new copies.

Named missing event files produce retained failed results, not silent successes.
Missing identity means the run cannot be assigned: preserve the directory and
resolve it against `batch-plan.json` before calling the dataset complete. Builder
exit 0 means result assembly was valid, not task success; inspect `outcome`.

## 5. Victoria's analysis after receipt

Use a dataset root containing one result per planned run, not all of `run-data`
(which may contain synthetic tests and previous attempts):

```bash
python scripts/analyze_runs.py --results-root "$DATASET" --output "$NEW_ANALYSIS"
python scripts/make_figures.py --gpu-results "$DATASET" --output "$NEW_FIGURES"
```

Add `--correctness "$CPU_MATRIX"` to the figure command only when authoritative
retained RQ1 results are available. RQ1 is not a dependency for inspecting GPU
results. Regenerate to another fresh figure directory and compare every file.

Confirm the declared pair counts against the batch plan, outcomes/invalid counts,
actual fault positions, compatible hardware/context/cache configuration, reset
windows, H/S inputs and missing cells. A missing/invalid/failed pair blocks the
cell estimate; incomplete cells remain visible. p95 is descriptive. q is an
assumed task-failure probability; the break-even curve is a point-estimate
sensitivity analysis, not a measured crash rate. Predictions and synthetic layout
tests never become measured cells.

The software handoff is ready when its offline interface tests pass. That does
not certify the real worker API or restart path: the four-run integrated pilot
and supervisor integration are still Shirley/Lordphone's execution gate.
