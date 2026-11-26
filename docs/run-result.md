# Run-result contract (C6) for the main GPU matrix

Owner: Victoria · Consumers: all result owners · Schema: `schemas/run-result.schema.json` version `1.3.0`

Execution/data-return checklist: [RTX 5090 handoff](gpu-handoff.md). The current
1.3.0 handoff requires `task_id` and checked model-mode protocol evidence for a
completed OutcomeServe result. Legacy 1.1.0/1.2.0 results remain schema-valid and
readable by generic analysis; they are not promoted to 1.3.0 formal GPU matrix
samples or figures. Rebuild from retained evidence using a new output path when
the stronger evidence is available; never relabel an old result's version.

This contract says what each main-matrix run must retain so that `scripts/analyze_runs.py` can pair, validate and summarize it without hand edits. It does not say how the runs are executed; batch scheduling, process lifecycle and fault injection stay with their owners (work distribution §3, §6). CPU and existing pilot results keep writing version `1.1.0` and remain valid.

## One directory per run

| File | Content | Producer |
| --- | --- | --- |
| `identity.json` | The fields in the table below, nothing else | Benchmark supervisor |
| `measurements.jsonl` (one or more files) | `measurement-event` 1.1.0 events. Every file of a run uses the same `run_id`, `configuration_id` and `clock_id` | Supervisor, runner |
| `verification.json` | `{"business": verify_business(...), "protocol": verify_workload(..., token_source="model")}`. A completed OutcomeServe run requires a valid completed protocol report with task/context identity and task success. Restart has no protocol report; a bare business report is accepted only for that baseline | Supervisor, after the runner exits |
| `resources.json` | `ResourceMonitor.stop()` summary, including `measurement_valid` | Supervisor |
| `runner-exit.json` | `{"exit_code": <int>, "failure_reason": <string or null>}` of the last runner attempt | Supervisor |
| `runtime.sqlite`, `workload.json`, `context.json` | Consistent database, independent business requests and exact immutable context specification, for independent re-verification | Supervisor |
| `configuration.json`, `resource-samples.jsonl`, attempt/fault logs | Launch/cache configuration, original samples, fault and replacement-worker evidence | Supervisor |
| `pre-fault.sqlite`, `recovery-request.json` | Faulted OutcomeServe snapshot and actual replacement prefix/prompt IDs; see the handoff guide | Supervisor |
| `actual-prefix-lengths.json` | Actual per-step prompt/generated/committed lengths for cost-model calibration | Supervisor |

Event files may also be in the runner's raw format (one JSON object per line with `event_type`, `monotonic_ns`, `clock_id`, other fields). The builder converts those lines with the run's `run_id` and `configuration_id` and keeps their clock; conflicting identity fields already present in a raw line are rejected. The runner must therefore get the supervisor's `clock_id`, or the run is invalid for mixing clocks. Sequence numbers are local to each file. A failed valid protocol check makes the run a `RECOVERY_VIOLATION`. A missing/unreadable/wrong-task/wrong-context/wrong-mode report, or an intermediate checkpoint offered as completion, is incomplete infrastructure evidence and cannot establish success.

Then build the result once; it refuses to overwrite an existing `result.json`:

```bash
python scripts/build_run_result.py <run-dir> --events measurements.jsonl runner-events.jsonl
```

Every event file named by `--events` is required. If a named file is missing, its
name is recorded in `evidence_errors`; a run with an established fault remains a
valid failed run rather than a silently completed or excluded sample. If even
the fault/start evidence is unavailable it is INVALID. For independent rechecks
use `verify_run.py --with-business --token-source model --output <new-file>`
(OutcomeServe) or `--business-only` (restart), then pass
`--verification-file verification-rechecked.json` to the builder. All output
files are created exclusively; original evidence is never overwritten.

Builder exit code 0 means a valid result was assembled, including valid failed runs; inspect `outcome`/`passed` for task success. Exit code 2 denotes invalid evidence or a CLI error. Exclusive file creation prevents concurrent builders from overwriting a result.

## Identity fields

| Field | Values |
| --- | --- |
| `run_id` | Unique; equals the `run_id` in every measurement event |
| `task_id` | This run's task in workload/database/protocol report. It may differ between the two paired methods; the business fixture hash must still match |
| `seed`, `method` | Integer seed; `restart` or `outcomeserve` |
| `tier` | `pilot`, `1`, `1H`, `2`, `3`. Pilot runs get their own pair key (`s<seed>:gpu-pilot:...`) and are excluded from matrix cells, H and S; `gpu-matrix.json` reports their outcomes under `pilot_runs_excluded` |
| `no_fault` | `true` for 1H or a fault-free pilot; never for a one-fault tier |
| `fault_domain`, `boundary` | Main matrix: `inference`, `DURING_GENERATION`. No-fault runs record the same values as the matrix they belong to, because one analysis accepts only one fault domain and boundary |
| `prefix_target_tokens` | `2048`, `8192`, `16384` |
| `requested_progress` | `0.1`, `0.5`, `0.9`; `null` when `no_fault` |
| `actual_progress` | Completed steps at the fault / 20; 1.0 for a completed no-fault task |
| `actual_prefix_tokens_at_fault` | Optional: committed prefix tokens when the fault was applied |
| `apc` | Boolean, always recorded |
| `method_order` | `1` or `2`: position within this seed/prefix/progress pair. The two methods must use different positions. This validates pair-local order only; it does not prove alternation or randomization across seeds, which remains the scheduler's responsibility |
| `configuration_id` | Identifies this run's configuration; every event must match it. Method-specific configuration IDs may differ within a pair |
| `context_spec_id`, `business_fixture_sha256` | Same for both methods of a pair |
| `hardware_id`, `gpu_name`, `driver_version` | GPU UUID, model name and driver. Different GPU models are never pooled |
| `runtime_versions`, `environment_lock_sha256` | As in the pilot |

## Events used for timing

Intervals follow `docs/measurement.md`. An interval is `null` unless both boundaries exist exactly once, in order, on one clock. The first-token metric is the exception to uniqueness: take the earliest token at/after the fault and at/before verified completion, if completion was recorded. A completed one-fault run requires both whole-task and fault-to-verified intervals. A missing/duplicated/inverted completion boundary after an established fault is an infrastructure failure, not grounds to exclude the run as INVALID.

| Result field | From | To |
| --- | --- | --- |
| `task_to_verified_seconds` | `TASK_STARTED` | `VERIFIED_COMPLETE` |
| `fault_to_verified_seconds` (primary) | `FAULT_APPLIED` | `VERIFIED_COMPLETE` |
| `fault_to_first_new_token_seconds` (secondary) | `FAULT_APPLIED` | first `FIRST_NEW_TOKEN` at or after the fault |
| `reset_seconds` | `TOOL_RESET_STARTED` | `TOOL_RESET_FINISHED` |
| `inference_restart_seconds` | `INFERENCE_RESTART_REQUESTED` | `INFERENCE_REPLACEMENT_READY` |

## Outcome

Every run gets exactly one `outcome`. **No outcome is silently excluded from a cell.** INVALID runs remain in raw data, outcome counts and pair diagnostics. Any invalid, valid failed, missing or incompatible pair member suppresses that cell's paired estimate and CI. `complete_pairs` is diagnostic when `estimate` is null; it is not a success-only estimate. `invalid_runs` and `valid_failed_runs` distinguish unusable evidence from recovery/model/infrastructure failures. This rule applies equally to CPU/legacy comparisons, GPU primary/secondary/resource comparisons, H, S and break-even.

Reruns use fresh directories and unique seed/repetition identities, retain all original attempts, and are never silently substituted for an invalid or failed member. Duplicate method entries for a pair are rejected. An unmeasured cell is absent; a measured cell with unusable pairs remains present with null estimates and reasons. Completed-run p95 is explicitly descriptive and is always accompanied by all-run outcome counts.

| Outcome | Rule |
| --- | --- |
| `INVALID` | The fault was not established (a one-fault run without exactly one `FAULT_APPLIED`, or a fault in a no-fault run), fault precedes task start, events mix clocks, runs or configurations, or `TASK_STARTED` is not unique |
| `MODEL_ERROR` | Runner exit code 5: the generated tool call did not match the business request |
| `RECOVERY_VIOLATION` | The business verifier finds duplicate effects, an incomplete effect audit or a committed tool result that disagrees with the oracle (also after a halt), or, for a run that did not abstain, an inventory mismatch; or an identity-matched valid model-mode protocol/replay check fails (OutcomeServe) |
| `ABSTAINED` | Runner exit code 3 with available verification, no duplicate effects, a complete effect audit and no tool-result mismatch. Incomplete final inventory is expected after a halt and does not by itself reclassify abstention as a recovery violation. Query capability alone does not prove availability at decision time; a final snapshot cannot establish false-abstain accuracy |
| `INFRASTRUCTURE_FAILURE` | Missing required event files or protocol evidence, wrong protocol identity/mode/checkpoint, no verification, a missing `VERIFIED_COMPLETE`, incomplete resource sampling, or any other exit |
| `COMPLETED` | Exit code 0, business verification passes for this task with zero duplicates, timing/resources are complete, and OutcomeServe additionally has a valid completed model-mode protocol report with matching task/context and task_success=true. Replay evidence is optional; if supplied, it must pass. The only outcome with `passed: true` |

Exit code 5 for model errors has been requested from the runner owner. Until it exists, a model error surfaces as `INFRASTRUCTURE_FAILURE` with its `failure_reason`, so it is still counted, just not as its own category.

## Analysis outputs

`scripts/analyze_runs.py --results-root <dir> --output <new-dir>` keeps its existing files and adds:

- `outcomes.json`: outcome counts and reasons per method for every cell;
- `gpu-matrix.json`: for each measured (L, f) cell, ΔT (restart − OutcomeServe, fault to verified completion) with a paired 95% bootstrap CI, the secondary first-new-token difference, reset time, descriptive p95, resource differences and outcomes; H(L) from the no-fault pairs; S(L, f) and the break-even q = H / (H + S) from whole-task times; and the Tier 1H precision check;
- `gpu-heatmap.csv`: one row per measured cell for the 3 × 3 figure. Cells that were not measured have no row; predictions are never written into it.

GPU resource differences include process CPU time, GPU allocation-seconds, peak summed RSS and peak GPU memory. Missing resource metrics remain null and suppress only that metric's estimate; values are never fabricated. Sampled GPU utilization remains in the raw resource samples; this pipeline does not infer utilization from allocation time.

One GPU analysis accepts only one GPU model/driver, hardware UUID, context, runtime/environment lock, APC setting and fault domain/boundary. Analyze other configurations separately; H from one configuration must not be combined with S from another. The analyzer rejects mixed input before creating its output directory.

**Tier 1H precision check.** Use 2000 paired bootstrap resamples. Only evaluate when all three Tier 1 prefix cells have positive, usable whole-task savings and all three prefixes have usable H confidence intervals. Otherwise the check is null (undetermined), not a recommendation to stop sampling. If the full width of H's 95% CI, at any prefix length, is larger than 0.25 × the smallest Tier 1 whole-task saving S, report a recommendation for 10 more 1H seeds before Tier 2. This implements the proposed decision rule without changing its sample-size policy or running extra experiments. Break-even q is a point estimate from H/S, not a confidence interval or an observed crash rate.

## Report figures and tables

`scripts/make_figures.py` regenerates every RQ1–RQ3 figure and table from retained results, never from a previous analysis output:

```cmd
uv run scripts\make_figures.py --output run-data\report-figures --correctness run-data\correctness-matrix --gpu-results <gpu results dir>
```

Either input may be left out. The script writes:
- `rq1-correctness.pdf/png` and `rq1-correctness-table.tex`;
- `rq2-delta-t-heatmap.pdf/png`, where unmeasured cells are hatched and labelled, never filled;
- `rq2-resource-table.tex`;
- `rq3-break-even.pdf/png`, with q on a log scale and labelled as an assumption;
- `figures-data.json`, every plotted number.

Rerunning on the same results gives byte-identical files. That is the reproducibility check before the data freeze: regenerate into a new directory and compare `figures-data.json`.

The GPU-free handoff regression in `tests/test_gpu_handoff.py` independently
verifies synthetic SQLite artifacts, builds successful/failed/pilot results,
analyzes them, and compares **all** PNG/PDF/JSON/LaTeX outputs from two figure
generations. Its synthetic fixtures are not GPU observations. Byte identity is
checked within the same installed plotting/dependency environment; keep that
environment pinned when regenerating on another machine.
