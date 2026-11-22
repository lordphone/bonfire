# Run-result contract (C6) for the main GPU matrix

Owner: Victoria · Consumers: all result owners · Schema: `schemas/run-result.schema.json` version `1.2.0`

This contract says what each main-matrix run must retain so that `scripts/analyze_runs.py` can pair, validate and summarize it without hand edits. It does not say how the runs are executed; batch scheduling, process lifecycle and fault injection stay with their owners (work distribution §3, §6). CPU and existing pilot results keep writing version `1.1.0` and remain valid.

## One directory per run

| File | Content | Producer |
| --- | --- | --- |
| `identity.json` | The fields in the table below, nothing else | Benchmark supervisor |
| `measurements.jsonl` (one or more files) | `measurement-event` 1.1.0 events. Every file of a run uses the same `run_id`, `configuration_id` and `clock_id` | Supervisor, runner |
| `verification.json` | Output of `verify_business` (either bare or as `{"business": ...}`) | Supervisor, after the runner exits |
| `resources.json` | `ResourceMonitor.stop()` summary, including `measurement_valid` | Supervisor |
| `runner-exit.json` | `{"exit_code": <int>, "failure_reason": <string or null>}` of the last runner attempt | Supervisor |

Raw `runner-events.jsonl` from Shirley's runner is not a `measurement-event` stream: the supervisor must retain it and supply schema-valid measurement events with run/configuration identity. Sequence numbers are local to each producer; timing uses their shared monotonic clock. This change does not implement that supervisor adapter.

Then build the result once; it refuses to overwrite an existing `result.json`:

```bash
python scripts/build_run_result.py <run-dir> --events measurements.jsonl runner-measurements.jsonl
```

Builder exit code 0 means a valid result was assembled, including valid failed runs; inspect `outcome`/`passed` for task success. Exit code 2 denotes invalid evidence or a CLI error. Exclusive file creation prevents concurrent builders from overwriting a result.

## Identity fields

| Field | Values |
| --- | --- |
| `run_id` | Unique; equals the `run_id` in every measurement event |
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
| `RECOVERY_VIOLATION` | The verifier finds duplicate effects, an incomplete effect audit or a committed tool result that disagrees with the oracle (also after a halt), or, for a run that did not abstain, an inventory mismatch |
| `ABSTAINED` | Runner exit code 3 with available verification, no duplicate effects, a complete effect audit and no tool-result mismatch. Incomplete final inventory is expected after a halt and does not by itself reclassify abstention as a recovery violation. Query capability alone does not prove availability at decision time; a final snapshot cannot establish false-abstain accuracy |
| `INFRASTRUCTURE_FAILURE` | No verification, a missing `VERIFIED_COMPLETE`, incomplete resource sampling, or any other exit |
| `COMPLETED` | Exit code 0, the verifier passes with zero duplicates, and the timing window is complete. The only outcome with `passed: true` |

Exit code 5 for model errors has been requested from the runner owner. Until it exists, a model error surfaces as `INFRASTRUCTURE_FAILURE` with its `failure_reason`, so it is still counted, just not as its own category.

## Analysis outputs

`scripts/analyze_runs.py --results-root <dir> --output <new-dir>` keeps its existing files and adds:

- `outcomes.json`: outcome counts and reasons per method for every cell;
- `gpu-matrix.json`: for each measured (L, f) cell, ΔT (restart − OutcomeServe, fault to verified completion) with a paired 95% bootstrap CI, the secondary first-new-token difference, reset time, descriptive p95, resource differences and outcomes; H(L) from the no-fault pairs; S(L, f) and the break-even q = H / (H + S) from whole-task times; and the Tier 1H precision check;
- `gpu-heatmap.csv`: one row per measured cell for the 3 × 3 figure. Cells that were not measured have no row; predictions are never written into it.

GPU resource differences include process CPU time, GPU allocation-seconds, peak summed RSS and peak GPU memory. Missing resource metrics remain null and suppress only that metric's estimate; values are never fabricated. Sampled GPU utilization remains in the raw resource samples; this pipeline does not infer utilization from allocation time.

One GPU analysis accepts only one GPU model/driver, hardware UUID, context, runtime/environment lock, APC setting and fault domain/boundary. Analyze other configurations separately; H from one configuration must not be combined with S from another. The analyzer rejects mixed input before creating its output directory.

**Tier 1H precision check.** Use 2000 paired bootstrap resamples. Only evaluate when all three Tier 1 prefix cells have positive, usable whole-task savings and all three prefixes have usable H confidence intervals. Otherwise the check is null (undetermined), not a recommendation to stop sampling. If the full width of H's 95% CI, at any prefix length, is larger than 0.25 × the smallest Tier 1 whole-task saving S, report a recommendation for 10 more 1H seeds before Tier 2. This implements the proposed decision rule without changing its sample-size policy or running extra experiments. Break-even q is a point estimate from H/S, not a confidence interval or an observed crash rate.
