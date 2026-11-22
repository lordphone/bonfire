# OutcomeServe

[Proposal](proposal/proposal.tex): persist tool intents, reconcile committed outcomes, and atomically advance context. Rebuild lost inference state from saved token IDs.

| Directory | Purpose |
| --- | --- |
| `outcomeserve/` | Storage, transactional tools, durable logs, independent verification, model interfaces, and resource sampling |
| `experiments/cpu/` | Scripted CPU fault and baseline experiments |
| `experiments/gpu/` | Inference-process restart pilot with a real model |
| `workloads/` | Deterministic inventory tasks |
| `analysis/pipeline/` | Paired statistics and cost analysis |
| `schemas/`, `infra/` | Interface formats, environment lock, and experiment configuration |

[Protocol](docs/protocol.md) | [Measurement](docs/measurement.md) | [Responsibilities](docs/work-distribution.md) | [Results](docs/reports/experiment-results.md)

## Environment and tests

CPU execution requires Python 3.12 or newer. On macOS or Linux, set up and test from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
python scripts/validate_recovery_contracts.py
python scripts/validate_infrastructure_contracts.py
```

Process-tree and resource sampling read `/proc` on Linux and use the Win32 API on Windows. macOS is not supported yet, so the tests that start and kill real worker processes fail there.

In Windows CMD, using the existing .venv:

```cmd
cd /d D:\A\bonfire
.venv\Scripts\python.exe --version
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\python.exe scripts\validate_recovery_contracts.py
.venv\Scripts\python.exe scripts\validate_infrastructure_contracts.py
```

## Live recovery verification

Each output directory must be new. Choose another name when repeating a run. The script creates a database, starts a runner and an independent tool service, executes inventory transactions over loopback HTTP, terminates the runner, recovers, and verifies the result. It does not load saved results. CPU tasks use scripted traces without model inference, so a run normally takes a few seconds.

```cmd
.venv\Scripts\python.exe scripts\run_cpu_experiment.py --output-dir run-data\order-clean --steps 20 --no-fault
.venv\Scripts\python.exe scripts\run_cpu_experiment.py --output-dir run-data\order-recovery --method outcomeserve --steps 20 --progress 0.9 --seed 73
type run-data\order-recovery\precondition.json
type run-data\order-recovery\measurements.jsonl
type run-data\order-recovery\worker-events.jsonl
type run-data\order-recovery\verification.json
```

Expect valid=true, passed=true, and duplicate_effects=0 for recovery. The precondition shows a COMMITTED tool outcome. Worker events show reconciliation returning COMMITTED in the replacement epoch, followed by step completion without resubmission. Tool fault windows require a write; a read at the requested position moves the target to the next write. For 20 steps, a requested 90% position normally resolves to 95%; inspect actual_progress.

Run the independent protocol verifier and an unsafe control:

```cmd
.venv\Scripts\python.exe scripts\verify_run.py --db run-data\order-recovery\runtime.sqlite --workload run-data\order-recovery\workload.json
.venv\Scripts\python.exe scripts\run_cpu_experiment.py --output-dir run-data\order-naive --method naive_retry --steps 20 --progress 0.9 --seed 73
```

At the default post-commit boundary, naive retry should produce valid=true, passed=false, duplicate_effects>0, and exit code 1. This is a detected business error. A normal process exit does not establish correct completion.

Use the same entry point for other cases: `--fault-domain tool_response` drops an acknowledgment; `--boundary` accepts PRE_DISPATCH, POST_TOOL_COMMIT_PRE_RESULT_LOG (default), POST_CONTEXT_COMMIT, DURING_GENERATION, or DURING_RECOVERY.

## Method comparison and analysis

```cmd
.venv\Scripts\python.exe scripts\run_experiment_matrix.py --output-dir run-data\order-correctness --purpose correctness --steps 20 --progress 0.9 --seeds 33
.venv\Scripts\python.exe scripts\run_experiment_matrix.py --output-dir run-data\order-performance --purpose performance --seeds 33 34
.venv\Scripts\python.exe scripts\analyze_runs.py --results-root run-data\order-performance --output run-data\order-analysis
```

Correctness compares four methods while retaining tool state. Performance compares OutcomeServe and restart with and without a fault; restart resets the isolated database and includes reset time. Run directories contain summaries. Analysis produces analysis.json, heatmap-cells.csv, break-even.json, and failure-probability-curve.csv. Two seeds only validate the pipeline.

## GPU pilot

Run `scripts/run_gpu_pilot.py --config <config-file> --output-dir <new-directory>`. Start from infra/gpu-pilot.example.json, replace the local model snapshot path and GPU UUID, and use the Linux/GPU environment pinned by infra/environment.lock.json. The pilot requires an exclusive GPU, NVML, the actual template hash, explicit thinking/APC settings, and exact token-ID and streaming interfaces. Unsupported serving capabilities cause failure.

The experiment plan is in infra/experiment-plan.json. Real GPU measurements have not been collected. CPU data and HTTP mock tests do not establish GPU performance.