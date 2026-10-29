# OutcomeServe

Crash-consistent recovery for agentic LLM serving. CMU 11-868 LLM Systems, Team 33.

When an agent runner or inference worker crashes mid-task, OutcomeServe reconciles each tool call against a durable log under a stable operation ID. It then resumes from the committed token prefix, or stops with `RECOVERY_ABSTAIN` when it cannot establish a tool's outcome.

## Setup

Python 3.12 or 3.13 works for the CPU path. GPU runs use the Python 3.12 inside the pinned vLLM container (see `infra/environment.lock.json`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Check that everything works:

```bash
python -m pytest -q
python scripts/validate_week1_contracts.py
```

## Try a crash and recovery

The CPU path uses a scripted agent and SQLite stand-ins for the log and the tool service, so it needs no GPU. The commands below run in a temporary directory.

```bash
cd "$(mktemp -d)"
python -m outcomeserve.runner --db state.sqlite --task t1 --telemetry events.jsonl --fence --crash-at POST_TOOL_COMMIT_PRE_RESULT_LOG --crash-step 3
python -m outcomeserve.runner --db state.sqlite --task t1 --telemetry events.jsonl --fence --recovery-step 3
```

`--fence` advances the task's owner epoch before each run, which is the supervisor's job once `outcomeserve/supervisor/` exists. The first command kills itself with SIGKILL right after the step 3 tool call commits. The second one reconciles step 3 and finishes the task, and it prints `"status": "COMPLETED"`. Each run appends its events to `events.jsonl`. When `--db` points at a database inside the repository, make sure `*.sqlite` and `*.jsonl` stay out of commits.

## Layout

Each package's `__init__.py` names its owner and reviewer, and `.github/CODEOWNERS` requests reviews to match. The full plan, with its contracts and deadlines, is in [docs/work-distribution.md](docs/work-distribution.md).

| Path | What | Owner |
| --- | --- | --- |
| `outcomeserve/runner/`, `outcomeserve/recovery/` | Agent runner, recovery controller | Shirley |
| `analysis/costmodel/` | Cost model | Shirley |
| `outcomeserve/serving/`, `supervisor/`, `faults/`, `telemetry/` | vLLM serving, supervisor and fencing, fault injection, telemetry | Lordphone |
| `schemas/`, `infra/`, `docs/infrastructure/` | Week 1 contracts and environment lock | Lordphone |
| `outcomeserve/tools/`, `log/`, `verify/` | Tool service, durable log, verifier | Victoria |
| `workloads/`, `analysis/pipeline/` | Workloads, aggregation and plotting | Victoria |

## Working together

- Make every change through a PR. A PR needs one approving reviewer, and CI must pass on it.
- A contract change (C1–C6, or anything in `schemas/`) needs approval from every consumer, and it must bump `schema_version`.
- Each Sunday, add what you finished to [CONTRIBUTIONS.md](CONTRIBUTIONS.md). Decisions that change scope go in [docs/decisions.md](docs/decisions.md).
