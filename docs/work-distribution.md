# OutcomeServe Six-Week Work Distribution

## Objective

Deliver a reproducible prototype and evaluation of crash-consistent recovery for long-horizon, tool-using LLM serving. The system must reconcile durable tool outcomes with the committed conversation prefix, rebuild inference state through vLLM re-prefill, and enter a deterministic `RECOVERY_ABSTAIN` state when the outcome of a non-idempotent external action cannot be established.

Quality takes priority over equal division of hours. Ownership follows subsystem boundaries and the project critical path.

## Scope Boundaries

The project includes:

- fail-stop crashes of the agent runner or inference worker;
- a durable conversation and operation log;
- stable operation IDs and transactional tool-result reconciliation;
- token-prefix reconstruction and vLLM re-prefill;
- deterministic fault and acknowledgment-loss injection;
- commit-aware safe halt through `RECOVERY_ABSTAIN`;
- correctness, recovery-latency, logging-overhead, and resume-versus-restart evaluation.

The project does not include:

- RL training or learned abstention policies;
- direct KV-cache replication or live migration;
- complete sandbox or filesystem rollback;
- multi-tenant scheduling or general service high availability;
- exactly-once guarantees for arbitrary external APIs;
- recovery from durable-log, transactional-database, or complete-node loss.

## System Invariant

For every completed agent step, the following must represent the same logical action:

1. one unique tool-side outcome;
2. one durable tool result in the transcript;
3. one committed token-prefix advancement;
4. one resumable model context.

Recovery behavior is deterministic:

| Reconciled outcome | Required behavior |
| --- | --- |
| `COMMITTED` | Retrieve the stored result, commit it to context if necessary, and resume. |
| `NOT_COMMITTED` | Execute once under the stable operation ID, commit the result, and resume. |
| `UNKNOWN` | Persist `RECOVERY_ABSTAIN`, stop before re-execution, and surface the operation ID and reason for manual resolution. |

## Primary Ownership

| Workstream | Directly responsible | Primary responsibilities | Required reviewer |
| --- | --- | --- | --- |
| Agent runtime and recovery | Shirley Li | Agent runner, vLLM integration, prefix reconstruction, re-prefill, recovery controller, and `UNKNOWN -> RECOVERY_ABSTAIN` transition | Victoria Zhu for recovery protocol; Lordphone Wen for vLLM integration |
| Infrastructure and performance | Lordphone Wen | CPU/GPU environment, worker supervision, deterministic fault injection, acknowledgment-loss injection, hardware telemetry, and performance benchmarks | Shirley Li for fault injection; Victoria Zhu for performance methodology |
| Durable state and correctness | Victoria Zhu | Transactional tool service, durable-log schema, operation-ID reconciliation, trace reconstruction, invariant checking, paired workloads, and metric aggregation | Shirley Li for tool semantics; Lordphone Wen for metrics |
| Integration and delivery | All members | End-to-end integration, experiments, analysis, demo, report, and presentation | Cross-review by all members |

## Interface Freeze

The following interfaces must be frozen by the end of Week 1. Later changes require an interface-change note and review from both affected owners.

### Operation identity

Each tool action uses a deterministic operation ID derived from at least:

- `task_id`;
- `step_id`;
- tool name;
- canonical arguments hash.

Reusing an operation ID with different arguments is an error, not a retry.

### Durable record

Each step record must contain:

- task and step IDs;
- operation ID;
- tool name and canonical arguments hash;
- operation status and stored result;
- committed token boundary;
- model ID and version;
- recovery status;
- preparation, execution, commit, failure, and recovery timestamps.

### Fault injection

Every injected failure must be addressable by:

- experiment ID;
- task and step ID;
- failure boundary;
- deterministic seed;
- worker process ID;
- injection timestamp.

### Experiment record

Every measured run must record:

- Git commit SHA;
- complete configuration;
- workload and failure seeds;
- model and hardware identifiers;
- raw timing and correctness events;
- validity status and any exclusion reason.

### Baseline semantics

- External tool state persists across crashes for all correctness comparisons.
- Full restart must not silently reset inventory, orders, or other external effects.
- A resettable-environment restart is a separately labeled recomputation experiment, not the correctness baseline.
- Reset time is measured as `T_reset` and reported separately rather than treated as free.
- Comparable methods use the same model, tools, task seed, failure point, and worker-start policy.

## Six-Week Timeline

### Week 1: Architecture, Interfaces, and Acceptance Criteria

**Goal:** Remove ambiguity before implementation begins.

#### Shirley Li

- Define the runner lifecycle: task creation, generation, prepare, dispatch, reconcile, context commit, recovery, completion, and abstention.
- Specify the recovery-controller inputs, outputs, and legal state transitions.
- Define which uncommitted generation may be discarded after a crash.
- Specify token-ID, model-version, and committed-prefix persistence for re-prefill.
- Write state-machine and recovery-protocol pseudocode.

#### Lordphone Wen

- Pin Python, PyTorch, CUDA, vLLM, and Qwen3-4B versions.
- Document CPU and GPU execution environments.
- Specify worker launch, health detection, crash detection, and replacement behavior.
- Define deterministic fault-injection and acknowledgment-loss APIs.
- Define measurement boundaries for worker startup, reconciliation, re-prefill, and crash-to-next-token latency.

#### Victoria Zhu

- Freeze the durable-log schema.
- Specify the transactional tool-service API and operation-ID contract.
- Define `PREPARED`, `COMMITTED`, `NOT_COMMITTED`, `UNKNOWN`, and `RECOVERY_ABSTAIN` semantics.
- Define the cross-layer invariant checker and final-state verifier.
- Specify paired should-resume and should-abstain workloads.

#### Shared deliverables

- architecture diagram;
- recovery state machine;
- interface specification;
- durable-log schema;
- fault-boundary catalog;
- experiment configuration schema;
- repository structure and CPU smoke-test entry point.

#### Exit gate

- Each crash boundary has an expected post-recovery state.
- Operation-ID generation and collision behavior are unambiguous.
- `PREPARED` is explicitly documented as insufficient evidence of non-execution.
- Resume and abstain decisions are deterministic.
- All baselines use the same model, tool implementation, workload seed, and worker-start policy.

### Week 2: Independent Component Implementation

**Goal:** Complete independently testable CPU components before integration.

#### Shirley Li — scripted agent runtime

- Implement a deterministic scripted-model backend.
- Implement the agent execution loop and step progression.
- Implement tool-call preparation and durable intent recording.
- Implement the recovery controller and transcript reconstruction.
- Implement the terminal `RECOVERY_ABSTAIN` path.
- Test `COMMITTED`, `NOT_COMMITTED`, and `UNKNOWN` decisions independently.

#### Lordphone Wen — supervision and fault injection

- Implement the worker launcher and health monitor.
- Implement deterministic process termination at a requested task step.
- Implement acknowledgment-loss injection without undoing the tool-side effect.
- Implement replacement-worker startup and lifecycle logging.
- Implement CPU/GPU telemetry collection keyed by experiment ID.
- Verify that the same seed triggers the same failure boundary.

#### Victoria Zhu — transactional tool service

- Implement stable-ID execution using database transactions.
- Store the side effect and result atomically for supported tools.
- Return the stored result for duplicate submissions.
- Reject the same operation ID with different canonical arguments.
- Implement auditable read, idempotent-write, and non-idempotent-write tools.
- Implement an external-tool mock that can deliberately return `UNKNOWN`.

#### Shared deliverables

- independently passing component tests;
- deterministic five-step workload;
- auditable inventory and order state;
- structured event traces;
- repeatable worker failure injection.

#### Exit gate

The team must demonstrate:

1. a tool side effect commits;
2. its acknowledgment is lost;
3. the runner crashes;
4. the replacement runner queries the same operation ID;
5. the original stored result is returned;
6. no duplicate side effect occurs.

### Week 3: CPU End-to-End Crash Consistency

**Goal:** Prove protocol correctness before spending GPU time.

#### Shirley Li

- Integrate the runner, recovery controller, durable log, and tool service.
- Implement committed-prefix advancement.
- Prevent generation from continuing before reconciliation completes.
- Implement full-restart, naive-retry, always-halt, and transcript-only baselines.
- Emit every recovery state transition to the structured trace.

#### Lordphone Wen

- Integrate the fault injector with the complete runner.
- Exercise failures during generation, before dispatch, after tool commit but before result logging, and after context commit.
- Exercise an additional crash during recovery.
- Attach configuration hash, workload seed, failure seed, and process IDs to each run.
- Provide a one-command CPU correctness suite.

#### Victoria Zhu

- Implement the cross-layer invariant checker.
- Implement paired should-resume and should-abstain cases.
- Calculate duplicate-effect, state-mismatch, unsafe-retry, correct-resume, false-abstention, manual-escalation, and task-success metrics.
- Produce human-readable diagnostics for every invariant violation.
- Audit baseline behavior against the same final-state verifier.

#### Correctness matrix

- task lengths: 5, 10, and 20 tool steps;
- failures: generation, pre-dispatch, post-tool-commit/pre-result-log, and post-context-commit;
- outcomes: `COMMITTED`, `NOT_COMMITTED`, and `UNKNOWN` where applicable;
- policies: full restart, naive retry, always halt, transcript-only, and OutcomeServe;
- at least 50 deterministic paired seeds per applicable correctness cell;
- identical seeds across policies within each cell;
- exact 95% Clopper-Pearson intervals for binary error rates, including the upper confidence bound when zero failures are observed.

#### Exit gate

- Zero observed duplicate effects for supported tools.
- Zero unsafe retries for `UNKNOWN` outcomes.
- No false abstention in should-resume cases.
- Transcript and tool state agree after recovery.
- Recovery remains safe when the recovery path crashes again.
- The complete CPU correctness suite passes repeatedly before GPU experiments begin.

### Week 4: vLLM Integration and Performance Instrumentation

**Goal:** Connect the verified protocol to real inference and produce trustworthy measurements.

#### Shirley Li

- Integrate Qwen3-4B with vLLM.
- Persist exact token IDs and the model identifier.
- Reconstruct the committed prefix after worker replacement.
- Submit the recovered prefix for vLLM re-prefill.
- Validate 2K-token recovery before extending to 8K and 16K.
- Confirm that correctness does not rely on bitwise-identical future sampling.

#### Lordphone Wen

- Establish the GPU worker lifecycle and replacement path.
- Measure worker startup, reconciliation, re-prefill, and first-token timing separately.
- Capture GPU utilization, memory use, and re-prefill throughput.
- Keep worker-start policy identical across comparable methods.
- Label cold restart, process restart, and warm replacement as separate conditions.
- Automate batches and raw-result collection without manual log copying.

#### Victoria Zhu

- Validate real-model traces against the same cross-layer invariant.
- Verify alignment between token-prefix and tool-result boundaries.
- Build the unified metrics pipeline.
- Detect incomplete, corrupt, or configuration-mismatched runs.
- Preserve excluded runs and explicit exclusion reasons.

#### Performance smoke matrix

- prefixes: 2K, 8K, and 16K tokens;
- task horizons: 5 and 20 tool steps;
- failures: early and late in the task;
- methods: full restart, transcript-only, and OutcomeServe;
- controls: failure-free runs with and without durable logging.

#### Exit gate

- All three prefix lengths recover successfully.
- Recovery latency is decomposed into startup, reconciliation, and re-prefill.
- Re-prefilled tokens, regenerated tokens, and repeated tool work are measurable.
- Telemetry aligns with the correct experiment IDs.
- At least one complete batch runs without manual intervention.

### Week 5: Formal Experiments and Data Freeze

**Goal:** Produce the complete evidence used in the report.

#### Correctness evaluation — Victoria leads

- Run every applicable Week 3 correctness cell with at least 50 paired deterministic seeds.
- Audit duplicate effects, state mismatches, unsafe retries, correct resumes, false abstentions, task success, and manual escalations.
- Explain why naive retry, always halt, and transcript-only fail in their expected cases.
- Have Shirley review every protocol-level interpretation.
- Have Lordphone verify that the intended fault actually occurred in each run.
- Report exact 95% Clopper-Pearson intervals for binary error rates and the upper confidence bound for every zero-observed-error claim.

#### Recovery performance — Lordphone leads

- Evaluate 2K, 8K, and 16K prefixes.
- Evaluate approximately 25%, 50%, and 75% completed-task fractions.
- Compare full restart, transcript-only, and OutcomeServe.
- Use identical task seeds and failure points across methods.
- Run at least 20 repetitions for each selected headline condition.
- Report p50 and p95 recovery latency, end-to-end latency, regenerated tokens, re-prefilled tokens, repeated tool work, GPU utilization, and re-prefill throughput.

#### Resume-versus-restart crossover — Shirley and Lordphone

Measure:

```text
T_resume  = T_startup + T_reconcile + T_prefill(L)
T_restart = T_startup + T_regenerate(k) + T_tools(k) + T_reset
```

Report a heatmap of:

```text
Delta T(L, k) = T_restart - T_resume
```

A positive value means resume is faster. Reset cost must be reported separately and never treated as free for external state.

#### Failure-free overhead — Lordphone leads

Compare logging disabled, durable logging enabled, and the complete OutcomeServe protocol using identical workloads. Report end-to-end latency, tokens per second, per-step tool latency, database-write latency, and GPU utilization.

#### Data-quality rules

- Generate every figure from retained structured raw data.
- Never hand-edit result CSV files.
- Store configuration, seed, commit SHA, hardware, and model identity with every run.
- Retain invalid runs and document exclusion reasons.
- Use paired seeds and confidence intervals, not averages alone.
- Freeze the primary dataset at the end of the week.

#### Exit gate

- Every planned headline claim maps to a frozen result.
- Every figure can be regenerated automatically.
- Confidence intervals and run counts are reported.
- No baseline receives an unfair environment reset or worker-start advantage.

### Week 6: Audit, Analysis, Demo, and Final Delivery

**Goal:** Stabilize the artifact and communicate only evidence-supported conclusions.

#### Monday — data audit

- Check for missing conditions, configuration drift, incomplete traces, and mixed hardware.
- Identify conclusions with insufficient evidence.
- Review all excluded runs.

#### Tuesday — bounded reruns

Rerun only conditions with missing data, unexpectedly wide confidence intervals, invalid baseline configuration, or incomplete recovery traces. Do not add new system features.

#### Wednesday — code and experiment freeze

- freeze release configuration;
- complete environment and dependency documentation;
- provide one-command CPU smoke test;
- provide one-command demo;
- provide one-command reproduction for selected headline results;
- produce the artifact manifest and release tag.

#### Thursday — report and figures

Complete:

1. system architecture diagram;
2. recovery state machine;
3. correctness comparison table;
4. crash-to-next-token latency plot;
5. recovery-time decomposition;
6. failure-free overhead plot;
7. resume-versus-restart crossover heatmap;
8. limitations and reproducibility sections.

#### Friday — demo and presentation

Demo two fixed scenarios:

1. **Commit-aware resume:** the tool commits, its acknowledgment is lost, the worker crashes, the replacement runner retrieves the original result by operation ID, re-prefills, resumes, and produces no duplicate effect.
2. **Commit-aware safe halt:** a non-idempotent external action is dispatched, its result cannot be queried, the worker crashes, recovery obtains `UNKNOWN`, persists `RECOVERY_ABSTAIN`, and stops without retrying.

#### Final exit gate

- A fresh environment can run the CPU smoke test from the README.
- The demo requires no manual database edits.
- All report figures regenerate from committed scripts and retained data.
- Every headline claim cites an experiment and configuration.
- Limitations explicitly state the surviving durable-service and single-runner assumptions.
- The report does not claim arbitrary exactly-once execution, direct KV-cache recovery, or a learned abstention policy.

## Review and Integration Rules

- The change-type review matrix below is authoritative when a workstream spans multiple review specialties.
- Feature work lands through pull requests, not direct commits to `main`.
- Every pull request states how to test the change and what result to expect.
- Protocol or API changes update the design documentation in the same pull request.
- Performance changes define measurement start and end boundaries.
- Experiment-result changes include configuration and commit SHA.
- Each critical subsystem receives review from the neighboring subsystem owner.

| Change type | Author | Required reviewer |
| --- | --- | --- |
| Recovery protocol | Shirley | Victoria |
| Durable-log or tool semantics | Victoria | Shirley |
| Fault injection | Lordphone | Shirley |
| Performance methodology | Lordphone | Victoria |
| Correctness metrics | Victoria | Lordphone |
| vLLM integration | Shirley | Lordphone |
| Final claims | Any member | All members |

## Weekly Operating Cadence

- **Monday:** Fix the weekly exit gate and the pull requests required to reach it.
- **Wednesday:** Run an end-to-end integration check rather than separate subsystem status reports.
- **Friday:** Demonstrate the milestone against the exit gate. Do not advance an unverified dependency to the next phase.
- **Daily asynchronous update:** completed work, next action, and blocking issue.

## Scope Reduction Order

If the schedule slips, preserve work in this order:

1. stable operation IDs and transactional reconciliation;
2. durable log and committed prefix;
3. core `UNKNOWN -> RECOVERY_ABSTAIN` safe halt and paired resume/abstain correctness cases;
4. crash-recovery correctness;
5. full-restart, naive-retry, and OutcomeServe comparisons;
6. real vLLM re-prefill;
7. correctness and recovery-latency evaluation;
8. resume-versus-restart crossover analysis;
9. warm replacement, additional abstention workloads, and snapshot-interval ablations.

Never preserve an extension by weakening the first four items. The minimum high-quality result is a correct protocol, deterministic failures, independently verified outcomes, and reproducible experiments.

## Definition of Done

The project is complete only when:

- supported tool operations show zero observed duplicate effects in the evaluated matrix;
- `UNKNOWN` outcomes never trigger blind re-execution;
- transcript, tool state, and committed prefix agree after recovery;
- CPU correctness tests and selected GPU experiments are reproducible;
- baselines use fair and documented policies;
- recovery costs are decomposed and reported with uncertainty;
- the crossover analysis identifies where resume ceases to outperform restart;
- the artifact includes setup, smoke-test, demo, experiment, and figure-generation instructions;
- the report separates verified results from limitations and future work.
