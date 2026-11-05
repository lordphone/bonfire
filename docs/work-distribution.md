# OutcomeServe Work Distribution — Proposal-Aligned Version

Shirley Li · 2026-11-04

> 本文件以最新版 proposal《OutcomeServe: Crash-Consistent Recovery for Long-Horizon Agentic LLM Serving》和 PR #7/#8 的实际实现为范围基准。它取代旧版中关于 warm replica、KV recovery、decode checkpoint、并发调度和完整 LangGraph 集成的安排；计划中的 ownership 不代替已合并代码、review 和实验记录所证明的实际贡献。

## 1. 项目目标与范围

OutcomeServe 要回答一个明确的系统问题：当 agent runner 或 inference process 在长任务中崩溃时，系统能否在不重复外部工具副作用的前提下，从 durable transcript 恢复，并减少相对 full restart 的完成时间和资源消耗。

### Must scope

- runner crash 与 inference-process crash 两个故障域；
- durable log、stable operation ID、transactional tool outcome reconciliation；
- `COMMITTED`、`NOT_COMMITTED`、`UNKNOWN` 三种恢复结果；
- `UNKNOWN -> RECOVERY_ABSTAIN`，在后续 state-changing action 前停止；
- 保存 token IDs、model revision 和 template version，重建 committed prefix；
- inference process 丢失 KV 后，用真实 vLLM/Qwen3-4B 完成 re-prefill；
- RTX 5090 作为当前实测 GPU 平台；PSC H100/L40S 只作为备选平台，数据不跨硬件混合；
- APC 必须作为显式配置记录；runner-crash diagnostic 报告 APC on/off，主 inference-restart 对比不假设可复用 KV；
- CPU correctness matrix；
- full restart、naive retry、transcript-only checkpointing、OutcomeServe 四种方法；
- end-to-end latency、CPU/RAM/GPU resource metrics 和 failure-free overhead；
- paired seeds、重复测量、置信区间和假设故障概率下的 break-even 分析。

### Non-goals

- scheduler 或 multi-subagent dependency scheduling；
- sandbox/filesystem rollback；
- 精确 hardware cycles、FLOPs 或能耗测量；
- 完整 LangGraph/Temporal framework integration；
- 真实生产 crash rate 的测量或推断；
- durable storage、tool database 与 runner 同时丢失的恢复保证。

### Future work（不进入当前交付范围）

- warm replica 或 live migration；
- KV-cache replication、LMCache 或其他 KV restore path；
- decode checkpoint；
- concurrency sweep 和 scheduler policy。

PR #8 的 spike 证明 cold worker startup 在 RTX 5090 上主导 inference-process recovery，也证明 APC 会显著改变 runner-only recovery；这些结果用于解释当前 cold-restart 路径和未来方向，不自动把 warm replica 或 decode checkpoint 加回当前 scope。

## 2. 三个研究问题与结果归属

每个研究问题有一个结果 owner，但跨模块证据由三人共同提供。每个人独立负责期末报告中的一个主要结果小节。

| Research question | Result owner | Supporting owners | 核心输出 |
| --- | --- | --- | --- |
| **RQ1 — Correctness:** 在各故障边界下，OutcomeServe 是否避免重复副作用、保持 transcript/tool state 一致，并在结果不可判定时安全停止？ | Victoria | Shirley：协议与恢复控制；Lordphone：故障确实发生的证据 | correctness matrix、invariant violations、`UNKNOWN`/abstain 结果 |
| **RQ2 — Recovery cost:** 在 2K/8K/16K prefix 和 10/50/90% task progress 下，resume 相对 restart 节省多少完成时间和资源？ | Lordphone | Shirley：恢复路径与 baselines；Victoria：配对和统计 | 9-cell heatmap、latency decomposition、resource table |
| **RQ3 — Overhead and break-even:** durable recovery 的 failure-free overhead 是多少；在什么假设故障概率下收益超过开销？ | Shirley | Lordphone：原始资源测量；Victoria：置信区间与可复现分析 | logging-overhead figure、break-even curve、protocol/cost interpretation |

## 3. Primary ownership

### Shirley — runtime, protocol, vLLM recovery integration

Shirley 对 protocol semantics 和 end-to-end recovery path 负责，不单独承担全部 storage 或 benchmark 实现。

**Owns**

- agent runner 和 recovery controller；
- tool contract 的语义：stable operation ID、argument fingerprint、query/dedup 能力和 `UNKNOWN` 行为；
- `prepare -> reconcile -> commit context / abstain` 状态机；
- vLLM recovery integration：将 Victoria 重建的 committed token prefix 提交给 inference worker；
- full restart、naive retry、transcript-only 和 OutcomeServe 模式；
- RQ3 cost model、break-even 定义和最终解释；
- 全项目接口协调、proposal/report 方法部分和最终统稿。

**Does not own**

- tool database 和 durable log 的具体存储实现；
- worker process launch/restart、GPU monitoring 或 benchmark batch execution；
- 统计分析流水线。

**Primary reviewers**

- Victoria reviews tool semantics、durable record interaction 和 abstention behavior；
- Lordphone reviews vLLM client boundary、worker readiness 和 timing events。

### Lordphone — process lifecycle, faults, resources, GPU benchmarks

Lordphone 对“故障是否按定义发生”和“性能数据是否可比较”负责。

**Owns**

- inference worker 与 runner 的 process lifecycle、startup、readiness、termination 和 restart；
- deterministic fault injection，包括 runner boundaries 和 inference-process crash；
- supervisor/fencing，防止旧 runner 在恢复后继续修改状态；
- CPU time、RSS、GPU memory、sampled utilization 和 GPU allocation-seconds 采集；
- RTX 5090 spike、planned main matrix 的自动化和原始数据保留；
- startup、reconciliation、prefill、first-new-token、verified-completion 时间分解；
- RQ2 performance/resource result section。

**Does not own**

- tool commit/reconciliation 的正确性定义；
- transcript prefix 的语义重建；
- 数据排除规则或最终统计方法的单方面决定。

**Primary reviewers**

- Shirley reviews lifecycle 与 recovery protocol 的接口；
- Victoria reviews measurement schema、paired-run identity 和 invalid-run evidence。

### Victoria — durable state, prefix reconstruction, verification, statistics

Victoria 对“恢复出来的状态是否正确”和“结论能否从原始数据复现”负责。

**Owns**

- durable log implementation 和 transaction boundaries；
- transactional tool service implementation，包括 stable-ID dedup、stored result、fingerprint mismatch 和 query API；
- 从 durable records 重建 canonical committed token prefix；
- transcript/tool-state invariant checker 和 final-state verifier；
- scripted CPU workloads、paired should-resume/should-abstain cases；
- run-result schema、aggregation pipeline、bootstrap confidence intervals 和 descriptive p95；
- RQ1 correctness result section，以及 RQ2/RQ3 的统计复核。

**Does not own**

- tool contract 的最终 protocol semantics；
- process-level fault injection 或 GPU worker restart；
- vLLM serving configuration 的最终决定。

**Primary reviewers**

- Shirley reviews tool behavior、prefix boundary 和 recovery-facing API；
- Lordphone reviews timestamps、resource samples 和 fault evidence。

### Shared responsibilities

- end-to-end integration；
- experiment reruns and anomaly review；
- demo、poster/presentation 和 final report；
- 每人维护自己的 `CONTRIBUTIONS.md` 记录；
- 任何会改变 proposal scope 的决定都写入 `docs/decisions.md`。

## 4. Module owner and reviewer map

| Module / artifact | Owner | Required reviewer | Path |
| --- | --- | --- | --- |
| Agent runner | Shirley | Victoria | `outcomeserve/runner/` |
| Recovery controller and baselines | Shirley | Victoria | `outcomeserve/recovery/` |
| Tool contract semantics and cost model | Shirley | Victoria / Lordphone | `docs/protocol.md`, `analysis/costmodel/` |
| vLLM recovery client and prefix submission | Shirley | Lordphone | `outcomeserve/serving/` client boundary |
| Worker lifecycle and supervisor | Lordphone | Shirley | `outcomeserve/serving/`, `outcomeserve/supervisor/` |
| Fault injection and fencing | Lordphone | Shirley / Victoria | `outcomeserve/faults/`, fault schemas |
| Resource telemetry | Lordphone | Victoria | `outcomeserve/telemetry/`, measurement schemas |
| GPU spike, pilot and benchmark automation | Lordphone | Shirley / Victoria | `experiments/spike/`, `experiments/gpu/`, benchmark scripts |
| Transactional tool service | Victoria | Shirley | `outcomeserve/tools/` |
| Durable log | Victoria | Shirley | `outcomeserve/log/` |
| Canonical prefix reconstruction | Victoria | Shirley | log/context reconstruction code |
| Workloads and final-state verifier | Victoria | Shirley | `workloads/`, `outcomeserve/verify/` |
| Aggregation and figures | Victoria | Lordphone | `analysis/pipeline/`, analysis scripts |
| Report integration and final claims | Shirley | All | report/presentation artifacts |

An owner writes the implementation and acceptance evidence. A reviewer checks the contract and result; review does not transfer ownership.

## 5. Cross-person interface contracts

### C1 — Durable record schema

- **Owner:** Victoria
- **Consumers:** Shirley recovery controller; Lordphone telemetry correlation
- **Must include:** `task_id`, `step_id`, `op_id`, tool name, canonical argument fingerprint, owner epoch, record type, exact token IDs, result, model revision and immutable `context_spec_id`
- **Required records:** `STEP_PREPARED`, `TOOL_RESULT_COMMITTED`, `CONTEXT_COMMITTED`, `RECOVERY_ABSTAIN`
- **Acceptance:** recovery can reconstruct the last committed boundary without reading process memory

### C2 — Tool contract

- **Semantic owner:** Shirley
- **Implementation owner:** Victoria
- **Must define:** `submit`, `query`, stable-ID deduplication, fingerprint mismatch and tool capability metadata
- **Required outcomes:** `COMMITTED`, `NOT_COMMITTED`, `UNKNOWN`, `STALE_EPOCH`, `FINGERPRINT_MISMATCH`
- **Acceptance:** an ambiguous non-idempotent action never triggers blind re-execution

### C3 — Runner/inference boundary

- **Recovery owner:** Shirley
- **Lifecycle owner:** Lordphone
- **Must define:** exact token-ID input, streaming output identity, model/template configuration, readiness, restart completion and no-reusable-KV condition
- **Acceptance:** after inference-process restart, the same committed prefix is submitted and actual re-prefill is measured

### C4 — Fault injection

- **Owner:** Lordphone
- **Consumers:** Shirley controller; Victoria verifier
- **Required boundaries:** `PRE_DISPATCH`, `POST_TOOL_COMMIT_PRE_RESULT_LOG`, `POST_CONTEXT_COMMIT`, `DURING_RECOVERY` and `DURING_GENERATION`
- **Acceptance:** every run records evidence that the requested boundary was reached before the process was terminated or acknowledgment was dropped

### C5 — Measurement events

- **Owner:** Lordphone
- **Analysis owner:** Victoria
- **Required events:** task start, fault/process exit, worker start/ready, reconciliation start/end, prefill start/end, first new token, verified completion and resource samples
- **Acceptance:** all comparable methods use the same measurement window and process set

### C6 — Run result and statistics

- **Owner:** Victoria
- **Consumers:** all result owners
- **Must include:** task/seed/fault/method identity, configuration ID, hardware/model identity, validity status, exclusion reason, correctness outputs, timing decomposition and resource summary
- **Acceptance:** every table and figure can be regenerated from retained structured data without hand-edited CSVs

Any contract change requires the owner, every consumer and at least one reviewer to approve it. Schema changes increment `schema_version`.

## 6. Experiment responsibility

### CPU correctness matrix — Victoria leads

**Shirley provides** protocol implementations and four method modes.

**Lordphone provides** deterministic fault hooks and evidence.

**Victoria runs and analyzes** the matrix.

Required coverage:

- task lengths: 5, 10 and 20 steps;
- supported idempotent/deduplicated tools and an unqueryable non-idempotent case;
- five deterministic boundaries: pre-dispatch, post-tool-commit/pre-result-log, post-result-log, during recovery and inference-generation simulation;
- methods: naive retry, full restart, transcript-only and OutcomeServe;
- verifier outputs: duplicates, transcript/tool mismatch, unsafe retry, abstention and task completion.

Exit criteria:

- zero observed duplicate effects for supported tools;
- zero unsafe retry when outcome is `UNKNOWN`;
- no continuation after `RECOVERY_ABSTAIN`;
- repeated crash during recovery preserves the invariant;
- model errors are reported separately from recovery violations.

### RTX 5090 spike — Lordphone leads（PR #8）

PR #8 completed the midterm sizing spike on one local RTX 5090 with the pinned vLLM 0.32.0 image and Qwen3-4B BF16. It retained raw telemetry and measured:

- cold worker start with warm caches, plus one diagnostic empty-driver-JIT observation (`n=1`)；
- 2K/8K/16K prefill with APC off and repeated-prefix APC hits；
- single-stream decode and agent-prompt answer lengths；
- runner `SIGKILL` recovery with APC off/on；
- inference-process `SIGKILL` followed by a fresh vLLM container and re-prefill；
- GPU utilization, memory, power, temperature and SM-clock samples at 200 ms intervals。

The spike uses five repetitions per regular cell and is preliminary evidence, not the planned ≥20-repeat main result. Its scripted local durable-log replay is not a substitute for PR #7's real tool reconciliation. Before the main matrix, the integrated pilot must connect PR #7's committed-prefix/tool protocol to the real inference restart path and end at the shared verifier.

### Main GPU matrix — Lordphone leads, Victoria analyzes

| Factor | Values |
| --- | --- |
| Prefix length | 2K, 8K, 16K |
| Approximate completed fraction | 10%, 50%, 90% |
| Method | full restart, OutcomeServe |
| Repetitions | 20 paired seeds per cell |

Total: `3 × 3 × 2 × 20 = 360` main runs.

- Shirley owns method correctness and baseline configuration.
- Lordphone owns batch execution, lifecycle, fault evidence and resource collection.
- Victoria owns pairing, exclusions, confidence intervals and figures.
- `crash-to-correct-completion` is the primary endpoint; next-token latency is secondary.
- Both methods start from the same pre-fault fixture and end only after the same verifier passes.
- Reset cost is included and reported separately for the resettable restart comparison.
- RTX 5090 is the current primary platform. A run records GPU UUID, driver, model/context identity and cache state; results from any backup H100/L40S platform form separate cells and are never pooled.
- APC is always recorded. Runner-only diagnostics retain APC on/off because the surviving inference process can preserve prefix cache; the primary cold inference-restart comparison does not claim reusable KV or an APC hit after process loss.

### Failure-free overhead and break-even — Shirley leads

- Compare the same representative workload with durability disabled and enabled.
- Lordphone collects end-to-end time and CPU/RAM/GPU resource measurements.
- Victoria computes paired uncertainty.
- Shirley defines and interprets:

```text
H = T_OutcomeServe,no-fault - T_baseline,no-fault
S = T_restart,one-fault - T_OutcomeServe,one-fault
Expected benefit(q) = q*S - (1-q)*H
```

`q` is an assumed probability of at most one task failure under the measured fault-position distribution. It is not a claimed production crash rate.

## 7. Current implementation status and merge order

### PR #7 — Victoria: merged durable protocol, CPU correctness and paired analysis

PR #7 merged on 2026-11-04 as commit `f79b7d7`. Its implemented scope includes the durable log/tool transaction prototype, independent business/protocol verifiers, scripted CPU fault comparisons, baselines and paired analysis. The merged result is the current RQ1 correctness foundation; remaining work expands the correctness matrix and connects the same transactional reconciliation path to real vLLM recovery.

### PR #8 — Lordphone: open RTX 5090 spike and midterm RQ2 evidence

PR #8 is based on merged PR #7 at current head `e3d4490`. The independently reviewed spike evidence includes 2,630 schema-valid telemetry events in 48 files, retained raw data, figures and a midterm RQ2 section. Regular measurement cells use five repetitions; the 86-second empty-driver-JIT result is a single diagnostic observation. The earlier measurement and future-work wording blockers are resolved; the PR remains open under `CHANGES_REQUESTED` pending work-distribution coordination and final re-review.

Before merge, PR #8 must:

1. drop its small edit to the superseded work-distribution text and retain this proposal-aligned file after this PR lands；
2. preserve its RTX 5090 environment evidence and PR #7's protocol/schema state；
3. pass final review on the current rebased branch before becoming final project evidence。

### Integration order

1. Merge this proposal-aligned work-distribution document as a small docs-only PR.
2. Update PR #8 onto the resulting `main`, retain this work-distribution file while preserving its RTX 5090 environment evidence, then re-review and merge.
3. Connect PR #7's transactional reconciliation and shared verifier to PR #8's real inference restart path.
4. Integrate the midterm sections and figures, compile the complete artifact and audit every claim against retained evidence.

## 8. Remaining timeline

The dates below translate the proposal's eight-week stages into the actual remaining schedule. If the course confirms different final-report or presentation dates, only the writing milestones move; implementation scope does not expand.

| Date | Gate | Shirley | Lordphone | Victoria |
| --- | --- | --- | --- | --- |
| **Nov 3–4** | Midterm freeze | Review PR #7/#8; integrate and compile the complete midterm artifact | Retain spike evidence; align PR #8 docs and request final re-review | Merge PR #7 and hand off its correctness evidence |
| **Nov 5–9** | CPU chain integrated | Four modes and recovery controller integration | Deterministic runner/tool fault hooks | Transactional log/tool service, prefix reconstruction and verifier |
| **Nov 10–15** | Correctness gate + integrated GPU pilot | Fix protocol failures; review baselines | Connect real restart/telemetry to the integrated protocol | Complete CPU matrix and correctness report |
| **Nov 16–22** | Main GPU experiments | Validate fixtures and method semantics; cost model v1 | Run 360 paired GPU runs | Monitor data quality; aggregate partial results |
| **Nov 23–25** | Data freeze | Break-even analysis and claim audit | Rerun invalid/missing cells | Final figures, CIs and reproducibility check |
| **Nov 26–29** | Draft sections | Introduction, related work, method and RQ3 | RQ2 performance/resource results | RQ1 correctness and statistical methods |
| **Nov 30–Dec 4** | Final delivery | Integrate report, demo and presentation | Validate all reported numbers | Regenerate figures and contribution table |

## 9. Scope cuts and stop conditions

| Trigger | Required action |
| --- | --- |
| CPU invariant tests are not green by Nov 15 | Stop GPU expansion; fix correctness before collecting performance claims |
| GPU pilot cannot complete stable structured tool calls | Use constrained output/schema fixes; do not turn the project into a model-capability benchmark |
| Inference restart does not demonstrably lose request KV | Fix the fault domain; do not report runner-only restart as inference-loss recovery |
| Pilot projects more than the 90–120 GPU-hour cap | Reduce optional matrix settings while preserving paired seeds and 20 repeats for retained cells |
| Primary GPU platform becomes unavailable | Move to approved backup hardware and report each hardware configuration separately |
| Main matrix is incomplete by Nov 25 | Report the completed preregistered subset; do not add warm replica, concurrency or new baselines |
| Any method has materially more invalid/model-error runs | Report failure counts and reasons; do not compare successful runs only |

The following are future work and are automatically rejected as late scope expansion: warm replica, KV checkpoint/replication, decode checkpoint, concurrency sweep, scheduler, real cloud migration, full workflow-framework integration and exact hardware-cycle profiling.

## 10. Collaboration and contribution rules

- Every PR has one owner and at least one required reviewer from the tables above.
- Cross-person contract changes require all affected owners to approve.
- Every experiment has one named result owner even when multiple people run it.
- Every Sunday, each person updates `CONTRIBUTIONS.md` with files, functions, experiments and PR links.
- Raw experiment outputs are immutable; corrections happen in parsers or analysis code.
- Invalid runs remain in the dataset with explicit reasons.
- Results reported on different GPU models are never pooled without a hardware factor.
- Final-report contribution claims come from merged PRs, retained experiment records and `CONTRIBUTIONS.md`, not from planned ownership alone.

## 11. Minimum acceptable final result

The project is complete at minimum when it has:

1. a CPU-validated prepare/reconcile/commit-or-abstain protocol;
2. deterministic evidence across the five fault boundaries;
3. zero observed duplicate effects for supported tools and no unsafe retry for `UNKNOWN`;
4. a real vLLM inference-process restart followed by token-prefix re-prefill;
5. a paired restart-versus-OutcomeServe performance comparison ending at correct task completion;
6. CPU time, peak host RAM, peak GPU memory/utilization and GPU allocation-seconds reported with clear measurement windows;
7. failure-free overhead and an explicitly assumption-based break-even analysis;
8. regenerable tables/figures and a concrete per-member contribution record.
