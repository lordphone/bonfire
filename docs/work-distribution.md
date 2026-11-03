# Project responsibilities

Ownership follows Team and Milestones in the [proposal](../proposal/proposal.tex).

| Owner | Responsibilities | Collaboration |
| --- | --- | --- |
| Shirley Li | Runtime, tool contract, vLLM integration, recovery, and report coordination | Align C1/C2, prefixes, and UNKNOWN handling with Victoria; validate inference lifecycle and model configuration with Lordphone |
| Lordphone Wen | Process lifecycle, fault injection, resource monitoring, and benchmarks | Integrate inference serving with Shirley; align injection preconditions, measurement events, and paired data with Victoria |
| Victoria Zhu | Durable logging, prefix reconstruction, verifiers, and statistical analysis | Provide atomic commit and context identity to the runtime; cross-check business/protocol evidence and resource statistics |
| Everyone | Integration, reproduction, writing, demos, and presentation | Review shared interfaces and distinguish implementation, tests, and measurements |

Victoria has provided the log/tool transaction prototype, independent oracles, CPU fault comparisons, and paired analysis, primarily in outcomeserve/log, tools, verify, experiments/cpu, and analysis/pipeline. Process monitoring and the GPU pilot provide integration support. Long-term ownership remains as listed above; code availability does not establish team review or real GPU integration.

Victoria's next steps are to validate exact prefixes and atomic commits with the runtime, validate real faults, sampling, and model-error classification with the benchmark owner, then analyze formal GPU data and write results. The course milestones and compute budget are defined in the proposal; matrix configuration is in infra/experiment-plan.json.
GPU measurements run on Lordphone's local RTX 5090 (single GPU, process-restart recovery); PSC Bridges-2 is a fallback and its results are reported separately. The environment is in [docs/infrastructure/environment.md](infrastructure/environment.md) and the Week 1 spike in [experiments/spike/report.md](../experiments/spike/report.md). Warm-replica failover and decode checkpointing are not implemented or measured and remain future work.
