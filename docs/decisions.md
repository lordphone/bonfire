# Decision log

Record decisions that change scope, contracts, or experiment design: what was decided, why, and who decided. Add the newest entry at the top.

| Date | Decision | Why | Decided by |
| --- | --- | --- | --- |
| 10/27 | Work is divided by research question (plan A in `docs/work-distribution.md`) | Each person needs a results section they can write on their own, and coupled modules should share an owner | Shirley |
| 10/27 | Every reported GPU number comes from PSC Bridges-2 H100, with L40S as the fallback (reported separately). V100 is not used | V100 has no BF16, and the pinned vLLM 0.32 reportedly does not support sm_70 | Shirley |
