# Recovery protocol

Tools execute sequentially, with one runner owning each task. Logs and tool state survive runner or inference failure. The prototype uses one SQLite file for ownership, logs, and tool tables, but tool and log commits are separate transactions. SQLite uses WAL and FULL synchronous mode. Cross-database fencing and storage failure are outside the validated scope.

## Identity and context

task_id, step_id, and op_id remain stable across recovery; epoch is not part of operation identity. Each task binds a fixed model/tokenizer revision, template SHA256, thinking setting, and backend through context_spec_id. Save exact input/output token IDs and the complete canonical prefix; decoding and re-tokenizing text does not preserve this evidence.

## C1: durable log

Implementation: outcomeserve/log. Format: schemas/durable-log.schema.json.

| Record | Content and purpose |
| --- | --- |
| STEP_PREPARED | Tool name, arguments, fingerprint, generation_prompt_token_ids, and generated_token_ids; durable before dispatch |
| TOOL_RESULT_COMMITTED | Tool result, fingerprint, and commit receipt |
| CONTEXT_COMMITTED | Complete prefix token IDs and the associated result record ID |
| RECOVERY_ABSTAIN | UNKNOWN and its reason; durably stop the task |

The store assigns sequence numbers starting at 1 without gaps; record_id is an idempotency key. All writes validate owner_epoch.

`append(task_id, record_id, expected_sequence, owner_epoch, record_body)` only accepts prepare or abstain records. Retrying identical content under the same ID is idempotent. Conflicts and invalid transitions do not modify the log.

`commit_step(prepared, tool_response, token_ids, owner_epoch=..., expected_sequence=..., commit_id=...)` requires durable prepare and COMMITTED C2 evidence with matching identity, epoch, and fingerprint. One transaction writes the result, context, and prefix pointer. Direct result/context append is prohibited. Success returns COMMITTED; an identical retry returns ALREADY_COMMITTED. Rejections include STALE_EPOCH, SEQUENCE_CONFLICT, RECORD_ID_CONFLICT, and INVALID_TRANSITION. JSON comparison distinguishes true from 1.

read returns records; committed_prefix reads the pointer and context in one snapshot. A crash may leave prepare or a complete commit; a result-only tail is invalid. Discard incomplete generation that has no prepare record. Decode checkpoints are not maintained.

## C2: transactional tool service

Implementation: outcomeserve/tools. Format: schemas/tool-service.schema.json. An independent HTTP process supports read_inventory and decrement_inventory. One transaction records the operation ID, business effect, result, receipt, and effect audit. Duplicate operations return the saved result. Business rejections, such as insufficient inventory, are also saved and deduplicated.

submit carries task/step/op IDs, owner_epoch, tool name, arguments, and fingerprint; query carries the same identity and fingerprint. The fingerprint is SHA256 over canonical JSON of the tool name and arguments. Arguments accept JSON safe integers, but not floats or nonfinite numbers. Changed arguments for an existing operation return FINGERPRINT_MISMATCH; step/op identity collisions are protocol errors.

| Query/response status | Runner action |
| --- | --- |
| COMMITTED | Reuse the saved result/receipt and atomically commit context |
| NOT_COMMITTED | An authoritative query proves absence; safely submit the original op ID |
| UNKNOWN | Non-execution cannot be established; durably abstain |
| STALE_EPOCH | Stop writes from the old runner |
| FINGERPRINT_MISMATCH | Reject changes to operation meaning |

Capabilities declare deduplication, linearizable queries, and atomic effect/result storage. These guarantees do not extend to arbitrary external APIs. A heartbeat cannot prove the outcome of the last tool operation. Log commits neither query tool tables nor call tools inside the transaction; independent verifiers check the actual evidence.

## Faults and replacement

C5 request format: schemas/fault-injection.schema.json. Requests include domain, boundary, action, task/step/config/worker identity, seed, and one_shot. An injection_id cannot be reused with changed content.

| Boundary | Precondition |
| --- | --- |
| DURING_GENERATION | At least one token exists; generation is unfinished and has no prepare record |
| PRE_DISPATCH | Prepare is durable; the tool has not committed |
| POST_TOOL_COMMIT_PRE_RESULT_LOG | Tool committed; result/context is not yet logged |
| POST_CONTEXT_COMMIT | Context committed atomically; the next step has not started |
| DURING_RECOVERY | Replacement entered reconciliation; the final outcome is not yet logged |

Resolve fault_step from requested_progress before execution. Tool windows require a write; if the position is a read, choose the next write and record actual_progress. Once fixed, the request does not move dynamically. The supervisor checks barrier/database preconditions before injection; a mismatch makes the run invalid.

The runner domain terminates the entire owned process tree, including children of the Windows venv launcher. Before replacement, compare-and-swap advances epoch; C1/C2 reject old writes. Replacement reconciles pending operations before generation. DURING_RECOVERY includes preparatory and recovery faults and is a separate correctness case.

The inference domain terminates inference only during real unfinished generation. The runner and tool service remain alive; epoch is unchanged. A fresh inference process has no reusable request KV and must replay the prefix. Startup rejects an occupied endpoint and checks the owned process and /v1/models model ID. Snapshot/configuration checks establish the model revision separately.

The tool_response domain suppresses an acknowledgment only after commit. It does not undo the effect or replace the runner. A timeout does not imply KV loss. Cleanup targets only processes owned by the current experiment.

## Verification and baselines

outcomeserve/verify/business.py checks inventory, tool results, logical steps, and effect audit for all four methods. verify/__init__.py independently checks OutcomeServe C1 ordering, identity, prefix, and pointer. These checks are complementary and do not substitute runtime state-machine validation for an independent oracle.

Baselines separate physical attempt IDs from logical steps, avoiding accidental deduplication by OutcomeServe. Correctness retains tool state. Performance restart resets only a single-task isolated simulator and includes reset time; real external effects are not assumed reversible. Timing and statistics follow the [measurement contract](measurement.md).