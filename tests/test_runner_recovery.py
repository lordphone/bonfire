import importlib.util
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from outcomeserve.recovery.interfaces import RecordType, StaleEpochError, ToolOutcome
from outcomeserve.recovery.opid import fingerprint, op_id
from outcomeserve.recovery.stubs import SqliteLog, SqliteToolService
from outcomeserve.runner.agent import Action, Boundary, FaultPlan, Runner, scripted_trace
from outcomeserve.telemetry import JsonlTelemetry, WorkerIdentity, cpu_run_context, read_events


ROOT = Path(__file__).resolve().parents[1]
TASK = "task-1"
ORDER_STEPS = [1, 3, 5, 7, 9]
FIRST_RUN_BOUNDARIES = [b for b in Boundary if b is not Boundary.DURING_RECOVERY]
LINEAGE = f"runner-{TASK}"


def load_validator():
    spec = importlib.util.spec_from_file_location("week1_validator", ROOT / "scripts" / "validate_week1_contracts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_runner(db: Path, trace: str = "orders", fault: FaultPlan | None = None) -> Runner:
    return Runner(SqliteLog(db), SqliteToolService(db), TASK, scripted_trace(trace), fault)


def launch(db: Path, *extra: str, trace: str = "orders") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "outcomeserve.runner", "--db", str(db), "--task", TASK, "--trace", trace, *extra],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )


def crash_then_recover(db: Path, boundary: Boundary, step: int, trace: str = "orders"):
    """Play the supervisor: fence each attempt with a new epoch, kill the first, replace it."""
    supervisor = Supervisor(db, trace)
    supervisor.attempt("--crash-at", boundary.value, "--crash-step", str(step), expect_kill=True)
    second = supervisor.attempt()
    return second, (json.loads(second.stdout) if second.stdout else None)


class Supervisor:
    """Test stand-in for Lordphone's supervisor: epochs, launches, CRASH_DETECTED."""

    def __init__(self, db: Path, trace: str = "orders"):
        self.db = db
        self.trace = trace
        self.log = SqliteLog(db)
        self.telemetry_path = db.with_suffix(".jsonl")
        self.telemetry = JsonlTelemetry(self.telemetry_path, cpu_run_context(TASK, f"run-{TASK}", f"{trace}:10"))
        self.attempts = 0
        self.crash_step: int | None = None

    def attempt(self, *extra: str, expect_kill: bool = False) -> subprocess.CompletedProcess:
        epoch = self.log.advance_epoch(TASK)
        self.attempts += 1
        args = ["--telemetry", str(self.telemetry_path), "--attempt", str(self.attempts)]
        if self.crash_step is not None:
            args += ["--recovery-step", str(self.crash_step)]
        proc = launch(self.db, *args, *extra, trace=self.trace)
        if expect_kill:
            assert proc.returncode == -signal.SIGKILL, proc.stderr
            fault = read_events(self.telemetry_path)[-1]
            assert fault["event_type"] == "FAULT_INJECTED"
            self.crash_step = int(fault["step_id"].removeprefix("step-"))
            worker = WorkerIdentity(LINEAGE, f"{LINEAGE}-attempt-{self.attempts}", self.attempts, epoch)
            self.telemetry.emit("CRASH_DETECTED", self.crash_step, worker)
        return proc


def order_op(step: int) -> str:
    s = scripted_trace()[step]
    return op_id(TASK, step, s.tool, s.args)


@pytest.fixture
def reference_tokens(tmp_path):
    outcome = make_runner(tmp_path / "reference.sqlite").run()
    assert outcome.status == "COMPLETED"
    return list(outcome.token_ids)


def test_failure_free_run_commits_every_step_once(tmp_path):
    db = tmp_path / "state.sqlite"
    outcome = make_runner(db).run()

    assert outcome.status == "COMPLETED"
    assert SqliteToolService(db).effect_counts() == {order_op(i): 1 for i in ORDER_STEPS}
    committed = [r.step_id for r in SqliteLog(db).records(TASK) if r.type is RecordType.CONTEXT_COMMITTED]
    assert committed == list(range(10))


@pytest.mark.parametrize("boundary", FIRST_RUN_BOUNDARIES)
@pytest.mark.parametrize("step", [2, 3])
def test_sigkill_at_boundary_recovers_to_identical_prefix_without_duplicate_effects(
    tmp_path, reference_tokens, boundary, step
):
    db = tmp_path / "state.sqlite"
    second, summary = crash_then_recover(db, boundary, step)

    assert second.returncode == 0, second.stderr
    assert summary["status"] == "COMPLETED"
    assert summary["token_ids"] == reference_tokens
    assert SqliteToolService(db).effect_counts() == {order_op(i): 1 for i in ORDER_STEPS}
    committed = [r.step_id for r in SqliteLog(db).records(TASK) if r.type is RecordType.CONTEXT_COMMITTED]
    assert committed == list(range(10))


def test_unqueryable_tool_abstains_instead_of_reexecuting_after_commit(tmp_path):
    db = tmp_path / "state.sqlite"
    second, summary = crash_then_recover(db, Boundary.POST_TOOL_COMMIT_PRE_RESULT_LOG, 3, trace="legacy")

    assert second.returncode == 3
    assert summary == {"status": "ABSTAINED", "next_step": 3, "token_ids": summary["token_ids"]}
    legacy = scripted_trace("legacy")[3]
    legacy_op = op_id(TASK, 3, legacy.tool, legacy.args)
    assert SqliteToolService(db).effect_counts()[legacy_op] == 1
    abstain = [r for r in SqliteLog(db).records(TASK) if r.type is RecordType.RECOVERY_ABSTAIN]
    assert len(abstain) == 1 and abstain[0].op_id == legacy_op

    # A later replacement stays halted and never retries the effect.
    SqliteLog(db).advance_epoch(TASK)
    third = launch(db, trace="legacy")
    assert third.returncode == 3
    assert SqliteToolService(db).effect_counts()[legacy_op] == 1


def test_unqueryable_tool_abstains_even_when_crash_preceded_dispatch(tmp_path):
    # The log cannot tell PRE_DISPATCH from a lost commit for this tool, so the
    # controller must halt; this is the expected false-abstain case.
    db = tmp_path / "state.sqlite"
    second, summary = crash_then_recover(db, Boundary.PRE_DISPATCH, 3, trace="legacy")

    assert second.returncode == 3 and summary["status"] == "ABSTAINED"
    legacy = scripted_trace("legacy")[3]
    assert op_id(TASK, 3, legacy.tool, legacy.args) not in SqliteToolService(db).effect_counts()


def test_dropped_acknowledgment_is_reconciled_by_query(tmp_path, reference_tokens):
    db = tmp_path / "state.sqlite"
    fault = FaultPlan(Boundary.POST_TOOL_COMMIT_PRE_RESULT_LOG, 3, Action.DROP_ACKNOWLEDGMENT)
    outcome = make_runner(db, fault=fault).run()

    assert outcome.status == "COMPLETED"
    assert list(outcome.token_ids) == reference_tokens
    assert SqliteToolService(db).effect_counts() == {order_op(i): 1 for i in ORDER_STEPS}


def test_dropped_acknowledgment_on_unqueryable_tool_abstains(tmp_path):
    db = tmp_path / "state.sqlite"
    fault = FaultPlan(Boundary.POST_TOOL_COMMIT_PRE_RESULT_LOG, 3, Action.DROP_ACKNOWLEDGMENT)
    outcome = make_runner(db, trace="legacy", fault=fault).run()

    assert outcome.status == "ABSTAINED" and outcome.next_step == 3
    assert [r.type for r in SqliteLog(db).records(TASK)][-1] is RecordType.RECOVERY_ABSTAIN


def test_drop_acknowledgment_is_rejected_outside_its_boundary():
    with pytest.raises(ValueError):
        FaultPlan(Boundary.PRE_DISPATCH, 3, Action.DROP_ACKNOWLEDGMENT)


def test_fenced_runner_cannot_write_log_or_apply_effects(tmp_path):
    db = tmp_path / "state.sqlite"
    log, tools = SqliteLog(db), SqliteToolService(db)
    stale = make_runner(db)
    log.advance_epoch(TASK)

    with pytest.raises(StaleEpochError):
        stale.run()
    step = scripted_trace()[1]
    response = tools.submit(
        order_op(1), fingerprint(step.tool, step.args), step.tool, step.args, stale.epoch, TASK
    )
    assert response.outcome is ToolOutcome.STALE_EPOCH
    assert tools.effect_counts() == {}


def test_reusing_op_id_with_different_arguments_is_rejected(tmp_path):
    tools = SqliteToolService(tmp_path / "state.sqlite")
    step = scripted_trace()[1]
    first = tools.submit(order_op(1), fingerprint(step.tool, step.args), step.tool, step.args, 0, TASK)
    other_args = {**step.args, "qty": 99}
    second = tools.submit(order_op(1), fingerprint(step.tool, other_args), step.tool, other_args, 0, TASK)

    assert first.outcome is ToolOutcome.COMMITTED
    assert second.outcome is ToolOutcome.FINGERPRINT_MISMATCH
    assert tools.effect_counts() == {order_op(1): 1}


@pytest.mark.parametrize("boundary", FIRST_RUN_BOUNDARIES)
def test_crash_and_recovery_telemetry_passes_the_week1_trace_validator(tmp_path, boundary):
    db = tmp_path / "state.sqlite"
    second, _ = crash_then_recover(db, boundary, 3)
    assert second.returncode == 0, second.stderr

    events = read_events(db.with_suffix(".jsonl"))
    load_validator().validate_trace(events)
    step3 = [e["event_type"] for e in events if e["step_id"] == "step-03"]
    assert step3[:2] == ["FAULT_INJECTED", "CRASH_DETECTED"]
    assert step3[-1] == "NEXT_TOKEN_EMITTED"
    reconciled = boundary in (Boundary.PRE_DISPATCH, Boundary.POST_TOOL_COMMIT_PRE_RESULT_LOG)
    assert ("RECONCILIATION_FINISHED" in step3) is reconciled


@pytest.mark.parametrize("first_crash", [Boundary.PRE_DISPATCH, Boundary.POST_TOOL_COMMIT_PRE_RESULT_LOG])
def test_crash_during_recovery_is_recovered_by_the_next_attempt(tmp_path, reference_tokens, first_crash):
    db = tmp_path / "state.sqlite"
    supervisor = Supervisor(db)
    supervisor.attempt("--crash-at", first_crash.value, "--crash-step", "3", expect_kill=True)
    supervisor.attempt("--crash-at", Boundary.DURING_RECOVERY.value, "--crash-step", "3", expect_kill=True)
    third = supervisor.attempt()

    assert third.returncode == 0, third.stderr
    assert json.loads(third.stdout)["token_ids"] == reference_tokens
    assert SqliteToolService(db).effect_counts() == {order_op(i): 1 for i in ORDER_STEPS}

    events = read_events(db.with_suffix(".jsonl"))
    load_validator().validate_trace(events)
    finished = [e for e in events if e["event_type"] == "RECONCILIATION_FINISHED"]
    # The killed recovery re-executed (or found) the effect; the next one finds it committed.
    assert [e["details"]["outcome"] for e in finished] == ["COMMITTED"]
    assert [e["worker"]["attempt"] for e in finished] == [3]
    started = [e["worker"]["attempt"] for e in events if e["event_type"] == "RECONCILIATION_STARTED"]
    assert started == [2, 3]


def test_abstain_telemetry_passes_the_week1_trace_validator(tmp_path):
    db = tmp_path / "state.sqlite"
    second, _ = crash_then_recover(db, Boundary.POST_TOOL_COMMIT_PRE_RESULT_LOG, 3, trace="legacy")
    assert second.returncode == 3

    events = read_events(db.with_suffix(".jsonl"))
    load_validator().validate_trace(events)
    tail = [(e["event_type"], e["details"].get("outcome")) for e in events[-2:]]
    assert tail == [("RECONCILIATION_FINISHED", "UNKNOWN"), ("RECOVERY_ABSTAIN", None)]


def test_readme_demo_writes_schema_valid_events_with_distinct_attempts(tmp_path):
    from jsonschema import Draft202012Validator

    db, events_path = tmp_path / "state.sqlite", tmp_path / "events.jsonl"
    common = ["--telemetry", str(events_path), "--fence"]
    first = launch(db, *common, "--crash-at", Boundary.POST_TOOL_COMMIT_PRE_RESULT_LOG.value, "--crash-step", "3")
    second = launch(db, *common, "--recovery-step", "3")

    assert first.returncode == -signal.SIGKILL
    assert second.returncode == 0 and json.loads(second.stdout)["status"] == "COMPLETED"
    schema = json.loads((ROOT / "schemas" / "telemetry-event.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    events = read_events(events_path)
    for event in events:
        validator.validate(event)
    assert {(e["worker"]["attempt"], e["worker"]["owner_epoch"]) for e in events} == {(1, 1), (2, 2)}


def test_telemetry_without_an_epoch_is_refused(tmp_path):
    proc = launch(tmp_path / "state.sqlite", "--telemetry", str(tmp_path / "events.jsonl"))
    assert proc.returncode == 2
    assert "--fence" in proc.stderr
