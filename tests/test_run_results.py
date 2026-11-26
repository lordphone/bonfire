"""C6 GPU run results: assembly from retained evidence and main-matrix analysis."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from analysis.pipeline import gpu_matrix_analysis, outcome_counts, paired_comparison
from analysis.runresult import assemble, assemble_directory, pair_key, read_events
from outcomeserve.contracts import ContractError, validate_schema

ROOT = Path(__file__).resolve().parents[1]
GOOD_RESOURCES = {"measurement_valid": True, "process_cpu_seconds": 3.0, "gpu_allocation_seconds": 40.0,
                  "peak_summed_rss_bytes": 10000000, "peak_gpu_memory_bytes": 8000000000}


def identity(method="outcomeserve", seed=33, prefix=8192, progress=0.5, no_fault=False, tier="1", order=None, **extra):
    order = order or (1 if (method == "restart") == (seed % 2 == 0) else 2)
    value = {"run_id": f"s{seed}-L{prefix}-p{progress}-{method}-{no_fault}", "task_id": "gpu-task", "seed": seed, "method": method,
             "tier": "1H" if no_fault else tier, "no_fault": no_fault, "fault_domain": "inference",
             "boundary": "DURING_GENERATION", "prefix_target_tokens": prefix,
             "requested_progress": None if no_fault else progress, "actual_progress": 1.0 if no_fault else progress,
             "apc": False, "method_order": order, "configuration_id": f"cfg-L{prefix}-p{progress}-{no_fault}",
             "context_spec_id": "ctx-qwen3-4b", "business_fixture_sha256": f"fixture-s{seed}",
             "hardware_id": "GPU-5090-uuid", "gpu_name": "NVIDIA GeForce RTX 5090", "driver_version": "580.95",
             "runtime_versions": {"vllm": "0.32.0"}, "environment_lock_sha256": "lock"}
    value.update(extra)
    return value


def events(ident, *, task=0.0, fault=None, token=None, reset=None, verified=None, clock="clock-a", failed=False):
    rows, stamps = [], [("TASK_STARTED", task)]
    if fault is not None:
        stamps.append(("FAULT_APPLIED", fault))
    if reset is not None:
        stamps += [("TOOL_RESET_STARTED", reset[0]), ("TOOL_RESET_FINISHED", reset[1])]
    if token is not None:
        stamps.append(("FIRST_NEW_TOKEN", token))
    if verified is not None:
        stamps.append(("VERIFICATION_FAILED" if failed else "VERIFIED_COMPLETE", verified))
    for sequence, (name, seconds) in enumerate(stamps, 1):
        rows.append({"schema_version": "1.1.0", "run_id": ident["run_id"], "configuration_id": ident["configuration_id"],
                     "clock_id": clock, "sequence": sequence, "event_type": name,
                     "monotonic_ns": int(seconds * 1e9), "details": {}})
    return rows


def verification(duplicates=0, success=True):
    return {"valid": True, "task_id": "gpu-task", "task_success": success,
            "metrics": {"duplicate_effects": duplicates, "tool_state_mismatches": 0,
                                                 "inventory_mismatches": 0, "audit_complete": True}}


def protocol_report(ident, *, exit_code=0):
    if ident["method"] != "outcomeserve":
        return None
    report = {"valid": True, "task_id": ident["task_id"], "context_spec_id": ident["context_spec_id"],
            "token_source": "model", "checkpoint": "halted" if exit_code == 3 else "completed",
            "consistency_pass": True, "task_success": exit_code == 0, "issues": []}
    return report


def completed(method, seed, *, prefix=8192, progress=0.5, no_fault=False, total=60.0, tier="1"):
    ident = identity(method, seed, prefix, progress, no_fault, tier=tier)
    if no_fault:
        evs = events(ident, verified=total)
    else:
        evs = events(ident, fault=total / 2, token=total / 2 + 5, verified=total)
    return assemble(ident, evs, verification(), GOOD_RESOURCES, runner_exit_code=0, protocol=protocol_report(ident))


def test_completed_run_has_primary_and_secondary_intervals():
    ident = identity("restart", 34)
    result = assemble(ident, events(ident, task=1, fault=31, reset=(32, 33), token=50, verified=81), verification(),
                      GOOD_RESOURCES, runner_exit_code=0)
    assert result["outcome"] == "COMPLETED" and result["passed"] and result["valid"]
    assert result["task_to_verified_seconds"] == 80 and result["fault_to_verified_seconds"] == 50
    assert result["fault_to_first_new_token_seconds"] == 19 and result["reset_seconds"] == 1
    assert result["pair_key"] == "s34:gpu:L8192:p0.5:inference:DURING_GENERATION:apc-false:False"
    assert result["gpu_matrix"]["tier"] == "1" and "tier" not in result


def test_first_new_token_before_the_fault_is_not_the_recovery_endpoint():
    ident = identity()
    rows = events(ident, fault=30, token=40, verified=60)
    rows.insert(1, {**rows[0], "event_type": "FIRST_NEW_TOKEN", "monotonic_ns": int(5e9), "sequence": 99})
    result = assemble(ident, rows, verification(), GOOD_RESOURCES, runner_exit_code=0, protocol=protocol_report(ident))
    assert result["fault_to_first_new_token_seconds"] == 10


@pytest.mark.parametrize("exit_code, verify, outcome", [
    (5, None, "MODEL_ERROR"),
    (3, verification(success=False), "ABSTAINED"),
    (0, verification(duplicates=1, success=False), "RECOVERY_VIOLATION"),
    (1, None, "INFRASTRUCTURE_FAILURE"),
])
def test_failures_after_a_valid_injection_stay_valid_and_are_classified(exit_code, verify, outcome):
    ident = identity()
    result = assemble(ident, events(ident, fault=30, verified=60, failed=True), verify, GOOD_RESOURCES,
                      runner_exit_code=exit_code, failure_reason="boom", protocol=protocol_report(ident, exit_code=exit_code))
    assert result["outcome"] == outcome and result["valid"] and not result["passed"]
    assert result["fault_applied"] and result["failure_reason"] == "boom"


@pytest.mark.parametrize("mutate, reason", [
    (lambda rows: [r for r in rows if r["event_type"] != "FAULT_APPLIED"], "0 FAULT_APPLIED"),
    (lambda rows: rows[:1] + [{**rows[1], "clock_id": "other"}] + rows[2:], "more than one clock_id"),
    (lambda rows: [{**r, "run_id": "someone-else"} for r in rows], "another run_id"),
])
def test_unestablished_fault_or_mixed_clocks_are_invalid_not_failed(mutate, reason):
    ident = identity()
    result = assemble(ident, mutate(events(ident, fault=30, verified=60)), verification(), GOOD_RESOURCES,
                      runner_exit_code=0)
    assert result["outcome"] == "INVALID" and not result["valid"] and not result["passed"]
    assert any(reason in item for item in result["invalid_reasons"])
    assert result["fault_to_verified_seconds"] is None


def test_completed_run_without_resource_window_is_not_a_timing_sample():
    ident = identity()
    result = assemble(ident, events(ident, fault=30, verified=60), verification(), None, runner_exit_code=0, protocol=protocol_report(ident))
    assert result["outcome"] == "INFRASTRUCTURE_FAILURE" and not result["passed"]


def test_identity_must_be_complete_and_closed():
    with pytest.raises(ValueError, match="missing tier"):
        assemble({k: v for k, v in identity().items() if k != "tier"}, [], None, None, runner_exit_code=0)
    with pytest.raises(ValueError, match="unknown fields"):
        assemble(identity(colour="red"), [], None, None, runner_exit_code=0)


def test_schema_rejects_gpu_results_without_matrix_identity_or_with_contradictory_outcome():
    result = completed("outcomeserve", 33)
    validate_schema("run-result", result)
    with pytest.raises(ContractError):
        validate_schema("run-result", {k: v for k, v in result.items() if k != "gpu_matrix"})
    with pytest.raises(ContractError):
        validate_schema("run-result", {**result, "outcome": "MODEL_ERROR"})
    with pytest.raises(ContractError):
        validate_schema("run-result", {**result, "requested_progress": 0.3})


def test_cli_builds_from_a_run_directory_and_never_overwrites(tmp_path):
    ident = identity()
    run = tmp_path / ident["run_id"]
    run.mkdir()
    (run / "identity.json").write_text(json.dumps(ident))
    rows = events(ident, fault=30, token=35, verified=60)
    (run / "measurements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:2]))
    (run / "runner-measurements.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[2:]))
    (run / "verification.json").write_text(json.dumps({"business": verification(), "protocol": protocol_report(ident)}))
    (run / "resources.json").write_text(json.dumps(GOOD_RESOURCES))
    (run / "runner-exit.json").write_text(json.dumps({"exit_code": 0}))
    command = [sys.executable, str(ROOT / "scripts/build_run_result.py"), str(run),
               "--events", "measurements.jsonl", "runner-measurements.jsonl"]
    first = subprocess.run(command, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert json.loads((run / "result.json").read_text())["fault_to_first_new_token_seconds"] == 5
    again = subprocess.run(command, capture_output=True, text=True)
    assert again.returncode != 0 and "never rewritten" in again.stderr
    assert assemble_directory(run, event_files=["measurements.jsonl", "runner-measurements.jsonl"])["passed"]


def test_bad_event_line_is_rejected(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"event_type": "TASK_STARTED"}\n')
    with pytest.raises(ContractError):
        read_events([path])


def test_pairs_need_both_order_positions_and_the_same_tier():
    a, b = completed("restart", 33, total=70), completed("outcomeserve", 33, total=60)
    assert paired_comparison([a, b])["complete_pairs"] == 1
    same_order = {**b, "gpu_matrix": {**b["gpu_matrix"], "method_order": a["gpu_matrix"]["method_order"]}}
    assert "positions 1 and 2" in paired_comparison([a, same_order])["incomplete_pairs"][0]["reason"]
    other_tier = {**b, "gpu_matrix": {**b["gpu_matrix"], "tier": "2"}}
    assert "gpu_matrix.tier" in paired_comparison([a, other_tier])["incomplete_pairs"][0]["reason"]


def matrix(seeds=range(33, 43), prefixes=((2048, 48), (16384, 71))):
    rows = []
    for seed in seeds:
        jitter = (seed % 3) * 0.5
        for prefix, base in prefixes:
            rows += [completed("restart", seed, prefix=prefix, total=base + 8 + jitter),
                     completed("outcomeserve", seed, prefix=prefix, total=base + jitter),
                     completed("restart", seed, prefix=prefix, no_fault=True, total=base / 2),
                     completed("outcomeserve", seed, prefix=prefix, no_fault=True, total=base / 2 + 1 + jitter / 4)]
    return rows


def test_gpu_matrix_reports_delta_t_overhead_and_break_even_per_prefix():
    result = gpu_matrix_analysis(matrix())
    assert result["platform"] == {"gpu_name": "NVIDIA GeForce RTX 5090", "driver_version": "580.95"}
    cells = {(c["prefix_target_tokens"], c["requested_progress"]): c for c in result["cells"]}
    assert set(cells) == {(2048, 0.5), (16384, 0.5)}
    cell = cells[(2048, 0.5)]
    assert cell["delta_T_fault_to_verified"]["estimate"]["mean_restart_minus_outcomeserve"] == pytest.approx(4)
    assert cell["whole_task_saving_S"]["estimate"]["mean_restart_minus_outcomeserve"] == pytest.approx(8)
    h = result["overhead_H"][2048]["H_seconds"]
    assert h == pytest.approx(1.125, abs=0.2)
    assert cell["break_even"]["break_even_q"] == pytest.approx(h / (h + 8))
    assert cell["outcomes"]["outcomeserve"]["COMPLETED"] == 10
    # Partial matrices cannot establish the smallest saving across all Tier 1 cells.
    assert result["tier_1H_precision"] is None


def test_a_model_error_suppresses_the_cell_estimate_and_is_counted():
    rows = matrix(range(33, 36))
    victim = next(r for r in rows if r["method"] == "outcomeserve" and not r["no_fault"]
                  and r["gpu_matrix"]["prefix_target_tokens"] == 2048)
    ident = identity("outcomeserve", victim["seed"], 2048, 0.5)
    rows[rows.index(victim)] = assemble(ident, events(ident, fault=20, verified=40, failed=True), None,
                                        GOOD_RESOURCES, runner_exit_code=5, failure_reason="tool call mismatch")
    cell = next(c for c in gpu_matrix_analysis(rows)["cells"] if c["prefix_target_tokens"] == 2048)
    assert cell["delta_T_fault_to_verified"]["estimate"] is None
    assert cell["outcomes"]["outcomeserve"]["MODEL_ERROR"] == 1
    assert cell["outcomes"]["outcomeserve"]["reasons"] == {"tool call mismatch": 1}


def test_gpu_platforms_are_never_pooled():
    rows = matrix(range(33, 35))
    rows[0] = {**rows[0], "gpu_matrix": {**rows[0]["gpu_matrix"], "gpu_name": "NVIDIA H100"}}
    with pytest.raises(ValueError, match="never pooled"):
        gpu_matrix_analysis(rows)


def test_outcome_counts_classify_older_results_without_guessing():
    rows = [{"method": "restart", "valid": False, "invalid_reasons": ["boundary missed"]},
            {"method": "restart", "valid": True, "passed": False, "model_or_backend_error": "x"},
            {"method": "restart", "valid": True, "passed": True}]
    counts = outcome_counts(rows)["restart"]
    assert (counts["INVALID"], counts["MODEL_ERROR"], counts["COMPLETED"]) == (1, 1, 1)


def test_analyze_runs_writes_gpu_heatmap(tmp_path):
    root = tmp_path / "results"
    for index, row in enumerate(matrix(range(33, 36))):
        folder = root / f"run-{index}"
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(row))
    output = tmp_path / "analysis"
    done = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_runs.py"), "--results-root", str(root),
                           "--output", str(output)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    with (output / "gpu-heatmap.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert {(r["prefix_target_tokens"], r["requested_progress"]) for r in rows} == {("2048", "0.5"), ("16384", "0.5")}
    assert all(r["complete_pairs"] == "3" and r["non_completed_runs"] == "0" for r in rows)
    assert (output / "outcomes.json").exists() and (output / "gpu-matrix.json").exists()


def test_pair_key_is_stable():
    assert pair_key(identity(no_fault=True)) == "s33:gpu:L8192:pnone:none:none:apc-false:True"


@pytest.mark.parametrize("no_fault", [False, True])
def test_invalid_member_is_retained_and_blocks_all_relevant_estimates(no_fault):
    rows = matrix(range(33, 36))
    victim = next(r for r in rows if r["method"] == "outcomeserve" and r["no_fault"] == no_fault
                  and r["gpu_matrix"]["prefix_target_tokens"] == 2048)
    ident = identity("outcomeserve", victim["seed"], 2048, no_fault=no_fault)
    bad_events = events(ident, verified=60) if not no_fault else events(ident, fault=30, verified=60)
    rows[rows.index(victim)] = assemble(ident, bad_events, verification(), GOOD_RESOURCES, runner_exit_code=0, protocol=protocol_report(ident))
    result = gpu_matrix_analysis(rows)
    cell = next(c for c in result["cells"] if c["prefix_target_tokens"] == 2048)
    assert result["outcomes_all_runs"]["outcomeserve"]["INVALID"] == 1
    assert cell["break_even"] is None
    if no_fault:
        assert result["overhead_H"][2048]["H_seconds"] is None
        assert cell["delta_T_fault_to_verified"]["estimate"] is not None
    else:
        for comparison in (cell["delta_T_fault_to_verified"], cell["whole_task_saving_S"],
                           cell["secondary_first_new_token"], *cell["resources"].values()):
            assert comparison["estimate"] is None
            assert comparison["total_pairs"] == 3 and comparison["complete_pairs"] == 2
            assert comparison["invalid_runs"] == 1 and comparison["valid_failed_runs"] == 0
            assert comparison["incomplete_pairs"]


def test_missing_member_and_valid_failure_are_not_silently_filtered():
    rows = [completed(m, seed) for seed in (33, 34) for m in ("restart", "outcomeserve")]
    assert paired_comparison(rows[:-1])["estimate"] is None
    ident = identity("outcomeserve", 34)
    rows[-1] = assemble(ident, events(ident, fault=30), None, GOOD_RESOURCES, runner_exit_code=1)
    comparison = paired_comparison(rows)
    assert comparison["estimate"] is None and comparison["valid_failed_runs"] == 1
    assert comparison["invalid_runs"] == 0 and comparison["total_pairs"] == 2


@pytest.mark.parametrize("task,fault,verified,outcome", [
    (10, 5, 60, "INVALID"),
    (0, 70, 60, "INFRASTRUCTURE_FAILURE"),
    (10, 30, 5, "INFRASTRUCTURE_FAILURE"),
])
def test_impossible_timing_never_becomes_a_completed_sample(task, fault, verified, outcome):
    ident = identity()
    result = assemble(ident, events(ident, task=task, fault=fault, verified=verified), verification(),
                      GOOD_RESOURCES, runner_exit_code=0, protocol=protocol_report(ident))
    assert result["outcome"] == outcome and not result["passed"]
    assert result["failure_reason"]


def test_duplicate_completion_and_token_after_completion_do_not_define_endpoints():
    ident = identity()
    rows = events(ident, fault=30, token=70, verified=60)
    result = assemble(ident, rows, verification(), GOOD_RESOURCES, runner_exit_code=0, protocol=protocol_report(ident))
    assert result["passed"] and result["fault_to_first_new_token_seconds"] is None
    rows.append({**rows[-1], "sequence": 99})
    result = assemble(ident, rows, verification(), GOOD_RESOURCES, runner_exit_code=0, protocol=protocol_report(ident))
    assert result["outcome"] == "INFRASTRUCTURE_FAILURE" and result["valid"]


def test_incomplete_audit_is_never_completed_even_with_zero_duplicate_counter():
    ident = identity()
    report = verification()
    report["metrics"]["audit_complete"] = False
    result = assemble(ident, events(ident, fault=30, verified=60), report, GOOD_RESOURCES, runner_exit_code=0)
    assert result["outcome"] == "RECOVERY_VIOLATION"


def test_abstention_is_not_inferred_to_be_unsafe_from_incomplete_final_inventory():
    ident = identity()
    report = verification(success=False)
    report["metrics"]["inventory_mismatches"] = 2
    result = assemble(ident, events(ident, fault=30), report, GOOD_RESOURCES, runner_exit_code=3, protocol=protocol_report(ident, exit_code=3))
    assert result["outcome"] == "ABSTAINED" and result["valid"]
    assert "not established" in result["failure_reason"]


@pytest.mark.parametrize("field,value", [
    ("no_fault", "false"), ("actual_progress", 1.1), ("context_spec_id", ""),
    ("runtime_versions", {}), ("valid", False),
])
def test_new_schema_rejects_ambiguous_identity_and_validity(field, value):
    with pytest.raises(ContractError):
        validate_schema("run-result", {**completed("outcomeserve", 33), field: value})


@pytest.mark.parametrize("field,value", [
    ("apc", True), ("context_spec_id", "other-context"), ("hardware_id", "other-GPU"),
    ("runtime_versions", {"vllm": "other"}), ("environment_lock_sha256", "other-lock"),
    ("fault_domain", "runner"), ("boundary", "PRE_DISPATCH"),
])
def test_h_and_s_cannot_be_combined_across_configurations(field, value):
    rows = matrix(range(33, 35))
    # Alter a complete no-fault pair: pair-local checks alone would miss this.
    for row in rows:
        if row["no_fault"]:
            row[field] = value
    with pytest.raises(ValueError, match=f"different {field}"):
        gpu_matrix_analysis(rows)


def test_ram_and_gpu_memory_differences_use_paired_samples_without_inventing_values():
    rows = matrix(range(33, 35))
    for row in rows:
        row["resources"] = dict(row["resources"])
        if row["method"] == "restart":
            row["resources"]["peak_summed_rss_bytes"] += 100
            row["resources"]["peak_gpu_memory_bytes"] += 200
    resources = gpu_matrix_analysis(rows)["cells"][0]["resources"]
    assert resources["resources.peak_summed_rss_bytes"]["estimate"]["mean_restart_minus_outcomeserve"] == 100
    assert resources["resources.peak_gpu_memory_bytes"]["estimate"]["mean_restart_minus_outcomeserve"] == 200
    del rows[0]["resources"]["peak_gpu_memory_bytes"]
    resources = gpu_matrix_analysis(rows)["cells"][0]["resources"]
    assert resources["resources.peak_gpu_memory_bytes"]["estimate"] is None
    assert resources["resources.peak_summed_rss_bytes"]["estimate"] is not None


def test_precision_check_requires_all_prefixes_and_positive_savings():
    rows = matrix(range(33, 36), prefixes=((2048, 48), (8192, 56), (16384, 71)))
    precision = gpu_matrix_analysis(rows)["tier_1H_precision"]
    assert precision["extend_tier_1H"] is False and precision["threshold_seconds"] == 2
    for row in rows:
        if not row["no_fault"] and row["gpu_matrix"]["prefix_target_tokens"] == 2048:
            row["task_to_verified_seconds"] = 100 if row["method"] == "outcomeserve" else 60
    assert gpu_matrix_analysis(rows)["tier_1H_precision"] is None


def test_analyzer_retains_invalid_cell_and_rejects_mixed_input_before_writing(tmp_path):
    rows = [completed(m, seed) for seed in (33, 34) for m in ("restart", "outcomeserve")]
    ident = identity("outcomeserve", 34)
    rows[-1] = assemble(ident, events(ident, verified=60), verification(), GOOD_RESOURCES, runner_exit_code=0, protocol=protocol_report(ident))
    paths = []
    for i, row in enumerate(rows):
        path = tmp_path / f"result-{i}.json"
        path.write_text(json.dumps(row))
        paths.append(str(path))
    output = tmp_path / "analysis"
    command = [sys.executable, str(ROOT / "scripts/analyze_runs.py"), "--results", *paths, "--output", str(output)]
    done = subprocess.run(command, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    data = json.loads((output / "gpu-matrix.json").read_text())
    assert data["cells"][0]["delta_T_fault_to_verified"]["estimate"] is None
    assert data["outcomes_all_runs"]["outcomeserve"]["INVALID"] == 1
    with (output / "gpu-heatmap.csv").open() as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 1 and csv_rows[0]["delta_T_mean_seconds"] == ""
    assert csv_rows[0]["non_completed_runs"] == "1"
    rows[0]["apc"] = True
    Path(paths[0]).write_text(json.dumps(rows[0]))
    rejected_output = tmp_path / "rejected"
    done = subprocess.run(command[:-1] + [str(rejected_output)], capture_output=True, text=True)
    assert done.returncode != 0 and "different apc" in done.stderr
    assert not rejected_output.exists()


def test_abstention_with_a_wrong_committed_result_is_a_recovery_violation():
    ident = identity()
    report = verification(success=False)
    report["metrics"].update(inventory_mismatches=2, tool_state_mismatches=1)
    result = assemble(ident, events(ident, fault=30), report, GOOD_RESOURCES, runner_exit_code=3, protocol=protocol_report(ident, exit_code=3))
    assert result["outcome"] == "RECOVERY_VIOLATION"


def test_pilot_runs_on_matrix_seeds_are_counted_but_never_matrix_samples(tmp_path):
    rows = matrix(range(33, 35))
    pilot = [completed(m, 33, prefix=2048, tier="pilot") for m in ("restart", "outcomeserve")]
    assert pilot[0]["pair_key"].startswith("s33:gpu-pilot:")
    with_pilot, without = gpu_matrix_analysis(rows + pilot), gpu_matrix_analysis(rows)
    assert with_pilot["cells"] == without["cells"] and with_pilot["overhead_H"] == without["overhead_H"]
    assert with_pilot["pilot_runs_excluded"]["outcomeserve"]["COMPLETED"] == 1
    paths = []
    for index, row in enumerate(rows + pilot):
        path = tmp_path / f"r{index}.json"
        path.write_text(json.dumps(row))
        paths.append(str(path))
    done = subprocess.run([sys.executable, str(ROOT / "scripts/analyze_runs.py"), "--results", *paths,
                           "--output", str(tmp_path / "out")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
