"""Launch one runner attempt; the supervisor sets --epoch and any pause boundary.

    python -m outcomeserve.runner --db runtime.sqlite --workload workload.json \
        --tool-endpoint http://127.0.0.1:PORT --inference-endpoint http://127.0.0.1:8000 \
        --context-spec context.json --template examples/gpu-pilot-prompt.txt \
        --events runner-events.jsonl --clock-id CLOCK --epoch 1

Exit codes: 0 completed, 3 abstained, 4 restart baseline needs a supervisor reset.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

from outcomeserve.runner.recovering import BOUNDARIES, RecoveringRunner, RunnerConfig, build_instruction
from outcomeserve.serving.client import CompletionBackend
from outcomeserve.storage import StateStore
from outcomeserve.tools.client import ToolClient
from workloads import load_workload


def main(argv=None):
    parser = argparse.ArgumentParser(prog="outcomeserve.runner")
    for name in ("db", "workload", "tool-endpoint", "inference-endpoint", "context-spec", "template", "events", "clock-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--method", choices=["outcomeserve", "restart"], default="outcomeserve")
    parser.add_argument("--seed", type=int, default=33)
    parser.add_argument("--padding-repeats", type=int, default=0)
    parser.add_argument("--pause-at", choices=BOUNDARIES)
    parser.add_argument("--pause-step", help="step_id to pause at, e.g. step-03")
    parser.add_argument("--ready-file")
    parser.add_argument("--inference-wait", type=float, default=600.0)
    args = parser.parse_args(argv)
    if (args.pause_at is None) != (args.pause_step is None) or (args.pause_at and not args.ready_file):
        parser.error("--pause-at, --pause-step and --ready-file go together")

    spec = json.loads(Path(args.context_spec).read_text(encoding="utf-8"))
    template = Path(args.template).read_text(encoding="utf-8")
    if hashlib.sha256(template.encode()).hexdigest() != spec["chat_template_sha256"]:
        parser.error("prompt template differs from the pinned context digest")
    workload = load_workload(args.workload)
    events = Path(args.events)

    def emit(event_type, step_id, **details):
        with events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event_type": event_type, "monotonic_ns": time.perf_counter_ns(),
                                     "clock_id": args.clock_id, "epoch": args.epoch, "step_id": step_id,
                                     **details}) + "\n")

    config = RunnerConfig(workload=workload, context_spec=spec, epoch=args.epoch, method=args.method, seed=args.seed,
                          instruction=build_instruction(template, workload, args.padding_repeats),
                          pause_at=args.pause_at, pause_step=args.pause_step,
                          ready_file=Path(args.ready_file) if args.ready_file else None,
                          inference_wait_seconds=args.inference_wait)
    runner = RecoveringRunner(config, store=StateStore(args.db), tools=ToolClient(args.tool_endpoint),
                              model=CompletionBackend(args.inference_endpoint, spec["model_id"], spec), emit=emit)
    emit("RUNNER_STARTED", "none", method=args.method)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
