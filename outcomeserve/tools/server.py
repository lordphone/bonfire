"""Surviving loopback tool-service process for local correctness experiments."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Event

from outcomeserve.storage import StateStore
from outcomeserve.tools import ToolService
from outcomeserve.tools.baselines import AttemptToolService
from outcomeserve.contracts import _unique_keys


def serve(database, barrier_file):
    store = StateStore(database)
    released = Event()
    control = {"drop": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def do_GET(self):
            self.reply({"ready": True, "pid": os.getpid()})

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4_000_000:
                    raise ValueError("invalid request size")
                envelope = json.loads(self.rfile.read(size), object_pairs_hook=_unique_keys)
                if self.path == "/release":
                    control["drop"] = bool(envelope.get("drop"))
                    released.set()
                    self.reply({"released": True})
                    return
                def after_commit(response):
                    if envelope.get("pause_after_commit"):
                        released.clear()
                        control["drop"] = False
                        with Path(barrier_file).open("w", encoding="utf-8") as stream:
                            stream.write(json.dumps({**response, "logical_step_id": envelope.get("logical_step_id", response["step_id"])}))
                            stream.flush()
                            os.fsync(stream.fileno())
                        if not released.wait(30) or control["drop"]:
                            raise TimeoutError("acknowledgment suppressed after durable tool commit")
                request = envelope["request"]
                if self.path == "/submit":
                    if "logical_step_id" in envelope:
                        service = AttemptToolService(store, after_commit=after_commit)
                        response = service.submit(logical_step_id=envelope["logical_step_id"],
                                                  attempt=envelope["execution_attempt"], **request)
                    else:
                        response = ToolService(store, after_commit=after_commit).submit(**request)
                elif self.path == "/query":
                    response = ToolService(store).query(**request)
                else:
                    raise ValueError("unknown tool endpoint")
                self.reply(response)
            except TimeoutError as exc:
                self.reply({"error": str(exc)}, 504)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply({"error": str(exc)}, 400)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    print(json.dumps({"port": server.server_port, "pid": os.getpid()}), flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--barrier-file", required=True)
    args = parser.parse_args()
    serve(args.db, args.barrier_file)


if __name__ == "__main__":
    main()
