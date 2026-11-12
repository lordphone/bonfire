"""Deterministic stand-in for the vLLM OpenAI server, for CPU-only recovery tests.

Token IDs are Unicode code points, so a token-ID prompt can be decoded exactly.
The "model" answers with the JSON request that follows the last step prompt in
its input, which makes output a pure function of the prompt: a run that crashes
and replays its durable prefix must end with exactly the failure-free prefix.

Streams use chunked transfer and SSE like vLLM, end with a finish_reason and
[DONE], and pace chunks so a supervisor can kill the process mid-generation.

    python tests/fake_vllm.py --port 0 --model Qwen/Qwen3-4B
prints {"port": ..., "pid": ...} once it is listening.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import sys
import time

STEP_PROMPT = "\nExecute next request as JSON tool_name/args: "


def tokenize(text):
    return [ord(c) for c in text]


def answer_for(prompt_ids):
    text = "".join(chr(t) for t in prompt_ids)
    start = text.rfind(STEP_PROMPT)
    if start < 0:
        raise ValueError("prompt has no step request")
    return json.dumps(json.loads(text[start + len(STEP_PROMPT):]))


def serve(port, model, chunk_chars, chunk_delay):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def send_json(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/v1/models":
                self.send_json({"object": "list", "data": [{"id": model, "object": "model"}]})
            else:
                self.send_json({"error": "not found"}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                self.send_json({"tokens": tokenize(body["prompt"])})
                return
            if self.path != "/v1/completions" or body.get("model") != model:
                self.send_json({"error": "unsupported request"}, 400)
                return
            text = answer_for(body["prompt"])
            if not body.get("stream"):
                self.send_json({"choices": [{"text": text, "token_ids": tokenize(text), "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": len(text)}})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            pieces = [text[i:i + chunk_chars] for i in range(0, len(text), chunk_chars)]
            for index, piece in enumerate(pieces):
                finish = "stop" if index == len(pieces) - 1 else None
                self.write_chunk({"choices": [{"text": piece, "token_ids": tokenize(piece), "finish_reason": finish}]})
                time.sleep(chunk_delay)
            self.write_chunk("[DONE]")
            self.wfile.write(b"0\r\n\r\n")

        def write_chunk(self, value):
            line = ("data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n").encode()
            self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(json.dumps({"port": server.server_address[1], "pid": os.getpid()}), flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--chunk-chars", type=int, default=4)
    parser.add_argument("--chunk-delay", type=float, default=0.002)
    args = parser.parse_args()
    serve(args.port, args.model, args.chunk_chars, args.chunk_delay)


if __name__ == "__main__":
    sys.exit(main())
