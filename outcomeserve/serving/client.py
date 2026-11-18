"""Strict token-ID completion client for an explicitly configured serving API."""

import http.client
import json
from urllib.request import Request, urlopen


class InferenceError(RuntimeError):
    pass


class InferenceUnavailable(InferenceError):
    """The inference process went away: connection failure or a stream cut before finish_reason.

    Unlike other InferenceErrors this is not a model or protocol failure; the
    caller may wait for a replacement worker and replay the committed prefix.
    """


TRANSPORT_ERRORS = (OSError, http.client.HTTPException)


class CompletionBackend:
    """Endpoint must support token-ID prompts and return generated token IDs.

    Missing token IDs fail closed: re-tokenizing generated text is not replay.
    Pilot must check support in the actual locked vLLM release.
    """

    def __init__(self, endpoint, model, context_spec, *, timeout=120):
        from outcomeserve.context import context_spec_id
        self.endpoint, self.model, self.timeout = endpoint.rstrip("/"), model, timeout
        self.context_spec = context_spec
        self.context_id = context_spec_id(context_spec)
        if context_spec["backend"] != "vllm" or model != context_spec["model_id"]:
            raise ValueError("serving backend/model differs from pinned context")

    def tokenize(self, text):
        request = Request(self.endpoint + "/tokenize", data=json.dumps({"model": self.model, "prompt": text}).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                tokens = json.load(response)["tokens"]
        except TRANSPORT_ERRORS as exc:
            raise InferenceUnavailable(f"tokenize failed: {exc}") from exc
        if not isinstance(tokens, list) or not tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise InferenceError("tokenize endpoint did not return token IDs")
        return tokens

    def generate_until_fault(self, prefix_token_ids, on_first_token, *, max_tokens=256, seed=33):
        """Confirm an unfinished stream token before triggering inference loss."""
        payload = {"model": self.model, "prompt": prefix_token_ids, "max_tokens": max_tokens,
                   "temperature": 0, "seed": seed, "return_token_ids": True, "stream": True}
        request = Request(self.endpoint + "/v1/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=self.timeout) as response:
            for line in response:
                if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                message = json.loads(line[5:])
                choices = message.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                tokens = choice.get("token_ids")
                if tokens and choice.get("finish_reason") is None:
                    if any(type(token) is not int or token < 0 for token in tokens):
                        raise InferenceError("invalid streamed token IDs")
                    on_first_token(tokens)
                    return
        raise InferenceError("no unfinished token-ID stream observed; fault precondition not met")

    def generate_stream(self, prefix_token_ids, on_first_token, *, max_tokens=256, seed=33):
        payload = {"model": self.model, "prompt": prefix_token_ids, "max_tokens": max_tokens,
                   "temperature": 0, "seed": seed, "return_token_ids": True, "stream": True}
        request = Request(self.endpoint + "/v1/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        tokens, text, observed, finished = [], [], False, False
        try:
            with urlopen(request, timeout=self.timeout) as response:
                for line in response:
                    if not line.endswith(b"\n"):
                        break  # cut mid-event: the server died while writing
                    if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                        continue
                    choices = json.loads(line[5:]).get("choices", [])
                    if not choices:
                        continue
                    choice = choices[0]
                    chunk = choice.get("token_ids", [])
                    if not isinstance(chunk, list) or any(type(t) is not int or t < 0 for t in chunk):
                        raise InferenceError("stream requires exact per-chunk generated token IDs")
                    if chunk and not observed:
                        on_first_token(chunk)
                        observed = True
                    tokens.extend(chunk)
                    text.append(choice.get("text", ""))
                    finished = finished or choice.get("finish_reason") is not None
        except TRANSPORT_ERRORS as exc:
            raise InferenceUnavailable(f"stream broke after {len(tokens)} tokens: {exc}") from exc
        if not finished:
            # A killed server can close the socket cleanly; without finish_reason
            # the partial text is lost generation, not a malformed tool call.
            raise InferenceUnavailable(f"stream ended without finish_reason after {len(tokens)} tokens")
        if not tokens:
            raise InferenceError("stream did not expose generated token IDs")
        try:
            call = json.loads("".join(text))
        except ValueError as exc:
            raise InferenceError("stream output is not a structured tool call") from exc
        if not isinstance(call, dict) or set(call) != {"tool_name", "args"} or not isinstance(call["args"], dict):
            raise InferenceError("invalid streamed tool call")
        return {"call": call, "generated_token_ids": tokens, "context_spec_id": self.context_id}

    def generate(self, prefix_token_ids, *, max_tokens=256, seed=33):
        if not isinstance(prefix_token_ids, list) or any(type(token) is not int or token < 0 for token in prefix_token_ids):
            raise ValueError("exact prompt token IDs required")
        payload = {"model": self.model, "prompt": prefix_token_ids, "max_tokens": max_tokens,
                   "temperature": 0, "seed": seed, "return_token_ids": True}
        request = Request(self.endpoint + "/v1/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                answer = json.load(response)
        except TRANSPORT_ERRORS as exc:
            raise InferenceUnavailable(f"completion failed: {exc}") from exc
        choice = answer["choices"][0]
        token_ids = choice.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids or any(type(token) is not int or token < 0 for token in token_ids):
            raise InferenceError("server must return exact generated token_ids; no text re-tokenization fallback")
        try:
            call = json.loads(choice["text"])
        except (ValueError, KeyError) as exc:
            raise InferenceError("model output is not a structured tool call") from exc
        if not isinstance(call, dict) or set(call) != {"tool_name", "args"} or not isinstance(call["args"], dict):
            raise InferenceError("expected tool_name/args object")
        return {"call": call, "generated_token_ids": token_ids, "usage": answer.get("usage", {}),
                "context_spec_id": self.context_id}
