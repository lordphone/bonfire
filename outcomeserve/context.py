"""Immutable context identity; replay uses stored IDs rather than new generation."""

import hashlib
import json


def context_spec_id(spec):
    required = {"model_id", "model_revision", "tokenizer_id", "tokenizer_revision",
                "chat_template_sha256", "enable_thinking", "backend"}
    if set(spec) != required:
        raise ValueError("context spec requires exact model/tokenizer/template/thinking/backend fields")
    if type(spec["enable_thinking"]) is not bool:
        raise ValueError("enable_thinking must be boolean")
    for key in required - {"enable_thinking"}:
        if not isinstance(spec[key], str) or not spec[key]:
            raise ValueError(f"invalid context spec field: {key}")
    for key in ("model_revision", "tokenizer_revision"):
        if len(spec[key]) != 40 or any(c not in "0123456789abcdef" for c in spec[key]):
            raise ValueError(f"{key} must be an immutable Git revision")
    digest = spec["chat_template_sha256"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("template must have an immutable SHA-256 digest")
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return "context-" + hashlib.sha256(encoded).hexdigest()


def scripted_context(revision):
    return {"model_id": "outcomeserve/scripted-trace", "model_revision": revision,
            "tokenizer_id": "outcomeserve/scripted-integers", "tokenizer_revision": revision,
            "chat_template_sha256": hashlib.sha256(b"scripted-full-prefix").hexdigest(),
            "enable_thinking": False, "backend": "scripted"}
