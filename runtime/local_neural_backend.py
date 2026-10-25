"""Exact loopback JSON-HTTP boundary for Mosaic's learned thinking Core.

The backend CALLS an already-serving neural model. It never starts/stops a
service, changes weights, loads adapters, opens a non-loopback URL, executes
a tool, or confers decision authority. Endpoint metadata is not attestation.
"""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from .strict_json import loads as strict_json_loads


class LoopbackBackendError(RuntimeError):
    """Fail closed if the local model endpoint violates its declared contract."""


class _RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LoopbackBackendError("HTTP redirect prohibited for local Core inference")


def _parse_json(body: bytes) -> dict[str, object]:
    try:
        parsed = strict_json_loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LoopbackBackendError("Model endpoint returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise LoopbackBackendError("Model endpoint JSON must be an object")
    return parsed


class LoopbackCoreInference:
    """Callable adapter from the local Qwen HTTP API to NeuralThinkingCore.

    The host and route are compile-time fixed at 127.0.0.1. The only selectable
    network coordinate is a local port, allowing isolated test servers. This
    also prevents accidental public-provider calls or proxy redirects.
    """

    def __init__(
        self,
        *,
        port: int = 18081,
        model_id: str = "qwen3.5-4b-local",
        timeout_seconds: float = 40,
    ):
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("port must be integer 1..65535")
        if not isinstance(model_id, str) or not model_id.strip() or len(model_id) > 128:
            raise ValueError("model_id is required and bounded")
        if (type(timeout_seconds) not in (int, float)
                or not 0 < timeout_seconds <= 120):
            raise ValueError("timeout_seconds must be 0..120 seconds")
        self.port = port
        self.model_id = model_id
        self.timeout_seconds = float(timeout_seconds)
        self.endpoint = f"http://127.0.0.1:{port}"
        # Disable ambient HTTP_PROXY/HTTPS_PROXY and never follow redirects.
        self._opener = build_opener(ProxyHandler({}), _RejectRedirect())

    def _request_json(
        self,
        path: str,
        *,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        body = None if payload is None else json.dumps(
            payload, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        req = Request(
            self.endpoint + path,
            data=body,
            headers={"Accept": "application/json",
                     **({"Content-Type": "application/json"} if body is not None else {})},
            method="POST" if body is not None else "GET",
        )
        try:
            with self._opener.open(req, timeout=self.timeout_seconds) as response:
                if response.status != 200:
                    raise LoopbackBackendError(
                        f"Local model request returned HTTP {response.status}"
                    )
                if response.geturl() != self.endpoint + path:
                    raise LoopbackBackendError("Model endpoint changed origin or route")
                raw = response.read(65537)
        except LoopbackBackendError:
            raise
        except HTTPError as exc:
            exc.close()
            raise LoopbackBackendError("Local neural endpoint rejected request") from exc
        except (OSError, URLError) as exc:
            raise LoopbackBackendError("Local neural endpoint unavailable") from exc
        if len(raw) > 65536:
            raise LoopbackBackendError("Model response exceeds fixed byte limit")
        return _parse_json(raw)

    def preflight(self) -> dict[str, object]:
        result = self._request_json("/v1/models")
        rows = result.get("data")
        if (
            result.get("object") != "list"
            or not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
            or rows[0].get("id") != self.model_id
            or rows[0].get("object") != "model"
        ):
            raise LoopbackBackendError("Unexpected advertised model identifier")
        return {
            "status": "ENDPOINT_MODEL_ID_ADVERTISED_NOT_BEHAVIOR_VERIFIED",
            "endpoint": self.endpoint,
            "model_id": self.model_id,
            "model_identity_cryptographically_verified": False,
            "reasoning_quality_verified": False,
            "service_mutated": False,
        }

    def __call__(self, messages: tuple[dict[str, str], ...]) -> str:
        if (not isinstance(messages, tuple) or len(messages) != 2
                or any(type(item) is not dict or set(item) != {"role", "content"}
                       for item in messages)
                or [item["role"] for item in messages] != ["system", "user"]
                or any(not isinstance(item["content"], str)
                       or not item["content"].strip() for item in messages)):
            raise LoopbackBackendError("Invalid Core request message structure")
        encoded = json.dumps(messages, ensure_ascii=False).encode("utf-8")
        if len(encoded) > 16384:
            raise LoopbackBackendError("Core request message byte limit exceeded")

        # A model ID response means availability, not trustworthy model lineage.
        self.preflight()
        result = self._request_json(
            "/v1/chat/completions",
            payload={
                "model": self.model_id,
                "messages": list(messages),
                "tools": [],
                "stream": False,
            },
        )
        if result.get("object") != "chat.completion" or result.get("model") != self.model_id:
            raise LoopbackBackendError("Model response identity/schema mismatch")
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise LoopbackBackendError("Model response choice count mismatch")
        choice = choices[0]
        if not isinstance(choice, dict) or choice.get("index") != 0:
            raise LoopbackBackendError("Model response choice identity invalid")
        if choice.get("finish_reason") != "stop":
            raise LoopbackBackendError("Model response incomplete or tool-routed")
        msg = choice.get("message")
        if (
            not isinstance(msg, dict)
            or msg.get("role") != "assistant"
            or "tool_calls" in msg and msg["tool_calls"]
            or not isinstance(msg.get("content"), str)
            or not msg["content"].strip()
        ):
            raise LoopbackBackendError("Local model did not provide a plain-text answer")
        return msg["content"]
