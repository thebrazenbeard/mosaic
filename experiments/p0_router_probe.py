"""Conservative live device probe for Mosaic P0, using a dedicated llama.cpp router.

Unlike runtime.residency's analytical envelope this invokes real model loads,
inference, and unloads, and samples total device VRAM with nvidia-smi. The
receipt is *not* a P0 pass: allocations, reserved memory, offload, and model
worker process lifecycle still need separate attestation.

Requires a dedicated, localhost-bound llama-server in router mode. Never point
this at a shared router: /models/unload changes its runtime state.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import threading
import time
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class ProbeError(RuntimeError):
    """A hardware, router, or evidence-boundary precondition failed."""


_MAX_CONTROL_REPLY_BYTES = 1024 * 1024
_MAX_STREAM_LINE_BYTES = 64 * 1024
_MAX_STREAM_REPLY_BYTES = 2 * 1024 * 1024
_MAX_STREAM_EVENTS = 2048


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request: Request, fp: object, code: int,
                         msg: str, headers: object, newurl: str) -> None:
        raise ProbeError("router redirected a request outside the pinned endpoint")


def require_loopback_router(url: str) -> str:
    """Avoid sending model-management commands to network services."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ProbeError("router must be a bare localhost HTTP origin")
    return url.rstrip("/")


def parse_nvidia_mib(line: str) -> tuple[int, int]:
    """Parse --query-gpu=memory.used,memory.total --format=csv,noheader,nounits."""
    try:
        parts = line.strip().split(",")
        if len(parts) != 2:
            raise ValueError("bad field count")
        used, total = (int(value.strip()) for value in parts)
        if total <= 0 or used < 0 or used > total:
            raise ValueError("impossible reported device memory")
        return used, total
    except ValueError as exc:
        raise ProbeError(f"invalid nvidia-smi memory sample: {line!r}") from exc


class NvidiaSampler:
    """Capture total device usage; no claim about per-model allocations."""

    def __init__(self, binary: str = "nvidia-smi", gpu_index: int = 0) -> None:
        if gpu_index < 0:
            raise ProbeError("GPU index must be non-negative")
        self.binary = binary
        self.gpu_index = gpu_index
        self.samples: list[dict[str, float | int]] = []
        self.errors: list[str] = []
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "NvidiaSampler":
        self._process = subprocess.Popen(
            [
                self.binary,
                "--id=" + str(self.gpu_index),
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
                "--loop-ms=250",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        def read_samples() -> None:
            assert self._process is not None and self._process.stdout is not None
            for raw in self._process.stdout:
                if not raw.strip():
                    continue
                try:
                    used, total = parse_nvidia_mib(raw)
                except ProbeError as exc:
                    self.errors.append(str(exc))
                    continue
                self.samples.append(
                    {"monotonic_s": time.monotonic(), "used_mib": used, "total_mib": total}
                )

        self._thread = threading.Thread(target=read_samples, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)


class Router:
    def __init__(self, url: str, timeout_s: float = 90.0) -> None:
        self.url = require_loopback_router(url)
        self.timeout_s = timeout_s
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def request(self, path: str, payload: dict[str, object] | None = None) -> dict[str, object]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {} if payload is None else {"Content-Type": "application/json"}
        request = Request(self.url + path, data=body, headers=headers)
        deadline = time.monotonic() + self.timeout_s
        try:
            with self._opener.open(request, timeout=self.timeout_s) as response:
                chunks: list[bytes] = []
                consumed = 0
                while True:
                    if time.monotonic() >= deadline:
                        raise ProbeError(f"router API {path} exceeded wall deadline")
                    # HTTPResponse.read1 returns the available buffered bytes
                    # without demanding a full fixed-sized read. Read bounded
                    # chunks so a local misbehaving router cannot return an
                    # unbounded JSON object.
                    chunk = response.read1(8192)
                    if not chunk:
                        break
                    consumed += len(chunk)
                    if consumed > _MAX_CONTROL_REPLY_BYTES:
                        raise ProbeError(f"router API {path} response exceeds byte limit")
                    chunks.append(chunk)
                if time.monotonic() >= deadline:
                    raise ProbeError(f"router API {path} exceeded wall deadline")
                output = json.loads(b"".join(chunks))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise ProbeError(f"router API {path} failed: {exc}") from exc
        if not isinstance(output, dict):
            raise ProbeError(f"router {path} returned a non-object")
        return output

    def model_records(self) -> dict[str, dict[str, object]]:
        output = self.request("/models")
        rows = output.get("data")
        if not isinstance(rows, list):
            raise ProbeError("router /models lacks data array")
        records: dict[str, dict[str, object]] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ProbeError("malformed router model record")
            if row["id"] in records:
                raise ProbeError("duplicate router model ID")
            records[row["id"]] = row
        return records

    def statuses(self) -> dict[str, str]:
        records = self.model_records()
        result = {}
        for name, row in records.items():
            status = row.get("status")
            result[name] = status.get("value", "") if isinstance(status, dict) else ""
        return result

    def paths(self) -> dict[str, str]:
        records = self.model_records()
        result = {}
        for name, row in records.items():
            path = row.get("path")
            if isinstance(path, str) and path:
                result[name] = path
        return result

    def load(self, model: str) -> None:
        reply = self.request("/models/load", {"model": model})
        if reply.get("success") is not True:
            raise ProbeError(f"load was not acknowledged for {model!r}")

    def unload(self, model: str) -> None:
        reply = self.request("/models/unload", {"model": model})
        if reply.get("success") is not True:
            raise ProbeError(f"unload was not acknowledged for {model!r}")

    def wait_status(self, model: str, expected_loaded: bool) -> None:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            status = self.statuses().get(model)
            if status == ("loaded" if expected_loaded else "unloaded"):
                return
            if status in {"failed", "error", None}:
                raise ProbeError(f"router reported terminal/missing status {status!r} for {model!r}")
            time.sleep(0.2)
        raise ProbeError(f"router model {model!r} did not reach desired loaded={expected_loaded}")

    def generate_stream(self, model: str) -> dict[str, float | int]:
        payload = {
            "model": model,
            "prompt": "Return the word ready.",
            "n_predict": 12,
            "stream": True,
        }
        req = Request(
            self.url + "/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        first: float | None = None
        last: float | None = None
        tokens = 0
        received_bytes = 0
        frames = 0
        finished = False
        deadline = started + self.timeout_s
        try:
            with self._opener.open(req, timeout=self.timeout_s) as response:
                while True:
                    if time.monotonic() >= deadline:
                        raise ProbeError(f"model {model!r} streaming deadline exceeded")
                    raw = response.readline(_MAX_STREAM_LINE_BYTES + 1)
                    if not raw:
                        break
                    if time.monotonic() >= deadline:
                        raise ProbeError(f"model {model!r} streaming deadline exceeded")
                    received_bytes += len(raw)
                    if len(raw) > _MAX_STREAM_LINE_BYTES or (
                        received_bytes > _MAX_STREAM_REPLY_BYTES
                    ):
                        raise ProbeError(f"model {model!r} streaming response exceeds byte limit")
                    if not raw.startswith(b"data: "):
                        continue
                    frames += 1
                    if frames > _MAX_STREAM_EVENTS:
                        raise ProbeError(f"model {model!r} streaming response exceeds event limit")
                    body = raw[6:].strip()
                    if body == b"[DONE]":
                        finished = True
                        break
                    event = json.loads(body)
                    if not isinstance(event, dict):
                        continue
                    if event.get("content"):
                        tokens += 1
                        now = time.monotonic()
                        if first is None:
                            first = now
                        last = now
                    if event.get("stop") is True:
                        finished = True
                        break
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            raise ProbeError(f"model {model!r} streaming inference failed: {exc}") from exc
        if not finished:
            raise ProbeError(f"model {model!r} stream ended without a completion marker")
        if first is None or last is None or tokens == 0:
            raise ProbeError(f"model {model!r} generated no observed token")
        elapsed = max(last - first, 0.001)
        return {
            "first_token_latency_s": round(first - started, 4),
            "observed_content_chunks": tokens,
            "observed_chunks_per_s_after_first": round(max(0, tokens - 1) / elapsed, 4),
            "end_to_end_s": round(time.monotonic() - started, 4),
        }


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_probe(
    router: Router,
    model_a: str,
    model_b: str,
    cycles: int,
    sampler: NvidiaSampler,
    model_subjects: dict[str, dict[str, object]],
    *,
    environment_subject: dict[str, object] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    if cycles < 10 or model_a == model_b:
        raise ProbeError("require at least ten A-B-A cycles and two distinct models")
    for model in (model_a, model_b):
        data = model_subjects.get(model)
        if (
            not isinstance(data, dict)
            or type(data.get("size_bytes")) is not int
            or data["size_bytes"] <= 0
            or not isinstance(data.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", data["sha256"]) is None
        ):
            raise ProbeError("each model requires an exact file size and lowercase SHA-256")
    statuses = router.statuses()
    if model_a not in statuses or model_b not in statuses:
        raise ProbeError("both models must be advertised by dedicated router")
    if (
        any(status in {"loaded", "loading", "unloading"} for status in statuses.values())
        or statuses[model_a] != "unloaded"
        or statuses[model_b] != "unloaded"
    ):
        raise ProbeError("dedicated router must start with zero loaded models")
    advertised_paths = router.paths()
    for model in (model_a, model_b):
        exact_path = model_subjects[model].get("path")
        if not isinstance(exact_path, str) or not exact_path:
            raise ProbeError("model subject requires a canonical file path")
        observed_path = advertised_paths.get(model)
        if (
            not isinstance(observed_path, str)
            or os.path.normcase(os.path.normpath(observed_path))
               != os.path.normcase(os.path.normpath(exact_path))
        ):
            raise ProbeError("router-advertised model path does not match hashed subject")

    steps: list[dict[str, object]] = []
    succeeded_cycles = 0
    for index in range(cycles):
        for ordinal, model in enumerate((model_a, model_b, model_a)):
            loaded = False
            load_attempted = False
            primary_failure: BaseException | None = None
            started = clock()
            try:
                load_attempted = True
                router.load(model)
                loaded = True
                router.wait_status(model, True)
                load_s = clock() - started
                generation = router.generate_stream(model)
            except BaseException as exc:
                primary_failure = exc
                raise
            finally:
                if load_attempted:
                    try:
                        unload_started = clock()
                        if loaded:
                            router.unload(model)
                            router.wait_status(model, False)
                        else:
                            # The router can load a model but lose its HTTP
                            # acknowledgement. A failed load call therefore
                            # does not prove nothing was loaded. Inspect only
                            # this model's status and clean it up if needed;
                            # never unload unrelated router model IDs.
                            status = router.statuses().get(model)
                            if status in {"loaded", "loading"}:
                                router.unload(model)
                                router.wait_status(model, False)
                            elif status == "unloading":
                                router.wait_status(model, False)
                            elif status != "unloaded":
                                raise ProbeError(
                                    f"cannot establish safe unload state for {model!r}"
                                )
                        unload_s = clock() - unload_started
                    except Exception as cleanup_exc:
                        if primary_failure is None:
                            raise
                        # Do not overwrite the original load/inference failure;
                        # preserve the cleanup failure as an explicit diagnostic.
                        primary_failure.add_note(
                            f"uncertain router cleanup for {model!r}: {cleanup_exc!r}"
                        )
            steps.append(
                {
                    "cycle_index": index,
                    "step_index": ordinal,
                    "model": model,
                    "load_s": round(load_s, 4),
                    "unload_s": round(unload_s, 4),
                    **generation,
                }
            )
        succeeded_cycles += 1

    if not sampler.samples:
        raise ProbeError("no device memory samples; hardware observation unavailable")
    total_mib_set = {int(s["total_mib"]) for s in sampler.samples}
    if len(total_mib_set) != 1:
        raise ProbeError("device memory ceiling changed during run")
    total_mib = next(iter(total_mib_set))
    peak_mib = max(int(s["used_mib"]) for s in sampler.samples)
    subject = {
        "model_subjects": model_subjects,
        "environment_subject": environment_subject or {},
        "sequence": [s["model"] for s in steps],
        "cycles": cycles,
    }
    package_sum = int(model_subjects[model_a]["size_bytes"]) + int(model_subjects[model_b]["size_bytes"])
    return {
        "schema_id": "MOSAIC_P0_ROUTER_DEVICE_OBSERVATION_V1",
        "qualification": "PARTIAL_REAL_HARDWARE_OBSERVATION_NOT_P0_PASS",
        "p0_pass": False,
        "measured_device_total_vram": True,
        "per_model_allocated_reserved_verified": False,
        "child_process_restarts_excluded": False,
        "shared_device_interference_excluded": False,
        "subject_sha256": sha256(
            json.dumps(subject, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "device_total_mib": total_mib,
        "device_peak_used_mib": peak_mib,
        "device_baseline_used_mib": int(sampler.samples[0]["used_mib"]),
        "device_sample_count": len(sampler.samples),
        "logical_model_package_pool_bytes": package_sum,
        "logical_package_pool_exceeds_physical_vram": package_sum > total_mib * 1024 * 1024,
        "environment_subject": environment_subject or {},
        "completed_aba_cycles": succeeded_cycles,
        "steps": steps,
        "limitations": [
            "nvidia-smi totals cover all processes, not only Mosaic allocations",
            "250ms polling can miss transient true peaks and excludes allocator reserved metrics",
            "llama.cpp router can retain its main PID while child workers restart",
            "KV/workspace, offload topology, CPU memory, and host-device bytes not yet attested",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-url", required=True)
    parser.add_argument("--dedicated-router", action="store_true")
    parser.add_argument("--model-a", required=True)
    parser.add_argument("--model-b", required=True)
    parser.add_argument("--model-a-file", type=Path, required=True)
    parser.add_argument("--model-b-file", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not args.dedicated_router:
        parser.error("must explicitly attest --dedicated-router; this probe changes router model state")
    subjects = {}
    for label, path in ((args.model_a, args.model_a_file), (args.model_b, args.model_b_file)):
        if not path.is_file():
            parser.error(f"missing local model file: {path}")
        subjects[label] = {"file_name": path.name, "path": str(path.resolve()),
                           "size_bytes": path.stat().st_size, "sha256": file_sha256(path)}
    router = Router(args.router_url)
    environment_subject = {"platform": platform.platform(), "device_monitor": args.nvidia_smi,
                           "device_gpu_index": args.gpu_index,
                           "router_runtime_version": "UNATTESTED_NOT_CAPTURED"}
    with NvidiaSampler(args.nvidia_smi, args.gpu_index) as sampler:
        wait_until = time.monotonic() + 5
        while not sampler.samples and time.monotonic() < wait_until:
            time.sleep(0.1)
        if not sampler.samples:
            raise ProbeError("nvidia-smi did not provide a baseline sample")
        receipt = run_probe(router, args.model_a, args.model_b, args.cycles, sampler, subjects,
                            environment_subject=environment_subject)
    result = json.dumps(receipt, indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(result + "\n", encoding="utf-8")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
