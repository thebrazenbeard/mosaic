"""Experimental same-PID llama.cpp neural weight-load probe (not a P0 pass).

This extends the metadata-only ABI check to actual GGUF tensor allocation.
It deliberately does not initialize a context, decode tokens, or claim shared
learned state. Requires a previously inspected, exact SHA-pinned Win64 DLL.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Callable

try:
    from experiments.p0_inprocess_abi_probe import NativeLlama, NativeProbeError, _sha256
except ModuleNotFoundError:  # direct invocation with experiments as sys.path[0]
    from p0_inprocess_abi_probe import NativeLlama, NativeProbeError, _sha256


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_uint32), ("memory_load", ctypes.c_uint32),
        ("total_phys", ctypes.c_uint64), ("avail_phys", ctypes.c_uint64),
        ("total_page", ctypes.c_uint64), ("avail_page", ctypes.c_uint64),
        ("total_virtual", ctypes.c_uint64), ("avail_virtual", ctypes.c_uint64),
        ("avail_extended_virtual", ctypes.c_uint64),
    ]


def free_physical_mib() -> int:
    if os.name != "nt":
        raise NativeProbeError("live weight test requires inspected Windows host")
    status = _MemoryStatus()
    status.length = ctypes.sizeof(_MemoryStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise NativeProbeError("GlobalMemoryStatusEx failed")
    return int(status.avail_phys // (1024 * 1024))


def gpu_memory_mib() -> tuple[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=8, check=True,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise NativeProbeError("expected exactly one NVIDIA device")
    match = re.fullmatch(r"\s*(\d+)\s*,\s*(\d+)\s*", rows[0])
    if not match:
        raise NativeProbeError("unrecognized nvidia-smi memory report")
    return int(match.group(1)), int(match.group(2))


class Telemetry:
    """Best-effort GPU sampler; no claim to allocator/transfer attribution."""

    def __init__(self, sample: Callable[[], tuple[int, int]] = gpu_memory_mib) -> None:
        self.sample = sample
        self.samples: list[dict[str, object]] = []
        self.errors: list[str] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Telemetry":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                total, used = self.sample()
                self.samples.append({"monotonic_ns": time.monotonic_ns(),
                                     "total_mib": total, "used_mib": used})
            except (OSError, subprocess.SubprocessError, NativeProbeError) as exc:
                self.errors.append(str(exc))
            self._stop.wait(0.5)

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)


def observed_gpu_peak_mib(
    samples: list[dict[str, object]], steps: list[dict[str, object]],
) -> int | None:
    """Include synchronous checkpoints if the polling thread misses a peak."""
    observed = [s["used_mib"] for s in samples]
    for step in steps:
        observed.extend([step["device_used_before_mib"],
                         step["device_used_loaded_mib"],
                         step["device_used_freed_mib"]])
    return max(observed, default=None)


class WeightBackend(NativeLlama):
    """Exact pinned DLL with real neural tensor allocation, bounded GPU layers."""

    def __init__(self, dll: Path, digest: str, ggml_digest: str) -> None:
        super().__init__(dll, digest)
        ggml_dll = self.dll.parent / "ggml.dll"
        # The same package's GGML registry is required for actual tensor
        # allocation; vocabulary-only metadata loading did not exercise it.
        if (type(ggml_digest) is not str or
                re.fullmatch(r"[0-9a-f]{64}", ggml_digest) is None or
                _sha256(ggml_dll) != ggml_digest):
            self._search_handle.close()
            raise NativeProbeError("GGML registry DLL differs from inspected bytes")
        self.ggml = ctypes.CDLL(str(ggml_dll))
        self.ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
        self.ggml.ggml_backend_load_all_from_path.restype = None
        self.ggml_sha256 = ggml_digest
        self.lib.llama_model_n_layer.argtypes = [ctypes.c_void_p]
        self.lib.llama_model_n_layer.restype = ctypes.c_int32
        self.lib.llama_model_n_params.argtypes = [ctypes.c_void_p]
        self.lib.llama_model_n_params.restype = ctypes.c_uint64

    def __enter__(self) -> "WeightBackend":
        self.ggml.ggml_backend_load_all_from_path(os.fsencode(str(self.dll.parent)))
        super().__enter__()
        return self

    def load_weights(self, path: Path, gpu_layers: int) -> tuple[object, dict[str, int]]:
        if not self._started:
            raise NativeProbeError("native backend not initialized")
        if type(gpu_layers) is not int or not 0 <= gpu_layers <= 32:
            raise NativeProbeError("gpu_layers must be 0..32")
        params = self.lib.llama_model_default_params()
        params.n_gpu_layers = gpu_layers
        params.vocab_only = False
        params.no_alloc = False
        model = self.lib.llama_model_load_from_file(os.fsencode(str(path)), params)
        if not model:
            raise NativeProbeError(f"native weight load failed for {path.name}")
        return model, {"n_layer": self.lib.llama_model_n_layer(model),
                       "n_params": self.lib.llama_model_n_params(model)}


def run_weight_cycle(
    loader: object, subjects: dict[str, dict[str, object]],
    *, cycles: int, gpu_layers: int, min_free_ram_mib: int,
    available_ram: Callable[[], int] = free_physical_mib,
    gpu: Callable[[], tuple[int, int]] = gpu_memory_mib,
    pid: Callable[[], int] = os.getpid,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    if type(cycles) is not int or not 1 <= cycles <= 10:
        raise NativeProbeError("cycles must be 1..10")
    if type(gpu_layers) is not int or not 0 <= gpu_layers <= 32:
        raise NativeProbeError("gpu_layers must be 0..32")
    if type(min_free_ram_mib) is not int or min_free_ram_mib < 2048:
        raise NativeProbeError("minimum free RAM must be >= 2048 MiB")
    if set(subjects) != {"A", "B"}:
        raise NativeProbeError("two distinct model subjects required")
    for subject in subjects.values():
        if (type(subject) is not dict or not isinstance(subject.get("path"), Path)
                or type(subject.get("sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", subject["sha256"]) is None
                or type(subject.get("size_bytes")) is not int
                or subject["size_bytes"] <= 0):
            raise NativeProbeError("invalid model subject manifest")
    if subjects["A"]["sha256"] == subjects["B"]["sha256"]:
        raise NativeProbeError("models must be distinct neural revisions")

    process_pid = pid()
    steps: list[dict[str, object]] = []
    for cycle in range(cycles):
        for role in ("A", "B", "A"):
            ram_before = available_ram()
            if ram_before < min_free_ram_mib:
                raise NativeProbeError(
                    f"memory safety stop: {ram_before} MiB free < {min_free_ram_mib} MiB")
            device_total, device_used_before = gpu()
            started = clock()
            model, metadata = loader.load_weights(subjects[role]["path"], gpu_layers)
            try:
                if pid() != process_pid:
                    raise NativeProbeError("process PID changed after native weight load")
                _, device_used_loaded = gpu()
                ram_loaded = available_ram()
                if metadata["n_layer"] <= 0 or metadata["n_params"] <= 0:
                    raise NativeProbeError("loaded model has invalid neural dimensions")
            finally:
                loader.free(model)
            if pid() != process_pid:
                raise NativeProbeError("process PID changed after native weight free")
            _, device_used_freed = gpu()
            steps.append({
                "cycle": cycle, "role": role, "neural_weight_load_free_ms":
                round((clock() - started) * 1000, 2),
                "device_total_mib": device_total,
                "device_used_before_mib": device_used_before,
                "device_used_loaded_mib": device_used_loaded,
                "device_used_freed_mib": device_used_freed,
                "free_ram_before_mib": ram_before,
                "free_ram_loaded_mib": ram_loaded,
                **metadata,
            })
    return {
        "schema_id": "MOSAIC_P0_INPROCESS_NEURAL_WEIGHT_V1",
        "qualification": "NEURAL_WEIGHTS_LOAD_FREE_ONLY_NOT_P0_PASS",
        "p0_pass": False,
        "neural_weights_loaded": True,
        "inference_completed": False,
        "gpu_residency_attested": False,
        "child_process_restarts_excluded": False,
        "shared_learned_state_demonstrated": False,
        "same_python_pid_during_calls": True,
        "process_pid": process_pid,
        "neural_aba_cycles": cycles,
        "gpu_layers_requested": gpu_layers,
        "subjects": {key: {k: (str(v) if isinstance(v, Path) else v)
                           for k, v in item.items()} for key, item in subjects.items()},
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--dll-sha256", required=True)
    parser.add_argument("--ggml-sha256", required=True)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--gpu-layers", type=int, default=16)
    parser.add_argument("--min-free-ram-mib", type=int, default=2300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("native live weight probe is Win64-only")
    if not 1 <= args.cycles <= 10 or not 0 <= args.gpu_layers <= 32:
        parser.error("cycles=1..10 and gpu-layers=0..32 required")
    if args.min_free_ram_mib < 2048:
        parser.error("min-free-ram-mib must be >= 2048")
    try:
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        parser.error(f"receipt must be a new private file: {exc}")
    with os.fdopen(fd, "w", encoding="utf-8") as sink:
        paths = {"A": args.model_a.resolve(strict=True),
                 "B": args.model_b.resolve(strict=True)}
        subjects = {name: {"path": path, "size_bytes": path.stat().st_size,
                           "sha256": _sha256(path)} for name, path in paths.items()}
        with WeightBackend(args.dll, args.dll_sha256, args.ggml_sha256) as backend:
            with Telemetry() as telemetry:
                receipt = run_weight_cycle(
                    backend, subjects, cycles=args.cycles,
                    gpu_layers=args.gpu_layers,
                    min_free_ram_mib=args.min_free_ram_mib,
                )
            receipt["gpu_total_samples"] = len(telemetry.samples)
            receipt["gpu_sample_errors"] = telemetry.errors
            receipt["gpu_peak_observed_mib"] = observed_gpu_peak_mib(
                telemetry.samples, receipt["steps"])
            receipt["native_version"] = backend.version
            receipt["native_dll_sha256"] = backend.sha256
            receipt["ggml_dll_sha256"] = backend.ggml_sha256
        sink.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"qualification": receipt["qualification"],
                          "cycles": receipt["neural_aba_cycles"],
                          "pid": receipt["process_pid"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
