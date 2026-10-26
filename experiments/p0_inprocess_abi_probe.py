"""Bounded llama.cpp same-PID ABI preflight; deliberately NOT a Mosaic P0 pass.

This tests actual native GGUF *metadata* load/free calls within one Python
process, without launching model workers or allocating neural weight tensors.
It is a prerequisite for, not a substitute for, a later in-process inference
and GPU-residency experiment. The ABI is pinned to the locally inspected
llama.cpp 0.4.1-dev Win64 model-params layout.
"""

from __future__ import annotations

import argparse
import ctypes
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
from typing import Callable


class NativeProbeError(RuntimeError):
    """Unsafe ABI, invalid subject, or failed native model operation."""


class _ModelParams(ctypes.Structure):
    # Exact layout from llama.cpp/include/llama.h at locally inspected Win64
    # 0.4.1-dev. A future ABI change MUST NOT silently reuse this layout.
    _fields_ = [
        ("devices", ctypes.c_void_p),
        ("tensor_buft_overrides", ctypes.c_void_p),
        ("n_gpu_layers", ctypes.c_int32),
        ("split_mode", ctypes.c_int32),
        ("load_mode", ctypes.c_int32),
        ("lazy_mode", ctypes.c_int32),
        ("main_gpu", ctypes.c_int32),
        ("tensor_split", ctypes.c_void_p),
        ("progress_callback", ctypes.c_void_p),
        ("progress_callback_user_data", ctypes.c_void_p),
        ("kv_overrides", ctypes.c_void_p),
        ("vocab_only", ctypes.c_bool),
        ("check_tensors", ctypes.c_bool),
        ("use_extra_bufts", ctypes.c_bool),
        ("no_host", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("load_mtp", ctypes.c_bool),
    ]


class NativeLlama:
    """Native ABI owner, restricted to non-allocating CPU metadata loads."""

    def __init__(self, dll: Path, expected_sha256: str) -> None:
        if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
            raise NativeProbeError("only inspected Win64 llama.cpp ABI is supported")
        if ctypes.sizeof(_ModelParams) != 80:
            raise NativeProbeError("llama model-params ABI size mismatch")
        if (type(expected_sha256) is not str or
                re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
            raise NativeProbeError("native DLL requires pinned lowercase SHA-256")
        self.dll = dll.resolve(strict=True)
        if self.dll.name.lower() != "llama.dll":
            raise NativeProbeError("expected the exact llama.dll library")
        # `0.4.1-dev` is not a unique ABI revision. Attest the precise DLL
        # bytes BEFORE loading any native code or attempting struct returns.
        if _sha256(self.dll) != expected_sha256:
            raise NativeProbeError("native DLL digest differs from inspected ABI")
        # Keep the DLL search handle alive for backend dependencies.
        self._search_handle = os.add_dll_directory(str(self.dll.parent))
        self.lib = ctypes.CDLL(str(self.dll))
        self.lib.llama_version.restype = ctypes.c_char_p
        version = self.lib.llama_version()
        if version != b"0.4.1-dev":
            self._search_handle.close()
            raise NativeProbeError(f"uninspected llama.dll version: {version!r}")
        self.lib.llama_model_default_params.restype = _ModelParams
        self.lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, _ModelParams]
        self.lib.llama_model_load_from_file.restype = ctypes.c_void_p
        self.lib.llama_model_free.argtypes = [ctypes.c_void_p]
        self.lib.llama_model_free.restype = None
        self.lib.llama_backend_init.argtypes = []
        self.lib.llama_backend_free.argtypes = []
        self.version = version.decode("ascii")
        self.sha256 = expected_sha256
        self._started = False

    def __enter__(self) -> "NativeLlama":
        self.lib.llama_backend_init()
        self._started = True
        return self

    def __exit__(self, *_: object) -> None:
        try:
            if self._started:
                self.lib.llama_backend_free()
                self._started = False
        finally:
            self._search_handle.close()

    def load_metadata(self, path: Path) -> object:
        if not self._started:
            raise NativeProbeError("native backend is not initialized")
        params = self.lib.llama_model_default_params()
        # This is intentionally NOT neural loading or inference. All tensor
        # allocations and GPU offloading remain disabled for this preflight.
        params.n_gpu_layers = 0
        params.vocab_only = True
        params.no_alloc = True
        model = self.lib.llama_model_load_from_file(os.fsencode(str(path)), params)
        if not model:
            raise NativeProbeError(f"native metadata load failed for {path.name}")
        return model

    def free(self, model: object) -> None:
        self.lib.llama_model_free(model)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_metadata_cycles(
    loader: object,
    subjects: dict[str, dict[str, object]],
    cycles: int,
    *,
    pid: Callable[[], int] = os.getpid,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    if type(cycles) is not int or not 1 <= cycles <= 10:
        raise NativeProbeError("cycles must be 1..10")
    if set(subjects) != {"A", "B"}:
        raise NativeProbeError("exactly two distinct model subjects required")
    for name in ("A", "B"):
        item = subjects[name]
        if (type(item) is not dict or not isinstance(item.get("path"), Path)
                or type(item.get("size_bytes")) is not int
                or item["size_bytes"] <= 0
                or type(item.get("sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None):
            raise NativeProbeError("subjects require pinned path, size, and SHA-256")
    if subjects["A"]["sha256"] == subjects["B"]["sha256"]:
        raise NativeProbeError("A and B must have distinct model digests")
    start_pid = pid()
    steps: list[dict[str, object]] = []
    for cycle in range(cycles):
        for role in ("A", "B", "A"):
            before = clock()
            model = loader.load_metadata(subjects[role]["path"])
            try:
                if pid() != start_pid:
                    raise NativeProbeError("process identity changed after native load")
            finally:
                loader.free(model)
            if pid() != start_pid:
                raise NativeProbeError("process identity changed after native free")
            steps.append({"cycle": cycle, "role": role,
                          "load_free_elapsed_ms": round((clock() - before) * 1000, 3)})
    return {
        "schema_id": "MOSAIC_P0_INPROCESS_ABI_METADATA_V1",
        "qualification": "METADATA_ONLY_NATIVE_ABI_PREFLIGHT_NOT_P0_PASS",
        "p0_pass": False,
        "neural_weights_loaded": False,
        "inference_completed": False,
        "gpu_residency_attested": False,
        "child_process_restarts_excluded": False,
        "same_python_pid_during_calls": True,
        "process_pid": start_pid,
        "metadata_aba_cycles": cycles,
        "subjects": {key: {k: (str(v) if isinstance(v, Path) else v)
                           for k, v in item.items()} for key, item in subjects.items()},
        "steps": steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--dll-sha256", required=True)
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        # Reserve the private receipt before hashing or model metadata loads.
        fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except OSError as exc:
        parser.error(f"receipt must be a new private file: {exc}")
    with os.fdopen(fd, "w", encoding="utf-8") as sink:
        paths = {"A": args.model_a.resolve(strict=True),
                 "B": args.model_b.resolve(strict=True)}
        subjects = {name: {"path": path, "size_bytes": path.stat().st_size,
                           "sha256": _sha256(path)} for name, path in paths.items()}
        with NativeLlama(args.dll, args.dll_sha256) as backend:
            receipt = run_metadata_cycles(backend, subjects, args.cycles)
            receipt["native_version"] = backend.version
            receipt["native_dll_sha256"] = backend.sha256
        sink.write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"qualification": receipt["qualification"],
                          "cycles": receipt["metadata_aba_cycles"],
                          "pid": receipt["process_pid"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
