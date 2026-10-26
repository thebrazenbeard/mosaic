# MOSAIC P0 — Residency Feasibility V1

Status: **predeclared experiment design**

## Question

Can Mosaic keep a logical capability pool larger than accelerator VRAM while guaranteeing that the active resident set remains within a fixed VRAM ceiling?

This phase does **not** test intelligence. It tests whether the hardware/runtime premise is physically workable.

## Experimental in-process ABI seam (metadata only)

`experiments/p0_inprocess_abi_probe.py` is a Win64-only native llama.cpp
**metadata** load/free preflight. It uses the actual `llama.dll` API within a
single Python process instead of a router that creates per-model workers.
It performs an A→B→A sequence 1–10 times and records native metadata-handle
load/free timings. It pins both model file hashes and a caller-supplied exact
`llama.dll` SHA-256; the inspected `0.4.1-dev` API and its 80-byte Win64
`llama_model_params` structure are required. Invoke only with a new output
path, e.g.:

```powershell
python -m experiments.p0_inprocess_abi_probe `
  --dll C:\path\to\llama.dll --dll-sha256 EXACT_INSPECTED_DLL_SHA256 `
  --model-a D:\path\to\A.gguf --model-b D:\path\to\B.gguf `
  --cycles 10 --out D:\path\to\new-metadata-receipt.json
```

**This is deliberately not a P0 pass.** The native call is made with
`vocab_only=true`, `no_alloc=true`, and `n_gpu_layers=0`: no neural weights,
GPU residency, KV cache, or inference are exercised. Staying in one Python PID
does not independently exclude unobserved child processes. This small step
only checks a usable ABI for a future persistent-worker implementation, which
must separately implement live decode, allocator/transfer telemetry, model
identity, worker/child-process tracking, and the full P0 gate.

## Target envelope

Primary target:

- GPU class: ~4 GB VRAM
- only one specialist needs to be resident at a time
- Boot/Core may be CPU-resident, GPU-resident, or hybrid, but its memory cost must be measured
- system RAM and storage may hold nonresident specialists

The exact model families are intentionally not frozen yet.

## Measurements

For every load/run/unload cycle record:

- specialist package size on disk;
- parameter count and quantization;
- peak allocated VRAM;
- peak reserved VRAM;
- KV-cache residency;
- activation/workspace overhead;
- Boot/Core residency;
- CPU RAM usage;
- specialist load time;
- unload time;
- first-token latency after a swap;
- tokens/second once warm;
- bytes transferred between host and device.

## Pass condition

P0 is a **PASS** for an exact subject only if all of the following are true:

1. peak measured device residency stays below the declared physical VRAM ceiling;
2. Specialist A can be unloaded and Specialist B loaded without process restart;
3. repeated A -> B -> A swaps succeed at least 10 consecutive times;
4. no swap requires reloading the complete Mosaic capability pool;
5. peak residency and swap latency are recorded by machine-readable receipt;
6. the exact model revisions, quantization, runtime versions, and hardware are bound into the receipt.

## Fail conditions

A run fails if:

- CUDA/accelerator OOM occurs;
- the runtime silently spills required active tensors to a different device while still claiming the target ceiling;
- the full capability pool must remain resident;
- measured memory exceeds the ceiling;
- a restart is required to change specialists;
- results cannot be reproduced from the recorded subject.

## Important distinction

A P0 pass establishes only:

> "This exact modular runtime can swap a larger total parameter pool through a smaller active VRAM envelope."

It does **not** establish that the specialists compose into one agent or equal a monolithic larger model.

## Partial device evidence from a dedicated llama.cpp router

`experiments/p0_router_probe.py` can exercise the installed router's real
`/models/load`, streaming `/completion`, and `/models/unload` endpoints through
ten A→B→A cycles while sampling total NVIDIA device memory. It requires a
**dedicated local router** with no models already loaded; it does not launch
or stop an existing service. Keep `--models-max 1`, set model sources/presets
outside shared services, and check `GET /models` for the exact model IDs.

```sh
python experiments/p0_router_probe.py \
  --router-url http://127.0.0.1:8517 --dedicated-router \
  --model-a EXACT_ROUTER_ID_A --model-b EXACT_ROUTER_ID_B \
  --model-a-file /path/to/model-a.gguf \
  --model-b-file /path/to/model-b.gguf \
  --out p0-observation.json
```

The probe hashes the model files, records loading/unloading and first-streamed
content timing, and emits an explicitly **partial** machine receipt. The
`--out` file must be a new path: the probe reserves it exclusively **before**
hashing model files or contacting the router and refuses to replace an existing
file or symlink. A failed/interrupted experiment may leave an empty reserved
file; that is **not a receipt** or evidence of completed cycles. Select a new
path for any retry and preserve the failed attempt for investigation.

The monitor's `nvidia-smi` device totals include other processes and may miss
transient peaks; the receipt does not attest per-model allocated/reserved
bytes, offload, KV/cache overhead, PCIe transfer, or stable model-worker PIDs.
llama.cpp router mode may spawn a fresh worker process on model load even
though the router process stays alive. Consequently **the probe cannot on its
own establish a Mosaic P0 pass**, especially the process-restart and complete
allocator-observation conditions above. It is a reproducible hardware-evidence
collection step, not a relaxation of the gate.
