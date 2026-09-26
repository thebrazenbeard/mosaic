# MOSAIC P0A — Analytical Residency Envelope V1

Status: **executable analytical preflight; not hardware qualification**

This preflight turns the Phase-0 memory-budget equation into deterministic code before any model download, GPU execution, or training.

It answers one narrow question:

> Given declared byte budgets for Boot/Core and each specialist, can a single-specialist swap sequence fit below a declared VRAM ceiling while the logical total pool is larger than that ceiling?

It deliberately does **not** measure:
- CUDA allocated/reserved memory;
- framework/kernel overhead not included in the declared inputs;
- actual unloading;
- host↔device transfer;
- load/unload latency;
- first-token latency;
- tokens/second;
- ten-cycle swap stability.

Therefore:

`ANALYTICAL_FEASIBILITY_ONLY != MOSAIC_P0_PASS`

The executable receipt binds the exact declared inputs by canonical SHA-256 and reports:
- declared ceiling;
- total declared GPU pool;
- modeled peak resident bytes;
- per-step modeled residency;
- whether the logical pool exceeds the ceiling;
- whether every modeled step fits.

The first hardware P0 experiment must replace modeled byte inputs with measured runtime values and satisfy all conditions already frozen in `MOSAIC_P0_RESIDENCY_V1.md`.
