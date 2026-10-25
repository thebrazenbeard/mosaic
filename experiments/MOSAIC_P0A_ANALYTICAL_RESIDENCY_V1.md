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

The legacy `logical_pool_exceeds_vram_ceiling` compares the sum of all
specialists' **declared weights and runtime buffers** plus Boot/Core GPU bytes
to the ceiling. It does not establish that learned parameters alone exceed
VRAM. The separate `total_declared_specialist_weight_bytes` and
`specialist_weight_pool_exceeds_vram_ceiling` fields isolate specialist
weights. They exclude Boot/Core learned weights because its declaration does
not separate weights from other GPU allocations. A false weight-only result
does not imply the complete learned parameter pool fits; it means that claim
cannot be established from the specialist weights alone.

The first hardware P0 experiment must replace modeled byte inputs with measured runtime values and satisfy all conditions already frozen in `MOSAIC_P0_RESIDENCY_V1.md`.

## Swap-transition headroom preflight

`runtime.residency.evaluate_transition_headroom` adds an **independent analytical
check** of peak allocation *during* a switch, not just after it. It supports
two explicitly assumed policies:

- `evict_then_load`: the old specialist's complete declared footprint is freed
  before the next is loaded (the P0/P1 design requires this ordering).
- `load_then_evict`: both complete declared specialist footprints overlap
  during a transition (a conservative bound, not observed loading behavior).

The caller can add a declared `persistent_overhead_bytes` margin for allocator,
driver, resident runtime and other fixed allocations not included in
`boot_core_gpu_bytes`. Passing zero is **not** a measurement of zero overhead.
Model package size on disk does not establish GPU allocation or host RAM use.

Reproduce the small synthetic test with:

```shell
python -m unittest discover -s tests -p test_swap_transitions.py -v
```

Its two 50/60-byte specialists each fit with a 10-byte Core under a 100-byte
ceiling. An `A -> B` transition models 70 bytes after eviction, but 120 bytes
if the old and new specialist footprints overlap. This exposes a policy
dependency that a post-swap residency-only calculation would miss.

The receipt binds policy, overhead, sequence and declared model budgets to a
deterministic SHA-256 digest, and reports signed headroom. It does **not**
instrument load/unload operations, verify that eviction happens, quantify
fragmentation, host RAM, CPU↔GPU bandwidth, swap time, or P0 success.
