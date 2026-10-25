# CURRENT — Mosaic

Audit snapshot: 2026-10-25. Recheck live `main`, open pull requests, and their exact heads before acting; this page is not a runtime health receipt.

## Implemented and unqualified

Mosaic remains an experimental **modular-model architecture**, not a trained 20B-equivalent agent. Canonical `main` contains bounded deterministic research harnesses **and guarded source interfaces** for thinking-Core experiments:

- `runtime/residency.py`: declared-byte residency, sequential swaps, and analytical transition headroom, **not** measured accelerator load/unload.
- `experiments/p0_router_probe.py`: explicitly gated ten-cycle A→B→A load/infer/unload probe with bounded local router traffic and total NVIDIA-memory sampling. It has **not** established a live hardware result or P0 pass; llama.cpp router worker restarts and incomplete allocation telemetry remain limitations.
- `runtime/handoff.py`: revision-aware, UTF-8-checked state handoff and RESET / MOSAIC STATE / FULL HISTORY / ORDINARY RETRIEVAL protocol comparisons, **not** demonstrated identity continuity or learned specialist transfer. The queried key's latest revision is selected independently of unrelated keys.
- `runtime/thinking_core.py`, `runtime/neural_core.py`, `runtime/local_neural_backend.py`: source-level Core deliberation, challenge handling and bounded loopback neural inference. Injected verifier assertions are **unattested** and cannot set `claim_verified=True`; no protected-effect execution authority is granted.
- `runtime/rezon_advisory.py`: exact-task, digest-bound Rezon collective first answers as **unverified specialist advice**; SQLite-store contract tests use synthetic model answers.
- `runtime/challenge_eval.py`: offline, declared-cost paired scorer for Core-only, neutral second-pass and challenged arms. Its two-call V1 rejects unaccounted learned-critic calls and zero-call successes; synthetic labels are not a model-quality result.
- `tests/`: source-level harness assertions, **not** neural, GPU, or cross-host qualification.

No real specialist checkpoint family, demonstrated parameter swap, GPU residency measurement, repeated clean-run continuity, or Mosaic P0/P1 pass is established by these modules. The original ~20B logical pool and ~4 GB VRAM goal are research targets.

`vera-mono` has already adapted Mosaic's state-handoff *source* into `packages/vera_core/src/vera_core/state_handoff.py`, with a donor record at `provenance/donors/mosaic_state_handoff.json`. That adaptation does not grant identity continuity, authority, model training, or an active Mosaic runtime integration.

## Security and open research branches

Historical `main@276282a1d4a34588f34a4d18ab3da76a84b6e48b` contained a concealed VS Code folder-open execution path and a disguised `.lkf` asset. Security containment PR [#13](https://github.com/thebrazenbeard/mosaic/pull/13) removed those source assets and added a regression guard. Merged PRs [#16](https://github.com/thebrazenbeard/mosaic/pull/16), [#18](https://github.com/thebrazenbeard/mosaic/pull/18), [#19](https://github.com/thebrazenbeard/mosaic/pull/19), and [#20](https://github.com/thebrazenbeard/mosaic/pull/20) introduced and hardened bounded thinking-Core interfaces. PRs [#22](https://github.com/thebrazenbeard/mosaic/pull/22), [#25](https://github.com/thebrazenbeard/mosaic/pull/25), [#24](https://github.com/thebrazenbeard/mosaic/pull/24), and [#26](https://github.com/thebrazenbeard/mosaic/pull/26) subsequently added transition accounting, handoff/bridge/P0 instrumentation, P1 key integrity and cost-scoring checks. Exact `main@748532b044ec88fc1c5ef080f90d2bf5e85a4631` passed 125 source tests with local Rezon test dependency and loopback fixture access; this is **not** live Mosaic neural/GPU qualification.

**Treat the sanitized main after #13 as a prerequisite to any model-branch promotion.** Draft research PRs [#10](https://github.com/thebrazenbeard/mosaic/pull/10), [#11](https://github.com/thebrazenbeard/mosaic/pull/11), and [#12](https://github.com/thebrazenbeard/mosaic/pull/12) were authored against older source and may reintroduce the concealed asset if merged unchanged. Rebase or reconstruct legitimate code on a verified sanitized head, then rerun the containment guard and tests before review.

The neural-Core challenge reported in PR #12 used a real local model with scripted objections, **not independent learned specialists**, and scored 6/10 before versus 6/10 after challenge: no demonstrated accuracy improvement or neural qualification.

## Next executable gates

1. **Source trust:** keep the containment guard green and rebuild any retained research work on sanitized source.
2. **P0 hardware:** run the guarded probe only on an explicitly dedicated router/device; measure weight/KV/activation/workspace residency, actual eviction, worker lifecycle, swap latency, and repeated cycles under a fixed budget. Total `nvidia-smi` usage is insufficient alone.
3. **P1 continuity:** run genuinely different specialists after full unload/reload, with predeclared identity/authority boundaries and RESET / FULL HISTORY / RETRIEVAL controls.
4. **Neural benefit:** independently trained or identified models, unseen holdouts, honest three-call learned-critic accounting and cost-matched controls, and raw receipt preservation before any superiority claim.

No deployment, training promotion, privileged effects, or merge is authorized by this status page. Patrick retains protected-effect authority under `BT2_ASSIGNMENT.md`.
