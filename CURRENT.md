# CURRENT — Mosaic

Audit snapshot: 2026-10-25. Recheck live `main`, open pull requests, and their exact heads before acting; this page is not a runtime health receipt.

## Implemented and unqualified

Mosaic remains an experimental **modular-model architecture**, not a trained 20B-equivalent agent. Canonical `main` contains bounded deterministic research harnesses **and guarded source interfaces** for thinking-Core experiments:

- `runtime/residency.py`: declared-byte residency and sequential-swap feasibility accounting, **not** measured accelerator load/unload.
- `runtime/handoff.py`: revision-aware state handoff and RESET / MOSAIC STATE / FULL HISTORY / ORDINARY RETRIEVAL protocol comparisons, **not** demonstrated identity continuity or learned specialist transfer.
- `runtime/thinking_core.py`, `runtime/neural_core.py`, `runtime/local_neural_backend.py`: source-level Core deliberation, challenge handling and bounded loopback neural inference. Injected verifier assertions are **unattested** and cannot set `claim_verified=True`; no protected-effect execution authority is granted.
- `tests/`: source-level harness assertions, **not** neural, GPU, or cross-host qualification.

No real specialist checkpoint family, demonstrated parameter swap, GPU residency measurement, repeated clean-run continuity, or Mosaic P0/P1 pass is established by these modules. The original ~20B logical pool and ~4 GB VRAM goal are research targets.

`vera-mono` has already adapted Mosaic's state-handoff *source* into `packages/vera_core/src/vera_core/state_handoff.py`, with a donor record at `provenance/donors/mosaic_state_handoff.json`. That adaptation does not grant identity continuity, authority, model training, or an active Mosaic runtime integration.

## Security and open research branches

Historical `main@276282a1d4a34588f34a4d18ab3da76a84b6e48b` contained a concealed VS Code folder-open execution path and a disguised `.lkf` asset. Security containment PR [#13](https://github.com/thebrazenbeard/mosaic/pull/13) removed those source assets and added a regression guard. Merged PRs [#16](https://github.com/thebrazenbeard/mosaic/pull/16), [#18](https://github.com/thebrazenbeard/mosaic/pull/18), [#19](https://github.com/thebrazenbeard/mosaic/pull/19), and [#20](https://github.com/thebrazenbeard/mosaic/pull/20) subsequently introduced and hardened bounded thinking-Core interfaces. Exact `main@21623bbc8e3d7fa49c45f285e0c9afb1dc41730b` passed 70 source tests, including the editor-startup regression; this is not runtime neural qualification.

**Treat the sanitized main after #13 as a prerequisite to any model-branch promotion.** Draft research PRs [#10](https://github.com/thebrazenbeard/mosaic/pull/10), [#11](https://github.com/thebrazenbeard/mosaic/pull/11), and [#12](https://github.com/thebrazenbeard/mosaic/pull/12) were authored against older source and may reintroduce the concealed asset if merged unchanged. Rebase or reconstruct legitimate code on a verified sanitized head, then rerun the containment guard and tests before review.

The neural-Core challenge reported in PR #12 used a real local model with scripted objections, **not independent learned specialists**, and scored 6/10 before versus 6/10 after challenge: no demonstrated accuracy improvement or neural qualification.

## Next executable gates

1. **Source trust:** keep the containment guard green and rebuild any retained research work on sanitized source.
2. **P0 hardware:** predeclare a real accelerator protocol and measure weight/KV/activation/workspace residency, actual eviction, swap latency, and repeated cycles under a fixed budget.
3. **P1 continuity:** run genuinely different specialists after full unload/reload, with predeclared identity/authority boundaries and RESET / FULL HISTORY / RETRIEVAL controls.
4. **Neural benefit:** independently trained or identified models, unseen holdouts, cost-matched neutral second-pass controls, and raw receipt preservation before any superiority claim.

No deployment, training promotion, privileged effects, or merge is authorized by this status page. Patrick retains protected-effect authority under `BT2_ASSIGNMENT.md`.
