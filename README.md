> **License:** Source-visible, not open source. Original material is proprietary. Commercial use, redistribution, hosted-service use, and commercial derivative products require written permission. See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). Separately identified third-party components retain their own licenses.

# Mosaic

**A research toolkit for modular AI reasoning and bounded model residency.**

Mosaic combines a protocol-level design for a continuity-bearing Core with executable Python harnesses for residency accounting, specialist challenges, evidence boundaries, and comparison baselines. It explores modular-agent operation under constrained compute; it is not a trained or production-qualified modular model.

The research objective is to test whether a logical system can use more specialist capability than fits in accelerator memory simultaneously, by loading only the capacity needed for a task. The resulting memory, latency, continuity, and accuracy tradeoffs must be measured rather than assumed.

## Reusable architecture pattern (research status)

Mosaic explores a logical agent assembled from a **continuity-bearing Core** and **swappable specialist partitions**. For example, a host could distinguish language, code, planning, and visual specialists while keeping task state and evidence provenance in a separately governed coordination layer. These are **candidate roles**, not shipped model weights or confirmed interchangeable transformer shards.

Compute residency is a measurable resource budget rather than a fixed workstation configuration. A small-memory GPU is one possible research target, with the specialist footprint, loading latency, and accuracy compared against full-history and retrieval baselines. No particular VRAM size or local host is a prerequisite for inspecting the repository's source-only harnesses.

The original named model-partition proposal and historical parameter estimates are preserved as research context in [Project Seed](docs/PROJECT_SEED.md). Their names and sizes are hypotheses, not advertised product artifacts.

## Central questions

1. Can a single logical agent be composed from specialist parameter partitions without duplicating a full general model inside every specialist?
2. Can identity, continuity, memory, task state, and routing survive specialist swaps?
3. Can a specialist be loaded only when needed while another is evicted, keeping active VRAM bounded?
4. Can the same logical system scale upward—sequential specialists on small hardware, multiple concurrent specialists on larger hardware?
5. Is it better to train a large teacher first and distill partitions from it, train specialists first and integrate them, or co-train a shared architecture?
6. What information must be shared across every specialist, and what can genuinely be specialized?
7. When does the cost of loading/switching experts erase the memory advantage?

## What Mosaic is not

Mosaic is not currently a claim that an ordinary dense 20B transformer can be cut into four arbitrary files and retain identical behavior. Dense transformer representations are distributed, and that assumption must be tested rather than smuggled into the design.

It is also not HC-Brain. Concepts may later be borrowed where useful, but Mosaic begins as an independent architecture centered on **parameter residency, specialization, composition, and continuity under constrained compute**.

## Current status

**Research architecture plus bounded executable research harnesses.**

`main` now contains deterministic Python harnesses under `runtime/` with tests under `tests/` for:

- analytical residency accounting;
- a safety-gated, **not-yet-live-qualified** GPU-router load/unload probe (`experiments/p0_router_probe.py`);
- state handoff / continuity protocol mechanics;
- the frozen P1 comparison cases across RESET, MOSAIC STATE, FULL HISTORY, and ORDINARY RETRIEVAL conditions;
- an offline three-arm challenge scorer (`runtime/challenge_eval.py`) whose two-call V1 rejects uncounted learned-critic costs.
- a separate [V2 learned-critic cost-control protocol](experiments/MOSAIC_LEARNED_CRITIC_COST_CONTROL_V2.md) and scorer (`runtime/challenge_eval_v2.py`) for caller-reported, three-call matched comparisons; neither authenticates model identity or measured compute.

Those harnesses make parts of the research protocol executable; they are not a trained Mosaic model, do not demonstrate real specialist unloading/swapping on an accelerator, and do not establish a Mosaic P1 pass. The model partition architecture, training recipe, and empirical capability claims remain research subjects.

### Neural handoff experiments (2026-10-26)

[Recorded latent handoff research](experiments/MOSAIC_LATENT_VN1_RESULT_20261026.md) documents a narrow hidden-tensor exchange between separately loaded pretrained SmolLM2 models and a learned handoff between small networks. The toy experiment beat a truncated eight-record proxy but **tied its full-history control**; its nominal latent was larger than a packed 16-bit record. A separate [pretrained bridge pilot](experiments/MOSAIC_PRETRAINED_BRIDGE_PILOT1_20261026.md) scored **64/128**, tied with full-history and reset controls. These observations are specific to the reported experimental setup, not performance guarantees for another host.

**Shared learned reasoning across pretrained specialists and superiority over genuine retrieval remain HOLD.** The [BM25 lexical retrieval control](experiments/p1_bm25_retrieval_control.py) adds a document-ranking comparison with abstention and byte-budget accounting; it does not itself establish a P1 pass.

### Guarded thinking-Core source on `main`

After the editor-startup security containment in merged PR #13, merged PR #16 added guarded reasoning-Core source without importing older branches' startup tasks or disguised assets. Subsequent PRs #18, #19, and #20 hardened reviewer identities, nested JSON error handling, and verifier-assertion boundaries. `runtime/thinking_core.py` governs Core deliberation and specialist objections; `runtime/neural_core.py` defines strict model-produced reasoning and reconsideration; `runtime/local_neural_backend.py` is a bounded loopback-only adapter to an already-serving model. The Core remains a reasoning participant, not merely a router. Neither a model response nor specialist objection grants protected-effect authority.

These are source interfaces and regression tests on `main`, not an installed Mosaic model or a measured capability improvement. Injected verifier assertions remain **unattested**; `claim_verified` cannot become true without a future trusted verifier mechanism. The earlier real-Core/scripted-critic comparison was 6/10 versus 6/10, with no net accuracy gain. Independent learned specialists, authenticated model weights, real VRAM swaps, and neutral-second-pass held-out validation remain open. Do not merge the old unsanitized PRs #10–#12 as-is.

See:

- [Rezon Unverified Advisory Bridge V1](docs/MOSAIC_REZON_ADVISORY_V1.md) — exact-task, read-only Rezon collective first-answer adapter; synthetic real-store contract test, no trusted evidence or independent learned-specialist qualification.
- [Project Seed](docs/PROJECT_SEED.md)
- [Architecture V0](docs/ARCHITECTURE_V0.md)
- [Research Plan](docs/RESEARCH_PLAN.md)
- [Terminology](docs/TERMS.md)
- [P0 Residency Feasibility](experiments/MOSAIC_P0_RESIDENCY_V1.md)
- [P1 Cross-Specialist Continuity](experiments/MOSAIC_P1_CONTINUITY_V1.md)
