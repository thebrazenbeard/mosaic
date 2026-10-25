> **License:** Source-visible, not open source. Original material is proprietary. Commercial use, redistribution, hosted-service use, and commercial derivative products require written permission. See [LICENSE](LICENSE) and [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md). Separately identified third-party components retain their own licenses.

# Mosaic

**An AI architecture whose total capability can be larger than the model currently loaded into VRAM.**

Mosaic is an experimental modular-agent architecture for composing a persistent coordination layer and swappable specialist model partitions into one logical agent under constrained hardware.

The motivating target is deliberately aggressive: explore whether a system with the capability pool of a much larger model can operate on a roughly **4 GB VRAM-class GPU** by keeping only the currently needed specialist capacity resident.

## Project seed

The original concept is:

- `VERA-MODEL20B`: the logical whole-agent system, initially imagined as roughly a 20B-class model / ~11 GB weight package.
- `VERA-MODEL20Ba`: specialist partition, e.g. coding.
- `VERA-MODEL20Bb`: specialist partition, e.g. conversation.
- `VERA-MODEL20Bc`: specialist partition, e.g. reasoning/coding or another learned specialty.
- `VERA-MODEL20Bd`: specialist partition, e.g. image/visual work.
- `VERA-MODEL20B_boot`: a comparatively small boot/core architecture that allows the partitions to operate individually while preserving their composition as one logical agent.

The exact parameter counts, partition boundaries, training method, and resident-memory budget are **not fixed yet**. The point of the repository is to find out which version of this idea actually works.

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
- state handoff / continuity protocol mechanics;
- the frozen P1 comparison cases across RESET, MOSAIC STATE, FULL HISTORY, and ORDINARY RETRIEVAL conditions.

Those harnesses make parts of the research protocol executable; they are not a trained Mosaic model, do not demonstrate real specialist unloading/swapping on an accelerator, and do not establish a Mosaic P1 pass. The model partition architecture, training recipe, and empirical capability claims remain research subjects.

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
