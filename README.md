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

**Research / architecture seed.** No implementation or training recipe is canonical yet.

See:

- [Project Seed](docs/PROJECT_SEED.md)
- [Architecture V0](docs/ARCHITECTURE_V0.md)
- [Research Plan](docs/RESEARCH_PLAN.md)
- [Terminology](docs/TERMS.md)
- [P0 Residency Feasibility](experiments/MOSAIC_P0_RESIDENCY_V1.md)
- [P1 Cross-Specialist Continuity](experiments/MOSAIC_P1_CONTINUITY_V1.md)
