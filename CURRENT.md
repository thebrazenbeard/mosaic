# CURRENT — Mosaic

Last updated: 2026-09-18

## Status

Mosaic is an active **BT2 project assignment** in foundation/research stage.

No Mosaic implementation, trained model, training recipe, or claim of equivalence to a monolithic 20B model is currently canonical.

## Originating concept

The project began with this semantic design:

- one logical `VERA-MODEL20B`-class agent;
- several smaller specialist partitions, initially imagined around ~3 GB each;
- a smaller `VERA-MODEL20B_boot` / Core responsible for allowing the partitions to operate independently while composing as one agent;
- target hardware motivation: approximately 4 GB VRAM.

The important research question is not merely routing requests to different models. It is whether specialist partitions can be swapped while preserving enough shared continuity that the system still behaves as one logical agent.

## Active work

Draft PR #1 on `project/foundation-v1` currently contains:

- preserved project seed;
- exploratory Architecture V0;
- research plan;
- common terminology;
- P0 predeclared residency-feasibility experiment;
- P1 predeclared cross-specialist-continuity experiment.

Exact branch head when this file was written:

`f13a1740a6c3e0040af35a47582babd9a44e20ee`

A BT2 worker must fresh-check the PR/head before carrying this state forward.

## Immediate next Mosaic frontier

Do not jump directly into 20B training.

The intended order is:

1. verify active-residency/swap feasibility under a constrained VRAM ceiling (P0);
2. verify cross-specialist state continuity after full specialist unload/reload (P1);
3. compare ordinary retrieval, structured state, and learned/shared state;
4. only then choose a training/decomposition strategy for larger specialists.

## Current boundaries

- Draft PR #1 remains unmerged.
- Exploratory size figures are target sketches, not implementation facts.
- HC-Brain is not Mosaic's template.
- No protected effect is implied by BT2 assignment status.
