# Mosaic Latent-State Residency Design

Status: RESEARCH DESIGN / NOT A TRAINED MODEL  
Date: 2026-10-02  
Source subject: `thebrazenbeard/mosaic@3a442e94b4f5b8fa365e2206c40405df6961e18d`

## Purpose

Extend Mosaic's constrained-VRAM research with a second residency axis: not only which specialist parameters are resident, but how much shared task/context state remains resident at full resolution.

Mosaic already separates the logical whole from the accelerator-resident set and explicitly budgets weights, KV cache, activations, workspace, and overhead. This design adds experimentally testable multi-resolution state handoff without claiming that ordinary dense models can consume arbitrary latent vectors.

## State classes

- `EXACT_BACKING`: source material or immutable source reference.
- `HIGH_FIDELITY_STATE`: compact structured handoff.
- `SEMANTIC_LATENT_STATE`: learned or deterministic compact representation.
- `ABSTRACT_HANDOFF`: low-cost summary/decision/task state.

Every compact handoff binds provenance and an exact-backing route when exact recovery is required.

## Research question

Can Mosaic reduce total active VRAM/RAM/context burden by jointly optimizing:

`resident specialist parameters + resident task state + KV/context state`

without increasing correction burden or cross-specialist continuity errors beyond a frozen baseline?

## Integration with existing P0/P1 direction

P0 residency accounting gains explicit state-residency terms.

P1 cross-specialist continuity gains additional arms:

- FULL HISTORY;
- STRUCTURED STATE;
- MULTI_RESOLUTION STATE;
- MULTI_RESOLUTION + SELECTIVE REHYDRATION.

A passes-only-on-summary benchmark is insufficient. Cases must include late exact-detail queries whose required fact was not obviously salient at compression time.

## Required metrics

- peak accelerator memory where measurable;
- resident model bytes;
- active context/KV proxy or measured KV bytes;
- shared-state bytes;
- swap/load latency;
- rehydration latency;
- exact-detail recovery;
- stale-state errors;
- user correction burden;
- whole-task accuracy.

## Boundary

This repo owns the residency/composition experiment. It should not become the canonical Vera runtime implementation.

The model-independent exactness invariant is:

`LOSSY_HANDOFF != EXACT_EVIDENCE`

A specialist may operate from compact state, but exact reconstruction claims require verified backing state or an explicitly qualified learned decoder under a separately defined fidelity ceiling.

## Research lineage

Relevant neighboring work:

- BLT dynamic latent patches: https://arxiv.org/abs/2412.09871
- ICAE compressed memory slots: https://arxiv.org/abs/2307.06945
- DMC adaptive KV compression: https://arxiv.org/abs/2403.09636
- Latent Context Compilation: https://arxiv.org/abs/2602.21221

## First implementation slice

Do not train a model first. Extend the deterministic residency/handoff harness so a test case can carry an exact backing payload, a compact handoff representation, a byte/token cost, and an on-demand rehydration event. Compare it against existing RESET/FULL HISTORY/ordinary retrieval conditions under the same frozen tasks.
