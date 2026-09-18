# Mosaic Research Plan

Status: **initial plan; revise as evidence arrives**

The first goal is not to train a 20B model. The first goal is to establish whether the architecture's central claims survive small, cheap experiments.

## Phase 0 — Define the claim precisely

Build an executable memory/residency model.

Measure:

- weight residency;
- peak VRAM;
- swap time;
- CPU RAM requirement;
- storage bandwidth;
- KV-cache growth;
- end-to-end latency.

Qualification question:

**Can a multi-specialist runtime remain within a fixed VRAM ceiling while swapping capability modules predictably?**

## Phase 1 — Prove handoff without pretending it is intelligence

Use small existing models or toy modules.

Create tasks where:

1. Specialist A receives information that Specialist B never sees directly.
2. A writes to the Mosaic state interface.
3. A is fully unloaded.
4. B is loaded later.
5. B must correctly act on A's information.

Test at least:

- referent continuity;
- user corrections;
- superseded state;
- decision finality;
- project/current-state tracking;
- structured artifact handoff.

Compare against:

- no persistent state;
- full history;
- ordinary textual retrieval;
- structured shared state.

Measure **user correction burden** and stale-state errors, not only task accuracy.

## Phase 2 — Distinguish routing from composition

Construct tasks that genuinely require more than one specialist.

Examples:

- interpret a user requirement -> reason about design -> implement code -> explain result;
- inspect an image -> infer a structured fact -> use it in a later language task;
- plan -> code -> debug, with each stage using a different specialist.

A router passes if it chooses the right model.

Mosaic passes only if later specialists correctly inherit and use the relevant earlier state.

## Phase 3 — Test specialization strategies

Compare at least three approaches where practical:

1. independently trained specialists;
2. teacher-distilled specialists from a larger common model;
3. shared-base or jointly trained specialists.

Hold evaluation tasks fixed before looking at results.

Measure:

- specialist quality;
- cross-specialist handoff quality;
- duplicated parameter burden;
- active VRAM;
- total parameter pool;
- load/swap latency;
- whole-task accuracy;
- correction burden.

## Phase 4 — Scale the parameter pool

Only after smaller experiments establish that the interface works should Mosaic scale toward the motivating 20B-class capability pool.

The important scaling curve is not only performance versus total parameters.

Track:

**performance / active VRAM / total parameter pool / latency / energy / continuity error**

A larger Mosaic that technically runs but spends most of its time moving weights may not be useful.

## Phase 5 — Whole-versus-part experiments

For a candidate system, test:

- each specialist alone;
- Core/Boot + each specialist;
- sequential specialist composition;
- concurrent composition where hardware permits;
- the monolithic teacher or closest comparable dense baseline.

This directly tests the originating idea that the partitions can operate as parts **and** compose into one whole.

## Evidence discipline

A first success establishes feasibility under one exact configuration.

A repeated success establishes repeatability under closely matched conditions.

A pattern begins when the effect survives independent repetitions, different task families, or changed implementations.

Do not promote a mechanism from "works once" to "architecture verified" without those repetitions.

Negative results are retained. Frozen failed experiments should not be rewritten after outcomes are known.
