# MOSAIC P1 — Cross-Specialist Continuity V1

Status: **predeclared experiment design**

## Question

Can two separately loaded specialists behave as stages of one continuing agent when the first specialist is fully unloaded before the second is loaded?

## Core test

Specialist A receives information or performs work that Specialist B cannot reconstruct from B's final prompt alone.

Sequence:

1. Boot/Core starts with empty test state.
2. Specialist A is loaded.
3. A receives the first-stage task.
4. A writes only through the declared Mosaic handoff interface.
5. A is fully unloaded from accelerator memory.
6. Specialist B is loaded.
7. B receives a final task that requires information created or learned during A's stage.
8. B must answer or act correctly using only the allowed shared state.

The benchmark must prevent the final prompt from trivially containing the answer.

## Conditions

Run the same cases under at least these conditions:

### RESET

A is unloaded and no continuity-bearing state is provided to B.

Purpose: estimate what B can reconstruct or guess without persistence.

### MOSAIC STATE

A is unloaded and B receives only the declared Mosaic shared state.

Purpose: test the architecture being proposed.

### FULL HISTORY

B receives the complete relevant prior interaction/work product.

Purpose: establish an upper-bound control for whether B is capable of solving the task when information is perfectly available.

### ORDINARY RETRIEVAL

B receives a conventional textual/structured retrieval memory using an equivalent source record.

Purpose: distinguish Mosaic-specific handoff value from simply preserving text.

## Task families

At minimum:

- referent preservation;
- correction of an earlier fact;
- supersession/currentness;
- decision finality;
- authority change;
- user preference;
- project status;
- artifact/work-product handoff.

Later stages should add code and multimodal tasks.

## Primary measurements

- final task accuracy;
- user correction burden;
- stale/superseded-answer errors;
- persistent-only correct cases;
- reset-only correct cases;
- Mosaic-state-only correct cases relative to ordinary retrieval;
- handoff payload bytes;
- handoff encode/decode latency;
- specialist swap latency;
- total wall-clock time.

## Minimum evidence pattern

The first successful run establishes feasibility for one exact configuration.

A second independently repeated run establishes repeatability under closely matched conditions.

A third successful run, preferably with a different task family or specialist pair, begins to establish a reproducible pattern.

No single success is promoted to a general architectural claim.

## Pass condition for an exact P1 subject

A candidate Mosaic handoff passes only if:

1. MOSAIC STATE materially outperforms RESET on predeclared continuity cases;
2. stale-state errors do not increase relative to RESET;
3. FULL HISTORY demonstrates that remaining failures are not simply beyond Specialist B's capability;
4. Specialist A is actually absent from accelerator residency before B begins;
5. the handoff payload is bounded and measured;
6. the result reproduces in a second clean run.

Comparison with ORDINARY RETRIEVAL is reported separately. Mosaic does not receive special credit merely for doing what a simpler retrieval system already does as well or better.

## Failure interpretation

If FULL HISTORY succeeds while MOSAIC STATE fails, the likely failure is information preservation, selection, or currentness—not Specialist B's reasoning capability.

If RESET and MOSAIC STATE perform similarly, the state mechanism has not demonstrated useful continuity.

If MOSAIC STATE beats RESET but loses to ordinary retrieval, persistence works but the proposed handoff representation has not justified its extra complexity.

If all conditions fail similarly, the benchmark or specialist capability must be examined before blaming persistence.
