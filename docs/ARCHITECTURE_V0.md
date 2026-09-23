# Mosaic Architecture V0

Status: **exploratory, not frozen**

## 1. Logical whole versus resident set

Mosaic treats the logical agent and the currently resident compute as separate things.

The **logical whole** is the complete Mosaic capability pool plus the state required to maintain continuity.

The **resident set** is the subset of model parameters and runtime state currently loaded onto the accelerator.

A small-hardware runtime may load specialists sequentially. A larger machine may keep several or all specialists resident concurrently. The logical identity of the system should not depend on that deployment topology.

## 2. Provisional component model

### Boot / Core

The originating design calls this `VERA-MODEL20B_boot`.

Its exact form is open. At minimum it may need to provide:

- startup and model discovery;
- task decomposition and specialist selection;
- continuity across specialist swaps;
- persistent-memory access;
- current working-state handoff;
- capability and resource accounting;
- provenance of which specialist produced which state;
- conflict/arbitration behavior when specialists disagree;
- final composition of work when more than one specialist participates.

A key research question is how much learned capability belongs here. A tiny deterministic router and a learned general-purpose core are very different architectures.

### Specialist partitions

Specialists are separately loadable capability packages.

Initial examples:

- code / software engineering;
- conversation / language interaction;
- deeper reasoning or an additional technical specialty;
- visual or image-generation capability.

The list is provisional. A specialist does not have to use the same neural architecture as every other specialist if the common interface is sufficiently well specified.

### Shared state

Specialist handoff requires more than natural-language summaries if Mosaic is to behave like one integrated agent.

Candidate state channels include:

- durable semantic memory;
- current task graph;
- structured facts and provenance;
- unresolved questions;
- goals and constraints;
- tool/environment observations;
- compact learned latent state;
- specialist-produced artifacts.

Mosaic should test textual, structured, and learned-state interfaces rather than assume one representation is sufficient.

## 3. Execution modes

### Single-specialist mode

Boot/Core selects one specialist, loads it, performs the bounded work, records state, and unloads it.

### Sequential-composition mode

A task passes through multiple specialists one after another while Boot/Core preserves and transfers working state.

Example:

```
request
  -> interpret / plan
  -> load reasoning specialist
  -> preserve state
  -> unload reasoning
  -> load coding specialist
  -> preserve state
  -> unload coding
  -> compose response
```

### Concurrent mode

On hardware with sufficient memory, multiple specialists may remain resident and cooperate without swap latency.

The same logical Mosaic should ideally support both sequential and concurrent execution.

## 4. Memory budget model

The design target is not "4 GB of weights."

The actual device budget must include:

```
resident model weights
+ KV cache
+ activations
+ temporary workspaces
+ framework / kernel overhead
+ multimodal buffers where applicable
<= available VRAM
```

This means a nominal 3 GB specialist can still be too large for a 4 GB card. Measurement must use real peak residency, not just file size.

## 5. Composition is the hard problem

Routing is not enough.

A system that sends coding requests to one model and chat requests to another is a model gateway, not necessarily one agent.

Mosaic's stronger claim requires successful preservation of:

- referents;
- corrections;
- currentness;
- goals;
- decisions;
- preferences;
- intermediate reasoning products;
- relevant learned state;
- behavioral continuity.

If specialist swaps repeatedly force the user to re-establish context, the architecture has failed its central purpose even if each specialist is individually strong.

## 6. What must be experimentally demonstrated

A Mosaic prototype should not claim whole-agent composition until evidence shows that:

1. specialists can be swapped under the target memory ceiling;
2. state survives those swaps;
3. later specialists correctly use earlier specialist state;
4. stale state is superseded rather than merely accumulated;
5. task outcomes improve over a comparable small single model;
6. the result is reproducible across more than one task family;
7. gains are not explained only by giving Mosaic more tokens, retrieval, or wall-clock compute.

## 7. Explicitly unresolved

- Whether the Boot/Core is neural, deterministic, hybrid, or itself partitioned.
- Whether specialists share frozen base layers, adapters, experts, or no weights at all.
- Whether specialist state exchange should be latent, symbolic, textual, or mixed.
- Whether the project should begin from an existing dense teacher or train a modular model directly.
- Whether image generation belongs inside the same parameter family or behind a common Mosaic interface.
- Whether the whole system can approach a dense 20B model's generality under the desired memory ceiling.
