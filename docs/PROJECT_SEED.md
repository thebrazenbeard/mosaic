# Mosaic Project Seed

## Origin

Mosaic began from a hardware-constrained model-partitioning idea:

> What if a roughly 20B-class agent, around an 11 GB model package, could be represented by several roughly 3 GB specialist models plus a roughly 1 GB boot architecture that allows the specialists to operate as one logical agent or independently?

The initial sketch was:

| Logical component | Initial role sketch | Initial size sketch |
| --- | --- | ---: |
| VERA-MODEL20B | whole logical agent | ~20B / ~11 GB package |
| VERA-MODEL20Ba | coding specialist | ~3 GB |
| VERA-MODEL20Bb | conversation specialist | ~3 GB |
| VERA-MODEL20Bc | second coding/reasoning specialist | ~3 GB |
| VERA-MODEL20Bd | image-rendering / visual specialist | ~3 GB |
| VERA-MODEL20B_boot | architecture tying the partitions together | ~1 GB |

These numbers preserve the originating idea; they are **not yet an implementation contract**.

## Why the idea exists

A conventional model is commonly treated as if the total parameter set must be available to the accelerator during use. Mosaic asks a different question:

**Can total learned capability be much larger than the active parameter residency at any one moment?**

The hardware target that motivated the idea is a roughly 4 GB VRAM-class GPU.

The desired behavior is not merely "four unrelated small models behind a router." The desired behavior is one logical agent whose specialists can be swapped according to the work being performed while continuity survives the swap.

## The overlap realization

A specialist cannot be assumed to contain only its specialty.

A coding specialist still needs language, instruction following, semantic interpretation, context handling, and shared concepts. A conversation specialist still needs reasoning. A visual specialist may need language and world-model interfaces. Training complete specialists independently therefore risks duplicating a large amount of general capability.

That creates a central Mosaic design problem:

**Which capabilities belong to shared substrate, and which capabilities can be moved into specialist partitions?**

One possible answer is a shared resident core plus narrower specialists. Another is additional partitions. Another is teacher-derived shared representations. Mosaic does not preselect the winner.

## Training-order question

Three broad strategies are intentionally left open:

### Large teacher first

Train or begin with a capable large model, then use it to distill or train smaller specialists and a composition layer.

Potential benefit: the specialists inherit a common teacher and may more easily share semantics.

Potential risk: a dense teacher may not decompose cleanly; distillation can lose interactions that depended on distributed parameters.

### Specialists first

Train specialists independently, then train a coordinator/interface to compose them.

Potential benefit: each specialist can become very strong in its domain.

Potential risk: internal representations may be incompatible and shared capabilities may be redundantly learned.

### Joint / staged co-training

Train shared substrate, specialists, and routing/composition in a coordinated curriculum.

Potential benefit: the architecture can learn to specialize while preserving interoperability.

Potential risk: training becomes substantially more complex and expensive.

## Non-negotiable research distinction

Mosaic must distinguish:

- **total parameter pool** from **active parameter residency**;
- **one logical agent** from **one simultaneously resident neural network**;
- **specialist composition** from **ordinary request routing**;
- **persistent continuity** from **specialist-local temporary state**;
- **a successful prototype** from **equivalence to a monolithic 20B model**.

The project begins by testing those distinctions rather than assuming they are already solved.
