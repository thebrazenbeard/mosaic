# BT2 Assignment

Mosaic is a **Build Team Two (BT2) project assignment**.

## Authority and coordination

- Owner / protected-effect authority: Patrick.
- BT2 is responsible for architecture, implementation, experimentation, verification, and review work on Mosaic under the current BT2 Project Instructions.
- Non-PR inter-agent coordination belongs on `thebrazenbeard/chat-communication-bus`.
- Pull requests remain the review/integration surface for Mosaic source changes.
- Do not merge, deploy, publish trained artifacts, spend paid compute, or make other protected effects without current owner authority.

## Project purpose

Mosaic explores whether one logical AI agent can have a total learned capability pool substantially larger than the model parameters currently resident in accelerator memory.

The originating target is a roughly 20B-class logical agent operating on constrained hardware by composing a persistent Boot/Core with swappable specialist model partitions.

## Current source state

- Repository: `thebrazenbeard/mosaic`
- Canonical branch: `main`
- Active foundation branch: `project/foundation-v1`
- Active review: Draft PR #1
- Foundation branch head at this record: `f13a1740a6c3e0040af35a47582babd9a44e20ee`

The exploratory architecture and experiment designs are intentionally not canonical merely because they exist on the draft branch.

## BT2 control-chat orientation rule

A fresh BT2 control chat should read, in order:

1. `README.md`
2. `BT2_ASSIGNMENT.md`
3. `CURRENT.md`
4. Draft PR #1 and its exact current head
5. On that branch: `docs/PROJECT_SEED.md`, `docs/ARCHITECTURE_V0.md`, `docs/RESEARCH_PLAN.md`, and `experiments/`

Do not infer current project state from conversation memory when these surfaces are available.
