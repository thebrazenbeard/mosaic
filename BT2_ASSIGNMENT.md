# BT2 Assignment

Mosaic is a **Build Team Two (BT2) project assignment**.

## Authority and coordination

- Owner / protected-effect authority: Patrick.
- BT2 is responsible for architecture, implementation, experimentation, verification, and review work on Mosaic under the current BT2 Project Instructions.
- Non-PR inter-agent coordination belongs on `thebrazenbeard/chat-communication-bus`.
- Pull requests remain the review/integration surface for Mosaic source changes.
- Do not merge, deploy, publish trained artifacts, spend paid compute, or make other protected effects without current owner authority.

## Project purpose and currentness

Mosaic tests whether a persistent neural Boot/Core and swappable specialized learned capacity can behave as one logical agent while keeping active accelerator memory below its total capability pool. Around 20B logical parameters and 4 GB VRAM are motivating hypotheses, not established implementation limits.

Read [`CURRENT.md`](CURRENT.md), then exact live `main` and open PR heads. In particular, inspect merged [security containment PR #13](https://github.com/thebrazenbeard/mosaic/pull/13) before attempting to promote historical model research PRs #10–#12. Those draft heads were based on an unsafe editor-startup source tree; stale branch CI or an earlier source-only result cannot waive containment review.

Canonical code provides analytical residency accounting, deterministic handoff comparison, and guarded thinking-Core/loopback inference source interfaces (merged PRs #16, #18, #19, #20). These interfaces can call an already-serving local model but do not constitute a qualified installed Mosaic neural Core. Actual specialist checkpoints, authenticated model identity, accelerator swap measurements, and cross-specialist continuity are **not qualified**. A source adaptation of Mosaic handoff in `vera-mono` is not a live Mosaic service or evidence of identity continuity.

## Worker orientation

1. Fresh-read `README.md`, `CURRENT.md`, default-branch head, open PRs and dependency/stacking relationships.
2. Read `runtime/`, `tests/`, and predeclared experiments under `experiments/` before asserting capability.
3. Check security provenance and any source/startup loaders before running workspace tasks, editor commands, or dependency scripts.
4. Keep documented architecture, test results, real model probes, hardware measurements, and independent qualification in separate evidence classes.
5. Preserve exact head SHA, test commands, hostile findings and unresolved review disagreements in PR review/BT2 coordination.

The older `project/foundation-v1` / PR #1 orientation was a historical foundation snapshot; do not mistake its September 2026 head for current source authority.
