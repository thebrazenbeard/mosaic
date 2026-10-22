# Mosaic thinking Core and specialist disagreements — candidate V1

**Status:** executable **protocol** with a model-call boundary and fake-model tests; **not** a deployed thinking Core, trained specialist, neural qualification, independent verifier installation, or Mosaic P0/P1 PASS. This proposal supplements, and does not replace, the experimental Architecture V0. Source/merge/installation/effect claims must remain separate.

## Corrected architectural interpretation

Mosaic's Boot/Core is **intended to do actual thinking and reasoning**. It is not merely a deterministic router or a memory database. Its coherent reasoning should remain present when specialist modules are unavailable. Specialists supply additional learned capabilities, candidate solutions, counterarguments, and evidence; they do not become a new owner of the system's identity or final reasoning.

The Core can be a modest learned neural model backed by deterministic identity, authority, state, memory and provenance software. A *separate* verification subsystem checks what it can, rather than letting a small Core arbitrarily override larger experts. A thinking Core and a software governance substrate are compatible.

The Core's smaller capacity may still limit judgment. This protocol **does not claim that a 1 GB model can reliably evaluate a 3 GB specialist**. It makes uncertainty explicit instead of hiding that asymmetry.

## Deliberation protocol

1. **THINK:** the learned Core returns an initial, explicit position on a bounded proposition with its reasoning. The model-facing interface is `runtime.neural_core.NeuralThinkingCore`, with an injected inference callable. This component cannot self-award verification.
2. **CHALLENGE:** one or more specialist models independently review that position. Their return values are suggestions/objections, never authority or a verified fact.
3. **VERIFY:** an independent, externally implemented verifier may return a proposition-bound verdict and evidence reference. A raw specialist opinion does not count. An arbitrary string labeled `evidence_ref` is not proof that a real trusted verifier exists; integration must authenticate/verifiably bind that source.
4. **RECONSIDER:** when a specialist dissents or evidence contradicts the first position, the Core itself must generate a second reasoned position. It must acknowledge each dissenting specialist; ignoring material objections blocks the result.
5. **CLASSIFY:** the deterministic runtime applies the result ceiling. Discretionary low-risk disagreement may produce a **CONTESTED_ADVISORY** answer, never a verified claim. A consequential unresolved disagreement becomes **HOLD_UNVERIFIED_DISPUTE**. If independent evidence contradicts the final Core position, return **BLOCKED_BY_EVIDENCE**, even if the Core remains confident.
6. **AUTHORIZE:** for any protected effect, even evidence-supported consensus produces **PROPOSED_EFFECT_REQUIRES_AUTHORIZATION**. Neither a Core nor a specialist nor a self-asserted verifier can execute it through this protocol. The user/owner's existing authorization machinery remains a separate prerequisite.
7. **RECORD:** a deterministic content SHA-256 binds the task, both Core positions, objections, provided verifier metadata, disposition, and explicit no-execution/claim ceiling. A SHA-based receipt is **not** a signature and does not authenticate a model or verifier.

For unresolved controversial or non-testable claims, there is no algorithm that can simply choose the more intelligent model. The Core may still choose an interpretation for a low-risk advisory answer, disclosing conflict and limits; consequential actions are held. Verified contradictory evidence supersedes unsupported model preferences, not because the verifier is a smarter brain but because external evidence has a separately established basis.

## What this implementation does and doesn't do

`runtime.thinking_core` is a strict in-memory protocol/arbitrator. `runtime.neural_core` turns a *supplied* generation backend into a model-driven THINK and RECONSIDER boundary with strict JSON parsing, distinct system/user messages, and untrusted specialist objections transmitted as data. It does **not** instantiate a model, load a LoRA adapter, switch specialists on the GPU, persist state across process restart, authenticate provenance, or prove that the Core performs useful reasoning.

The bounded tests use fake Core responses and synthetic verifier fixtures to check the *policy*, not the correctness of the claims. A real prototype needs a qualified resident Core inference backend, actual independent specialist adapters/models, permission-scoped verifier connectors, an event-sourced durable task state layer, and hardware P0/P1 observations.

## Experiments required before adopting the design

The first adversarial study should compare (A) Core alone, (B) Core plus specialist opinions, (C) Core plus independently checked evidence, and (D) a comparable single generalist with equal resource and token budgets. Include misleading confident specialists, failures the Core can't comprehend, unreliable verifiers, contradictory advice, stale state, protected effects, and at least one genuinely cross-specialist task. Measure final task accuracy, unresolved/false-certification rates, latency, extra tokens, memory, and user correction burden.

**Success means the Core's thinking is measurably useful while remaining correctable, not that it can declare itself sovereign or always win disagreements.**
