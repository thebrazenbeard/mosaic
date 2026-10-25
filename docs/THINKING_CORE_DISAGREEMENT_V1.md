# Mosaic thinking Core and specialist disagreements — candidate V1

**Status:** executable **protocol** with a model-call boundary and fake-model tests; **not** a deployed thinking Core, trained specialist, neural qualification, independent verifier installation, or Mosaic P0/P1 PASS. This proposal supplements, and does not replace, the experimental Architecture V0. Source/merge/installation/effect claims must remain separate.

## Corrected architectural interpretation

Mosaic's Boot/Core is **intended to do actual thinking and reasoning**. It is not merely a deterministic router or a memory database. Its coherent reasoning should remain present when specialist modules are unavailable. Specialists supply additional learned capabilities, candidate solutions, counterarguments, and evidence; they do not become a new owner of the system's identity or final reasoning.

The Core can be a modest learned neural model backed by deterministic identity, authority, state, memory and provenance software. A *separate* verification subsystem checks what it can, rather than letting a small Core arbitrarily override larger experts. A thinking Core and a software governance substrate are compatible.

The Core's smaller capacity may still limit judgment. This protocol **does not claim that a 1 GB model can reliably evaluate a 3 GB specialist**. It makes uncertainty explicit instead of hiding that asymmetry.

## Deliberation protocol

1. **THINK:** the learned Core returns an initial, explicit position on a bounded proposition with its reasoning. The model-facing interface is `runtime.neural_core.NeuralThinkingCore`, with an injected inference callable. This component cannot self-award verification.
2. **CHALLENGE:** one or more specialist models independently review that position. Their return values are suggestions/objections, never authority or a verified fact.
3. **EXAMINE:** an externally injected verifier may return a task-bound verdict and evidence reference. The exact task digest rejects accidental cross-case replay, but the caller controls the verifier and can forge the digest. Neither `verifier_id` nor `evidence_ref` is an attestation; this source module has no trusted-verifier registry.
4. **RECONSIDER:** when a specialist dissents or evidence contradicts the first position, the Core itself must generate a second reasoned position. It must acknowledge each dissenting specialist; ignoring material objections blocks the result.
5. **CLASSIFY:** the deterministic runtime applies the result ceiling. Discretionary low-risk disagreement may produce **CONTESTED_ADVISORY**; a consequential unresolved disagreement produces **HOLD_UNVERIFIED_DISPUTE**. Conflicting verifier assertions produce **HOLD_UNATTESTED_VERIFIER_CONFLICT**, and agreement can produce **VERIFIER_ASSERTED_ADVISORY**. Neither status authenticates the assertion. `claim_verified` and `verified_proposition_is_false` remain false in this source-only engine.
6. **AUTHORIZE:** protected effects remain owner-gated, even when caller-provided verifier assertions agree with the Core. `effect_executed=False` describes only engine-owned actions: injected Core, specialist and verifier Python callbacks are executable and **not sandboxed or monitored** by this module. The user/owner's authorization machinery is a separate prerequisite for any protected side effect.
7. **RECORD:** a deterministic content SHA-256 binds the task, both Core positions, objections, provided verifier metadata, disposition, and explicit no-execution/claim ceiling. Receipt schema `MOSAIC_THINKING_CORE_DELIBERATION_V2` distinguishes the stricter, unattested verdict contract from older V1 receipts; no V1 receipt should be silently promoted. A SHA-based receipt is **not** a signature and does not authenticate a model or verifier.

For unresolved controversial or non-testable claims, there is no algorithm that can simply choose the more intelligent model. The Core may still choose an interpretation for a low-risk advisory answer, disclosing conflict and limits; consequential actions are held. A future independently authenticated verifier integration could bind real evidence to an exact task and permit a stronger verdict, but the current source-only protocol must not claim that qualification.

## What this implementation does and doesn't do

`runtime.thinking_core` is a strict in-memory protocol/arbitrator. `runtime.neural_core` turns a *supplied* generation backend into a model-driven THINK and RECONSIDER boundary with strict JSON parsing, distinct system/user messages, and untrusted specialist objections transmitted as data. It does **not** instantiate a model, load a LoRA adapter, switch specialists on the GPU, persist state across process restart, authenticate provenance, or prove that the Core performs useful reasoning.

The bounded tests use fake Core responses and synthetic verifier fixtures to check the *policy*, not the correctness of the claims. A real prototype needs a qualified resident Core inference backend, actual independent specialist adapters/models, permission-scoped verifier connectors, an event-sourced durable task state layer, and hardware P0/P1 observations.

## Experiments required before adopting the design

The first adversarial study should compare (A) Core alone, (B) Core plus specialist opinions, (C) Core plus independently checked evidence, and (D) a comparable single generalist with equal resource and token budgets. Include misleading confident specialists, failures the Core can't comprehend, unreliable verifiers, contradictory advice, stale state, protected effects, and at least one genuinely cross-specialist task. Measure final task accuracy, unresolved/false-certification rates, latency, extra tokens, memory, and user correction burden.

**Success means the Core's thinking is measurably useful while remaining correctable, not that it can declare itself sovereign or always win disagreements.**
