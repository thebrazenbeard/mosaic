# Mosaic ↔ Rezon advisory bridge V1

Status: **source interface with synthetic real-store contract test**, not a
trained Mosaic, distinct-model neural integration, authenticated host, or P1 PASS.

Pinned inspection baseline (2026-10-25):

| Component | Inspected `main` |
| --- | --- |
| Mosaic | `c58c27b17bf7f395349f9da5498a1326e8b804c0` |
| Rezon | `7bbc676baf4621888377783d763e12f1c3c95806` |
| vera-mono | `d106d64304b75a710675362ef3dd48ac67bbe3fb` |

The existing vera-mono
`packages/vera_core/src/vera_core/external_rezon_collective.py` is a
read-only authenticated-loopback **observation** client. It validates
Rezon gateway response framing and a source snapshot but does not expose a
verified answer or permit effects. This bridge does not modify that contract
or dynamically import Rezon into Mosaic's protected Core runtime.

## Exact operation

1. Create a Rezon collective episode with
   `runtime.rezon_advisory.rezon_task_prompt(task)` as its exact prompt.
2. Enroll the intended independent agent IDs in the episode. Have actual
   Rezon participants submit JSON answers with exactly `proposition_id`,
   `supports` (boolean), and `rationale` (nonempty string); the Rezon store
   records claims but does not attest model weight identity or truth.
3. Once all participant answers are present, pass a **separately obtained**
   `CollectiveStore.snapshot(episode).to_dict()` to
   `RezonAdvisorySpecialist(task, snapshot, agent_id)`.
4. Pass that adapter as a normal specialist to
   `CoreDeliberationEngine.deliberate(...)`. Its `review` ignores the Core's
   initial stance and supplies only the independent first-round model answer.
   Rezon peer reviews or syntheses are never promoted to evidence or
   substituted for that answer.
5. Log `adapter.provenance` with the Mosaic receipt if the operator needs an
   episode/digest/contribution correlation. Its status is always
   `UNVERIFIED_REZON_ADVISORY`.

The adapter validates the exact task prompt, complete Rezon roster, answer
and peer-review envelopes, duplicate identities, content bounds, Rezon's
canonical contribution-input digest, synthesis labels, and strict typed
JSON output (including duplicate-key rejection). It binds to the **entire**
Mosaic task, so reuse across an unrelated case with the same proposition ID
fails closed. The digest is collision-resistant *content consistency*, not
producer authentication: an attacker with complete control of the snapshot
can fabricate a self-consistent snapshot and recompute its digest. Downstream
Mosaic decisions retain `claim_verified=False`; no protected effect is
authorized or executed by the bridge.

## Verification

Without optional Rezon on the import path:

```shell
python -m unittest tests.test_rezon_advisory -q
```

With a fresh copy of Rezon adjacent to Mosaic, this executes its actual
`CollectiveStore` SQLite round trip, including a changed digest after a peer
review and a current unverified synthesis:

```shell
PYTHONPATH=../rezon/src:. python -m unittest tests.test_rezon_advisory -v
```

Synthetic answer strings in those tests are **fixtures**. They do not prove
separate learned specialists, training, provider authenticity, real VRAM
eviction, swap latency, or the preregistered P1 controls. A next-stage
live-model qualification must record the model artifact hash, engine/version,
host, inference timing, which participants were actually independent, and
held-out reset/full-history/retrieval baselines, then independently audit
the entire run.

### Fail-closed limits

- No startup task, font asset, automatic consent, external URL, or network
  operation is introduced.
- An incomplete collective cannot become Mosaic specialist advice.
- Rezon's `input_digest` can become stale after an appended peer review;
  capture a fresh snapshot and validate the new digest.
- Source review and passing tests are **not** hardware inference measurements.
- Caller-owned Python callbacks remain outside Mosaic's effect-observation
  and sandboxing boundary. No verifier or authorization trust root is added.
