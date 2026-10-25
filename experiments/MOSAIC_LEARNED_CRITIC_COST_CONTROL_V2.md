# Mosaic P1 — learned-critic three-call cost control V2

**Status:** separately implemented *offline, caller-reported* scoring protocol.
It does **not** execute neural inference, prove that a critic learned, authenticate
models or token counters, independently freeze a corpus, or qualify P1.
The V1 scripted-challenge scorer remains unchanged and cannot score a learned
critic because its two-call budget omits a critic invocation.

## Three same-case arms

| Arm | Call 1 | Call 2 | Call 3 | Use |
| --- | --- | --- | --- | --- |
| `core_only` | Core initial | — | — | Descriptive one-call baseline, not cost matched |
| `neutral_three_call` | Core initial | Same claimed critic model, neutral task | Core reconsider | Three-call resource control |
| `challenged` | Core initial | Same claimed critic model, learned objection task | Core reconsider | Candidate intervention |

The two three-call arms are scored on the **same complete case set** with
the same self-declared Core and critic model IDs, same initial Core output
SHA-256/input/output token receipt, exactly three ordered inference calls per
successful arm, and one token ceiling. Critic requests must differ only in the
pre-registered intervention versus neutral instruction. The scorer validates
declared receipt structure and compares the reported sum of input and output
tokens to an explicit tolerance; it makes no claim those receipts are true or
the neutral task is substantively fair. Even matched total tokens can conceal
different compute cost, hardware residency, latency or thinking-token usage.

Required top-level JSON object fields are:

| Field | Contents |
| --- | --- |
| `schema_id` | `MOSAIC_LEARNED_CRITIC_PAIRED_INPUT_V2` |
| `experiment_id` | Bounded nonempty identifier |
| `challenge_source` | `learned` (caller claim; not attested) |
| `model_manifest` | `core_model_claimed_id`, `critic_model_claimed_id` (distinct strings) |
| `corpus` | `corpus_id`, `claimed_frozen_at_utc` (UTC seconds), `declared_sha256` |
| `budget` | `matched_inference_calls: 3`, `max_total_tokens_per_arm`, `token_delta_tolerance` |
| `cases` | Nonempty array of 1–1000 cases, no duplicates |

Each case has `case_id`, `prompt`, boolean `expected_supports`, and the three
named arms above. Each arm has `status` (`ok`, `abstain`, or
`output_contract_error`), `supports` (boolean only when `ok`; otherwise null),
and `calls` (ordered, possibly incomplete on non-success). Each call has
`role`, `model_id`, positive `input_tokens`, `output_tokens`, nonnegative
`elapsed_ms`, and `output_sha256` of exactly 64 lowercase hex characters.
**Successful calls require positive output tokens** and all role-correct
calls, including the separate middle critic invocation. There is no
zero-call/zero-token success escape. Neutral/challenged partial or errored
cases remain in paired scoring and make that case's reported budget unmatched.

## Corpus binding is not a freeze attestation

`corpus_digest(cases)` deterministically hashes the ordered subset
`[{case_id, prompt, expected_supports}, ...]`, using sorted-key compact UTF-8
JSON. The supplied digest must match the case data exactly; the returned input
digest binds the entire report input. Both hashes can be recomputed and
redeclared by anyone controlling the input. In particular, neither proves the
cases were unseen, the gold oracle was independent, the benchmark was
precommitted before generating answers, or the reported first-turn Core
outputs actually match. For a real study, an independent steward must register
the holdout prior to collection, retain private cases and timestamps, and
preserve signed raw prompts, responses, model revisions, token counter logs,
pricing/hardware conditions, randomized ordering, and an audit trail.

## Executing the scorer

```sh
python -m unittest discover -s tests -p test_challenge_eval_v2.py -v
python -m runtime.challenge_eval_v2 < independently_collected_v2_input.json
```

The new regression suite uses **synthetic, self-generated receipts** only.
For that fixture, Core-only is 1/3 correct; both three-call arms are 2/3;
challenged-minus-neutral paired wins/losses/ties are 1/1/1 (net zero).
These are arithmetic test vectors, **not an empirical learned-model result**.

The scorer always returns `neural_benefit_qualified: false`,
`reported_token_counts_independently_measured: false`,
`model_identity_attested: false`,
`learned_weights_independently_attested: false`,
`corpus_freeze_independently_attested: false`, and
`independent_evaluator_custody: false`. A future independent inference run
must establish all of those separately and test multiple seeds, misleading
critics, leakage controls, domain shift and cost-matched effects before
claiming specialist benefit or a self-improving model.
