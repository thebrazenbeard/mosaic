# Mosaic P1 V3 declared-frozen Python-semantics corpus

Status: **FROZEN INPUT CANDIDATE / NO LIVE P1 QUALIFICATION**.

This is a new, separately identifiable 12-case research corpus. It is **not**
the unpublished V2 corpus whose scratch workspace was reset. Its digest is
different and no attempt is made to recover that corpus's former provenance.

## Exact input subject

- File: `experiments/fixtures/MOSAIC_P1_PYTHON_SEMANTICS_FROZEN_V3.json`
- Corpus ID: `p1-python-semantics-balanced-v3`
- Declared local freeze time: `2026-10-26T09:17:38Z` (re-freeze after hostile review found label-leaking IDs)
- Canonical case digest: `7d482b7b4561ee049ff8d153d151dc48ed17182e89e549d31aa7e92f4ccc4477`
- Digest definition: `runtime.challenge_eval_v2.corpus_digest(cases)`:
  SHA-256 of JSON with exactly `case_id`, `prompt`,
  `expected_supports`, sorted keys, compact separators, UTF-8, case order
  retained.
- Balance: 12 cases, 6 true/6 false, six contrasting operator pairs.
- Case IDs: opaque truth-neutral `p1c01` through `p1c12`; unlike the
  rejected first freeze they encode no gold label. Gold remains in the local
  fixture, so the public file still cannot serve as a blinded holdout.
- Gold: deterministic ordinary Python 3 expressions; tests re-evaluate
  their operations without executing untrusted fixture text.
- Previous explored `len([])` pair is excluded. This is still a
  **public fixture**, not an independently secluded or sealed holdout.

## Contract for a future real-model three-arm comparison

1. Record the exact model file hashes, backend versions, CPU/GPU device,
   router ownership, prompt template and token/temperature settings **before**
   any run. Reserve an exclusive test instance; do not repurpose a shared
   inference service's loaded weights or routing state.
2. For each frozen case, the task's `proposition_id` is the exact
   `case_id`. A Core answer must be strict JSON with exactly
   `proposition_id` (matching), `supports` (boolean), `reasoning`
   (nonblank string), and `acknowledged_objections` (distinct nonblank
   specialist ID strings). JSON-object shape without those constraints or
   bare `{"supports":true}` is **not** a passing Core result.
3. Compare (a) Core only, (b) neutral critic plus Core reconsideration, and
   (c) challenged critic plus Core reconsideration. Both three-call arms
   reuse the *same recorded initial Core call*; record every actual
   model response, request, model ID, token accounting, latency, and failure.
   Preserve raw traces outside the public repository until sanitized.
4. Lock a randomization scheme for neutral-vs-challenged order before running;
   if fixed order is used, report it as a material confounder. Do not consult
   the gold label when composing prompts, challenges, or retries.
5. Cost-match neutral and challenged three-call arms and score with V2.
   **A learned-critic intervention requires independently identified
   learned specialist weights**; merely prompting a different model to
   object establishes only a *model-generated challenge*.
6. Report all contract failures and negative results without dropping cases.
   Model-set or prompt tuning on these cases converts them to exploratory
   data, requiring a fresh independent holdout for any efficacy claim.

## Scope and limitations

This commit freezes auditable data and exact source tests, **not experimental
results**. The existing one-case smoke collector cannot run this 12-case
fixture unmodified and is not a qualification runner. This public fixture
cannot independently prove its own timestamp, no leakage, model identity,
training, held-out benefit, independent replication, GPU residency, or
self-learning.

Run offline validation without starting a model:

```sh
python -m unittest discover -s tests -p test_p1_frozen_v3.py -v
```
