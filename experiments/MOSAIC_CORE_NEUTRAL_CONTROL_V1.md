# Mosaic Core — neutral second-pass paired scoring V1

**Status:** executable offline *scorer* with one synthetic fixture; **not** a live
neural evaluation, actual cost-matched inference run, independent gold-label
verification, specialist training, or demonstration of improved reasoning.

## Experimental comparison

Use the **same frozen case IDs, expected answers, and model revision** for:

1. `core_only`: one Core answer; a descriptive low-cost baseline.
2. `neutral_second_pass`: two Core calls, the second prompted to reconsider
   without a specialist objection.
3. `challenged`: an initial Core call followed by a **scripted, non-model**
   challenge and Core reconsideration; include challenge overhead in supplied
   `total_tokens` and elapsed time.

The challenge and neutral arms have the same **declared** two-call/token
ceilings. The scorer separately tests *observed supplied* calls and token-use
parity, with an explicit per-case tolerance. One call versus two is *not* a
cost-matched superiority test; only challenge versus neutral can be compared
on that resource basis. All three arms must contain every case, with no
post-hoc case dropping. `ok`, `abstain`, and `output_contract_error` are distinct;
abstentions and errors do not count as correct guesses.

V1 deliberately rejects `challenge_source=learned` or `mixed`: two Core calls
plus a learned critic require at least a third model invocation, but V1 has no
separate critic-call accounting or three-call matched neutral control. A future
schema needs explicit Core/critic calls and matched total inference budgets.
Successful `ok` rows must report at least one call and a nonzero token count;
all resource numbers are still unauthenticated declarations.

For a real experiment, freeze the independent case oracle and model/critic
identities **before** generating outcomes; preserve raw generations, token
accounting, prompts, service versions, timestamps, and evaluator custody
outside this file. A self-reported SHA-256 or model name does not authenticate
them. Include misleading critics and neutral review prompts with identical
caps. Run held-out and domain-shift cases; avoid making the evaluator's own
case corpus public to the evaluated model.

## Reproduce the scorer (no model or network access)

From the repository root:

```shell
python -m runtime.challenge_eval < experiments/fixtures/MOSAIC_CORE_NEUTRAL_SYNTHETIC_V1.json
python -m unittest discover -s tests -p test_challenge_eval.py -v
```

The **synthetic** fixture has one challenged-only win, one challenged-only
loss, and one tie versus a neutral second pass. Thus both arms score 2/3,
with net gain zero; the descriptive one-call Core-only baseline scores 1/3.
The input digest binds all supplied fields for reproducibility, **not**
truthfulness, predeclaration, external model identity, or independent custody.

## Fail-closed interpretation

The scorer rejects malformed typed fields, duplicate case IDs, extra fields,
nonboolean gold/answers, impossible reported budgets, and false claims of an
answer when an arm aborted or abstained. It reports unbalanced realized
resource usage rather than silently labeling the paired result cost-matched.
The report always sets:

- `gold_labels_independently_authenticated = false`
- `model_identity_attested = false`
- `independent_evaluator_custody = false`
- `neural_benefit_qualified = false`

This is protocol/score arithmetic only. Mosaic's claimed single logical agent,
actual independent specialist benefit, accelerator swaps, and P0/P1 remain
unverified by this artifact.
