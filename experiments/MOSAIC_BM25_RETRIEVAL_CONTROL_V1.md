# Mosaic P1 lexical retrieval control — initial executable comparator

**Status: source/test PASS; model comparison UNKNOWN; P1 HOLD.**

`p1_bm25_retrieval_control.py` implements a deterministic BM25 lexical ranker over
16 independently indexed flag-observation documents. Each document carries its
original index and value; the query contains only truth-neutral terms.
This is **actual text document retrieval**, unlike the prior
`bits[:, :8]` MLP labeled `ORDINARY_RETRIEVAL_8_RECORDS`.

The bounded reader calculates whether its retrieved observations logically
force the yes/no answer. If missing documents could reverse the conclusion,
it **abstains** instead of guessing or treating abstention as accuracy.
Metrics separate `correct`, `incorrect`, `abstain`, coverage, and accuracy
conditional on deciding. BM25 scores use `k1=1.2`, `b=0.75` and a documented
recency tie-break for equally relevant records. `top_k=16` sees all observations.

Run a posthoc diagnostic on the original **already-seen** two research splits
only in the isolated Torch-enabled experiment environment:

```powershell
$env:PYTHONPATH='experiments'
python -c "from p1_toy_latent_benchmark import episodes; from p1_bm25_retrieval_control import evaluate_cases; d=episodes(); print(evaluate_cases(d['run_a']+d['run_b'], top_k=8)); print(evaluate_cases(d['run_a']+d['run_b'], top_k=16))"
```

These splits were already inspected while developing the neural models. Their
results **cannot** be presented as a fresh blind confirmatory P1 test.

## What a fair next benchmark must add

1. Freeze a **new**, disjoint held-out cohort and hash its generator and case
   identities **before** model selection or training. Do not use its scores to
   tune either approach. Publish every paired prediction, not only aggregates.
2. Train and validate **both** latent-state and full-retrieval *target-side
   answer heads*, with predeclared equivalent budgets/steps, independent seeds
   (at least three), and an identical held-out decision set. Do not compare a
   trained neural head against an untrained frozen-LM prompt and call it a win.
3. Report retrieved payload bytes and token counts, ranker/index residency,
   peak RSS/VRAM, model swapping time, per-case latency, and head parameters.
   The 16-bit full history can be packed into **2 bytes** while the existing
   latent has **32 bytes**; that representation is a stronger trivial
   comparator than an arbitrarily truncated eight-record baseline.
4. Add no-state, rolled-state, wrong-specialist and id-swap controls; require
   correct specialist weight/revision identities and actual source eviction.
   Ensure source state contributes causal task information rather than answer
   leakage through a prompt, retrieval label, or direct raw-history bypass.
5. A positive comparison must be qualified to its exact **compute/context
   budget** and task distribution. Even a significant budgeted win is not a
   proof of *unrestricted* retrieval superiority or a shared continuously
   learned reasoning state.

This first step is a **methodological control**, not a P1 pass.
