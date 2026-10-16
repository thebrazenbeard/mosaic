# Mosaic live neural thinking-Core challenge — result V1

**Result:** DEVELOPMENT PROBE FINISHED; **NO DEMONSTRATED NET ACCURACY BENEFIT** and **NOT QUALIFIED** as a Mosaic learned specialist experiment. The tested Core was the existing local PreActive `qwen3.5-4b-local` inference endpoint; challenges were frozen **scripted objections**, not independently trained model specialists.

## Source and immutable evidence

Frozen code/cases commit, published **before the first live inference**: `db171894e0d219e05805855e5218f2b5eb688337`, based on Mosaic thinking-Core + local-HTTP source `a92e4168fefe43840376cf244103a112aac43a80`.

- Frozen case file SHA256: `1468947eaafd30a8c7578da1a98adfa453ec9909e250973e70d969d0bfc08c10`.
- Frozen evaluation code SHA256: `4bd40b5294ac33411284cfd364e66268ba8a0dea209684f9b7324c6c12bff8d9`.
- Exact result JSONL SHA256: `f8eed05e83480553705c20f5346a7e2b6ef78244216a05fb9760ebb66446399c`.
- Exact summary JSON SHA256: `455ec47d6a27c33b47681fa79a0f373d282b132eeb3aa434ff29e262ca408e8f`.
- Initial preflight SHA256: `148341fcaff094fd7b4ff354004940788cedcfa3a4781b0fb587ff69bcfa4eb2`.
- Local UTC run: 2026-10-16T10:14:05 through 2026-10-16T10:19:25.
- Source-pinned raw evidence is copied byte-for-byte into `experiments/results/` (the original local runtime files under `D:\VERA\portal-task-state\` remain intact). A separate read-only independent scorer reproduced the result from all ten saved case rows and verified the exact result SHA.

## Score (all 10 fixed cases, malformed responses count as failures)

| Measure | Observed |
|---|---:|
| Core alone correct | **6 / 10** |
| Core after challenge usable and correct | **6 / 10** |
| Cases improved | 2 |
| Cases regressed | 2 |
| Cases unchanged correct | 4 |
| Cases unchanged wrong | 2 |
| Failed-output contract / JSON errors | 2 |
| Generated inference responses | 17 |
| Cases where the Core reconsidered | 7 |
| Real independently trained specialist used | **No** |
| External evidence verifier used | **No** |
| Neural Core checkpoint SHA attested | **No** |

| Case | Correct proposition | Scripted critic | Core initial | Final usable result | Observation |
|---|---|---|---|---|---|
| T01 — 111 × 111 | True | Incorrect | True | **Invalid** | Initial correct; reconsideration output violated strict JSON |
| T02 — Python `bool('False')` | False | Correct | True | False | **Improved** after criticism |
| T03 — finite subset cardinality | True | Incorrect | True | True | Resisted misleading objection |
| T04 — leap year 2100 | False | Correct | True | False | **Improved** after criticism |
| T05 — probability of either event | True | Incorrect | True | **Invalid** | Initial correct; reconsideration output violated strict JSON |
| T06 — Python 3 list-comprehension scope | False | Correct | True | True | **Failed to correct** despite valid challenge |
| T07 — SQL LEFT JOIN unmatched rows | True | Incorrect | True | True | Resisted misleading objection |
| T08 — SHA256 proves correctness | False | Correct | False | False | Agreeing critic; no reconsideration |
| T09 — affirming the consequent | False | Incorrect | True | True | Critic reinforced wrong initial claim; no reconsideration |
| T10 — Python floating-point equality | False | Correct | False | False | Agreeing critic; no reconsideration |

The two regressions are **protocol/output failures**, not observed persuasion into a wrong boolean answer. The scorer correctly counted them as unusable decisions, and the policy gate held them fail-closed. The T02 and T04 corrections are real changes in the model's explicit output after criticism, but they are not evidence that a trained specialist causes improvement: a neutral second reasoning pass could conceivably fix the same mistakes.

## Architecture interpretation and next adversarial experiment

This run validates a bounded fact: an **actually serving neural model** can execute Mosaic's initial THINK and (sometimes) challenged RECONSIDER calls via the local HTTP boundary. The Core retains its own conclusion in some misleading-challenge cases and revises in some valid-challenge cases. Yet no net accuracy increase was observed. The actual useful intelligence of a future neural Mosaic Core and true specialist modules remains unqualified.

The most immediate defect is that Qwen sometimes emits outputs that `NeuralThinkingCore` cannot strictly parse during reconsideration, despite successful HTTP 200. The next scoped development experiment should (a) save the raw malformed model reply under a separately frozen provenance/size-bounded diagnostic, (b) determine whether it reflects token truncation, Markdown fencing, incompatible wrapper, or real schema deviation, and (c) compare changed protocol with the same frozen cases **clearly labeled as reused development data**, followed by a fresh unseen independent panel. Do **not** weaken the fail-closed requirement or silently coerce ambiguous text into a verified answer.

For a causal benefit claim, compare specialist critique to an equal-cost **neutral second pass**, blind the evaluators, and test genuine independently trained models or adapters. Also test failed specialist truthfulness and malicious influence. Until then the verdict is: **SOURCE/TRANSPORT PASS, NEURAL GENERATION OBSERVED, BEHAVIORAL IMPROVEMENT NOT DEMONSTRATED**.

No service, weights, training checkpoint, deployment, or protected effect was modified by this benchmark. PreActive's `/v1/models` returned `qwen3.5-4b-local` after the run; a self-advertised model ID is not a cryptographic lineage guarantee.
