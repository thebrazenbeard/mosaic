# Mosaic P1 — isolated one-case live learned-critic smoke run

**Not P1 qualification.** This collector uses two actually invoked local model
IDs in a dedicated `llama.cpp` router and submits reported counts to the V2
offline scorer. It reuses one observed initial Core call, then runs a neutral
critic + Core reconsideration and a challenged critic + Core reconsideration.
Five physical requests produce the conceptual one-call and paired three-call
arms. Core messages must return a strict one-field `supports` JSON object;
thinking tags, extra fields, and unparsable outputs are contract failures.

Run only on an explicitly dedicated, idle localhost router with a single loaded
model maximum. It **loads and unloads models** and must not touch a shared
Ollama or inference service. The caller supplies exact local model file paths,
one previously declared case, and an asserted corpus timestamp. With both
models unloaded and the router locked to `127.0.0.1`, for example:

```powershell
python -m experiments.p1_live_three_arm `
  --router-url http://127.0.0.1:8518 --dedicated-router `
  --core-model core-4b --critic-model critic-1p5b `
  --core-model-file D:\OllamaModels\blobs\EXACT_CORE_BLOB `
  --critic-model-file D:\OllamaModels\blobs\EXACT_CRITIC_BLOB `
  --case-file experiments\fixtures\MOSAIC_P1_PUBLIC_SMOKE_CASE_V1.json `
  --experiment-id p1-smoke-20261025 --corpus-id public-smoke-v1 `
  --frozen-at ACTUAL_PRE_RUN_UTC_TIMESTAMP --n-predict 24 `
  --out p1-local-raw-observation.json
```

Replace `EXACT_*_BLOB` with separately identified and hashed real local files.
The file records all raw model replies, exact submitted prompts, call order,
self-reported token counts, model file digests, V2 input and V2 score. Treat it
as a potentially sensitive *local* raw trace; do not publish it automatically.

The corpus here is a **public smoke case**, not an unseen holdout. The declared
freeze time is not independently attested. No model weights, router metadata,
token counters, loaded-worker identity, hardware residency, critic independence,
or gold oracle is independently authenticated. Fixed neutral-before-challenged
order and prompt/context differences remain confounders. The V2 scorer always
reports `neural_benefit_qualified: false`; one case cannot estimate a benefit.

Source-only unit tests use a fake router and do not load weights or make any
network request:

```sh
python -m unittest discover -s tests -p test_p1_live_three_arm.py -v
```
