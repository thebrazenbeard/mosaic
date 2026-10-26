# Pretrained-to-pretrained learned latent bridge — first negative pilot

**Observed qualification: HOLD / NEGATIVE.** This is a post-hoc exploratory test after the count toy result, not a preregistered P1 experiment. Run on real Windows VN1 in `%TEMP%\mosaic-latent-vn1-20261026` with PyTorch CPU `2.8.0+cpu`, Transformers `4.57.1`; the shared Ollama service was not touched.

Models are the two distinct original pinned Hugging Face SmolLM2 models: source 135M `12fd25f77366fa6b3b4b768ec3050bf629380bac`, target 360M `a10cc1512eabd3dde888204e902eca88bddb4951`. The 135M model produced 640 mean-pooled 576-dimensional hidden vectors from binary-observation text and was unloaded (weakref released) **before** the 360M model was loaded. Frozen source embedding tensor SHA256 `cfa48981a3adee8733f1f6b9f20302dea51d56e8165b92fff3354f0ad75cbb1b`.

The 360M model's pretrained weights were frozen. A learned 576→128→960 projection provided one continuous prefix via `inputs_embeds`; supervised training optimized only adapter weights for next-token yes/no choice. Pilot fixed settings: seed `20261026`, first 512 of the existing 4,096 training rows, batch `4`, AdamW learning rate `0.005`, `120` steps. Trained bridge checkpoint SHA256 `f16f23ba18054c0fb6aae3e64d1a9222eed92e24bd545fba7efd16c986644098`. End-to-end elapsed `169.332 s` on VN1 CPU; first-step loss `1.00897`, final step loss `0.88115` (intermediate steps varied).

| Arm | Correct / 128 heldout | Interpretation |
| --- | ---: | --- |
| Trained continuous-prefix 135M→360M | **64 / 128 (50%)** | FAIL |
| 360M model with full raw observation history | **64 / 128 (50%)** | Equal chance-level comparator |
| 360M model without transferred state | **64 / 128 (50%)** | Equal chance-level reset |

**Counterdiagnostic (same data, post-hoc):** A small learned MLP head applied to the frozen 135M hidden vectors plus query-threshold one-hot, *without* the 360M model, increased accuracy over 100 epochs from 64/128 to **122/128 (95.3%)**. This is source-state decodability evidence, not pretrained-to-pretrained reasoning continuity. No fresh sealed holdout was reserved for the diagnostic, so do not use it for a P1 pass.

> **Hostile challenge:** The 360M model's binary next-token interface remained at chance for the raw full-history condition too. Thus the experiment does not isolate a deficient cross-model bridge from a deficient answer-token/prompt interface, and the training/evaluation budgets are not comparable to an optimized retrieval-augmented specialist. These samples cannot establish that Mosaic beats conventional retrieval.

**Next experiment, not yet qualified:** Train a small decoder head on the target model's hidden representation alongside the latent-prefix bridge, and train a matched decoder head for a full-record retrieval arm. Use a new frozen heldout cohort and two independent training seeds; account for all weights/compute and include zero/rolled-latent, shuffled-source and equal-token controls. Hold both P1 and unrestricted retrieval superiority until those falsifiers are survived.
