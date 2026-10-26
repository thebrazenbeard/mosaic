# Mosaic VN1 learned-state research — 2026-10-26

**Research verdict: narrow toy result PASS; pretrained-LLM learned continuity HOLD; unrestricted retrieval superiority NOT DEMONSTRATED.**

## Source / live machine boundary

Experiment source: `experiments/p1_toy_latent_benchmark.py`; exact executed final scorer SHA256 `554d6e9af7898d45c4e277a108841f16bdd92c54030a044e8b8a7be57bf162b2`. Initial training used the same train entrypoint before the post-result zero/rolled-state control was added: SHA256 `5804382d06a26a91e8e5c978eed1c9a5a87e8a58d92f9e3dd951ec435e3153a2`. The control does not retrain any weights or change the heldout cases.

Hardware: real Executor `VERA-NODE-1` Windows device, Python 3.13.15, CPU-only PyTorch `2.8.0+cpu`, 10.28 GiB available RAM before setup; disk initially 140.18 GiB free. Isolated `%TEMP%\mosaic-latent-vn1-20261026\venv`; two running Ollama processes remained running through and after the probe. No Ollama weights, processes, configuration or shared services were intentionally altered.

Pretrained model wiring: `HuggingFaceTB/SmolLM2-135M-Instruct@12fd25f77366fa6b3b4b768ec3050bf629380bac` (weights SHA256 `5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`) emitted a 576-d real hidden tensor, unloaded, then `HuggingFaceTB/SmolLM2-360M-Instruct@a10cc1512eabd3dde888204e902eca88bddb4951` (weights SHA256 `e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e`) accepted it through an untrained projection and `inputs_embeds`, without text substitution. Its logits changed by `4.135274887084961`, 1.6 seconds elapsed. **This is a real neural connection, not a trained LLM transfer.**

## Trained toy transfer — original preregistered outcome

Frozen dataset SHA256 `4618d1f3182bbcf4f5d7035958c468e8dfb0fc7b0306c1136a7a80d017d3baab`; seed `20261026`; 4096 training and 512 unused development cases; two disjoint balanced 64-case heldout slices, sharing a seeded generator stream and one trained weight set. Source encoder and target decoder trained 100 epochs with AdamW, 19.253 seconds. Tensor channel only: eight float32 values / 32 bytes, carrying the learned sufficient statistic of 16 binary records. Distinct OS processes loaded source and target weights sequentially; no source model or raw records entered the target process.

| Arm | Run A | Run B |
| --- | ---: | ---: |
| MOSAIC_LEARNED_STATE | 64/64 (100%) | 64/64 (100%) |
| ORDINARY_RETRIEVAL, eight indexed records | 38/64 (59.375%) | 36/64 (56.25%) |
| FULL_HISTORY, all 16 records | 64/64 (100%) | 64/64 (100%) |
| RESET, query only | 32/64 (50%) | 32/64 (50%) |

Paired Mosaic wins/losses vs retrieval: run A `26/0`, exact one-sided `p=1.4901161193847656e-08`; run B `28/0`, `p=3.725290298461914e-09`. Both pass the frozen +5 pp / p<.05 / >=80% narrow gate. Frozen source weights SHA256 `c259f3f546ff241f169bd5a69ee4fe4e1ea3dfaadfff49f1f6d28de62dc70e82`; target weights `25c23ab4b51b089005e6c87507e8e3a71af68cc40e52f212e32485969d4ffc9c`.

Post-result controls on unchanged frozen weights/cases: zeroing state drops accuracy to 32/64 on both; rolling state across different cases drops to 30/64 and 31/64. Raw paired case identifiers, correct labels, every arm prediction, post-hoc ablation predictions, weight hashes and gate results: [`results/P0_NEURAL_COUNT_V1.json`](results/P0_NEURAL_COUNT_V1.json).

## Material hostile challenges

> **The retrieval budget is constrained.** The text-record comparator sees only eight of 16 records. If ordinary retrieval is allowed all 16, the full-history reader ties Mosaic at 100%. A blanket claim that latent state is better than conventional retrieval would be false.

> **The learned specialists are small purpose-built networks, not pretrained language models.** The distinct pretrained SmolLM2 pair passed *untrained* continuous-input wiring only. Neither result alone or in combination proves learned state transfer between the two SmolLM2 models.

> **The learned pipeline has extra encoder parameters and training compute.** The retrieval reader was trained with the same episodes, epochs and learning rate, but its restricted input cannot reconstruct the absent eight event values. This is evidence for task-specific compression under a handoff budget, not a fully cost-equal comparison with unconstrained retrieval.

> **No external independent qualification yet.** The live host receipts and paired scorer have not been independently audited; reproduction from pinned source and machine is the next gate. The two heldout slices are not independent full training replications.

**Next qualification:** Train a bridge that injects *learned* source state from the frozen 135M pretrained language model into the distinct 360M language model, with matched context budgets, full retrieval, two independently trained seeds, negative controls and authenticated source/model receipt. Keep Mosaic's general P1 claim HOLD until it wins a properly controlled frozen benchmark.
