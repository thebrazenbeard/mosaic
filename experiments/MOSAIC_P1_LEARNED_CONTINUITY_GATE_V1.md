# Mosaic P1 learned-state continuity — preregistered qualification gate V1

Status: **research gate; no qualified results**. Frozen before any new P1 learned-state result is collected under this protocol.

## Question and falsifiers

Can two separately loaded neural specialists transfer information through a **trained continuous-state adapter** and improve unseen final-task accuracy over conventional retrieval using the same stage-A source and a declared bounded information/compute budget?

- **H1 / feasibility:** A's source-bound intermediate state survives complete A unload and is consumed by B after B is loaded. Falsifier: missing worker/unload evidence, receiver not using the trained adapter, or a test that can be solved from B's final prompt alone.
- **H2 / comparative benefit:** MOSAIC STATE outperforms ORDINARY RETRIEVAL on disjoint, preregistered heldout cases in two clean runs. Falsifier: tied or lower paired accuracy, increased stale errors, or failure under a matched cost/information budget.
- **H3 / learned mechanism:** An immutable, nonzero-parameter adapter checkpoint is trained only on the training split and used for the reported receiver run. Falsifier: deterministic JSON/current-value lookup, untrained/identical weights, any training/heldout overlap, or a purely textual handoff relabeled as latent state.

## Confirmatory experiment P1-LATENT-01 (not yet executed)

1. Register immutable model and adapter file SHA-256s, architectures, software versions, the training/development/heldout split, and a heldout-corpus commitment **before model calls**. Independent custody is required for a definitive claim.
2. Sample at least **64 heldout cases**, balanced across the eight existing P1 families. Use the same stage-A source and final query under RESET, MOSAIC STATE, FULL HISTORY and ORDINARY RETRIEVAL. The final B prompt may not leak the heldout answer.
3. Train only the bridge adapter on training cases. Explicitly log optimizer/step count, adapter weights/checkpoint, change from initialization, and leakage checks. Expose learned continuous state to B through a verified supported mechanism (e.g. receiver input embeddings or KV translation), **not** text injected as if it were a latent state.
4. A and B must be distinct pinned neural models. Record their physical process identities, unload/load acknowledgements, GPU/RAM telemetry, request timing, and receipt bytes. Do not replace this with endpoint metadata or claimed model IDs.
5. Randomize trial and condition order with a pinned seed; use an independent scorer blinded to arm names until predictions are fixed. Include a frozen equivalent-source conventional retrieval control and the full-history capability control. Match input payload budgets and report all actual costs; do not discard errors or failed cases.
6. Perform a second clean independently witnessed run after process restart with the same frozen weights/split. Report per-case paired wins/losses/ties, stale errors, correction burden, payload bytes, latency and token/compute use.

## Decision rule

No P1 learned-state qualification unless (a) every hardware/training/heldout prerequisite is independently attested, (b) MOSAIC STATE beats RESET without increased stale errors, (c) FULL HISTORY proves B can solve remaining tasks when given complete information, (d) MOSAIC STATE beats ORDINARY RETRIEVAL by at least 5 percentage points **and** a one-sided exact paired discordance test gives p < 0.05 in *both* clean runs, and (e) reported costs satisfy the declared budget. Failed experiments remain visible.

The source scorer can only evaluate **caller-supplied receipts**. It must return an explicitly *unattested* classification even when those receipts meet thresholds. A model agreement, file path, self-declared training manifest, local unit test, or one successful case does not independently authenticate training, model-worker identity, or continuity.

The executable score-only interface is
`runtime.learned_continuity_gate.evaluate_learned_continuity`. Input has an
immutable experiment ID, exact SHA-256 model/adapter/corpus commitments, two
to four declared runs on the **same balanced 64..512 heldout case subjects**,
and an exact four-arm prediction/byte/time/token receipt for every case.
Each run has a common payload cap and a predeclared maximum reported token
total **per arm**. Predictions are compared against caller-provided gold with
an exact one-sided paired discordance test. Crucially, the output always says
`UNATTESTED_DECLARED_EVIDENCE_NOT_P1_PASS` and sets
`p1_learned_state_verified=false`; supplied flags are never treated as an
independent witness.

## Current baseline and blocking resources (2026-10-26)

- Deterministic `runtime.handoff` eight-case harness: RESET 0/8; MOSAIC STATE, FULL HISTORY and ORDINARY RETRIEVAL each 8/8. This is not neural inference or comparative benefit.
- Existing live P1 V2 collector is a different question (Core/critic intervention), not cross-specialist continuity.
- Lappy live read: RTX3050 4 GiB, about 2,046,124 KiB free RAM at preflight; Python Torch 2.14.1 **CPU-only**, Transformers available, no current Mosaic router listener. Current GGUF-router HTTP API does not establish that a continuous state was extracted or injected. A new hardware run is on HOLD pending a safe host-memory preflight, bridge implementation, and independent benchmark custody.

## Research/evidence log

- **E1 SOURCE:** Mosaic `main@6a8cbceea75fe8993ea7dcf4295cbe2b298c6381`, `runtime/handoff.py`, tests and frozen P1 V1, read 2026-10-26.
- **E2 TEST:** `python -m unittest tests.test_p1_handoff -q` passed 12/12; 8-case deterministic counts recorded above.
- **E3 HOST READ:** Lappy PowerShell `Get-CimInstance Win32_OperatingSystem`, `nvidia-smi`, Torch/Transformers import checks; about 2 GiB free and Torch CUDA unavailable at observation time.
- **E4 RESEARCH:** Dery et al., *Latent Space Communication via K-V Cache Alignment*, arXiv:2601.06123, and Liu et al., *Dual-Cache Latent Space Communication between Heterogeneous Language Models*, arXiv:2608.20617. Both describe *trained* cross-model latent/cache adapters; their published findings do not verify a Mosaic implementation or the local llama.cpp route.
- **E5 RED/GREEN TEST:** Fresh new scorer regression cases rejected a not-yet-implemented gate and an unmatched declared token budget; final focused `tests.test_p1_learned_continuity_gate` 10/10 PASS. The deliberately favorable synthetic data in unit tests exercises reporting, **not** a neural run or evidence that Mosaic wins.
- **E6 OFFICIAL BACKEND CONTRACT:** Current ggml-org/llama.cpp `tools/server/README.md` (retrieved 2026-10-26) documents message completion, embeddings output and same-model slot-save/restore routes; no reviewed public cross-model hidden-state injection path was established for Mosaic's installed pinned router. A different verified backend or an explicitly implemented continuous-state interface is needed before P1-LATENT-01.

Exploratory experiments and deviations must be appended here, never represented as if they were predeclared confirmatory successes.
