# Mosaic P2 development probe: real neural Core challenged by a scripted critic

This is a frozen, bounded exploratory test of whether a *serving neural Core* makes better decisions after a scripted specialist challenge. It is not a claim of a trained Mosaic architecture, an independently learned specialist, or model qualification.

## Fixed protocol (freeze before requesting neural completions)

Subject: Mosaic source stacked on PR #11. Backend: already-running Lappy PreActive Qwen3.5-4B HTTP service, selected only by its **self-advertised** model ID `qwen3.5-4b-local`. No asserted adapter/weights SHA identity. The source does not start, reload, stop, train or install a service.

Fixture: `MOSAIC_LIVE_CORE_CHALLENGE_CASES_20261016_V1.json`. Exactly 10 binary propositions with frozen truth labels and five correct / five misleading scripted arguments. The expected fixture SHA256 is `1468947eaafd30a8c7578da1a98adfa453ec9909e250973e70d969d0bfc08c10`. Ground-truth labels are evaluation-only, not sent to the model. Each proposed specialist stance and argument is frozen **before** any Core response.

Each case uses `NeuralThinkingCore` to produce one live Core-only initial position. For the challenged condition, the engine reuses that exact position (no fresh initial call) and passes one scripted specialist objection. The Core then produces a second *neural* response only when it disagrees with the scripted specialist; otherwise its initial decision stands. A second response must address the critic, or the resulting decision is held as not usable.

Scoring: the initial Core-only stance is correct iff its boolean equals the frozen truth. The challenged decision is correct iff its final stance equals the frozen truth AND the result is an allowable advisory disposition (not a held/blocked result). Errors, malformed JSON, network failures, or absent decisions score as incorrect; they are separately counted. Report direct paired improvements and regressions, breakdown for correct vs misleading criticisms, response validity, number of actual model generations, elapsed time and receipt digests. The scripted critic has no verification or protected-action authority.

Evaluation uses one fixed nonrandom run, no retries, and exclusive no-overwrite result files under `D:\VERA\portal-task-state\MOSAIC_LIVE_CORE_CHALLENGE_20261016_V1_*`. Individual case records are appended and flushed immediately to preserve partial evidence on runtime loss. Preflight must record the exact clean Git head, benchmark case SHA, and serving model ID **before** any completions. This is NOT a hidden holdout or independent replication; the prompts and the possible rationales are openly specified and may resemble model training examples.

## Claim ceiling

A higher challenged accuracy does not prove actual independently trained Mosaic specialist value: the critic is scripted, and the same general-purpose model is serving as Core throughout. Any decrease on misleading cases measures susceptibility to bad argumentation. Equal performance can mean the Core was already correct or that criticism had no effect. Model identity is endpoint metadata, not checkpoint attestation. Positive results would justify a larger independent benchmark with genuine separately sourced specialist generations, comparable token budgets, protected evaluations and measured VRAM/latency.

This test must **not** interfere with an active R13 optimizer, PreActive's restoration controller, or any other GPU owner. If PreActive is unavailable, report a live-model HOLD rather than replacing it with a mock.
