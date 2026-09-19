# MOSAIC P1A — Deterministic Continuity Protocol Harness V1

Status: **protocol harness / synthetic / not P1 qualification**

This artifact makes the predeclared P1 comparison executable without using a model, accelerator, training run, or hidden semantic grader.

It covers the four frozen conditions:

- RESET
- MOSAIC STATE
- FULL HISTORY
- ORDINARY RETRIEVAL

and the eight minimum task families:

- referent preservation;
- correction;
- supersession/currentness;
- decision finality;
- authority change;
- user preference;
- project status;
- artifact/work-product handoff.

The harness verifies that the state interface can preserve and resolve these synthetic state transitions, that superseded values do not become current, and that all four conditions use the same case set.

It measures:
- final deterministic accuracy;
- correction burden;
- stale-state errors;
- encoded payload bytes.

It deliberately does not claim to measure:
- actual accelerator residency;
- actual Specialist-A unload before Specialist-B execution;
- swap latency;
- model encode/decode latency;
- model reasoning quality;
- clean-run repeatability with independent model processes.

The ordinary-retrieval control is allowed to match or beat the synthetic Mosaic state. No special credit is awarded merely for reproducing retrieval.

The receipt hard-codes:

`p1_pass_eligible = false`

because the frozen P1 contract requires actual accelerator unload and a second clean run.

`DETERMINISTIC_PROTOCOL_HARNESS_ONLY != MOSAIC_P1_PASS`
