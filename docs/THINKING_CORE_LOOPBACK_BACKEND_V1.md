# Thinking Core HTTP transport — V1

This source-controlled candidate connects Mosaic's neural thinking Core to an already-running local Qwen-style HTTP endpoint. It does not create a trained Mosaic Core or instantiate an inference server.

`LoopbackCoreInference` is supplied to `NeuralThinkingCore` as its generation function. Core THINK and Core RECONSIDER are actual inference calls when a service is available. The specialist and independent verification protocols remain separate.

The endpoint is fixed to loopback 127.0.0.1 (default port 18081). There are no external host options. Proxies and redirects are disabled. An advertised `qwen3.5-4b-local` model ID must match before generation. The model's self-reported ID is **not cryptographic proof of its checkpoint or adapter lineage**.

Only a system/user message pair, an empty tool list, and non-streaming completion are permitted. The client rejects HTTP failures, extra choices, tool calls, incomplete output, invalid roles, large requests/responses and malformed JSON. The existing PreActive server caps its own generation at 256 new tokens and ignores any suggested client-side token cap. Strict reasoning JSON parsing remains in `NeuralThinkingCore`.

Tests exercise real HTTP requests against an isolated CPU-only loopback fixture (not the neural model), including Core THINK -> specialist objection -> Core RECONSIDER. Such passing tests qualify the transport and protocol only; they do not qualify neural reasoning, specialist quality, P0/P1 residency/continuity, or deployment.

No live PreActive request is made while Vera R13 training temporarily occupies Lappy's GPU. This adapter never starts or stops a service, switches weights, or authorizes protected effects.

Usage with an already-running local service:

```python
from runtime.local_neural_backend import LoopbackCoreInference
from runtime.neural_core import NeuralThinkingCore
from runtime.thinking_core import CoreDeliberationEngine, DeliberationTask

core = NeuralThinkingCore(LoopbackCoreInference())
task = DeliberationTask("test-001", "supported", "Is this proposition supported?")
receipt = CoreDeliberationEngine().deliberate(task, core, [])
# Without independent evidence, the result remains UNVERIFIED_ADVISORY.
```

The next verification subject is a **predeclared live-model evaluation**, after the running training controller restores its serving process. Any production use requires independent identity, model provenance and authority qualification.
