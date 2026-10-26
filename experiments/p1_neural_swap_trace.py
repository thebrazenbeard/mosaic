"""Real source-model eviction and target-model continuous-prefix injection probe.

This is NOT a trained-state P1 result. It qualifies tensor plumbing only.
Loads two pinned, independently stored HF neural models with no remote code.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import time
import weakref

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    torch.set_num_threads(4)
    root = Path(os.environ["MOSAIC_LATENT_ROOT"]).resolve(strict=True)
    source = root / "models" / "SmolLM2-135M-Instruct"
    target = root / "models" / "SmolLM2-360M-Instruct"
    if source == target or not source.exists() or not target.exists():
        raise RuntimeError("Two separate staged weight sets are required")
    begin = time.monotonic()
    source_model = AutoModelForCausalLM.from_pretrained(
        source, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").eval()
    source_tokenizer = AutoTokenizer.from_pretrained(
        source, local_files_only=True, trust_remote_code=False)
    source_ids = source_tokenizer(
        "The hidden rule is that blue means the gate is open, and red means shut.",
        return_tensors="pt").input_ids
    with torch.inference_mode():
        source_output = source_model(source_ids, output_hidden_states=True)
        latent = source_output.hidden_states[-1][0, -1].float().contiguous().cpu()
    state_path = root / "source_latent.safetensors"
    save_file({"state": latent}, str(state_path))
    source_ref = weakref.ref(source_model)
    source_hidden_dim = latent.numel()
    del source_model, source_tokenizer, source_ids, source_output, latent
    gc.collect()
    if source_ref() is not None:
        raise RuntimeError("Source model still referenced at swap boundary")
    source_evicted = time.monotonic()
    target_model = AutoModelForCausalLM.from_pretrained(
        target, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager").eval()
    target_tokenizer = AutoTokenizer.from_pretrained(
        target, local_files_only=True, trust_remote_code=False)
    query_ids = target_tokenizer("The gate is blue. The gate is", return_tensors="pt").input_ids
    query_embed = target_model.get_input_embeddings()(query_ids)
    state = load_file(str(state_path))["state"]
    torch.manual_seed(20261026)
    bridge = torch.nn.Linear(source_hidden_dim, query_embed.shape[-1], bias=False)
    continuous_prefix = bridge(state[None, :]).unsqueeze(1)
    with torch.inference_mode():
        no_state = target_model(inputs_embeds=query_embed).logits[0, -1]
        with_state = target_model(
            inputs_embeds=torch.cat((continuous_prefix, query_embed), dim=1)).logits[0, -1]
    delta = float((no_state - with_state).abs().max())
    if delta <= 0 or not torch.isfinite(with_state).all():
        raise RuntimeError("Continuous prefix failed to affect target logits")
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    receipt = {
        "kind": "real_neural_tensor_plumbing_untrained",
        "source_hidden_dim": source_hidden_dim,
        "target_hidden_dim": int(query_embed.shape[-1]),
        "source_evicted_before_target_load": True,
        "latent_safetensors_sha256": digest,
        "state_is_continuous_float32": str(state.dtype) == "torch.float32",
        "state_text_prompted_to_target": False,
        "target_used_inputs_embeds": True,
        "prefix_changed_logits_max_abs": delta,
        "source_stage_seconds": round(source_evicted - begin, 3),
        "total_seconds": round(time.monotonic() - begin, 3),
        "bridge_was_trained": False,
        "comparative_advantage_verified": False,
    }
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
