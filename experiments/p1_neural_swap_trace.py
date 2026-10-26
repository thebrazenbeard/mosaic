"""Real source-model eviction and target-model continuous-prefix injection probe.

This is NOT a trained-state P1 result. It qualifies tensor plumbing only.
Loads two pinned, independently stored HF neural models with no remote code.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
import weakref

import torch
from safetensors.torch import load_file, save as serialize_safetensors
from transformers import AutoModelForCausalLM, AutoTokenizer


EXPECTED_WEIGHTS = {
    "SmolLM2-135M-Instruct": "5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c",
    "SmolLM2-360M-Instruct": "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e",
}
EXPECTED_RUNTIME_ASSETS = {
    "SmolLM2-135M-Instruct": {
        "config.json": "8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a",
        "generation_config.json": "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013",
        "special_tokens_map.json": "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
        "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        "tokenizer_config.json": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
        "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
    },
    "SmolLM2-360M-Instruct": {
        "config.json": "224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee",
        "generation_config.json": "87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013",
        "special_tokens_map.json": "2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52",
        "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
        "tokenizer_config.json": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
        "vocab.json": "82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79",
    },
}


def weight_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    torch.set_num_threads(4)
    root = Path(os.environ["MOSAIC_LATENT_ROOT"]).resolve(strict=True)
    source = root / "models" / "SmolLM2-135M-Instruct"
    target = root / "models" / "SmolLM2-360M-Instruct"
    if (source == target or not source.is_dir() or not target.is_dir()
            or (root / "models").is_symlink() or source.is_symlink()
            or target.is_symlink() or source.samefile(target)):
        raise RuntimeError("Two separate staged weight sets are required")
    for model_path in (source, target):
        weights = model_path / "model.safetensors"
        if weights.is_symlink() or weight_digest(weights) != EXPECTED_WEIGHTS[model_path.name]:
            raise RuntimeError(f"Weight identity mismatch: {model_path.name}")
        for filename, expected in EXPECTED_RUNTIME_ASSETS[model_path.name].items():
            asset = model_path / filename
            if asset.is_symlink() or weight_digest(asset) != expected:
                raise RuntimeError(f"Runtime config/tokenizer identity mismatch: {model_path.name}/{filename}")
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
    if state_path.is_symlink():
        raise RuntimeError("Latent output symlink refused")
    serialized = serialize_safetensors({"state": latent})
    try:
        with state_path.open("xb") as file:
            file.write(serialized)
    except FileExistsError:
        if state_path.is_symlink() or state_path.read_bytes() != serialized:
            raise RuntimeError("Prior latent receipt differs; refusing overwrite")
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
        # Equal-length controls isolate the source-dependent tensor from the
        # positional effect of adding any prefix token.
        zero_state = target_model(inputs_embeds=torch.cat((
            torch.zeros_like(continuous_prefix), query_embed), dim=1)).logits[0, -1]
        with_state = target_model(
            inputs_embeds=torch.cat((continuous_prefix, query_embed), dim=1)).logits[0, -1]
    delta = float((zero_state - with_state).abs().max())
    if not math.isfinite(delta) or delta <= 0 or not torch.isfinite(
            with_state).all() or not torch.isfinite(zero_state).all():
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
        "zero_prefix_equal_sequence_length_control": True,
        "source_stage_seconds": round(source_evicted - begin, 3),
        "total_seconds": round(time.monotonic() - begin, 3),
        "bridge_was_trained": False,
        "comparative_advantage_verified": False,
    }
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
