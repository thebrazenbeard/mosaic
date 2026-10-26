"""Stage two immutable small language models for a cross-specialist latent probe.

Experimental prerequisite only: downloading weights proves neither learning nor P1.
The caller chooses an isolated cache through MOSAIC_LATENT_ROOT.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import model_info, snapshot_download


MODEL_REVISIONS = {
    "HuggingFaceTB/SmolLM2-135M-Instruct": "12fd25f77366fa6b3b4b768ec3050bf629380bac",
    "HuggingFaceTB/SmolLM2-360M-Instruct": "a10cc1512eabd3dde888204e902eca88bddb4951",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root_value = os.environ.get("MOSAIC_LATENT_ROOT")
    if not root_value:
        raise SystemExit("MOSAIC_LATENT_ROOT must point to an isolated experiment directory")
    root = Path(root_value).resolve(strict=True)
    if not root.is_dir():
        raise SystemExit("MOSAIC_LATENT_ROOT is not a directory")
    records = []
    stage_dir = root / "models"
    if stage_dir.is_symlink():
        raise RuntimeError("Model staging directory must not be a symlink")
    for model_id, revision in MODEL_REVISIONS.items():
        info = model_info(model_id, revision=revision, files_metadata=True)
        if info.sha != revision:
            raise ValueError(f"Mismatched immutable model revision for {model_id}")
        destination = root / "models" / model_id.split("/")[-1]
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            raise RuntimeError(f"Unsafe model output directory: {model_id}")
        destination.mkdir(parents=True, exist_ok=True)
        if any(path.is_symlink() for path in destination.rglob("*")):
            raise RuntimeError(f"Existing symlink inside model staging directory: {model_id}")
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_dir=str(destination),
            allow_patterns=("*.json", "*.safetensors", "tokenizer.model"),
        )
        weights = sorted(destination.glob("*.safetensors"))
        if not weights:
            raise RuntimeError(f"Model weights not found for {model_id}")
        files = [
            {"name": path.name, "size_bytes": path.stat().st_size,
             "sha256": sha256_file(path)}
            for path in weights
        ]
        record = {"model_id": model_id, "revision": info.sha,
                  "destination": str(destination), "files": files}
        records.append(record)
        print("STAGED " + json.dumps(record, sort_keys=True), flush=True)
    print("MODEL_PAIR_READY " + json.dumps(records, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
