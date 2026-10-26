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


MODEL_IDS = (
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
)


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
    for model_id in MODEL_IDS:
        info = model_info(model_id, files_metadata=True)
        if not info.sha or len(info.sha) != 40:
            raise ValueError(f"Unpinned model revision for {model_id}")
        destination = root / "models" / model_id.split("/")[-1]
        destination.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            revision=info.sha,
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
