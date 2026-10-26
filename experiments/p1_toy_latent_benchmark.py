"""P0-NEURAL-COUNT-01: actual learned tensor swap vs bounded retrieval.

Only a narrow synthetic capability test. Not a pretrained-LLM P1 proof.
Train and evaluation use distinct process entrypoints; heldout never tunes weights.
Dependencies: torch, safetensors. Run in an isolated experiment directory.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import gc
import weakref
from math import comb

import torch
from safetensors.torch import load_file, save as serialize_safetensors


THRESHOLDS = (6, 7, 8, 9, 10, 11, 12, 13)
TRAIN_COUNT, DEV_COUNT, HELDOUT_COUNT = 512, 64, 8
EPOCHS, BATCH, SEED = 100, 128, 20261026


def model(*widths: int) -> torch.nn.Module:
    layers: list[torch.nn.Module] = []
    for a, b in zip(widths[:-2], widths[1:-1]):
        layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
    layers += [torch.nn.Linear(widths[-2], widths[-1])]
    return torch.nn.Sequential(*layers)


def build_models() -> tuple[torch.nn.Module, ...]:
    return (model(16, 64, 32, 8), model(16, 64, 64, 2),
            model(16, 64, 64, 2), model(24, 64, 64, 2),
            model(8, 64, 64, 2))


def episodes() -> dict[str, list[dict[str, int]]]:
    rng = random.Random(SEED)
    used: set[int] = set()
    result: dict[str, list[dict[str, int]]] = {}
    for split, n_per_family in (("train", TRAIN_COUNT), ("dev", DEV_COUNT),
                                ("run_a", HELDOUT_COUNT), ("run_b", HELDOUT_COUNT)):
        rows: list[dict[str, int]] = []
        for family, threshold in enumerate(THRESHOLDS):
            assert n_per_family % 2 == 0
            for label in (0, 1):
                for _ in range(n_per_family // 2):
                    for _attempt in range(100000):
                        count = rng.choice((threshold - 2, threshold - 1)
                                           if label == 0 else (threshold, threshold + 1))
                        bits = rng.sample(range(16), count)
                        mask = sum(1 << bit for bit in bits)
                        if mask not in used:
                            used.add(mask)
                            rows.append({"mask": mask, "family": family, "label": label})
                            break
                    else:
                        raise RuntimeError("Unique-mask balanced sampler exhausted")
        rng.shuffle(rows)
        result[split] = rows
    assert len(used) == sum(map(len, result.values()))
    return result


def features(rows: list[dict[str, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bits = torch.tensor([[float((row["mask"] >> j) & 1) for j in range(16)]
                         for row in rows], dtype=torch.float32)
    family = torch.nn.functional.one_hot(
        torch.tensor([row["family"] for row in rows]), num_classes=8).float()
    label = torch.tensor([row["label"] for row in rows], dtype=torch.long)
    return bits, family, label


def dataset_sha(data: dict[str, list[dict[str, int]]]) -> str:
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def frozen_root() -> Path:
    root = Path(os.environ["MOSAIC_LATENT_ROOT"]).resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("Experiment root missing")
    child = root / "toy_count_v1"
    if child.is_symlink() or (child.exists() and not child.is_dir()):
        raise RuntimeError("Unsafe experiment output directory")
    return child


def write_once_or_match(path: Path, payload: bytes) -> None:
    """Never follow an existing output symlink or overwrite a prior receipt."""
    parent = frozen_root()
    if path.parent != parent or not parent.is_dir() or parent.is_symlink():
        raise RuntimeError("Output parent differs from bounded experiment directory")
    if path.is_symlink():
        raise RuntimeError("Output path is a symlink")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        prior = path.read_bytes()
        # Early Windows receipts were written in text mode (CRLF). A replay
        # compares canonical UTF-8 JSON while keeping those originals intact.
        if path.suffix == ".json":
            prior = prior.replace(b"\r\n", b"\n")
        if path.is_symlink() or prior != payload:
            raise RuntimeError(f"Existing output does not match: {path.name}")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train() -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    root = frozen_root()
    root.mkdir(parents=True, exist_ok=False)
    data = episodes()
    initial_fingerprint = dataset_sha(data)
    source, target, retrieval, full, reset = build_models()
    net = (source, target, retrieval, full, reset)
    optimizer = torch.optim.AdamW(
        [parameter for member in net for parameter in member.parameters()],
        lr=0.003, weight_decay=0.0001)
    raw, query, truth = features(data["train"])
    start = time.monotonic()
    for epoch in range(EPOCHS):
        indices = torch.randperm(len(raw))
        for subset in indices.split(BATCH):
            bits, fam, label = raw[subset], query[subset], truth[subset]
            logits_state = target(torch.cat((source(bits), fam), dim=1))
            logits_retrieval = retrieval(torch.cat((bits[:, :8], fam), dim=1))
            logits_full = full(torch.cat((bits, fam), dim=1))
            logits_reset = reset(fam)
            loss = sum(torch.nn.functional.cross_entropy(pred, label) for pred in
                       (logits_state, logits_retrieval, logits_full, logits_reset))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch in (0, 24, 49, 74, EPOCHS - 1):
            print(f"TRAIN_EPOCH={epoch+1} LOSS={float(loss):.5f}", flush=True)
    filenames = ("source.safetensors", "target.safetensors",
                 "retrieval.safetensors", "full.safetensors", "reset.safetensors")
    for member, filename in zip(net, filenames):
        write_once_or_match(root / filename, serialize_safetensors({
            k: v.detach().contiguous().cpu()
            for k, v in member.state_dict().items()}))
    manifest = {"benchmark": "P0-NEURAL-COUNT-01", "seed": SEED, "epochs": EPOCHS,
                "train_count": len(data["train"]), "dev_count": len(data["dev"]),
                "dataset_sha256": initial_fingerprint,
                "training_seconds": round(time.monotonic() - start, 3),
                "weights": {n: file_sha(root / n) for n in filenames}}
    write_once_or_match(root / "train_manifest.json",
                        json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8"))
    print("TRAIN_COMPLETE " + json.dumps(manifest, sort_keys=True), flush=True)


def load_weights(net: torch.nn.Module, path: Path) -> torch.nn.Module:
    net.load_state_dict(load_file(str(path)), strict=True)
    return net.eval()


def verify_manifest(root: Path, data: dict[str, list[dict[str, int]]]) -> dict:
    manifest = json.loads((root / "train_manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset_sha256"] != dataset_sha(data):
        raise RuntimeError("Dataset fingerprint mismatch")
    for name, expected in manifest["weights"].items():
        if file_sha(root / name) != expected:
            raise RuntimeError(f"Weight checksum mismatch: {name}")
    return manifest


def emit_source_state() -> None:
    torch.set_num_threads(4)
    root = frozen_root()
    data = episodes()
    verify_manifest(root, data)
    source = load_weights(model(16, 64, 32, 8), root / "source.safetensors")
    for name in ("run_a", "run_b"):
        raw, family, _ = features(data[name])
        with torch.inference_mode():
            latent = source(raw)
        if latent.shape != (64, 8) or not torch.isfinite(latent).all():
            raise RuntimeError("Non-finite or malformed continuous source state")
        path = root / f"{name}_state.safetensors"
        write_once_or_match(path, serialize_safetensors({
            "continuous_state": latent.contiguous(),
            "query_onehot": family.contiguous()}))
        print("SOURCE_EMITTED " + json.dumps({"run": name, "path_sha256": file_sha(path),
            "latent_min": float(latent.min()), "latent_max": float(latent.max()),
            "raw_records_persisted": False}, sort_keys=True), flush=True)


def target_only() -> None:
    torch.set_num_threads(4)
    root = frozen_root()
    target = load_weights(model(16, 64, 64, 2), root / "target.safetensors")
    for name in ("run_a", "run_b"):
        values = load_file(str(root / f"{name}_state.safetensors"))
        if set(values) != {"continuous_state", "query_onehot"}:
            raise RuntimeError("Unapproved state channel to target")
        with torch.inference_mode():
            state = values["continuous_state"]
            query = values["query_onehot"]
            logits = target(torch.cat((state, query), dim=1))
            predictions = logits.argmax(dim=1).tolist()
            zeros = target(torch.cat((torch.zeros_like(state), query), dim=1))
            rolled = target(torch.cat((torch.roll(state, 1, dims=0), query), dim=1))
        write_once_or_match(root / f"{name}_neural_predictions.json",
                            json.dumps(predictions).encode("utf-8"))
        write_once_or_match(root / f"{name}_ablations.json", json.dumps({
            "zero_state": zeros.argmax(dim=1).tolist(),
            "rolled_state": rolled.argmax(dim=1).tolist(),
        }).encode("utf-8"))
        print("TARGET_PREDICTED " + json.dumps({"run": name, "count": len(predictions),
            "state_sha256": file_sha(root / f"{name}_state.safetensors")},
            sort_keys=True), flush=True)


def paired_one_sided_p(mosaic: list[bool], baseline: list[bool]) -> tuple[int, int, float]:
    wins = sum(a and not b for a, b in zip(mosaic, baseline))
    losses = sum(b and not a for a, b in zip(mosaic, baseline))
    discordant = wins + losses
    p = sum(comb(discordant, i) for i in range(wins, discordant + 1)) / (2 ** discordant)
    return wins, losses, p


def compare() -> None:
    torch.set_num_threads(4)
    root = frozen_root()
    data = episodes()
    manifest = verify_manifest(root, data)
    # Read back both genuine learned checkpoint computations instead of trusting
    # user-replaceable neural_predictions.json or source_latent.safetensors.
    source = load_weights(model(16, 64, 32, 8), root / "source.safetensors")
    for name in ("run_a", "run_b"):
        raw, query, _ = features(data[name])
        receipt = load_file(str(root / f"{name}_state.safetensors"))
        if set(receipt) != {"continuous_state", "query_onehot"}:
            raise RuntimeError(f"Unapproved state keys in {name}")
        with torch.inference_mode():
            expected = source(raw)
        if not torch.equal(receipt["continuous_state"], expected):
            raise RuntimeError(f"Source state does not match checkpoint + dataset: {name}")
        if not torch.equal(receipt["query_onehot"], query):
            raise RuntimeError(f"Target query does not match frozen case: {name}")
    source_ref = weakref.ref(source)
    del source
    gc.collect()
    if source_ref() is not None:
        raise RuntimeError("Source model still resident before target verification")
    target = load_weights(model(16, 64, 64, 2), root / "target.safetensors")
    for name in ("run_a", "run_b"):
        receipt = load_file(str(root / f"{name}_state.safetensors"))
        with torch.inference_mode():
            expected = target(torch.cat((receipt["continuous_state"],
                                         receipt["query_onehot"]), dim=1)).argmax(1).tolist()
        declared = json.loads((root / f"{name}_neural_predictions.json").read_text())
        if expected != declared:
            raise RuntimeError(f"Target predictions fail checkpoint replay: {name}")
    del target
    gc.collect()
    retrieval = load_weights(model(16, 64, 64, 2), root / "retrieval.safetensors")
    full = load_weights(model(24, 64, 64, 2), root / "full.safetensors")
    reset = load_weights(model(8, 64, 64, 2), root / "reset.safetensors")
    results = []
    for name in ("run_a", "run_b"):
        raw, query, label = features(data[name])
        neural = json.loads((root / f"{name}_neural_predictions.json").read_text())
        ablations = json.loads((root / f"{name}_ablations.json").read_text())
        if len(neural) != len(label) or any(x not in (0, 1) for x in neural):
            raise RuntimeError("Invalid neural prediction count or label")
        if set(ablations) != {"zero_state", "rolled_state"} or any(
            len(ablations[condition]) != len(label) or any(
                answer not in (0, 1) for answer in ablations[condition])
            for condition in ("zero_state", "rolled_state")
        ):
            raise RuntimeError("Invalid ablation prediction count or label")
        with torch.inference_mode():
            retr = retrieval(torch.cat((raw[:, :8], query), dim=1)).argmax(1).tolist()
            history = full(torch.cat((raw, query), dim=1)).argmax(1).tolist()
            blank = reset(query).argmax(1).tolist()
        truth = label.tolist()
        neural_ok = [n == y for n, y in zip(neural, truth)]
        retrieval_ok = [n == y for n, y in zip(retr, truth)]
        wins, losses, p = paired_one_sided_p(neural_ok, retrieval_ok)
        accuracies = {"MOSAIC_LEARNED_STATE": sum(neural_ok) / 64,
                      "ORDINARY_RETRIEVAL_8_RECORDS": sum(retrieval_ok) / 64,
                      "FULL_HISTORY": sum(n == y for n, y in zip(history, truth)) / 64,
                      "RESET": sum(n == y for n, y in zip(blank, truth)) / 64,
                      "POST_HOC_ZERO_STATE": sum(n == y for n, y in zip(
                          ablations["zero_state"], truth)) / 64,
                      "POST_HOC_ROLLED_STATE": sum(n == y for n, y in zip(
                          ablations["rolled_state"], truth)) / 64}
        pass_gate = (accuracies["MOSAIC_LEARNED_STATE"] >= 0.8
                     and accuracies["MOSAIC_LEARNED_STATE"] - accuracies[
                         "ORDINARY_RETRIEVAL_8_RECORDS"] >= 0.05
                     and p < 0.05)
        results.append({"run": name, "accuracies": accuracies, "wins": wins,
                        "losses": losses, "one_sided_exact_p": p, "run_gate_pass": pass_gate,
                        "case_ids": [hashlib.sha256(f'{row["mask"]}:{row["family"]}'.encode()).hexdigest()
                                     for row in data[name]],
                        "truth": truth, "neural_predictions": neural,
                        "post_hoc_ablations": ablations,
                        "retrieval_predictions": retr, "history_predictions": history,
                        "reset_predictions": blank})
    final = {"status": "NARROW_TOY_PASS" if all(row["run_gate_pass"] for row in results)
             else "HOLD_RETRIEVAL_ADVANTAGE_NOT_DEMONSTRATED",
             "not_pretrained_llm_p1": True, "eight_record_retrieval_only": True,
             "manifest": manifest, "runs": results}
    write_once_or_match(root / "results.json",
                        json.dumps(final, sort_keys=True, indent=2).encode("utf-8"))
    print("FINAL_METRICS " + json.dumps({"status": final["status"],
        "runs": [{k: row[k] for k in ("run", "accuracies", "wins", "losses",
                                      "one_sided_exact_p", "run_gate_pass")}
                 for row in results]}, sort_keys=True), flush=True)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in (
        "train", "emit", "target", "compare", "evaluate"):
        raise SystemExit("Usage: python p1_toy_latent_benchmark.py train|emit|target|compare|evaluate")
    stage = sys.argv[1]
    if stage == "train":
        train()
    elif stage == "emit":
        emit_source_state()
    elif stage == "target":
        target_only()
    elif stage == "compare":
        compare()
    else:
        for action in ("emit", "target", "compare"):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), action],
                           check=True)


if __name__ == "__main__":
    main()
