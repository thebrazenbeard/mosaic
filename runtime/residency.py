from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable, Mapping


class ResidencyModelError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SpecialistResidency:
    specialist_id: str
    package_bytes: int
    weight_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    workspace_bytes: int

    def gpu_bytes(self) -> int:
        values = (
            self.weight_bytes,
            self.kv_cache_bytes,
            self.activation_bytes,
            self.workspace_bytes,
        )
        if not self.specialist_id.strip():
            raise ResidencyModelError("specialist_id is required")
        if any(value < 0 for value in values):
            raise ResidencyModelError("residency byte counts must be non-negative")
        if self.package_bytes < 0:
            raise ResidencyModelError("package_bytes must be non-negative")
        return sum(values)


@dataclass(frozen=True, slots=True)
class ResidencyEnvelope:
    vram_ceiling_bytes: int
    boot_core_gpu_bytes: int
    specialists: tuple[SpecialistResidency, ...]

    def validate(self) -> None:
        if self.vram_ceiling_bytes <= 0:
            raise ResidencyModelError("vram_ceiling_bytes must be positive")
        if self.boot_core_gpu_bytes < 0:
            raise ResidencyModelError("boot_core_gpu_bytes must be non-negative")
        ids = [item.specialist_id for item in self.specialists]
        if len(ids) != len(set(ids)):
            raise ResidencyModelError("specialist_id values must be unique")
        for item in self.specialists:
            item.gpu_bytes()


def canonical_json_sha256(value: Mapping[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_swap_sequence(
    envelope: ResidencyEnvelope,
    sequence: Iterable[str],
) -> dict[str, object]:
    """Evaluate an analytical single-specialist residency sequence.

    This models declared byte budgets only. It does not inspect CUDA/device
    state and therefore cannot establish Mosaic P0 hardware qualification.
    """

    envelope.validate()
    by_id = {item.specialist_id: item for item in envelope.specialists}
    ordered = tuple(sequence)
    if not ordered:
        raise ResidencyModelError("swap sequence must not be empty")

    steps: list[dict[str, object]] = []
    max_resident = 0
    for index, specialist_id in enumerate(ordered):
        specialist = by_id.get(specialist_id)
        if specialist is None:
            raise ResidencyModelError(f"unknown specialist in sequence: {specialist_id}")
        modeled = envelope.boot_core_gpu_bytes + specialist.gpu_bytes()
        max_resident = max(max_resident, modeled)
        steps.append(
            {
                "index": index,
                "specialist_id": specialist_id,
                "modeled_resident_bytes": modeled,
                "within_declared_ceiling": modeled <= envelope.vram_ceiling_bytes,
            }
        )

    total_pool = envelope.boot_core_gpu_bytes + sum(
        item.gpu_bytes() for item in envelope.specialists
    )
    input_subject = {
        "vram_ceiling_bytes": envelope.vram_ceiling_bytes,
        "boot_core_gpu_bytes": envelope.boot_core_gpu_bytes,
        "specialists": [
            {
                "specialist_id": item.specialist_id,
                "package_bytes": item.package_bytes,
                "weight_bytes": item.weight_bytes,
                "kv_cache_bytes": item.kv_cache_bytes,
                "activation_bytes": item.activation_bytes,
                "workspace_bytes": item.workspace_bytes,
            }
            for item in envelope.specialists
        ],
        "sequence": list(ordered),
    }

    return {
        "schema_id": "MOSAIC_ANALYTICAL_RESIDENCY_RECEIPT_V1",
        "qualification": "ANALYTICAL_FEASIBILITY_ONLY",
        "hardware_measurement_performed": False,
        "input_sha256": canonical_json_sha256(input_subject),
        "vram_ceiling_bytes": envelope.vram_ceiling_bytes,
        "total_declared_gpu_parameter_and_runtime_pool_bytes": total_pool,
        "modeled_peak_resident_bytes": max_resident,
        "logical_pool_exceeds_vram_ceiling": total_pool > envelope.vram_ceiling_bytes,
        "all_modeled_steps_within_ceiling": all(
            bool(step["within_declared_ceiling"]) for step in steps
        ),
        "steps": steps,
        "claim_ceiling": (
            "Analytical byte accounting only; does not establish measured peak VRAM, "
            "actual unload behavior, swap latency, ten-cycle stability, or Mosaic P0 PASS."
        ),
    }
