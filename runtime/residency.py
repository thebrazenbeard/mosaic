from __future__ import annotations

from dataclasses import asdict, dataclass
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
            self.package_bytes,
            self.weight_bytes,
            self.kv_cache_bytes,
            self.activation_bytes,
            self.workspace_bytes,
        )
        if not isinstance(self.specialist_id, str) or not self.specialist_id.strip():
            raise ResidencyModelError("specialist_id is required")
        try:
            self.specialist_id.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ResidencyModelError("specialist_id must be valid UTF-8 text") from exc
        if any(type(value) is not int for value in values):
            raise ResidencyModelError("residency byte counts must be integers")
        if any(value < 0 for value in values):
            raise ResidencyModelError("residency byte counts must be non-negative")
        return sum(values[1:])


@dataclass(frozen=True, slots=True)
class ResidencyEnvelope:
    vram_ceiling_bytes: int
    boot_core_gpu_bytes: int
    specialists: tuple[SpecialistResidency, ...]

    def validate(self) -> None:
        if type(self.vram_ceiling_bytes) is not int:
            raise ResidencyModelError("vram_ceiling_bytes must be an integer")
        if self.vram_ceiling_bytes <= 0:
            raise ResidencyModelError("vram_ceiling_bytes must be positive")
        if type(self.boot_core_gpu_bytes) is not int:
            raise ResidencyModelError("boot_core_gpu_bytes must be an integer")
        if self.boot_core_gpu_bytes < 0:
            raise ResidencyModelError("boot_core_gpu_bytes must be non-negative")
        for item in self.specialists:
            if not isinstance(item, SpecialistResidency):
                raise ResidencyModelError("specialists must contain specialist records")
            item.gpu_bytes()
        # Validate specialist_id shape before constructing a set; malformed
        # unhashable IDs must fail with the domain error, not a raw TypeError.
        ids = [item.specialist_id for item in self.specialists]
        if len(ids) != len(set(ids)):
            raise ResidencyModelError("specialist_id values must be unique")


def canonical_json_sha256(value: Mapping[str, object]) -> str:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ResidencyModelError("receipt must contain valid UTF-8 text") from exc
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
        if not isinstance(specialist_id, str) or not specialist_id.strip():
            raise ResidencyModelError("sequence items must be specialist IDs")
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
    # Runtime buffers are not learned parameters. This strict lower bound on
    # total learned weights deliberately excludes Boot/Core, whose GPU budget
    # combines weights and other allocations without a separate breakdown.
    specialist_weights = sum(item.weight_bytes for item in envelope.specialists)
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
        "total_declared_specialist_weight_bytes": specialist_weights,
        "specialist_weight_pool_exceeds_vram_ceiling": (
            specialist_weights > envelope.vram_ceiling_bytes
        ),
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


def evaluate_transition_headroom(
    envelope: ResidencyEnvelope,
    sequence: Iterable[str],
    *,
    policy: str = "evict_then_load",
    persistent_overhead_bytes: int = 0,
) -> dict[str, object]:
    """Model transient peaks between specialists, with declared inputs only.

    ``evict_then_load`` requires full release of the previous specialist before
    loading the next. ``load_then_evict`` conservatively includes both whole
    declared specialist footprints during the swap. Neither behavior has been
    observed or enforced by this analytical function. This intentionally does
    not estimate latency, physical fragmentation, CPU RAM, or transfer traffic.
    """

    envelope.validate()
    if policy not in ("evict_then_load", "load_then_evict"):
        raise ResidencyModelError("unknown swap policy")
    if type(persistent_overhead_bytes) is not int or persistent_overhead_bytes < 0:
        raise ResidencyModelError("persistent_overhead_bytes must be a non-negative integer")
    ordered = tuple(sequence)
    if not ordered:
        raise ResidencyModelError("swap sequence must not be empty")
    if any(not isinstance(item, str) for item in ordered):
        raise ResidencyModelError("specialist ID in sequence must be a string")

    specialists = {item.specialist_id: item for item in envelope.specialists}
    if any(item not in specialists for item in ordered):
        raise ResidencyModelError("unknown specialist in sequence")

    constant = envelope.boot_core_gpu_bytes + persistent_overhead_bytes
    previous: str | None = None
    peak = constant
    transitions: list[dict[str, object]] = []
    for index, item in enumerate(ordered):
        current_bytes = specialists[item].gpu_bytes()
        if previous is None:
            kind = "initial_load"
            transition_peak = constant + current_bytes
        elif previous == item:
            kind = "reuse"
            transition_peak = constant + current_bytes
        else:
            kind = "swap"
            previous_bytes = specialists[previous].gpu_bytes()
            transition_peak = constant + current_bytes
            if policy == "load_then_evict":
                transition_peak += previous_bytes
        peak = max(peak, transition_peak)
        transitions.append({
            "index": index,
            "from_specialist_id": previous,
            "to_specialist_id": item,
            "kind": kind,
            "modeled_transition_peak_bytes": transition_peak,
            "within_declared_ceiling": transition_peak <= envelope.vram_ceiling_bytes,
        })
        previous = item

    subject = {
        "envelope": {
            "vram_ceiling_bytes": envelope.vram_ceiling_bytes,
            "boot_core_gpu_bytes": envelope.boot_core_gpu_bytes,
            "specialists": [asdict(item) for item in envelope.specialists],
        },
        "sequence": list(ordered),
        "policy": policy,
        "persistent_overhead_bytes": persistent_overhead_bytes,
    }
    return {
        "schema_id": "MOSAIC_ANALYTICAL_SWAP_TRANSITION_RECEIPT_V1",
        "qualification": "ANALYTICAL_TRANSITION_PREFLIGHT_ONLY",
        "hardware_measurement_performed": False,
        "input_sha256": canonical_json_sha256(subject),
        "swap_policy_assumed_not_observed": policy,
        "persistent_overhead_bytes": persistent_overhead_bytes,
        "vram_ceiling_bytes": envelope.vram_ceiling_bytes,
        "modeled_peak_transition_bytes": peak,
        "headroom_bytes": envelope.vram_ceiling_bytes - peak,
        "all_modeled_transitions_fit": all(
            bool(row["within_declared_ceiling"]) for row in transitions
        ) and constant <= envelope.vram_ceiling_bytes,
        "transitions": transitions,
        "claim_ceiling": (
            "Declared-byte transition arithmetic only; swap order, device releases, "
            "driver allocations, CPU RAM, transfer costs, latency and P0 remain unverified."
        ),
    }
