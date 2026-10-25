"""Mosaic V1 thinking-Core decision protocol: advice != verification != authority.

The learned/neural Core owns *reasoning*. Specialists challenge and supply
claims. A separate verifier may supply evidence. This module checks contract
boundaries and records conflicts; it is not a neural model or proof of intelligence.
No operation here executes a proposed protected effect.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
import json
from typing import Protocol, Sequence


class ProtocolViolation(ValueError):
    """Contract violation; do not continue with a fabricated decision."""


class DecisionStatus(str, Enum):
    UNVERIFIED_ADVISORY = "UNVERIFIED_ADVISORY"
    CONTESTED_ADVISORY = "CONTESTED_ADVISORY"
    HOLD_UNACKNOWLEDGED_OBJECTION = "HOLD_UNACKNOWLEDGED_OBJECTION"
    HOLD_UNVERIFIED_DISPUTE = "HOLD_UNVERIFIED_DISPUTE"
    BLOCKED_BY_EVIDENCE = "BLOCKED_BY_EVIDENCE"
    EVIDENCE_SUPPORTED_ADVISORY = "EVIDENCE_SUPPORTED_ADVISORY"
    PROPOSED_EFFECT_REQUIRES_AUTHORIZATION = "PROPOSED_EFFECT_REQUIRES_AUTHORIZATION"


@dataclass(frozen=True, slots=True)
class DeliberationTask:
    case_id: str
    proposition_id: str
    question: str
    consequential: bool = False
    protected_effect: bool = False


@dataclass(frozen=True, slots=True)
class CorePosition:
    proposition_id: str
    supports: bool
    reasoning: str
    acknowledged_objections: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SpecialistObjection:
    specialist_id: str
    proposition_id: str
    supports: bool
    rationale: str


@dataclass(frozen=True, slots=True)
class EvidenceVerdict:
    proposition_id: str
    supports: bool | None
    verifier_id: str
    evidence_ref: str
    rationale: str


class ThinkingCore(Protocol):
    """Must be implemented by the intelligent/neural Core, not by the router."""
    def think(self, task: DeliberationTask) -> CorePosition: ...
    def reconsider(
        self,
        task: DeliberationTask,
        prior: CorePosition,
        objections: tuple[SpecialistObjection, ...],
        evidence: EvidenceVerdict | None,
    ) -> CorePosition: ...


class Specialist(Protocol):
    specialist_id: str

    def review(
        self,
        task: DeliberationTask,
        core_position: CorePosition,
    ) -> SpecialistObjection: ...


class IndependentVerifier(Protocol):
    def examine(
        self,
        task: DeliberationTask,
        proposal: CorePosition,
        objections: tuple[SpecialistObjection, ...],
    ) -> EvidenceVerdict: ...


@dataclass(frozen=True, slots=True)
class DeliberationReceipt:
    task: DeliberationTask
    core_initial: CorePosition
    core_final: CorePosition
    objections: tuple[SpecialistObjection, ...]
    evidence: EvidenceVerdict | None
    status: DecisionStatus
    claim_verified: bool
    verified_proposition_is_false: bool
    effect_executed: bool
    claim_ceiling: str
    receipt_sha256: str


def _required(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolViolation(f"{label} must be a nonempty string")


def _identity(value: str, label: str) -> None:
    _required(value, label)
    if value != value.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ProtocolViolation(f"{label} is not a canonical identity")


def _stance(value: object, label: str, *, nullable: bool = False) -> None:
    if type(value) is not bool and not (nullable and value is None):
        raise ProtocolViolation(f"{label} must be an explicit boolean"
                                + (" or unknown" if nullable else ""))


def _task(task: DeliberationTask) -> None:
    _required(task.case_id, "case_id")
    _required(task.proposition_id, "proposition_id")
    _required(task.question, "question")
    _stance(task.consequential, "consequential")
    _stance(task.protected_effect, "protected_effect")


def _core(
    position: CorePosition, task: DeliberationTask,
    *, allowed_acknowledgments: set[str],
) -> None:
    if not isinstance(position, CorePosition):
        raise ProtocolViolation("Core did not return a CorePosition")
    if position.proposition_id != task.proposition_id:
        raise ProtocolViolation("Core proposition is not the task proposition")
    _stance(position.supports, "core stance")
    _required(position.reasoning, "core reasoning")
    acknowledgments = position.acknowledged_objections
    if type(acknowledgments) is not tuple:
        raise ProtocolViolation("Acknowledged objection IDs must be a tuple")
    for name in acknowledgments:
        _identity(name, "acknowledged objection ID")
    if len({name.casefold() for name in acknowledgments}) != len(acknowledgments):
        raise ProtocolViolation("Duplicate acknowledged objection ID")
    if not set(acknowledgments) <= allowed_acknowledgments:
        raise ProtocolViolation("Core acknowledged an unknown or unreviewed specialist")


def _objection(obj: SpecialistObjection, task: DeliberationTask, expected: str) -> None:
    if not isinstance(obj, SpecialistObjection):
        raise ProtocolViolation("Specialist did not return a SpecialistObjection")
    _identity(obj.specialist_id, "specialist ID")
    if obj.specialist_id != expected or obj.specialist_id.casefold() == "core":
        raise ProtocolViolation("Specialist identity mismatch or impersonation")
    if obj.proposition_id != task.proposition_id:
        raise ProtocolViolation("Specialist proposition does not match task")
    _stance(obj.supports, "specialist stance")
    _required(obj.rationale, "specialist rationale")


def _evidence(
    result: EvidenceVerdict,
    task: DeliberationTask,
    specialist_ids: set[str],
) -> None:
    if not isinstance(result, EvidenceVerdict):
        raise ProtocolViolation("Verifier did not return an EvidenceVerdict")
    if result.proposition_id != task.proposition_id:
        raise ProtocolViolation("Verifier proposition does not match task")
    _stance(result.supports, "verified stance", nullable=True)
    _identity(result.verifier_id, "verifier identity")
    if result.verifier_id.casefold() in {"core", *(s.casefold() for s in specialist_ids)}:
        raise ProtocolViolation("Independent verifier cannot be Core or a specialist")
    _required(result.rationale, "verifier rationale")
    if result.supports is not None:
        _required(result.evidence_ref, "independent evidence reference")
    elif result.evidence_ref:
        raise ProtocolViolation("Unknown verdict cannot assert verified evidence")


def _receipt_digest(fields: dict[str, object]) -> str:
    serialized = json.dumps(
        fields, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(serialized).hexdigest()


class CoreDeliberationEngine:
    """Boundary checks around neural reasoning, not an automated intelligence.

    The Core constructs and revises positions. Specialists can challenge them.
    Only external verifier evidence can verify a proposition. Protected effects
    remain unexecuted irrespective of Core, specialist, or verifier confidence.
    """

    def deliberate(
        self,
        task: DeliberationTask,
        core: ThinkingCore,
        specialists: Sequence[Specialist],
        verifier: IndependentVerifier | None = None,
    ) -> DeliberationReceipt:
        if not isinstance(task, DeliberationTask):
            raise ProtocolViolation("Task type mismatch")
        _task(task)
        # Reject malformed reviewer rosters before calling a neural inference
        # backend or any specialist callback. Neither is a safe preflight probe.
        reviewers = tuple(specialists)
        ids = [specialist.specialist_id for specialist in reviewers]
        for name in ids:
            _identity(name, "specialist ID")
            if name.casefold() == "core":
                raise ProtocolViolation("Specialist may not impersonate Core")
        if len({name.casefold() for name in ids}) != len(ids):
            raise ProtocolViolation("duplicate specialist identity")

        initial = core.think(task)
        _core(initial, task, allowed_acknowledgments=set())
        objections = tuple(
            specialist.review(task, initial) for specialist in reviewers
        )
        for obj, expected in zip(objections, ids, strict=True):
            _objection(obj, task, expected)

        evidence = None if verifier is None else verifier.examine(
            task, initial, objections
        )
        if evidence is not None:
            _evidence(evidence, task, set(ids))

        dissent = tuple(
            obj for obj in objections if obj.supports != initial.supports
        )
        independent_conflict = (
            evidence is not None and evidence.supports is not None
            and evidence.supports != initial.supports
        )
        needs_reconsideration = bool(dissent or independent_conflict)
        final = (
            core.reconsider(task, initial, objections, evidence)
            if needs_reconsideration else initial
        )
        _core(final, task, allowed_acknowledgments=set(ids))

        acknowledged = set(final.acknowledged_objections)
        missing = {obj.specialist_id for obj in dissent} - acknowledged
        evidence_known = evidence is not None and evidence.supports is not None
        aligns_with_evidence = evidence_known and final.supports == evidence.supports
        conflict_with_evidence = evidence_known and not aligns_with_evidence

        # A verified disagreement can veto the Core's position, but a
        # specialist cannot replace the Core as the origin of final reasoning.
        # An unsupported specialist dispute can be disclosed or held, never
        # upgraded to independent verification.
        if missing:
            status = DecisionStatus.HOLD_UNACKNOWLEDGED_OBJECTION
        elif conflict_with_evidence:
            status = DecisionStatus.BLOCKED_BY_EVIDENCE
        elif dissent and not evidence_known and (task.consequential or task.protected_effect):
            status = DecisionStatus.HOLD_UNVERIFIED_DISPUTE
        elif task.protected_effect:
            status = DecisionStatus.PROPOSED_EFFECT_REQUIRES_AUTHORIZATION
        elif aligns_with_evidence:
            status = DecisionStatus.EVIDENCE_SUPPORTED_ADVISORY
        elif dissent:
            status = DecisionStatus.CONTESTED_ADVISORY
        else:
            status = DecisionStatus.UNVERIFIED_ADVISORY

        claim_verified = bool(aligns_with_evidence and status in {
            DecisionStatus.EVIDENCE_SUPPORTED_ADVISORY,
            DecisionStatus.PROPOSED_EFFECT_REQUIRES_AUTHORIZATION,
        })
        ceiling = (
            "MECHANICAL_DECISION_PROTOCOL_ONLY_NOT_NEURAL_QUALIFICATION: "
            "receipt records a Core position, not proof of model reasoning; "
            "verifier provenance must be assessed separately; no effects executed"
        )
        fields = {
            "schema": "MOSAIC_THINKING_CORE_DELIBERATION_V1",
            "task": asdict(task),
            "core_initial": asdict(initial),
            "core_final": asdict(final),
            "objections": [asdict(item) for item in objections],
            "evidence": None if evidence is None else asdict(evidence),
            "status": status.value,
            "claim_verified": claim_verified,
            "verified_proposition_is_false": bool(
                evidence_known and evidence.supports is False
            ),
            "effect_executed": False,
            "claim_ceiling": ceiling,
        }
        return DeliberationReceipt(
            task=task,
            core_initial=initial,
            core_final=final,
            objections=objections,
            evidence=evidence,
            status=status,
            claim_verified=claim_verified,
            verified_proposition_is_false=fields["verified_proposition_is_false"],
            effect_executed=False,
            claim_ceiling=ceiling,
            receipt_sha256=_receipt_digest(fields),
        )
