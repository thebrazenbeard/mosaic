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
    HOLD_UNATTESTED_VERIFIER_CONFLICT = "HOLD_UNATTESTED_VERIFIER_CONFLICT"
    VERIFIER_ASSERTED_ADVISORY = "VERIFIER_ASSERTED_ADVISORY"
    # V1 historical labels: this unauthenticated source engine never emits
    # these stronger statuses; no trusted-verifier path exists yet.
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
    # Binds a verifier's assertion to the *entire* task, not merely a reusable
    # proposition ID. This is not a signature or verifier attestation.
    subject_sha256: str = ""


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
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ProtocolViolation(f"{label} must be valid UTF-8 text") from exc


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
    if not isinstance(result.evidence_ref, str):
        raise ProtocolViolation("verifier evidence_ref must be a string")
    if result.supports is not None:
        _required(result.evidence_ref, "independent evidence reference")
    elif result.evidence_ref != "":
        raise ProtocolViolation("Unknown verdict cannot assert verified evidence")
    if not isinstance(result.subject_sha256, str) or (
        len(result.subject_sha256) != 64 or
        any(char not in "0123456789abcdef" for char in result.subject_sha256)
    ):
        raise ProtocolViolation("Verifier subject_sha256 must be a lowercase SHA-256 hex string")
    if result.subject_sha256 != subject_digest(task):
        raise ProtocolViolation("Verifier evidence subject binding does not match task")


def subject_digest(task: DeliberationTask) -> str:
    """Deterministic exact-task binding; *not* independent verification."""
    _task(task)
    return _receipt_digest({"task": asdict(task)})


def _receipt_digest(fields: dict[str, object]) -> str:
    try:
        serialized = json.dumps(
            fields, sort_keys=True, separators=(",", ":"), allow_nan=False,
            ensure_ascii=False,
        ).encode("utf-8")
    except (UnicodeEncodeError, TypeError, ValueError) as exc:
        raise ProtocolViolation("receipt cannot be encoded as canonical UTF-8 JSON") from exc
    return sha256(serialized).hexdigest()


class CoreDeliberationEngine:
    """Boundary checks around neural reasoning, not an automated intelligence.

    The Core constructs and revises positions. Specialists can challenge them.
    A caller-injected verifier can supply assertions, not authenticated proof.
    No trust root is configured here, so claim verification remains false.
    The engine itself executes no protected effects; injected Python callbacks
    are outside its effect-observation and sandboxing boundary.
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

        # A verifier-asserted disagreement can require a HOLD, but a
        # specialist cannot replace the Core as the origin of final reasoning.
        # An arbitrary verifier ID/evidence string is not an attestation.
        if missing:
            status = DecisionStatus.HOLD_UNACKNOWLEDGED_OBJECTION
        elif conflict_with_evidence:
            status = DecisionStatus.HOLD_UNATTESTED_VERIFIER_CONFLICT
        elif dissent and not evidence_known and (task.consequential or task.protected_effect):
            status = DecisionStatus.HOLD_UNVERIFIED_DISPUTE
        elif task.protected_effect:
            status = DecisionStatus.PROPOSED_EFFECT_REQUIRES_AUTHORIZATION
        elif aligns_with_evidence:
            status = DecisionStatus.VERIFIER_ASSERTED_ADVISORY
        elif dissent:
            status = DecisionStatus.CONTESTED_ADVISORY
        else:
            status = DecisionStatus.UNVERIFIED_ADVISORY

        # In-memory provider injection authenticates neither verifier origin
        # nor its evidence. A digest prevents accidental subject replay, not
        # forgery. Do not promote caller assertions to verified facts.
        claim_verified = False
        ceiling = (
            "MECHANICAL_DECISION_PROTOCOL_ONLY_NOT_NEURAL_QUALIFICATION: "
            "receipt records a Core position, not proof of model reasoning; "
            "verifier assertions are UNATTESTED; claim_verified remains false; "
            "effect_executed=false covers engine-owned actions only; "
            "caller-injected callbacks are not sandboxed or monitored"
        )
        fields = {
            "schema": "MOSAIC_THINKING_CORE_DELIBERATION_V2",
            "task": asdict(task),
            "core_initial": asdict(initial),
            "core_final": asdict(final),
            "objections": [asdict(item) for item in objections],
            "evidence": None if evidence is None else asdict(evidence),
            "status": status.value,
            "claim_verified": claim_verified,
            "verified_proposition_is_false": False,
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
