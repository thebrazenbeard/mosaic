"""Read-only Rezon collective answer -> Mosaic specialist *advice* adapter.

Rezon's exported snapshot is a claimed data object. Its content digest detects
accidental or uncoordinated edits but does not authenticate its origin, model
weights, independent host, or truth. No Rezon module or network client is
imported into Mosaic's Core; a caller explicitly supplies an observed snapshot.
"""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import re

from .strict_json import loads as strict_json_loads
from .thinking_core import (
    CorePosition, DeliberationTask, SpecialistObjection, subject_digest,
)


class RezonAdvisoryError(ValueError):
    """The external Rezon claim cannot safely become a Mosaic objection."""


_EPISODE = re.compile(r"collective-[a-f0-9]{32}\Z")
_AGENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")
_SHA = re.compile(r"[a-f0-9]{64}\Z")
_SCHEMA = "MOSAIC_REZON_SPECIALIST_TASK_V1"
_MAX_SNAPSHOT_BYTES = 1_000_000


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise RezonAdvisoryError("Rezon snapshot is not canonical JSON") from exc


def _utf8(value: object, limit: int) -> bool:
    if type(value) is not str or not value.strip() or len(value) > limit:
        return False
    try:
        value.encode("utf-8")
    except UnicodeError:
        return False
    return True


def rezon_task_prompt(task: DeliberationTask) -> str:
    """Canonical Rezon prompt for one exact Mosaic task and typed advisory.

    The prompt is copied verbatim into Rezon's episode. Rezon participants
    independently return JSON with proposition_id, supports, rationale.
    """
    subject_digest(task)  # Mosaic validates all task fields and UTF-8 first.
    prompt = _canonical({
        "schema": _SCHEMA,
        "task": asdict(task),
        "instruction": (
            "Analyze independently; return ONLY JSON with exact fields "
            "proposition_id (same string), supports (boolean), rationale "
            "(nonempty string). This is unverified specialist advice. "
            "Do not claim authorization, authenticated sources, or verification."
        ),
    })
    if len(prompt) > 16_384:
        raise RezonAdvisoryError("Mosaic task exceeds Rezon prompt limit")
    return prompt


def _refs(value: object) -> bool:
    return type(value) is list and len(value) <= 32 and (
        all(_utf8(item, 1024) for item in value)
        and len(set(value)) == len(value)
    )


def _contributions(value: object, *, kind: str, agents: tuple[str, ...],
                   answer_ids: set[str]) -> tuple[list[dict[str, object]], set[str]]:
    if type(value) is not list or len(value) > len(agents):
        raise RezonAdvisoryError("invalid Rezon contribution list")
    seen: set[str] = set()
    canonical: list[dict[str, object]] = []
    for item in value:
        if type(item) is not dict or set(item) != {
            "id", "agent_id", "kind", "content", "source_refs", "target_ids"
        }:
            raise RezonAdvisoryError("invalid Rezon contribution fields")
        agent = item["agent_id"]
        if type(agent) is not str or agent not in agents or agent in seen or (
            item["kind"] != kind or item["id"] != f"{kind}:{agent}"
        ):
            raise RezonAdvisoryError("invalid Rezon contribution identity")
        if not _utf8(item["content"], 32_768) or not _refs(item["source_refs"]) or (
            not _refs(item["target_ids"])
        ):
            raise RezonAdvisoryError("invalid Rezon contribution content or references")
        targets = set(item["target_ids"])
        if (kind == "answer" and targets) or (kind == "review" and (
            not targets or f"answer:{agent}" in targets or not targets <= answer_ids
        )):
            raise RezonAdvisoryError("invalid Rezon contribution review targets")
        seen.add(agent)
        canonical.append({
            "contribution_id": item["id"], "agent_id": agent, "kind": kind,
            "content": item["content"], "source_refs": item["source_refs"],
            "target_ids": item["target_ids"],
        })
    return canonical, seen


def _snapshot_advisory(task: DeliberationTask, snapshot: object,
                       specialist_id: str) -> tuple[SpecialistObjection, dict[str, str]]:
    if type(snapshot) is not dict or set(snapshot) != {
        "episode_id", "prompt", "agents", "answers", "reviews", "syntheses",
        "missing_agents", "complete", "input_digest",
    }:
        raise RezonAdvisoryError("invalid Rezon snapshot fields")
    serialized = _canonical(snapshot)
    if len(serialized.encode("utf-8")) > _MAX_SNAPSHOT_BYTES:
        raise RezonAdvisoryError("Rezon snapshot exceeds bounded input")
    episode_id = snapshot["episode_id"]
    if type(episode_id) is not str or not _EPISODE.fullmatch(episode_id):
        raise RezonAdvisoryError("invalid Rezon episode identity")
    if snapshot["prompt"] != rezon_task_prompt(task):
        raise RezonAdvisoryError("Rezon prompt is not bound to this task")
    agents = snapshot["agents"]
    if type(agents) is not list or not 1 <= len(agents) <= 32 or (
        not all(type(agent) is str and _AGENT.fullmatch(agent) for agent in agents)
        or agents != sorted(set(agents))
    ):
        raise RezonAdvisoryError("invalid Rezon worker roster")
    if specialist_id not in agents or specialist_id.casefold() == "core":
        raise RezonAdvisoryError("expected Rezon specialist is absent or reserved")
    answer_claims, answered = _contributions(
        snapshot["answers"], kind="answer", agents=tuple(agents), answer_ids=set()
    )
    answer_ids = {f"answer:{name}" for name in answered}
    review_claims, _ = _contributions(
        snapshot["reviews"], kind="review", agents=tuple(agents), answer_ids=answer_ids
    )
    missing = sorted(set(agents) - answered)
    if snapshot["missing_agents"] != missing or type(snapshot["complete"]) is not bool or (
        snapshot["complete"] != (not missing)
    ) or missing:
        raise RezonAdvisoryError("incomplete or unsupported Rezon round")
    basis = {
        "episode_id": episode_id, "prompt": snapshot["prompt"],
        "agents": agents,
        "contributions": sorted(answer_claims + review_claims,
                                key=lambda item: (str(item["kind"]), str(item["agent_id"]))),
    }
    digest = sha256(_canonical(basis).encode("utf-8")).hexdigest()
    if type(snapshot["input_digest"]) is not str or not _SHA.fullmatch(
        snapshot["input_digest"]
    ) or digest != snapshot["input_digest"]:
        raise RezonAdvisoryError("Rezon contribution digest mismatch")
    syntheses = snapshot["syntheses"]
    if type(syntheses) is not list or len(syntheses) > 32:
        raise RezonAdvisoryError("invalid Rezon synthesis list")
    synthesis_ids: set[str] = set()
    for item in syntheses:
        if type(item) is not dict or set(item) != {
            "synthesis_id", "integrator_id", "input_digest", "content",
            "status", "is_current",
        }:
            raise RezonAdvisoryError("invalid Rezon synthesis fields")
        integrator, previous_digest = item["integrator_id"], item["input_digest"]
        if type(integrator) is not str or not _AGENT.fullmatch(integrator) or (
            type(previous_digest) is not str or not _SHA.fullmatch(previous_digest)
        ) or item["synthesis_id"] != f"synthesis:{previous_digest}:{integrator}" or (
            type(item["is_current"]) is not bool
            or item["is_current"] != (previous_digest == digest)
        ) or item["status"] != "unverified_model_synthesis" or (
            not _utf8(item["content"], 32_768)
        ) or item["synthesis_id"] in synthesis_ids:
            raise RezonAdvisoryError("invalid or falsely promoted Rezon synthesis")
        synthesis_ids.add(item["synthesis_id"])
    # A synthesis is never used as a specialist answer: only the selected
    # immutable first-round model contribution supplies Mosaic's objection.
    selected = next((claim for claim in answer_claims
                     if claim["agent_id"] == specialist_id), None)
    if selected is None:
        raise RezonAdvisoryError("expected Rezon specialist answer missing")
    try:
        answer = strict_json_loads(selected["content"])
    except (ValueError, TypeError) as exc:
        raise RezonAdvisoryError("Rezon answer is not strict JSON") from exc
    if type(answer) is not dict or set(answer) != {
        "proposition_id", "supports", "rationale"
    }:
        raise RezonAdvisoryError("Rezon advisory answer fields violate the schema")
    if answer["proposition_id"] != task.proposition_id or (
        type(answer["supports"]) is not bool
    ) or not _utf8(answer["rationale"], 32_768):
        raise RezonAdvisoryError("invalid Rezon advisory task stance or rationale")
    return (SpecialistObjection(
        specialist_id=specialist_id, proposition_id=task.proposition_id,
        supports=answer["supports"], rationale=answer["rationale"],
    ), {
        "episode_id": episode_id, "input_digest": digest,
        "contribution_id": selected["contribution_id"],
        "evidence_status": "UNVERIFIED_REZON_ADVISORY",
    })


class RezonAdvisorySpecialist:
    """Frozen unverified Rezon initial-answer specialist, no external effects."""

    def __init__(self, task: DeliberationTask, snapshot: dict[str, object],
                 specialist_id: str) -> None:
        self._objection, self._provenance = _snapshot_advisory(
            task, snapshot, specialist_id,
        )
        self.specialist_id = specialist_id
        self._task = task

    @property
    def provenance(self) -> dict[str, str]:
        return dict(self._provenance)

    def review(self, task: DeliberationTask,
               core_position: CorePosition) -> SpecialistObjection:
        if task != self._task or not isinstance(core_position, CorePosition) or (
            core_position.proposition_id != task.proposition_id
        ):
            raise RezonAdvisoryError("Rezon advisory bound to another task/Core position")
        return self._objection
