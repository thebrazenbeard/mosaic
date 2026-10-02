from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable


CONDITIONS = (
    "RESET",
    "MOSAIC_STATE",
    "FULL_HISTORY",
    "ORDINARY_RETRIEVAL",
    "MULTI_RESOLUTION_STATE",
    "MULTI_RESOLUTION_REHYDRATE",
)


class HandoffProtocolError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class StateEvent:
    event_id: str
    family: str
    key: str
    value: str
    revision: int

    def validate(self) -> None:
        if not self.event_id.strip() or not self.family.strip() or not self.key.strip():
            raise HandoffProtocolError("event_id, family, and key are required")
        if self.revision < 1:
            raise HandoffProtocolError("revision must be positive")


@dataclass(frozen=True, slots=True)
class ContinuityCase:
    case_id: str
    family: str
    events: tuple[StateEvent, ...]
    query_key: str
    expected_value: str

    def validate(self) -> None:
        if not self.case_id.strip() or not self.family.strip():
            raise HandoffProtocolError("case_id and family are required")
        if not self.events:
            raise HandoffProtocolError("case requires stage-A events")
        for event in self.events:
            event.validate()
            if event.family != self.family:
                raise HandoffProtocolError("event family mismatch")
        revisions = [event.revision for event in self.events]
        if len(revisions) != len(set(revisions)):
            raise HandoffProtocolError("event revisions must be unique within a case")
        current = max(self.events, key=lambda event: event.revision)
        if current.key != self.query_key or current.value != self.expected_value:
            raise HandoffProtocolError(
                "expected result must bind to the highest-revision queried event"
            )


def _encode(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _mosaic_state(events: tuple[StateEvent, ...]) -> dict[str, object]:
    by_key: dict[str, StateEvent] = {}
    superseded: dict[str, list[str]] = {}
    for event in sorted(events, key=lambda item: item.revision):
        previous = by_key.get(event.key)
        if previous is not None:
            superseded.setdefault(event.key, []).append(previous.event_id)
        by_key[event.key] = event

    return {
        "schema_id": "MOSAIC_HANDOFF_STATE_V1",
        "current": {
            key: {
                "event_id": event.event_id,
                "family": event.family,
                "value": event.value,
                "revision": event.revision,
            }
            for key, event in sorted(by_key.items())
        },
        "superseded_event_ids": {
            key: ids for key, ids in sorted(superseded.items())
        },
    }


def _full_history(events: tuple[StateEvent, ...]) -> list[dict[str, object]]:
    return [
        {
            "event_id": event.event_id,
            "family": event.family,
            "key": event.key,
            "value": event.value,
            "revision": event.revision,
        }
        for event in events
    ]


def _multi_resolution_state(events: tuple[StateEvent, ...]) -> dict[str, object]:
    by_key: dict[str, StateEvent] = {}
    for event in sorted(events, key=lambda item: item.revision):
        by_key[event.key] = event
    return {
        "schema_id": "MOSAIC_MULTI_RESOLUTION_STATE_V1",
        "current_index": {
            key: [event.event_id, event.revision]
            for key, event in sorted(by_key.items())
        },
    }


def _rehydrate_current_event(
    events: tuple[StateEvent, ...],
    compact_state: dict[str, object],
    query_key: str,
) -> StateEvent | None:
    current_index = compact_state["current_index"]
    if not isinstance(current_index, dict):
        raise HandoffProtocolError("multi-resolution current_index must be an object")
    locator = current_index.get(query_key)
    if locator is None:
        return None
    if (
        not isinstance(locator, list)
        or len(locator) != 2
        or not isinstance(locator[0], str)
        or not isinstance(locator[1], int)
    ):
        raise HandoffProtocolError("multi-resolution locator is malformed")
    event_id, revision = locator
    matches = [
        event
        for event in events
        if event.event_id == event_id and event.revision == revision
    ]
    if len(matches) != 1:
        raise HandoffProtocolError("multi-resolution backing locator is not exact")
    return matches[0]


def _ordinary_retrieval(events: tuple[StateEvent, ...], query_key: str) -> dict[str, object]:
    relevant = [event for event in events if event.key == query_key]
    if not relevant:
        return {"records": []}
    current = max(relevant, key=lambda event: event.revision)
    return {
        "records": [
            {
                "source": "ordinary-retrieval",
                "key": current.key,
                "value": current.value,
                "revision": current.revision,
            }
        ]
    }


def _specialist_b_answer(condition: str, payload: object, query_key: str) -> str:
    if condition in ("RESET", "MULTI_RESOLUTION_STATE"):
        return "UNKNOWN"
    if condition == "MOSAIC_STATE":
        current = payload["current"]  # type: ignore[index]
        record = current.get(query_key)  # type: ignore[union-attr]
        return "UNKNOWN" if record is None else str(record["value"])
    if condition == "FULL_HISTORY":
        records = [
            item for item in payload  # type: ignore[union-attr]
            if item["key"] == query_key
        ]
        if not records:
            return "UNKNOWN"
        current = max(records, key=lambda item: int(item["revision"]))
        return str(current["value"])
    if condition == "ORDINARY_RETRIEVAL":
        records = payload["records"]  # type: ignore[index]
        return "UNKNOWN" if not records else str(records[0]["value"])
    raise HandoffProtocolError(f"unknown condition: {condition}")


def run_case(case: ContinuityCase, condition: str) -> dict[str, object]:
    case.validate()
    if condition not in CONDITIONS:
        raise HandoffProtocolError(f"unknown condition: {condition}")

    rehydration_count = 0
    rehydration_bytes = 0
    rehydrated_event_ids: list[str] = []
    backing_state_bytes = 0

    if condition == "RESET":
        payload: object = {}
        answer = _specialist_b_answer(condition, payload, case.query_key)
    elif condition == "MOSAIC_STATE":
        payload = _mosaic_state(case.events)
        answer = _specialist_b_answer(condition, payload, case.query_key)
    elif condition == "FULL_HISTORY":
        payload = _full_history(case.events)
        answer = _specialist_b_answer(condition, payload, case.query_key)
    elif condition == "ORDINARY_RETRIEVAL":
        payload = _ordinary_retrieval(case.events, case.query_key)
        answer = _specialist_b_answer(condition, payload, case.query_key)
    else:
        payload = _multi_resolution_state(case.events)
        backing = _full_history(case.events)
        backing_state_bytes = len(_encode(backing))
        if condition == "MULTI_RESOLUTION_STATE":
            answer = "UNKNOWN"
        else:
            event = _rehydrate_current_event(
                case.events,
                payload,
                case.query_key,
            )
            if event is None:
                answer = "UNKNOWN"
            else:
                exact_record = {
                    "event_id": event.event_id,
                    "family": event.family,
                    "key": event.key,
                    "value": event.value,
                    "revision": event.revision,
                }
                exact_bytes = _encode(exact_record)
                answer = event.value
                rehydration_count = 1
                rehydration_bytes = len(exact_bytes)
                rehydrated_event_ids = [event.event_id]

    encoded = _encode(payload)
    stale_values = {
        event.value
        for event in case.events
        if event.key == case.query_key and event.value != case.expected_value
    }
    correct = answer == case.expected_value
    stale_error = answer in stale_values and not correct

    return {
        "case_id": case.case_id,
        "family": case.family,
        "condition": condition,
        "answer": answer,
        "expected_value": case.expected_value,
        "correct": correct,
        "stale_error": stale_error,
        "correction_burden": 0 if correct else 1,
        "payload_bytes": len(encoded),
        "active_state_bytes": len(encoded),
        "backing_state_bytes": backing_state_bytes,
        "rehydration_count": rehydration_count,
        "rehydration_bytes": rehydration_bytes,
        "rehydrated_event_ids": rehydrated_event_ids,
    }


def run_benchmark(cases: Iterable[ContinuityCase]) -> dict[str, object]:
    cases = tuple(cases)
    if not cases:
        raise HandoffProtocolError("benchmark requires at least one case")
    if len({case.case_id for case in cases}) != len(cases):
        raise HandoffProtocolError("case_id values must be unique")
    for case in cases:
        case.validate()

    results = [
        run_case(case, condition)
        for condition in CONDITIONS
        for case in cases
    ]

    by_condition: dict[str, dict[str, object]] = {}
    for condition in CONDITIONS:
        subset = [row for row in results if row["condition"] == condition]
        by_condition[condition] = {
            "cases": len(subset),
            "correct": sum(bool(row["correct"]) for row in subset),
            "accuracy": sum(bool(row["correct"]) for row in subset) / len(subset),
            "stale_errors": sum(bool(row["stale_error"]) for row in subset),
            "correction_burden": sum(int(row["correction_burden"]) for row in subset),
            "payload_bytes_total": sum(int(row["payload_bytes"]) for row in subset),
            "active_state_bytes_total": sum(
                int(row["active_state_bytes"]) for row in subset
            ),
            "backing_state_bytes_total": sum(
                int(row["backing_state_bytes"]) for row in subset
            ),
            "rehydration_count": sum(
                int(row["rehydration_count"]) for row in subset
            ),
            "rehydration_bytes_total": sum(
                int(row["rehydration_bytes"]) for row in subset
            ),
        }

    mosaic = by_condition["MOSAIC_STATE"]
    reset = by_condition["RESET"]
    history = by_condition["FULL_HISTORY"]

    return {
        "schema_id": "MOSAIC_P1_PROTOCOL_HARNESS_RECEIPT_V1",
        "qualification": "DETERMINISTIC_PROTOCOL_HARNESS_ONLY",
        "task_families": sorted({case.family for case in cases}),
        "conditions": by_condition,
        "mosaic_outperforms_reset_in_harness": (
            float(mosaic["accuracy"]) > float(reset["accuracy"])
        ),
        "mosaic_stale_errors_not_above_reset": (
            int(mosaic["stale_errors"]) <= int(reset["stale_errors"])
        ),
        "full_history_capability_control_succeeds": float(history["accuracy"]) == 1.0,
        "actual_accelerator_unload_verified": False,
        "swap_latency_measured": False,
        "encode_decode_latency_measured": False,
        "independent_clean_repeat_completed": False,
        "p1_pass_eligible": False,
        "claim_ceiling": (
            "Deterministic protocol harness only; does not establish actual specialist "
            "accelerator unload, swap latency, clean-run repeatability, or Mosaic P1 PASS."
        ),
        "results": results,
    }
