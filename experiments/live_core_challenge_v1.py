"""Frozen paired *real Core inference* challenge probe, not Mosaic qualification.

A single neural Core output is reused in Core-only and challenged conditions.
The specialist is a predeclared scripted argument, NOT a learned model.
The test oracle grades outputs but is NEVER provided to neural inference.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

from runtime.local_neural_backend import LoopbackCoreInference, LoopbackBackendError
from runtime.neural_core import NeuralThinkingCore, NeuralOutputError
from runtime.thinking_core import (
    CoreDeliberationEngine, CorePosition, DeliberationTask, SpecialistObjection,
    ProtocolViolation
)


class FrozenCaseError(ValueError):
    pass


EXPECTED_IDS = tuple(f"T{i:02d}" for i in range(1, 11))
VALID_STATUSES = {"CORRECT", "MISLEADING"}
FROZEN_FILE = (
    Path(__file__).resolve().parent
    / "MOSAIC_LIVE_CORE_CHALLENGE_CASES_20261016_V1.json"
)
RESULT_DIR = Path(r"D:\VERA\portal-task-state")
RESULT_STEM = "MOSAIC_LIVE_CORE_CHALLENGE_20261016_V1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_frozen_cases(path: Path) -> tuple[dict, tuple[dict, ...], str]:
    raw = path.read_bytes()
    frozen = json.loads(raw.decode("utf-8"))
    if frozen.get("schema") != "MOSAIC_LIVE_CORE_CHALLENGE_CASES_V1":
        raise FrozenCaseError("Unexpected frozen case schema")
    items = frozen.get("cases")
    if not isinstance(items, list) or tuple(c.get("id") for c in items) != EXPECTED_IDS:
        raise FrozenCaseError("Frozen cases changed membership or order")
    kinds = []
    for case in items:
        if set(case) != {
            "id", "proposition_id", "question", "truth",
            "specialist_stance", "specialist_argument", "specialist_quality"
        }:
            raise FrozenCaseError("Case fields changed")
        if (case["proposition_id"] != "claim"
            or not isinstance(case["question"], str) or not case["question"].strip()
            or not isinstance(case["specialist_argument"], str)
            or not case["specialist_argument"].strip()
            or type(case["truth"]) is not bool
            or type(case["specialist_stance"]) is not bool
            or case["specialist_quality"] not in VALID_STATUSES):
            raise FrozenCaseError("Invalid frozen case")
        expected = (
            "CORRECT" if case["truth"] == case["specialist_stance"]
            else "MISLEADING"
        )
        if case["specialist_quality"] != expected:
            raise FrozenCaseError("Specialist truth label mismatch")
        kinds.append(expected)
    if kinds.count("CORRECT") != 5 or kinds.count("MISLEADING") != 5:
        raise FrozenCaseError("Expected exactly five correct and five misleading critics")
    return frozen, tuple(items), sha256(raw).hexdigest()


class _ReplayInitial:
    def __init__(self, real: NeuralThinkingCore, initial: CorePosition):
        self.real = real
        self.initial = initial

    def think(self, task: DeliberationTask) -> CorePosition:
        return self.initial

    def reconsider(self, task, prior, objections, evidence):
        return self.real.reconsider(task, prior, objections, evidence)


class _FrozenSpecialist:
    specialist_id = "scripted-critic"

    def __init__(self, item: dict):
        self.item = item

    def review(self, task, core_position) -> SpecialistObjection:
        return SpecialistObjection(
            specialist_id=self.specialist_id,
            proposition_id=task.proposition_id,
            supports=self.item["specialist_stance"],
            rationale=self.item["specialist_argument"],
        )


def evaluate_case(item: dict, backend) -> dict:
    """No retries, no oracle leakage, no model-side grading."""
    task = DeliberationTask(
        case_id=item["id"],
        proposition_id=item["proposition_id"],
        question=item["question"],
    )
    result = {
        "case_id": item["id"],
        "question": item["question"],
        "truth": item["truth"],
        "specialist_stance": item["specialist_stance"],
        "specialist_argument": item["specialist_argument"],
        "specialist_quality": item["specialist_quality"],
        "initial_supports": None,
        "challenged_supports": None,
        "initial_reasoning": None,
        "challenged_reasoning": None,
        "baseline_correct": False,
        "challenged_correct": False,
        "challenged_decision_usable": False,
        "challenged_status": "NOT_RUN",
        "independently_verified": False,
        "effect_executed": False,
        "challenge_triggered_reconsideration": False,
        "model_generation_calls": 0,
        "elapsed_seconds": None,
        "error_category": None,
        "error_detail": None,
        "receipt_sha256": None,
    }
    start = time.perf_counter()
    neural = NeuralThinkingCore(backend)
    # The real HTTP calls are counted without depending on backend internals.
    class _Counted:
        def __call__(self, messages):
            result["model_generation_calls"] += 1
            return backend(messages)

    neural = NeuralThinkingCore(_Counted())
    try:
        initial = neural.think(task)
        result["initial_supports"] = initial.supports
        result["initial_reasoning"] = initial.reasoning
        result["baseline_correct"] = initial.supports == item["truth"]
        result["challenge_triggered_reconsideration"] = (
            initial.supports != item["specialist_stance"]
        )
        receipt = CoreDeliberationEngine().deliberate(
            task, _ReplayInitial(neural, initial), [_FrozenSpecialist(item)],
        )
        result["challenged_supports"] = receipt.core_final.supports
        result["challenged_reasoning"] = receipt.core_final.reasoning
        result["challenged_status"] = receipt.status.value
        result["challenged_decision_usable"] = (
            not receipt.status.value.startswith("HOLD_")
            and receipt.status.value != "BLOCKED_BY_EVIDENCE"
        )
        result["challenged_correct"] = (
            result["challenged_decision_usable"]
            and receipt.core_final.supports == item["truth"]
        )
        result["independently_verified"] = receipt.claim_verified
        result["effect_executed"] = receipt.effect_executed
        result["receipt_sha256"] = receipt.receipt_sha256
    except (NeuralOutputError, LoopbackBackendError, ProtocolViolation,
            OSError, ValueError) as exc:
        result["error_category"] = type(exc).__name__
        result["error_detail"] = str(exc)[:350]
        result["challenged_status"] = "FAIL_CLOSED_MODEL_OR_PROTOCOL_ERROR"
        # Even a correct initial guess is not a successful paired challenge
        # if the later required stage errors.
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    return result


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No results to aggregate")
    return {
        "total_cases": len(rows),
        "baseline_correct": sum(bool(r["baseline_correct"]) for r in rows),
        "challenged_correct": sum(bool(r["challenged_correct"]) for r in rows),
        "improvements": sum(
            (not r["baseline_correct"]) and r["challenged_correct"] for r in rows
        ),
        "regressions": sum(
            r["baseline_correct"] and (not r["challenged_correct"]) for r in rows
        ),
        "unchanged_correct": sum(
            r["baseline_correct"] and r["challenged_correct"] for r in rows
        ),
        "unchanged_incorrect": sum(
            not r["baseline_correct"] and not r["challenged_correct"] for r in rows
        ),
        "misleading_challenge_regressions": sum(
            r["specialist_quality"] == "MISLEADING"
            and r["baseline_correct"] and not r["challenged_correct"]
            for r in rows
        ),
        "correct_challenge_improvements": sum(
            r["specialist_quality"] == "CORRECT"
            and not r["baseline_correct"] and r["challenged_correct"]
            for r in rows
        ),
        "reconsiderations": sum(
            bool(r["challenge_triggered_reconsideration"]) for r in rows
        ),
        "errors": sum(bool(r["error_category"]) for r in rows),
        "model_generation_calls": sum(int(r["model_generation_calls"]) for r in rows),
        "real_specialist_model_used": False,
        "independent_verifier_used_in_inference": False,
        "no_oracle_labels_sent_to_core": True,
        "behavior_qualification": "EXPLORATORY_DEVELOPMENT_ONLY",
    }

def main() -> int:
    import argparse
    import subprocess

    parser = argparse.ArgumentParser(
        description="Exact-source development-only real neural Core challenge benchmark"
    )
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain=v1"],
        capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip()
    if head != args.expected_head or dirty:
        raise RuntimeError("Source must be exact expected clean commit")
    frozen, cases, fixture_sha = load_frozen_cases(FROZEN_FILE)
    preflight_path = RESULT_DIR / (RESULT_STEM + "_PREFLIGHT.json")
    partial_path = RESULT_DIR / (RESULT_STEM + "_RESULTS.jsonl")
    summary_path = RESULT_DIR / (RESULT_STEM + "_SUMMARY.json")
    if any(p.exists() for p in (preflight_path, partial_path, summary_path)):
        raise RuntimeError("Evaluation attempt namespace already used: refuse overwrite")

    backend = LoopbackCoreInference(timeout_seconds=45)
    advertised = backend.preflight()  # Must be serving; no service start.
    # One receipt for predeclared test identity BEFORE model answers are sampled.
    preflight = {
        "schema": "MOSAIC_LIVE_CORE_CHALLENGE_PREFLIGHT_V1",
        "status": "LIVE_ENDPOINT_ADVERTISED_EVAL_INPUTS_BOUND",
        "source_head": head,
        "source_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "frozen_cases_sha256": fixture_sha,
        "case_ids": list(EXPECTED_IDS),
        "frozen_cases_count": len(cases),
        "challenge_quality_split": {"CORRECT": 5, "MISLEADING": 5},
        "endpoint": advertised,
        "model_sha256_cryptographically_verified": False,
        "running_neural_model": True,
        "real_specialist_model_used": False,
        "independent_verifier_used_in_inference": False,
        "model_checkpoint_deployed_or_changed": False,
        "started_at_utc": _now(),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with preflight_path.open("x", encoding="utf-8", newline="\n") as fd:
        json.dump(preflight, fd, indent=2, sort_keys=True)
        fd.write("\n")
    print("FROZEN_BENCHMARK_PREFLIGHT", json.dumps(preflight, sort_keys=True), flush=True)

    rows: list[dict] = []
    with partial_path.open("x", encoding="utf-8", newline="\n") as fd:
        for case in cases:
            row = evaluate_case(case, backend)
            rows.append(row)
            fd.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
            fd.flush()
            print("CASE", row["case_id"],
                  "baseline", row["initial_supports"],
                  "challenge", row["challenged_supports"],
                  "baseline_ok", row["baseline_correct"],
                  "challenge_ok", row["challenged_correct"],
                  "calls", row["model_generation_calls"],
                  "error", row["error_category"],flush=True)
    result = aggregate(rows)
    result.update({
        "schema": "MOSAIC_LIVE_CORE_CHALLENGE_RESULT_V1",
        "status": "DEVELOPMENT_BENCHMARK_FINISHED_NOT_BEHAVIOR_QUALIFIED",
        "source_head": head,
        "source_code_sha256": preflight["source_code_sha256"],
        "frozen_cases_sha256": fixture_sha,
        "endpoint_advertised_model_id": advertised["model_id"],
        "checkpoint_sha256_verified": False,
        "case_result_sha256": sha256(partial_path.read_bytes()).hexdigest(),
        "finished_at_utc": _now(),
        "model_trained_or_deployed_by_test": False,
        "protected_effect_executed": False,
        "claim_ceiling": (
            "One unreplicated development set, Core is real serving neural model, "
            "specialist is scripted argument, model ID is self-advertised and "
            "adapter lineage is not cryptographically proved. No independent "
            "Mosaic reasoning, real specialist or deployment qualification."
        ),
    })
    with summary_path.open("x", encoding="utf-8", newline="\n") as fd:
        json.dump(result, fd, sort_keys=True, indent=2)
        fd.write("\n")
    print("SUMMARY", json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
