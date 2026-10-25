"""Offline paired Mosaic Core experiment scorer; never an evidence verifier.

Consumes caller-supplied fixed rows. It does not run a model, assess the
correctness of supplied gold labels, authenticate model identity or claim a
learned specialist improvement. See experiments/MOSAIC_CORE_NEUTRAL_CONTROL_V1.md.
"""
from __future__ import annotations

from hashlib import sha256
import json
import sys

from .strict_json import loads as strict_json_loads


SCHEMA = "MOSAIC_CORE_CHALLENGE_PAIRED_INPUT_V1"
ARMS = ("core_only", "neutral_second_pass", "challenged")
STATUSES = ("ok", "abstain", "output_contract_error")


class ChallengeEvalError(ValueError):
    """Invalid or incomparable caller-supplied experiment record."""


def _keys(value: object, names: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != names:
        raise ChallengeEvalError(f"{label} fields do not match the schema")
    return value


def _id(value: object, label: str) -> str:
    if (type(value) is not str or not 1 <= len(value) <= 128
            or value != value.strip()):
        raise ChallengeEvalError(f"{label} must be a bounded canonical string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ChallengeEvalError(f"{label} contains invalid Unicode") from exc
    return value


def _int(value: object, label: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ChallengeEvalError(f"{label} must be an integer in {low}..{high}")
    return value


def _arm(value: object, label: str, calls: int, token_ceiling: int) -> dict[str, object]:
    arm = _keys(value, {"status", "supports", "model_calls", "total_tokens",
                        "elapsed_ms"}, label)
    status = arm["status"]
    if status not in STATUSES or type(status) is not str:
        raise ChallengeEvalError(f"{label} has unknown status")
    if status == "ok":
        if type(arm["supports"]) is not bool:
            raise ChallengeEvalError(f"{label} ok status requires boolean answer")
    elif arm["supports"] is not None:
        raise ChallengeEvalError(f"{label} non-ok status cannot assert an answer")
    _int(arm["model_calls"], f"{label} model_calls", 0, calls)
    _int(arm["total_tokens"], f"{label} total_tokens", 0, token_ceiling)
    _int(arm["elapsed_ms"], f"{label} elapsed_ms", 0, 1_000_000_000)
    return arm


def evaluate_paired(value: object) -> dict[str, object]:
    """Score all three arms on identical supplied cases and disclosed budgets.

    The challenged versus neutral pair has the same *declared* model-call and
    token ceiling. Measured call counts and token use are separately checked;
    unbalanced cases remain visible and invalidate observed cost matching.
    """
    doc = _keys(value, {"schema_id", "experiment_id", "challenge_source",
                        "declared_budget", "cases"}, "experiment")
    if doc["schema_id"] != SCHEMA:
        raise ChallengeEvalError("unknown experiment schema")
    _id(doc["experiment_id"], "experiment_id")
    if doc["challenge_source"] not in ("scripted", "learned", "mixed"):
        raise ChallengeEvalError("challenge_source must disclose declared source kind")
    budget = _keys(doc["declared_budget"], {
        "core_only_calls", "matched_arm_calls", "max_total_tokens_per_case",
        "token_delta_tolerance"}, "declared budget")
    if budget["core_only_calls"] != 1 or type(budget["core_only_calls"]) is not int:
        raise ChallengeEvalError("core_only_calls must be exactly one")
    if budget["matched_arm_calls"] != 2 or type(budget["matched_arm_calls"]) is not int:
        raise ChallengeEvalError("matched_arm_calls must be exactly two")
    limit = _int(budget["max_total_tokens_per_case"], "token ceiling", 1, 10_000_000)
    tolerance = _int(budget["token_delta_tolerance"], "token tolerance", 0, limit)
    rows = doc["cases"]
    if type(rows) is not list or not 1 <= len(rows) <= 1000:
        raise ChallengeEvalError("cases must be a bounded nonempty array")

    parsed: list[dict[str, object]] = []
    seen: set[str] = set()
    for index, candidate in enumerate(rows):
        case = _keys(candidate, {"case_id", "expected_supports", *ARMS},
                     f"case {index}")
        case_id = _id(case["case_id"], "case_id")
        if case_id in seen:
            raise ChallengeEvalError("case IDs must be unique")
        seen.add(case_id)
        if type(case["expected_supports"]) is not bool:
            raise ChallengeEvalError("expected_supports must be a boolean gold label")
        outcome = {"case_id": case_id, "expected_supports": case["expected_supports"]}
        for arm_name in ARMS:
            arm_calls = 1 if arm_name == "core_only" else 2
            outcome[arm_name] = _arm(case[arm_name], arm_name, arm_calls, limit)
        parsed.append(outcome)

    def correct(case: dict[str, object], arm: str) -> bool:
        result = case[arm]
        return (result["status"] == "ok"
                and result["supports"] is case["expected_supports"])

    aggregates: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        aggregates[arm] = {
            "cases": len(parsed),
            "correct": sum(correct(case, arm) for case in parsed),
            "accuracy": sum(correct(case, arm) for case in parsed) / len(parsed),
            "contract_errors": sum(case[arm]["status"] == "output_contract_error"
                                   for case in parsed),
            "abstentions": sum(case[arm]["status"] == "abstain" for case in parsed),
            "model_calls_total": sum(case[arm]["model_calls"] for case in parsed),
            "total_tokens": sum(case[arm]["total_tokens"] for case in parsed),
            "elapsed_ms_total": sum(case[arm]["elapsed_ms"] for case in parsed),
        }

    wins = losses = ties = 0
    unmatched: list[str] = []
    case_results: list[dict[str, object]] = []
    for case in parsed:
        challenged = correct(case, "challenged")
        neutral = correct(case, "neutral_second_pass")
        if challenged and not neutral:
            wins += 1
        elif neutral and not challenged:
            losses += 1
        else:
            ties += 1
        c = case["challenged"]
        n = case["neutral_second_pass"]
        balanced = (c["model_calls"] == 2 and n["model_calls"] == 2
                    and abs(c["total_tokens"] - n["total_tokens"]) <= tolerance)
        if not balanced:
            unmatched.append(case["case_id"])
        case_results.append({
            "case_id": case["case_id"],
            "challenged_correct": challenged,
            "neutral_correct": neutral,
            "resource_usage_matched": balanced,
            "challenged_model_calls": c["model_calls"],
            "neutral_model_calls": n["model_calls"],
            "absolute_token_usage_delta": abs(c["total_tokens"] - n["total_tokens"]),
        })

    try:
        encoded = json.dumps(doc, sort_keys=True, separators=(",", ":"),
                             allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ChallengeEvalError("experiment cannot be canonically encoded") from exc
    return {
        "schema_id": "MOSAIC_CORE_CHALLENGE_PAIRED_REPORT_V1",
        "qualification": "UNTRUSTED_OFFLINE_PAIRED_SCORE_ONLY",
        "experiment_id": doc["experiment_id"],
        "challenge_source_declared_unverified": doc["challenge_source"],
        "input_sha256": sha256(encoded).hexdigest(),
        "cases": len(parsed),
        "arms": aggregates,
        "paired": {
            "comparison": "challenged_minus_neutral_second_pass",
            "wins": wins, "losses": losses, "ties": ties,
            "net_correct_delta": wins - losses,
            "declared_resource_budget_matched": True,
            "observed_resource_usage_matched": not unmatched,
            "unmatched_cases": unmatched,
            "case_results": case_results,
        },
        "gold_labels_independently_authenticated": False,
        "model_identity_attested": False,
        "independent_evaluator_custody": False,
        "neural_benefit_qualified": False,
        "claim_ceiling": (
            "Caller-supplied outcomes/gold/cost data only. Matched declared budgets "
            "and paired arithmetic do not prove actual model identity, independent "
            "critic benefit, frozen predeclaration or Mosaic neural qualification."
        ),
    }


def main() -> int:
    """Read one bounded strict JSON document on stdin and print the score."""
    raw = sys.stdin.buffer.read(1_048_577)
    if len(raw) > 1_048_576:
        print("input exceeds 1 MiB limit", file=sys.stderr)
        return 2
    try:
        document = strict_json_loads(raw.decode("utf-8"))
        report = evaluate_paired(document)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        print(f"invalid benchmark input: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
