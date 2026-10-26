"""V2 offline cost-matched learned-critic scorer (untrusted supplied receipts).

This module does not invoke or authenticate models. Every call, token count,
model ID, gold label and digest is provided by the caller. See the V2 protocol.
V1 remains intact for its separately budgeted scripted-challenge experiment.
"""
from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
import re
import sys

from .strict_json import loads as strict_json_loads


INPUT_SCHEMA = "MOSAIC_LEARNED_CRITIC_PAIRED_INPUT_V2"
REPORT_SCHEMA = "MOSAIC_LEARNED_CRITIC_PAIRED_REPORT_V2"
ARMS = ("core_only", "neutral_three_call", "challenged")
STATUSES = ("ok", "abstain", "output_contract_error")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
UTC_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
CALL_KEYS = {"role", "model_id", "input_tokens", "output_tokens",
             "elapsed_ms", "output_sha256"}
ARM_KEYS = {"status", "supports", "calls"}
CASE_KEYS = {"case_id", "prompt", "expected_supports", *ARMS}
CALL_ROLES = {
    "core_only": ("core_initial",),
    "neutral_three_call": ("core_initial", "neutral_critic", "core_reconsider"),
    "challenged": ("core_initial", "learned_critic", "core_reconsider"),
}


class LearnedChallengeEvalError(ValueError):
    """Invalid or insufficient caller-supplied learned-critic experiment record."""


def _keys(value: object, names: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != names:
        raise LearnedChallengeEvalError(f"{label} fields do not match the schema")
    return value


def _string(value: object, label: str, ceiling: int = 128) -> str:
    if (type(value) is not str or not 1 <= len(value) <= ceiling
            or value != value.strip()):
        raise LearnedChallengeEvalError(f"{label} must be a bounded canonical string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LearnedChallengeEvalError(f"{label} contains invalid Unicode") from exc
    return value


def _int(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise LearnedChallengeEvalError(
            f"{label} must be an integer in {minimum}..{maximum}")
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or SHA256.fullmatch(value) is None:
        raise LearnedChallengeEvalError(f"{label} must be lowercase SHA-256 hex")
    return value


def _canonical_digest(value: object, label: str) -> str:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise LearnedChallengeEvalError(f"{label} cannot be canonically encoded") from exc
    return sha256(encoded).hexdigest()


def corpus_digest(cases: list[dict]) -> str:
    """SHA-256 of *supplied* case IDs, prompts and golds in input order.

    Integrity against accidental change, not proof of pre-registration, secrecy,
    absence of leakage, or an independently administered frozen holdout.
    """
    return _canonical_digest([
        {"case_id": case["case_id"], "prompt": case["prompt"],
         "expected_supports": case["expected_supports"]}
        for case in cases
    ], "corpus")


def _arm(value: object, label: str, manifest: dict, token_ceiling: int) -> dict:
    arm = _keys(value, ARM_KEYS, label)
    status = arm["status"]
    if type(status) is not str or status not in STATUSES:
        raise LearnedChallengeEvalError(f"{label} has unknown status")
    if status == "ok":
        if type(arm["supports"]) is not bool:
            raise LearnedChallengeEvalError(f"{label} ok requires boolean answer")
    elif arm["supports"] is not None:
        raise LearnedChallengeEvalError(f"{label} non-ok cannot assert an answer")

    calls = arm["calls"]
    required = CALL_ROLES[label]
    if (type(calls) is not list or len(calls) > len(required)
            or (status == "ok" and len(calls) != len(required))):
        raise LearnedChallengeEvalError(
            f"{label} success requires {len(required)} individually recorded calls")

    total = elapsed = 0
    for index, raw_call in enumerate(calls):
        call = _keys(raw_call, CALL_KEYS, f"{label} call {index}")
        role = required[index]
        if type(call["role"]) is not str or call["role"] != role:
            raise LearnedChallengeEvalError(f"{label} call {index} role mismatch")
        expected_model = (manifest["critic_model_claimed_id"]
                          if index == 1 else manifest["core_model_claimed_id"])
        if type(call["model_id"]) is not str or call["model_id"] != expected_model:
            raise LearnedChallengeEvalError(
                f"{label} call {index} claimed model ID mismatch")
        input_tokens = _int(call["input_tokens"], f"{label} input_tokens", 1, 10_000_000)
        output_tokens = _int(call["output_tokens"], f"{label} output_tokens",
                             1 if status == "ok" else 0, 10_000_000)
        elapsed += _int(call["elapsed_ms"], f"{label} elapsed_ms", 0, 1_000_000_000)
        _digest(call["output_sha256"], f"{label} output_sha256")
        total += input_tokens + output_tokens
    if total > token_ceiling:
        raise LearnedChallengeEvalError(f"{label} exceeds declared total token ceiling")
    return {"status": status, "supports": arm["supports"], "calls": calls,
            "reported_total_tokens": total, "reported_elapsed_ms": elapsed}


def evaluate_learned_pair(value: object) -> dict:
    """Score three caller-reported arms, transparently disclosing mismatch.

    Real experimental qualification requires separately attested inference
    traces, sealed corpus, model custody and replication. Not provided here.
    """
    doc = _keys(value, {"schema_id", "experiment_id", "challenge_source",
                        "model_manifest", "corpus", "budget", "cases"}, "experiment")
    if doc["schema_id"] != INPUT_SCHEMA:
        raise LearnedChallengeEvalError("unknown experiment schema")
    _string(doc["experiment_id"], "experiment_id")
    if type(doc["challenge_source"]) is not str or doc["challenge_source"] != "learned":
        raise LearnedChallengeEvalError("V2 requires claimed learned critic source")

    manifest = _keys(doc["model_manifest"], {"core_model_claimed_id",
                                             "critic_model_claimed_id"}, "model manifest")
    core_id = _string(manifest["core_model_claimed_id"], "core model claimed ID")
    critic_id = _string(manifest["critic_model_claimed_id"], "critic model claimed ID")
    if core_id == critic_id:
        raise LearnedChallengeEvalError("Core and critic claimed model IDs must differ")

    corpus = _keys(doc["corpus"], {"corpus_id", "claimed_frozen_at_utc",
                                   "declared_sha256"}, "corpus")
    _string(corpus["corpus_id"], "corpus_id")
    frozen = corpus["claimed_frozen_at_utc"]
    if type(frozen) is not str or UTC_TIMESTAMP.fullmatch(frozen) is None:
        raise LearnedChallengeEvalError("corpus claimed frozen time must be UTC timestamp")
    try:
        datetime.strptime(frozen, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise LearnedChallengeEvalError("corpus claimed frozen time is invalid") from exc
    declared_digest = _digest(corpus["declared_sha256"], "corpus declared_sha256")

    budget = _keys(doc["budget"], {"matched_inference_calls",
                                   "max_total_tokens_per_arm",
                                   "token_delta_tolerance"}, "budget")
    if type(budget["matched_inference_calls"]) is not int or budget["matched_inference_calls"] != 3:
        raise LearnedChallengeEvalError("learned critic requires exactly three calls per matched arm")
    ceiling = _int(budget["max_total_tokens_per_arm"], "token ceiling", 1, 10_000_000)
    tolerance = _int(budget["token_delta_tolerance"], "token tolerance", 0, ceiling)

    rows = doc["cases"]
    if type(rows) is not list or not 1 <= len(rows) <= 1000:
        raise LearnedChallengeEvalError("cases must be bounded nonempty array")
    seen = set()
    parsed = []
    for index, candidate in enumerate(rows):
        case = _keys(candidate, CASE_KEYS, f"case {index}")
        case_id = _string(case["case_id"], "case_id")
        if case_id in seen:
            raise LearnedChallengeEvalError("case IDs must be unique")
        seen.add(case_id)
        _string(case["prompt"], f"case {case_id} prompt", 8192)
        if type(case["expected_supports"]) is not bool:
            raise LearnedChallengeEvalError("expected_supports must be a boolean gold label")
        outcome = {"case_id": case_id, "expected_supports": case["expected_supports"]}
        for label in ARMS:
            outcome[label] = _arm(case[label], label, manifest, ceiling)
        initial_calls = [outcome[arm]["calls"][0] for arm in ARMS
                         if outcome[arm]["calls"]]
        for initial in initial_calls[1:]:
            baseline = initial_calls[0]
            # One physical initial Core invocation is shared between arms.
            # Its entire exact-schema receipt (including elapsed time) must
            # agree, even though caller-reported values remain unauthenticated.
            if initial != baseline:
                raise LearnedChallengeEvalError(
                    f"{case_id} initial Core receipt differs across paired arms")
        parsed.append(outcome)

    observed_digest = corpus_digest(rows)
    if observed_digest != declared_digest:
        raise LearnedChallengeEvalError("corpus digest does not match supplied gold/prompt")

    def correct(row: dict, arm: str) -> bool:
        answer = row[arm]
        return answer["status"] == "ok" and answer["supports"] is row["expected_supports"]

    aggregates = {}
    for label in ARMS:
        count = sum(correct(row, label) for row in parsed)
        aggregates[label] = {
            "cases": len(parsed), "correct": count, "accuracy": count / len(parsed),
            "contract_errors": sum(row[label]["status"] == "output_contract_error"
                                   for row in parsed),
            "abstentions": sum(row[label]["status"] == "abstain" for row in parsed),
            "inference_calls": sum(len(row[label]["calls"]) for row in parsed),
            "reported_total_tokens": sum(row[label]["reported_total_tokens"]
                                          for row in parsed),
            "reported_elapsed_ms": sum(row[label]["reported_elapsed_ms"]
                                       for row in parsed),
        }

    wins = losses = ties = 0
    unmatched = []
    case_results = []
    for row in parsed:
        challenged_correct = correct(row, "challenged")
        neutral_correct = correct(row, "neutral_three_call")
        if challenged_correct and not neutral_correct:
            wins += 1
        elif neutral_correct and not challenged_correct:
            losses += 1
        else:
            ties += 1
        challenged = row["challenged"]
        neutral = row["neutral_three_call"]
        token_delta = abs(challenged["reported_total_tokens"]
                          - neutral["reported_total_tokens"])
        balanced = (len(challenged["calls"]) == 3 and len(neutral["calls"]) == 3
                    and challenged["status"] == "ok" and neutral["status"] == "ok"
                    and token_delta <= tolerance)
        if not balanced:
            unmatched.append(row["case_id"])
        case_results.append({
            "case_id": row["case_id"],
            "challenged_correct": challenged_correct,
            "neutral_correct": neutral_correct,
            "reported_cost_matched": balanced,
            "reported_token_delta": token_delta,
            "challenged_calls": len(challenged["calls"]),
            "neutral_calls": len(neutral["calls"]),
        })

    return {
        "schema_id": REPORT_SCHEMA,
        "qualification": "UNTRUSTED_OFFLINE_THREE_CALL_PAIRED_SCORE_ONLY",
        "experiment_id": doc["experiment_id"],
        "input_sha256": _canonical_digest(doc, "experiment"),
        "cases": len(parsed),
        "challenge_source_claimed_unverified": doc["challenge_source"],
        "arms": aggregates,
        "paired": {
            "comparison": "challenged_minus_neutral_three_call",
            "wins": wins, "losses": losses, "ties": ties,
            "net_correct_delta": wins - losses,
            "declared_inference_budget_matched": True,
            "all_reported_costs_matched": not unmatched,
            "unmatched_cases": unmatched,
            "case_results": case_results,
        },
        "corpus": {
            "corpus_id": corpus["corpus_id"],
            "claimed_frozen_at_utc": frozen,
            "declared_sha256": declared_digest,
            "observed_sha256": observed_digest,
        },
        "reported_token_counts_independently_measured": False,
        "corpus_freeze_independently_attested": False,
        "learned_weights_independently_attested": False,
        "model_identity_attested": False,
        "independent_evaluator_custody": False,
        "neural_benefit_qualified": False,
        "claim_ceiling": (
            "Supplied token/call receipts, gold labels and hashes only. Matching self-reported "
            "three-call budgets is not authenticated metering, a sealed unseen holdout, "
            "proof of learned weights or demonstrated specialist benefit."
        ),
    }


def main() -> int:
    raw = sys.stdin.buffer.read(1_048_577)
    if len(raw) > 1_048_576:
        print("input exceeds 1 MiB limit", file=sys.stderr)
        return 2
    try:
        document = strict_json_loads(raw.decode("utf-8"))
        report = evaluate_learned_pair(document)
    except (UnicodeDecodeError, ValueError, TypeError) as exc:
        print(f"invalid benchmark input: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
