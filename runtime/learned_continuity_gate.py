"""Strict comparison of *unattested* cross-model learned-continuity receipts.

This module never calls models, trains adapters, or authenticates a claimed
hardware operation. It cannot award P1 PASS, even if supplied data is ideal.
"""

from __future__ import annotations

from collections import Counter
from math import comb
import re

from .handoff import CONDITIONS


SCHEMA = "MOSAIC_P1_LEARNED_CONTINUITY_EVIDENCE_V1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
FAMILIES = frozenset({
    "REFERENT_PRESERVATION", "CORRECTION", "SUPERSESSION_CURRENTNESS",
    "DECISION_FINALITY", "AUTHORITY_CHANGE", "USER_PREFERENCE",
    "PROJECT_STATUS", "ARTIFACT_HANDOFF",
})


class LearnedContinuityError(ValueError):
    pass


def _object(value: object, expected: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != expected:
        raise LearnedContinuityError(f"{label} must contain exactly {sorted(expected)}")
    return value


def _string(value: object, label: str, *, maximum: int = 256) -> str:
    if type(value) is not str or not 1 <= len(value) <= maximum or value != value.strip():
        raise LearnedContinuityError(f"{label} must be bounded canonical text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise LearnedContinuityError(f"{label} invalid Unicode") from exc
    return value


def _digest(value: object, label: str) -> str:
    if type(value) is not str or SHA256.fullmatch(value) is None:
        raise LearnedContinuityError(f"{label} must be lower-case sha256")
    return value


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise LearnedContinuityError(f"{label} outside bounded integer range")
    return value


def _flag(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise LearnedContinuityError(f"{label} must be boolean")
    return value


def _paired_p(wins: int, losses: int) -> float:
    """Exact one-sided conditional binomial test of discordant case outcomes."""
    n = wins + losses
    if not n:
        return 1.0
    return sum(comb(n, k) for k in range(wins, n + 1)) / (2 ** n)


def evaluate_learned_continuity(receipt: object) -> dict[str, object]:
    value = _object(receipt, {"schema_id", "experiment_id", "adapter", "subjects",
                              "heldout_sha256", "runs"}, "experiment")
    if value["schema_id"] != SCHEMA:
        raise LearnedContinuityError("unsupported learned continuity schema")
    identifier = _string(value["experiment_id"], "experiment ID", maximum=128)
    adapter = _object(value["adapter"], {
        "kind", "checkpoint_sha256", "training_corpus_sha256",
        "parameters_trained", "optimizer_steps", "train_heldout_disjoint_claimed",
        "receiver_consumed_continuous_state_claimed",
    }, "adapter")
    if adapter["kind"] != "trained_latent_adapter":
        raise LearnedContinuityError("trained latent adapter is required; text is not learned state")
    adapter_sha = _digest(adapter["checkpoint_sha256"], "adapter checkpoint")
    train_sha = _digest(adapter["training_corpus_sha256"], "training corpus")
    hold_sha = _digest(value["heldout_sha256"], "heldout corpus")
    if train_sha == hold_sha:
        raise LearnedContinuityError("training and heldout must be disjoint")
    _integer(adapter["parameters_trained"], "trained parameters", 1, 100_000_000)
    _integer(adapter["optimizer_steps"], "optimizer steps", 1, 10_000_000)
    split_claim = _flag(adapter["train_heldout_disjoint_claimed"], "split claim")
    injected_claim = _flag(adapter["receiver_consumed_continuous_state_claimed"],
                           "latent receiver claim")

    subjects = _object(value["subjects"], {
        "specialist_a_sha256", "specialist_b_sha256"}, "model subjects")
    a_sha = _digest(subjects["specialist_a_sha256"], "specialist A")
    b_sha = _digest(subjects["specialist_b_sha256"], "specialist B")
    if a_sha == b_sha or adapter_sha in {a_sha, b_sha}:
        raise LearnedContinuityError("distinct specialist and adapter subjects required")

    runs = value["runs"]
    if type(runs) is not list or not 2 <= len(runs) <= 4:
        raise LearnedContinuityError("at least two and at most four clean runs required")
    seen_runs: set[str] = set()
    reference_cases: dict[str, tuple] | None = None
    reported: list[dict[str, object]] = []
    run_candidates: list[bool] = []
    for run in runs:
        item = _object(run, {"run_id", "unload_before_receiver_claimed",
                             "independent_replay_claimed", "max_payload_bytes",
                             "max_reported_tokens_per_arm", "cases"}, "run")
        run_id = _string(item["run_id"], "run ID", maximum=128)
        if run_id in seen_runs:
            raise LearnedContinuityError("duplicate run ID")
        seen_runs.add(run_id)
        unload_claim = _flag(item["unload_before_receiver_claimed"], "unload claim")
        replay_claim = _flag(item["independent_replay_claimed"], "replay claim")
        cap = _integer(item["max_payload_bytes"], "payload budget", 1, 8192)
        token_cap = _integer(item["max_reported_tokens_per_arm"],
                             "token budget", 1, 10_000_000)
        cases = item["cases"]
        if type(cases) is not list or not 64 <= len(cases) <= 512:
            raise LearnedContinuityError("run requires 64..512 heldout cases")
        case_subjects: dict[str, tuple] = {}
        counts = {name: 0 for name in CONDITIONS}
        stale = {name: 0 for name in CONDITIONS}
        payloads = {name: 0 for name in CONDITIONS}
        elapsed = {name: 0 for name in CONDITIONS}
        tokens = {name: 0 for name in CONDITIONS}
        wins = losses = 0
        families: Counter[str] = Counter()
        for case in cases:
            row = _object(case, {"case_id", "family", "expected", "superseded", "conditions"},
                          "case")
            case_id = _string(row["case_id"], "case ID", maximum=128)
            family = _string(row["family"], "case family", maximum=64)
            if family not in FAMILIES:
                raise LearnedContinuityError("unknown case family")
            families[family] += 1
            expected = _string(row["expected"], "gold value", maximum=512)
            old = row["superseded"]
            if (type(old) is not list or len(old) > 16 or
                    any(type(v) is not str for v in old)):
                raise LearnedContinuityError("invalid superseded values")
            superseded = tuple(_string(v, "superseded value", maximum=512) for v in old)
            if case_id in case_subjects:
                raise LearnedContinuityError("duplicate case ID")
            case_subjects[case_id] = (family, expected, superseded)
            arms = _object(row["conditions"], set(CONDITIONS), "four conditions")
            correct: dict[str, bool] = {}
            for name in CONDITIONS:
                arm = _object(arms[name], {"answer", "payload_bytes", "elapsed_ms",
                                          "reported_tokens"}, "arm")
                answer = _string(arm["answer"], "answer", maximum=512)
                size = _integer(arm["payload_bytes"], "payload size", 0, 8192)
                if size > cap:
                    raise LearnedContinuityError("payload budget exceeded")
                millis = _integer(arm["elapsed_ms"], "elapsed time", 0, 3_600_000)
                claimed_tokens = _integer(arm["reported_tokens"], "token cost", 0, 100_000)
                # RESET cannot claim continuity by returning an UNKNOWN sentinel.
                is_correct = answer == expected and not (name == "RESET" and answer == "UNKNOWN")
                correct[name] = is_correct
                counts[name] += int(is_correct)
                stale[name] += int(answer in superseded and not is_correct)
                payloads[name] += size
                elapsed[name] += millis
                tokens[name] += claimed_tokens
            wins += int(correct["MOSAIC_STATE"] and not correct["ORDINARY_RETRIEVAL"])
            losses += int(correct["ORDINARY_RETRIEVAL"] and not correct["MOSAIC_STATE"])
        if (len(cases) % len(FAMILIES) or set(families) != FAMILIES or
                any(families[family] != len(cases) // len(FAMILIES)
                    for family in FAMILIES)):
            raise LearnedContinuityError("all eight task families must be balanced")
        if reference_cases is not None and case_subjects != reference_cases:
            raise LearnedContinuityError("repetitions require the same heldout case subjects")
        reference_cases = case_subjects
        if any(total_tokens > token_cap for total_tokens in tokens.values()):
            raise LearnedContinuityError("reported token budget exceeded")
        total = len(cases)
        improvement = (counts["MOSAIC_STATE"] - counts["ORDINARY_RETRIEVAL"]) / total
        p = _paired_p(wins, losses)
        candidate = (unload_claim and replay_claim and
                     counts["MOSAIC_STATE"] > counts["RESET"] and
                     counts["FULL_HISTORY"] == total and
                     stale["MOSAIC_STATE"] <= stale["RESET"] and
                     improvement >= 0.05 and p < 0.05)
        run_candidates.append(candidate)
        reported.append({
            "run_id": run_id, "cases": total, "correct": counts,
            "accuracy": {name: counts[name] / total for name in CONDITIONS},
            "stale_errors": stale, "payload_bytes_total": payloads,
            "reported_elapsed_ms_total": elapsed, "reported_token_total": tokens,
            "paired_wins": wins, "paired_losses": losses,
            "paired_ties": total - wins - losses,
            "paired_one_sided_p": p,
            "mosaic_minus_retrieval_accuracy": improvement,
            "declared_run_threshold_met": candidate,
        })

    return {
        "schema_id": "MOSAIC_P1_LEARNED_CONTINUITY_SCORE_V1",
        "experiment_id": identifier,
        "qualification": "UNATTESTED_DECLARED_EVIDENCE_NOT_P1_PASS",
        "run_count": len(runs), "case_count_per_run": len(reference_cases or {}),
        "declared_thresholds_met": split_claim and injected_claim and all(run_candidates),
        "p1_learned_state_verified": False,
        "independent_corpus_custody_verified": False,
        "physical_worker_and_unload_verified": False,
        "trained_checkpoint_and_injection_verified": False,
        "budget_and_metering_independently_verified": False,
        "runs": reported,
        "claim_ceiling": (
            "Only schema-validated, self-reported predictions, costs, model digests, "
            "train/holdout disjointness and adapter usage. This is not hardware, "
            "training, model-identity, leakage or Mosaic P1 qualification."
        ),
    }
