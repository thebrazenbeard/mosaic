"""Adversarial, synthetic contracts; these are not neural P1 results."""
from __future__ import annotations

from copy import deepcopy
import unittest

from runtime.learned_continuity_gate import (
    LearnedContinuityError,
    evaluate_learned_continuity,
)


FAMILIES = (
    "REFERENT_PRESERVATION", "CORRECTION", "SUPERSESSION_CURRENTNESS",
    "DECISION_FINALITY", "AUTHORITY_CHANGE", "USER_PREFERENCE",
    "PROJECT_STATUS", "ARTIFACT_HANDOFF",
)


def fixture(*, benefit: bool = False) -> dict:
    cases = []
    for i in range(64):
        gold = f"opaque-gold-{i}"
        retrieval_correct = not benefit or i >= 32
        conditions = {}
        for name in ("RESET", "MOSAIC_STATE", "FULL_HISTORY", "ORDINARY_RETRIEVAL"):
            answer = (
                "UNKNOWN" if name == "RESET"
                else "wrong" if name == "ORDINARY_RETRIEVAL" and not retrieval_correct
                else gold
            )
            conditions[name] = {"answer": answer, "payload_bytes": 512,
                                "elapsed_ms": 100, "reported_tokens": 30}
        cases.append({"case_id": f"case-{i:03d}", "family": FAMILIES[i % 8],
                      "expected": gold, "superseded": [f"stale-{i}"],
                      "conditions": conditions})
    return {
        "schema_id": "MOSAIC_P1_LEARNED_CONTINUITY_EVIDENCE_V1",
        "experiment_id": "synthetic-contract-unit-test",
        "adapter": {
            "kind": "trained_latent_adapter", "checkpoint_sha256": "a" * 64,
            "training_corpus_sha256": "b" * 64, "parameters_trained": 1024,
            "optimizer_steps": 100, "train_heldout_disjoint_claimed": True,
            "receiver_consumed_continuous_state_claimed": True,
        },
        "subjects": {"specialist_a_sha256": "c" * 64,
                     "specialist_b_sha256": "d" * 64},
        "heldout_sha256": "e" * 64,
        "runs": [
            {"run_id": "repeat-one", "unload_before_receiver_claimed": True,
             "independent_replay_claimed": True, "max_payload_bytes": 1024,
             "max_reported_tokens_per_arm": 2500,
             "cases": deepcopy(cases)},
            {"run_id": "repeat-two", "unload_before_receiver_claimed": True,
             "independent_replay_claimed": True, "max_payload_bytes": 1024,
             "max_reported_tokens_per_arm": 2500,
             "cases": deepcopy(cases)},
        ],
    }


class LearnedContinuityGateTests(unittest.TestCase):
    def test_tie_with_conventional_retrieval_never_qualifies(self):
        report = evaluate_learned_continuity(fixture())
        self.assertFalse(report["declared_thresholds_met"])
        self.assertEqual(report["runs"][0]["mosaic_minus_retrieval_accuracy"], 0.0)
        self.assertEqual(report["runs"][0]["paired_one_sided_p"], 1.0)
        self.assertFalse(report["p1_learned_state_verified"])

    def test_even_positive_synthetic_case_is_unattested(self):
        report = evaluate_learned_continuity(fixture(benefit=True))
        self.assertTrue(report["declared_thresholds_met"])
        self.assertEqual(report["runs"][0]["paired_wins"], 32)
        self.assertEqual(report["runs"][0]["paired_losses"], 0)
        self.assertLess(report["runs"][0]["paired_one_sided_p"], 0.05)
        self.assertFalse(report["p1_learned_state_verified"])
        self.assertEqual(report["qualification"],
                         "UNATTESTED_DECLARED_EVIDENCE_NOT_P1_PASS")

    def test_retrieval_favoring_second_run_blocks_candidate(self):
        data = fixture(benefit=True)
        for item in data["runs"][1]["cases"]:
            item["conditions"]["MOSAIC_STATE"]["answer"] = "wrong"
        report = evaluate_learned_continuity(data)
        self.assertFalse(report["declared_thresholds_met"])

    def test_text_handoff_must_not_be_relabelled_learned_latent(self):
        data = fixture(benefit=True)
        data["adapter"]["kind"] = "text_state"
        with self.assertRaisesRegex(LearnedContinuityError, "trained latent"):
            evaluate_learned_continuity(data)

    def test_requires_two_clean_runs(self):
        data = fixture(benefit=True)
        data["runs"].pop()
        with self.assertRaisesRegex(LearnedContinuityError, "two.*runs"):
            evaluate_learned_continuity(data)

    def test_disjoint_training_and_holdout_subjects_required(self):
        data = fixture(benefit=True)
        data["heldout_sha256"] = data["adapter"]["training_corpus_sha256"]
        with self.assertRaisesRegex(LearnedContinuityError, "disjoint"):
            evaluate_learned_continuity(data)
        data = fixture(benefit=True)
        data["subjects"]["specialist_b_sha256"] = data["subjects"]["specialist_a_sha256"]
        with self.assertRaisesRegex(LearnedContinuityError, "distinct"):
            evaluate_learned_continuity(data)

    def test_changed_gold_or_missing_arm_across_repetitions_is_rejected(self):
        data = fixture(benefit=True)
        data["runs"][1]["cases"][0]["expected"] = "different-gold"
        with self.assertRaisesRegex(LearnedContinuityError, "same heldout"):
            evaluate_learned_continuity(data)
        data = fixture(benefit=True)
        del data["runs"][0]["cases"][0]["conditions"]["ORDINARY_RETRIEVAL"]
        with self.assertRaisesRegex(LearnedContinuityError, "four conditions"):
            evaluate_learned_continuity(data)

    def test_stale_state_and_unmatched_payload_budget_block_candidate(self):
        data = fixture(benefit=True)
        data["runs"][0]["cases"][0]["conditions"]["MOSAIC_STATE"]["answer"] = "stale-0"
        self.assertFalse(evaluate_learned_continuity(data)["declared_thresholds_met"])
        data = fixture(benefit=True)
        data["runs"][0]["cases"][0]["conditions"]["MOSAIC_STATE"]["payload_bytes"] = 1025
        with self.assertRaisesRegex(LearnedContinuityError, "payload budget"):
            evaluate_learned_continuity(data)

    def test_declared_token_budget_is_enforced_for_each_arm(self):
        data = fixture(benefit=True)
        data["runs"][0]["cases"][0]["conditions"]["MOSAIC_STATE"]["reported_tokens"] = 5000
        with self.assertRaisesRegex(LearnedContinuityError, "token budget"):
            evaluate_learned_continuity(data)

    def test_task_families_must_be_balanced_not_merely_present(self):
        data = fixture(benefit=True)
        data["runs"][0]["cases"][0]["family"] = FAMILIES[1]
        with self.assertRaisesRegex(LearnedContinuityError, "balanced"):
            evaluate_learned_continuity(data)


if __name__ == "__main__":
    unittest.main()
