"""Check the arithmetic of the published VN1 P0 toy receipt.

This is consistency checking, NOT attestation that its host executed the models.
"""

from __future__ import annotations

import copy
import json
from math import comb
from pathlib import Path
import unittest


RESULT = (Path(__file__).resolve().parents[1] / "experiments" / "results" /
          "P0_NEURAL_COUNT_V1.json")


class P0NeuralReceiptConsistencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_narrow_claim_is_not_general_p1(self):
        self.assertEqual(self.receipt["experimental_status"], "NARROW_TOY_PASS")
        self.assertTrue(self.receipt["limitations"]["eight_record_retrieval_only"])
        self.assertTrue(self.receipt["limitations"]["not_pretrained_llm_p1"])
        self.assertEqual(self.receipt["manifest"]["train_count"], 4096)
        self.assertEqual(self.receipt["manifest"]["dev_count"], 512)

    def test_predictions_fully_reproduce_metrics_and_gate(self):
        self.assertEqual(len(self.receipt["runs"]), 2)
        all_ids = []
        for row in self.receipt["runs"]:
            truth = row["truth"]
            self.assertEqual(len(truth), 64)
            self.assertEqual(sum(truth), 32)
            self.assertTrue(all(x in (0, 1) for x in truth))
            self.assertEqual(len(row["case_ids"]), 64)
            all_ids += row["case_ids"]
            conditions = {
                "MOSAIC_LEARNED_STATE": row["neural_predictions"],
                "ORDINARY_RETRIEVAL_8_RECORDS": row["retrieval_predictions"],
                "FULL_HISTORY": row["history_predictions"],
                "RESET": row["reset_predictions"],
                "POST_HOC_ZERO_STATE": row["post_hoc_ablations"]["zero_state"],
                "POST_HOC_ROLLED_STATE": row["post_hoc_ablations"]["rolled_state"],
            }
            for name, predicted in conditions.items():
                self.assertEqual(len(predicted), 64)
                self.assertTrue(all(x in (0, 1) for x in predicted))
                self.assertEqual(sum(x == y for x, y in zip(predicted, truth)) / 64,
                                 row["accuracies"][name])
            mosaic_ok = [x == y for x, y in zip(conditions["MOSAIC_LEARNED_STATE"], truth)]
            retrieval_ok = [x == y for x, y in zip(conditions["ORDINARY_RETRIEVAL_8_RECORDS"], truth)]
            wins = sum(a and not b for a, b in zip(mosaic_ok, retrieval_ok))
            losses = sum(b and not a for a, b in zip(mosaic_ok, retrieval_ok))
            self.assertEqual(wins, row["wins"])
            self.assertEqual(losses, row["losses"])
            n = wins + losses
            one_sided = sum(comb(n, k) for k in range(wins, n + 1)) / (2 ** n)
            self.assertEqual(one_sided, row["one_sided_exact_p"])
            self.assertEqual(row["run_gate_pass"],
                             row["accuracies"]["MOSAIC_LEARNED_STATE"] >= 0.8
                             and row["accuracies"]["MOSAIC_LEARNED_STATE"] -
                             row["accuracies"]["ORDINARY_RETRIEVAL_8_RECORDS"] >= 0.05
                             and one_sided < 0.05)
        self.assertEqual(len(set(all_ids)), 128)
        expected_status = ("NARROW_TOY_PASS" if all(
            row["run_gate_pass"] for row in self.receipt["runs"])
            else "HOLD_RETRIEVAL_ADVANTAGE_NOT_DEMONSTRATED")
        self.assertEqual(self.receipt["experimental_status"], expected_status)

    def test_artifact_hashes_are_present_and_distinct(self):
        weights = self.receipt["manifest"]["weights"]
        self.assertEqual(set(weights), {"source.safetensors", "target.safetensors",
                                        "retrieval.safetensors", "full.safetensors",
                                        "reset.safetensors"})
        self.assertEqual(len(set(weights.values())), len(weights))
        for digest in weights.values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)

    def test_success_status_cannot_survive_a_failing_run(self):
        corrupted = copy.deepcopy(self.receipt)
        row = corrupted["runs"][0]
        row["neural_predictions"] = row["retrieval_predictions"][:]
        row["accuracies"]["MOSAIC_LEARNED_STATE"] = row["accuracies"][
            "ORDINARY_RETRIEVAL_8_RECORDS"]
        row["wins"] = 0
        row["losses"] = 0
        row["one_sided_exact_p"] = 1.0
        row["run_gate_pass"] = False
        checker = type(self)("test_predictions_fully_reproduce_metrics_and_gate")
        checker.receipt = corrupted
        with self.assertRaises(AssertionError):
            checker.test_predictions_fully_reproduce_metrics_and_gate()


if __name__ == "__main__":
    unittest.main()
