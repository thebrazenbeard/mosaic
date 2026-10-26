"""Pure standard-library contract checks for a real lexical retrieval control."""

import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from p1_bm25_retrieval_control import (  # noqa: E402
    Observation,
    create_observations,
    retrieve_bm25,
    answer_from_evidence,
    evaluate_cases,
)


class BM25ControlTests(unittest.TestCase):
    def test_bm25_ranks_real_observation_documents_deterministically(self):
        docs = create_observations(0b1011001101010001)
        ranked = retrieve_bm25(docs, "observed flag state", top_k=8)
        self.assertEqual([item.position for item in ranked], list(range(15, 7, -1)))
        self.assertEqual(len(set(item.position for item in ranked)), 8)
        self.assertTrue(all("flag" in item.text for item in ranked))

    def test_lexical_relevance_can_override_recency(self):
        docs = [
            Observation(position=0, value=1, text="observed flag phase transition"),
            Observation(position=1, value=0, text="observed flag ordinary state"),
        ]
        selected = retrieve_bm25(docs, "phase transition", top_k=1)
        self.assertEqual([item.position for item in selected], [0])

    def test_full_retrieval_answers_from_all_evidence(self):
        docs = create_observations(0b1111000011110000)
        self.assertTrue(answer_from_evidence(retrieve_bm25(docs, "observed flag state", top_k=16), threshold=8, total=16))
        self.assertFalse(answer_from_evidence(retrieve_bm25(docs, "observed flag state", top_k=16), threshold=9, total=16))

    def test_bounded_retrieval_abstains_when_missing_data_changes_answer(self):
        docs = create_observations(0xFFFF)
        self.assertIsNone(answer_from_evidence(retrieve_bm25(docs, "observed flag state", top_k=8), threshold=15, total=16))
        docs = create_observations(0)
        self.assertFalse(answer_from_evidence(retrieve_bm25(docs, "observed flag state", top_k=8), threshold=9, total=16))

    def test_invalid_budgets_rejected_instead_of_silent_truncation(self):
        docs = create_observations(0)
        for budget in (0, -1, 17):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                retrieve_bm25(docs, "observed flag state", top_k=budget)

    def test_evidence_boolean_and_fractional_fields_cannot_count_as_observations(self):
        invalid = (
            Observation(position=True, value=1, text="bool position"),
            Observation(position=0, value=True, text="bool value"),
            Observation(position=0.5, value=1, text="fractional position"),
        )
        for observation in invalid:
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                answer_from_evidence([observation], threshold=1, total=16)

    def test_case_labels_must_be_binary_integers_not_boolean_or_fractional(self):
        for label in (True, False, 1.0, 0.0):
            with self.subTest(label=label), self.assertRaises(ValueError):
                evaluate_cases([{"mask": 0, "family": 0, "label": label}], top_k=16)

    def test_evaluation_records_coverage_and_byte_accounting(self):
        rows = [
            {"mask": 0xFFFF, "family": 7, "label": 1},
            {"mask": 0, "family": 0, "label": 0},
        ]
        result = evaluate_cases(rows, top_k=16)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["abstain"], 0)
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["packed_full_history_bytes_per_case"], 2)
        self.assertTrue(result["retrieved_text_bytes_total"] > 2 * len(rows))
        self.assertEqual(result["status"], "LEXICAL_RETRIEVAL_CONTROL_NOT_P1")


if __name__ == "__main__":
    unittest.main()
