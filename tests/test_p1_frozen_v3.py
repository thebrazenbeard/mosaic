"""Immutable-input regression gate for the declared P1 V3 research corpus.

These tests are deterministic Python oracles, not model inference or evidence
that the corpus freeze was witnessed by an independent steward.
"""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import unittest

from runtime.challenge_eval_v2 import corpus_digest
from runtime.neural_core import NeuralOutputError, _parse_position
from runtime.strict_json import loads as strict_json_loads
from runtime.thinking_core import DeliberationTask


FILE = (Path(__file__).resolve().parent.parent / "experiments" / "fixtures" /
        "MOSAIC_P1_PYTHON_SEMANTICS_FROZEN_V3.json")
DIGEST = "7d482b7b4561ee049ff8d153d151dc48ed17182e89e549d31aa7e92f4ccc4477"


class FrozenP1V3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = strict_json_loads(FILE.read_text(encoding="utf-8"))
        cls.rows = cls.fixture["cases"]

    def test_declared_freeze_and_exact_digest(self):
        self.assertEqual(set(self.fixture), {
            "schema_id", "corpus_id", "claimed_frozen_at_utc",
            "declared_sha256", "cases"
        })
        self.assertEqual(
            self.fixture["schema_id"], "MOSAIC_P1_DECLARED_FROZEN_CASES_V3"
        )
        self.assertEqual(
            self.fixture["corpus_id"], "p1-python-semantics-balanced-v3"
        )
        self.assertEqual(
            self.fixture["claimed_frozen_at_utc"], "2026-10-26T09:17:38Z"
        )
        datetime.strptime(
            self.fixture["claimed_frozen_at_utc"], "%Y-%m-%dT%H:%M:%SZ"
        )
        self.assertEqual(self.fixture["declared_sha256"], DIGEST)
        self.assertEqual(corpus_digest(self.rows), DIGEST)

    def test_twelve_unique_balanced_cases_and_no_old_pilot_prompts(self):
        self.assertEqual(len(self.rows), 12)
        self.assertEqual(
            Counter(row["expected_supports"] for row in self.rows),
            Counter({True: 6, False: 6})
        )
        self.assertEqual(len(set(row["case_id"] for row in self.rows)), 12)
        self.assertEqual(len(set(row["prompt"] for row in self.rows)), 12)
        for row in self.rows:
            self.assertEqual(set(row), {
                "case_id", "prompt", "expected_supports"
            })
            self.assertTrue(row["prompt"].startswith("In Python 3, is the proposition "))
            self.assertNotIn("len([])", row["prompt"])
            self.assertIs(type(row["expected_supports"]), bool)
            # proposition_id is sent to the Core. It must not reveal gold.
            self.assertRegex(row["case_id"], r"\Ap1c\d{2}\Z")
            self.assertNotIn("true", row["case_id"])
            self.assertNotIn("false", row["case_id"])
        self.assertEqual(
            [row["case_id"] for row in self.rows],
            [f"p1c{n:02d}" for n in range(1, 13)]
        )

    def test_gold_labels_against_independent_python_operations(self):
        # Controlled, hand-specified operations; never eval fixture text.
        oracles = {
            "p1c01": sum([2, 3, 5]) == 11,
            "p1c02": sorted([3, 1, 2]) == [1, 2, 3],
            "p1c03": 7 // 2 == 4,
            "p1c04": bool("") is False,
            "p1c05": len(set([1, 1, 2])) == 2,
            "p1c06": sum(range(4)) == 10,
            "p1c07": sum([2, 3, 5]) == 10,
            "p1c08": bool("0") is False,
            "p1c09": sorted([3, 1, 2]) == [3, 2, 1],
            "p1c10": len(set([1, 1, 2])) == 3,
            "p1c11": 7 // 2 == 3,
            "p1c12": sum(range(4)) == 6,
        }
        self.assertEqual(set(oracles), {row["case_id"] for row in self.rows})
        for row in self.rows:
            with self.subTest(case_id=row["case_id"]):
                self.assertIs(oracles[row["case_id"]], row["expected_supports"])

    def test_core_fully_bound_four_field_answer_not_bare_stance(self):
        for row in self.rows:
            with self.subTest(case_id=row["case_id"]):
                task = DeliberationTask(
                    case_id=row["case_id"],
                    proposition_id=row["case_id"],
                    question=row["prompt"]
                )
                position = {
                    "proposition_id": row["case_id"],
                    "supports": row["expected_supports"],
                    "reasoning": "Deterministic Python statement evaluated.",
                    "acknowledged_objections": [],
                }
                parsed = _parse_position(json.dumps(position), task)
                self.assertIs(parsed.supports, row["expected_supports"])
                with self.assertRaises(NeuralOutputError):
                    _parse_position(
                        json.dumps({"supports": row["expected_supports"]}),
                        task
                    )

    def test_any_gold_flip_invalidates_frozen_digest(self):
        copy_rows = [dict(item) for item in self.rows]
        copy_rows[0]["expected_supports"] = not copy_rows[0]["expected_supports"]
        self.assertNotEqual(corpus_digest(copy_rows), DIGEST)


if __name__ == "__main__":
    unittest.main()
