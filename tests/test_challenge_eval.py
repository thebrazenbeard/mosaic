"""Offline paired scoring only; supplied outcomes are not neural evidence."""
import copy
import subprocess
import sys
import unittest

from runtime.challenge_eval import ChallengeEvalError, evaluate_paired


def row(supports, *, status="ok", tokens=120, calls=2, ms=100):
    return {
        "status": status, "supports": supports, "model_calls": calls,
        "total_tokens": tokens, "elapsed_ms": ms,
    }


def fixture():
    return {
        "schema_id": "MOSAIC_CORE_CHALLENGE_PAIRED_INPUT_V1",
        "experiment_id": "synthetic-001",
        "challenge_source": "scripted",
        "declared_budget": {
            "core_only_calls": 1, "matched_arm_calls": 2,
            "max_total_tokens_per_case": 200,
            "token_delta_tolerance": 10,
        },
        "cases": [
            {
                "case_id": "gain", "expected_supports": True,
                "core_only": row(False, tokens=70, calls=1),
                "neutral_second_pass": row(False, tokens=130),
                "challenged": row(True, tokens=120),
            },
            {
                "case_id": "regression", "expected_supports": False,
                "core_only": row(True, tokens=80, calls=1),
                "neutral_second_pass": row(False, tokens=153),
                "challenged": row(True, tokens=150),
            },
            {
                "case_id": "tie", "expected_supports": True,
                "core_only": row(True, tokens=60, calls=1),
                "neutral_second_pass": row(True, tokens=110),
                "challenged": row(True, tokens=110),
            },
        ],
    }


class ChallengeEvalTests(unittest.TestCase):
    def test_paired_win_loss_tie_and_no_fabricated_neural_benefit(self):
        report = evaluate_paired(fixture())
        self.assertEqual(report["cases"], 3)
        self.assertEqual(report["arms"]["core_only"]["correct"], 1)
        self.assertEqual(report["arms"]["neutral_second_pass"]["correct"], 2)
        self.assertEqual(report["arms"]["challenged"]["correct"], 2)
        self.assertEqual(report["paired"]["wins"], 1)
        self.assertEqual(report["paired"]["losses"], 1)
        self.assertEqual(report["paired"]["ties"], 1)
        self.assertEqual(report["paired"]["net_correct_delta"], 0)
        self.assertEqual(
            [(item["case_id"], item["challenged_correct"], item["neutral_correct"])
             for item in report["paired"]["case_results"]],
            [("gain", True, False), ("regression", False, True), ("tie", True, True)],
        )
        self.assertTrue(report["paired"]["observed_resource_usage_matched"])
        self.assertFalse(report["neural_benefit_qualified"])
        self.assertFalse(report["gold_labels_independently_authenticated"])

    def test_token_or_call_imbalance_is_reported_not_silently_called_matched(self):
        supplied = fixture()
        supplied["cases"][0]["challenged"]["total_tokens"] = 145
        supplied["cases"][1]["challenged"]["model_calls"] = 1
        report = evaluate_paired(supplied)
        self.assertFalse(report["paired"]["observed_resource_usage_matched"])
        self.assertEqual(report["paired"]["unmatched_cases"], ["gain", "regression"])
        self.assertEqual([item["resource_usage_matched"] for item in
                          report["paired"]["case_results"]], [False, False, True])

    def test_abstain_and_contract_error_cannot_count_as_correct(self):
        supplied = fixture()
        supplied["cases"][2]["challenged"] = row(
            None, status="output_contract_error", tokens=50, calls=1)
        report = evaluate_paired(supplied)
        self.assertEqual(report["arms"]["challenged"]["correct"], 1)
        self.assertEqual(report["arms"]["challenged"]["contract_errors"], 1)
        self.assertFalse(report["paired"]["observed_resource_usage_matched"])

    def test_rejects_duplicate_case_ids_unbounded_usage_and_boolean_as_integer(self):
        changes = (
            lambda x: x["cases"][1].update(case_id="gain"),
            lambda x: x["cases"][0]["challenged"].update(total_tokens=201),
            lambda x: x["cases"][0]["challenged"].update(model_calls=True),
            lambda x: x["cases"][0]["challenged"].update(elapsed_ms=-1),
            lambda x: x["cases"][0]["challenged"].update(supports=None),
        )
        for change in changes:
            supplied = copy.deepcopy(fixture())
            change(supplied)
            with self.subTest(change=str(change)):
                with self.assertRaises(ChallengeEvalError):
                    evaluate_paired(supplied)

    def test_rejects_extra_fields_and_noncanonical_budget(self):
        supplied = fixture()
        supplied["cases"][0]["challenged"]["verified"] = True
        with self.assertRaises(ChallengeEvalError):
            evaluate_paired(supplied)
        supplied = fixture()
        supplied["declared_budget"]["matched_arm_calls"] = 3
        with self.assertRaises(ChallengeEvalError):
            evaluate_paired(supplied)

    def test_canonical_digest_is_stable_to_mapping_key_order_not_outcome_changes(self):
        a = fixture()
        b = {key: a[key] for key in reversed(tuple(a.keys()))}
        self.assertEqual(evaluate_paired(a)["input_sha256"],
                         evaluate_paired(b)["input_sha256"])
        changed = copy.deepcopy(a)
        changed["cases"][0]["challenged"]["supports"] = False
        self.assertNotEqual(evaluate_paired(a)["input_sha256"],
                            evaluate_paired(changed)["input_sha256"])

    def test_cli_rejects_duplicate_json_keys_without_scoring(self):
        run = subprocess.run(
            [sys.executable, "-m", "runtime.challenge_eval"],
            input='{"schema_id":"first","schema_id":"second"}',
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(run.returncode, 2)
        self.assertEqual(run.stdout, "")
        self.assertIn("duplicate JSON object key", run.stderr)


if __name__ == "__main__":
    unittest.main()
