"""V2 cost-matched learned-critic protocol arithmetic, never neural evidence."""

import copy
import json
import subprocess
import sys
import unittest

from runtime.challenge_eval_v2 import (
    LearnedChallengeEvalError,
    corpus_digest,
    evaluate_learned_pair,
)


CORE = "qwen-core-claim"
CRITIC = "granite-critic-claim"


def call(role, model_id, *, ins=40, outs=10, digest="a"):
    return {
        "role": role, "model_id": model_id,
        "input_tokens": ins, "output_tokens": outs,
        "elapsed_ms": 100, "output_sha256": digest * 64,
    }


def fixture():
    cases = []
    for ident, gold, neutral, challenged in (
        ("gain", True, False, True),
        ("regression", False, False, True),
        ("tie", True, True, True),
    ):
        cases.append({
            "case_id": ident,
            "prompt": "Decide a frozen yes/no proposition: " + ident,
            "expected_supports": gold,
            "core_only": {
                "status": "ok", "supports": False,
                "calls": [call("core_initial", CORE, digest="a")],
            },
            "neutral_three_call": {
                "status": "ok", "supports": neutral,
                "calls": [
                    call("core_initial", CORE, digest="a"),
                    call("neutral_critic", CRITIC, digest="b"),
                    call("core_reconsider", CORE, digest="c"),
                ],
            },
            "challenged": {
                "status": "ok", "supports": challenged,
                "calls": [
                    call("core_initial", CORE, digest="a"),
                    call("learned_critic", CRITIC, digest="d"),
                    call("core_reconsider", CORE, digest="e"),
                ],
            },
        })
    return {
        "schema_id": "MOSAIC_LEARNED_CRITIC_PAIRED_INPUT_V2",
        "experiment_id": "local-synthetic-cost-match",
        "challenge_source": "learned",
        "model_manifest": {
            "core_model_claimed_id": CORE,
            "critic_model_claimed_id": CRITIC,
        },
        "corpus": {
            "corpus_id": "synthetic-fixture-only",
            "claimed_frozen_at_utc": "2026-10-25T00:00:00Z",
            "declared_sha256": corpus_digest(cases),
        },
        "budget": {
            "matched_inference_calls": 3,
            "max_total_tokens_per_arm": 400,
            "token_delta_tolerance": 0,
        },
        "cases": cases,
    }


class LearnedChallengeEvalTests(unittest.TestCase):
    def test_three_call_neutral_and_learned_pairs_score_all_cases(self):
        report = evaluate_learned_pair(fixture())
        self.assertEqual(report["cases"], 3)
        self.assertEqual(report["arms"]["core_only"]["correct"], 1)
        self.assertEqual(report["arms"]["neutral_three_call"]["correct"], 2)
        self.assertEqual(report["arms"]["challenged"]["correct"], 2)
        self.assertEqual(report["paired"]["wins"], 1)
        self.assertEqual(report["paired"]["losses"], 1)
        self.assertEqual(report["paired"]["ties"], 1)
        self.assertEqual(report["paired"]["net_correct_delta"], 0)
        self.assertTrue(report["paired"]["all_reported_costs_matched"])
        self.assertEqual(report["arms"]["neutral_three_call"]["inference_calls"], 9)
        self.assertEqual(report["arms"]["challenged"]["inference_calls"], 9)
        self.assertFalse(report["neural_benefit_qualified"])
        self.assertFalse(report["learned_weights_independently_attested"])
        self.assertFalse(report["corpus_freeze_independently_attested"])

    def test_incomplete_three_call_or_zero_token_success_rejected(self):
        mutations = (
            lambda x: x["cases"][0]["challenged"]["calls"].pop(1),
            lambda x: x["cases"][0]["neutral_three_call"]["calls"].pop(),
            lambda x: x["cases"][0]["core_only"]["calls"].clear(),
            lambda x: x["cases"][0]["challenged"]["calls"][1].update(output_tokens=0),
            lambda x: x["cases"][0]["neutral_three_call"]["calls"][0].update(input_tokens=0),
        )
        for mutate in mutations:
            data = fixture()
            mutate(data)
            with self.subTest(mutation=str(mutate)):
                with self.assertRaises(LearnedChallengeEvalError):
                    evaluate_learned_pair(data)

    def test_middle_call_uses_same_claimed_critic_model_but_distinct_role(self):
        mutations = (
            lambda x: x["cases"][0]["neutral_three_call"]["calls"][1].update(
                model_id="other-model"),
            lambda x: x["cases"][0]["challenged"]["calls"][1].update(
                model_id=CORE),
            lambda x: x["cases"][0]["challenged"]["calls"][1].update(
                role="neutral_critic"),
            lambda x: x["model_manifest"].update(critic_model_claimed_id=CORE),
        )
        for mutate in mutations:
            data = fixture()
            mutate(data)
            with self.subTest(mutation=str(mutate)):
                with self.assertRaises(LearnedChallengeEvalError):
                    evaluate_learned_pair(data)

    def test_same_initial_core_answer_required_for_paired_arms(self):
        data = fixture()
        data["cases"][0]["challenged"]["calls"][0]["output_sha256"] = "f" * 64
        with self.assertRaisesRegex(LearnedChallengeEvalError, "initial"):
            evaluate_learned_pair(data)
        data = fixture()
        data["cases"][0]["challenged"]["calls"][0]["input_tokens"] += 1
        with self.assertRaisesRegex(LearnedChallengeEvalError, "initial"):
            evaluate_learned_pair(data)
        # The initial Core invocation is physically shared, so a different
        # duration cannot belong to the same receipt even when all model,
        # output and token fields are otherwise identical.
        data = fixture()
        data["cases"][0]["challenged"]["calls"][0]["elapsed_ms"] += 1
        with self.assertRaisesRegex(LearnedChallengeEvalError, "initial"):
            evaluate_learned_pair(data)

    def test_token_imbalance_is_disclosed_not_silently_cost_matched(self):
        data = fixture()
        data["cases"][0]["challenged"]["calls"][1]["output_tokens"] += 20
        report = evaluate_learned_pair(data)
        self.assertFalse(report["paired"]["all_reported_costs_matched"])
        self.assertEqual(report["paired"]["unmatched_cases"], ["gain"])
        self.assertEqual(report["paired"]["case_results"][0]["reported_token_delta"], 20)
        self.assertFalse(report["neural_benefit_qualified"])

    def test_declared_frozen_digest_binds_gold_and_prompt_but_is_unattested(self):
        data = fixture()
        original = evaluate_learned_pair(data)
        self.assertEqual(original["corpus"]["observed_sha256"],
                         data["corpus"]["declared_sha256"])
        self.assertFalse(original["corpus_freeze_independently_attested"])
        for field, new in (("expected_supports", False), ("prompt", "leaked prompt")):
            changed = fixture()
            changed["cases"][0][field] = new
            with self.subTest(field=field):
                with self.assertRaisesRegex(LearnedChallengeEvalError, "corpus"):
                    evaluate_learned_pair(changed)

    def test_non_ok_cannot_claim_answer_or_correctness(self):
        data = fixture()
        data["cases"][0]["challenged"]["status"] = "output_contract_error"
        data["cases"][0]["challenged"]["supports"] = None
        data["cases"][0]["challenged"]["calls"].pop()
        report = evaluate_learned_pair(data)
        self.assertEqual(report["arms"]["challenged"]["correct"], 1)
        self.assertFalse(report["paired"]["all_reported_costs_matched"])
        data["cases"][0]["challenged"]["supports"] = True
        with self.assertRaises(LearnedChallengeEvalError):
            evaluate_learned_pair(data)

    def test_rejects_duplicate_case_extra_keys_and_malformed_types(self):
        mutations = (
            lambda x: x["cases"][1].update(case_id="gain"),
            lambda x: x["cases"][0]["challenged"].update(verified=True),
            lambda x: x["cases"][0]["challenged"]["calls"][0].update(output_tokens=True),
            lambda x: x["cases"][0].update(expected_supports="false"),
            lambda x: x["budget"].update(matched_inference_calls=2),
            lambda x: x["corpus"].update(claimed_frozen_at_utc="yesterday"),
        )
        for mutate in mutations:
            data = fixture()
            mutate(data)
            with self.subTest(mutation=str(mutate)):
                with self.assertRaises(LearnedChallengeEvalError):
                    evaluate_learned_pair(data)

    def test_cli_rejects_duplicate_json_keys_and_limits_input(self):
        proc = subprocess.run(
            [sys.executable, "-m", "runtime.challenge_eval_v2"],
            input='{"schema_id":"bad","schema_id":"bad-again"}',
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("duplicate JSON object key", proc.stderr)
        proc = subprocess.run(
            [sys.executable, "-m", "runtime.challenge_eval_v2"],
            input="x" * 1_048_577,
            text=True, capture_output=True, check=False,
        )
        self.assertEqual(proc.returncode, 2)
        self.assertIn("1 MiB", proc.stderr)


if __name__ == "__main__":
    unittest.main()
