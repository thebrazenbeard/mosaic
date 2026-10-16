"""Frozen live neural Core challenge benchmark, without doing live inference in tests."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.live_core_challenge_v1 import (
    load_frozen_cases, evaluate_case, aggregate, FrozenCaseError
)

ROOT=Path(__file__).resolve().parents[1]
CASES=ROOT/"experiments/MOSAIC_LIVE_CORE_CHALLENGE_CASES_20261016_V1.json"


class ScriptedBackend:
    def __init__(self, initial, revised=None):
        self.responses=[initial]+([] if revised is None else [revised])
        self.messages=[]

    def __call__(self, messages):
        self.messages.append(messages)
        return json.dumps(self.responses.pop(0))


def position(supports, reason="Reasoned answer", acknowledged=()):
    return {
        "proposition_id":"claim",
        "supports":supports,
        "reasoning":reason,
        "acknowledged_objections":list(acknowledged),
    }


class LiveChallengeProtocolTests(unittest.TestCase):
    def test_frozen_cases_have_balanced_truth_and_misleading_critics(self):
        frozen,cases,sha=load_frozen_cases(CASES)
        self.assertEqual(len(cases),10)
        self.assertEqual({case["id"] for case in cases},{
            "T01","T02","T03","T04","T05",
            "T06","T07","T08","T09","T10"})
        self.assertEqual(sum(c["specialist_quality"]=="CORRECT" for c in cases),5)
        self.assertEqual(sum(c["specialist_quality"]=="MISLEADING" for c in cases),5)
        self.assertEqual(sha,hashlib.sha256(CASES.read_bytes()).hexdigest())

    def test_no_dissent_does_not_spend_another_model_call(self):
        _,cases,_=load_frozen_cases(CASES)
        c=next(c for c in cases if c["id"]=="T02")
        backend=ScriptedBackend(position(False))
        result=evaluate_case(c,backend)
        self.assertEqual(result["initial_supports"],False)
        self.assertEqual(result["challenged_supports"],False)
        self.assertTrue(result["baseline_correct"])
        self.assertTrue(result["challenged_correct"])
        self.assertEqual(result["model_generation_calls"],1)
        self.assertFalse(result["challenge_triggered_reconsideration"])

    def test_misleading_specialist_can_make_a_correct_core_wrong(self):
        _,cases,_=load_frozen_cases(CASES)
        c=next(c for c in cases if c["id"]=="T01")
        backend=ScriptedBackend(position(True),position(False,acknowledged=("scripted-critic",)))
        result=evaluate_case(c,backend)
        self.assertTrue(result["baseline_correct"])
        self.assertFalse(result["challenged_correct"])
        self.assertEqual(result["model_generation_calls"],2)
        self.assertEqual(result["challenged_status"],"CONTESTED_ADVISORY")
        self.assertTrue(result["challenge_triggered_reconsideration"])
        self.assertFalse(result["independently_verified"])

    def test_unacknowledged_objection_is_not_actionable(self):
        _,cases,_=load_frozen_cases(CASES)
        c=next(c for c in cases if c["id"]=="T01")
        backend=ScriptedBackend(position(True),position(True,acknowledged=()))
        result=evaluate_case(c,backend)
        self.assertEqual(result["challenged_status"],"HOLD_UNACKNOWLEDGED_OBJECTION")
        self.assertFalse(result["challenged_decision_usable"])
        self.assertFalse(result["independently_verified"])

    def test_nonjson_live_core_output_counts_as_failure_not_correct(self):
        _,cases,_=load_frozen_cases(CASES)
        c=cases[0]
        backend=ScriptedBackend(position(True))
        backend.responses=["unparseable output"]
        result=evaluate_case(c,backend)
        self.assertFalse(result["baseline_correct"])
        self.assertFalse(result["challenged_correct"])
        self.assertEqual(result["error_category"],"NeuralOutputError")
        self.assertFalse(result["challenged_decision_usable"])

    def test_aggregate_exposes_regressions_and_success_denominator(self):
        _,cases,_=load_frozen_cases(CASES)
        rows=[
            evaluate_case(cases[0],ScriptedBackend(position(True),position(False,acknowledged=("scripted-critic",)))),
            evaluate_case(cases[1],ScriptedBackend(position(True),position(False,acknowledged=("scripted-critic",)))),
        ]
        result=aggregate(rows)
        self.assertEqual(result["total_cases"],2)
        self.assertEqual(result["baseline_correct"],1)
        self.assertEqual(result["challenged_correct"],1)
        self.assertEqual(result["improvements"],1)
        self.assertEqual(result["regressions"],1)
        self.assertEqual(result["model_generation_calls"],4)
        self.assertFalse(result["real_specialist_model_used"])

    def test_frozen_cases_reject_ground_truth_mismatch(self):
        with TemporaryDirectory() as td:
            path=Path(td)/"cases.json"
            frozen=json.loads(CASES.read_text(encoding="utf8"))
            frozen["cases"][0]["specialist_quality"]="CORRECT"
            path.write_text(json.dumps(frozen))
            with self.assertRaises(FrozenCaseError):
                load_frozen_cases(path)


if __name__=="__main__":
    unittest.main()
