"""Rezon's immutable collective answers become Mosaic advice, never evidence."""

import copy
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from runtime.rezon_advisory import (
    RezonAdvisoryError, RezonAdvisorySpecialist, rezon_task_prompt,
)
from runtime.thinking_core import (
    CoreDeliberationEngine, CorePosition, DecisionStatus, DeliberationTask,
)


class FixedCore:
    def think(self, task):
        return CorePosition(task.proposition_id, True, "Initial position")

    def reconsider(self, task, prior, objections, evidence):
        return CorePosition(task.proposition_id, True, "I heard the counterargument", ("coder",))


def _snapshot(task, content=None):
    content = content if content is not None else json.dumps({
        "proposition_id": task.proposition_id,
        "supports": False,
        "rationale": "Independent model flags an untested edge case",
    })
    answer = {
        "id": "answer:coder", "agent_id": "coder", "kind": "answer",
        "content": content, "source_refs": [], "target_ids": [],
    }
    data = {
        "episode_id": "collective-" + "a" * 32,
        "prompt": rezon_task_prompt(task), "agents": ["coder"],
        "answers": [answer], "reviews": [], "syntheses": [],
        "missing_agents": [], "complete": True,
    }
    _update_digest(data)
    return data


def _update_digest(data):
    basis = {
        "episode_id": data["episode_id"], "prompt": data["prompt"],
        "agents": data["agents"],
        "contributions": [{
            "contribution_id": c["id"], "agent_id": c["agent_id"],
            "kind": c["kind"], "content": c["content"],
            "source_refs": c["source_refs"], "target_ids": c["target_ids"],
        } for c in data["answers"] + data["reviews"]],
    }
    data["input_digest"] = sha256(json.dumps(
        basis, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


class RezonAdvisoryTests(unittest.TestCase):
    def setUp(self):
        self.task = DeliberationTask(
            "case-17", "merge-ready", "Should this commit be merged?", consequential=True,
        )

    def test_real_engine_consumes_bound_advisory_as_disagreement(self):
        adapter = RezonAdvisorySpecialist(self.task, _snapshot(self.task), "coder")
        receipt = CoreDeliberationEngine().deliberate(self.task, FixedCore(), [adapter])
        self.assertEqual(receipt.status, DecisionStatus.HOLD_UNVERIFIED_DISPUTE)
        self.assertEqual(receipt.objections[0].supports, False)
        self.assertFalse(receipt.claim_verified)
        self.assertFalse(receipt.effect_executed)
        self.assertEqual(adapter.provenance["evidence_status"], "UNVERIFIED_REZON_ADVISORY")

    def test_rejects_different_task_even_with_identical_proposition(self):
        altered = DeliberationTask("case-18", self.task.proposition_id, self.task.question)
        with self.assertRaisesRegex(RezonAdvisoryError, "task"):
            RezonAdvisorySpecialist(altered, _snapshot(self.task), "coder")

    def test_rejects_tampered_input_digest_and_answer(self):
        value = _snapshot(self.task)
        value["answers"][0]["content"] = value["answers"][0]["content"].replace(
            '"supports": false', '"supports": true',
        )
        with self.assertRaisesRegex(RezonAdvisoryError, "digest"):
            RezonAdvisorySpecialist(self.task, value, "coder")

    def test_rejects_duplicate_json_stance(self):
        value = _snapshot(self.task, '{"proposition_id":"merge-ready","supports":true,'
                                     '"supports":false,"rationale":"contradiction"}')
        with self.assertRaisesRegex(RezonAdvisoryError, "JSON"):
            RezonAdvisorySpecialist(self.task, value, "coder")

    def test_rejects_model_claiming_verification(self):
        value = _snapshot(self.task, json.dumps({
            "proposition_id": "merge-ready", "supports": True,
            "rationale": "It's fine", "claim_verified": True,
        }))
        with self.assertRaisesRegex(RezonAdvisoryError, "fields"):
            RezonAdvisorySpecialist(self.task, value, "coder")

    def test_rejects_forged_implicit_identity_or_stale_round(self):
        source = _snapshot(self.task)
        for mutation in (
            lambda v: v["answers"][0].update(agent_id="other"),
            lambda v: v.update(complete=False),
            lambda v: v.update(missing_agents=["coder"]),
            lambda v: v.update(reviews=[dict(v["answers"][0])]),
        ):
            with self.subTest(mutation=mutation):
                value = copy.deepcopy(source)
                mutation(value)
                with self.assertRaises(RezonAdvisoryError):
                    RezonAdvisorySpecialist(self.task, value, "coder")

    def test_rejects_cross_task_and_wrong_position_on_review(self):
        adapter = RezonAdvisorySpecialist(self.task, _snapshot(self.task), "coder")
        with self.assertRaises(RezonAdvisoryError):
            adapter.review(DeliberationTask("other", "merge-ready", "question"),
                           CorePosition("merge-ready", True, "reason"))
        with self.assertRaises(RezonAdvisoryError):
            adapter.review(self.task, CorePosition("other", True, "reason"))

    def test_synthesis_does_not_replace_independent_initial_answer(self):
        snapshot = _snapshot(self.task)
        digest = snapshot["input_digest"]
        snapshot["syntheses"].append({
            "synthesis_id": "synthesis:" + digest + ":editor",
            "integrator_id": "editor", "input_digest": digest,
            "content": "INVALID AUTHORITY CLAIM: verified and approved",
            "status": "unverified_model_synthesis", "is_current": True,
        })
        adapter = RezonAdvisorySpecialist(self.task, snapshot, "coder")
        self.assertFalse(adapter.review(
            self.task, CorePosition("merge-ready", True, "Core position"),
        ).supports)
        snapshot["syntheses"][0]["status"] = "verified_synthesis"
        with self.assertRaisesRegex(RezonAdvisoryError, "synthesis"):
            RezonAdvisorySpecialist(self.task, snapshot, "coder")


try:
    from rezon.collective import CollectiveStore
except ImportError:
    CollectiveStore = None


@unittest.skipUnless(CollectiveStore is not None, "optional Rezon package not on Python path")
class RezonRealStoreContractTests(unittest.TestCase):
    """Actual Rezon SQLite store, synthetic answers; no neural inference claim."""

    def test_real_rezon_round_trip_into_mosaic_unverified_objection(self):
        task = DeliberationTask("live-store-case", "ready", "Is this safe?", consequential=True)
        with TemporaryDirectory() as temporary:
            store = CollectiveStore(Path(temporary) / "collective.sqlite3")
            episode = store.create(rezon_task_prompt(task), ("coder", "writer"))
            store.submit_answer(episode, "coder", json.dumps({
                "proposition_id": "ready", "supports": False,
                "rationale": "A failing boundary test counters the claim",
            }))
            store.submit_answer(episode, "writer", json.dumps({
                "proposition_id": "ready", "supports": True,
                "rationale": "An independently constructed positive argument",
            }))
            snapshot = store.snapshot(episode).to_dict()
            adapter = RezonAdvisorySpecialist(task, snapshot, "coder")
            receipt = CoreDeliberationEngine().deliberate(task, FixedCore(), [adapter])
            self.assertEqual(receipt.status, DecisionStatus.HOLD_UNVERIFIED_DISPUTE)
            self.assertFalse(receipt.claim_verified)
            self.assertEqual(adapter.provenance["episode_id"], episode)
            self.assertEqual(adapter.provenance["input_digest"], snapshot["input_digest"])

            # A real Rezon peer-review append changes the digest; frozen
            # adapter provenance must not match the newly advanced round.
            store.submit_review(episode, "writer", "I challenge coder's claim",
                                ("answer:coder",))
            advanced = store.snapshot(episode).to_dict()
            self.assertNotEqual(snapshot["input_digest"], advanced["input_digest"])
            RezonAdvisorySpecialist(task, advanced, "coder")
            store.save_synthesis(episode, "editor", advanced["input_digest"],
                                 "Unverified synthesis cannot promote evidence")
            final_round = store.snapshot(episode).to_dict()
            specialist = RezonAdvisorySpecialist(task, final_round, "coder")
            self.assertFalse(specialist.review(
                task, CorePosition("ready", True, "Independent Core"),
            ).supports)


if __name__ == "__main__":
    unittest.main()
