"""Hostile contract tests for an intelligent Core and fallible specialists."""
import unittest

from runtime.thinking_core import (
    DeliberationTask, CorePosition, SpecialistObjection, EvidenceVerdict,
    CoreDeliberationEngine, DecisionStatus, ProtocolViolation, subject_digest,
)


class StubCore:
    def __init__(self, initial, revised):
        self.initial = initial
        self.revised = revised
        self.call_log = []

    def think(self, task):
        self.call_log.append(("think", task.case_id))
        return self.initial

    def reconsider(self, task, prior, objections, evidence):
        self.call_log.append(("reconsider", task.case_id, tuple(x.specialist_id for x in objections)))
        return self.revised


class StubSpecialist:
    def __init__(self, name, stance, reason="Specialist argument"):
        self.specialist_id = name
        self.stance = stance
        self.reason = reason
        self.call_count = 0

    def review(self, task, core_position):
        self.call_count += 1
        return SpecialistObjection(
            specialist_id=self.specialist_id,
            proposition_id=task.proposition_id,
            supports= self.stance,
            rationale=self.reason,
        )


class StubVerifier:
    def __init__(self, value=None, issuer="independent-harness"):
        self.value=value
        self.issuer=issuer
        self.calls=0

    def examine(self, task, proposal, objections):
        self.calls+=1
        return EvidenceVerdict(
            proposition_id=task.proposition_id,
            supports=self.value,
            verifier_id=self.issuer,
            evidence_ref="sha256:test-frozen-fixture" if self.value is not None else "",
            rationale="Fixture-based check" if self.value is not None else "No independent fact check",
            subject_sha256=subject_digest(task),
        )


def task(*, consequential=False, protected=False):
    return DeliberationTask(
        case_id="case-17",
        proposition_id="requirements-met",
        question="Does this implementation satisfy the invariant?",
        consequential=consequential,
        protected_effect=protected,
    )


def pos(stance, *, acknowledged=(), reason="Reasoned position"):
    return CorePosition(
        proposition_id="requirements-met",
        supports=stance,
        reasoning=reason,
        acknowledged_objections=acknowledged,
    )


class ThinkingCoreDecisionTests(unittest.TestCase):
    def test_thinking_core_not_specialist_owns_final_reasoning(self):
        core=StubCore(pos(True),pos(False,acknowledged=("auditor",)))
        specialist=StubSpecialist("auditor",False)
        result=CoreDeliberationEngine().deliberate(task(),core,[specialist])
        self.assertEqual(core.call_log[0][0],"think")
        self.assertEqual(core.call_log[1][0],"reconsider")
        self.assertEqual(result.core_final.supports,False)
        self.assertEqual(result.status,DecisionStatus.CONTESTED_ADVISORY)
        self.assertFalse(result.claim_verified)
        self.assertFalse(result.effect_executed)
        self.assertEqual(result.objections[0].specialist_id,"auditor")

    def test_specialist_cannot_force_final_answer_without_core_revision(self):
        core=StubCore(pos(True),pos(False,acknowledged=()))
        result=CoreDeliberationEngine().deliberate(
            task(),core,[StubSpecialist("auditor",False)])
        self.assertEqual(result.status,DecisionStatus.HOLD_UNACKNOWLEDGED_OBJECTION)
        self.assertFalse(result.claim_verified)
        self.assertFalse(result.effect_executed)

    def test_unattested_verifier_conflict_holds_even_after_core_rejects_specialist(self):
        core=StubCore(pos(True),pos(True,acknowledged=("auditor",)))
        result=CoreDeliberationEngine().deliberate(
            task(),core,[StubSpecialist("auditor",False)],StubVerifier(False))
        self.assertEqual(result.status,DecisionStatus.HOLD_UNATTESTED_VERIFIER_CONFLICT)
        self.assertFalse(result.verified_proposition_is_false)
        self.assertFalse(result.claim_verified)
        self.assertFalse(result.effect_executed)

    def test_core_can_accept_specialist_on_unattested_evidence_without_verification(self):
        core=StubCore(pos(True),pos(False,acknowledged=("auditor",)))
        result=CoreDeliberationEngine().deliberate(
            task(),core,[StubSpecialist("auditor",False)],StubVerifier(False))
        self.assertEqual(result.status,DecisionStatus.VERIFIER_ASSERTED_ADVISORY)
        self.assertFalse(result.claim_verified)
        self.assertEqual(result.core_final.supports,False)

    def test_consequential_unverified_dispute_holds(self):
        core=StubCore(pos(True),pos(True,acknowledged=("auditor",)))
        result=CoreDeliberationEngine().deliberate(
            task(consequential=True),core,[StubSpecialist("auditor",False)])
        self.assertEqual(result.status,DecisionStatus.HOLD_UNVERIFIED_DISPUTE)
        self.assertFalse(result.effect_executed)

    def test_protected_effect_requires_owner_even_with_verifier_assertion(self):
        core=StubCore(pos(True),pos(True))
        result=CoreDeliberationEngine().deliberate(
            task(protected=True),core,[StubSpecialist("auditor",True)],StubVerifier(True))
        self.assertEqual(result.status,DecisionStatus.PROPOSED_EFFECT_REQUIRES_AUTHORIZATION)
        self.assertFalse(result.claim_verified)
        self.assertFalse(result.effect_executed)

    def test_claim_without_challenge_is_not_magically_verified(self):
        core=StubCore(pos(True),pos(True))
        result=CoreDeliberationEngine().deliberate(task(),core,[])
        self.assertEqual(result.status,DecisionStatus.UNVERIFIED_ADVISORY)
        self.assertFalse(result.claim_verified)
        self.assertFalse(result.effect_executed)

    def test_core_cannot_self_verify(self):
        core=StubCore(pos(True),pos(True))
        with self.assertRaisesRegex(ProtocolViolation, "verifier"):
            CoreDeliberationEngine().deliberate(task(),core,[],StubVerifier(True,issuer="core"))

    def test_verifier_must_bind_evidence_to_subject(self):
        class InvalidVerifier:
            def examine(self, task, position, objections):
                return EvidenceVerdict(task.proposition_id,True,"auditor","", "unsupported")
        core=StubCore(pos(True),pos(True))
        with self.assertRaisesRegex(ProtocolViolation, "evidence"):
            CoreDeliberationEngine().deliberate(task(),core,[],InvalidVerifier())

    def test_wrong_proposition_or_duplicate_reviewers_cannot_be_laundered(self):
        core=StubCore(CorePosition("other",True,"Wrong proposition"),pos(True))
        with self.assertRaisesRegex(ProtocolViolation, "proposition"):
            CoreDeliberationEngine().deliberate(task(),core,[])
        core=StubCore(pos(True),pos(True))
        with self.assertRaisesRegex(ProtocolViolation, "duplicate"):
            CoreDeliberationEngine().deliberate(task(),core,
                [StubSpecialist("auditor",True),StubSpecialist("auditor",True)])

    def test_verifier_contradiction_without_specialist_still_forces_second_core_thought(self):
        core=StubCore(pos(True),pos(False,reason="Independent evidence proves opposite"))
        result=CoreDeliberationEngine().deliberate(
            task(),core,[],StubVerifier(False))
        self.assertEqual(len(core.call_log),2)
        self.assertEqual(result.status,DecisionStatus.VERIFIER_ASSERTED_ADVISORY)
        self.assertFalse(result.verified_proposition_is_false)

    def test_unknown_verifier_cannot_resolve_a_consequential_dispute(self):
        core=StubCore(pos(True),pos(True,acknowledged=("auditor",)))
        result=CoreDeliberationEngine().deliberate(
            task(consequential=True),core,[StubSpecialist("auditor",False)],
            StubVerifier(None))
        self.assertEqual(result.status,DecisionStatus.HOLD_UNVERIFIED_DISPUTE)
        self.assertFalse(result.claim_verified)

    def test_specialist_cannot_impersonate_external_verifier(self):
        core=StubCore(pos(True),pos(True))
        with self.assertRaisesRegex(ProtocolViolation,"verifier"):
            CoreDeliberationEngine().deliberate(
                task(),core,[StubSpecialist("auditor",True)],
                StubVerifier(True,issuer="auditor"))

    def test_core_must_acknowledge_all_dissenters(self):
        core=StubCore(pos(True),pos(False,acknowledged=("auditor",)))
        result=CoreDeliberationEngine().deliberate(
            task(),core,[StubSpecialist("auditor",False),StubSpecialist("critic",False)])
        self.assertEqual(result.status,DecisionStatus.HOLD_UNACKNOWLEDGED_OBJECTION)

    def test_invalid_reviewer_preflight_precedes_core_inference(self):
        for reviewers in (
            [StubSpecialist("core", True)],
            [StubSpecialist("auditor", True), StubSpecialist("AUDITOR", False)],
            [StubSpecialist(" core ", True)],
            [StubSpecialist("auditor ", True)],
        ):
            with self.subTest(reviewers=[x.specialist_id for x in reviewers]):
                core = StubCore(pos(True), pos(True))
                with self.assertRaises(ProtocolViolation):
                    CoreDeliberationEngine().deliberate(task(), core, reviewers)
                self.assertEqual(core.call_log, [])
                self.assertTrue(all(x.call_count == 0 for x in reviewers))

    def test_initial_and_final_core_cannot_fabricate_acknowledgments(self):
        cases = (
            (pos(True, acknowledged=("auditor",)), pos(True), []),
            (pos(True), pos(False, acknowledged=("auditor", "ghost")),
             [StubSpecialist("auditor", False)]),
            (pos(True), pos(False, acknowledged=("auditor", "auditor")),
             [StubSpecialist("auditor", False)]),
            (pos(True), pos(False, acknowledged="auditor"),
             [StubSpecialist("auditor", False)]),
        )
        for initial, revised, reviewers in cases:
            with self.subTest(initial=initial, revised=revised):
                with self.assertRaises(ProtocolViolation):
                    CoreDeliberationEngine().deliberate(
                        task(), StubCore(initial, revised), reviewers
                    )

    def test_verifier_whitespace_impersonation_is_invalid(self):
        core = StubCore(pos(True), pos(True))
        with self.assertRaises(ProtocolViolation):
            CoreDeliberationEngine().deliberate(
                task(), core, [StubSpecialist("auditor", True)],
                StubVerifier(True, issuer=" auditor ")
            )

    def test_dispute_record_is_reproducible_and_conveys_uncertainty(self):
        core=StubCore(pos(True),pos(False,acknowledged=("auditor",)))
        a=CoreDeliberationEngine().deliberate(task(),core,[StubSpecialist("auditor",False)])
        core2=StubCore(pos(True),pos(False,acknowledged=("auditor",)))
        b=CoreDeliberationEngine().deliberate(task(),core2,[StubSpecialist("auditor",False)])
        self.assertEqual(a.receipt_sha256,b.receipt_sha256)
        self.assertNotEqual(a.receipt_sha256,"")
        self.assertIn("NOT_NEURAL_QUALIFICATION",a.claim_ceiling)
        self.assertIn("callbacks are not sandboxed",a.claim_ceiling)

    def test_unregistered_self_asserted_verifier_is_never_authenticated(self):
        class UnregisteredVerifier:
            def examine(self, current, proposal, objections):
                return EvidenceVerdict(
                    proposition_id=current.proposition_id,
                    supports=True,
                    verifier_id="unregistered-fake-verifier",
                    evidence_ref="evidence://case-A-only",
                    rationale="No authenticated evidence",
                    subject_sha256=subject_digest(current),
                )

        for case_id in ("case-A", "case-B"):
            current=DeliberationTask(case_id,"requirements-met","Does this satisfy the invariant?")
            result=CoreDeliberationEngine().deliberate(
                current,StubCore(pos(True),pos(True)),[],UnregisteredVerifier())
            self.assertFalse(result.claim_verified)
            self.assertFalse(result.verified_proposition_is_false)
            self.assertEqual(result.status,DecisionStatus.VERIFIER_ASSERTED_ADVISORY)

    def test_replayed_evidence_cannot_bind_to_a_different_case_or_question(self):
        first=DeliberationTask("case-A","requirements-met","Question for case A")
        class ReplayedVerifier:
            def examine(self, current, proposal, objections):
                return EvidenceVerdict(
                    proposition_id=current.proposition_id,
                    supports=True, verifier_id="unregistered-fake-verifier",
                    evidence_ref="evidence://case-A-only", rationale="Replayed",
                    subject_sha256=subject_digest(first),
                )
        for current in (
            DeliberationTask("case-B","requirements-met","Question for case A"),
            DeliberationTask("case-A","requirements-met","Different question"),
        ):
            with self.subTest(current=current),self.assertRaisesRegex(ProtocolViolation,"subject"):
                CoreDeliberationEngine().deliberate(
                    current,StubCore(pos(True),pos(True)),[],ReplayedVerifier())

    def test_unpaired_surrogate_does_not_escape_domain_error_boundary(self):
        unsafe=CorePosition("requirements-met",True,chr(0xD800))
        with self.assertRaisesRegex(ProtocolViolation,"UTF-8"):
            CoreDeliberationEngine().deliberate(task(),StubCore(unsafe,unsafe),[])

    def test_unknown_verdict_rejects_falsey_nonstring_evidence_references(self):
        for bad_ref in ([],False,0,{},None):
            class BadUnknownVerifier:
                def examine(self, current, proposal, objections):
                    return EvidenceVerdict(
                        proposition_id=current.proposition_id,
                        supports=None,verifier_id="external-harness",
                        evidence_ref=bad_ref,rationale="No admissible claim",
                        subject_sha256=subject_digest(current),
                    )
            with self.subTest(value=bad_ref),self.assertRaisesRegex(ProtocolViolation,"evidence_ref"):
                CoreDeliberationEngine().deliberate(
                    task(),StubCore(pos(True),pos(True)),[],BadUnknownVerifier())

    def test_verifier_subject_digest_requires_strict_lowercase_sha256(self):
        valid=subject_digest(task())
        for bad_digest in ([],None,False,"",valid.upper(),valid[:-1],"g"*64):
            class BadDigestVerifier:
                def examine(self, current, proposal, objections):
                    return EvidenceVerdict(
                        current.proposition_id,True,"external-harness",
                        "sha256:fixture","Claimed fixture",subject_sha256=bad_digest,
                    )
            with self.subTest(value=bad_digest),self.assertRaisesRegex(ProtocolViolation,"subject_sha256"):
                CoreDeliberationEngine().deliberate(
                    task(),StubCore(pos(True),pos(True)),[],BadDigestVerifier())


if __name__ == "__main__":
    unittest.main()
