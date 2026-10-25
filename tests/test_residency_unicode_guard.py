"""Malformed text must not leak raw encoding errors from residency receipts."""

import unittest

from runtime.residency import (
    ResidencyEnvelope,
    ResidencyModelError,
    SpecialistResidency,
    canonical_json_sha256,
    evaluate_swap_sequence,
    evaluate_transition_headroom,
)


class ResidencyUnicodeGuardTests(unittest.TestCase):
    def test_unpaired_surrogate_specialist_id_fails_closed_in_both_receipts(self):
        bad_id = chr(0xD800)
        envelope = ResidencyEnvelope(128, 0, (SpecialistResidency(bad_id, 1, 1, 0, 0, 0),))
        for evaluate in (evaluate_swap_sequence, evaluate_transition_headroom):
            with self.subTest(evaluator=evaluate.__name__):
                with self.assertRaises(ResidencyModelError):
                    evaluate(envelope, [bad_id])

    def test_canonical_digest_rejects_invalid_unicode_with_domain_error(self):
        with self.assertRaises(ResidencyModelError):
            canonical_json_sha256({"id": chr(0xD800)})

    def test_legitimate_unicode_identity_remains_supported(self):
        envelope = ResidencyEnvelope(128, 0, (SpecialistResidency("café", 1, 1, 0, 0, 0),))
        receipt = evaluate_transition_headroom(envelope, ["café"])
        self.assertTrue(receipt["all_modeled_transitions_fit"])
        self.assertEqual(receipt["modeled_peak_transition_bytes"], 1)


if __name__ == "__main__":
    unittest.main()
