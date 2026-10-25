"""Analytical swap-transition risks, not a hardware P0 qualification."""

import unittest

from runtime.residency import (
    ResidencyEnvelope,
    ResidencyModelError,
    SpecialistResidency,
    evaluate_transition_headroom,
)


class SwapTransitionTests(unittest.TestCase):
    def envelope(self):
        # Two specialists each fit alone under the 100-byte ceiling, but
        # together exceed it. Byte-scale numbers make independent arithmetic
        # easy to audit; they are not device measurements.
        return ResidencyEnvelope(
            vram_ceiling_bytes=100,
            boot_core_gpu_bytes=10,
            specialists=(
                SpecialistResidency("A", 45, 40, 5, 5, 0),
                SpecialistResidency("B", 55, 50, 5, 5, 0),
            ),
        )

    def test_evict_then_load_fits_but_overlap_fails(self):
        env = self.envelope()
        evict = evaluate_transition_headroom(env, ["A", "B", "A"],
                                             policy="evict_then_load")
        overlap = evaluate_transition_headroom(env, ["A", "B", "A"],
                                               policy="load_then_evict")
        self.assertEqual(evict["modeled_peak_transition_bytes"], 70)
        self.assertEqual(evict["headroom_bytes"], 30)
        self.assertTrue(evict["all_modeled_transitions_fit"])
        self.assertEqual(overlap["modeled_peak_transition_bytes"], 120)
        self.assertEqual(overlap["headroom_bytes"], -20)
        self.assertFalse(overlap["all_modeled_transitions_fit"])
        self.assertFalse(overlap["hardware_measurement_performed"])

    def test_same_specialist_is_not_double_counted(self):
        receipt = evaluate_transition_headroom(self.envelope(), ["A", "A"],
                                               policy="load_then_evict")
        self.assertEqual(receipt["modeled_peak_transition_bytes"], 60)
        self.assertEqual(receipt["transitions"][1]["kind"], "reuse")

    def test_persistent_overhead_reduces_headroom_and_binds_digest(self):
        baseline = evaluate_transition_headroom(self.envelope(), ["A", "B"],
                                                policy="evict_then_load")
        reserved = evaluate_transition_headroom(self.envelope(), ["A", "B"],
                                                policy="evict_then_load",
                                                persistent_overhead_bytes=35)
        self.assertEqual(reserved["modeled_peak_transition_bytes"], 105)
        self.assertEqual(reserved["headroom_bytes"], -5)
        self.assertNotEqual(reserved["input_sha256"], baseline["input_sha256"])
        self.assertEqual(reserved["input_sha256"], evaluate_transition_headroom(
            self.envelope(), ["A", "B"], policy="evict_then_load",
            persistent_overhead_bytes=35)["input_sha256"])

    def test_invalid_policy_budget_and_unhashable_sequence_fail_closed(self):
        env = self.envelope()
        for invalid_policy in ("hybrid", "", None):
            with self.subTest(policy=invalid_policy):
                with self.assertRaises(ResidencyModelError):
                    evaluate_transition_headroom(env, ["A", "B"], policy=invalid_policy)
        for invalid_overhead in (-1, True, 0.5, "10"):
            with self.subTest(overhead=invalid_overhead):
                with self.assertRaises(ResidencyModelError):
                    evaluate_transition_headroom(env, ["A"],
                                                 persistent_overhead_bytes=invalid_overhead)
        with self.assertRaisesRegex(ResidencyModelError, "specialist ID"):
            evaluate_transition_headroom(env, [["A"]])


if __name__ == "__main__":
    unittest.main()
