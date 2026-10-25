"""Adversarial contract tests against Mosaic's synthetic research harness.

These intentionally expose presently unguarded input/provenance invariants.
"""

import math
import unittest

from runtime.handoff import ContinuityCase, HandoffProtocolError, StateEvent, run_case
from runtime.residency import (
    ResidencyEnvelope,
    ResidencyModelError,
    SpecialistResidency,
    evaluate_swap_sequence,
)


class HandoffIntegrityTests(unittest.TestCase):
    def test_duplicate_event_ids_cannot_represent_distinct_revisions(self):
        case = ContinuityCase(
            case_id="duplicate-provenance",
            family="CORRECTION",
            events=(
                StateEvent("same-id", "CORRECTION", "claim", "obsolete", 1),
                StateEvent("same-id", "CORRECTION", "claim", "current", 2),
            ),
            query_key="claim",
            expected_value="current",
        )
        with self.assertRaisesRegex(HandoffProtocolError, "event_id"):
            run_case(case, "MOSAIC_STATE")

    def test_fractional_revisions_rejected(self):
        case = ContinuityCase(
            case_id="fractional-revision",
            family="CORRECTION",
            events=(StateEvent("e1", "CORRECTION", "claim", "current", 1.5),),
            query_key="claim",
            expected_value="current",
        )
        with self.assertRaisesRegex(HandoffProtocolError, "revision"):
            run_case(case, "MOSAIC_STATE")

    def test_nan_revisions_rejected(self):
        case = ContinuityCase(
            case_id="nan-revision",
            family="CORRECTION",
            events=(StateEvent("e1", "CORRECTION", "claim", "current", math.nan),),
            query_key="claim",
            expected_value="current",
        )
        with self.assertRaisesRegex(HandoffProtocolError, "revision"):
            run_case(case, "MOSAIC_STATE")

    def test_boolean_revisions_rejected(self):
        case = ContinuityCase(
            case_id="boolean-revision",
            family="CORRECTION",
            events=(StateEvent("e1", "CORRECTION", "claim", "current", True),),
            query_key="claim",
            expected_value="current",
        )
        with self.assertRaisesRegex(HandoffProtocolError, "revision"):
            run_case(case, "MOSAIC_STATE")

    def test_reset_unknown_literal_cannot_be_counted_as_continuity(self):
        case = ContinuityCase(
            case_id="unknown-as-data",
            family="CORRECTION",
            events=(StateEvent("e1", "CORRECTION", "claim", "UNKNOWN", 1),),
            query_key="claim",
            expected_value="UNKNOWN",
        )
        row = run_case(case, "RESET")
        self.assertFalse(row["correct"])

    def test_reset_unknown_literal_cannot_be_counted_as_stale_memory(self):
        case = ContinuityCase(
            case_id="unknown-as-stale-data",
            family="CORRECTION",
            events=(
                StateEvent("e1", "CORRECTION", "claim", "UNKNOWN", 1),
                StateEvent("e2", "CORRECTION", "claim", "current", 2),
            ),
            query_key="claim",
            expected_value="current",
        )
        row = run_case(case, "RESET")
        self.assertFalse(row["stale_error"])


class ResidencyInputIntegrityTests(unittest.TestCase):
    def test_unhashable_specialist_id_rejected_with_domain_error(self):
        envelope = ResidencyEnvelope(
            vram_ceiling_bytes=100,
            boot_core_gpu_bytes=0,
            specialists=(SpecialistResidency([], 1, 1, 0, 0, 0),),
        )
        with self.assertRaisesRegex(ResidencyModelError, "specialist_id"):
            evaluate_swap_sequence(envelope, ["part"])

    def test_fractional_bytes_rejected(self):
        envelope = ResidencyEnvelope(
            vram_ceiling_bytes=100,
            boot_core_gpu_bytes=0,
            specialists=(SpecialistResidency("part", 1, 1.25, 0, 0, 0),),
        )
        with self.assertRaisesRegex(ResidencyModelError, "byte"):
            evaluate_swap_sequence(envelope, ["part"])

    def test_nan_core_bytes_rejected_before_receipt(self):
        envelope = ResidencyEnvelope(
            vram_ceiling_bytes=100,
            boot_core_gpu_bytes=math.nan,
            specialists=(SpecialistResidency("part", 1, 1, 0, 0, 0),),
        )
        with self.assertRaisesRegex(ResidencyModelError, "byte"):
            evaluate_swap_sequence(envelope, ["part"])

    def test_boolean_byte_budget_rejected(self):
        envelope = ResidencyEnvelope(
            vram_ceiling_bytes=100,
            boot_core_gpu_bytes=False,
            specialists=(SpecialistResidency("part", 1, 1, 0, 0, 0),),
        )
        with self.assertRaisesRegex(ResidencyModelError, "byte"):
            evaluate_swap_sequence(envelope, ["part"])


if __name__ == "__main__":
    unittest.main()
