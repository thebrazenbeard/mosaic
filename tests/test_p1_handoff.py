import unittest

from runtime.handoff import (
    CONDITIONS,
    ContinuityCase,
    HandoffProtocolError,
    StateEvent,
    run_benchmark,
    run_case,
)
from tests.p1_cases import CASES


class P1ProtocolHarnessTests(unittest.TestCase):
    def test_all_predeclared_minimum_task_families_are_present(self):
        receipt = run_benchmark(CASES)
        self.assertEqual(
            set(receipt["task_families"]),
            {
                "REFERENT_PRESERVATION",
                "CORRECTION",
                "SUPERSESSION_CURRENTNESS",
                "DECISION_FINALITY",
                "AUTHORITY_CHANGE",
                "USER_PREFERENCE",
                "PROJECT_STATUS",
                "ARTIFACT_HANDOFF",
            },
        )

    def test_all_four_conditions_run_same_case_set(self):
        receipt = run_benchmark(CASES)
        for condition in CONDITIONS:
            self.assertEqual(receipt["conditions"][condition]["cases"], len(CASES))

    def test_reset_has_no_continuity_information(self):
        receipt = run_benchmark(CASES)
        self.assertEqual(receipt["conditions"]["RESET"]["correct"], 0)
        self.assertEqual(receipt["conditions"]["RESET"]["accuracy"], 0.0)

    def test_mosaic_state_and_full_history_resolve_current_values(self):
        receipt = run_benchmark(CASES)
        self.assertEqual(receipt["conditions"]["MOSAIC_STATE"]["correct"], len(CASES))
        self.assertEqual(receipt["conditions"]["FULL_HISTORY"]["correct"], len(CASES))
        self.assertEqual(receipt["conditions"]["MOSAIC_STATE"]["stale_errors"], 0)

    def test_ordinary_retrieval_is_reported_without_special_credit(self):
        receipt = run_benchmark(CASES)
        self.assertEqual(
            receipt["conditions"]["ORDINARY_RETRIEVAL"]["accuracy"],
            receipt["conditions"]["MOSAIC_STATE"]["accuracy"],
        )

    def test_harness_cannot_self_award_p1(self):
        receipt = run_benchmark(CASES)
        self.assertTrue(receipt["mosaic_outperforms_reset_in_harness"])
        self.assertTrue(receipt["mosaic_stale_errors_not_above_reset"])
        self.assertTrue(receipt["full_history_capability_control_succeeds"])
        self.assertFalse(receipt["actual_accelerator_unload_verified"])
        self.assertFalse(receipt["independent_clean_repeat_completed"])
        self.assertFalse(receipt["p1_pass_eligible"])
        self.assertEqual(receipt["qualification"], "DETERMINISTIC_PROTOCOL_HARNESS_ONLY")

    def test_superseded_state_is_not_returned_as_current(self):
        case = next(item for item in CASES if item.family == "SUPERSESSION_CURRENTNESS")
        row = run_case(case, "MOSAIC_STATE")
        self.assertEqual(row["answer"], "current-v2")
        self.assertFalse(row["stale_error"])

    def test_duplicate_revisions_fail_closed(self):
        bad = ContinuityCase(
            case_id="bad",
            family="CORRECTION",
            events=(
                StateEvent("e1", "CORRECTION", "fact", "old", 1),
                StateEvent("e2", "CORRECTION", "fact", "new", 1),
            ),
            query_key="fact",
            expected_value="new",
        )
        with self.assertRaisesRegex(HandoffProtocolError, "revisions must be unique"):
            run_case(bad, "MOSAIC_STATE")

    def test_latest_unrelated_key_does_not_invalidate_queried_state(self):
        # Real handoff snapshots can contain several facts; a later edit to
        # another key must not hide the latest revision of the queried fact.
        case = ContinuityCase(
            case_id="mixed-key-history",
            family="CORRECTION",
            events=(
                StateEvent("fact-old", "CORRECTION", "fact", "old", 1),
                StateEvent("fact-new", "CORRECTION", "fact", "current", 3),
                StateEvent("other-new", "CORRECTION", "other", "other-value", 5),
            ),
            query_key="fact",
            expected_value="current",
        )
        for condition in ("MOSAIC_STATE", "FULL_HISTORY", "ORDINARY_RETRIEVAL"):
            with self.subTest(condition=condition):
                row = run_case(case, condition)
                self.assertTrue(row["correct"])
                self.assertEqual(row["answer"], "current")

    def test_query_key_not_present_in_events_fails_closed(self):
        case = ContinuityCase(
            case_id="missing-key",
            family="CORRECTION",
            events=(StateEvent("e1", "CORRECTION", "other", "value", 1),),
            query_key="fact",
            expected_value="value",
        )
        with self.assertRaisesRegex(HandoffProtocolError, "query_key.*events"):
            run_case(case, "MOSAIC_STATE")

    def test_unpaired_unicode_state_fails_closed_for_all_conditions(self):
        invalid = chr(0xD800)
        case = ContinuityCase(
            case_id="invalid-unicode", family="CORRECTION",
            events=(StateEvent("e1", "CORRECTION", "fact", invalid, 1),),
            query_key="fact", expected_value=invalid,
        )
        for condition in CONDITIONS:
            with self.subTest(condition=condition):
                with self.assertRaises(HandoffProtocolError):
                    run_case(case, condition)

    def test_invalid_unicode_identifier_and_valid_unicode_value(self):
        invalid = chr(0xD800)
        malformed = ContinuityCase(
            case_id="bad-id", family="CORRECTION",
            events=(StateEvent(invalid, "CORRECTION", "fact", "value", 1),),
            query_key="fact", expected_value="value",
        )
        with self.assertRaises(HandoffProtocolError):
            run_case(malformed, "RESET")
        supported = ContinuityCase(
            case_id="utf8-valid", family="CORRECTION",
            events=(StateEvent("e1", "CORRECTION", "café", "☕ ready", 1),),
            query_key="café", expected_value="☕ ready",
        )
        self.assertEqual(run_case(supported, "MOSAIC_STATE")["answer"], "☕ ready")


if __name__ == "__main__":
    unittest.main()
