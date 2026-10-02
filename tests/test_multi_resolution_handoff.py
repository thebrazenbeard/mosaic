import unittest

from runtime.handoff import CONDITIONS, run_benchmark, run_case
from tests.p1_cases import CASES


class MultiResolutionHandoffTests(unittest.TestCase):
    def test_conditions_include_compact_and_rehydrating_arms(self):
        self.assertIn("MULTI_RESOLUTION_STATE", CONDITIONS)
        self.assertIn("MULTI_RESOLUTION_REHYDRATE", CONDITIONS)

    def test_compact_state_does_not_pretend_to_contain_exact_value(self):
        case = CASES[0]
        compact = run_case(case, "MULTI_RESOLUTION_STATE")
        history = run_case(case, "FULL_HISTORY")

        self.assertEqual(compact["answer"], "UNKNOWN")
        self.assertFalse(compact["correct"])
        self.assertEqual(compact["rehydration_count"], 0)
        self.assertEqual(compact["rehydration_bytes"], 0)
        self.assertLess(compact["active_state_bytes"], history["payload_bytes"])

    def test_selective_rehydration_restores_only_queried_current_event(self):
        case = next(item for item in CASES if len(item.events) > 1)
        row = run_case(case, "MULTI_RESOLUTION_REHYDRATE")
        current = max(case.events, key=lambda event: event.revision)

        self.assertTrue(row["correct"])
        self.assertEqual(row["answer"], case.expected_value)
        self.assertEqual(row["rehydration_count"], 1)
        self.assertEqual(row["rehydrated_event_ids"], [current.event_id])
        self.assertGreater(row["rehydration_bytes"], 0)
        self.assertGreater(row["backing_state_bytes"], row["rehydration_bytes"])

    def test_rehydrating_arm_matches_full_history_accuracy_with_lower_active_state(self):
        receipt = run_benchmark(CASES)
        rehydrated = receipt["conditions"]["MULTI_RESOLUTION_REHYDRATE"]
        history = receipt["conditions"]["FULL_HISTORY"]
        compact = receipt["conditions"]["MULTI_RESOLUTION_STATE"]

        self.assertEqual(rehydrated["accuracy"], history["accuracy"])
        self.assertEqual(rehydrated["accuracy"], 1.0)
        self.assertLess(
            rehydrated["active_state_bytes_total"],
            history["payload_bytes_total"],
        )
        self.assertGreater(rehydrated["rehydration_bytes_total"], 0)
        self.assertEqual(compact["rehydration_bytes_total"], 0)
        self.assertFalse(receipt["p1_pass_eligible"])

    def test_rehydration_is_deterministic(self):
        case = CASES[0]

        first = run_case(case, "MULTI_RESOLUTION_REHYDRATE")
        second = run_case(case, "MULTI_RESOLUTION_REHYDRATE")

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
