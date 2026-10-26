import unittest

from pathlib import Path

from experiments.p0_inprocess_abi_probe import NativeProbeError, run_metadata_cycles


SUBJECTS = {
    "A": {"path": Path("a.gguf"), "size_bytes": 100, "sha256": "a" * 64},
    "B": {"path": Path("b.gguf"), "size_bytes": 200, "sha256": "b" * 64},
}


class FakeLoader:
    def __init__(self):
        self.loaded = None
        self.events = []

    def load_metadata(self, path):
        if self.loaded is not None:
            raise AssertionError("previous model was not freed")
        self.loaded = object()
        self.events.append(("load", path.name))
        return self.loaded

    def free(self, model):
        if model is not self.loaded:
            raise AssertionError("wrong handle freed")
        self.events.append(("free", model))
        self.loaded = None


class InProcessAbiProbeTests(unittest.TestCase):
    def test_ten_aba_cycles_reuse_same_pid_but_never_claim_p0(self):
        backend = FakeLoader()
        receipt = run_metadata_cycles(backend, SUBJECTS, 10, pid=lambda: 251)
        self.assertEqual(receipt["metadata_aba_cycles"], 10)
        self.assertEqual(len(receipt["steps"]), 30)
        self.assertEqual([s["role"] for s in receipt["steps"]], ["A", "B", "A"] * 10)
        self.assertEqual([e[1] for e in backend.events if e[0] == "load"],
                         ["a.gguf", "b.gguf", "a.gguf"] * 10)
        self.assertEqual(len([e for e in backend.events if e[0] == "free"]), 30)
        self.assertIsNone(backend.loaded)
        self.assertTrue(receipt["same_python_pid_during_calls"])
        for not_proven in ("p0_pass", "neural_weights_loaded", "inference_completed",
                           "gpu_residency_attested", "child_process_restarts_excluded"):
            self.assertFalse(receipt[not_proven])

    def test_process_identity_change_fails_and_still_frees_handle(self):
        backend = FakeLoader()
        readings = iter([1, 2])
        with self.assertRaisesRegex(NativeProbeError, "process identity changed"):
            run_metadata_cycles(backend, SUBJECTS, 1, pid=lambda: next(readings))
        self.assertEqual([e[0] for e in backend.events], ["load", "free"])
        self.assertIsNone(backend.loaded)

    def test_failing_load_does_not_fake_success_or_free_unowned_handle(self):
        class BrokenLoader(FakeLoader):
            def load_metadata(self, path):
                raise NativeProbeError("native load failed")

        backend = BrokenLoader()
        with self.assertRaisesRegex(NativeProbeError, "native load failed"):
            run_metadata_cycles(backend, SUBJECTS, 1)
        self.assertEqual(backend.events, [])

    def test_ambiguous_or_invalid_subjects_rejected_without_loading(self):
        backend = FakeLoader()
        bad = (
            {"A": SUBJECTS["A"]},
            {"A": SUBJECTS["A"], "B": {**SUBJECTS["B"], "sha256": "a" * 64}},
            {"A": SUBJECTS["A"], "B": {**SUBJECTS["B"], "size_bytes": True}},
            {"A": SUBJECTS["A"], "B": {**SUBJECTS["B"], "sha256": "z" * 64}},
        )
        for subject in bad:
            with self.subTest(subject=subject), self.assertRaises(NativeProbeError):
                run_metadata_cycles(backend, subject, 1)
        self.assertEqual(backend.events, [])

    def test_cycle_limit_fails_closed(self):
        for cycles in (0, 11, True, 1.5, "10"):
            with self.subTest(cycles=cycles), self.assertRaises(NativeProbeError):
                run_metadata_cycles(FakeLoader(), SUBJECTS, cycles)


if __name__ == "__main__":
    unittest.main()
