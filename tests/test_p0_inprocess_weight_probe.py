"""Behavior and safety tests for the real-neural-weight P0 experiment."""

from pathlib import Path
import unittest

from experiments.p0_inprocess_abi_probe import NativeProbeError
from experiments.p0_inprocess_weight_probe import observed_gpu_peak_mib, run_weight_cycle


class _Loader:
    def __init__(self, *, bad_dimensions=False):
        self.calls = []
        self.bad_dimensions = bad_dimensions

    def load_weights(self, path, layers):
        self.calls.append(("load", path.name, layers))
        return path.name, {"n_layer": 0 if self.bad_dimensions else 24,
                           "n_params": 1_600_000_000}

    def free(self, model):
        self.calls.append(("free", model))


SUBJECTS = {
    "A": {"path": Path("a.gguf"), "size_bytes": 12345, "sha256": "a" * 64},
    "B": {"path": Path("b.gguf"), "size_bytes": 23456, "sha256": "b" * 64},
}


class WeightProbeTests(unittest.TestCase):
    def test_synchronous_gpu_peak_is_not_lost_to_polling_aliasing(self):
        self.assertEqual(observed_gpu_peak_mib(
            [{"used_mib": 1380}], [{"device_used_before_mib": 1376,
                                      "device_used_loaded_mib": 2328,
                                      "device_used_freed_mib": 1380}]), 2328)
        self.assertIsNone(observed_gpu_peak_mib([], []))

    def test_real_load_contract_is_not_automatic_p0_pass(self):
        loader = _Loader()
        result = run_weight_cycle(
            loader, SUBJECTS, cycles=2, gpu_layers=12,
            min_free_ram_mib=2300, available_ram=lambda: 3000,
            gpu=lambda: (4096, 1500), pid=lambda: 42,
        )
        self.assertEqual(result["neural_aba_cycles"], 2)
        self.assertEqual(len(result["steps"]), 6)
        self.assertEqual([s["role"] for s in result["steps"]],
                         ["A", "B", "A", "A", "B", "A"])
        self.assertEqual(len([c for c in loader.calls if c[0] == "free"]), 6)
        self.assertTrue(result["neural_weights_loaded"])
        self.assertFalse(result["p0_pass"])
        self.assertFalse(result["inference_completed"])
        self.assertFalse(result["shared_learned_state_demonstrated"])

    def test_preflight_stops_before_loading_under_ram_floor(self):
        loader = _Loader()
        with self.assertRaisesRegex(NativeProbeError, "memory safety stop"):
            run_weight_cycle(loader, SUBJECTS, cycles=1, gpu_layers=16,
                             min_free_ram_mib=2300,
                             available_ram=lambda: 2299,
                             gpu=lambda: (4096, 1500))
        self.assertEqual(loader.calls, [])

    def test_invalid_neural_dimensions_free_handle(self):
        loader = _Loader(bad_dimensions=True)
        with self.assertRaisesRegex(NativeProbeError, "neural dimensions"):
            run_weight_cycle(loader, SUBJECTS, cycles=1, gpu_layers=16,
                             min_free_ram_mib=2300,
                             available_ram=lambda: 3000,
                             gpu=lambda: (4096, 1500))
        self.assertEqual(loader.calls[:2], [("load", "a.gguf", 16),
                                           ("free", "a.gguf")])

    def test_rejects_same_revision_and_bad_cycle_bounds(self):
        loader = _Loader()
        repeated = {"A": SUBJECTS["A"], "B": SUBJECTS["A"]}
        with self.assertRaisesRegex(NativeProbeError, "distinct"):
            run_weight_cycle(loader, repeated, cycles=1, gpu_layers=16,
                             min_free_ram_mib=2300)
        for value in (0, 11, True):
            with self.subTest(value=value), self.assertRaises(NativeProbeError):
                run_weight_cycle(loader, SUBJECTS, cycles=value, gpu_layers=16,
                                 min_free_ram_mib=2300)

    def test_pid_change_always_frees_model(self):
        loader = _Loader()
        values = iter([7, 8])
        with self.assertRaisesRegex(NativeProbeError, "PID changed"):
            run_weight_cycle(loader, SUBJECTS, cycles=1, gpu_layers=16,
                             min_free_ram_mib=2300,
                             available_ram=lambda: 3000,
                             gpu=lambda: (4096, 1500), pid=lambda: next(values))
        self.assertEqual(loader.calls[:2], [("load", "a.gguf", 16),
                                           ("free", "a.gguf")])


if __name__ == "__main__":
    unittest.main()
