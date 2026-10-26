import unittest
from io import BytesIO, StringIO
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import mock

from experiments.p0_router_probe import (
    ProbeError,
    Router,
    main,
    parse_nvidia_mib,
    require_loopback_router,
    reserve_receipt,
    run_probe,
)


class FakeRouter:
    def __init__(self, *, preloaded=False):
        self.loaded = "A" if preloaded else None
        self.transitions = []

    def statuses(self):
        return {
            name: ("loaded" if self.loaded == name else "unloaded")
            for name in ("A", "B")
        }

    def paths(self):
        return {"A": "/models/a.gguf", "B": "/models/b.gguf"}

    def load(self, name):
        if self.loaded is not None:
            raise AssertionError("overlap")
        self.loaded = name
        self.transitions.append(("load", name))

    def unload(self, name):
        if self.loaded != name:
            raise AssertionError("wrong unload")
        self.loaded = None
        self.transitions.append(("unload", name))

    def wait_status(self, name, expected_loaded):
        assert (self.loaded == name) == expected_loaded

    def generate_stream(self, name):
        if self.loaded != name:
            raise AssertionError("no live model")
        return {"first_token_latency_s": 0.5, "observed_content_chunks": 2}


class FakeSampler:
    samples = [
        {"monotonic_s": 1.0, "used_mib": 1200, "total_mib": 4096},
        {"monotonic_s": 2.0, "used_mib": 2800, "total_mib": 4096},
    ]


MODEL_FILES = {
    "A": {"sha256": "a" * 64, "size_bytes": 1646573120, "path": "/models/a.gguf"},
    "B": {"sha256": "b" * 64, "size_bytes": 3072993056, "path": "/models/b.gguf"},
}


class P0RouterProbeTests(unittest.TestCase):
    @staticmethod
    def _argv_for_output(directory: Path, output: Path) -> list[str]:
        model_a = directory / "a.gguf"
        model_b = directory / "b.gguf"
        model_a.write_bytes(b"a")
        model_b.write_bytes(b"b")
        return [
            "p0_router_probe.py", "--router-url", "http://127.0.0.1:8517",
            "--dedicated-router", "--model-a", "A", "--model-b", "B",
            "--model-a-file", str(model_a), "--model-b-file", str(model_b),
            "--out", str(output),
        ]

    def test_existing_receipt_refused_before_hashing_or_router_access(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "protected.json"
            output.write_text("sentinel receipt", encoding="utf-8")
            argv = self._argv_for_output(directory, output)
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("experiments.p0_router_probe.file_sha256",
                           side_effect=AssertionError("hashing must not run")),
                mock.patch("experiments.p0_router_probe.Router",
                           side_effect=AssertionError("router must not run")),
                mock.patch.object(sys, "stderr", StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                main()
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(output.read_text(encoding="utf-8"), "sentinel receipt")

    def test_symlinked_receipt_refused_without_modifying_target(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            target = directory / "protected.json"
            target.write_text("target sentinel", encoding="utf-8")
            link = directory / "receipt-link.json"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError) as exc:
                self.skipTest(f"symlink unavailable: {exc}")
            argv = self._argv_for_output(directory, link)
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("experiments.p0_router_probe.file_sha256",
                           side_effect=AssertionError("hashing must not run")),
                mock.patch("experiments.p0_router_probe.Router",
                           side_effect=AssertionError("router must not run")),
                mock.patch.object(sys, "stderr", StringIO()),
                self.assertRaises(SystemExit) as caught,
            ):
                main()
            self.assertEqual(caught.exception.code, 2)
            self.assertTrue(link.is_symlink())
            self.assertEqual(target.read_text(encoding="utf-8"), "target sentinel")

    def test_new_receipt_is_exclusive_and_private(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "new-receipt.json"
            with reserve_receipt(output) as receipt:
                receipt.write('{"qualification":"PARTIAL_NOT_P0_PASS"}\n')
            self.assertEqual(output.read_text(encoding="utf-8"),
                             '{"qualification":"PARTIAL_NOT_P0_PASS"}\n')
            with self.assertRaises(FileExistsError):
                reserve_receipt(output)
            if os.name == "posix":
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    @staticmethod
    def memory_router(response: bytes, *, timeout_s: float = 90.0) -> Router:
        class StaticOpener:
            def open(self, request, timeout):
                return BytesIO(response)

        router = Router("http://127.0.0.1:8500", timeout_s=timeout_s)
        router._opener = StaticOpener()
        return router

    def test_control_api_rejects_oversized_router_json(self):
        router = self.memory_router(b'{"data":"' + b'x' * (1024 * 1024) + b'"}')
        with self.assertRaisesRegex(ProbeError, "byte limit"):
            router.request("/models")

    def test_stream_requires_completion_marker(self):
        router = self.memory_router(b'data: {"content":"ready"}\n')
        with self.assertRaisesRegex(ProbeError, "completion marker"):
            router.generate_stream("A")

    def test_stream_caps_oversized_line_and_event_flood(self):
        for body in (
            b'data: ' + b' ' * (64 * 1024) + b'\n',
            b'data: {"content":"x"}\n' * 2049,
        ):
            with self.subTest(body_len=len(body)):
                router = self.memory_router(body)
                with self.assertRaisesRegex(ProbeError, "limit"):
                    router.generate_stream("A")

    def test_stream_deadline_is_checked_during_active_response(self):
        router = self.memory_router(
            b'data: {"content":"ready"}\ndata: {"content":"again","stop":true}\n',
            timeout_s=1.0,
        )
        with mock.patch("experiments.p0_router_probe.time.monotonic",
                        side_effect=(0.0, 0.2, 1.2)):
            with self.assertRaisesRegex(ProbeError, "deadline"):
                router.generate_stream("A")

    def test_valid_nvidia_sample(self):
        self.assertEqual(parse_nvidia_mib(" 1234, 4096\n"), (1234, 4096))

    def test_nvidia_bad_sample_fails(self):
        for bad in ("", "0,0", "4097,4096", "N/A,4096", "1,2,3"):
            with self.subTest(bad=bad), self.assertRaises(ProbeError):
                parse_nvidia_mib(bad)

    def test_only_loopback_router(self):
        self.assertEqual(require_loopback_router("http://127.0.0.1:8500/"), "http://127.0.0.1:8500")
        for bad in (
            "https://127.0.0.1:8500",
            "http://192.168.1.12:8500",
            "http://127.0.0.1:8500/admin",
            "http://a@127.0.0.1:8500",
            "http://localhost:8500?foo=bar",
        ):
            with self.subTest(bad=bad), self.assertRaises(ProbeError):
                require_loopback_router(bad)

    def test_ten_cycles_are_real_ordered_load_infer_unload_events(self):
        router = FakeRouter()
        times = iter(i / 10 for i in range(1000))
        receipt = run_probe(
            router,
            "A",
            "B",
            10,
            FakeSampler(),
            MODEL_FILES,
            clock=lambda: next(times),
        )
        self.assertEqual(receipt["completed_aba_cycles"], 10)
        self.assertEqual(len(receipt["steps"]), 30)
        self.assertEqual(router.transitions[:6], [
            ("load", "A"), ("unload", "A"),
            ("load", "B"), ("unload", "B"),
            ("load", "A"), ("unload", "A"),
        ])
        self.assertIsNone(router.loaded)
        self.assertEqual(receipt["device_peak_used_mib"], 2800)
        self.assertTrue(receipt["logical_package_pool_exceeds_physical_vram"])
        self.assertFalse(receipt["p0_pass"])
        self.assertFalse(receipt["per_model_allocated_reserved_verified"])

    def test_preloaded_router_rejected_without_changes(self):
        router = FakeRouter(preloaded=True)
        with self.assertRaisesRegex(ProbeError, "zero loaded"):
            run_probe(router, "A", "B", 10, FakeSampler(), MODEL_FILES)
        self.assertEqual(router.transitions, [])

    def test_false_proof_rejected(self):
        for left, right, cycles in (("A", "A", 10), ("A", "B", 9)):
            with self.subTest(left=left, right=right, cycles=cycles):
                with self.assertRaises(ProbeError):
                    run_probe(FakeRouter(), left, right, cycles, FakeSampler(), MODEL_FILES)

    def test_missing_exact_model_subject_is_rejected_before_router_use(self):
        router = FakeRouter()
        with self.assertRaisesRegex(ProbeError, "exact file size"):
            run_probe(router, "A", "B", 10, FakeSampler(), {"A": {"sha256": "not a hash"}})
        self.assertEqual(router.transitions, [])

    def test_failed_inference_unloads_model_before_bubbling_error(self):
        class BrokenRouter(FakeRouter):
            def generate_stream(self, name):
                raise ProbeError("inference failed")

        router = BrokenRouter()
        with self.assertRaisesRegex(ProbeError, "inference failed"):
            run_probe(router, "A", "B", 10, FakeSampler(), MODEL_FILES)
        self.assertEqual(router.transitions, [("load", "A"), ("unload", "A")])
        self.assertIsNone(router.loaded)

    def test_lost_load_ack_after_router_accepts_model_still_unloads(self):
        class LostAckRouter(FakeRouter):
            def load(self, name):
                super().load(name)
                raise ProbeError("model loaded but acknowledgement lost")

        router = LostAckRouter()
        with self.assertRaisesRegex(ProbeError, "acknowledgement lost"):
            run_probe(router, "A", "B", 10, FakeSampler(), MODEL_FILES)
        self.assertEqual(router.transitions, [("load", "A"), ("unload", "A")])
        self.assertIsNone(router.loaded)

    def test_load_rejected_without_mutation_never_unloads(self):
        class RejectedRouter(FakeRouter):
            def load(self, name):
                raise ProbeError("load rejected before mutation")

        router = RejectedRouter()
        with self.assertRaisesRegex(ProbeError, "load rejected"):
            run_probe(router, "A", "B", 10, FakeSampler(), MODEL_FILES)
        self.assertEqual(router.transitions, [])
        self.assertIsNone(router.loaded)

    def test_failed_cleanup_preserves_primary_load_error_and_reports_uncertainty(self):
        class UnreachableRouter(FakeRouter):
            def load(self, name):
                super().load(name)
                raise ProbeError("load reply disconnected")

            def unload(self, name):
                raise ProbeError("unload reply disconnected")

        router = UnreachableRouter()
        with self.assertRaisesRegex(ProbeError, "load reply disconnected") as caught:
            run_probe(router, "A", "B", 10, FakeSampler(), MODEL_FILES)
        self.assertTrue(any("uncertain router cleanup" in note
                            for note in getattr(caught.exception, "__notes__", [])))
        self.assertEqual(router.transitions, [("load", "A")])

    def test_boolean_or_fake_hash_cannot_bind_model(self):
        for data in (
            {"size_bytes": True, "sha256": "a" * 64},
            {"size_bytes": 200, "sha256": "A" * 64},
            {"size_bytes": 200, "sha256": "a" * 63},
        ):
            with self.subTest(data=data), self.assertRaisesRegex(ProbeError, "SHA-256"):
                run_probe(FakeRouter(), "A", "B", 10, FakeSampler(), {"A": data, "B": MODEL_FILES["B"]})

    def test_model_id_cannot_be_bound_to_unrelated_file(self):
        class MisleadingRouter(FakeRouter):
            def paths(self):
                return {"A": "/models/not-a.gguf", "B": "/models/b.gguf"}

        router = MisleadingRouter()
        with self.assertRaisesRegex(ProbeError, "advertised model path"):
            run_probe(router, "A", "B", 10, FakeSampler(), MODEL_FILES)
        self.assertEqual(router.transitions, [])

    def test_preset_model_paths_are_read_from_router_status_args(self):
        # Observed llama.cpp 0.4.1-dev on Windows: preset entries omit top-level
        # ``path`` and advertise the exact model file in status.args instead.
        router = Router("http://127.0.0.1:8517")
        router.model_records = lambda: {
            "code-1p5b": {"status": {"value": "unloaded", "args": [
                "llama-server.exe", "--alias", "code-1p5b", "--model",
                "D:/OllamaModels/blobs/sha256-model-a",
            ]}},
            "vera-4b": {"status": {"value": "unloaded", "args": [
                "llama-server.exe", "--model", "D:/VERA/Models/gguf/model-b.gguf",
            ]}},
        }
        self.assertEqual(router.paths(), {
            "code-1p5b": "D:/OllamaModels/blobs/sha256-model-a",
            "vera-4b": "D:/VERA/Models/gguf/model-b.gguf",
        })

    def test_ambiguous_preset_model_path_fails_closed(self):
        router = Router("http://127.0.0.1:8517")
        for args in (
            ["--model", "/models/a.gguf", "--model", "/models/b.gguf"],
            ["--model"],
            ["--model", ""],
            ["--model", 3],
        ):
            with self.subTest(args=args):
                router.model_records = lambda args=args: {
                    "A": {"status": {"args": args}}
                }
                with self.assertRaises(ProbeError):
                    router.paths()


if __name__ == "__main__":
    unittest.main()
