"""Exercise real source guards without requiring optional PyTorch in CI."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "experiments" / "p1_toy_latent_benchmark.py"
RECEIPT = Path(__file__).resolve().parents[1] / "experiments" / "results" / "P0_NEURAL_COUNT_V1.json"


def source_function(name, **scope):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(item for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {"Path": Path, **scope}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


class HostEvidenceGuardsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))

    def test_rejects_empty_or_attacker_selected_checkpoint_names(self):
        expected = self.receipt["manifest"]["weights"]
        claims = {"dataset_sha256": "expected", "seed": 20261026, "epochs": 100,
                  "train_count": 4, "dev_count": 2, "weights": expected}
        calls = []
        check = source_function(
            "verify_manifest", json=json, dataset_sha=lambda _: "expected",
            EXACT_PUBLISHED_WEIGHTS=expected, SEED=20261026, EPOCHS=100,
            file_sha=lambda path: calls.append(path.name) or expected[path.name],
        )
        data = {"train": [1] * 4, "dev": [1] * 2}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "train_manifest.json"
            for untrusted in ({}, {"/dev/zero": "bad"},
                              {"source.safetensors": expected["source.safetensors"]}):
                manifest.write_text(json.dumps({**claims, "weights": untrusted}))
                with self.assertRaisesRegex(RuntimeError, "Manifest does not bind"):
                    check(root, data)
            self.assertEqual(calls, [])
            manifest.write_text(json.dumps(claims))
            self.assertEqual(check(root, data), claims)
            self.assertEqual(set(calls), set(expected))

    def test_existing_receipt_and_symlink_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            child = base / "toy_count_v1"
            child.mkdir()
            write = source_function("write_once_or_match", frozen_root=lambda: child)
            target = child / "receipt.json"
            write(target, b'{"case":1}')
            write(target, b'{"case":1}')
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                write(target, b'{"case":2}')
            self.assertEqual(target.read_bytes(), b'{"case":1}')
            sentinel = base / "sentinel.txt"
            sentinel.write_bytes(b"untouched")
            link = child / "other.json"
            try:
                link.symlink_to(sentinel)
            except OSError:
                self.skipTest("Symlink creation unavailable on this platform")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                write(link, b"damage")
            self.assertEqual(sentinel.read_bytes(), b"untouched")


if __name__ == "__main__":
    unittest.main()
