import unittest
import json
from pathlib import Path
import tempfile
from unittest import mock

from experiments.p1_live_three_arm import (
    LivePairError, _parse_core, collect, main, verify_dedicated_router,
)


CASE = {"case_id": "logic-01", "prompt": "Is 2 + 2 equal to 4?",
        "expected_supports": True}
SUBJECTS = {
    "core-4b": {"path": "/models/core.gguf", "sha256": "a" * 64,
                "size_bytes": 2000},
    "critic-1p5b": {"path": "/models/critic.gguf", "sha256": "b" * 64,
                   "size_bytes": 1000},
}


class FakeRouter:
    def __init__(self, *, preloaded=None, spoof=False, bad_core=False,
                 invalid_reply=None, invalid_on_critic=False):
        self.loaded = preloaded
        self.events = []
        self.spoof = spoof
        self.bad_core = bad_core
        self.invalid_reply = invalid_reply
        self.invalid_on_critic = invalid_on_critic
        self.core_count = 0

    def statuses(self):
        return {name: "loaded" if name == self.loaded else "unloaded"
                for name in SUBJECTS}

    def paths(self):
        return {name: subject["path"] for name, subject in SUBJECTS.items()}

    def load(self, model):
        if self.loaded is not None:
            raise AssertionError("overlapping model load")
        self.loaded = model
        self.events.append(("load", model))

    def wait_status(self, model, expected_loaded):
        if (self.loaded == model) != expected_loaded:
            raise AssertionError("invalid router state")

    def unload(self, model):
        if self.loaded != model:
            raise AssertionError("unloaded wrong model")
        self.loaded = None
        self.events.append(("unload", model))

    def request(self, path, payload):
        if path != "/v1/chat/completions" or payload["model"] != self.loaded:
            raise AssertionError("request to wrong model")
        if payload.get("tools") != [] or payload.get("stream") is not False:
            raise AssertionError("all inference calls must disable tools and streaming")
        model = self.loaded
        if model == "core-4b":
            fmt = payload.get("response_format")
            if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
                raise AssertionError("Core requires strict four-field schema mode")
            schema = fmt.get("json_schema", {}).get("schema", {})
            if (set(schema.get("properties", {})) != {
                "proposition_id", "supports", "reasoning", "acknowledged_objections",
            } or set(schema.get("required", [])) != set(schema["properties"])
                    or schema.get("additionalProperties") is not False):
                raise AssertionError("Core schema must match actual four-field contract")
            question = json.loads(payload["messages"][1]["content"])
            if question.get("proposition_id") != CASE["case_id"]:
                raise AssertionError("unbound Core proposition")
            self.core_count += 1
            content = ("not-json" if self.bad_core else json.dumps({
                "proposition_id": CASE["case_id"],
                "supports": False if self.core_count == 2 else True,
                "reasoning": "A model-produced stance, not independent proof.",
                "acknowledged_objections": [],
            }))
        else:
            if "response_format" in payload:
                raise AssertionError("critic output should not be forced into Core schema")
            content = "Neutral observation." if len(self.events) < 4 else "Adversarial objection."
        reply = {"model": "wrong" if self.spoof else model,
                 "choices": [{"index": 0, "finish_reason": "stop",
                              "message": {"role": "assistant", "content": content}}],
                 "usage": {"prompt_tokens": 42, "completion_tokens": 8}}
        if self.invalid_reply and ((model == "critic-1p5b") == self.invalid_on_critic):
            choice = reply["choices"][0]
            message = choice["message"]
            case = self.invalid_reply
            if case == "missing_model":
                del reply["model"]
            elif case == "missing_index":
                del choice["index"]
            elif case == "bool_index":
                choice["index"] = False
            elif case == "wrong_index":
                choice["index"] = 1
            elif case == "multiple_choices":
                reply["choices"].append(dict(choice))
            elif case == "missing_role":
                del message["role"]
            elif case == "wrong_role":
                message["role"] = "system"
            elif case == "missing_finish":
                del choice["finish_reason"]
            elif case == "length_finish":
                choice["finish_reason"] = "length"
            elif case == "tool_finish":
                choice["finish_reason"] = "tool_calls"
            elif case == "tool_calls":
                message["tool_calls"] = [{"id": "t1", "type": "function"}]
            elif case == "empty_tool_calls":
                message["tool_calls"] = []
            else:
                raise AssertionError(f"unexpected hostile variant: {case}")
        return reply


class LiveThreeArmTests(unittest.TestCase):
    def run_one(self, router):
        traces = []
        doc, report = collect(
            router, CASE, "core-4b", "critic-1p5b", SUBJECTS,
            "live-v2-01", "logic-v1", "2026-10-25T00:00:00Z", 192, traces,
        )
        return doc, report, traces

    def test_real_three_arm_shape_five_actual_calls_and_unload(self):
        router = FakeRouter()
        doc, report, traces = self.run_one(router)
        self.assertEqual(len(traces), 5)
        self.assertEqual(len(router.events), 10)
        self.assertIsNone(router.loaded)
        self.assertEqual([event for event, _ in router.events].count("load"), 5)
        self.assertEqual([trace["role"] for trace in traces], [
            "core_initial", "neutral_critic", "core_reconsider",
            "learned_critic", "core_reconsider"])
        case = doc["cases"][0]
        self.assertEqual([len(case[a]["calls"]) for a in (
            "core_only", "neutral_three_call", "challenged")], [1, 3, 3])
        self.assertEqual(case["core_only"]["calls"][0],
                         case["challenged"]["calls"][0])
        self.assertFalse(report["neural_benefit_qualified"])
        self.assertEqual(report["cases"], 1)

    def test_unparseable_core_remains_contract_error(self):
        doc, report, _ = self.run_one(FakeRouter(bad_core=True))
        self.assertEqual(doc["cases"][0]["core_only"]["status"],
                         "output_contract_error")
        self.assertFalse(report["neural_benefit_qualified"])
        self.assertEqual(_parse_core("<think>not JSON</think>", CASE["case_id"]),
                         ("output_contract_error", None))

    def test_actual_core_parser_rejects_identity_schema_and_reasoning_failures(self):
        correct = {
            "proposition_id": CASE["case_id"], "supports": True,
            "reasoning": "Arithmetic supports the proposition.",
            "acknowledged_objections": [],
        }
        self.assertEqual(_parse_core(json.dumps(correct), CASE["case_id"]),
                         ("ok", True))
        variants = (
            {**correct, "proposition_id": "different"},
            {**correct, "supports": 1},
            {**correct, "reasoning": "   "},
            {**correct, "extra": "injected"},
            {**correct, "acknowledged_objections": ["critic", "critic"]},
            {**correct, "acknowledged_objections": [""]},
        )
        for answer in variants:
            with self.subTest(answer=answer):
                self.assertEqual(_parse_core(json.dumps(answer), CASE["case_id"]),
                                 ("output_contract_error", None))
        self.assertEqual(_parse_core("```json\n"+json.dumps(correct)+"\n```",
                                     CASE["case_id"]),
                         ("output_contract_error", None))

    def test_preloaded_router_rejected_without_effects(self):
        router = FakeRouter(preloaded="core-4b")
        with self.assertRaisesRegex(LivePairError, "unloaded"):
            verify_dedicated_router(router, SUBJECTS)
        self.assertEqual(router.events, [])

    def test_model_spoof_fails_and_unloads(self):
        router = FakeRouter(spoof=True)
        with self.assertRaisesRegex(LivePairError, "different model"):
            self.run_one(router)
        self.assertIsNone(router.loaded)
        self.assertEqual(router.events, [("load", "core-4b"),
                                         ("unload", "core-4b")])

    def test_core_and_critic_fail_closed_on_hostile_completion_envelopes(self):
        variants = (
            "missing_model", "missing_index", "bool_index", "wrong_index",
            "multiple_choices", "missing_role", "wrong_role",
            "missing_finish", "length_finish", "tool_finish",
            "tool_calls",
        )
        for on_critic in (False, True):
            for variant in variants:
                with self.subTest(on_critic=on_critic,variant=variant):
                    router = FakeRouter(invalid_reply=variant,
                                        invalid_on_critic=on_critic)
                    with self.assertRaises(LivePairError):
                        self.run_one(router)
                    self.assertIsNone(router.loaded)
                    self.assertEqual(router.events, (
                        [("load", "core-4b"), ("unload", "core-4b"),
                         ("load", "critic-1p5b"), ("unload", "critic-1p5b")]
                        if on_critic else
                        [("load", "core-4b"), ("unload", "core-4b")]
                    ))

    def test_empty_tool_call_metadata_is_not_a_dispatched_tool(self):
        # Match the production LoopbackCoreInference boundary: an empty list
        # is metadata, while nonempty tool calls must fail on both models.
        for on_critic in (False, True):
            with self.subTest(on_critic=on_critic):
                router = FakeRouter(invalid_reply="empty_tool_calls",
                                    invalid_on_critic=on_critic)
                doc, report, traces = self.run_one(router)
                self.assertEqual(len(traces), 5)
                self.assertIsNone(router.loaded)
                self.assertFalse(report["neural_benefit_qualified"])

    def test_mismatched_model_path_rejected_before_loading(self):
        router = FakeRouter()
        router.paths = lambda: {"core-4b": "/wrong.gguf",
                                "critic-1p5b": "/models/critic.gguf"}
        with self.assertRaisesRegex(LivePairError, "differs"):
            self.run_one(router)
        self.assertEqual(router.events, [])

    def test_same_model_weights_under_different_ids_rejected(self):
        router = FakeRouter()
        same_weights = {key: dict(value) for key, value in SUBJECTS.items()}
        same_weights["critic-1p5b"]["sha256"] = same_weights["core-4b"]["sha256"]
        with self.assertRaisesRegex(LivePairError, "distinct model file digests"):
            collect(router, CASE, "core-4b", "critic-1p5b", same_weights,
                    "live-v2-01", "logic-v1", "2026-10-25T00:00:00Z", 192, [])
        self.assertEqual(router.events, [])

    def test_bad_metadata_and_bounded_case_rejected_before_model_calls(self):
        for label, overrides in (
            ("invalid freeze", {"frozen": "tomorrow"}),
            ("invalid calendar", {"frozen": "2026-02-30T00:00:00Z"}),
            ("oversized experiment ID", {"experiment_id": "x" * 129}),
            ("oversized corpus ID", {"corpus_id": "x" * 129}),
            ("oversized case prompt", {"case": {**CASE, "prompt": "x" * 8193}}),
            ("bad unicode", {"case": {**CASE, "case_id": "a\ud800"}}),
        ):
            with self.subTest(label=label):
                router = FakeRouter()
                args = {"case": CASE, "experiment_id": "live-v2-01",
                        "corpus_id": "logic-v1", "frozen": "2026-10-25T00:00:00Z"}
                args.update(overrides)
                with self.assertRaises(LivePairError):
                    collect(router, args["case"], "core-4b", "critic-1p5b", SUBJECTS,
                            args["experiment_id"], args["corpus_id"],
                            args["frozen"], 192, [])
                self.assertEqual(router.events, [])

    def test_insufficient_four_field_token_budget_rejected_before_model_load(self):
        router = FakeRouter()
        for cap in (0, 32, 95, 385, True, "192"):
            with self.subTest(cap=cap), self.assertRaises(LivePairError):
                collect(router, CASE, "core-4b", "critic-1p5b", SUBJECTS,
                        "live-v2-01", "logic-v1", "2026-10-25T00:00:00Z",
                        cap, [])
            self.assertEqual(router.events, [])

    def test_existing_output_file_cannot_be_overwritten_or_trigger_model_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "case.json"
            source.write_text('{"case_id":"a","prompt":"q","expected_supports":true}')
            existing = root / "result.json"
            existing.write_bytes(b"DO NOT REPLACE")
            argv = ["runner", "--router-url", "http://127.0.0.1:8518",
                    "--dedicated-router", "--core-model", "core-4b",
                    "--critic-model", "critic-1p5b", "--core-model-file", "any",
                    "--critic-model-file", "other", "--case-file", str(source),
                    "--experiment-id", "x", "--corpus-id", "c",
                    "--frozen-at", "2026-10-25T00:00:00Z", "--out", str(existing)]
            with mock.patch("sys.argv", argv), mock.patch(
                    "experiments.p1_live_three_arm._subject",
                    side_effect=AssertionError("should not hash models")):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(existing.read_bytes(), b"DO NOT REPLACE")

    def test_oversized_case_file_is_read_with_a_fixed_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "large-case.json"
            source.write_bytes(b"x" * 9000)
            target = Path(folder) / "result.json"
            argv = ["runner", "--router-url", "http://127.0.0.1:8518",
                    "--dedicated-router", "--core-model", "core-4b",
                    "--critic-model", "critic-1p5b", "--core-model-file", "any",
                    "--critic-model-file", "other", "--case-file", str(source),
                    "--experiment-id", "x", "--corpus-id", "c",
                    "--frozen-at", "2026-10-25T00:00:00Z", "--out", str(target)]
            with mock.patch("sys.argv", argv), mock.patch.object(
                    Path, "read_bytes", side_effect=AssertionError("unbounded read")):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
