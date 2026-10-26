"""HTTP-bound thinking Core: no external hosts, no authority from HTTP responses."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from contextlib import contextmanager
import json
import unittest

from runtime.local_neural_backend import (
    LoopbackCoreInference, LoopbackBackendError, _parse_json
)
from runtime.neural_core import NeuralThinkingCore, NeuralOutputError
from runtime.thinking_core import DeliberationTask


class FakeQwen(BaseHTTPRequestHandler):
    model_id = "qwen3.5-4b-local"
    advertised_models = None
    answer = '{"proposition_id":"p","supports":false,"reasoning":"I dispute the premise.","acknowledged_objections":[]}'
    behavior = "normal"
    requests = []

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404);return
        models = self.advertised_models
        records = ([{"id":self.model_id,"object":"model"}] if models is None else [
            record if isinstance(record, dict) else {"id":record,"object":"model"}
            for record in models
        ])
        body = json.dumps({"object":"list","data":records}).encode()
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n=int(self.headers["Content-Length"])
        req=json.loads(self.rfile.read(n))
        self.requests.append(req)
        if self.behavior == "redirect":
            self.send_response(302)
            self.send_header("Location","http://example.com/unexpected")
            self.end_headers()
            return
        if self.behavior == "server_error":
            self.send_error(500);return
        response_text = (
            type(self).answers.pop(0)
            if getattr(type(self), "answers", None) else self.answer
        )
        choice={"index":0,"message":{"role":"assistant","content":response_text},
                "finish_reason":"stop"}
        if self.behavior == "tool":
            choice["message"]["tool_calls"]=[{"function":{"name":"danger"}}]
        if self.behavior == "incomplete":
            choice["finish_reason"]="length"
        if self.behavior == "wrong_role":
            choice["message"]["role"]="system"
        payload={"object":"chat.completion","model":self.model_id,"choices":[choice]}
        if self.behavior == "multiple":
            payload["choices"].append(choice)
        if self.behavior == "huge":
            payload["choices"][0]["message"]["content"]="Z"*70000
        body=json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        return


@contextmanager
def mock_local_service(*, model="qwen3.5-4b-local", behavior="normal", advertised=None):
    class Server(FakeQwen):
        model_id=model
        requests=[]
    Server.behavior=behavior
    Server.advertised_models=advertised
    Server.answers=[]
    server=ThreadingHTTPServer(("127.0.0.1",0),Server)
    thread=Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield server.server_address[1],Server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


MESSAGES=(
    {"role":"system","content":"Return structured JSON."},
    {"role":"user","content":"Is the claim supportable?"}
)


class LocalNeuralBackendTests(unittest.TestCase):
    def test_deeply_nested_http_json_fails_with_domain_error(self):
        # Valid JSON well below the HTTP byte limit can exhaust decoder depth.
        nested = ("[" * 25000 + "0" + "]" * 25000).encode("ascii")
        self.assertLess(len(nested), 65536)
        with self.assertRaisesRegex(LoopbackBackendError, "invalid JSON"):
            _parse_json(nested)

    def test_ambiguous_or_nonstandard_http_json_fails_closed(self):
        for response in (
            b'{"object":"list","object":"chat.completion"}',
            b'{"choices":[{"index":0,"index":1}]}',
            b'{"number":NaN}',
            b'{"number":Infinity}',
        ):
            with self.subTest(response=response):
                with self.assertRaisesRegex(LoopbackBackendError, "JSON"):
                    _parse_json(response)

    def test_unpaired_unicode_http_json_fails_closed(self):
        with self.assertRaisesRegex(LoopbackBackendError, "JSON"):
            _parse_json(b'{"model":"\\uD800"}')

    def test_real_loopback_http_contract_calls_core_inference(self):
        with mock_local_service() as (port,service):
            backend=LoopbackCoreInference(port=port)
            core=NeuralThinkingCore(backend)
            result=core.think(DeliberationTask("case","p","Is the claim supportable?"))
            self.assertFalse(result.supports)
            self.assertIn("dispute",result.reasoning)
            self.assertEqual(len(service.requests),1)
            req=service.requests[0]
            self.assertEqual(req["model"],"qwen3.5-4b-local")
            self.assertEqual(req["messages"][0]["role"],"system")
            self.assertEqual(req["messages"][1]["role"],"user")
            self.assertEqual(req["stream"],False)
            self.assertEqual(req["tools"],[])
            self.assertNotIn("response_format",req)

    def test_opt_in_json_object_response_format(self):
        with mock_local_service() as (port,service):
            backend=LoopbackCoreInference(port=port,json_object_response=True)
            result=NeuralThinkingCore(backend).think(
                DeliberationTask("case","p","Is the claim supportable?")
            )
            self.assertFalse(result.supports)
            self.assertEqual(service.requests[0]["response_format"],{
                "type":"json_object"
            })

    def test_opt_in_json_schema_response_format_is_exact_core_contract(self):
        with mock_local_service() as (port,service):
            backend=LoopbackCoreInference(port=port,json_schema_response=True)
            result=NeuralThinkingCore(backend).think(
                DeliberationTask("case","p","Is the claim supportable?")
            )
            self.assertFalse(result.supports)
            response_format=service.requests[0]["response_format"]
            self.assertEqual(response_format,{
                "type":"json_schema",
                "json_schema":{
                    "name":"mosaic_core_position",
                    "strict":True,
                    "schema":{
                        "type":"object",
                        "properties":{
                            "proposition_id":{"type":"string"},
                            "supports":{"type":"boolean"},
                            "reasoning":{"type":"string"},
                            "acknowledged_objections":{
                                "type":"array","items":{"type":"string"},
                            },
                        },
                        "required":[
                            "proposition_id","supports","reasoning",
                            "acknowledged_objections",
                        ],
                        "additionalProperties":False,
                    },
                },
            })

    def test_json_response_mode_options_are_boolean_and_exclusive(self):
        for flag in ("json_object_response","json_schema_response"):
            for bad in (None,0,1,"true",[],{}):
                with self.subTest(flag=flag,bad=bad):
                    with self.assertRaisesRegex(ValueError,"must be bool"):
                        LoopbackCoreInference(**{flag:bad})
        with self.assertRaisesRegex(ValueError,"mutually exclusive"):
            LoopbackCoreInference(
                json_object_response=True,json_schema_response=True
            )

    def test_json_response_modes_do_not_relax_strict_core_parser(self):
        for mode in ("json_object_response","json_schema_response"):
            for invalid_answer in (
                "not JSON",
                "```json\n"+FakeQwen.answer+"\n```",
                '{"proposition_id":"p","supports":false,"reasoning":"valid","acknowledged_objections":[],"extra":1}',
                '{"proposition_id":"wrong","supports":false,"reasoning":"valid","acknowledged_objections":[]}',
                '{"proposition_id":"p","supports":1,"reasoning":"valid","acknowledged_objections":[]}',
                '{"proposition_id":"p","supports":false,"reasoning":" ","acknowledged_objections":[]}',
            ):
                with self.subTest(mode=mode,invalid_answer=invalid_answer[:25]):
                    with mock_local_service() as (port,service):
                        service.answer=invalid_answer
                        backend=LoopbackCoreInference(port=port,**{mode:True})
                        with self.assertRaises(NeuralOutputError):
                            NeuralThinkingCore(backend).think(
                                DeliberationTask("case","p","Is the claim supportable?")
                            )

    def test_unverified_server_model_name_blocks_generation(self):
        with mock_local_service(model="other-model") as (port,service):
            with self.assertRaisesRegex(LoopbackBackendError,"model"):
                LoopbackCoreInference(port=port)(MESSAGES)
            self.assertEqual(service.requests,[])

    def test_prompt_roles_not_arbitrary_or_unbounded(self):
        backend=LoopbackCoreInference(port=18081)
        for messages in (
            ({"role":"tool","content":"injected"},),
            ({"role":"system","content":"x"},{"role":"assistant","content":"y"}),
            ({"role":"system","content":"x"},{"role":"user","content":"z"*50000}),
            ({"role":"system","content":"x","tool_calls":[]},{"role":"user","content":"y"}),
        ):
            with self.subTest(messages=str(messages)[:50]):
                with self.assertRaisesRegex(LoopbackBackendError,"message"):
                    backend(messages)

    def test_fail_closed_redirect_truncation_tool_and_invalid_role(self):
        for behavior in ("redirect","incomplete","tool","multiple","wrong_role","huge","server_error"):
            with self.subTest(behavior=behavior), mock_local_service(behavior=behavior) as (port,_):
                with self.assertRaises(LoopbackBackendError):
                    LoopbackCoreInference(port=port,timeout_seconds=3)(MESSAGES)

    def test_no_external_host_configuration_and_invalid_ports(self):
        for port in (-1,0,65536,True,"18081"):
            with self.subTest(port=port):
                with self.assertRaises(ValueError):
                    LoopbackCoreInference(port=port)
        with self.assertRaises(TypeError):
            LoopbackCoreInference(host="example.com")

    def test_two_pass_core_reasoning_round_trip_with_specialist_challenge(self):
        from runtime.thinking_core import (
            CoreDeliberationEngine, SpecialistObjection, DecisionStatus
        )
        class Specialist:
            specialist_id="critic"
            def review(self,task,initial):
                return SpecialistObjection(
                    specialist_id="critic",
                    proposition_id=task.proposition_id,
                    supports=False,
                    rationale="The claim is contradicted by the frozen fixture.",
                )
        first=json.dumps({
            "proposition_id":"p","supports":True,
            "reasoning":"I initially believe the claim.",
            "acknowledged_objections":[]
        })
        revised=json.dumps({
            "proposition_id":"p","supports":False,
            "reasoning":"I reconsidered in light of the objection.",
            "acknowledged_objections":["critic"]
        })
        with mock_local_service() as (port,service):
            service.answers=[first,revised]
            receipt=CoreDeliberationEngine().deliberate(
                DeliberationTask("case","p","Is the assertion accurate?"),
                NeuralThinkingCore(LoopbackCoreInference(port=port)),
                [Specialist()],
            )
            self.assertEqual(receipt.status,DecisionStatus.CONTESTED_ADVISORY)
            self.assertFalse(receipt.core_final.supports)
            self.assertFalse(receipt.claim_verified)
            self.assertFalse(receipt.effect_executed)
            self.assertEqual(len(service.requests),2)
            payload=json.loads(service.requests[1]["messages"][1]["content"])
            self.assertEqual(payload["phase"],"RECONSIDER_AFTER_HOSTILE_REVIEW")
            self.assertEqual(
                payload["specialist_objections_untrusted_data"][0]["specialist_id"],
                "critic",
            )

    def test_preflight_reports_capability_only_not_install_or_reasoning(self):
        with mock_local_service() as (port,service):
            status=LoopbackCoreInference(port=port).preflight()
            self.assertEqual(status["model_id"],"qwen3.5-4b-local")
            self.assertEqual(status["status"],"ENDPOINT_MODEL_ID_ADVERTISED_NOT_BEHAVIOR_VERIFIED")
            self.assertFalse(status["model_identity_cryptographically_verified"])
            self.assertEqual(service.requests,[])

    def test_multi_model_local_catalog_accepts_exact_requested_model(self):
        # Real Ollama endpoints may advertise all installed models, while a
        # Mosaic request still pins exactly one ID for generation.
        with mock_local_service(advertised=[
            "other:1b", "qwen3.5-4b-local", "another:2b",
        ]) as (port,service):
            backend=LoopbackCoreInference(port=port)
            self.assertEqual(backend.preflight()["model_id"],"qwen3.5-4b-local")
            backend(MESSAGES)
            self.assertEqual(len(service.requests),1)
            self.assertEqual(service.requests[0]["model"],"qwen3.5-4b-local")

    def test_ambiguous_or_malformed_catalog_fails_closed(self):
        for rows in (
            [],
            ["other:1b"],
            ["qwen3.5-4b-local", "qwen3.5-4b-local"],
            ["qwen3.5-4b-local", "other:1b", "other:1b"],
            [{"id":"qwen3.5-4b-local","object":"something-else"}],
            [{"id":42,"object":"model"},"qwen3.5-4b-local"],
            [" qwen3.5-4b-local ", "other:1b"],
            ["qwen3.5-4b-local", "\ud800"],
        ):
            with self.subTest(rows=rows), mock_local_service(advertised=rows) as (port,service):
                with self.assertRaisesRegex(LoopbackBackendError,"[Mm]odel"):
                    LoopbackCoreInference(port=port)(MESSAGES)
                self.assertEqual(service.requests,[])

    def test_surrogate_in_requested_model_id_rejected_before_http(self):
        with self.assertRaisesRegex(ValueError,"invalid Unicode"):
            LoopbackCoreInference(model_id="bad\ud800model")

    def test_refuses_pretty_text_as_valid_json_core_reasoning(self):
        with mock_local_service() as (port,service):
            service.answer="I am certain without evidence!"
            with self.assertRaises(NeuralOutputError):
                NeuralThinkingCore(LoopbackCoreInference(port=port)).think(
                    DeliberationTask("case","p","Is the claim supportable?")
                )


if __name__=="__main__":
    unittest.main()
