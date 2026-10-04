"""CPU-only tests exercise real ASGI bytes, speculative batches and trailers."""
import asyncio
import importlib.util
import json
from pathlib import Path
import unittest
from types import SimpleNamespace

SOURCE = Path(__file__).resolve().parents[4] / "vllm/entrypoints/serve/instrumentator/request_timing.py"
spec = importlib.util.spec_from_file_location("request_timing", SOURCE)
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def event(content=None, *, tokens=None, usage=None, reasoning=False, finish=None):
    item = {"choices": []}
    if content is not None or finish is not None:
        choice = {"index": 0, "delta": {"reasoning" if reasoning else "content": content}, "finish_reason": finish}
        if tokens is not None:
            choice["token_ids"] = list(range(tokens))
        item["choices"] = [choice]
    if usage is not None:
        item["usage"] = {"prompt_tokens": 100, "completion_tokens": usage,
                         "completion_tokens_details": {"reasoning_tokens": 4}}
    return ("data: " + json.dumps(item, ensure_ascii=False) + "\r\n\r\n").encode()


class RequestTimingTest(unittest.TestCase):
    def sample(self, batch):
        meter = timing.StreamTiming(0)
        meter.feed(event("", usage=0), .5)
        meter.feed(event("思考", tokens=batch, reasoning=True), 2)
        meter.feed(event("answer", tokens=10-batch, finish="stop"), 5)
        meter.feed(event(usage=10), 11)
        meter.feed(b"data: [DONE]\n\n", 12, final=True)
        return meter.snapshot(12, complete=True)

    def test_real_llm_test_formula_includes_first_speculative_batch_and_trailer(self):
        for first_batch in [1, 4, 10]:
            result = self.sample(first_batch)
            self.assertEqual(result["prefill"], 50)
            self.assertEqual(result["decode"], 1)
            self.assertEqual(result["decodeSeconds"], 10)
            self.assertEqual(result["firstEmissionTokens"], first_batch)
            self.assertEqual(result["outputTokens"], 10)
            self.assertTrue(result["complete"])

    def test_utf8_and_sse_frame_across_transport_chunks(self):
        meter = timing.StreamTiming(0)
        data = event("中", tokens=4, usage=4, reasoning=True)
        for byte in data[:-1]:
            meter.feed(bytes([byte]), 1)
        meter.feed(data[-1:], 2)
        meter.feed(event("done", usage=10, finish="stop") + b"data: [DONE]\n\n", 5, final=True)
        result = meter.snapshot(5, complete=True)
        self.assertEqual(result["ttftSeconds"], 2)
        self.assertAlmostEqual(result["decode"], 10/3)
        self.assertEqual(result["firstEmissionTokens"], 4)

    def test_no_usage_does_not_estimate_from_text_or_chunk_count(self):
        meter = timing.StreamTiming(0)
        meter.feed(event("a long text"), 1)
        meter.feed(event("more text", finish="stop") + b"data: [DONE]\n\n", 2, final=True)
        result = meter.snapshot(2, complete=True)
        self.assertIsNone(result["prefill"])
        self.assertIsNone(result["decode"])
        self.assertIsNone(result["firstEmissionTokens"])

    def test_usage_trailer_in_same_transport_chunk_does_not_become_first_batch(self):
        meter = timing.StreamTiming(0)
        meter.feed(event("hello") + event(usage=10), 2)
        self.assertIsNone(meter.first_emission_tokens)

    def test_active_role_only_frame_waits_for_content_then_final_fallback_is_allowed(self):
        meter=timing.StreamTiming(0)
        meter.feed(event("",usage=4),1)
        self.assertIsNone(meter.snapshot(1)["prefill"])
        self.assertIsNone(meter.snapshot(1)["decode"])
        meter.feed(event("",usage=4,finish="stop")+b"data: [DONE]\n\n",3,final=True)
        final=meter.snapshot(3,complete=True)
        self.assertEqual(final["prefill"],100)
        self.assertEqual(final["decode"],2)
        self.assertEqual(final["boundary"],"first_chunk_fallback")

    def test_invalid_and_unfinished_streams_do_not_publish_completed_rates(self):
        for suffix in [b"", b'data: {"error":{"message":"failed"}}\n\n',
                       b'data: {invalid}\n\n', b'data: {"choices":[{"index":1}]}\n\n']:
            meter = timing.StreamTiming(0)
            meter.feed(event("hello", usage=10), 1)
            meter.feed(suffix, 2, final=True)
            result = meter.snapshot(2, complete=True)
            self.assertFalse(result["complete"])
            self.assertIsNone(result["decode"])

    def test_newest_active_request_isolated_from_other_concurrent_updates(self):
        updates = []
        store = timing.RequestTimingStore(lambda row, active: updates.append((row, active)))
        older = store.start(0)
        store.feed(older, event("older", usage=5), 1)
        newer = store.start(2)
        store.feed(newer, event("newer", usage=2), 3)
        expected = updates[-1][0]
        store.feed(older, event("older still", usage=8), 4)
        self.assertEqual(updates[-1][0], expected)
        store.discard(newer, 5)
        self.assertEqual(updates[-1][0]["outputTokens"], 8)
        store.discard(older, 6)
        self.assertIsNone(updates[-1][0]["decode"])
        self.assertFalse(updates[-1][1])

    def test_exact_engine_metadata_supports_clients_without_include_usage(self):
        async def run():
            seen, clock = [], [0]
            store = timing.RequestTimingStore(lambda row, active: seen.append(row))
            metadata = SimpleNamespace(final_usage_info=None)
            async def app(scope, receive, send):
                await send({"type":"http.response.start","status":200,"headers":[(b"content-type", b"text/event-stream")]})
                clock[0] = 2
                await send({"type":"http.response.body","body":event("thinking", reasoning=True),"more_body":True})
                metadata.final_usage_info = SimpleNamespace(model_dump=lambda:{"prompt_tokens":100,"completion_tokens":10})
                clock[0] = 12
                await send({"type":"http.response.body","body":event("last",finish="stop")+b"data: [DONE]\n\n","more_body":False})
            async def send(message):
                pass
            wrapper = timing.RequestTimingMiddleware(app, store, clock=lambda:clock[0])
            await wrapper({"type":"http","method":"POST","path":"/v1/chat/completions","state":{"request_metadata":metadata}},None,send)
            self.assertEqual(seen[-1]["prefill"],50)
            self.assertEqual(seen[-1]["decode"],1)
            self.assertTrue(seen[-1]["complete"])
        asyncio.run(run())

    def test_private_exact_counts_are_live_for_plain_dsh_and_external_streams(self):
        async def run(path, fail=False):
            seen, sent, clock = [], [], [0]
            store = timing.RequestTimingStore(lambda row, active: seen.append((row,active)))
            metadata = SimpleNamespace(final_usage_info=None)
            first = event("thinking",reasoning=True)
            second = event("answer")
            async def app(scope, receive, send):
                await send({"type":"http.response.start","status":200,"headers":[(b"content-type",b"text/event-stream")]})
                metadata.__dict__["_sm75_stream_usage"]={"prompt_tokens":100,"completion_tokens":4}
                clock[0]=2
                await send({"type":"http.response.body","body":first,"more_body":True})
                self.assertEqual(seen[-1][0]["prefill"],50)
                self.assertEqual(seen[-1][0]["decode"],4000)
                self.assertTrue(seen[-1][1])
                metadata.__dict__["_sm75_stream_usage"]={"prompt_tokens":100,"completion_tokens":10}
                clock[0]=5
                await send({"type":"http.response.body","body":second,"more_body":True})
                self.assertAlmostEqual(seen[-1][0]["decode"],10/3)
                self.assertTrue(seen[-1][1])
                if fail: raise RuntimeError("cancelled engine")
                metadata.final_usage_info=SimpleNamespace(model_dump=lambda:{"prompt_tokens":100,"completion_tokens":11})
                clock[0]=12
                await send({"type":"http.response.body","body":event("",finish="stop")+b"data: [DONE]\n\n","more_body":False})
            async def send(message): sent.append(message)
            wrapper=timing.RequestTimingMiddleware(app,store,clock=lambda:clock[0])
            try:
                await wrapper({"type":"http","method":"POST","path":path,"state":{"request_metadata":metadata}},None,send)
            except RuntimeError:
                self.assertTrue(fail)
            self.assertEqual(sent[1]["body"],first)
            self.assertEqual(sent[2]["body"],second)
            self.assertNotIn(b'"usage"',sent[1]["body"])
            self.assertFalse(seen[-1][1])
            if fail: self.assertIsNone(seen[-1][0]["decode"])
            else: self.assertAlmostEqual(seen[-1][0]["decode"],1.1)
        for path in ("/v1/chat/completions","/v1/completions"):
            asyncio.run(run(path)); asyncio.run(run(path,True))

    def test_explicit_client_disconnect_invalidates_even_if_app_finishes(self):
        async def run():
            seen, clock = [], [0]
            store = timing.RequestTimingStore(lambda row, active:seen.append(row))
            async def app(scope, receive, send):
                await send({"type":"http.response.start","status":200,"headers":[(b"content-type",b"text/event-stream")]})
                clock[0] = 2
                await send({"type":"http.response.body","body":event("first",usage=4),"more_body":True})
                await receive()
                clock[0] = 5
                await send({"type":"http.response.body","body":event("last",usage=10,finish="stop")+b"data: [DONE]\n\n","more_body":False})
            async def receive():
                return {"type":"http.disconnect"}
            async def send(message):
                pass
            wrapper = timing.RequestTimingMiddleware(app,store,clock=lambda:clock[0])
            await wrapper({"type":"http","method":"POST","path":"/v1/completions"},receive,send)
            self.assertIsNone(seen[-1]["decode"])
            self.assertFalse(store.active)
        asyncio.run(run())

    def test_metric_registration_is_idempotent_on_minimum_prometheus_version(self):
        try:
            from prometheus_client import CollectorRegistry, generate_latest
        except ImportError:
            self.skipTest("prometheus_client required for registration integration")
        registry = CollectorRegistry()
        app = SimpleNamespace(state=SimpleNamespace(), add_middleware=lambda *args,**kwargs:None)
        timing.attach_request_timing(app,registry)
        first = app.state.sm75_request_timing
        timing.attach_request_timing(app,registry)
        self.assertIs(app.state.sm75_request_timing,first)
        metrics = generate_latest(registry).decode()
        self.assertEqual(metrics.count("# HELP vllm:sm75_request_decode_tokens_per_second"),1)
        self.assertIn("vllm:sm75_request_decode_tokens_per_second NaN",metrics)

    def test_asgi_pass_through_and_failure_cleanup(self):
        async def run(fail=False):
            sent, published, clock = [], [], [0]
            store = timing.RequestTimingStore(lambda row, active: published.append((row, active)))
            messages = [
                {"type":"http.response.start", "status":200, "headers":[(b"content-type", b"text/event-stream; charset=utf-8")]},
                {"type":"http.response.body", "body":event("first", usage=4), "more_body":True},
                {"type":"http.response.body", "body":event("last", usage=10, finish="stop")+b"data: [DONE]\n\n", "more_body":False},
            ]
            async def app(scope, receive, send):
                for i, message in enumerate(messages):
                    if fail and i == 2:
                        raise RuntimeError("stream aborted")
                    clock[0] = i + 1
                    await send(message)
            async def send(message):
                sent.append(message)
            wrapper = timing.RequestTimingMiddleware(app, store, clock=lambda:clock[0])
            try:
                await wrapper({"type":"http","method":"POST","path":"/v1/chat/completions"}, None, send)
            except RuntimeError:
                self.assertTrue(fail)
            self.assertEqual(sent, messages[:2] if fail else messages)
            self.assertFalse(store.active)
            self.assertFalse(published[-1][1])
            if fail:
                self.assertIsNone(published[-1][0]["decode"])
            else:
                self.assertEqual(published[-1][0]["decode"], 10)
        asyncio.run(run())
        asyncio.run(run(True))

if __name__ == "__main__":
    unittest.main()
