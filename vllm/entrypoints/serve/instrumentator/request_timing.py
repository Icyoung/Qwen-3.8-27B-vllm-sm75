# SPDX-License-Identifier: Apache-2.0
"""Passive OpenAI SSE timing, with the pinned llm_speedtest client formula.

Prefill = exact prompt usage / time to first nonempty content or reasoning.
Decode = ALL exact completion usage / (stream end - first content).
The first speculative batch is not subtracted. Usage/DONE trailers belong to
this decode window. This is API-server transport timing, not client network
latency or engine-only phase time. No text, keys, prompts or request IDs persist.
"""
from __future__ import annotations

import codecs
import json
import math
import time

CONTRACT = "llm-speedtest-client-v1"


def _tokens(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


class StreamTiming:
    def __init__(self, started):
        self.started = started
        self.updated = started
        self.first_json = None
        self.first_content = None
        self.last_content = None
        self.prompt_tokens = None
        self.output_tokens = None
        self.first_emission_tokens = None
        self.done = False
        self.finished_choice = False
        self.invalid = False
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = ""
        self.data = []

    def event(self, data, now):
        if data.strip() == "[DONE]":
            self.done = True
            return
        try:
            item = json.loads(data)
        except (TypeError, ValueError):
            self.invalid = True
            return
        if not isinstance(item, dict) or item.get("error"):
            self.invalid = True
            return
        if self.first_json is None:
            self.first_json = now
        choices = item.get("choices") or []
        if not isinstance(choices, list) or len(choices) > 1:
            self.invalid = True
            return
        choice = choices[0] if choices else {}
        if not isinstance(choice, dict) or choice.get("index", 0) != 0:
            self.invalid = True
            return
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        if not isinstance(delta, dict) or not isinstance(message, dict):
            self.invalid = True
            return
        content = (delta.get("reasoning_content") or delta.get("reasoning") or
                   delta.get("content") or message.get("content") or choice.get("text"))
        first_content_event = bool(content) and self.first_content is None
        if content:
            if self.first_content is None:
                self.first_content = now
                ids = choice.get("token_ids")
                if isinstance(ids, list) and ids:
                    self.first_emission_tokens = len(ids)
            self.last_content = now
        self.finished_choice |= choice.get("finish_reason") is not None
        self.usage(item.get("usage"), first_content_event)

    def usage(self, usage, first_content_event=False):
        if not isinstance(usage, dict):
            return
        prompt = _tokens(usage.get("prompt_tokens"))
        output = _tokens(usage.get("completion_tokens"))
        # vLLM completion_tokens already includes reasoning tokens.
        if prompt is not None:
            if self.prompt_tokens is not None and prompt != self.prompt_tokens:
                self.invalid = True
            self.prompt_tokens = prompt
        if output is not None:
            if self.output_tokens is not None and output < self.output_tokens:
                self.invalid = True
            self.output_tokens = output
            if first_content_event and self.first_emission_tokens is None and output > 0:
                self.first_emission_tokens = output

    def feed(self, chunk, now, final=False):
        self.updated = now
        if self.invalid:
            return
        try:
            self.buffer += self.decoder.decode(chunk, final=final)
        except UnicodeDecodeError:
            self.invalid = True
            return
        # Bound only unframed data. Large complete chunks are consumed below.
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                if self.data:
                    self.event("\n".join(self.data), now)
                    self.data.clear()
            elif line.startswith("data:"):
                self.data.append(line[5:].lstrip(" "))
            if len(self.buffer) + sum(map(len, self.data)) > 4 * 1024 * 1024:
                self.invalid = True
                return
        if len(self.buffer) + sum(map(len, self.data)) > 4 * 1024 * 1024:
            self.invalid = True
        if final:
            if self.buffer.startswith("data:"):
                self.data.append(self.buffer[5:].lstrip(" "))
            if self.data:
                self.event("\n".join(self.data), now)
            self.buffer = ""
            self.data.clear()

    def snapshot(self, now, complete=False):
        # A role/control-only frame cannot be called TTFT while a real content
        # event may still arrive. The benchmark's first-JSON fallback is final-only.
        boundary = self.first_content if self.first_content is not None else self.first_json if complete else None
        valid = not self.invalid and boundary is not None and now >= boundary >= self.started
        if complete and not (self.done or self.finished_choice):
            valid = False
        prefill_seconds = max(boundary - self.started, .001) if valid else None
        decode_seconds = max(now - boundary, .001) if valid else None
        return {
            "contract": CONTRACT,
            "observedAt": now,
            "prefill": self.prompt_tokens / prefill_seconds if valid and self.prompt_tokens is not None and self.prompt_tokens > 0 else None,
            "decode": self.output_tokens / decode_seconds if valid and self.output_tokens is not None and self.output_tokens > 0 else None,
            "promptTokens": self.prompt_tokens,
            "outputTokens": self.output_tokens,
            "firstEmissionTokens": self.first_emission_tokens,
            "ttftSeconds": prefill_seconds,
            "decodeSeconds": decode_seconds,
            "boundary": "first_content_token" if self.first_content is not None else "first_chunk_fallback" if boundary is not None else None,
            "complete": complete and valid,
        }


class RequestTimingStore:
    """Select newest started active stream, else newest successful completion.

    A new request invalidates an older completed sample. Failed/cancelled requests
    never restore that old sample. An active older request may resume visibility
    when the newer concurrent request finishes. Gauge timestamps enforce expiry
    at the consumer; no prompt/response data enters metrics.
    """
    def __init__(self, publish):
        self.publish = publish
        self.active = {}
        self.sequence = 0
        self.last = None

    def start(self, now):
        self.sequence += 1
        key = self.sequence
        self.active[key] = StreamTiming(now)
        self.last = None
        self.emit(now)
        return key

    def emit(self, now):
        if self.active:
            meter = self.active[max(self.active)]
            self.publish(meter.snapshot(meter.updated), True)
        elif self.last is not None:
            self.publish(self.last, False)
        else:
            self.publish({"prefill": None, "decode": None, "observedAt": now}, False)

    def feed(self, key, body, now, final=False, usage=None):
        meter = self.active[key]
        meter.feed(body, now, final=final)
        if usage is not None:
            meter.usage(usage)
        if final:
            result = meter.snapshot(now, complete=True)
            self.last = result if result["complete"] else None
            del self.active[key]
        self.emit(now)

    def discard(self, key, now):
        self.active.pop(key, None)
        self.last = None
        self.emit(now)


class RequestTimingMiddleware:
    def __init__(self, app, store, clock=time.perf_counter):
        self.app, self.store, self.clock = app, store, clock

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "POST" or scope.get("path") not in ("/v1/chat/completions", "/v1/completions"):
            return await self.app(scope, receive, send)
        key = self.store.start(self.clock())
        streaming = False
        ended = False
        disconnected = False

        async def observed_receive():
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def observed_send(message):
            nonlocal streaming, ended
            if message["type"] == "http.response.start":
                headers = {k.lower(): v.lower() for k, v in message.get("headers", [])}
                streaming = message.get("status") == 200 and headers.get(b"content-type", b"").startswith(b"text/event-stream")
            await send(message)
            if message["type"] == "http.response.body":
                final = not message.get("more_body", False)
                if streaming and not disconnected:
                    metadata = scope.get("state", {}).get("request_metadata")
                    # The reviewed generators publish their existing exact counters
                    # before yielding each SSE event, even without include_usage.
                    usage = getattr(metadata, "_sm75_stream_usage", None)
                    if final:
                        final_usage = getattr(metadata, "final_usage_info", None)
                        if final_usage is not None:
                            usage = final_usage.model_dump() if hasattr(final_usage, "model_dump") else final_usage
                    self.store.feed(key, message.get("body", b""), self.clock(), final=final, usage=usage)
                    ended = final
                elif final:
                    self.store.discard(key, self.clock())
                    ended = True

        try:
            return await self.app(scope, observed_receive, observed_send)
        finally:
            if not ended or disconnected:
                self.store.discard(key, self.clock())


def attach_request_timing(app, registry):
    """Use the same engine middleware for Standard, Ultra, DSH and external APIs."""
    if getattr(app.state, "sm75_request_timing", None) is not None:
        return
    from prometheus_client import Gauge

    gauges = {name: Gauge("vllm:sm75_request_" + name, help_text, registry=registry,
                          multiprocess_mode="liveall") for name, help_text in {
        "prefill_tokens_per_second": "Latest request exact prompt tokens / first-content latency (llm_speedtest formula)",
        "decode_tokens_per_second": "Latest request all exact output tokens / first-content-to-stream-end time (llm_speedtest formula)",
        "timing_updated_seconds": "Unix timestamp of the latest observed request timing update",
        "timing_active": "1 while the displayed request is active, 0 when completed/unavailable",
    }.items()}

    def publish(sample, active):
        for metric, key in (("prefill_tokens_per_second", "prefill"), ("decode_tokens_per_second", "decode")):
            value = sample.get(key)
            gauges[metric].set(value if value is not None and math.isfinite(value) else float("nan"))
        gauges["timing_active"].set(int(active))
        observed = sample.get("observedAt", time.perf_counter())
        gauges["timing_updated_seconds"].set(time.time() - max(0, time.perf_counter() - observed))

    store = RequestTimingStore(publish)
    app.state.sm75_request_timing = store
    publish({}, False)
    app.add_middleware(RequestTimingMiddleware, store=store)
