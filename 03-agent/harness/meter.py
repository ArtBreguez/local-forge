"""A metering proxy between an agent (OpenCode) and the forge endpoint (phase 4).

It forwards every request unchanged except for the settings that define config A (they are forced, so the agent cannot
change them): temperature 0, a fixed seed, thinking off (`chat_template_kwargs.enable_thinking=false`), prompt caching on.
It counts what the agent asks for: requests, requests that carry tools (the agent's steps), tool calls the model made, tokens,
errors (a context overflow is an HTTP 400). Phase 5: it also keeps llama-server's own `timings` of every answer (generated tokens and
milliseconds, new prompt tokens and milliseconds, speculative drafts proposed and accepted), so that the speed of an arm is measured on the
server's clock and not on the proxy's. Stdlib only; one thread per connection; streams are passed through line by line.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


class Meter:
    def __init__(self, upstream: str, seed: int = 1, thinking: bool = False):
        u = urllib.parse.urlparse(upstream)
        self.host, self.port, self.base = u.hostname or "127.0.0.1", u.port or 80, u.path.rstrip("/")
        self.seed, self.thinking = seed, thinking
        self.lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def start(self) -> "Meter":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def totals(self) -> dict[str, Any]:
        with self.lock:
            recs = list(self.records)
        pred_n, pred_ms = sum(r.get("pred_n") or 0 for r in recs), sum(r.get("pred_ms") or 0 for r in recs)
        pp_n, pp_ms = sum(r.get("pp_n") or 0 for r in recs), sum(r.get("pp_ms") or 0 for r in recs)
        return {"gen_tok_s": round(1000 * pred_n / pred_ms, 2) if pred_ms else None, "pp_tok_s": round(1000 * pp_n / pp_ms, 1) if pp_ms else None,
                "draft_n": sum(r.get("draft_n") or 0 for r in recs), "draft_accepted": sum(r.get("draft_accepted") or 0 for r in recs),
                "requests": len(recs), "steps": sum(1 for r in recs if r.get("tools")), "tool_calls": sum(r.get("tool_calls", 0) for r in recs),
                "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in recs), "completion_tokens": sum(r.get("completion_tokens") or 0 for r in recs),
                "errors": sum(1 for r in recs if r.get("status", 200) >= 400), "model_seconds": round(sum(r.get("secs", 0) for r in recs), 1),
                "context_overflow": sum(1 for r in recs if r.get("status") == 400 and "context" in (r.get("error") or "").lower())}

    def force(self, body: dict[str, Any]) -> dict[str, Any]:
        body["temperature"] = 0
        body["seed"] = self.seed
        body["cache_prompt"] = True
        kw = dict(body.get("chat_template_kwargs") or {})
        kw["enable_thinking"] = self.thinking
        body["chat_template_kwargs"] = kw
        body.pop("top_p", None)
        body.pop("top_k", None)
        if body.get("stream"):
            body["stream_options"] = {**(body.get("stream_options") or {}), "include_usage": True}
        return body

    def _handler(self):
        meter = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"          # the body ends when the connection closes: no length needed for streams

            def log_message(self, *a: Any) -> None:
                pass

            def _forward(self, method: str) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                rec: dict[str, Any] = {"t": round(time.time(), 1), "path": self.path}
                body: dict[str, Any] | None = None
                if method == "POST" and raw:
                    try:
                        body = meter.force(json.loads(raw))
                        raw = json.dumps(body).encode("utf-8")
                        rec["tools"] = bool(body.get("tools"))
                    except ValueError:
                        body = None
                t0 = time.monotonic()
                conn = HTTPConnection(meter.host, meter.port, timeout=3600)
                headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
                headers["Content-Length"] = str(len(raw))
                path = meter.base + self.path[len("/v1"):] if self.path.startswith("/v1") else self.path
                try:
                    conn.request(method, path, body=raw if raw else None, headers=headers)
                    resp = conn.getresponse()
                    self.send_response(resp.status)
                    for k, v in resp.getheaders():
                        if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                            self.send_header(k, v)
                    self.end_headers()
                    rec["status"] = resp.status
                    buf = b""
                    while True:
                        chunk = resp.read1(8192) if hasattr(resp, "read1") else resp.read(8192)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                        buf += chunk
                        if len(buf) > 4_000_000:
                            buf = buf[-2_000_000:]
                    self._summarize(rec, buf, body)
                except (OSError, ConnectionError) as e:
                    rec.setdefault("status", 599)
                    rec["error"] = f"{type(e).__name__}: {e}"[:200]
                finally:
                    conn.close()
                    rec["secs"] = round(time.monotonic() - t0, 2)
                    with meter.lock:
                        meter.records.append(rec)

            @staticmethod
            def _timings(rec: dict[str, Any], tm: dict[str, Any]) -> None:
                rec["pred_n"], rec["pred_ms"] = tm.get("predicted_n"), tm.get("predicted_ms")
                rec["pp_n"], rec["pp_ms"] = tm.get("prompt_n"), tm.get("prompt_ms")
                rec["draft_n"], rec["draft_accepted"] = tm.get("draft_n"), tm.get("draft_n_accepted")

            def _summarize(self, rec: dict[str, Any], buf: bytes, body: dict[str, Any] | None) -> None:
                text = buf.decode("utf-8", "replace")
                names: set[int] = set()
                if rec.get("status", 200) >= 400:
                    rec["error"] = text[:300]
                    return
                if body and body.get("stream"):
                    for line in text.splitlines():
                        if not line.startswith("data:") or line.strip() == "data: [DONE]":
                            continue
                        try:
                            ev = json.loads(line[5:])
                        except ValueError:
                            continue
                        for ch in ev.get("choices") or []:
                            for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                                if (tc.get("function") or {}).get("name"):
                                    names.add(tc.get("index", 0))
                            if ch.get("finish_reason"):
                                rec["finish"] = ch["finish_reason"]
                        u = ev.get("usage") or {}
                        if u:
                            rec["prompt_tokens"], rec["completion_tokens"] = u.get("prompt_tokens"), u.get("completion_tokens")
                        tm = ev.get("timings") or {}
                        if tm:
                            self._timings(rec, tm)
                        if tm and rec.get("prompt_tokens") is None:
                            rec["prompt_tokens"], rec["completion_tokens"] = tm.get("prompt_n"), tm.get("predicted_n")
                    rec["tool_calls"] = len(names)
                else:
                    try:
                        data = json.loads(text)
                    except ValueError:
                        return
                    u = data.get("usage") or {}
                    rec["prompt_tokens"], rec["completion_tokens"] = u.get("prompt_tokens"), u.get("completion_tokens")
                    if data.get("timings"):
                        self._timings(rec, data["timings"])
                    ch = (data.get("choices") or [{}])[0]
                    rec["finish"] = ch.get("finish_reason")
                    rec["tool_calls"] = len((ch.get("message") or {}).get("tool_calls") or [])

            def do_POST(self) -> None:
                self._forward("POST")

            def do_GET(self) -> None:
                self._forward("GET")

        return H
