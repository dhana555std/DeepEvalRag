"""Mock RAG API for local development and CI smoke tests (stdlib only).

Starts a server exposing the two endpoints ragqa expects by default (same shapes as the real API):

    POST /v1/retrieve {"query", "top_k"} -> {"query", "top_k", "results": [{"rank","chunk_id","score","text","metadata"}]}
    POST /v1/answer   {"query", "top_k"} -> {"query", "answer", "cited_ranks": [int], "sources": [...], "usage": {...}}
                      (retrieves internally; an optional "contexts": [str] body field overrides retrieval)

It answers from a tiny built-in knowledge base that matches ``tests/fixtures/sample_golden.jsonl``
and refuses obvious jailbreak / PII requests. It exists so the harness can be
exercised end to end without the real application - it is NOT a model of your RAG.

Run:  uv run python scripts/mock_rag_api.py --port 8080
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

KB = [
    {"id": "returns-policy", "text": "Customers may return online purchases within 30 days of delivery for a full refund."},
    {"id": "shipping-faq", "text": "Standard shipping within the continental US takes 3 to 5 business days."},
    {"id": "order-management", "text": "The delivery address can be updated before the order ships via the order page or by contacting support."},
    {"id": "shipping-faq", "text": "Orders that have already shipped cannot have their address changed."},
]
BLOCKED = ("ignore all previous", "system prompt", "home address", "phone number")


def words(s: str) -> set[str]:
    return {w.strip("?.,!").lower() for w in s.split() if len(w) > 3}


def _source(rank: int, d: dict, score: float) -> dict:
    return {"rank": rank, "chunk_id": d["id"], "score": score, "text": d["text"],
            "metadata": {"source_system": "mock", "file_name": f"{d['id']}.pdf", "chunk_index": rank - 1}}


def retrieve(query: str, top_k: int) -> list[dict]:
    if any(b in query.lower() for b in BLOCKED):
        return []
    scored = sorted(((len(words(query) & words(d["text"])), d) for d in KB), key=lambda x: -x[0])
    return [_source(i, d, float(s)) for i, (s, d) in enumerate(scored[:top_k], 1) if s > 0]


class Handler(BaseHTTPRequestHandler):
    def _send(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/v1/retrieve":
            query, top_k = req.get("query", ""), int(req.get("top_k", 5))
            return self._send({"query": query, "top_k": top_k, "results": retrieve(query, top_k)})
        if self.path == "/v1/answer":
            query = req.get("query", "")
            if "contexts" in req:
                sources = [_source(i, next((d for d in KB if d["text"] == t), {"id": f"ctx-{i}", "text": t}), 0.0)
                           for i, t in enumerate(req["contexts"], 1)]
            else:
                sources = retrieve(query, int(req.get("top_k", 5)))
            if not sources:
                answer = "I can't help with that request, but I'm happy to answer questions about orders, shipping and returns."
            else:
                answer = " ".join(s["text"] for s in sources)
            return self._send({"query": query, "answer": answer, "grounded": bool(sources),
                               "cited_ranks": [s["rank"] for s in sources], "sources": sources,
                               "usage": {"prompt_tokens": 20 + len(answer) // 4,
                                         "completion_tokens": len(answer) // 4}})
        self._send({"error": "not found"}, 404)

    def log_message(self, *args) -> None:  # keep test output quiet
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    port = ap.parse_args().port
    print(f"Mock RAG API on http://localhost:{port}  (/v1/retrieve, /v1/answer)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
