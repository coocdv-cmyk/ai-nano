"""EXP-015 N5: local UI/API exercising the same code paths the matrix measures.

Endpoints use the real implementations — `encode_text` (N3), `MemoryFabric.recall`
(N0/N3), `Engine.generate` with the shipped N2 default (non-speculative) — not
reimplementations.  Binds to loopback only; accepts text, never file paths.

Usage:
    python EXPERIMENTS/EXP-015-n5-release/serve.py --port 8765
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega_nano.core.engine import Engine  # noqa: E402
from omega_nano.memory import MemoryFabric  # noqa: E402
from omega_nano.semantic import encode_text  # noqa: E402

MATRIX = ROOT / "EXPERIMENTS/EXP-015-n5-release/runs/n5-v1/matrix.json"
CKPT = ROOT / "EXPERIMENTS/EXP-012-n2-mtp/runs/mtp-v1/mtp-self.pt"
DOCS = (
    "The deployment server runs on port 8443",
    "عاصمة مصر هي القاهرة",
    "Cache keys must include both the user and the locale",
    "A verifier accepts a patch only after the tests execute",
)
_ENGINE: Engine | None = None


def engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine.load(CKPT, packed=True)
    return _ENGINE


def generate(prompt: bytes, n: int) -> dict:
    if not 1 <= n <= 4096:
        raise ValueError("n must be in 1..4096")
    eng = engine()
    t0 = time.perf_counter()
    out = eng.generate(prompt, n, temperature=0, seed=0)  # default: non-speculative
    dt = time.perf_counter() - t0
    return {"bytes_hex": out.hex(), "n": len(out), "tok_per_s": round(len(out) / dt, 1),
            "speculative": False}


def retrieve(query: str, k: int = 3) -> dict:
    with tempfile.TemporaryDirectory() as td:
        mem = MemoryFabric(Path(td) / "m.db")
        try:
            for i, text in enumerate(DOCS):
                mem.remember(text, source=f"doc-{i}")
            hits = mem.recall(query, k=k)
            return {"hits": [{"text": h.text, "source": h.source} for h in hits]}
        finally:
            mem.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: object) -> None:
        pass  # keep the local UI quiet

    def _send(self, code: int, payload: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, code: int, obj: object) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        if self.path in ("/", "/index.html"):
            rows = []
            if MATRIX.is_file():
                data = json.loads(MATRIX.read_text(encoding="utf-8"))
                rows = [f"<li>{html.escape(e['track'])} · {html.escape(e['metric'])}: "
                        f"<b>{html.escape(str(e.get('display', 'n/a')))}</b> {html.escape(e['unit'])}</li>"
                        for e in data["rows"]]
            page = ("<!doctype html><meta charset='utf-8'><title>OMEGA-NANO N5</title>"
                    "<h1>OMEGA-NANO release matrix</h1><ul>" + "".join(rows) + "</ul>"
                    "<p>JSON: <a href='/api/matrix'>/api/matrix</a></p>"
                    "<p>POST /api/encode, /api/retrieve, /api/generate with JSON bodies.</p>")
            self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/matrix":
            if not MATRIX.is_file():
                self._json(404, {"error": "matrix.json not built"})
                return
            data = json.loads(MATRIX.read_text(encoding="utf-8"))
            self._json(200, {"integrity": data["integrity"], "rows": data["rows"]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/api/encode", "/api/retrieve", "/api/generate"):
            self._json(404, {"error": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(min(size, 65536)) or b"{}")
            if self.path == "/api/encode":
                text = str(body.get("text", ""))[:4096]
                vec = encode_text(text)
                self._json(200, {"dim": len(vec), "l2": round(sum(v * v for v in vec) ** 0.5, 6),
                                 "first8": vec[:8]})
            elif self.path == "/api/retrieve":
                self._json(200, retrieve(str(body.get("query", ""))[:1024]))
            else:
                prompt = str(body.get("prompt", "def f(x):"))[:1024].encode("utf-8")
                self._json(200, generate(prompt, int(body.get("n", 32))))
        except (ValueError, KeyError, OSError) as exc:
            self._json(400, {"error": str(exc)})


def main() -> None:
    ap = argparse.ArgumentParser(description="Local N5 release UI/API")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"serving on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
