"""EXP-015 (N5) release gate: provenance integrity and local API paths."""
from __future__ import annotations

import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "EXPERIMENTS" / "EXP-015-n5-release"
if str(EXP) not in sys.path:
    sys.path.insert(0, str(EXP))

import serve  # noqa: E402


@pytest.fixture(scope="module")
def api():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{srv.server_port}"
    yield base
    srv.shutdown()
    srv.server_close()


def _get(base: str, path: str) -> tuple[int, bytes]:
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return r.status, r.read()


def _post(base: str, path: str, obj: dict) -> dict:
    req = urllib.request.Request(
        base + path, data=json.dumps(obj).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def test_matrix_has_no_unbacked_claim():
    data = json.loads((EXP / "runs/n5-v1/matrix.json").read_text(encoding="utf-8"))
    assert data["integrity"]["clean"] is True
    assert data["integrity"]["resolved"] == data["integrity"]["claims"] >= 19
    for row in data["rows"]:
        assert row["command"] and row["source"]
        assert row["value"] is not None and row["ok"] is True


def test_api_exercises_the_measured_paths(api):
    status, page = _get(api, "/")
    assert status == 200 and b"release matrix" in page
    status, body = _get(api, "/api/matrix")
    assert status == 200 and json.loads(body)["integrity"]["clean"] is True

    enc = _post(api, "/api/encode", {"text": "ذاكرة التخزين"})
    assert enc["dim"] == 512 and 0.99 < enc["l2"] < 1.01

    gen = _post(api, "/api/generate", {"prompt": "def f(x):", "n": 16})
    assert gen["n"] == 16 and len(gen["bytes_hex"]) == 32 and gen["speculative"] is False

    ret = _post(api, "/api/retrieve", {"query": "which port does deployment use"})
    assert ret["hits"] and "8443" in ret["hits"][0]["text"]
