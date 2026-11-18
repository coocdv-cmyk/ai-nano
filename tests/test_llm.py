"""The LLM tier against a stub OpenAI-compatible server (no real model needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from omega_nano.engine import Engine
from omega_nano.model import TaskStatus

PLAN = {
    "understanding": {"objective": "greet", "constraints": []},
    "steps": [
        {"tool": "math.eval", "args": {"expr": "40+2"}, "expect": [{"kind": "output_equals", "field": "value", "value": "42"}]},
        {"tool": "fs.write", "args": {"path": "answer.txt", "content": "The answer is ${s0.value}"},
         "expect": [{"kind": "file_sha256", "path": "answer.txt", "sha256": "@content_sha256"}]},
    ],
}


class Stub(BaseHTTPRequestHandler):
    reply = PLAN

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["messages"][0]["role"] == "system" and "fs.write" in body["messages"][0]["content"]
        data = json.dumps({"choices": [{"message": {"content": "Here:\n" + json.dumps(Stub.reply)}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_llm_tier_plans_open_ended_goal(cfg, server):
    cfg.llm_url = server
    eng = Engine(cfg)
    t = eng.submit("figure out the answer to everything and write it down")
    assert t.tier == "t2:llm"
    t = eng.run(t.id)
    assert t.status == TaskStatus.DONE
    assert (cfg.workspace / "answer.txt").read_text(encoding="utf-8") == "The answer is 42"
    eng.close()


def test_llm_plan_with_unknown_tool_is_rejected(cfg, server):
    Stub.reply = {"steps": [{"tool": "os.format_disk", "args": {}}]}
    try:
        cfg.llm_url = server
        eng = Engine(cfg)
        t = eng.submit("do something weird")
        assert t.status == TaskStatus.BLOCKED and "unknown tool" in t.summary
        eng.close()
    finally:
        Stub.reply = PLAN


def test_llm_unreachable_falls_back_to_blocked(cfg):
    cfg.llm_url = "http://127.0.0.1:9"  # nothing listens here
    eng = Engine(cfg)
    eng.llm.timeout_s = 2
    t = eng.submit("open ended goal nobody can parse")
    assert t.status == TaskStatus.BLOCKED and "llm_error" in t.understanding
    eng.close()
