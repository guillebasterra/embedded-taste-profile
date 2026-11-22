import json
import os
import subprocess
import sys

import pytest

from taste.serve import tools
from taste.serve.server import build_server


def _texts(result):
    return [c.text for c in result.content if c.type == "text"]


async def test_server_registers_exactly_the_four_tools(built):
    listed = await build_server(built.home).list_tools()
    assert sorted(t.name for t in listed) == sorted(tools.TOOL_NAMES)
    for t in listed:
        assert t.description == tools.TOOL_DESCRIPTIONS[t.name]


def test_descriptions_say_when_to_call():
    d = tools.TOOL_DESCRIPTIONS
    assert "BEFORE" in d["get_taste"] and "BEFORE" in d["get_exemplars"]
    assert "AFTER" in d["critique"] and "screenshot" in d["critique"]
    assert "EVERY time" in d["log_feedback"] and "corrects" in d["log_feedback"]


async def test_input_schemas(built):
    listed = {t.name: t for t in await build_server(built.home).list_tools()}
    schema = lambda name: listed[name].model_dump(by_alias=True)["inputSchema"]  # noqa: E731
    assert schema("get_taste").get("required", []) == []
    assert schema("get_exemplars")["required"] == ["query"]
    assert schema("critique")["required"] == ["image_path"]
    fb = schema("log_feedback")
    assert fb["required"] == ["verdict", "note"]
    assert fb["properties"]["verdict"]["enum"] == ["like", "dislike", "rule"]


async def test_call_get_taste(built):
    server = build_server(built.home)
    assert "Top-level taste summary." in _texts(await server.call_tool("get_taste", {}))[0]
    full = await server.call_tool("get_taste", {"domain": "web", "detail": "full"})
    assert "CARD-MD" in _texts(full)[0]


async def test_call_get_exemplars_returns_image_content(built, scoring):
    result = await build_server(built.home).call_tool("get_exemplars", {"query": "blue"})
    kinds = [c.type for c in result.content]
    assert kinds == ["text", "image", "image"]
    image = result.content[1].model_dump(by_alias=True)
    assert image["mimeType"] == "image/jpeg" and len(image["data"]) > 100


async def test_call_critique_and_log_feedback(built, scoring):
    server = build_server(built.home)
    path = str(built.abs("images/dark-00.jpg"))
    text = _texts(await server.call_tool("critique", {"image_path": path}))[0]
    assert "Dark Brutalist principle A" in text and "Dark Brutalist avoid X" in text

    out = await server.call_tool("log_feedback", {"verdict": "dislike", "note": "too loud",
                                                  "image_path": path, "domain": "web"})
    assert "Logged dislike feedback" in _texts(out)[0]
    (fb,) = built.read_feedback()
    assert fb.note == "too loud" and built.abs(fb.image_path).exists()


async def test_bad_input_reaches_the_agent_as_a_readable_error(built):
    server = build_server(built.home)
    with pytest.raises(Exception, match="note must not be empty"):
        await server.call_tool("log_feedback", {"verdict": "like", "note": " "})
    with pytest.raises(Exception):
        await server.call_tool("log_feedback", {"verdict": "meh", "note": "x"})
    assert built.read_feedback() == []


def test_serve_speaks_only_mcp_on_stdout(built):
    """Real stdio round trip: every stdout line must be a JSON-RPC message."""
    env = {**os.environ, "TASTE_EMBEDDER": "fake"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "taste.cli", "serve", "--home", str(built.home)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    seen = []

    def send(**kw):
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", **kw}) + "\n")
        proc.stdin.flush()

    def reply(msg_id):
        while True:
            line = proc.stdout.readline()
            assert line, "server closed stdout early: " + proc.stderr.read()[-2000:]
            message = json.loads(line)  # anything that is not JSON fails here
            seen.append(message)
            if message.get("id") == msg_id:
                return message

    try:
        send(id=1, method="initialize", params={
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}})
        assert reply(1)["result"]["serverInfo"]["name"] == "taste"
        send(method="notifications/initialized")
        send(id=2, method="tools/list")
        listed = reply(2)["result"]["tools"]
        assert sorted(t["name"] for t in listed) == sorted(tools.TOOL_NAMES)
        send(id=3, method="tools/call", params={"name": "get_taste", "arguments": {}})
        assert "Top-level taste summary." in reply(3)["result"]["content"][0]["text"]
        send(id=4, method="tools/call", params={
            "name": "log_feedback", "arguments": {"verdict": "like", "note": " "}})
        bad = reply(4)["result"]
        assert bad["isError"] and "note must not be empty" in bad["content"][0]["text"]
    finally:
        proc.stdin.close()
        proc.wait(timeout=20)
        rest = proc.stdout.read()
        proc.stdout.close()
        proc.stderr.close()
    assert all(m.get("jsonrpc") == "2.0" for m in seen)
    assert all(json.loads(line) for line in rest.splitlines() if line.strip())
