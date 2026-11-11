"""The MCP JSON-RPC layer, driven with a fake handler -- a failure here is a protocol failure."""

import io
import json
import sys

import pytest

from compas_singular.mcp import protocol


class Fake(object):
    def list_tools(self):
        return [{"name": "echo", "description": "echo", "inputSchema": {"type": "object"}}]

    def call_tool(self, name, arguments):
        if name == "boom":
            raise RuntimeError("handler exploded")
        if name == "refuse":
            return {"ok": False, "reason": "not today"}
        return {"ok": True, "got": arguments}

    def list_resources(self):
        return [{"uri": "guidance://quality", "name": "quality"}]

    def read_resource(self, uri):
        return "body of " + uri if uri.startswith("guidance://") else None


class Noisy(Fake):
    def call_tool(self, name, arguments):
        print("THIS PRINT WOULD CORRUPT THE STREAM")
        return {"ok": True}


@pytest.fixture
def dispatcher():
    return protocol.Dispatcher(Fake(), instructions="hello")


def _call(d, rid, method, params=None):
    message = {"jsonrpc": "2.0", "id": rid, "method": method}
    if params is not None:
        message["params"] = params
    return d.handle(message)


def test_initialize_negotiates_and_advertises_honestly(dispatcher):
    out = _call(dispatcher, 1, "initialize", {"protocolVersion": "2024-11-05"})["result"]
    assert out["protocolVersion"] == "2024-11-05"
    assert "name" in out["serverInfo"]
    assert out["instructions"] == "hello"
    assert set(out["capabilities"]) == {"tools", "resources"}
    fallback = _call(dispatcher, 1, "initialize", {"protocolVersion": "1999-01-01"})["result"]
    assert fallback["protocolVersion"] == protocol.PREFERRED_VERSION


def test_a_notification_is_never_answered(dispatcher):
    assert dispatcher.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert dispatcher.initialized is True
    assert dispatcher.handle({"jsonrpc": "2.0", "method": "notifications/whatever"}) is None


def test_a_refusing_tool_is_a_result_not_an_error(dispatcher):
    ok = _call(dispatcher, 2, "tools/call", {"name": "echo", "arguments": {"a": 1}})["result"]
    assert ok["isError"] is False
    assert json.loads(ok["content"][0]["text"])["got"] == {"a": 1}
    ref = _call(dispatcher, 3, "tools/call", {"name": "refuse"})
    assert ref["result"]["isError"] is True
    # still a result, so the model reads the reason
    assert "error" not in ref and "not today" in ref["result"]["content"][0]["text"]


def test_a_crashing_handler_does_not_kill_the_session(dispatcher, capsys):
    boom = _call(dispatcher, 4, "tools/call", {"name": "boom"})
    assert boom["error"]["code"] == protocol.INTERNAL_ERROR
    assert "exploded" in boom["error"]["message"]
    assert "RuntimeError" in capsys.readouterr().err


@pytest.mark.parametrize(
    "method, params, code",
    [
        ("nope", None, protocol.METHOD_NOT_FOUND),
        ("prompts/list", None, protocol.METHOD_NOT_FOUND),  # an unadvertised capability
        ("tools/call", {}, protocol.INVALID_PARAMS),
        ("tools/call", {"name": "echo", "arguments": 5}, protocol.INVALID_PARAMS),
        ("resources/read", {"uri": "nope://x"}, protocol.INVALID_PARAMS),
    ],
)
def test_protocol_faults_are_json_rpc_errors(dispatcher, method, params, code):
    assert _call(dispatcher, 5, method, params)["error"]["code"] == code


def test_a_message_that_is_not_an_object_is_an_invalid_request(dispatcher):
    assert dispatcher.handle("banana")["error"]["code"] == protocol.INVALID_REQUEST


def test_the_stdio_loop_frames_skips_blanks_survives_junk_and_batches():
    lines = [
        json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "",
        "this is not json",
        json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
        json.dumps([{"jsonrpc": "2.0", "method": "notifications/initialized"}]),
        json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
    ]
    sink = io.StringIO()
    protocol.serve(protocol.Dispatcher(Fake()), stdin=io.StringIO("\n".join(lines) + "\n"), stdout=sink, guard=False)
    replies = [json.loads(line) for line in sink.getvalue().splitlines() if line.strip()]
    assert len(replies) == 4
    assert "Content-Length" not in sink.getvalue()
    assert [r.get("id") for r in replies] == [1, None, 2, 3]
    assert replies[1]["error"]["code"] == protocol.PARSE_ERROR
    assert replies[3]["result"] == {}


def test_the_stdout_guard_keeps_a_stray_print_off_the_stream(capsys):
    sink = io.StringIO()
    stdout = sys.stdout
    request = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "chatty"}}
    protocol.serve(protocol.Dispatcher(Noisy()), stdin=io.StringIO(json.dumps(request) + "\n"), stdout=sink)
    assert "CORRUPT" not in sink.getvalue()
    assert all(json.loads(line) for line in sink.getvalue().splitlines() if line.strip())
    assert "CORRUPT" in capsys.readouterr().err
    assert sys.stdout is stdout
