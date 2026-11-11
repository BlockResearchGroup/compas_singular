"""JSON-RPC 2.0 over stdio for MCP, in the standard library (Rhino runs Python 3.9).

Newline-delimited; stdout is reserved for protocol frames and ``print`` goes to stderr.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import json
import sys
import traceback
from typing import Any
from typing import Callable

__all__ = [
    "PROTOCOL_VERSIONS",
    "PREFERRED_VERSION",
    "IMAGE_KEY",
    "JsonRpcError",
    "Dispatcher",
    "serve",
]


#: Versions this server can speak, newest first. ``initialize`` echoes the
#: client's choice when it is one of these, and otherwise answers with the
#: preferred one and lets the client decide whether to continue.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
PREFERRED_VERSION = PROTOCOL_VERSIONS[0]

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class JsonRpcError(Exception):
    """A fault in the protocol, not in a tool. Becomes an ``error`` object."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        Exception.__init__(self, message)
        self.code = code
        self.message = message
        self.data = data

    def to_object(self) -> dict[str, Any]:
        error = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


#: A tool attaches an image by putting ``{'data': <base64>, 'mimeType': ...}``
#: under this key. It is pulled out before the payload is rendered as text, so
#: a hundred kilobytes of base64 never lands in the JSON a model reads. The
#: underscore marks it as out of band rather than part of the tool's result.
IMAGE_KEY = "_image"


def _text_content(payload: Any) -> list[dict[str, Any]]:
    """A result dict as an MCP text content block of indented JSON."""
    if isinstance(payload, str):
        text = payload
    else:
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    return [{"type": "text", "text": text}]


class Dispatcher(object):
    """Turns a decoded JSON-RPC message into a decoded reply, or ``None``.

    Parameters
    ----------
    handler : object
        Supplies the content. Every method is optional, and the ones present
        decide what ``capabilities`` advertises:

        ``list_tools()`` -> ``[{name, description, inputSchema}, ...]``
        ``call_tool(name, arguments)`` -> ``dict``
        ``list_resources()`` -> ``[{uri, name, ...}, ...]``
        ``read_resource(uri)`` -> ``str`` or ``{text, mimeType}``
        ``list_prompts()`` -> ``[{name, description, arguments}, ...]``
        ``get_prompt(name, arguments)`` -> ``str`` or ``{description, messages}``
    name : str, optional
    version : str, optional
    instructions : str, optional
        Shown to the client once, at ``initialize``. Say what the server is for.
    """

    def __init__(self, handler: Any, name: str = "compas_singular", version: str = "0.1.0", instructions: str | None = None) -> None:
        self.handler = handler
        self.name = name
        self.version = version
        self.instructions = instructions
        self.initialized = False
        self.client_version: str | None = None

    # --------------------------------------------------------------------
    # what we can do
    # --------------------------------------------------------------------

    def capabilities(self) -> dict[str, Any]:
        """Advertise only what the handler actually implements."""
        capabilities = {}
        if hasattr(self.handler, "list_tools"):
            capabilities["tools"] = {}
        if hasattr(self.handler, "list_resources"):
            capabilities["resources"] = {}
        if hasattr(self.handler, "list_prompts"):
            capabilities["prompts"] = {}
        return capabilities

    # --------------------------------------------------------------------
    # the methods
    # --------------------------------------------------------------------

    def _initialize(self, params: dict[str, Any] | None) -> dict[str, Any]:
        wanted = (params or {}).get("protocolVersion")
        self.client_version = wanted
        version = wanted if wanted in PROTOCOL_VERSIONS else PREFERRED_VERSION
        result = {
            "protocolVersion": version,
            "capabilities": self.capabilities(),
            "serverInfo": {"name": self.name, "version": self.version},
        }
        if self.instructions:
            result["instructions"] = self.instructions
        return result

    def _tools_list(self, params: dict[str, Any] | None) -> dict[str, Any]:
        return {"tools": list(self.handler.list_tools())}

    def _tools_call(self, params: dict[str, Any] | None) -> dict[str, Any]:
        params = params or {}
        name = params.get("name")
        if not name:
            raise JsonRpcError(INVALID_PARAMS, "tools/call needs a 'name'")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise JsonRpcError(INVALID_PARAMS, "'arguments' must be an object")
        payload = self.handler.call_tool(name, arguments)
        image = None
        if isinstance(payload, dict) and IMAGE_KEY in payload:
            image = payload.pop(IMAGE_KEY)
        failed = isinstance(payload, dict) and payload.get("ok") is False
        content = _text_content(payload)
        if image and image.get("data"):
            content.append({"type": "image", "data": image["data"], "mimeType": image.get("mimeType", "image/png")})
        return {"content": content, "isError": bool(failed)}

    def _resources_list(self, params: dict[str, Any] | None) -> dict[str, Any]:
        return {"resources": list(self.handler.list_resources())}

    def _resources_read(self, params: dict[str, Any] | None) -> dict[str, Any]:
        uri = (params or {}).get("uri")
        if not uri:
            raise JsonRpcError(INVALID_PARAMS, "resources/read needs a 'uri'")
        found = self.handler.read_resource(uri)
        if found is None:
            raise JsonRpcError(INVALID_PARAMS, "no resource at {!r}".format(uri))
        if isinstance(found, str):
            found = {"text": found}
        return {"contents": [{"uri": uri, "mimeType": found.get("mimeType", "text/markdown"), "text": found.get("text", "")}]}

    def _prompts_list(self, params: dict[str, Any] | None) -> dict[str, Any]:
        return {"prompts": list(self.handler.list_prompts())}

    def _prompts_get(self, params: dict[str, Any] | None) -> dict[str, Any]:
        params = params or {}
        name = params.get("name")
        if not name:
            raise JsonRpcError(INVALID_PARAMS, "prompts/get needs a 'name'")
        found = self.handler.get_prompt(name, params.get("arguments") or {})
        if found is None:
            raise JsonRpcError(INVALID_PARAMS, "no prompt named {!r}".format(name))
        if isinstance(found, str):
            found = {"messages": [{"role": "user", "content": {"type": "text", "text": found}}]}
        return found

    def _ping(self, params: dict[str, Any] | None) -> dict[str, Any]:
        return {}

    def methods(self) -> dict[str, Callable[[dict[str, Any] | None], dict[str, Any]]]:
        table = {"initialize": self._initialize, "ping": self._ping}
        if hasattr(self.handler, "list_tools"):
            table["tools/list"] = self._tools_list
            table["tools/call"] = self._tools_call
        if hasattr(self.handler, "list_resources"):
            table["resources/list"] = self._resources_list
            table["resources/read"] = self._resources_read
        if hasattr(self.handler, "list_prompts"):
            table["prompts/list"] = self._prompts_list
            table["prompts/get"] = self._prompts_get
        return table

    # --------------------------------------------------------------------
    # dispatch
    # --------------------------------------------------------------------

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """One decoded message in, one decoded reply out (or ``None``).

        Parameters
        ----------
        message : dict

        Returns
        -------
        dict or None
            ``None`` for a notification, which must not be answered.
        """
        if not isinstance(message, dict):
            return _error_reply(None, INVALID_REQUEST, "message is not an object")

        method = message.get("method")
        message_id = message.get("id")
        is_notification = "id" not in message

        if not method:
            if is_notification:
                return None
            return _error_reply(message_id, INVALID_REQUEST, "no 'method'")

        # Notifications carry state changes we acknowledge by acting, not by
        # replying. ``notifications/initialized`` is the handshake completing.
        if is_notification:
            if method == "notifications/initialized":
                self.initialized = True
            return None

        function = self.methods().get(method)
        if function is None:
            return _error_reply(message_id, METHOD_NOT_FOUND, "unknown method {!r}".format(method))
        try:
            return {"jsonrpc": "2.0", "id": message_id, "result": function(message.get("params"))}
        except JsonRpcError as exc:
            return {"jsonrpc": "2.0", "id": message_id, "error": exc.to_object()}
        except Exception as exc:
            # An unexpected fault in a handler is this server's bug, not the
            # client's. Report it as an internal error WITH the traceback on
            # stderr, so the session survives and the cause is recoverable.
            traceback.print_exc(file=sys.stderr)
            return _error_reply(message_id, INTERNAL_ERROR, "{}: {}".format(type(exc).__name__, exc))


def _error_reply(message_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    error = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": message_id, "error": error}


# ==============================================================================
# the stdio loop
# ==============================================================================


def serve(dispatcher: Dispatcher, stdin: Any = None, stdout: Any = None, guard: bool = True) -> None:
    """Read newline-delimited JSON from stdin and write replies to stdout until stdin closes.

    Rebinds ``sys.stdout`` to stderr for the duration unless ``guard`` is off.

    Parameters
    ----------
    dispatcher : Dispatcher
    stdin : file, optional
        Defaults to ``sys.stdin``. The tests pass a ``StringIO``.
    stdout : file, optional
        Defaults to ``sys.stdout`` as it was when ``serve`` was called.
    guard : bool, optional
        Rebind ``sys.stdout`` to stderr for the duration. On by default and
        independent of ``stdout``, so the tests exercise the same guard
        production runs under rather than a path only they take.
    """
    source = stdin if stdin is not None else sys.stdin
    sink = stdout if stdout is not None else sys.stdout

    saved = sys.stdout
    if guard:
        sys.stdout = sys.stderr
    try:
        for line in source:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError as exc:
                _write(sink, _error_reply(None, PARSE_ERROR, "invalid JSON: {}".format(exc)))
                continue
            # A batch is a list. Answer each member, and send nothing at all if
            # every one of them was a notification.
            if isinstance(message, list):
                replies = [r for r in (dispatcher.handle(m) for m in message) if r is not None]
                for reply in replies:
                    _write(sink, reply)
                continue
            reply = dispatcher.handle(message)
            if reply is not None:
                _write(sink, reply)
    finally:
        if guard:
            sys.stdout = saved


def _write(sink: Any, payload: Any) -> None:
    """One message, one line, flushed.

    Flushing every message is not optional: the client is blocked on our reply,
    and a buffered stdout would deadlock the session until the buffer filled.
    """
    sink.write(json.dumps(payload, default=str))
    sink.write("\n")
    sink.flush()
