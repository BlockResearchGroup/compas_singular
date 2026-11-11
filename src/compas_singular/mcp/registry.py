"""The MCP tool table: one decorator carrying each tool's schema and annotations, one dispatch point.

A refusal is a ``{'ok': False, 'reason': ...}`` return value, never an exception.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

if TYPE_CHECKING:
    from compas_singular.mcp.session import MeshSession


__all__ = ["ToolSpec", "TOOLS", "tool", "tool_names", "schemas", "call"]


class ToolSpec(object):
    """One callable tool: the function, what it is for, and its arguments."""

    def __init__(
        self,
        name: str,
        function: Callable[..., dict[str, Any]],
        description: str,
        schema: dict[str, Any],
        read_only: bool = False,
        destructive: bool = False,
        idempotent: bool = False,
        open_world: bool = False,
        title: str | None = None,
    ) -> None:
        self.name = name
        self.function = function
        self.description = description
        self.schema = schema
        self.read_only = read_only
        #: Removes or overwrites something a person would miss.
        self.destructive = destructive
        self.idempotent = idempotent
        #: Reaches beyond this process -- the Rhino document, or the library.
        self.open_world = open_world
        self.title = title or name

    def annotations(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "readOnlyHint": self.read_only,
            "destructiveHint": self.destructive,
            "idempotentHint": self.idempotent,
            "openWorldHint": self.open_world,
        }

    def __repr__(self) -> str:
        return "<ToolSpec {}>".format(self.name)


TOOLS: dict[str, ToolSpec] = {}


def tool(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: Sequence[str] = (),
    read_only: bool = False,
    destructive: bool = False,
    idempotent: bool = False,
    open_world: bool = False,
    title: str | None = None,
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Register a tool. ``properties`` is the JSON-schema body of its arguments."""

    def wrap(function: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        schema = {
            "type": "object",
            "properties": dict(properties or {}),
            "required": list(required),
            "additionalProperties": False,
        }
        TOOLS[name] = ToolSpec(name, function, description, schema, read_only=read_only, destructive=destructive, idempotent=idempotent, open_world=open_world, title=title)
        return function

    return wrap


def tool_names() -> list[str]:
    return sorted(TOOLS)


def schemas() -> list[dict[str, Any]]:
    """``[{name, description, inputSchema, annotations}, ...]`` for ``tools/list``.

    Note ``inputSchema``, camelCase -- MCP's spelling, which is not the
    ``input_schema`` of Anthropic's own tool API.
    """
    return [{"name": spec.name, "description": spec.description, "inputSchema": spec.schema, "annotations": spec.annotations()} for spec in (TOOLS[name] for name in tool_names())]


def call(session: MeshSession, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a tool by name. An unknown name or bad arguments is a refusal.

    Parameters
    ----------
    session : MeshSession
    name : str
    arguments : dict, optional

    Returns
    -------
    dict
    """
    spec = TOOLS.get(name)
    if spec is None:
        return {"ok": False, "reason": "no tool named {!r}".format(name), "available": tool_names()}
    arguments = dict(arguments or {})
    try:
        return spec.function(session, **arguments)
    except TypeError as exc:
        # Python names the IMPLEMENTATION in the message, which is not a name
        # the caller has ever seen. Swap the tool's own name in so the
        # correction is obvious, and list what it does accept.
        detail = str(exc).replace(spec.function.__name__, name)
        accepted = sorted(spec.schema["properties"])
        return {"ok": False, "reason": "bad arguments for {}: {}".format(name, detail), "accepted": accepted}
