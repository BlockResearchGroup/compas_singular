"""MCP tools that save a finished session and recall ones with a similar problem shape.

Bad examples are kept and returned too.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas_singular.mcp import library
from compas_singular.mcp.registry import tool

if TYPE_CHECKING:
    from compas_singular.mcp.session import MeshSession


__all__ = []

VERDICTS = ("good", "acceptable", "bad")


def _fingerprint_of(session: MeshSession) -> dict[str, Any]:
    metrics = session.quality() or {}
    return library.fingerprint(metrics, walls=len(session.walls), faces=metrics.get("faces"))


@tool(
    "recall_examples",
    "Find sessions already worked on meshes shaped like this one, ranked by "
    "how alike the PROBLEM is -- boundary loop count, size band, and which "
    "measure was worst -- not by wording. Call this BEFORE deciding what to do, "
    "not after. Returns summaries with a uri; read the full write-up through "
    'resources/read on that uri. Includes examples marked "bad" on purpose: '
    "they say what not to repeat.",
    properties={
        "limit": {"type": "integer", "description": "How many to return. Default 3."},
        "verdicts": {"type": "array", "items": {"type": "string"}, "description": "Which verdicts to include. Default all three: good, acceptable, bad."},
    },
    read_only=True,
    idempotent=True,
    title="Recall similar sessions",
)
def _t_recall_examples(session: MeshSession, limit: int = 3, verdicts: Sequence[str] | None = None) -> dict[str, Any]:
    wanted = tuple(verdicts) if verdicts else VERDICTS
    bad = [v for v in wanted if v not in VERDICTS]
    if bad:
        return {"ok": False, "reason": "unknown verdict(s) {}; use good, acceptable or bad".format(", ".join(repr(v) for v in bad))}
    target = _fingerprint_of(session) if session.loaded else {}
    found = library.recall(target, limit=limit, verdicts=wanted)
    if not found:
        return {
            "ok": True,
            "fingerprint": target,
            "examples": [],
            "reading": "The corpus is empty -- nothing has been saved yet. Save this session with save_example when it is done, and the next one will have something to read.",
        }
    return {"ok": True, "fingerprint": target, "examples": found, "reading": "Closest first. Read the full write-up with resources/read on the uri."}


@tool(
    "save_example",
    "Write this session into the corpus as a worked example, so later sessions "
    "can read it. Records the starting and finishing quality, every step with "
    "its arguments, the remarks, and your verdict. Save BAD runs too -- a run "
    "that went wrong is evidence, and recall returns it deliberately. Be honest "
    'in the verdict: "good" means worth copying, "acceptable" means it worked '
    'but there was a better route, "bad" means do not repeat this. Writes a '
    "file into the library on disk.",
    properties={
        "name": {
            "type": "string",
            "description": "Short name, slugged into a filename. An existing name is not overwritten -- a suffix is added, because the older run is evidence too.",
        },
        "verdict": {"type": "string", "enum": list(VERDICTS), "description": "good, acceptable or bad. See guidance://remarks."},
        "lesson": {"type": "string", "description": "The one thing a later session should take from this. The most useful field in the record."},
        "instruction": {"type": "string", "description": "What you were asked to do."},
    },
    required=("name", "verdict"),
    open_world=True,
    title="Save a worked example",
)
def _t_save_example(session: MeshSession, name: str, verdict: str, lesson: str | None = None, instruction: str | None = None) -> dict[str, Any]:
    if verdict not in VERDICTS:
        return {"ok": False, "reason": "verdict must be one of good, acceptable, bad; got {!r}".format(verdict)}
    if not session.history:
        return {"ok": False, "reason": "nothing has been done in this session, so there is no example to save"}

    steps = []
    for entry in session.history:
        skip = ("step", "action", "at", "before", "after", "improved", "all_improved", "layer", "undone")
        steps.append(
            {
                "action": entry["action"],
                "arguments": dict((k, v) for k, v in entry.items() if k not in skip),
                "all_improved": entry.get("all_improved"),
                # Kept, not dropped: a rejected attempt is the evidence recall is
                # meant to return. A reader filters on this if it wants the route.
                "undone": bool(entry.get("undone")),
                "note": "; ".join(session.remarks_for(entry["step"])) or None,
            }
        )

    first = next((e["after"] for e in session.history if e.get("after")), None)
    record = {
        # ``name`` is slugged into the filename by save_example; ``title`` keeps
        # what was actually typed, which is what a reader recognises.
        "name": name,
        "title": name,
        "verdict": verdict,
        "instruction": instruction,
        "lesson": lesson,
        "source": session.source,
        "before": session.history[0].get("before") or first,
        "after": session.quality(),
        "fingerprint": _fingerprint_of(session),
        "steps": steps,
        "remarks": [r["text"] for r in session.remarks],
    }
    try:
        path = library.save_example(record)
    except OSError as exc:
        return {"ok": False, "reason": "could not write into the library at {}: {}".format(library.library_root(), exc)}
    saved = library._slug(name)
    return {
        "ok": True,
        "path": path,
        "steps": len(steps),
        "remarks": len(record["remarks"]),
        "verdict": verdict,
        "uri": library.EXAMPLE_SCHEME + saved,
        "reading": "Saved. Later sessions on a mesh shaped like this one will find it through recall_examples.",
    }
