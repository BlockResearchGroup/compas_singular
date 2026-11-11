"""MCP tools that change the dense mesh's topology by adding and removing a line (a strip).

A line whose strip reaches a non-quad face is refused; undo keeps a whole-mesh copy.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas_singular.editing.denseeditor import DenseMeshEditor
from compas_singular.mcp.bridge import wire
from compas_singular.mcp.describe import describe
from compas_singular.mcp.handle import resolve
from compas_singular.mcp.library import thresholds
from compas_singular.mcp.registry import tool

if TYPE_CHECKING:
    from compas.datastructures import Mesh
    from compas_singular.mcp.session import MeshSession


__all__ = []


_EDGE = {
    "type": "array",
    "items": {"type": "string"},
    "minItems": 2,
    "maxItems": 2,
    "description": "Two vertex handles naming ONE edge of the dense mesh -- neighbours, e.g. from inspect's worst_face_at and a vertex next to it. Any edge of the line works.",
}


def _refuse(reason: str, **extra: Any) -> dict[str, Any]:
    out = {"ok": False, "reason": reason}
    out.update(extra)
    return out


def _editor(session: MeshSession) -> tuple[DenseMeshEditor | None, dict[str, Any] | None]:
    """``(editor, None)`` on the dense mesh in hand, or ``(None, refusal)``; a plain Mesh is rebuilt as a QuadMesh."""
    if not session.loaded:
        return None, _refuse("no dense mesh is loaded -- coarse_densify, rhino_pull or load_mesh first")
    mesh = session.mesh
    if not hasattr(mesh, "collect_polyedges"):
        from compas_singular.datastructures import QuadMesh

        mesh = wire.mesh_from_wire(wire.mesh_to_wire(mesh), cls=QuadMesh)
        session.mesh = mesh
    walls = [[list(p) for p in getattr(w, "points", w)] for w in session.walls]
    return DenseMeshEditor(mesh, walls=walls or None), None


def _edge(mesh: Mesh, handles: Sequence[str]) -> tuple[tuple[Any, Any] | None, str | None]:
    """``((u, v), None)`` for two handles that are neighbours, else ``(None, reason)``."""
    if len(handles) != 2:
        return None, "an edge needs exactly two vertex handles"
    keys = []
    for item in handles:
        key, how = resolve(mesh, item)
        if key is None:
            return None, "could not place {!r} on the dense mesh: {}".format(item, how)
        keys.append(key)
    u, v = keys
    if v not in mesh.vertex_neighbors(u):
        return None, ("{} and {} are both on the mesh but are not joined by an edge -- name two NEIGHBOURING vertices".format(*handles))
    return (u, v), None


def _reading(session: MeshSession) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    metrics = session.quality()
    told = describe(session.mesh, metrics, thresholds())
    return metrics, told


def _note_layout(session: MeshSession) -> str | None:
    if session.coarse is None:
        return None
    return (
        "this edit is on the DENSE mesh only. coarse_densify regenerates "
        "from the coarse layout and would discard it -- if the same change "
        "can be made there (coarse_add_strip / coarse_remove_strip / "
        "coarse_divide), it survives re-densifying."
    )


@tool(
    "dense_plan_line_removal",
    "Preview what removing the line through an edge of the DENSE mesh would "
    "cost, changing nothing: the whole strip through the edge goes (a quad "
    "mesh cannot lose one edge), its two sides are welded so surrounding "
    "vertices move, other strips lying inside it go too (collateral), and a "
    "hole or boundary left with too few splits collapses (boundaries_lost). "
    "Performs the deletion on a copy and reports what came out, because no "
    "cheap test predicts every failure. ok=false means dense_remove_line "
    "refuses for the same reason. Refuses a strip that crosses or reaches a "
    "face that is not a quad (a pole, or a polygon); other strips on the same "
    "mesh are fine.",
    properties={
        "edge": _EDGE,
        "preserve_boundaries": {"type": "boolean", "description": "Pre-split strips so no boundary collapses. Only acts when one is at risk. Default false."},
    },
    required=("edge",),
    read_only=True,
    idempotent=True,
    title="Plan a dense line removal",
)
def _t_dense_plan_line_removal(session: MeshSession, edge: Sequence[str], preserve_boundaries: bool = False) -> dict[str, Any]:
    editor, refusal = _editor(session)
    if refusal:
        return refusal
    uv, reason = _edge(editor.mesh, edge)
    if uv is None:
        return _refuse(reason)
    plan = editor.plan_strip_deletion(uv, preserve_boundaries=bool(preserve_boundaries))
    out = dict((k, plan.get(k)) for k in ("ok", "reason", "faces", "faces_before", "faces_after", "vertices_before", "vertices_after", "boundaries_lost"))
    out["ok"] = bool(plan.get("ok"))
    out["collateral"] = len(plan.get("collateral") or [])
    out["to_split"] = len(plan.get("to_split") or [])
    return out


@tool(
    "dense_remove_line",
    "Delete the whole strip through an edge of the DENSE mesh and weld its two "
    "sides together. Not one edge: the band of faces running the full width of "
    "the mesh goes, and the vertices either side move to meet. Call "
    "dense_plan_line_removal first. Refuses a strip that crosses or reaches a "
    "face that is not a quad, and a result with more non-quad faces than "
    "before. Snapshots the whole mesh first -- undo takes it back. The edit is "
    "lost if coarse_densify runs again.",
    properties={
        "edge": _EDGE,
        "preserve_boundaries": {"type": "boolean", "description": "Pre-split strips so no boundary collapses. Default false."},
    },
    required=("edge",),
    destructive=True,
    title="Remove a dense line",
)
def _t_dense_remove_line(session: MeshSession, edge: Sequence[str], preserve_boundaries: bool = False) -> dict[str, Any]:
    editor, refusal = _editor(session)
    if refusal:
        return refusal
    uv, reason = _edge(editor.mesh, edge)
    if uv is None:
        return _refuse(reason)
    before = session.quality()
    session.snapshot("before dense_remove_line", whole=True)
    ok, _notes = editor.remove_line(uv, preserve_boundaries=bool(preserve_boundaries))
    if not ok:
        session._undo.pop()
        return _refuse(editor.last_reason)
    session.mesh = editor.mesh
    session.mark_changed()
    after, told = _reading(session)
    deletion = dict(editor.last_deletion)
    entry = session.record("dense_remove_line", before=before, after=after, edge=list(edge), faces_out=deletion.get("faces_after"))
    out = {
        "ok": True,
        "step": entry["step"],
        "faces_before": deletion.get("faces_before"),
        "faces": deletion.get("faces_after"),
        "vertices": deletion.get("vertices_after"),
        "collateral": len(deletion.get("collateral") or []),
        "boundaries_lost": deletion.get("boundaries_lost"),
        "quality": after,
        "reading": told["reading"],
        "verdict": told["verdict"],
        "undo_available": session.can_undo(),
    }
    note = _note_layout(session)
    if note:
        out["note"] = note
    return out


@tool(
    "dense_add_line",
    "Add a line to the DENSE mesh: grow one new strip beside the whole polyedge "
    "through an edge, wall to wall (or all the way round a closed one). The new "
    "row is opened by an exact rule: only its own vertices move, and the ones "
    "on a wall stay on it. Nothing else moves -- smooth afterwards if wanted. The "
    "polyedge must be closed or end on the boundary at both ends, and may not "
    "touch a face that is not a quad (a pole, or a polygon). Snapshots the "
    "whole mesh first -- undo takes it back. The edit is lost if coarse_densify "
    "runs again.",
    properties={
        "edge": _EDGE,
    },
    required=("edge",),
    title="Add a dense line",
)
def _t_dense_add_line(session: MeshSession, edge: Sequence[str]) -> dict[str, Any]:
    editor, refusal = _editor(session)
    if refusal:
        return refusal
    uv, reason = _edge(editor.mesh, edge)
    if uv is None:
        return _refuse(reason)
    before = session.quality()
    session.snapshot("before dense_add_line", whole=True)
    ok, _notes = editor.add_line(uv)
    if not ok:
        session._undo.pop()
        return _refuse(editor.last_reason)
    session.mesh = editor.mesh
    session.mark_changed()
    after, told = _reading(session)
    addition = dict(editor.last_addition)
    entry = session.record("dense_add_line", before=before, after=after, edge=list(edge), faces_out=addition.get("faces_after"))
    out = {
        "ok": True,
        "step": entry["step"],
        "faces_before": addition.get("faces_before"),
        "faces": addition.get("faces_after"),
        "polyedge_vertices": addition.get("polyedge_vertices"),
        "closed": addition.get("closed"),
        "quality": after,
        "reading": told["reading"],
        "verdict": told["verdict"],
        "undo_available": session.can_undo(),
    }
    note = _note_layout(session)
    if note:
        out["note"] = note
    return out
