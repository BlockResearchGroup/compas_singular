"""MCP tools that load a mesh from disk and write one back, with Rhino closed."""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas_singular.mcp.describe import describe
from compas_singular.mcp.library import thresholds
from compas_singular.mcp.registry import tool
from compas_singular.mcp.tools_rhino import unseen_refusal

if TYPE_CHECKING:
    from compas_singular.mcp.session import MeshSession


__all__ = []


@tool(
    "load_mesh",
    "Read a quad mesh from a COMPAS json file into this session, replacing "
    "whatever was held. Use this to work with Rhino closed. Boundary curves are "
    "NOT in a mesh file, so pass them as walls if you have them -- without "
    "walls, boundary smoothing can only make the boundary smoother, never "
    "truer, and a faceted boundary stays faceted. Discards the undo history.",
    properties={
        "path": {"type": "string", "description": "Path to a .json written by Mesh.to_json."},
        "walls": {
            "type": "array",
            "description": "Optional boundary loops, as lists of [x, y, z] points: the outer loop first, then any inner ones. These are what boundary vertices slide along.",
            "items": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        },
    },
    required=("path",),
    open_world=True,
    title="Load a mesh from a file",
)
def _t_load_mesh(session: MeshSession, path: str, walls: Sequence[Sequence[Sequence[float]]] | None = None) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {"ok": False, "reason": "no file at {}".format(path)}
    try:
        import compas

        mesh = compas.json_load(path)
    except Exception as exc:
        return {"ok": False, "reason": "could not read {}: {}: {}".format(path, type(exc).__name__, exc)}
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        return {"ok": False, "reason": "{} holds a {}, not a mesh".format(path, type(mesh).__name__)}

    had_coarse = session.coarse is not None
    # A mesh file carries no guides or point features, so none survive from
    # whatever was loaded before -- they would belong to a different domain.
    session.adopt(mesh, walls=walls or [], guides=[], points=[], source={"kind": "file", "name": os.path.basename(path), "path": path})
    metrics = session.quality()
    told = describe(session.mesh, metrics, thresholds())
    session.record("load_mesh", before=None, after=metrics, path=path, walls=len(session.walls))
    out = {
        "ok": True,
        "path": path,
        "type": type(mesh).__name__,
        "vertices": mesh.number_of_vertices(),
        "faces": mesh.number_of_faces(),
        "walls": len(session.walls),
        "quality": metrics,
        "reading": told["reading"],
        "verdict": told["verdict"],
        "worst_face_at": told["worst_face_at"],
    }
    if had_coarse and session.coarse is None:
        out["coarse_cleared"] = "these walls differ from the domain the coarse layout was built from, so the layout was dropped"
    if not session.walls:
        out["warning"] = "no walls were given, so boundary vertices have nothing true to slide along. Smoothing will still run; it just cannot correct the boundary."
    return out


@tool(
    "save_mesh",
    "Write the mesh in hand to a COMPAS json file. Round-trips exactly, in "
    "double precision, keeping pole data -- unlike a push to Rhino, which goes "
    "through single-precision vertices. Use this to keep a result when Rhino is "
    "not involved. REFUSES until the mesh being saved has been drawn and "
    "looked at -- call inspect with image=true first.",
    properties={
        "path": {"type": "string", "description": "Where to write the .json."},
    },
    required=("path",),
    open_world=True,
    title="Save the mesh to a file",
)
def _t_save_mesh(session: MeshSession, path: str) -> dict[str, Any]:
    if not session.loaded:
        return {"ok": False, "reason": "no mesh is loaded, so there is nothing to save"}
    refusal = unseen_refusal(session)
    if refusal:
        return refusal
    folder = os.path.dirname(os.path.abspath(path))
    if folder and not os.path.isdir(folder):
        return {"ok": False, "reason": "no directory at {}".format(folder)}
    try:
        session.mesh.to_json(path)
    except Exception as exc:
        return {"ok": False, "reason": "could not write {}: {}: {}".format(path, type(exc).__name__, exc)}
    return {
        "ok": True,
        "path": path,
        "faces": session.mesh.number_of_faces(),
        "vertices": session.mesh.number_of_vertices(),
        "reading": "Written in double precision, with type information, so load_mesh gives back the same class.",
    }
