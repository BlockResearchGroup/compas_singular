"""MCP tools that pull geometry from a live Rhino document and push meshes back, via the file spool.

Pull once, improve, push at the end: a round trip loses precision to ``Point3f``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any

from compas_singular.mcp.bridge import spool
from compas_singular.mcp.bridge import wire
from compas_singular.mcp.describe import describe
from compas_singular.mcp.handle import vertex_handle
from compas_singular.mcp.library import thresholds
from compas_singular.mcp.registry import tool

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarsePseudoQuadMesh
    from compas_singular.mcp.session import MeshSession


__all__ = []

#: The layer a dense quad mesh lives on, in the layer scheme every ``CMD_``
#: command already uses.
DEFAULT_LAYER = "QuadMesh"


def _ask(verb: str, args: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    """One request to Rhino, and its answer. Never raises, never hangs."""
    try:
        request_id = spool.post(verb, args or {}, ttl=float(timeout))
    except OSError as exc:
        return {"ok": False, "reason": "could not write to the spool directory {}: {}".format(spool.default_directory(), exc)}
    return spool.wait(request_id, timeout=float(timeout))


def unseen_refusal(session: MeshSession) -> dict[str, Any] | None:
    """A refusal if the current mesh has not been rendered since it changed."""
    if session.visually_current:
        return None
    return {
        "ok": False,
        "reason": "this mesh has changed since it was last drawn, so "
        "nothing has actually looked at what is about to be "
        "delivered. Call inspect with image=true first, judge the "
        "boundaries, the point features, the guides and the "
        "continuity of the polyedges, and then deliver.",
        "fix": {"tool": "inspect", "arguments": {"image": True}},
    }


@tool(
    "rhino_status",
    "Whether Rhino is listening, which document is open, and how many requests "
    "are queued. Call this FIRST if a pull or push has failed: it separates "
    '"no link attached" -- the person must run CMD_mcp_link in Rhino -- from '
    '"attached but busy", which means Rhino is inside a command and the link is '
    "deferring rather than touching the document underneath them. Reads "
    "nothing from the document and changes nothing.",
    properties={
        "timeout": {"type": "number", "description": "Seconds to wait for the ping. Default 5."},
    },
    read_only=True,
    idempotent=True,
    open_world=True,
    title="Rhino link status",
)
def _t_rhino_status(session: MeshSession, timeout: float = 5.0) -> dict[str, Any]:
    state = spool.link_state()
    queued = len(spool.pending())
    if state is None:
        return {
            "ok": True,
            "attached": False,
            "queued": queued,
            "spool": spool.default_directory(),
            "reading": "No link is attached. Run CMD_mcp_link in Rhino to attach one, then retry. A pull or push that timed out was withdrawn, so nothing will fire late.",
        }
    # A live link answers a ping; a stale heartbeat means Rhino went away
    # without detaching, so say that rather than reporting it as attached.
    reply = _ask("ping", timeout=timeout)
    alive = bool(reply.get("ok"))
    return {
        "ok": True,
        "attached": alive,
        "queued": queued,
        "document": state.get("document"),
        "seconds_since_heartbeat": round(state.get("age", 0.0), 1),
        "stale": state.get("stale"),
        "spool": spool.default_directory(),
        "reading": (
            "Attached to {} and answering.".format(state.get("document") or "an unnamed document")
            if alive
            else "A link is registered but did not answer a ping. Rhino is either inside a command or has been closed without detaching."
        ),
    }


@tool(
    "rhino_pull",
    "Take the mesh and the boundary curves out of the open Rhino document into "
    "this session, replacing whatever was held. Reads the outer and inner "
    "boundaries, the guide curves, the point features, and the mesh on the "
    "named layer. The "
    "boundaries matter: they are the walls that boundary vertices SLIDE along "
    "during smoothing, and without them a faceted boundary can only get "
    "smoother, never truer. IF THERE IS NO MESH ON THAT LAYER BUT THERE IS A "
    "BOUNDARY, this still succeeds -- nothing is loaded to inspect or smooth "
    "yet, but there is enough to call create_coarse_mesh and build one. Reads "
    "the document and changes nothing in it. Discards the current undo "
    "history, since positions from one mesh cannot be applied to another. "
    "With selection=true it reads what the person has SELECTED instead of the "
    "fixed layers: closed curves are loops (the largest is the outer wall), "
    "open curves guides, points point features, the first mesh the mesh. "
    "For a COARSE LAYOUT use rhino_pull_coarse: this tool would load one as a "
    "plain dense mesh and lose its strips, densities and patterns.",
    properties={
        "layer": {"type": "string", "description": 'Layer holding the mesh. Default "QuadMesh" (the dense mesh). Ignored with selection=true.'},
        "selection": {"type": "boolean", "description": "Read the current selection instead of the Outer/Inner/Guides/PointFeatures layers. Default false."},
        "spacing": {
            "type": "number",
            "description": "How finely a CURVED boundary is sampled into a "
            "polyline. Default 0.125. Sample finer than the "
            "element size: these points become the wall, so the "
            "boundary is only as smooth as what is handed in.",
        },
        "timeout": {"type": "number", "description": "Seconds to wait for Rhino. Default 30."},
    },
    read_only=True,
    open_world=True,
    title="Pull from Rhino",
)
def _t_rhino_pull(session: MeshSession, layer: str = DEFAULT_LAYER, selection: bool = False, spacing: float = 0.125, timeout: float = 30.0) -> dict[str, Any]:
    reply = _ask("pull", {"layer": layer, "spacing": float(spacing), "selection": bool(selection)}, timeout=timeout)
    if not reply.get("ok"):
        return {"ok": False, "reason": reply.get("reason", "the pull failed"), "link": reply.get("link"), "timed_out": reply.get("timed_out")}

    result = reply.get("result") or {}
    payload = result.get("mesh")
    outer = result.get("outer") or []
    inners = result.get("inners") or []
    walls = ([outer] if outer else []) + list(inners)

    mesh = None
    if payload:
        try:
            mesh = wire.mesh_from_wire(payload)
        except wire.WireError as exc:
            return {"ok": False, "reason": "the mesh Rhino sent could not be rebuilt: {}".format(exc)}
    elif not walls:
        if selection:
            return {"ok": False, "reason": "nothing usable is selected in Rhino -- select a closed boundary curve (and any holes, guides, points or a mesh), then pull again"}
        return {"ok": False, "reason": "Rhino answered but there is no mesh on layer {!r} and no boundary either -- nothing came back to work with.".format(layer)}
    layer = result.get("layer") or layer

    had_coarse = session.coarse is not None
    session.adopt(
        mesh, walls=walls, guides=result.get("guides") or [], points=result.get("points") or [], source={"kind": "rhino", "layer": layer, "document": result.get("document")}
    )

    out = {"ok": True, "layer": layer, "document": result.get("document"), "walls": len(walls), "guides": len(session.guides), "point_features": len(session.points)}
    if had_coarse and session.coarse is None:
        out["coarse_cleared"] = "the domain in Rhino differs from the one the coarse layout was built from, so the layout was dropped -- call create_coarse_mesh again"
    if mesh is not None:
        metrics = session.quality()
        told = describe(session.mesh, metrics, thresholds())
        session.record("rhino_pull", before=None, after=metrics, layer=layer, walls=len(walls))
        out.update(
            {
                "vertices": mesh.number_of_vertices(),
                "faces": mesh.number_of_faces(),
                "quality": metrics,
                "reading": told["reading"],
                "verdict": told["verdict"],
                "worst_face_at": told["worst_face_at"],
            }
        )
    else:
        session.record("rhino_pull", before=None, after=None, layer=layer, walls=len(walls))
        out.update(
            {
                "vertices": None,
                "faces": None,
                "reading": "No mesh on layer {!r} -- pulled the boundary, "
                "{} guide(s) and {} point feature(s) only. "
                "Nothing is loaded to inspect or smooth yet; "
                "call create_coarse_mesh to build a coarse "
                "layout from this domain.".format(layer, len(session.guides), len(session.points)),
            }
        )
    if not walls:
        out["warning"] = (
            "no boundary curves came back, so the boundary "
            "vertices have nothing true to slide along. Smoothing "
            "can still run, but it will only make the boundary "
            "smoother, not truer."
        )
    return out


@tool(
    "rhino_push",
    "Bake the improved mesh back into the open Rhino document, on the given "
    "layer. WRITES TO A DOCUMENT SOMEBODY HAS OPEN: the mesh that was there is "
    'moved to a "::MCP::Before" sub-layer first so the two can be compared, and '
    "the write is one named undo step so it can be taken back in Rhino. It will "
    "not fire while Rhino is inside a command -- it waits. Push a pulled mesh "
    "only when it is actually better than it was pulled, and a mesh from "
    "coarse_densify only once it has been inspected and judged; otherwise say "
    "so and push nothing. Do not pull again afterwards to check: Rhino stores mesh vertices "
    "in single precision, so a round trip loses about 2e-6 per coordinate and "
    "the copy here is the accurate one. REFUSES until the mesh being pushed "
    "has been drawn and looked at -- call inspect with image=true first.",
    properties={
        "layer": {"type": "string", "description": 'Layer to bake onto. Default "QuadMesh".'},
        "timeout": {"type": "number", "description": "Seconds to wait for Rhino. Default 30."},
    },
    destructive=True,
    open_world=True,
    title="Push to Rhino",
)
def _t_rhino_push(session: MeshSession, layer: str = DEFAULT_LAYER, timeout: float = 30.0) -> dict[str, Any]:
    if not session.loaded:
        return {"ok": False, "reason": "no mesh is loaded, so there is nothing to push"}
    refusal = unseen_refusal(session)
    if refusal:
        return refusal
    payload = wire.mesh_to_wire(session.mesh)
    reply = _ask("push", {"layer": layer, "mesh": payload}, timeout=timeout)
    if not reply.get("ok"):
        return {
            "ok": False,
            "reason": reply.get("reason", "the push failed"),
            "link": reply.get("link"),
            "timed_out": reply.get("timed_out"),
            "in_progress": reply.get("in_progress"),
        }
    result = reply.get("result") or {}
    session.record("rhino_push", before=None, after=session.quality(), layer=layer)
    return {
        "ok": True,
        "layer": layer,
        "document": result.get("document"),
        "vertices": len(payload["vertices"]),
        "faces": len(payload["faces"]),
        "kept_previous_on": result.get("before_layer"),
        "undo_record": result.get("undo_record"),
        "reading": "Baked {} faces onto {}. The mesh that was there is on {}.".format(len(payload["faces"]), layer, result.get("before_layer") or "no backup layer"),
    }


# ==============================================================================
# the coarse layout
# ==============================================================================


def _float32_collapsed(mesh: CoarsePseudoQuadMesh) -> list[str]:
    """Faces Rhino would refuse, as handles: two corners equal at single precision."""
    import struct

    def f32(point: Any) -> tuple[float, ...]:
        return tuple(struct.unpack("3f", struct.pack("3f", *[float(c) for c in point[:3]])))

    bad = []
    for fkey in mesh.faces():
        corners = [f32(mesh.vertex_coordinates(v)) for v in mesh.face_vertices(fkey)]
        if len(set(corners)) < len(corners):
            centre = [sum(c[i] for c in corners) / len(corners) for i in range(3)]
            bad.append(vertex_handle(centre))
    return bad


def unseen_coarse_refusal(session: MeshSession) -> dict[str, Any] | None:
    """A refusal if the coarse layout has changed since it was last drawn."""
    if session.coarse_visually_current:
        return None
    return {
        "ok": False,
        "reason": "the coarse layout has changed since it was last drawn, so "
        "nothing has looked at what is about to be written into the "
        "document. Call coarse_inspect with image=true first -- "
        "check the strips, the poles and where the layout meets "
        "the walls -- and then push.",
        "fix": {"tool": "coarse_inspect", "arguments": {"image": True}},
    }


@tool(
    "rhino_push_coarse",
    "Write the coarse layout into the open Rhino document exactly where the "
    "CMD_ commands keep one -- the document's session, with its strips, "
    "densities, patterns and edge shapes, drawn on TopologyProblem::Skeleton::"
    "Mesh, ::Poles, ::Polylines and ::EdgeCurves. After it, CMD_densities, "
    "CMD_quad_mesh and CMD_edit_coarse_mesh carry on from this layout in Rhino. "
    "WRITES TO A DOCUMENT SOMEBODY HAS OPEN: what was drawn on those layers "
    'moves to Skeleton::MCP::Before, and the write is one undo step named "MCP '
    'push coarse" that takes the layout and its drawing back together. REFUSES '
    "until the current layout has been drawn and looked at -- call "
    "coarse_inspect with image=true first.",
    properties={
        "timeout": {"type": "number", "description": "Seconds to wait for Rhino. Default 30."},
    },
    destructive=True,
    open_world=True,
    title="Push the coarse layout to Rhino",
)
def _t_rhino_push_coarse(session: MeshSession, timeout: float = 30.0) -> dict[str, Any]:
    import compas
    from compas_singular.mcp import tools_coarse

    if session.coarse is None:
        return {"ok": False, "reason": "no coarse layout is loaded, so there is nothing to push"}
    refusal = unseen_coarse_refusal(session)
    if refusal:
        return refusal

    # The edge shapes FIRST: building them snaps boundary corners onto their
    # walls, and the baked mesh has to have the corners the curves end on.
    mapping = tools_coarse._edges_to_curves(session) or {}
    coarse = session.coarse

    collapsed = _float32_collapsed(coarse)
    if collapsed:
        return {
            "ok": False,
            "reason": "Rhino would refuse the whole layout: {} patch(es) have "
            "two corners that coincide at single precision, first "
            "at {}. coarse_move_corner or coarse_undo to open them "
            "up.".format(len(collapsed), collapsed[0]),
            "patches": collapsed[:5],
        }

    edge_curves = []
    for u, v in coarse.edges():
        curve = mapping.get((u, v)) or list(reversed(mapping.get((v, u)) or []))
        if len(curve) < 2:
            curve = [list(coarse.vertex_coordinates(u)), list(coarse.vertex_coordinates(v))]
        edge_curves.append([list(p) for p in curve])

    layout = coarse.copy()
    layout.attributes["quad_mesh"] = None
    layout.attributes["polygonal_mesh"] = None
    # ``mapping`` goes on the layout, so CMD_quad_mesh (or a later coarse_load)
    # reads the edge shapes straight off it.
    layout.set_edges_to_curves(mapping)
    # Replaced, not kept: a layout pulled from Rhino still carries the polylines
    # of the layout it was BEFORE this session edited it. Every edge, so
    # CMD_quad_mesh's traced-branch lookup finds the curves this session made --
    # a drawn cut, a warped corner -- and not only the ones its own
    # decomposition would have traced.
    layout.set_shape_polylines(edge_curves)
    tools_coarse._patterns(layout)

    reply = _ask("push_coarse", {"coarse": compas.json_dumps(layout)}, timeout=timeout)
    if not reply.get("ok"):
        return {
            "ok": False,
            "reason": reply.get("reason", "the push failed"),
            "link": reply.get("link"),
            "timed_out": reply.get("timed_out"),
            "in_progress": reply.get("in_progress"),
        }
    result = reply.get("result") or {}
    session.record("rhino_push_coarse", faces=coarse.number_of_faces(), strips=len(list(coarse.strips())))
    out = {
        "ok": True,
        "document": result.get("document"),
        "layers": result.get("layers"),
        "faces": coarse.number_of_faces(),
        "strips": len(list(coarse.strips())),
        "poles": result.get("poles"),
        "edge_curves": result.get("edge_curves"),
        "shaped_edges": sum(1 for curve in edge_curves if len(curve) > 2),
        "kept_previous_on": result.get("before_layer"),
        "undo_record": result.get("undo_record"),
        "reading": "Drew {} patches onto {}. Run CMD_densities or CMD_quad_mesh in Rhino to carry on from it.".format(
            coarse.number_of_faces(), (result.get("layers") or {}).get("mesh", "Skeleton::Mesh")
        ),
    }
    return out


# ==============================================================================
# pulling a coarse layout back
# ==============================================================================


def _layout_from_pull(result: dict[str, Any]) -> CoarsePseudoQuadMesh:
    """The layout ``pull_coarse`` sent back, with its shape polylines as user curves."""
    import compas
    from compas_singular.mcp import tools_coarse

    layout = compas.json_loads(result["coarse"])
    if not hasattr(layout, "strips"):
        raise TypeError("the session holds a {}, not a layout".format(type(layout).__name__))
    if not list(layout.strips()):
        layout.collect_strips()
    densities = layout.get_strip_densities()
    for skey in layout.strips():
        densities.setdefault(skey, 1)

    def key2(point: Any) -> tuple[float, float]:
        return (round(point[0], 3), round(point[1], 3))

    known = set((key2(c[0]), key2(c[-1])) for c in tools_coarse._user_curves(layout))
    edge_ends = set()
    for u, v in layout.edges():
        a, b = key2(layout.vertex_coordinates(u)), key2(layout.vertex_coordinates(v))
        edge_ends.add((a, b))
        edge_ends.add((b, a))
    extra = []
    for curve in layout.shape_polylines():
        if len(curve) < 3:
            continue
        ends = (key2(curve[0]), key2(curve[-1]))
        if ends in known or ends not in edge_ends:
            continue
        extra.append([list(p) for p in curve])
        known.add(ends)
        known.add((ends[1], ends[0]))
    if extra:
        layout.attributes["user_curves"] = tools_coarse._user_curves(layout) + extra
    return layout


@tool(
    "rhino_pull_coarse",
    "Take a COARSE LAYOUT out of the open Rhino document into this session -- "
    "the one CMD_coarse_mesh / CMD_edit_coarse_mesh / CMD_densities keep in the "
    "document's session, or one rhino_push_coarse wrote -- with its strips, "
    "densities, patterns and edge shapes, and the domain. The counterpart of "
    "rhino_push_coarse: edit in Rhino, pull back, keep editing here. Replaces "
    "the coarse layout held and its undo; if the domain differs, the dense mesh "
    "is dropped too. Reads the document and changes nothing in it.",
    properties={
        "spacing": {"type": "number", "description": "How finely a CURVED boundary is sampled. Default 0.125."},
        "timeout": {"type": "number", "description": "Seconds to wait for Rhino. Default 30."},
    },
    read_only=True,
    open_world=True,
    title="Pull the coarse layout from Rhino",
)
def _t_rhino_pull_coarse(session: MeshSession, spacing: float = 0.125, timeout: float = 30.0) -> dict[str, Any]:
    from compas_singular.mcp import tools_coarse

    reply = _ask("pull_coarse", {"spacing": float(spacing)}, timeout=timeout)
    if not reply.get("ok"):
        return {"ok": False, "reason": reply.get("reason", "the pull failed"), "link": reply.get("link"), "timed_out": reply.get("timed_out")}
    result = reply.get("result") or {}
    if not result.get("coarse"):
        return {
            "ok": False,
            "reason": "the session of {} holds no coarse layout -- run CMD_coarse_mesh there, or rhino_push_coarse one from here, first".format(result.get("document")),
        }
    try:
        layout = _layout_from_pull(result)
    except Exception as exc:
        return {"ok": False, "reason": "the layout Rhino sent could not be rebuilt: {}: {}".format(type(exc).__name__, exc)}

    outer = result.get("outer") or []
    walls = ([outer] if outer else []) + list(result.get("inners") or [])
    guides = result.get("guides") or []
    points = result.get("points") or []
    domain = ([[list(p) for p in w] for w in walls], [[list(p) for p in g] for g in guides], [list(p) for p in points])
    dropped = False
    if domain != session._domain():
        dropped = session.loaded
        session.adopt(None, walls=walls, guides=guides, points=points, source={"kind": "rhino_coarse", "document": result.get("document")})
    session.coarse = layout
    session.decomposition = None
    session._coarse_undo = []
    entry = session.record("rhino_pull_coarse", faces=layout.number_of_faces())
    out = {
        "ok": True,
        "step": entry["step"],
        "document": result.get("document"),
        "faces": layout.number_of_faces(),
        "strips": len(list(layout.strips())),
        "densities": dict((str(k), d) for k, d in layout.get_strip_densities().items()),
        "patterns": tools_coarse._pattern_counts(layout),
        "poles": tools_coarse._pole_handles(layout),
        "walls": len(session.walls),
        "guides": len(session.guides),
        "point_features": len(session.points),
        "shaped_edges_read": len(tools_coarse._user_curves(layout)),
        "dense_mesh_dropped": dropped,
    }
    if not walls:
        out["warning"] = "no boundary curves came back from Outer/Inner, so curved boundary edges will densify as straight chords"
    return out


# ==============================================================================
# markers
# ==============================================================================

MARKER_KINDS = ("worst", "singularity", "pole")


@tool(
    "rhino_push_markers",
    "Put labelled dots into the Rhino document where the person should look "
    "at the DENSE mesh in hand: the worst faces (labelled with their smallest "
    "corner angle), the irregular interior vertices (with their valence), and "
    "the poles. On TopologyProblem::MCP::Markers::worst / ::singularity / "
    '::pole, replacing the previous set, as one undo step named "MCP markers". '
    "clear=true removes them all. Writes only to that layer branch -- never to "
    "a mesh or a layout.",
    properties={
        "worst": {"type": "integer", "description": "How many worst faces to mark. Default 5; 0 marks none."},
        "singularities": {"type": "boolean", "description": "Mark irregular interior vertices. Default true."},
        "poles": {"type": "boolean", "description": "Mark poles. Default true."},
        "clear": {"type": "boolean", "description": "Remove every marker and add none. Default false."},
        "timeout": {"type": "number", "description": "Seconds to wait for Rhino. Default 30."},
    },
    open_world=True,
    title="Mark problems in Rhino",
)
def _t_rhino_push_markers(session: MeshSession, worst: int = 5, singularities: bool = True, poles: bool = True, clear: bool = False, timeout: float = 30.0) -> dict[str, Any]:
    from compas_singular.mcp.handle import _face_min_angle
    from compas_singular.mcp.handle import _singular_vertices

    markers = []
    if not clear:
        if not session.loaded:
            return {"ok": False, "reason": "no dense mesh is loaded, so there is nothing to mark -- coarse_densify or rhino_pull first"}
        mesh = session.mesh
        pole_keys = set(mesh.poles()) if hasattr(mesh, "poles") else set()
        if int(worst) > 0:
            ranked = sorted(mesh.faces(), key=lambda f: _face_min_angle(mesh, f))
            for fkey in ranked[: int(worst)]:
                corners = [mesh.vertex_coordinates(v) for v in mesh.face_vertices(fkey)]
                centre = [sum(c[i] for c in corners) / len(corners) for i in range(3)]
                markers.append({"kind": "worst", "point": centre, "text": "{:.1f} deg".format(_face_min_angle(mesh, fkey))})
        if singularities:
            for vkey in sorted(_singular_vertices(mesh) - pole_keys):
                markers.append({"kind": "singularity", "point": list(mesh.vertex_coordinates(vkey)), "text": "valence {}".format(len(mesh.vertex_neighbors(vkey)))})
        if poles:
            for vkey in sorted(pole_keys):
                markers.append({"kind": "pole", "point": list(mesh.vertex_coordinates(vkey)), "text": "pole"})

    reply = _ask("push_markers", {"markers": markers, "kinds": list(MARKER_KINDS)}, timeout=timeout)
    if not reply.get("ok"):
        return {
            "ok": False,
            "reason": reply.get("reason", "the push failed"),
            "link": reply.get("link"),
            "timed_out": reply.get("timed_out"),
            "in_progress": reply.get("in_progress"),
        }
    result = reply.get("result") or {}
    counts = dict((kind, 0) for kind in MARKER_KINDS)
    for marker in markers:
        counts[marker["kind"]] += 1
    return {
        "ok": True,
        "document": result.get("document"),
        "layer": result.get("layer"),
        "counts": counts,
        "cleared": bool(clear),
        "undo_record": result.get("undo_record"),
        "reading": (
            "Removed every marker."
            if clear
            else "Marked {} worst face(s), {} singularit(ies), {} pole(s) in Rhino.".format(counts["worst"], counts["singularity"], counts["pole"])
        ),
    }
