"""MCP tools for the coarse layout: create, set densities and patterns, densify, edit, save and load.

The coarse layout has its own undo stack, independent of the dense mesh's.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import os
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas_singular.algorithms.skeleton_decomposition import SkeletonDecomposition
from compas_singular.datastructures.mesh_quad_coarse.patterns import PATTERNS
from compas_singular.datastructures.mesh_quad_coarse.patterns import reconcile_strip_densities
from compas_singular.editing.coarseeditor import CoarseEditor
from compas_singular.editing.rebuild import warp_polyline
from compas_singular.mcp import render
from compas_singular.mcp.describe import describe
from compas_singular.mcp.handle import _parse as parse_handle
from compas_singular.mcp.handle import resolve
from compas_singular.mcp.handle import vertex_handle
from compas_singular.mcp.library import thresholds
from compas_singular.mcp.registry import tool

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarsePseudoQuadMesh
    from compas_singular.mcp.session import MeshSession


__all__ = []


def _needs_walls(session: MeshSession) -> dict[str, Any] | None:
    if not session.walls:
        return {"ok": False, "reason": "no boundary is loaded -- call rhino_pull, or load_mesh with walls, first: there is nothing to decompose without one"}
    return None


def _needs_coarse(session: MeshSession) -> dict[str, Any] | None:
    if session.coarse is None:
        return {"ok": False, "reason": "no coarse layout is loaded -- call create_coarse_mesh first"}
    return None


def _polyline_points(curve: Any) -> Any:
    """Plain points from a ``Polyline`` or an already-bare point list."""
    return getattr(curve, "points", curve)


def _close(a: Sequence[float], b: Sequence[float], tol: float = 1e-9) -> bool:
    return all(abs(a[i] - b[i]) <= tol for i in range(3))


def _open_loop(points: list[list[float]]) -> list[list[float]]:
    """Drop a duplicated closing point -- ``from_boundary`` wants an OPEN loop."""
    if len(points) > 1 and _close(points[0], points[-1]):
        return points[:-1]
    return points


def _editor(session: MeshSession, shapes: Sequence[Sequence[Any]] | None = None) -> CoarseEditor:
    """A fresh ``CoarseEditor`` on the session's layout, seeded with the shapes earlier cuts registered."""
    loops = [_open_loop([list(p) for p in _polyline_points(wall)]) for wall in session.walls] or None
    polylines = [[list(p) for p in curve] for curve in (shapes or []) if len(curve) > 2]
    editor = CoarseEditor(session.coarse, loops=loops, polylines=polylines)
    for curve in _user_curves(session.coarse):
        editor.curves[editor._curve_key(curve[0], curve[-1])] = curve
    editor._snapshot_curves = dict(editor.curves)
    return editor


def _resolve_edge(mesh: CoarsePseudoQuadMesh, handles: Sequence[str]) -> tuple[tuple[Any, ...] | None, str | None]:
    """Two vertex handles as a coarse edge ``(u, v)``, or ``(None, reason)``."""
    if len(handles) != 2:
        return None, "an edge needs exactly two vertex handles"
    keys = []
    for item in handles:
        key, how = resolve(mesh, item)
        if key is None:
            return None, "could not place {!r} on the coarse layout: {}".format(item, how)
        keys.append(key)
    return tuple(keys), None


def _pole_handles(mesh: CoarsePseudoQuadMesh) -> list[str]:
    poles = mesh.poles() if hasattr(mesh, "poles") else []
    return [vertex_handle(mesh.vertex_coordinates(key)) for key in poles]


# ------------------------------------------------------------------------------
# faces, which have no handle of their own either
# ------------------------------------------------------------------------------


def _inside(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    """Even-odd test in xy. The layout is planar; z plays no part."""
    x, y = point[0], point[1]
    inside = False
    for i in range(len(polygon)):
        (x0, y0), (x1, y1) = polygon[i - 1][:2], polygon[i][:2]
        if (y0 > y) != (y1 > y):
            if x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
                inside = not inside
    return inside


def _corners(coarse: CoarsePseudoQuadMesh, fkey: Any) -> list[list[float]]:
    """A face's distinct corners -- a pseudo-quad repeats its pole."""
    out = []
    for vkey in coarse.face_vertices(fkey):
        xyz = coarse.vertex_coordinates(vkey)
        if not out or xyz != out[-1]:
            out.append(xyz)
    if len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


def _face_point(coarse: CoarsePseudoQuadMesh, fkey: Any) -> str:
    """A point inside a face, used as its handle: the centroid, or a diagonal midpoint if that falls outside."""
    corners = _corners(coarse, fkey)
    centre = [sum(c[i] for c in corners) / len(corners) for i in range(3)]
    candidates = [centre] + [[(corners[i][k] + corners[(i + 2) % len(corners)][k]) / 2.0 for k in range(3)] for i in range(len(corners))]
    for point in candidates:
        if _inside(point, corners):
            return vertex_handle(point)
    return vertex_handle(centre)


def _face_at(coarse: CoarsePseudoQuadMesh, handle: str) -> tuple[Any, str | None]:
    """The face a handle points into, or ``(None, reason)``."""
    point = parse_handle(handle)
    if point is None:
        return None, "{!r} is not a handle -- use one from coarse_inspect".format(handle)
    for fkey in coarse.faces():
        if _inside(point, _corners(coarse, fkey)):
            return fkey, None
    # A handle is rounded, so one reported for a sliver patch can land a hair
    # outside it. Accept a face's own reported handle even then.
    for fkey in coarse.faces():
        if _face_point(coarse, fkey) == vertex_handle(point):
            return fkey, None
    return None, "no patch contains {}".format(handle)


def _patterns(coarse: CoarsePseudoQuadMesh) -> dict[Any, str]:
    """Complete and prune the face-to-pattern map in place."""
    patterns = coarse.attributes.setdefault("dense_pattern", {})
    faces = set(coarse.faces())
    for fkey in list(patterns):
        if fkey not in faces:
            del patterns[fkey]
    for fkey in faces:
        patterns.setdefault(fkey, "ortho")
    return patterns


def _pattern_counts(coarse: CoarsePseudoQuadMesh) -> dict[str, int]:
    counts = dict((name, 0) for name in PATTERNS)
    for pattern in _patterns(coarse).values():
        counts[pattern] = counts.get(pattern, 0) + 1
    return counts


def _raised(changes: dict[Any, tuple[Any, Any]]) -> dict[str, list[Any]]:
    """``reconcile_strip_densities``' ``{skey: (old, new)}``, JSON-ready."""
    return dict((str(skey), [old, new]) for skey, (old, new) in sorted(changes.items()))


def _user_curves(coarse: CoarsePseudoQuadMesh) -> list[list[list[float]]]:
    """The edge shapes a cut left in ``attributes['user_curves']``, as plain point lists."""
    return [[list(p) for p in curve] for curve in (coarse.attributes.get("user_curves") or []) if len(curve) >= 2]


def _closed(loop: Any) -> list[list[float]]:
    points = [list(p) for p in _polyline_points(loop)]
    if len(points) > 2 and not _close(points[0], points[-1]):
        points.append(list(points[0]))
    return points


def _edges_to_curves(session: MeshSession) -> dict[tuple[Any, Any], list[list[float]]] | None:
    """The shape of each coarse edge: from the decomposition, the saved curves or the walls, then cut shapes on top."""
    coarse = session.coarse
    mapping = None
    if session.decomposition is not None:
        try:
            mapping, _tally = session.decomposition.edges_to_curves(coarse=coarse)
        except Exception:
            mapping = None
    if not mapping and hasattr(coarse, "edges_to_curves"):
        mapping = coarse.edges_to_curves() or None
    user = _user_curves(coarse)
    if not mapping and (session.walls or user):
        # A strip edit or a cut clears the stored curves (they are keyed by
        # vertex), so a loaded layout that has been edited lands here.
        from compas_singular.datastructures import coarse_edges_to_curves

        mapping, _tally = coarse_edges_to_curves(coarse, loops=[_closed(wall) for wall in session.walls], polylines=user)
    if not mapping:
        return None
    mapping = dict(mapping)
    tol = 1e-3
    for u, v in coarse.edges():
        pu, pv = coarse.vertex_coordinates(u), coarse.vertex_coordinates(v)
        for curve in user:
            if _close(curve[0], pu, tol) and _close(curve[-1], pv, tol):
                shape = curve
            elif _close(curve[0], pv, tol) and _close(curve[-1], pu, tol):
                shape = list(reversed(curve))
            else:
                continue
            mapping[(u, v)] = shape
            mapping[(v, u)] = list(reversed(shape))
            break
    return mapping


# ------------------------------------------------------------------------------
# what a cut has to carry across its renumbering commit
# ------------------------------------------------------------------------------


def _distance_to_segment(point: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, bx, by = a[0], a[1], b[0], b[1]
    dx, dy = bx - ax, by - ay
    span = dx * dx + dy * dy
    t = 0.0 if span == 0 else max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / span))
    return ((point[0] - ax - t * dx) ** 2 + (point[1] - ay - t * dy) ** 2) ** 0.5


def _length(a: Sequence[float], b: Sequence[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _carry_densities(before: CoarsePseudoQuadMesh, after: CoarsePseudoQuadMesh) -> dict[Any, tuple[Any, int]]:
    """Give every strip of ``after`` a density from the strips of ``before`` it overlaps. ``{new skey: (old skey, density)}``."""
    old_edges = []
    for skey in before.strips():
        d = before.get_strip_density(skey)
        for u, v in before.strip_edges(skey):
            if u != v:
                pu, pv = before.vertex_coordinates(u), before.vertex_coordinates(v)
                old_edges.append((skey, d, pu, pv, _length(pu, pv)))

    carried = {}
    for skey in after.strips():
        votes = {}
        for u, v in after.strip_edges(skey):
            if u == v:
                continue
            pa, pb = after.vertex_coordinates(u), after.vertex_coordinates(v)
            best = None
            for old, d, pu, pv, span in old_edges:
                if span <= 0:
                    continue
                # Loose on purpose: a corner the cut put on a CURVED edge sits
                # on the curve, a sagitta off the chord, and a boundary corner
                # is snapped onto its wall at commit.
                off = _distance_to_segment(pa, pu, pv) + _distance_to_segment(pb, pu, pv)
                if off <= 0.2 * span and (best is None or off < best[0]):
                    best = (off, old, d, min(1.0, _length(pa, pb) / span))
            if best is not None:
                _off, old, d, fraction = best
                votes.setdefault(old, [d, []])[1].append(fraction)
        if not votes:
            carried[skey] = (None, 1)
            continue
        old = max(votes, key=lambda k: len(votes[k][1]))
        d, fractions = votes[old]
        share = sum(fractions) / len(fractions)
        carried[skey] = (old, max(1, int(d * share + 0.5)))
    for skey, (_old, d) in carried.items():
        after.set_strip_density(skey, d)
    return carried


def _carry_patterns(before: CoarsePseudoQuadMesh, after: CoarsePseudoQuadMesh) -> dict[Any, str]:
    """Every new patch takes the pattern of the old patch it lies inside."""
    old = [(_corners(before, fkey), pattern) for fkey, pattern in _patterns(before).items()]
    patterns = after.attributes.setdefault("dense_pattern", {})
    patterns.clear()
    for fkey in after.faces():
        point = parse_handle(_face_point(after, fkey))
        patterns[fkey] = next((pattern for corners, pattern in old if _inside(point, corners)), "ortho")
    return patterns


def _pinches(layout: CoarsePseudoQuadMesh, curves: Sequence[Sequence[Sequence[float]]], limit: float) -> list[tuple[str, float]]:
    """Where a new cut runs closer than ``limit`` to an edge it does not touch. ``[(handle, distance)]``, closest first."""
    found = []
    edges = [(layout.vertex_coordinates(u), layout.vertex_coordinates(v)) for u, v in layout.edges()]
    for curve in curves:
        ends = (curve[0], curve[-1])
        for point in curve[1:-1]:
            for pu, pv in edges:
                if any(_close(pu, end, 1e-3) or _close(pv, end, 1e-3) for end in ends):
                    continue
                d = _distance_to_segment(point, pu, pv)
                if d < limit:
                    found.append((d, vertex_handle(point)))
    found.sort()
    return [(handle, round(d, 4)) for d, handle in found[:3]]


def _crosses(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> bool:
    """Whether segments ab and cd properly cross in xy. Touching ends do not count."""

    def side(p: Sequence[float], q: Sequence[float], r: Sequence[float]) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1, d2 = side(c, d, a), side(c, d, b)
    d3, d4 = side(a, b, c), side(a, b, d)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _fold(mesh: CoarsePseudoQuadMesh, vkey: Any) -> str | None:
    """Why moving ``vkey`` broke the layout (crossed edges or an inverted patch), or ``None``."""
    here = mesh.vertex_coordinates(vkey)
    for n in mesh.vertex_neighbors(vkey):
        pn = mesh.vertex_coordinates(n)
        for u, v in mesh.edges():
            if vkey in (u, v) or n in (u, v):
                continue
            if _crosses(here, pn, mesh.vertex_coordinates(u), mesh.vertex_coordinates(v)):
                middle = [(mesh.vertex_coordinates(u)[i] + mesh.vertex_coordinates(v)[i]) / 2.0 for i in range(3)]
                return "the corner's edge to {} would cross the coarse edge at {} -- it has been dragged over a neighbouring patch".format(
                    vertex_handle(pn), vertex_handle(middle)
                )
    for fkey in mesh.vertex_faces(vkey):
        if fkey is None:
            continue
        corners = _corners(mesh, fkey)
        area2 = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(corners, corners[1:] + corners[:1]))
        if area2 <= 0.0:
            return "the patch at {} would turn inside out -- the corner has been dragged past its opposite side".format(_face_point(mesh, fkey))
    return None


def _nearest_wall(walls: Sequence[Any], point: Sequence[float]) -> tuple[int | None, float | None]:
    """Index of the wall loop closest to ``point``, and the distance."""
    best = (None, None)
    for index, wall in enumerate(walls):
        points = _closed(wall)
        for a, b in zip(points, points[1:]):
            d = _distance_to_segment(point, a, b)
            if best[1] is None or d < best[1]:
                best = (index, d)
    return best


def _wall_kink(walls: Sequence[Any], point: Sequence[float], degrees: float = 30.0, tol: float = 1e-3) -> bool:
    """Whether ``point`` sits on a wall vertex that turns by more than ``degrees``."""
    import math

    for wall in walls:
        points = _open_loop([list(p) for p in _polyline_points(wall)])
        n = len(points)
        for i in range(n):
            if _length(points[i], point) > tol:
                continue
            a, b, c = points[i - 1], points[i], points[(i + 1) % n]
            v1 = (b[0] - a[0], b[1] - a[1])
            v2 = (c[0] - b[0], c[1] - b[1])
            n1, n2 = math.hypot(*v1), math.hypot(*v2)
            if n1 == 0 or n2 == 0:
                continue
            cos = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
            if math.degrees(math.acos(cos)) > degrees:
                return True
    return False


def _snap_end(editor: CoarseEditor, point: Sequence[float], tol: float) -> tuple[list[float] | None, float | None]:
    """Snap one end of a drawn cut onto the nearest corner or edge within ``tol``. ``(point, distance)``."""
    mesh = editor.mesh
    best = None
    for vkey in mesh.vertices():
        xyz = mesh.vertex_coordinates(vkey)
        d = _length(point, xyz)
        if d <= editor.corner_tol() and (best is None or d < best[0]):
            best = (d, list(xyz))
    if best is None:
        for u, v in mesh.edges():
            pu, pv = mesh.vertex_coordinates(u), mesh.vertex_coordinates(v)
            dx, dy = pv[0] - pu[0], pv[1] - pu[1]
            span = dx * dx + dy * dy
            if span == 0:
                continue
            t = max(0.0, min(1.0, ((point[0] - pu[0]) * dx + (point[1] - pu[1]) * dy) / span))
            q = [pu[0] + t * dx, pu[1] + t * dy, 0.0]
            d = _length(point, q)
            if best is None or d < best[0]:
                best = (d, q)
    if best is None or best[0] > tol:
        return None, (best[0] if best else None)
    return best[1], best[0]


# ==============================================================================
# building
# ==============================================================================


@tool(
    "create_coarse_mesh",
    "Build a coarse quad layout from the domain's own boundary, line and point "
    "features -- via a topological-skeleton decomposition, not a read of "
    "hand-drawn patch curves. Needs a boundary already loaded: call rhino_pull "
    "(or load_mesh with walls) first. Point features become the POLES of the "
    "layout; line features are followed where the decomposition can manage it. "
    "Replaces any coarse layout already held and clears its undo history. This "
    "is step 1 of the basic workflow: read the result with coarse_inspect, set "
    "densities with coarse_set_density, then call coarse_densify.",
    properties={
        "target_length": {
            "type": "number",
            "description": "Background spacing the domain is triangulated at "
            "before decomposing -- NOT the quad size. Default is "
            "a small fraction of the bounding-box diagonal (the "
            "thesis rule). Too coarse a value and small features "
            "are missed entirely.",
        },
    },
    title="Create a coarse mesh",
)
def _t_create_coarse_mesh(session: MeshSession, target_length: float | None = None) -> dict[str, Any]:
    refusal = _needs_walls(session)
    if refusal:
        return refusal

    outer = _open_loop([list(p) for p in _polyline_points(session.walls[0])])
    inners = [_open_loop([list(p) for p in _polyline_points(wall)]) for wall in session.walls[1:]]
    guides = [[list(p) for p in _polyline_points(guide)] for guide in session.guides]
    points = [list(p) for p in session.points]

    try:
        decomposition = SkeletonDecomposition.from_boundary(
            outer, inner_boundaries=inners, polyline_features=guides, point_features=points, target_length=(float(target_length) if target_length is not None else None)
        )
        coarse = decomposition.coarse_mesh()
        # Freshly built, ``coarse`` carries no density data at all --
        # ``get_strip_density`` (which ``coarse_inspect`` and
        # ``densification`` both call) raises on a strip nobody has set one
        # for yet. Default every strip to 1, matching the default
        # ``CoarseQuadMesh.from_quad_mesh`` itself starts from.
        coarse.set_strips_density(1)
    except Exception as exc:
        return {"ok": False, "reason": "could not build a coarse layout: {}: {}".format(type(exc).__name__, exc)}

    session.coarse = coarse
    session.decomposition = decomposition
    session._coarse_undo = []
    poles = _pole_handles(coarse)
    strips = len(list(coarse.strips()))
    entry = session.record("create_coarse_mesh", faces=coarse.number_of_faces(), strips=strips, target_length=target_length)
    return {
        "ok": True,
        "step": entry["step"],
        "faces": coarse.number_of_faces(),
        "vertices": coarse.number_of_vertices(),
        "strips": strips,
        "poles": poles,
        "reading": "{} patch(es), {} strip(s), {} pole(s). Set strip densities next with coarse_set_density, then call coarse_densify.".format(
            coarse.number_of_faces(), strips, len(poles)
        ),
    }


@tool(
    "coarse_inspect",
    "Report on the coarse layout in hand: every strip with its skey, density "
    "and one representative coarse edge (two vertex handles) to address it by; "
    "every patch with a handle inside it, its dense pattern and the strips "
    "crossing it; pattern counts; and the poles. Read this before "
    "coarse_set_density, coarse_set_pattern or a strip edit -- neither a strip "
    "nor a patch has a handle of its own. SET image=true TO SEE THE LAYOUT: "
    "each strip is drawn in its own colour with its skey written on its "
    "representative edge, which is the only way to tell which band of edges a "
    "skey is before removing it.",
    properties={
        "image": {
            "type": "boolean",
            "description": "Draw the layout -- strips coloured and numbered, "
            "patterns marked, over the input walls, guides and "
            "point features. Off by default. Does NOT count as "
            "having looked at the DENSE mesh; only inspect does.",
        },
        "size": {"type": "integer", "description": "Image width and height in pixels. Default 800, max 1400."},
    },
    read_only=True,
    idempotent=True,
    title="Inspect the coarse layout",
)
def _t_coarse_inspect(session: MeshSession, image: bool = False, size: int | None = None) -> dict[str, Any]:
    refusal = _needs_coarse(session)
    if refusal:
        return refusal
    coarse = session.coarse
    strips = []
    for skey in sorted(coarse.strips()):
        u, v = render.strip_representative_edge(coarse, skey)
        strips.append(
            {
                "skey": skey,
                "density": coarse.get_strip_density(skey),
                "length": len(coarse.strip_edges(skey)),
                "edge": [vertex_handle(coarse.vertex_coordinates(u)), vertex_handle(coarse.vertex_coordinates(v))],
            }
        )
    patterns = _patterns(coarse)
    pseudo = getattr(coarse, "is_face_pseudo_quad", lambda fkey: False)
    faces = []
    for fkey in coarse.faces():
        try:
            crossing = sorted(set(coarse.face_strips(fkey)))
        except Exception:
            crossing = None
        faces.append({"at": _face_point(coarse, fkey), "pattern": patterns[fkey], "pseudo_quad": bool(pseudo(fkey)), "strips": crossing})
    payload = {
        "ok": True,
        "faces": coarse.number_of_faces(),
        "vertices": coarse.number_of_vertices(),
        "strips": strips,
        "patches": faces,
        "patterns": _pattern_counts(coarse),
        "poles": _pole_handles(coarse),
        "state": session.state(),
    }

    if image:
        size = render.DEFAULT_SIZE if size is None else int(size)
        try:
            layers, colors = render.coarse_scene(session)
            encoded = render.render_png_base64(session, width=size, height=size, layers=layers)
        except Exception as exc:
            payload["image_error"] = "could not draw the layout: {}: {}".format(type(exc).__name__, exc)
            encoded = None
        if encoded:
            # NOT session.mark_seen(): that gate is about the dense mesh a
            # delivery would ship, and this picture does not show it. It IS the
            # look rhino_push_coarse waits for.
            session.mark_coarse_seen()
            payload["image_legend"] = render.COARSE_LEGEND
            payload["strip_colors"] = dict((str(k), v) for k, v in colors.items())
            if len(colors) > len(render.STRIP_PALETTE):
                payload["image_caveat"] = "{} strips share {} colours, so two strips can look alike -- go by the numbers.".format(len(colors), len(render.STRIP_PALETTE))
            payload["_image"] = {"data": encoded, "mimeType": "image/png"}
    return payload


@tool(
    "coarse_set_density",
    "Set how many dense rows a strip -- or every strip -- densifies into. One "
    'selector covers every case: {"kind":"all","value":d} sets every strip to '
    'd; {"kind":"strip","skey":s,"value":d} sets one strip (skey from '
    'coarse_inspect); {"kind":"strips","skeys":[...],"value":d} sets several; '
    '{"kind":"target_length","value":t} sets every strip so its edges average '
    'about length t (add "skeys" to limit this to some strips); '
    '{"kind":"face_count","value":n} sets one density for every strip, aiming '
    "at about n dense faces in total. Snapshots the coarse layout first -- "
    "undo with coarse_undo.",
    properties={
        "density": {"type": "object", "description": 'One of the five shapes above; always needs "kind" and "value".'},
    },
    required=("density",),
    title="Set strip density",
)
def _t_coarse_set_density(session: MeshSession, density: dict[str, Any]) -> dict[str, Any]:
    refusal = _needs_coarse(session)
    if refusal:
        return refusal
    if not isinstance(density, dict):
        return {"ok": False, "reason": 'density must be an object, e.g. {"kind": "all", "value": 4}'}
    kind = density.get("kind")
    value = density.get("value")
    if kind not in ("all", "strip", "strips", "target_length", "face_count"):
        return {"ok": False, "reason": "unknown density kind {!r}; try 'all', 'strip', 'strips', 'target_length' or 'face_count'".format(kind)}
    if value is None:
        return {"ok": False, "reason": "density needs a 'value'"}
    if kind == "strip" and "skey" not in density:
        return {"ok": False, "reason": "kind 'strip' needs a 'skey'"}
    if kind == "strips" and not density.get("skeys"):
        return {"ok": False, "reason": "kind 'strips' needs a non-empty 'skeys'"}

    coarse = session.coarse
    # ``set_strip_density`` stores ANY key without complaint, so a mistyped
    # skey would add a density for a strip that does not exist and report ok.
    known = set(coarse.strips())
    asked = [density["skey"]] if kind == "strip" else list(density.get("skeys") or [])
    unknown = [s for s in asked if s not in known]
    if unknown:
        return {"ok": False, "reason": "no strip(s) {} -- skeys come from coarse_inspect".format(", ".join(repr(s) for s in unknown)), "skeys": sorted(known)}
    try:
        number = float(value)
    except (TypeError, ValueError):
        return {"ok": False, "reason": "'value' must be a number, got {!r}".format(value)}
    if kind == "target_length":
        if number <= 0:
            return {"ok": False, "reason": "a target_length must be above 0"}
    elif kind == "face_count":
        faces = coarse.number_of_faces()
        if number < faces:
            return {"ok": False, "reason": "face_count {} is below the {} patches the layout already has -- every strip needs a density of at least 1".format(int(number), faces)}
    elif int(number) < 1:
        return {"ok": False, "reason": "a density must be at least 1"}
    session.snapshot_coarse("before coarse_set_density")
    try:
        if kind == "all":
            coarse.set_strips_density(int(value))
        elif kind == "strip":
            coarse.set_strip_density(density["skey"], int(value))
        elif kind == "strips":
            coarse.set_strips_density(int(value), skeys=list(density["skeys"]))
        elif kind == "target_length":
            skeys = density.get("skeys")
            if skeys is None:
                coarse.set_strips_density_target(float(value))
            else:
                for skey in skeys:
                    coarse.set_strip_density_target(skey, float(value))
        else:  # face_count
            coarse.set_mesh_density_face_target(int(value))
    except Exception as exc:
        session.undo_coarse()
        return {"ok": False, "reason": "could not set density: {}: {}".format(type(exc).__name__, exc)}
    entry = session.record("coarse_set_density", density=density)
    return {"ok": True, "step": entry["step"], "densities": coarse.get_strip_densities(), "undo_available": session.can_undo_coarse()}


@tool(
    "coarse_set_pattern",
    'Choose how patches densify: "ortho" (the plain grid, the default), '
    '"diagonal" (the grid with both patch diagonals cut through it, so the '
    'dense mesh gains triangles along them) or "fan" (four polar fans meeting '
    "at the patch centre; on a triangular patch, three -- one per corner, the "
    'pole included). Select the patches with faces: {"kind":"all"}; '
    '{"kind":"strip","skey":s} -- every patch the strip crosses; or '
    '{"kind":"at","handles":[...]} -- handles from coarse_inspect\'s patches. '
    "COSTS DENSITY: diagonal and fan need both strips crossing a patch at the "
    "same EVEN density, and a strip runs on through other patches, so whole "
    "groups of strips are raised to the largest density among them, rounded up "
    "to even. That is applied here and returned as density_raised, so it is "
    "never a surprise at coarse_densify. Snapshots first -- undo with "
    "coarse_undo.",
    properties={
        "pattern": {"type": "string", "enum": list(PATTERNS)},
        "faces": {"type": "object", "description": 'Which patches. One of the three shapes above. Default {"kind":"all"}.'},
    },
    required=("pattern",),
    title="Set patch pattern",
)
def _t_coarse_set_pattern(session: MeshSession, pattern: str, faces: dict[str, Any] | None = None) -> dict[str, Any]:
    refusal = _needs_coarse(session)
    if refusal:
        return refusal
    if pattern not in PATTERNS:
        return {"ok": False, "reason": "unknown pattern {!r}; pick one of {}".format(pattern, ", ".join(PATTERNS))}
    coarse = session.coarse
    if pattern != "ortho" and not hasattr(coarse, "quad_mesh"):
        return {
            "ok": False,
            "reason": "this layout is a {}, which only densifies as ortho -- patterns need a CoarsePseudoQuadMesh, which is what create_coarse_mesh builds".format(
                type(coarse).__name__
            ),
        }

    selector = faces or {"kind": "all"}
    kind = selector.get("kind") if isinstance(selector, dict) else None
    if kind == "all":
        chosen = list(coarse.faces())
    elif kind == "strip":
        if selector.get("skey") not in set(coarse.strips()):
            return {"ok": False, "reason": "no strip {!r} -- skeys come from coarse_inspect".format(selector.get("skey"))}
        chosen = sorted(set(coarse.strip_faces(selector["skey"])))
    elif kind == "at":
        chosen, missing = [], []
        for handle in selector.get("handles") or []:
            fkey, reason = _face_at(coarse, handle)
            if fkey is None:
                missing.append(reason)
            else:
                chosen.append(fkey)
        if missing or not chosen:
            return {"ok": False, "reason": "could not place every handle in a patch: " + ("; ".join(missing[:3]) or "'at' needs 'handles'")}
    else:
        return {"ok": False, "reason": "faces must be {'kind': 'all'}, {'kind': 'strip', 'skey': s} or {'kind': 'at', 'handles': [...]}"}

    session.snapshot_coarse("before coarse_set_pattern")
    patterns = _patterns(coarse)
    for fkey in chosen:
        patterns[fkey] = pattern
    try:
        changes = reconcile_strip_densities(coarse, patterns)
    except Exception as exc:
        session.undo_coarse()
        return {"ok": False, "reason": "the layout cannot take that pattern: {}: {}".format(type(exc).__name__, exc)}
    entry = session.record("coarse_set_pattern", pattern=pattern, faces=selector, patches=len(chosen))
    payload = {
        "ok": True,
        "step": entry["step"],
        "set": len(chosen),
        "patterns": _pattern_counts(coarse),
        "density_raised": _raised(changes),
        "undo_available": session.can_undo_coarse(),
    }
    if changes:
        payload["note"] = (
            "{} strip density(ies) were raised so the pattern "
            "can tile -- see density_raised. A later "
            "coarse_set_density to an odd or unequal value is "
            "raised again at coarse_densify.".format(len(changes))
        )
    return payload


@tool(
    "coarse_densify",
    "Generate the dense quad mesh from the coarse layout and its strip "
    "densities, and load it as the working mesh -- replacing whatever "
    "inspect/relax/smooth_* were operating on and clearing the DENSE mesh undo "
    "history (the coarse layout and ITS undo are untouched). Curved boundaries "
    "keep their shape automatically when the layout came from "
    "create_coarse_mesh or coarse_load. Honours each patch's pattern "
    "(coarse_set_pattern); if a density set since then no longer suits a "
    "diagonal or fan patch it is raised first, reported as density_raised and "
    "undoable with coarse_undo. Does not solve a frame field: patch interiors "
    "are blended from their four sides, same as compas_singular has always "
    "done without one.",
    title="Densify the coarse layout",
)
def _t_coarse_densify(session: MeshSession) -> dict[str, Any]:
    refusal = _needs_coarse(session)
    if refusal:
        return refusal
    coarse = session.coarse
    edges_to_curves = _edges_to_curves(session)
    changes = {}
    try:
        if hasattr(coarse, "quad_mesh"):
            patterns = _patterns(coarse)
            # Find out on a copy first, so the layout is snapshotted only when
            # densifying is actually going to change it.
            if reconcile_strip_densities(coarse.copy(), dict(patterns)):
                session.snapshot_coarse("before coarse_densify raised densities")
                changes = reconcile_strip_densities(coarse, patterns)
            # ``quad_mesh`` is bit-identical to ``densification`` on an
            # all-ortho layout (test_tools_coarse checks it), and the only one
            # of the two that reads the patterns.
            dense = coarse.quad_mesh(overwrite_edges_to_curves=edges_to_curves)
        else:
            dense = coarse.densification(overwrite_edges_to_curves=edges_to_curves)
    except Exception as exc:
        return {"ok": False, "reason": "densification failed: {}: {}".format(type(exc).__name__, exc)}

    session.adopt(dense, source={"kind": "coarse_densify", "coarse_faces": coarse.number_of_faces()})
    metrics = session.quality()
    told = describe(session.mesh, metrics, thresholds())
    session.record("coarse_densify", before=None, after=metrics, coarse_faces=coarse.number_of_faces(), density_raised=_raised(changes))
    payload = {
        "ok": True,
        "vertices": dense.number_of_vertices(),
        "faces": dense.number_of_faces(),
        "patterns": (_pattern_counts(coarse) if hasattr(coarse, "quad_mesh") else None),
        "density_raised": _raised(changes),
        "quality": metrics,
        "reading": told["reading"],
        "verdict": told["verdict"],
    }
    if changes:
        payload["note"] = "{} strip density(ies) were raised first so the diagonal/fan patches could tile; coarse_undo puts them back.".format(len(changes))
    counts = payload["patterns"] or {}
    if counts.get("diagonal") or counts.get("fan"):
        # Measured on an L-shaped domain at density 4: ortho reads usable (min
        # 49.5), all-diagonal poor (min 16.7), all-fan unusable (min 8.5, max
        # 170). The patterns put those angles there on purpose.
        payload["pattern_caveat"] = (
            "This mesh uses diagonal and/or fan patterns, and the quality "
            "reading does not know that. Their triangles and fan poles have "
            "small and large corner angles BY DESIGN -- on a test domain an "
            'all-fan layout reads "unusable" where the same layout in ortho '
            'reads "usable". Do not smooth to chase those numbers: smoothing '
            "cannot remove a pattern's own angles, only distort the patches "
            "around them. Judge the ortho patches by the numbers, and the "
            "patterned ones by eye with inspect image=true."
        )
    return payload


# ==============================================================================
# editing -- strips only
# ==============================================================================


@tool(
    "coarse_plan_strip_deletion",
    "Preview what removing a strip would cost, without changing anything: "
    "face count before/after, how many OTHER strips it would take with it "
    "(collateral), whether it collapses a hole in the domain "
    "(boundaries_lost), and ok/reason -- ok false means coarse_remove_strip "
    "will refuse for the same reason. There is no cheaper way to know whether "
    "a strip can safely go: the only real test is performing the deletion on "
    "a copy and looking, which is exactly what this does.",
    properties={
        "edge": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
            "description": "Two vertex handles naming a coarse edge that belongs to the strip. Use this or skey.",
        },
        "skey": {"type": "integer", "description": "A strip key, from coarse_inspect. Use this or edge."},
        "preserve_boundaries": {"type": "boolean", "description": "Split a strip first if needed, rather than let this deletion collapse a hole. Default false."},
    },
    read_only=True,
    title="Plan a strip deletion",
)
def _t_coarse_plan_strip_deletion(session: MeshSession, edge: Sequence[str] | None = None, skey: Any = None, preserve_boundaries: bool = False) -> dict[str, Any]:
    refusal = _needs_coarse(session)
    if refusal:
        return refusal
    if edge is None and skey is None:
        return {"ok": False, "reason": "give either an edge (two vertex handles) or a skey"}
    edge_keys = None
    if edge is not None:
        edge_keys, err = _resolve_edge(session.coarse, edge)
        if err:
            return {"ok": False, "reason": err}

    editor = _editor(session)
    info = editor.plan_strip_deletion(edge=edge_keys, skey=skey, preserve_boundaries=bool(preserve_boundaries))
    info["ok"] = bool(info.get("ok"))
    return info


@tool(
    "coarse_remove_strip",
    "Delete the strip through an edge (or skey), collapsing that band of "
    "patches and welding its two sides together. This does NOT dissolve the "
    "picked line and merge the patches either side of it -- the strip runs "
    "the full width of the layout, so the whole band goes. Some strips "
    "cannot be removed at all, and there is no cheap predictor for which -- "
    "call coarse_plan_strip_deletion first if that matters. Snapshots the "
    "coarse layout first -- undo with coarse_undo.",
    properties={
        "edge": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
            "description": "Two vertex handles naming a coarse edge that belongs to the strip. Use this or skey.",
        },
        "skey": {"type": "integer", "description": "A strip key, from coarse_inspect. Use this or edge."},
        "preserve_boundaries": {"type": "boolean", "description": "Split a strip first if needed, rather than let this deletion collapse a hole. Default false."},
    },
    destructive=True,
    title="Remove a strip",
)
def _t_coarse_remove_strip(session: MeshSession, edge: Sequence[str] | None = None, skey: Any = None, preserve_boundaries: bool = False) -> dict[str, Any]:
    refusal = _needs_coarse(session)
    if refusal:
        return refusal
    if edge is None and skey is None:
        return {"ok": False, "reason": "give either an edge (two vertex handles) or a skey"}
    edge_keys = None
    if edge is not None:
        edge_keys, err = _resolve_edge(session.coarse, edge)
        if err:
            return {"ok": False, "reason": err}

    session.snapshot_coarse("before coarse_remove_strip")
    editor = _editor(session)
    ok, notes = editor.remove_strip(edge=edge_keys, skey=skey, preserve_boundaries=bool(preserve_boundaries))
    if not ok:
        session.undo_coarse()
        return {"ok": False, "reason": notes.get("error", str(notes))}
    layout, commit_notes = editor.commit()
    if layout is None:
        session.undo_coarse()
        return {"ok": False, "reason": commit_notes.get("error", "commit refused the edit")}
    entry = session.record("coarse_remove_strip", edge=edge, skey=skey, faces_out=layout.number_of_faces())
    return {
        "ok": True,
        "step": entry["step"],
        "faces": layout.number_of_faces(),
        "strips": len(list(layout.strips())),
        "notes": notes,
        "undo_available": session.can_undo_coarse(),
    }


@tool(
    "coarse_add_strip",
    "Add a strip to the coarse layout by unzipping a chain of at least three "
    "EXISTING corners into two -- the strip runs the band between them. The "
    "chain must run wall to wall, or close on itself. A corner has no handle "
    "of its own until you name one from an edge, so read coarse_inspect "
    "first. Strip labels and densities elsewhere survive this. Snapshots the "
    "coarse layout first -- undo with coarse_undo.",
    properties={
        "polyedge": {"type": "array", "items": {"type": "string"}, "minItems": 3, "description": "At least 3 vertex handles, in order, each joined to the next by a coarse edge."},
    },
    required=("polyedge",),
    title="Add a strip",
)
def _t_coarse_add_strip(session: MeshSession, polyedge: Sequence[str]) -> dict[str, Any]:
    refusal = _needs_coarse(session)
    if refusal:
        return refusal
    if len(polyedge) < 3:
        return {"ok": False, "reason": "a polyedge needs at least 3 vertex handles"}
    keys, missing = [], []
    for item in polyedge:
        key, how = resolve(session.coarse, item)
        if key is None:
            missing.append(item)
        else:
            keys.append(key)
    if missing:
        return {"ok": False, "reason": "could not place {} of {} handles on the coarse layout: {}".format(len(missing), len(polyedge), ", ".join(missing[:3]))}

    session.snapshot_coarse("before coarse_add_strip")
    editor = _editor(session)
    ok, notes = editor.add_strip(keys)
    if not ok:
        session.undo_coarse()
        return {"ok": False, "reason": notes.get("error", str(notes))}
    layout, commit_notes = editor.commit()
    if layout is None:
        session.undo_coarse()
        return {"ok": False, "reason": commit_notes.get("error", "commit refused the edit")}
    entry = session.record("coarse_add_strip", polyedge=list(polyedge), faces_out=layout.number_of_faces())
    return {
        "ok": True,
        "step": entry["step"],
        "faces": layout.number_of_faces(),
        "strips": len(list(layout.strips())),
        "notes": notes,
        "undo_available": session.can_undo_coarse(),
    }


@tool(
    "coarse_divide",
    "Split a band of patches in two, adding a strip. Two ways: {skey, t} "
    "splits every patch of that strip at parameter t (0.5 = down the middle); "
    "{points} cuts along a drawn line or polyline, and an arc sampled as a "
    "polyline stays an arc in the dense mesh. A cut must cross each patch "
    "through OPPOSITE sides (through adjacent sides it would slice a triangle "
    "off, and is refused) and cannot cross a patch with a pole. Its ends are "
    "snapped onto the nearest coarse edge within a quarter of the mean edge "
    "length -- snapped_by reports how far. A cut that stops on an interior "
    "edge is continued along its strip to the wall when extend is true (the "
    "default). EVERY strip key is renumbered by this: densities are carried "
    "over by position -- a divided strip's density is shared between its two "
    "halves -- and patterns by containment, but re-read coarse_inspect before "
    "using a skey again. Snapshots first -- undo with coarse_undo.",
    properties={
        "skey": {"type": "integer", "description": "The strip to divide, from coarse_inspect. Use this or points."},
        "t": {"type": "number", "description": "Where across the strip to split, strictly between 0 and 1. Default 0.5. Only with skey."},
        "points": {
            "type": "array",
            "minItems": 2,
            "items": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3},
            "description": "The cut as [x, y, z] points, first and last on (or near) a coarse edge. Use this or skey.",
        },
        "extend": {"type": "boolean", "description": "Continue a cut that stops on an interior edge along its strip to the wall. Default true; false refuses such a cut instead."},
    },
    title="Divide a strip",
)
def _t_coarse_divide(session: MeshSession, skey: Any = None, t: float | None = None, points: Sequence[Sequence[float]] | None = None, extend: bool = True) -> dict[str, Any]:
    refusal = _needs_coarse(session)
    if refusal:
        return refusal
    if (skey is None) == (points is None):
        return {"ok": False, "reason": "give exactly one of skey (divide that strip) or points (cut along a drawn curve)"}
    if t is not None and skey is None:
        return {"ok": False, "reason": "t only applies with skey"}

    before = session.coarse.copy()
    faces_before = before.number_of_faces()
    shapes = list((_edges_to_curves(session) or {}).values())
    editor = _editor(session, shapes=shapes)

    snapped = None
    if points is not None:
        cut = [[float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0] for p in points]
        tol = 0.25 * editor.mean_edge()
        snapped = []
        for index in (0, -1):
            where, distance = _snap_end(editor, cut[index], tol)
            if where is None:
                return {
                    "ok": False,
                    "reason": "the cut's {} point, {}, is {} away from "
                    "every coarse edge, further than the {:.3f} "
                    "snap allows -- start and end it on a coarse "
                    "edge (coarse_inspect image=true shows "
                    "them)".format("first" if index == 0 else "last", vertex_handle(cut[index]), "nowhere near" if distance is None else "{:.3f}".format(distance), tol),
                }
            cut[index] = where
            snapped.append(round(distance, 6))

    session.snapshot_coarse("before coarse_divide")
    if points is not None:
        ok, notes = editor.divide(cut, extend=bool(extend))
    else:
        ok, notes = editor.divide(skey=skey, t=0.5 if t is None else float(t))
    if not ok:
        session.undo_coarse()
        return {"ok": False, "reason": editor.last_reason, "snapped_by": snapped}
    seeded = set(editor._snapshot_curves)
    new_shapes = [curve for key, curve in editor.curves.items() if key not in seeded]
    mean_edge = editor.mean_edge()
    layout, commit_notes = editor.commit()
    if layout is None:
        session.undo_coarse()
        return {"ok": False, "reason": commit_notes.get("error", "commit refused the cut")}

    layout.collect_strips()
    carried = _carry_densities(before, layout)
    _carry_patterns(before, layout)
    raised = reconcile_strip_densities(layout, _patterns(layout))

    cut_notes = dict(notes)
    entry = session.record("coarse_divide", skey=skey, t=t, points=len(points) if points else None, extend=bool(extend), faces_out=layout.number_of_faces())
    strips = []
    for new in sorted(layout.strips()):
        u, v = render.strip_representative_edge(layout, new)
        strips.append(
            {
                "skey": new,
                "density": layout.get_strip_density(new),
                "from_skey": carried.get(new, (None, None))[0],
                "edge": [vertex_handle(layout.vertex_coordinates(u)), vertex_handle(layout.vertex_coordinates(v))],
            }
        )
    payload = {
        "ok": True,
        "step": entry["step"],
        "faces_before": faces_before,
        "faces": layout.number_of_faces(),
        "divided_skey": cut_notes.get("skey"),
        "patches_split": cut_notes.get("faces"),
        "strips": strips,
        "patterns": _pattern_counts(layout),
        "density_raised": _raised(raised),
        "shaped_edges": len(_user_curves(layout)),
        "lost_curves": commit_notes.get("lost_curves", 0),
        "undo_available": session.can_undo_coarse(),
        "note": "strip keys were renumbered -- from_skey says where each came from; divided_skey is the OLD key of the strip that was split.",
    }
    if snapped is not None:
        payload["snapped_by"] = snapped
    pinches = _pinches(layout, new_shapes, 0.1 * mean_edge)
    if pinches:
        payload["warning"] = (
            "the cut passes within {} of an existing coarse edge at {} -- the "
            "strip between them is a sliver and will densify into slivers at "
            "any density. Look with coarse_inspect image=true; coarse_undo and "
            "cut further away if that is not intended.".format(pinches[0][1], pinches[0][0])
        )
        payload["pinches"] = pinches
    return payload


@tool(
    "coarse_move_corner",
    "Move one corner of the coarse layout. An INTERIOR corner goes exactly "
    "where it is told. A BOUNDARY corner is projected back onto its wall so the "
    "outline is not eaten -- dragged far enough that the nearest wall is a "
    "different one (a hole, say), it is refused rather than moved onto it. "
    "Refused, too, if the corner would be dragged over a neighbouring patch or "
    "turn one of its own inside out. Topology, strip keys, densities and "
    "patterns are all unchanged. The shapes of the edges at the corner are "
    "warped onto its new position, so a curved edge stays curved. Snapshots "
    "first -- undo with coarse_undo.",
    properties={
        "corner": {"type": "string", "description": "The corner as a vertex handle -- from coarse_inspect (a strip edge, a pole) or the picture."},
        "to": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 3, "description": "Where to put it, as [x, y] or [x, y, z]."},
        "project": {
            "type": "boolean",
            "description": "Hold a boundary corner on its wall. Default true; turn off only deliberately -- off, the dense boundary leaves the wall there.",
        },
    },
    required=("corner", "to"),
    title="Move a coarse corner",
)
def _t_coarse_move_corner(session: MeshSession, corner: str, to: Sequence[float], project: bool = True) -> dict[str, Any]:
    refusal = _needs_coarse(session)
    if refusal:
        return refusal
    coarse = session.coarse
    vkey, how = resolve(coarse, corner)
    if vkey is None:
        return {"ok": False, "reason": "could not place {!r} on the coarse layout: {}".format(corner, how)}
    target = [float(to[0]), float(to[1]), 0.0]
    boundary = coarse.is_vertex_on_boundary(vkey)

    # The shapes of the corner's edges as they are now, to carry across. Read
    # BEFORE the move: afterwards nothing anchors them to this corner any more.
    mapping = _edges_to_curves(session) or {}
    old = list(coarse.vertex_coordinates(vkey))  # the read may snap it onto its wall
    shapes = {}
    for n in coarse.vertex_neighbors(vkey):
        if coarse.is_edge_on_boundary((vkey, n)):
            continue  # the wall arc is recomputed exactly
        curve = mapping.get((vkey, n))
        if curve is None and (n, vkey) in mapping:
            curve = list(reversed(mapping[(n, vkey)]))
        if curve is not None and len(curve) > 2:
            shapes[n] = curve

    editor = _editor(session)
    ok, notes = editor.move_vertex(vkey, target, project=bool(project))
    if not ok:
        return {"ok": False, "reason": editor.last_reason}
    moved = editor.mesh.vertex_coordinates(vkey)
    if boundary and project and session.walls:
        before_wall, _ = _nearest_wall(session.walls, old)
        after_wall, _ = _nearest_wall(session.walls, moved)
        if before_wall != after_wall:
            return {
                "ok": False,
                "reason": "that corner lies on wall {} and would be projected onto wall {} -- a boundary corner can slide along its own wall only; nothing was moved".format(
                    before_wall, after_wall
                ),
            }
    fold = _fold(editor.mesh, vkey)
    if fold:
        return {"ok": False, "reason": fold + "; nothing was moved"}

    session.snapshot_coarse("before coarse_move_corner")
    layout, commit_notes = editor.commit()
    if layout is None:
        session.undo_coarse()
        return {"ok": False, "reason": commit_notes.get("error", "commit refused the move")}

    # Re-anchor the interior shapes at the corner, and drop the stored ones
    # that still end where it used to be.
    here = layout.vertex_coordinates(vkey)
    kept = [c for c in _user_curves(layout) if not (_close(c[0], old, 1e-3) or _close(c[-1], old, 1e-3))]
    warped, straightened = 0, 0
    for n, curve in shapes.items():
        new = warp_polyline(curve, here, layout.vertex_coordinates(n))
        if new is None:
            straightened += 1
        else:
            kept.append(new)
            warped += 1
    layout.attributes["user_curves"] = kept

    entry = session.record("coarse_move_corner", corner=corner, to=target, project=bool(project))
    payload = {
        "ok": True,
        "step": entry["step"],
        "from": vertex_handle(old),
        "to": vertex_handle(here),
        "projected": bool(notes.get("projected")),
        "distance": round(_length(old, here), 6),
        "shapes_warped": warped,
        "shapes_straightened": straightened,
        "undo_available": session.can_undo_coarse(),
    }
    if boundary and project and _length(here, target) > 1e-6:
        payload["note"] = "a boundary corner stays on its wall, so it went to the nearest wall point to the one asked for"
    if boundary and _wall_kink(session.walls, old):
        payload["warning"] = (
            "this corner sat on a corner OF THE DOMAIN. The "
            "patch now wraps round that kink, which densifies "
            "into a flattened element there -- coarse_undo "
            "unless that is the intent."
        )
    return payload


@tool(
    "coarse_undo",
    "Put the coarse layout back to its last snapshot -- taken automatically "
    "before coarse_set_density, coarse_set_pattern, coarse_add_strip, "
    "coarse_remove_strip, coarse_divide, coarse_move_corner, and a "
    "coarse_densify that had to raise densities -- "
    "and mark the coarse steps taken since as undone in history. Independent "
    "of undo, which only ever touches the dense mesh.",
    destructive=True,
    title="Undo the last coarse edit",
)
def _t_coarse_undo(session: MeshSession) -> dict[str, Any]:
    ok, detail = session.undo_coarse()
    if not ok:
        return {"ok": False, "reason": detail}
    coarse = session.coarse
    return {"ok": True, "restored": detail, "faces": coarse.number_of_faces(), "strips": len(list(coarse.strips())), "state": session.state()}


# ==============================================================================
# keeping a layout -- across a restart, or for another session
# ==============================================================================

#: What a saved layout file says it is, so ``coarse_load`` can refuse a plain
#: mesh file rather than misread one.
LAYOUT_FORMAT = "compas_singular.mcp/coarse_layout"
LAYOUT_VERSION = 1


def _plain_curves(curves: Sequence[Any]) -> list[list[list[float]]]:
    return [[list(p) for p in _polyline_points(curve)] for curve in curves]


@tool(
    "coarse_save",
    "Write the coarse layout to a json file WITH its domain -- walls, guides "
    "and point features -- and the shape of every edge, so coarse_load gives "
    "back a layout that densifies exactly as this one does, curved boundaries "
    "and patterns included, in a later session or after a restart. Strip "
    "densities and patterns are kept; the undo history is not. Overwrites a "
    "file already at that path.",
    properties={
        "path": {"type": "string", "description": "Where to write the .json."},
    },
    required=("path",),
    open_world=True,
    title="Save the coarse layout",
)
def _t_coarse_save(session: MeshSession, path: str) -> dict[str, Any]:
    refusal = _needs_coarse(session)
    if refusal:
        return refusal
    folder = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(folder):
        return {"ok": False, "reason": "no directory at {}".format(folder)}
    import compas

    edges_to_curves = _edges_to_curves(session)
    layout = session.coarse.copy()
    if edges_to_curves and hasattr(layout, "set_edges_to_curves"):
        # The decomposition that could recompute these is not saved, so the
        # file has to carry them or a curved wall reloads as chords.
        layout.set_edges_to_curves(edges_to_curves)
    # A densified layout holds its dense mesh as an attribute. That is not part
    # of the layout, and would multiply the file size for nothing.
    layout.attributes["quad_mesh"] = None
    layout.attributes["polygonal_mesh"] = None
    _patterns(layout)

    record = {
        "format": LAYOUT_FORMAT,
        "version": LAYOUT_VERSION,
        "coarse": layout,
        "walls": _plain_curves(session.walls),
        "guides": _plain_curves(session.guides),
        "points": [list(p) for p in session.points],
        "source": session.source,
    }
    try:
        compas.json_dump(record, path)
    except Exception as exc:
        return {"ok": False, "reason": "could not write {}: {}: {}".format(path, type(exc).__name__, exc)}
    return {
        "ok": True,
        "path": path,
        "faces": layout.number_of_faces(),
        "strips": len(list(layout.strips())),
        "patterns": _pattern_counts(layout),
        "curved_edges": len(edges_to_curves or {}),
        "walls": len(session.walls),
    }


@tool(
    "coarse_load",
    "Read a layout written by coarse_save, with its walls, guides and point "
    "features, replacing the coarse layout held and clearing its undo. If the "
    "file's domain differs from the one in this session, it REPLACES that "
    "domain and the dense mesh is dropped too, since it belongs to the other "
    "domain -- dense_mesh_dropped says so. Then continue as after "
    "create_coarse_mesh: coarse_inspect, edits, coarse_densify.",
    properties={
        "path": {"type": "string", "description": "A .json written by coarse_save."},
    },
    required=("path",),
    open_world=True,
    title="Load a coarse layout",
)
def _t_coarse_load(session: MeshSession, path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {"ok": False, "reason": "no file at {}".format(path)}
    import compas

    try:
        record = compas.json_load(path)
    except Exception as exc:
        return {"ok": False, "reason": "could not read {}: {}: {}".format(path, type(exc).__name__, exc)}
    if not isinstance(record, dict) or record.get("format") != LAYOUT_FORMAT:
        return {"ok": False, "reason": "{} is not a layout written by coarse_save -- for a plain mesh file use load_mesh".format(path)}
    coarse = record.get("coarse")
    if not hasattr(coarse, "strips") or not coarse.number_of_faces():
        return {"ok": False, "reason": "{} holds no coarse layout".format(path)}
    if not list(coarse.strips()):
        coarse.collect_strips()
    densities = coarse.get_strip_densities()
    for skey in coarse.strips():
        densities.setdefault(skey, 1)
    # The saved shapes are keyed by vertex, and the first strip edit or cut
    # clears them. Kept again by geometry, they outlive that: the curved ones
    # join the user curves, which every edit carries and densifying lays over.
    if hasattr(coarse, "edges_to_curves"):
        known = set((round(c[0][0], 3), round(c[0][1], 3), round(c[-1][0], 3), round(c[-1][1], 3)) for c in _user_curves(coarse))
        extra = [
            [list(p) for p in curve]
            for curve in coarse.edges_to_curves().values()
            if len(curve) > 2 and (round(curve[0][0], 3), round(curve[0][1], 3), round(curve[-1][0], 3), round(curve[-1][1], 3)) not in known
        ]
        if extra:
            coarse.attributes["user_curves"] = _user_curves(coarse) + extra

    walls = record.get("walls") or []
    guides = record.get("guides") or []
    points = record.get("points") or []
    domain = ([[list(p) for p in w] for w in walls], [[list(p) for p in g] for g in guides], [list(p) for p in points])
    dropped = False
    if domain != session._domain():
        # A different domain: the dense mesh in hand was made for the old one.
        dropped = session.loaded
        session.adopt(None, walls=walls, guides=guides, points=points, source={"kind": "coarse_load", "path": path})
    session.coarse = coarse
    session.decomposition = None
    session._coarse_undo = []
    entry = session.record("coarse_load", path=path)
    return {
        "ok": True,
        "step": entry["step"],
        "path": path,
        "faces": coarse.number_of_faces(),
        "strips": len(list(coarse.strips())),
        "patterns": _pattern_counts(coarse),
        "poles": _pole_handles(coarse),
        "walls": len(session.walls),
        "guides": len(session.guides),
        "point_features": len(session.points),
        "dense_mesh_dropped": dropped,
    }
