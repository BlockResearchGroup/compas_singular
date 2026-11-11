"""Rebuild a hand-edited coarse layout (mesh or one closed polyline per patch) into a coarse mesh.

Welds corners, snaps boundary corners to the walls, repairs to all-quad; the field is not re-solved.
Design notes: ``design_notes/editing.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas.geometry import Polyline
from compas.geometry import distance_point_point
from compas.itertools import pairwise
from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.editing.repair import solve_non_quad_faces

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = ["coarse_from_skeleton", "warp_polyline", "face_polylines", "faces_from_geometry", "mesh_from_faces", "snap_to_loops", "closest_on_loop", "PRECISION"]


#: Decimals the corner weld rounds to. Matches ``_mesh_from_faces`` and is the
#: resolution ``from_polylines`` matches polyline endpoints at, so a layout that
#: went out through one route and came back through the other welds the same.
PRECISION = 3


# ------------------------------------------------------------------
# reading what Rhino handed back
# ------------------------------------------------------------------


def _is_mesh(thing: Any) -> bool:
    """Duck-typed: a compas mesh of any class, including the pseudo-quad ones."""
    return all(hasattr(thing, name) for name in ("vertices", "faces", "vertex_coordinates", "face_vertices"))


def _as_points(thing: Any) -> list[list[float]]:
    """A polyline, compas ``Polyline`` or point list as a point list, without a closing repeat."""
    points = [list(p)[:3] for p in getattr(thing, "points", thing)]
    points = [p + [0.0] * (3 - len(p)) for p in points]
    if len(points) > 1 and distance_point_point(points[0], points[-1]) < 1e-9:
        points = points[:-1]
    return points


def faces_from_geometry(geometry: Any) -> list[list[list[float]]]:
    """``[[corner, corner, ...], ...]`` from a mesh, closed polylines, or faces already given as point lists."""
    if _is_mesh(geometry):
        return [[list(geometry.vertex_coordinates(v)) for v in geometry.face_vertices(f)] for f in geometry.faces()]

    faces = []
    for item in geometry:
        points = _as_points(item)
        if len(points) >= 3:
            faces.append(points)
    return faces


# ------------------------------------------------------------------
# putting an edited corner back on the wall
# ------------------------------------------------------------------


def closest_on_loop(point: list[float], loop: list[list[float]]) -> tuple[float, list[float] | None]:
    """``(distance, projected point)`` for the closest point of a closed loop.

    The loop is treated as closed whether or not its last point repeats its
    first, matching every other loop consumer in this package.
    """
    best_d, best_p = float("inf"), None
    ring = list(loop) + list(loop[:1])
    for a, b in pairwise(ring):
        abx, aby = b[0] - a[0], b[1] - a[1]
        length2 = abx * abx + aby * aby
        if length2 == 0.0:
            continue
        t = ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby) / length2
        t = max(0.0, min(1.0, t))
        q = [a[0] + abx * t, a[1] + aby * t, 0.0]
        d = distance_point_point(point, q)
        if d < best_d:
            best_d, best_p = d, q
    return best_d, best_p


def snap_to_loops(mesh: "QuadMesh", loops: list[list[list[float]]], tol: float) -> tuple[int, int]:
    """Project the mesh's own boundary corners onto the domain walls; interior corners never move.

    Returns ``(moved, left_off_wall)``; the second should be 0.

    Returns
    -------
    (int, int)
        How many boundary corners were moved, and how many were further from
        every loop than ``tol`` and so left where they were. The second number
        should be 0; anything else means a patch is hanging off the domain.
    """
    if not loops or tol <= 0.0:
        return 0, 0

    moved = off = 0
    # ``vertices_on_boundarIES`` -- PLURAL. The singular form returns only the
    # LONGEST boundary, so every corner of every HOLE was reported as interior and
    # silently skipped: measured on an annulus, 12 of 16 boundary vertices seen and
    # all four of the hole's missed. It went unnoticed because the editor's own
    # preview used the same singular form, so the two agreed with each other while
    # both being wrong -- and because a generated layout already has its corners on
    # the walls, which is why no baseline row moves either way. It bites exactly
    # when a user DRAGS a hole corner, which is the case this seam exists for.
    for vkey in set(v for ring in mesh.vertices_on_boundaries() for v in ring):
        point = mesh.vertex_coordinates(vkey)
        best_d, best_p = float("inf"), None
        for loop in loops:
            d, q = closest_on_loop(point, loop)
            if d < best_d:
                best_d, best_p = d, q
        if best_p is None or best_d <= 1e-9:
            continue
        if best_d > tol:
            off += 1
            continue
        mesh.vertex_attributes(vkey, "xyz", best_p)
        moved += 1
    return moved, off


# ------------------------------------------------------------------
# faces -> mesh
# ------------------------------------------------------------------


def _key(point: list[float]) -> tuple[float, float]:
    return (round(point[0], PRECISION), round(point[1], PRECISION))


def mesh_from_faces(faces: list[list[list[float]]], cls: type = CoarsePseudoQuadMesh) -> tuple["QuadMesh | None", int]:
    """A coarse mesh from faces as corner point lists, welded at ``PRECISION``. ``None`` if nothing is left.

    Faces that repeat a corner are dropped as degenerate.
    """
    if not faces:
        return None, 0
    index = {}
    vertices = []
    cells = []
    dropped = 0
    for face in faces:
        cell = []
        for point in face:
            key = _key(point)
            if key not in index:
                index[key] = len(vertices)
                vertices.append([point[0], point[1], 0.0])
            cell.append(index[key])
        if len(set(cell)) == len(cell) and len(cell) >= 3:
            cells.append(cell)
        else:
            dropped += 1
    if not cells:
        return None, dropped
    return cls.from_vertices_and_faces(vertices, cells), dropped


# ------------------------------------------------------------------
# the entry point
# ------------------------------------------------------------------


def coarse_from_skeleton(
    geometry: Any,
    loops: list[list[list[float]]] | None = None,
    poles: Sequence[list[float]] = (),
    snap_tol: float = 0.0,
    cls: type = CoarsePseudoQuadMesh,
) -> tuple["QuadMesh | None", dict[str, Any]]:
    """A coarse quad layout from an edited patch skeleton (the wireframe handed back from Rhino).

    Pass one point per patch corner, not curve samples.

    Parameters
    ----------
    geometry : mesh or list
        A compas mesh, or closed polylines with ONE POINT PER PATCH CORNER.
        See design_notes/editing.md (rebuild.py) on why.
    loops : list[list[[x, y, z]]], optional
        Boundary loops -- outer first, then holes. Used for snapping and handed
        to ``repair.solve_non_quad_faces`` so a split vertex lands on the
        wall rather than on a chord.
    poles : list[[x, y, z]], optional
        Preferred pole positions. Pass the previous layout's poles so an
        untouched pseudo-quad keeps its collapsed corner where it was.
    snap_tol : float, optional
        How far a BOUNDARY corner may be projected onto a wall -- not which
        corners are eligible, which is topological. ``0`` disables snapping.
        A boundary corner further than this from every loop is left alone and
        counted in ``off_wall``, because at that distance the layout is wrong
        rather than imprecise and moving it would hide the fact.
    cls : type, optional

    Returns
    -------
    (mesh or None, dict)
        The layout, and what happened to it: ``faces_in``, ``faces_out``,
        ``vertices``, ``snapped``, ``off_wall``, ``degenerate``, ``sides`` (the
        side-count histogram BEFORE repair -- anything other than ``{4: n}``
        means the skeleton was not four-sided patches) and ``repair`` (the note
        ``repair.solve_non_quad_faces`` returned).
    """
    faces = faces_from_geometry(geometry)
    notes = {"faces_in": len(faces), "faces_out": 0, "vertices": 0, "snapped": 0, "off_wall": 0, "degenerate": 0, "sides": {}, "repair": ""}
    if not faces:
        return None, notes

    mesh, dropped = mesh_from_faces(faces, cls)
    notes["degenerate"] = dropped
    if mesh is None:
        return None, notes

    # Snapping needs the topology -- see ``snap_to_loops`` -- so it happens
    # after the weld and before anything is measured or validated.
    notes["snapped"], notes["off_wall"] = snap_to_loops(mesh, loops or [], snap_tol)

    sides = {}
    for fkey in mesh.faces():
        n = len(mesh.face_vertices(fkey))
        sides[n] = sides.get(n, 0) + 1
    notes["sides"] = sides

    # Whatever came back, it leaves here all-quad -- or all-quad plus registered
    # pseudo-quad poles, which is the same thing to ``densification``. Reused
    # wholesale: this is the identical repair the generated layout goes through.
    mesh, note = solve_non_quad_faces(mesh, cls, loops, poles=poles)
    notes["repair"] = note
    notes["faces_out"] = mesh.number_of_faces()
    notes["vertices"] = mesh.number_of_vertices()
    return mesh, notes


# ------------------------------------------------------------------
# moving a traced curve onto new endpoints
# ------------------------------------------------------------------


def warp_polyline(points: list[list[float]], pa: list[float], pb: list[float], limit: float = 1.0) -> list[list[float]] | None:
    """End-anchored warp of a traced polyline onto moved endpoints, blending by arc length.

    ``None`` when an end moves more than ``limit`` times the curve length.

    Parameters
    ----------
    points : list[[x, y, z]]
    pa, pb : [x, y, z]
    limit : float, optional
        Reject the warp when either end has to move further than ``limit``
        times the curve's own length. A displacement of that size is not this
        curve moved, it is a different curve, and warping it produces a
        self-crossing polyline that densifies into folded quads.

    Returns
    -------
    list[[x, y, z]] or None
        ``None`` when the warp was rejected, so the caller can fall back to the
        straight chord.
    """
    curve = [list(p) for p in points]
    if len(curve) < 2:
        return None

    cum = [0.0]
    for a, b in pairwise(curve):
        cum.append(cum[-1] + distance_point_point(a, b))
    total = cum[-1]
    if total <= 1e-12:
        return None

    d0 = [pa[i] - curve[0][i] for i in range(3)]
    d1 = [pb[i] - curve[-1][i] for i in range(3)]
    reach = max((d0[0] ** 2 + d0[1] ** 2) ** 0.5, (d1[0] ** 2 + d1[1] ** 2) ** 0.5)
    if reach > limit * total:
        return None

    out = []
    for point, s in zip(curve, cum):
        t = s / total
        out.append([point[i] + (1.0 - t) * d0[i] + t * d1[i] for i in range(3)])
    out[0] = [pa[0], pa[1], pa[2] if len(pa) > 2 else 0.0]
    out[-1] = [pb[0], pb[1], pb[2] if len(pb) > 2 else 0.0]
    return out


# ------------------------------------------------------------------
# what to bake into Rhino
# ------------------------------------------------------------------


def face_polylines(coarse: "QuadMesh") -> list[Polyline]:
    """One closed polyline per patch with corners only, to bake and edit.

    Returns
    -------
    list[compas.geometry.Polyline]
    """
    out = []
    for fkey in coarse.faces():
        corners = [list(coarse.vertex_coordinates(v)) for v in coarse.face_vertices(fkey)]
        out.append(Polyline(corners + corners[:1]))
    return out
