"""Global relaxation of the dense mesh after densification, with topological per-vertex constraints.

Sliding vertices keep their order; the result is kept only if no quality measure got worse.
"""

from __future__ import annotations

from math import cos
from math import radians
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import numpy as np

from compas.tolerance import TOL
from compas_singular.geometry.polyline import closest_on_polyline

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh


__all__ = ["relax_mesh", "relaxation_constraints"]

#: ``(iterations, damping)`` settings, gentlest first. The first accepted wins,
#: so a mesh takes the least relaxation that helps it.
SCHEDULE = ((20, 0.3), (50, 0.5), (100, 0.5))

#: Minimum spacing between two vertices sliding on one curve, as a fraction of
#: their mean spacing: 0 would let an edge collapse, 1 would freeze the chain.
MIN_GAP = 0.25


# ------------------------------------------------------------------
# curves and parameters
# ------------------------------------------------------------------


class _Curve(object):
    """A polyline that reports arc-length parameters as well as closest points."""

    def __init__(self, points: list[list[float]], closed: bool = False) -> None:
        self.points = [list(p)[:3] for p in points]
        self.closed = closed
        self.cumulative = [0.0]
        for a, b in zip(self.points, self.points[1:]):
            self.cumulative.append(self.cumulative[-1] + _distance(a, b))
        self.length = self.cumulative[-1]
        pts = np.array(self.points, dtype=float).reshape(-1, 3)
        self._a = pts[:-1]
        self._ab = pts[1:] - pts[:-1]
        self._len2 = (self._ab * self._ab).sum(axis=1)

    def project(self, xyz: list[float]) -> tuple[list[float] | None, float]:
        """``(point, parameter)`` of the closest point, parameter as arc length."""
        index, t, q, d = closest_on_polyline(xyz, self.points)
        if d == float("inf"):
            return None, 0.0
        return q, self._parameter(index, t)

    def _parameter(self, index: int, t: float) -> float:
        a, b = self.points[index], self.points[index + 1]
        abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        span = (abx * abx + aby * aby + abz * abz) ** 0.5
        return self.cumulative[index] + t * span

    def project_many(self, points: list[list[float]]) -> list[tuple[list[float] | None, float]]:
        """``project`` for many points, with the same results."""
        if not points or not len(self._a):
            return [self.project(p) for p in points]
        p = np.asarray([[x[0], x[1], x[2]] for x in points], dtype=float)
        a, ab, len2 = self._a[None, :, :], self._ab[None, :, :], self._len2[None, :]
        valid = len2 != 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            t = ((p[:, None, 0] - a[..., 0]) * ab[..., 0] + (p[:, None, 1] - a[..., 1]) * ab[..., 1] + (p[:, None, 2] - a[..., 2]) * ab[..., 2]) / len2
        t = np.clip(np.where(valid, t, 0.0), 0.0, 1.0)
        q = a + ab * t[..., None]
        d = np.sqrt(((q - p[:, None, :]) ** 2).sum(axis=2))
        d = np.where(valid, d, np.inf)
        near = d <= d.min(axis=1, keepdims=True) * (1.0 + 1e-9) + 1e-12

        out = []
        for row, xyz in enumerate(points):
            best = None
            for i in np.flatnonzero(near[row]).tolist():
                index, ti, qi, di = closest_on_polyline(xyz, self.points[i : i + 2])
                if di != float("inf") and (best is None or di < best[3]):
                    best = (i + index, ti, qi, di)
            if best is None:
                out.append((None, 0.0))
            else:
                out.append((best[2], self._parameter(best[0], best[1])))
        return out

    def point_at(self, parameter: float) -> list[float]:
        """The point at an arc-length parameter, clamped to the curve."""
        parameter = max(0.0, min(self.length, parameter))
        for i, (lo, hi) in enumerate(zip(self.cumulative, self.cumulative[1:])):
            if parameter <= hi:
                span = hi - lo
                t = 0.0 if span <= 0.0 else (parameter - lo) / span
                a, b = self.points[i], self.points[i + 1]
                return [a[k] + (b[k] - a[k]) * t for k in range(3)]
        return list(self.points[-1])


def _distance(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


# ------------------------------------------------------------------
# constraints
# ------------------------------------------------------------------


def _singularities(mesh: Mesh) -> set[int]:
    """Interior vertices whose valency is not 4, plus every pole."""
    out = set()
    for vertex in mesh.vertices():
        if mesh.is_vertex_on_boundary(vertex):
            continue
        if len(mesh.vertex_neighbors(vertex)) != 4:
            out.add(vertex)
    out.update(mesh.attributes.get("face_pole", {}).values())
    return out


def _boundary_curves(mesh: Mesh) -> list[tuple[list[int], _Curve]]:
    """The mesh's own boundary loops, as closed ``_Curve`` objects."""
    curves = []
    for loop in mesh.vertices_on_boundaries():
        keys = list(loop)
        while len(keys) > 1 and keys[-1] == keys[0]:
            keys.pop()
        if len(keys) < 3:
            continue
        points = [mesh.vertex_coordinates(v) for v in keys]
        curves.append((keys, _Curve(points + points[:1], closed=True)))
    return curves


def relaxation_constraints(
    mesh: Mesh,
    seam_edge: dict[str, tuple[int, int, int]] | None = None,
    edges_to_curves: dict[tuple[int, int], list[list[float]]] | None = None,
    corner_angle: float = 30.0,
    seams: str = "free",
    pin_singularities: bool = False,
) -> tuple[dict[_Curve, list[int]], set[int]]:
    """What each vertex may do.

    Returns
    -------
    (dict, set)
        ``chains`` maps a ``_Curve`` to the ordered vertices sliding on
        it; ``pinned`` is the set of vertices that may not move.
    """
    pinned = set(_singularities(mesh)) if pin_singularities else set()
    chains = {}

    # the outline: slide along it, pin the corners
    limit = cos(radians(180.0 - corner_angle))
    for keys, curve in _boundary_curves(mesh):
        count = len(keys)
        for i, vertex in enumerate(keys):
            a = mesh.vertex_coordinates(keys[i - 1])
            b = mesh.vertex_coordinates(vertex)
            c = mesh.vertex_coordinates(keys[(i + 1) % count])
            u = [b[k] - a[k] for k in range(3)]
            v = [c[k] - b[k] for k in range(3)]
            lu, lv = _distance(a, b), _distance(b, c)
            if lu <= 0.0 or lv <= 0.0:
                continue
            cosine = sum(u[k] * v[k] for k in range(3)) / (lu * lv)
            if cosine < limit:
                pinned.add(vertex)
        chains[curve] = [v for v in keys if v not in pinned]

    if seams == "free":
        return chains, pinned

    # the patch seams: each chain in its order along its own coarse edge
    if seam_edge and edges_to_curves:
        by_edge = {}
        for vertex in mesh.vertices():
            if mesh.is_vertex_on_boundary(vertex):
                continue
            found = seam_edge.get(TOL.geometric_key(mesh.vertex_coordinates(vertex)))
            if found is None:
                continue
            u, v, index = found
            by_edge.setdefault((u, v), []).append((index, vertex))

        for edge, members in by_edge.items():
            points = edges_to_curves.get(edge) or edges_to_curves.get((edge[1], edge[0]))
            if not points or len(points) < 2:
                continue
            members.sort()
            ordered = [vertex for _, vertex in members if vertex not in pinned]
            if not ordered:
                continue
            if seams == "fixed":
                pinned.update(ordered)
            else:
                chains[_Curve(points)] = ordered

    return chains, pinned


# ------------------------------------------------------------------
# the pass
# ------------------------------------------------------------------


def _quality(mesh: Mesh) -> tuple[float, float, float]:
    """``(min angle, max angle, max aspect)`` -- the three the gate compares."""
    from compas_singular.framefield.quality import mesh_quality

    q = mesh_quality(mesh)
    return q["min_angle"], q["max_angle"], q["aspect_max"]


def _positions(mesh: Mesh) -> dict[int, list[float]]:
    return {v: mesh.vertex_coordinates(v) for v in mesh.vertices()}


def _restore(mesh: Mesh, positions: dict[int, list[float]]) -> None:
    for vertex, xyz in positions.items():
        mesh.vertex_attributes(vertex, "xyz", xyz)


def _reproject(mesh: Mesh, chains: dict[_Curve, list[int]]) -> None:
    """Put every sliding vertex back on its curve, keeping its order along it."""
    for curve, ordered in chains.items():
        if not ordered:
            continue
        found = curve.project_many([mesh.vertex_coordinates(v) for v in ordered])
        wanted = [parameter for _, parameter in found]

        floor = MIN_GAP * (curve.length / (len(ordered) + 1))

        # a forward then a backward sweep leaves the parameters increasing and
        # on the curve, so no vertex passes its neighbour
        for i in range(1, len(wanted)):
            wanted[i] = max(wanted[i], wanted[i - 1] + floor)
        upper = curve.length if not curve.closed else curve.length - floor
        for i in range(len(wanted) - 1, -1, -1):
            cap = upper if i == len(wanted) - 1 else wanted[i + 1] - floor
            wanted[i] = min(wanted[i], cap)

        for vertex, parameter in zip(ordered, wanted):
            mesh.vertex_attributes(vertex, "xyz", curve.point_at(parameter))


def _smooth(
    mesh: Mesh,
    chains: dict[_Curve, list[int]],
    pinned: set[int],
    kmax: int,
    damping: float,
) -> None:
    """``kmax`` rounds of area-weighted smoothing, reprojecting the sliding
    vertices after each."""
    fixed = set(pinned)
    movable = [v for v in mesh.vertices() if v not in fixed]
    around = {v: [f for f in mesh.vertex_faces(v, ordered=True) if f is not None] for v in movable}
    for _ in range(kmax):
        area = {f: mesh.face_area(f) for f in mesh.faces()}
        centre = {f: list(mesh.face_centroid(f)) for f in mesh.faces()}
        centroid = {}
        for vertex in movable:
            faces = around[vertex]
            areas = [area[f] for f in faces]
            points = [centre[f] for f in faces]
            if not areas or not sum(areas):
                continue
            total = sum(areas)
            centroid[vertex] = [sum(a * p[i] for a, p in zip(areas, points)) / total for i in range(3)]

        for vertex, target in centroid.items():
            xyz = mesh.vertex_coordinates(vertex)
            mesh.vertex_attributes(vertex, "xyz", [xyz[i] + damping * (target[i] - xyz[i]) for i in range(3)])

        _reproject(mesh, chains)


def relax_mesh(
    mesh: Mesh,
    seam_edge: dict[str, tuple[int, int, int]] | None = None,
    edges_to_curves: dict[tuple[int, int], list[list[float]]] | None = None,
    corner_angle: float = 30.0,
    schedule: Sequence[tuple[int, float]] = SCHEDULE,
    seams: str = "free",
    pin_singularities: bool = False,
) -> dict[str, Any]:
    """**Relax a dense mesh globally**, its outline sliding along itself.

    Modifies ``mesh`` in place, and only if no quality measure gets worse.

    Parameters
    ----------
    mesh : Mesh
        The dense mesh.
    seam_edge : dict, optional
        ``{geometric_key: (u, v, index)}``, as in
        ``FieldDecomposition.densify_stats['seam_edge']``. Needed for ``seams``
        other than ``'free'``.
    edges_to_curves : dict, optional
        ``{(u, v): [point, ...]}``, from ``FieldDecomposition.edges_to_curves``.
    corner_angle : float, optional
        Kink angle, in degrees, past which a boundary vertex is a pinned corner.
    schedule : sequence[(int, float)], optional
        ``(iterations, damping)`` settings, gentlest first.
    seams : {'free', 'slide', 'fixed'}, optional
        What the patch-seam vertices may do. ``'free'`` (the default, and the
        only one measured to help) lets them leave their separatrix; ``'slide'``
        keeps them on it; ``'fixed'`` pins them.
    pin_singularities : bool, optional
        Hold irregular vertices and poles in place. Off by default.

    Returns
    -------
    dict
        ``accepted`` -- the setting used, or ``None`` if the mesh was left
        untouched; ``before`` / ``after`` -- ``(min angle, max angle, max
        aspect)``; ``sliding`` / ``pinned`` -- how many vertices of each kind.
    """
    chains, pinned = relaxation_constraints(
        mesh, seam_edge=seam_edge, edges_to_curves=edges_to_curves, corner_angle=corner_angle, seams=seams, pin_singularities=pin_singularities
    )

    before = _quality(mesh)
    original = _positions(mesh)
    sliding = sum(len(v) for v in chains.values())

    report = {"accepted": None, "before": before, "after": before, "sliding": sliding, "pinned": len(pinned)}

    done = None  # (iterations, damping) the mesh is at right now
    for kmax, damping in schedule:
        if done is not None and done[1] == damping and done[0] <= kmax:
            # the same damping, further on: continue rather than start again
            _smooth(mesh, chains, pinned, kmax - done[0], damping)
        else:
            _restore(mesh, original)
            _smooth(mesh, chains, pinned, kmax, damping)
        done = (kmax, damping)
        after = _quality(mesh)
        if after[0] >= before[0] and after[1] <= before[1] and after[2] <= before[2]:
            report["accepted"] = (kmax, damping)
            report["after"] = after
            return report

    _restore(mesh, original)
    return report
