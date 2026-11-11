"""Rebuild the shape of every coarse edge (wall arc, traced branch, or chord) from the Rhino document alone.

Tolerances allow for the float32 round trip of a baked mesh. Design notes: ``design_notes/datastructures.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING

from compas.geometry import distance_point_point
from compas.itertools import pairwise
from compas.tolerance import TOL
from compas_singular.geometry.polyline import project_on_polyline

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarsePseudoQuadMesh
    from compas_singular.datastructures import CoarseQuadMesh


__all__ = [
    "coarse_edges_to_curves",
    "snap_corners_to_walls",
    "mean_edge_length",
    "BoundaryLoop",
]


def mean_edge_length(coarse: "CoarseQuadMesh | CoarsePseudoQuadMesh") -> float:
    """Average coarse edge length, the scale every tolerance here is relative to."""
    lengths = [coarse.edge_length(edge) for edge in coarse.edges()]
    lengths = [length for length in lengths if length > 0.0]
    return sum(lengths) / len(lengths) if lengths else 1.0


def _clean(points: list[list[float]], tol: float = 1e-9) -> list[list[float]]:
    """Drop consecutive duplicates -- ``Polyline.point_at`` divides by them."""
    out = [list(points[0])]
    for point in points[1:]:
        if distance_point_point(out[-1], point) > tol:
            out.append(list(point))
    return out


class BoundaryLoop(object):
    """A closed domain wall, arc-length parametrised, stored doubled so an arc across the seam is a slice.

    Parameters
    ----------
    points : list[[x, y, z]]
        The loop, closed. A repeated closing point is dropped: the loop is
        stored open and closed by the doubling, so keeping it would put a
        zero-length segment in the middle of every arc that crosses the seam.
    """

    def __init__(self, points: list[list[float]]) -> None:
        pts = [[float(p[0]), float(p[1]), 0.0] for p in points]
        while len(pts) > 1 and distance_point_point(pts[0], pts[-1]) < 1e-9:
            pts = pts[:-1]
        self.points = pts
        # two full turns, so a half-turn arc starting anywhere is contiguous
        self.ring = pts + pts + pts[:1]
        cum = [0.0]
        for a, b in pairwise(self.ring):
            cum.append(cum[-1] + distance_point_point(a, b))
        self.cum = cum
        self.total = cum[len(pts)]

    def project(self, point: list[float]) -> tuple[float, float, list[float] | None]:
        """``(distance, arclength, closest point)`` for a point near the loop.

        Searched over ONE turn -- the second turn is the same loop and would
        only ever tie.
        """
        best = (float("inf"), 0.0, None)
        for i in range(len(self.points)):
            a, b = self.ring[i], self.ring[i + 1]
            abx, aby = b[0] - a[0], b[1] - a[1]
            length2 = abx * abx + aby * aby
            if length2 == 0.0:
                continue
            t = ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby) / length2
            t = max(0.0, min(1.0, t))
            q = [a[0] + abx * t, a[1] + aby * t, 0.0]
            d = distance_point_point(point, q)
            if d < best[0]:
                best = (d, self.cum[i] + t * (self.cum[i + 1] - self.cum[i]), q)
        return best

    def _point_at(self, s: float) -> list[float]:
        """The loop point at arclength ``s`` along the doubled ring."""
        for i in range(len(self.cum) - 1):
            if self.cum[i] <= s <= self.cum[i + 1]:
                seg = self.cum[i + 1] - self.cum[i]
                t = 0.0 if seg == 0.0 else (s - self.cum[i]) / seg
                a, b = self.ring[i], self.ring[i + 1]
                return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, 0.0]
        return list(self.ring[-1])

    def _span(self, s0: float, s1: float, flip: bool, pa: list[float], pb: list[float]) -> list[list[float]] | None:
        """One way round, as the loop's own points from ``pa`` to ``pb``, with the ends set exactly to them."""
        inner = [list(self.ring[i]) for i in range(len(self.ring)) if s0 + 1e-9 < self.cum[i] < s1 - 1e-9]

        # An arc SHORTER than the loop's own sampling step straddles no loop
        # point, so it would come back as two points -- the chord -- and be
        # rejected for having fewer than three. Sample it instead. The shape
        # barely matters at that length (0.002 sagitta on a 0.1 arc of a
        # radius-0.7 hole), but "no arc found" is a lie the tally then repeats.
        if len(inner) < 3:
            inner = [self._point_at(s0 + (s1 - s0) * k / 4.0) for k in (1, 2, 3)]

        head, tail = (pb, pa) if flip else (pa, pb)
        arc = _clean([list(head)] + inner + [list(tail)])
        if flip:
            arc.reverse()
        if len(arc) < 2:
            return None
        arc[0], arc[-1] = [pa[0], pa[1], 0.0], [pb[0], pb[1], 0.0]
        return arc

    def arcs(self, pa: list[float], pb: list[float]) -> list[list[list[float]]]:
        """Both ways round the loop from ``pa`` to ``pb``, shorter first.

        On a hole the edge can be the longer way, so both are offered.
        """
        _da, sa, _qa = self.project(pa)
        _db, sb, _qb = self.project(pb)

        forward = (sb - sa) % self.total
        backward = self.total - forward
        if forward <= 1e-12 or backward <= 1e-12:
            return []

        ahead = (sa, sa + forward, False)
        behind = (sb, sb + backward, True)
        order = (ahead, behind) if forward <= backward else (behind, ahead)
        return [arc for arc in (self._span(s0, s1, flip, pa, pb) for s0, s1, flip in order) if arc is not None]

    def arc(self, pa: list[float], pb: list[float]) -> list[list[float]] | None:
        """The shorter way round the loop from ``pa`` to ``pb``, or ``None``."""
        found = self.arcs(pa, pb)
        return found[0] if found else None


def _arc_is_one_edge(arc: list[list[float]] | None, pa: list[float], pb: list[float], corners: list[list[float]], tol: float) -> bool:
    """Whether this arc is one layout edge, i.e. no other boundary corner lies on it clear of its ends."""
    if not arc or len(arc) < 3:
        return bool(arc)

    xs = [p[0] for p in arc]
    ys = [p[1] for p in arc]
    lo_x, hi_x = min(xs) - tol, max(xs) + tol
    lo_y, hi_y = min(ys) - tol, max(ys) + tol

    for point in corners:
        # the arc's own endpoints are not "another corner"
        if distance_point_point(point, pa) < 1e-6 or distance_point_point(point, pb) < 1e-6:
            continue
        if not (lo_x <= point[0] <= hi_x and lo_y <= point[1] <= hi_y):
            continue
        d, s, total = project_on_polyline(point, arc)
        if d >= tol:
            continue
        # capped so the margin can never swallow the arc it is protecting
        margin = min(tol, 0.25 * total)
        if s <= margin or s >= total - margin:
            continue
        return False
    return True


def _wall_arc(loops: list[BoundaryLoop], pa: list[float], pb: list[float], corners: list[list[float]], wall_tol: float, arc_tol: float) -> list[list[float]] | None:
    """The piece of wall between two corners: the first arc, over nearby loops and both directions, no other corner lies on."""
    candidates = []
    for loop in loops:
        d = max(loop.project(pa)[0], loop.project(pb)[0])
        if d <= wall_tol:
            candidates.append((d, loop))
    candidates.sort(key=lambda pair: pair[0])

    for _d, loop in candidates:
        for arc in loop.arcs(pa, pb):
            # Two points IS the chord. Falling through lets the traced branch
            # have a go.
            if len(arc) < 3:
                continue
            if _arc_is_one_edge(arc, pa, pb, corners, arc_tol):
                return arc
    return None


def snap_corners_to_walls(
    coarse: "CoarseQuadMesh | CoarsePseudoQuadMesh", loops: "list[list[list[float]]] | tuple[list[list[float]], ...]" = (), wall_tol: float | None = None
) -> tuple[int, float]:
    """Move the layout's boundary corners onto the nearest wall, within ``wall_tol``. ``(moved, worst)``.

    Mutates ``coarse``; call it before ``coarse_edges_to_curves``.

    Parameters
    ----------
    coarse : CoarseQuadMesh or CoarsePseudoQuadMesh
        The layout. **Modified in place.**
    loops : list[list[[x, y, z]]], optional
        The domain walls, outer first, each closed -- the same ones handed to
        ``coarse_edges_to_curves``.
    wall_tol : float, optional
        Largest move allowed. Defaults to a quarter of the mean coarse edge
        length, matching ``coarse_edges_to_curves``.

    Returns
    -------
    tuple[int, float]
        How many corners moved, and the largest distance any of them moved.
    """
    walls = [BoundaryLoop(loop) for loop in (loops or []) if len(loop) >= 3]
    if not walls:
        return 0, 0.0
    if wall_tol is None:
        wall_tol = 0.25 * mean_edge_length(coarse)

    moved, worst = 0, 0.0
    for vertex in coarse.vertices():
        if not coarse.is_vertex_on_boundary(vertex):
            continue
        point = coarse.vertex_coordinates(vertex)
        best = None
        for wall in walls:
            d, _s, q = wall.project(point)
            if best is None or d < best[0]:
                best = (d, q)
        if best is None or best[1] is None:
            continue
        if best[0] <= 1e-12 or best[0] > wall_tol:
            continue
        coarse.vertex_attributes(vertex, "xyz", best[1])
        moved += 1
        worst = max(worst, best[0])
    return moved, worst


def coarse_edges_to_curves(
    coarse: "CoarseQuadMesh | CoarsePseudoQuadMesh",
    loops: "list[list[list[float]]] | tuple[list[list[float]], ...]" = (),
    polylines: "list[list[list[float]]] | tuple[list[list[float]], ...]" = (),
    wall_tol: float | None = None,
    precision: int | None = None,
) -> tuple[dict[tuple[int, int], list[list[float]]], dict[str, int]]:
    """``({(u, v): polyline}, tally)``: the shape of every coarse edge, for ``densification(edges_to_curves=...)``.

    The mapping is complete for every edge, as ``densification`` requires.

    Parameters
    ----------
    coarse : CoarseQuadMesh or CoarsePseudoQuadMesh
        The layout to densify. Read only.
    loops : list[list[[x, y, z]]], optional
        The domain walls, outer first, each closed. Sample them from the Rhino
        curves at whatever resolution the final mesh should follow -- these
        points ARE the boundary from here on, so an arc handed over as two
        points is a chord.
    polylines : list[list[[x, y, z]]], optional
        The decomposition's branches, as baked on ``Skeleton::Polylines``.
        Matched by the geometric key of their two ends.
    wall_tol : float, optional
        How far from a wall a boundary corner may sit and still be taken as on
        it. Defaults to a quarter of the mean coarse edge length, which is
        generous on purpose: a corner placed by the background triangulation
        lies on a CHORD of the wall, so on a tightly curved boundary it is a
        sagitta off it -- 0.03 for a 0.5 background on a 1-unit radius -- and
        the branch is already gated on the edge being topologically on the
        layout boundary and on ``_arc_is_one_edge``.
    precision : int, optional
        Decimals for the geometric key of the traced branch. The default is
        COMPAS's own (3), which is far coarser than the single-precision error a
        Rhino mesh round trip introduces and so is safe.

    Returns
    -------
    tuple[dict, dict]
        The mapping, complete for every edge of ``coarse``, and
        ``{'boundary': n, 'traced': n, 'chord': n, 'wall_missed': n}``.

        ``wall_missed`` (a subset of the others) counts boundary edges that did
        not get their wall arc; expect 0 on a generated layout.
    """
    mean = mean_edge_length(coarse)
    if wall_tol is None:
        wall_tol = 0.25 * mean
    arc_tol = 0.05 * mean

    walls = [BoundaryLoop(loop) for loop in (loops or []) if len(loop) >= 3]

    lookup = {}
    traced = []
    for polyline in polylines or []:
        points = [[float(p[0]), float(p[1]), float(p[2])] for p in polyline]
        if len(points) < 2:
            continue
        ka = TOL.geometric_key(points[0], precision)
        kb = TOL.geometric_key(points[-1], precision)
        # A two-point branch is kept even though it IS the chord, because the
        # tally is the point: 'chord' has to mean "no shape was found for this
        # edge", not "the shape found happens to be straight". Most interior
        # branches of a skeleton decomposition are straight, and counting them
        # as losses buries the boundary edge that is a real one.
        for key, curve in (((ka, kb), points), ((kb, ka), list(reversed(points)))):
            if len(lookup.get(key, ())) < len(curve):
                lookup[key] = curve
        traced.append(points)

    def _nearest_branch(pa: list[float], pb: list[float]) -> list[list[float]] | None:
        """The branch whose ends best match these ends by summed distance, allowing for a moved corner."""
        best, found = wall_tol, None
        for points in traced:
            for curve in (points, list(reversed(points))):
                score = distance_point_point(curve[0], pa) + distance_point_point(curve[-1], pb)
                if score < best:
                    best, found = score, curve
        return found

    # BOUNDARY corners only -- see _arc_is_one_edge. An interior corner cannot
    # end a boundary edge, so it is not evidence that an arc spans several.
    corners = [coarse.vertex_coordinates(w) for w in coarse.vertices() if coarse.is_vertex_on_boundary(w)]
    tally = {"boundary": 0, "traced": 0, "chord": 0, "wall_missed": 0}
    out = {}

    for u, v in coarse.edges():
        pa = coarse.vertex_coordinates(u)
        pb = coarse.vertex_coordinates(v)

        curve, how = None, "chord"

        on_wall = bool(walls) and coarse.is_edge_on_boundary((u, v))
        if on_wall:
            curve = _wall_arc(walls, pa, pb, corners, wall_tol, arc_tol)
            if curve is not None:
                how = "boundary"
            else:
                tally["wall_missed"] += 1

        if curve is None:
            curve = lookup.get((TOL.geometric_key(pa, precision), TOL.geometric_key(pb, precision)))
            if curve is None:
                curve = _nearest_branch(pa, pb)
            if curve is not None:
                curve = [list(point) for point in curve]
                how = "traced"

        if curve is None or len(curve) < 2:
            curve = [list(pa), list(pb)]
            how = "chord"
        else:
            # Both branches must end EXACTLY on the corners or the Coons patches
            # of the two faces sharing this edge disagree and will not weld.
            curve[0], curve[-1] = list(pa), list(pb)

        tally[how] += 1
        out[u, v] = curve

    return out, tally
