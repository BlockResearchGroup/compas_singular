"""Make the separatrix network a planar arrangement before the coarse layout is recovered from it.

Tiles walls, lands loose ends, splits at crossings and drops slivers, with shared node positions.
"""

from __future__ import annotations

from math import atan2
from typing import TYPE_CHECKING
from typing import Any

from compas.geometry import distance_point_point
from compas.itertools import pairwise
from compas_singular.geometry.polyline import closest_on_polyline
from compas_singular.geometry.polyline import loop_arc_lengths
from compas_singular.geometry.polyline import loop_parameter
from compas_singular.geometry.polyline import near_loop
from compas_singular.geometry.polyline import point_at_length
from compas_singular.geometry.polyline import polyline_length
from compas_singular.geometry.polyline import signed_area

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh


__all__ = ["planar_arrangement", "faces_with_repeated_vertices", "count_interior_crossings", "count_dangling_ends", "faces_from_arrangement"]


#: ``TOL.geometric_key`` rounds to three decimals, and that is how
#: ``from_polylines`` matches endpoints: an arc shorter than this is a node.
GKEY_RESOLUTION = 1e-3


def _seg_seg(
    p1: list[float],
    p2: list[float],
    p3: list[float],
    p4: list[float],
) -> tuple[float, float] | None:
    """Parameters ``(t, u)`` where two segments meet, or ``None``.

    Endpoints count as meeting. Collinear overlaps return ``None``: there is no
    single point to insert, and doubled-edge removal handles them.
    """
    d1x, d1y = p2[0] - p1[0], p2[1] - p1[1]
    d2x, d2y = p4[0] - p3[0], p4[1] - p3[1]
    den = d1x * d2y - d1y * d2x
    if abs(den) < 1e-14:
        return None
    ox, oy = p3[0] - p1[0], p3[1] - p1[1]
    t = (ox * d2y - oy * d2x) / den
    u = (ox * d1y - oy * d1x) / den
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return max(0.0, min(1.0, t)), max(0.0, min(1.0, u))
    return None


def _partition_loop(
    loop: list[list[float]],
    marks: list[list[float]],
    tol: float,
) -> list[list[list[float]]]:
    """Cut a closed loop into arcs at ``marks``, ordered by arc length, keeping the loop's own vertices."""
    _seg, cum = loop_arc_lengths(loop)
    total = cum[-1]
    if total <= 0.0:
        return []
    ring = list(loop) + list(loop[:1])

    placed = []
    for p in marks:
        d, s = loop_parameter(p, loop, cum)
        if d > tol:
            continue
        placed.append((s % total, [float(p[0]), float(p[1]), 0.0]))

    if len(placed) < 2:
        # fewer than two cuts leaves a closed arc, which bounds a degenerate patch
        placed = [(total * k / 3.0, None) for k in range(3)]

    placed.sort(key=lambda m: m[0])
    unique = [placed[0]]
    for s, p in placed[1:]:
        if s - unique[-1][0] > max(tol * 0.25, GKEY_RESOLUTION):
            unique.append((s, p))
    if len(unique) > 1 and (total - unique[-1][0]) + unique[0][0] <= max(tol * 0.25, GKEY_RESOLUTION):
        unique.pop()
    if len(unique) < 2:
        unique = [(total * k / 3.0, None) for k in range(3)]

    def point_at(s: float) -> list[float]:
        return point_at_length(ring, cum, s % total)

    arcs = []
    n = len(unique)
    for k in range(n):
        s0, p0 = unique[k]
        s1, p1 = unique[(k + 1) % n]
        a0 = list(p0) if p0 is not None else point_at(s0)
        a1 = list(p1) if p1 is not None else point_at(s1)
        s1w = s1 if s1 > s0 else s1 + total

        # the loop's own vertices in between, by arc length -- an arc that
        # wraps past the loop's seam takes its high-index vertices first
        between = []
        for i, sv in enumerate(cum[:-1]):
            for s in (sv, sv + total):
                if s0 + 1e-9 < s < s1w - 1e-9:
                    between.append((s, [float(c) for c in loop[i]]))
        between.sort(key=lambda m: m[0])

        arc = [a0]
        for _, p in between:
            if distance_point_point(p, arc[-1]) > GKEY_RESOLUTION:
                arc.append(p)
        if distance_point_point(a1, arc[-1]) <= GKEY_RESOLUTION and len(arc) > 1:
            arc.pop()
        arc.append(a1)
        if len(arc) >= 2 and polyline_length(arc) > GKEY_RESOLUTION:
            arcs.append(arc)
    return arcs


class _Nodes(object):
    """Positions that several polylines must agree on EXACTLY.

    Every shared point is looked up here and written back from here, so two
    polylines meeting at a crossing get byte-identical coordinates.
    """

    def __init__(self, tol: float, loops: list[list[list[float]]] | None = None) -> None:
        self.tol = tol
        self.loops = list(loops or [])
        self.points = []
        # id(point) -> ((x, y), on a wall?). Keyed by position as well as
        # identity, because landing a loose end moves a node in place.
        self._wall = {}

    def _on_wall(self, p: list[float]) -> bool:
        key = (p[0], p[1])
        cached = self._wall.get(id(p))
        if cached is not None and cached[0] == key:
            return cached[1]
        flag = any(near_loop([p], loop, self.tol * 0.25)[0] for loop in self.loops)
        self._wall[id(p)] = (key, flag)
        return flag

    def add(
        self,
        p: list[float],
        tol: float | None = None,
        exclude: list[float] | None = None,
    ) -> list[float]:
        """The canonical position for ``p``, creating a node if it is new."""
        tol = self.tol if tol is None else tol
        wall = self._on_wall(p) if self.loops else False
        for q in self.points:
            if exclude is not None and q is exclude:
                continue
            limit = self.tol * 0.25 if (wall and self._on_wall(q)) else tol
            if distance_point_point(p, q) < min(limit, tol):
                return q
        q = [float(p[0]), float(p[1]), 0.0]
        self.points.append(q)
        return q


def _normalise(cut: tuple[int, float], n_points: int) -> tuple[int, float]:
    """Move a cut at a segment's END onto the next segment's start, so the same
    crossing reported from both segments sorts as one cut."""
    i, t = cut
    if t > 1.0 - 1e-9 and i + 1 < n_points - 1:
        return i + 1, 0.0
    return i, t


def _split_chain(
    points: list[list[float]],
    cuts: list[tuple[int, float, list[float]]],
) -> list[list[list[float]]]:
    """Cut a polyline at ``(segment, t, position)`` marks. Ends are kept."""
    if not cuts:
        return [[list(p) for p in points]]
    cuts = sorted(cuts, key=lambda c: (c[0], c[1]))

    pieces = []
    current = [list(points[0])]
    k = 0
    for i in range(len(points) - 1):
        while k < len(cuts) and cuts[k][0] == i:
            q = cuts[k][2]
            k += 1
            if distance_point_point(current[-1], q) > 1e-12:
                current.append(list(q))
            if len(current) >= 2:
                pieces.append(current)
            current = [list(q)]
        if distance_point_point(current[-1], points[i + 1]) > 1e-12:
            current.append(list(points[i + 1]))
    if len(current) >= 2:
        pieces.append(current)
    return pieces


def planar_arrangement(
    boundary: list[list[list[float]]],
    others: list[list[list[float]]],
    tol: float,
    report: dict[str, Any] | None = None,
    loops: list[list[list[float]]] | None = None,
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """Split every polyline at every crossing so the network is a planar graph.

    Parameters
    ----------
    boundary : list[polyline]
        The wall split points, as ``build_network`` returns them: every end of
        every polyline here, and every end of ``others`` near a wall, becomes a
        split point of that wall.
    others : list[polyline]
        Separatrices.
    tol : float
        How far apart two points may be and still be one node. The background
        spacing is the right scale.
    report : dict, optional
        Updated in place with what the arrangement had to do.
    loops : list[list[[x, y, z]]]
        The domain's boundary loops, outer first. The walls are rebuilt from
        these as a true partition (``_partition_loop``).

    Returns
    -------
    (list[polyline], list[polyline])
        Wall arcs, in loop order, and the separatrix pieces.
    """
    if not loops:
        raise ValueError("planar_arrangement needs the boundary loops to rebuild the walls from")
    report = report if report is not None else {}

    # -- 0. the walls, tiled exactly once ---------------------------------------
    rebuilt = []
    for loop in loops:
        cum = loop_arc_lengths(loop)[1]
        marks = []
        for pl in list(boundary) + list(others):
            for p in (pl[0], pl[-1]):
                d, _ = loop_parameter(p, loop, cum)
                if d <= tol * 0.5:
                    marks.append(p)
        rebuilt.extend(_partition_loop(loop, marks, tol * 0.5))
    if rebuilt:
        boundary = rebuilt

    chains = [("b", [list(p) for p in pl]) for pl in boundary] + [("o", [list(p) for p in pl]) for pl in others]
    if not chains:
        return list(boundary), list(others)

    nodes = _Nodes(tol, loops)

    # -- 1. every existing endpoint is a node -----------------------------------
    # seeded first, so a crossing near one of them snaps to it, not vice versa
    for _, pts in chains:
        head = nodes.add(pts[0])
        pts[0] = list(head)
        pts[-1] = list(nodes.add(pts[-1], exclude=head))

    # -- 2. loose ends onto the polyline they nearly touch -----------------------
    # Per NODE, not per chain end: the arms meeting at a singularity share a
    # node and must move together. A node joining two or more chains is a real
    # junction and moves only to close an overshoot (``near_tol``); a degree-1
    # node has to attach to something, so it reaches further (``far_tol``). The
    # wall never moves -- the loose end lands on it, and step 3 splits it there.
    near_tol = tol * 0.35
    far_tol = tol * 2.5
    landed = 0
    for _ in range(4):
        moves = []
        for node_p in list(nodes.points):
            owners = [(i, end) for i, (_, pts) in enumerate(chains) for end in (0, -1) if distance_point_point(pts[end], node_p) < 1e-9]
            if not owners:
                continue
            reach = far_tol if len(owners) < 2 else near_tol
            own = set(i for i, _ in owners)
            best = None
            for j, (_kind, pts_j) in enumerate(chains):
                if j in own:
                    continue
                _, _, q, d = closest_on_polyline(node_p, pts_j)
                if d >= reach or d < 1e-9:
                    continue
                # meeting another chain end-on needs no split
                if min(distance_point_point(q, pts_j[0]), distance_point_point(q, pts_j[-1])) < near_tol:
                    continue
                if best is None or d < best[0]:
                    best = (d, q)
            if best is not None:
                moves.append((node_p, best[1], owners))
        if not moves:
            break
        for node_p, q, owners in moves:
            node_p[0], node_p[1] = q[0], q[1]
            for i, end in owners:
                chains[i][1][end] = list(node_p)
            landed += 1

    # -- 3. every crossing becomes a shared node -------------------------------
    cuts = [[] for _ in chains]
    crossings = 0
    for i in range(len(chains)):
        kind_i, pts_i = chains[i]
        for j in range(i + 1, len(chains)):
            kind_j, pts_j = chains[j]
            if kind_i == "b" and kind_j == "b":
                continue  # wall arcs meet at their ends and nowhere else
            for si, (a1, a2) in enumerate(pairwise(pts_i)):
                for sj, (b1, b2) in enumerate(pairwise(pts_j)):
                    hit = _seg_seg(a1, a2, b1, b2)
                    if hit is None:
                        continue
                    t, u = hit
                    p = [a1[0] + (a2[0] - a1[0]) * t, a1[1] + (a2[1] - a1[1]) * t, 0.0]
                    # a crossing at a polyline's END is already a node there:
                    # cut only the polyline it is interior to
                    near_i = min(distance_point_point(p, pts_i[0]), distance_point_point(p, pts_i[-1])) < near_tol
                    near_j = min(distance_point_point(p, pts_j[0]), distance_point_point(p, pts_j[-1])) < near_tol
                    if near_i and near_j:
                        continue
                    shared = nodes.add(p, near_tol)
                    if not near_i:
                        cuts[i].append(_normalise((si, t), len(pts_i)) + (shared,))
                    if not near_j:
                        cuts[j].append(_normalise((sj, u), len(pts_j)) + (shared,))
                    crossings += 1

    # -- 4. split, then rebuild the graph ----------------------------------------
    pieces = []
    for (kind, pts), marks in zip(chains, cuts):
        unique = []
        for m in marks:
            if not any(distance_point_point(m[2], n[2]) < 1e-9 and m[0] == n[0] for n in unique):
                unique.append(m)
        for piece in _split_chain(pts, unique):
            # an exact lookup, not a snap: every piece end is already a node
            piece[0] = list(nodes.add(piece[0], 1e-9))
            piece[-1] = list(nodes.add(piece[-1], 1e-9))
            pieces.append((kind, piece))

    # an arc shorter than the key resolution IS its own endpoint to ``from_polylines``
    kept = []
    collapsed = 0
    for kind, piece in pieces:
        if polyline_length(piece) <= GKEY_RESOLUTION or distance_point_point(piece[0], piece[-1]) <= GKEY_RESOLUTION and polyline_length(piece) < tol * 0.05:
            collapsed += 1
            continue
        kept.append((kind, piece))
    pieces = kept

    # -- 5. doubled edges ---------------------------------------------------------
    def key_of(piece: list[list[float]]) -> tuple[tuple[float, float], tuple[float, float]]:
        a = tuple(round(c, 9) for c in piece[0][:2])
        b = tuple(round(c, 9) for c in piece[-1][:2])
        return (a, b) if a <= b else (b, a)

    seen = {}
    unique_pieces = []
    doubled = 0
    for kind, piece in pieces:
        k = key_of(piece)
        mid = piece[len(piece) // 2]
        if any(distance_point_point(mid, other_mid) < tol for other_mid in seen.get(k, ())):
            doubled += 1
            continue
        seen.setdefault(k, []).append(mid)
        unique_pieces.append((kind, piece))
    pieces = unique_pieces

    # -- 6. prune what is still dangling ----------------------------------------
    pruned = 0
    while True:
        degree = {}
        for _, piece in pieces:
            for p in (piece[0], piece[-1]):
                k = tuple(round(c, 9) for c in p[:2])
                degree[k] = degree.get(k, 0) + 1
        drop = set()
        for idx, (kind, piece) in enumerate(pieces):
            if kind == "b":
                continue  # the wall is a closed loop; never cut it
            for p in (piece[0], piece[-1]):
                if degree[tuple(round(c, 9) for c in p[:2])] < 2:
                    drop.add(idx)
        if not drop:
            break
        pieces = [pc for idx, pc in enumerate(pieces) if idx not in drop]
        pruned += len(drop)

    report.update(
        {
            "wall_arcs_partitioned": len(rebuilt),
            "crossings_split": crossings,
            "ends_landed": landed,
            "arcs_collapsed": collapsed,
            "doubled_dropped": doubled,
            "dangling_pruned": pruned,
        }
    )

    out_boundary = [p for kind, p in pieces if kind == "b"]
    out_others = [p for kind, p in pieces if kind == "o"]
    return out_boundary, out_others


# ----------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------


def faces_with_repeated_vertices(mesh: Mesh) -> list[int]:
    """Faces that visit the same vertex -- or the same POSITION -- twice: the
    signature of a face walked on a graph that was not a planar embedding.

    Returns
    -------
    list[face key]
    """
    out = []
    for fkey in mesh.faces():
        fv = mesh.face_vertices(fkey)
        if len(set(fv)) != len(fv):
            out.append(fkey)
            continue
        keys = [tuple(round(c, 6) for c in mesh.vertex_coordinates(v)) for v in fv]
        if len(set(keys)) != len(keys):
            out.append(fkey)
    return out


def count_interior_crossings(polylines: list[list[list[float]]], tol: float) -> list[list[float]]:
    """Crossings in the INTERIOR of both polylines. Should be none."""
    found = []
    for i in range(len(polylines)):
        for j in range(i + 1, len(polylines)):
            a, b = polylines[i], polylines[j]
            for a1, a2 in pairwise(a):
                for b1, b2 in pairwise(b):
                    hit = _seg_seg(a1, a2, b1, b2)
                    if hit is None:
                        continue
                    t, _u = hit
                    p = [a1[0] + (a2[0] - a1[0]) * t, a1[1] + (a2[1] - a1[1]) * t, 0.0]
                    if any(distance_point_point(p, e) < tol for e in (a[0], a[-1], b[0], b[-1])):
                        continue
                    if not any(distance_point_point(p, q) < tol for q in found):
                        found.append(p)
    return found


def count_dangling_ends(
    boundary: list[list[list[float]]],
    others: list[list[list[float]]],
    tol: float,
) -> list[list[float]]:
    """Endpoints no second polyline reaches. Should be none."""
    nodes, degree = [], []

    def node(p: list[float]) -> int:
        for k, q in enumerate(nodes):
            if distance_point_point(p, q) < tol:
                return k
        nodes.append(p)
        degree.append(0)
        return len(nodes) - 1

    for pts in list(boundary) + list(others):
        degree[node(pts[0])] += 1
        degree[node(pts[-1])] += 1
    return [nodes[k] for k, d in enumerate(degree) if d < 2]


# ----------------------------------------------------------------------------
# face recovery
# ----------------------------------------------------------------------------


def _point_in_ring(p: list[float], ring: list[list[float]]) -> bool:
    """Ray casting. ``ring`` is a closed loop of [x, y, z]."""
    x, y = p[0], p[1]
    inside = False
    n = len(ring)
    for i in range(n):
        ax, ay = ring[i][0], ring[i][1]
        bx, by = ring[(i + 1) % n][0], ring[(i + 1) % n][1]
        if (ay > y) != (by > y):
            t = (y - ay) / (by - ay)
            if x < ax + t * (bx - ax):
                inside = not inside
    return inside


def _interior_sample(points: list[list[float]]) -> list[float] | None:
    """A point inside the polygon ``points``, or ``None``: the centroid if it is
    inside, else the first vertex-pair midpoint that is."""
    n = len(points)
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    if _point_in_ring([cx, cy], points):
        return [cx, cy, 0.0]
    for i in range(n):
        for j in range(i + 2, n):
            m = [(points[i][0] + points[j][0]) / 2.0, (points[i][1] + points[j][1]) / 2.0, 0.0]
            if _point_in_ring(m, points):
                return m
    return None


def faces_from_arrangement(
    boundary: list[list[list[float]]],
    others: list[list[list[float]]],
    loops: list[list[list[float]]],
) -> list[list[list[float]]]:
    """Recover the domain's faces from an arranged network by planar face walk.

    For networks whose patch corners all lie on walls, where ``from_polylines`` returns nothing.

    Parameters
    ----------
    boundary, others : list[list[[x, y, z]]]
        The network, already through ``planar_arrangement``.
    loops : list[list[[x, y, z]]]
        Domain boundary loops, outer first. Used only to test containment.

    Returns
    -------
    list[list[[x, y, z]]]
        One entry per face, its corner points in order.
    """

    def node(p: list[float]) -> tuple[float, float]:
        return (round(p[0] / GKEY_RESOLUTION), round(p[1] / GKEY_RESOLUTION))

    edges = []
    for pts in list(boundary) + list(others):
        a, b = node(pts[0]), node(pts[-1])
        if a != b:
            edges.append((a, b, list(pts)))

    def ends(h: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float], list[list[float]]]:
        i, d = h
        a, b, pts = edges[i]
        return (a, b, pts) if d > 0 else (b, a, pts[::-1])

    out = {}
    for i in range(len(edges)):
        for d in (1, -1):
            a, _b, pts = ends((i, d))
            ang = atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
            out.setdefault(a, []).append((ang, (i, d)))
    for a in out:
        out[a].sort()

    faces = []
    seen = set()
    for i in range(len(edges)):
        for d in (1, -1):
            start = (i, d)
            if start in seen:
                continue
            cycle = []
            h = start
            while h not in seen:
                seen.add(h)
                cycle.append(h)
                _a, b, _pts = ends(h)
                ring = out.get(b)
                rev = (h[0], -h[1])
                pos = next((k for k, (_ang, e) in enumerate(ring) if e == rev), None)
                if pos is None:
                    cycle = None
                    break
                h = ring[(pos - 1) % len(ring)][1]
            if not cycle:
                continue

            points = []
            for e in cycle:
                _a, _b, pts = ends(e)
                points.extend(pts[:-1])
            if len(points) < 3:
                continue
            if signed_area(points) <= 0.0:
                continue  # unbounded face, or wound backwards

            sample = _interior_sample(points)
            if sample is None:
                continue
            if not _point_in_ring(sample, loops[0]):
                continue  # outside the domain
            if any(_point_in_ring(sample, hole) for hole in loops[1:]):
                continue  # inside a hole

            faces.append([list(ends(e)[2][0]) for e in cycle])
    return faces
