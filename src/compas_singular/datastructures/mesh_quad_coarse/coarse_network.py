"""Read a coarse quad layout from a drawn network with one polyline per coarse edge.

Validates rather than repairs: a crossing, T-junction, dangling or loose curve raises, naming the curve.
Design notes: ``design_notes/datastructures.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import atan2
from math import radians

from compas.geometry import distance_point_point
from compas.itertools import pairwise
from compas.tolerance import TOL

__all__ = [
    "split_at_corners",
    "split_at_junctions",
    "weld_network",
    "check_network",
    "faces_from_network",
    "check_faces",
]


#: Two consecutive points of one polyline closer than this are the same point. Only
#: guards against zero-length segments, which ``Polyline.point_at`` divides by; the
#: WELD that decides topology happens at ``precision`` decimals, several orders
#: coarser.
EPS = 1e-9

#: Decimals ``TOL.geometric_key`` rounds to when no precision is given. Named here
#: only so a default validity tolerance can be derived from it.
DEFAULT_PRECISION = 3

#: A polyline vertex turning by less than this is collinear, not a corner. One
#: degree is Rhino's default angle tolerance -- a vertex Rhino itself would call
#: tangent is not split.
CORNER_ANGLE = radians(1.0)


# ----------------------------------------------------------------------------
# small geometry, all XY
# ----------------------------------------------------------------------------


def _flat(point: list[float]) -> list[float]:
    return [float(point[0]), float(point[1]), 0.0]


def _clean(points: list[list[float]]) -> list[list[float]]:
    """Point list, flattened to z = 0, with consecutive duplicates dropped."""
    out = []
    for point in points:
        p = _flat(point)
        if not out or distance_point_point(out[-1], p) > EPS:
            out.append(p)
    return out


def _shoelace(ring: list[list[float]]) -> float:
    """Twice the signed area of a closed ring whose last point is not repeated."""
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in pairwise(list(ring) + list(ring[:1])))


def _point_in_ring(point: list[float], ring: list[list[float]]) -> bool:
    """Ray casting in XY. ``ring`` is closed, its last point not repeated."""
    x, y = point[0], point[1]
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


def _distance_to_segment(point: list[float], a: list[float], b: list[float]) -> float:
    """Shortest distance from ``point`` to the segment ``ab``."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    length2 = abx * abx + aby * aby
    if length2 == 0.0:
        return distance_point_point(point, a)
    t = ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby) / length2
    t = max(0.0, min(1.0, t))
    return distance_point_point(point, [a[0] + abx * t, a[1] + aby * t, 0.0])


def _intersect(p1: list[float], p2: list[float], p3: list[float], p4: list[float]) -> tuple[float, float] | None:
    """Parameters ``(t, u)`` where segments ``p1p2`` and ``p3p4`` meet, or ``None``.

    Collinear overlaps return ``None``: there is no single point to report, and a
    curve drawn twice is caught by ``weld_network`` as a duplicate edge.
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


def _crossing(p1: list[float], p2: list[float], p3: list[float], p4: list[float]) -> list[float] | None:
    """Where segments ``p1p2`` and ``p3p4`` meet, or ``None``."""
    hit = _intersect(p1, p2, p3, p4)
    if hit is None:
        return None
    t = hit[0]
    return [p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t, 0.0]


def _locate(points: list[list[float]], point: list[float]) -> tuple[float, int, float]:
    """``(distance, segment, t)`` of the point of an open polyline nearest ``point``."""
    best = (float("inf"), 0, 0.0)
    for i, (a, b) in enumerate(pairwise(points)):
        abx, aby = b[0] - a[0], b[1] - a[1]
        length2 = abx * abx + aby * aby
        if length2 == 0.0:
            continue
        t = ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby) / length2
        t = max(0.0, min(1.0, t))
        d = distance_point_point(point, [a[0] + abx * t, a[1] + aby * t, 0.0])
        if d < best[0]:
            best = (d, i, t)
    return best


def _bbox(points: list[list[float]], pad: float) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _xy(point: list[float]) -> str:
    """A point as it should appear in an error message -- the user has to find it."""
    return "({:.3f}, {:.3f})".format(point[0], point[1])


# ----------------------------------------------------------------------------
# 0. reading a CAD polyline as edges
# ----------------------------------------------------------------------------


def _turn(a: list[float], b: list[float], c: list[float]) -> float:
    """Absolute turning angle at ``b`` walking ``a -> b -> c``, in radians."""
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    return abs(atan2(v1x * v2y - v1y * v2x, v1x * v2x + v1y * v2y))


def split_at_corners(points: list[list[float]], angle: float = CORNER_ANGLE) -> list[list[list[float]]]:
    """Cut a clicked polyline into one piece per straight run, at every vertex turning more than ``angle``.

    Never hand it a sampled curve; not called by ``weld_network``.

    Parameters
    ----------
    points : list[[x, y, z]]
    angle : float, optional
        Radians. Defaults to ``CORNER_ANGLE``, one degree.

    Returns
    -------
    list[list[[x, y, z]]]
        One point list per straight run. A polyline with no corner comes back
        whole.
    """
    points = _clean(points)
    if len(points) < 3:
        return [points] if len(points) == 2 else []

    closed = distance_point_point(points[0], points[-1]) <= EPS
    if closed:
        points[-1] = list(points[0])

    pieces = []
    current = [points[0]]
    for i in range(1, len(points) - 1):
        current.append(points[i])
        if _turn(points[i - 1], points[i], points[i + 1]) > angle:
            pieces.append(current)
            current = [points[i]]
    current.append(points[-1])
    pieces.append(current)

    if closed and len(pieces) > 1 and _turn(points[-2], points[0], points[1]) <= angle:
        # the seam is not a corner: the last run continues straight into the first
        pieces[0] = pieces.pop()[:-1] + pieces[0]

    return pieces


def _cut(points: list[list[float]], cuts: list[tuple[int, float, list[float]]], closed: bool, tol: float) -> list[list[list[float]]]:
    """Cut one polyline at ``(segment, t, point)`` marks, merged by arc length. Returns its pieces."""
    cum = [0.0]
    for a, b in pairwise(points):
        cum.append(cum[-1] + distance_point_point(a, b))
    total = cum[-1]
    if total <= EPS:
        return []

    marks = []
    for segment, t, point in cuts:
        at = cum[segment] + t * (cum[segment + 1] - cum[segment])
        marks.append((at % total if closed else at, list(point)))
    marks.sort(key=lambda mark: mark[0])

    kept = []
    for at, point in marks:
        if not closed and (at <= tol or at >= total - tol):
            continue
        if kept and at - kept[-1][0] <= tol:
            continue
        kept.append((at, point))
    if closed and len(kept) > 1 and kept[0][0] + total - kept[-1][0] <= tol:
        kept.pop()
    if not kept:
        return [points]

    if closed:
        ring = list(zip(cum[:-1], points[:-1]))
        vertices = ring + [(at + total, point) for at, point in ring]
        stops = kept + [(kept[0][0] + total, kept[0][1])]
    else:
        vertices = list(zip(cum, points))
        stops = [(0.0, points[0])] + kept + [(total, points[-1])]

    pieces = []
    for (s0, p0), (s1, p1) in pairwise(stops):
        piece = [list(p0)]
        for at, vertex in vertices:
            if s0 + EPS < at < s1 - EPS and distance_point_point(piece[-1], vertex) > EPS:
                piece.append(list(vertex))
        if distance_point_point(piece[-1], p1) > EPS:
            piece.append(list(p1))
        else:
            piece[-1] = list(p1)
        if len(piece) >= 2:
            pieces.append(piece)
    return pieces


def split_at_junctions(
    polylines: list[list[list[float]]], tol: float | None = None, precision: int | None = None, points: "list[list[float]] | tuple[list[float], ...]" = ()
) -> tuple[list[list[list[float]]], list[int]]:
    """Split every polyline wherever another ends on it or crosses it. ``(pieces, origin)``.

    Turns a drawing into a network ``weld_network`` accepts; dangling ends are left for it to refuse.

    Parameters
    ----------
    polylines : list[list[[x, y, z]]]
    tol : float, optional
        How close an end must be to another curve to be ON it. Defaults to the
        weld resolution, the same distance ``check_network`` refuses a
        T-junction at -- so what this splits is exactly what that would refuse.
    precision : int, optional
        Decimals of the weld, used only to derive the default ``tol``.
    points : list[[x, y, z]], optional
        Extra points to cut at. Any polyline passing within ``tol`` of one is
        split there, exactly as if a curve had ended on it. For a caller that
        knows where a corner is without having a curve that ends there -- a
        corner on a wall, marked by a piece of wall that is not itself an edge.

    Returns
    -------
    tuple[list[list[[x, y, z]]], list[int]]
        The pieces, and for each the index of the input polyline it came from --
        so a caller can map a later refusal back to what the user drew.
    """
    if tol is None:
        tol = 10.0 ** -(DEFAULT_PRECISION if precision is None else precision)

    lines = [_clean(polyline) for polyline in polylines]
    closed = [len(points) > 2 and distance_point_point(points[0], points[-1]) <= tol for points in lines]
    # A closed curve has no ends: its seam is just a point along it.
    ends = [[] if closed[i] or len(points) < 2 else [points[0], points[-1]] for i, points in enumerate(lines)]
    boxes = [_bbox(points, tol) if len(points) >= 2 else None for points in lines]
    cuts = [[] for _ in lines]
    extra = [_flat(point) for point in points]

    # landings -- every other curve's ends, and the extra points
    for i, line in enumerate(lines):
        if boxes[i] is None:
            continue
        candidates = [end for j, others in enumerate(ends) if j != i for end in others]
        for end in candidates + extra:
            if not (boxes[i][0] <= end[0] <= boxes[i][2] and boxes[i][1] <= end[1] <= boxes[i][3]):
                continue
            if any(distance_point_point(end, own) <= tol for own in ends[i]):
                continue  # a shared corner, not a landing
            d, segment, t = _locate(line, end)
            if d <= tol:
                cuts[i].append((segment, t, end))

    # crossings
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if boxes[i] is None or boxes[j] is None or not _overlap(boxes[i], boxes[j]):
                continue
            corners = ends[i] + ends[j]
            for si, (p1, p2) in enumerate(pairwise(lines[i])):
                box_i = _bbox([p1, p2], tol)
                if not _overlap(box_i, boxes[j]):
                    continue
                for sj, (p3, p4) in enumerate(pairwise(lines[j])):
                    if not _overlap(box_i, _bbox([p3, p4], tol)):
                        continue
                    hit = _intersect(p1, p2, p3, p4)
                    if hit is None:
                        continue
                    t, u = hit
                    point = [p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t, 0.0]
                    if any(distance_point_point(point, corner) <= tol for corner in corners):
                        continue  # at an end: a landing or a shared corner
                    cuts[i].append((si, t, point))
                    cuts[j].append((sj, u, point))

    pieces = []
    origin = []
    for i, points in enumerate(lines):
        if len(points) < 2:
            continue
        for piece in _cut(points, cuts[i], closed[i], tol):
            pieces.append(piece)
            origin.append(i)
    return pieces, origin


# ----------------------------------------------------------------------------
# 1. the weld
# ----------------------------------------------------------------------------


def weld_network(polylines: list[list[list[float]]], precision: int | None = None) -> tuple[list[list[float]], list[tuple[int, int, list[list[float]]]]]:
    """``(vertices, edges)`` from a drawn network. One polyline is one coarse edge.

    Parameters
    ----------
    polylines : list[list[[x, y, z]]]
        One polyline per coarse edge. A straight edge is two points; a curved one is
        however many the CAD session sampled it at.
    precision : int, optional
        Decimals the endpoints are welded at. The default is COMPAS's own, 3, which
        is the resolution ``Mesh.from_polylines`` matches endpoints at and the one a
        Rhino mesh round trip already rounds to.

    Returns
    -------
    tuple[list[[x, y, z]], list[(int, int, list[[x, y, z]])]]
        The corners, and one ``(start, end, points)`` per edge with ``points``
        running ``start`` -> ``end``. Corner indices are positional, so handing
        ``vertices`` straight to ``from_vertices_and_faces`` gives mesh keys that
        are these same indices.

    Raises
    ------
    ValueError
        On a polyline that is not an edge, or a second polyline between two corners
        that already have one.
    """
    vertices = []
    index = {}
    edges = []
    between = {}

    for i, polyline in enumerate(polylines):
        points = _clean(polyline)
        if len(points) < 2:
            raise ValueError("polyline {} is not an edge: it has fewer than two distinct points.".format(i))

        ends = []
        for point in (points[0], points[-1]):
            key = TOL.geometric_key(point, precision)
            if key not in index:
                index[key] = len(vertices)
                vertices.append(list(point))
            ends.append(index[key])
        a, b = ends

        if a == b:
            raise ValueError(
                "polyline {} starts and ends at the same corner {}. A coarse edge runs between two corners -- a closed curve is a patch outline, not an edge.".format(
                    i, _xy(points[0])
                )
            )

        pair = (a, b) if a < b else (b, a)
        if pair in between:
            raise ValueError(
                "polylines {} and {} both run between {} and {}. A mesh keys its "
                "half-edges by corner pair, so two edges cannot share both corners "
                "-- add a corner to one of them. (An annulus therefore needs at "
                "least three cuts, not two.)".format(between[pair], i, _xy(vertices[a]), _xy(vertices[b]))
            )
        between[pair] = i

        edges.append((a, b, points))

    return vertices, edges


# ----------------------------------------------------------------------------
# 2. the checks
# ----------------------------------------------------------------------------


def check_network(vertices: list[list[float]], edges: list[tuple[int, int, list[list[float]]]], tol: float) -> None:
    """Raise unless the drawn network is a clean planar arrangement.

    Checks T-junctions, crossings, dangling curves and loose pieces, most specific first.

    Parameters
    ----------
    vertices, edges : list
        As returned by ``weld_network``.
    tol : float
        Distance below which a point is ON a curve. The weld resolution is the right
        scale -- two points nearer than that are already one corner.

    Raises
    ------
    ValueError
    """
    for i, (a, b, points) in enumerate(edges):
        for vertex, point in enumerate(vertices):
            if vertex == a or vertex == b:
                continue
            if distance_point_point(point, points[0]) <= tol or distance_point_point(point, points[-1]) <= tol:
                continue
            if min(_distance_to_segment(point, p, q) for p, q in pairwise(points)) <= tol:
                raise ValueError("the corner at {} lies on the interior of polyline {}. That polyline is two coarse edges drawn as one -- split it there.".format(_xy(point), i))

    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            points_i = edges[i][2]
            points_j = edges[j][2]
            for p1, p2 in pairwise(points_i):
                for p3, p4 in pairwise(points_j):
                    hit = _crossing(p1, p2, p3, p4)
                    if hit is None:
                        continue
                    if any(distance_point_point(hit, end) <= tol for end in (points_i[0], points_i[-1], points_j[0], points_j[-1])):
                        continue
                    raise ValueError("polylines {} and {} cross at {} with no corner there. Split both at that point.".format(i, j, _xy(hit)))

    # Last, because it is the least specific: a corner that lands on a wall drawn as
    # one long curve is BOTH dangling and a T-junction, and "split that polyline" is
    # the message that names the fix. Only a corner touching nothing at all reaches
    # here.
    degree = [0] * len(vertices)
    for a, b, _points in edges:
        degree[a] += 1
        degree[b] += 1
    for vertex, count in enumerate(degree):
        if count < 2:
            raise ValueError(
                "the corner at {} is the end of one curve only. A dangling curve bounds no patch -- extend it to another corner, or delete it.".format(_xy(vertices[vertex]))
            )

    parent = list(range(len(vertices)))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _points in edges:
        parent[root(a)] = root(b)
    pieces = {}
    for i, (a, _b, _points) in enumerate(edges):
        pieces.setdefault(root(a), []).append(i)
    if len(pieces) > 1:
        # Name the piece with the SMALLEST extent, not the fewest curves: a hole
        # drawn inside a square has as many curves as the square does, and it is
        # the hole that is loose. The piece that spans the drawing is the layout.
        def extent(indices: list[int]) -> float:
            box = _bbox([p for i in indices for p in edges[i][2]], 0.0)
            return (box[2] - box[0]) * (box[3] - box[1])

        loose = min(pieces.values(), key=extent)
        raise ValueError(
            "the drawing is in {} separate pieces: polyline {} (and the {} other "
            "curve(s) joined to it) touches nothing else. A coarse layout is one "
            "connected drawing -- connect the piece, or delete it. A hole is "
            "connected by at least three division lines reaching it.".format(len(pieces), loose[0], len(loose) - 1)
        )


# ----------------------------------------------------------------------------
# 3. the face walk
# ----------------------------------------------------------------------------


def faces_from_network(vertices: list[list[float]], edges: list[tuple[int, int, list[list[float]]]], holes: list[list[float]] | None = None) -> list[list[int]]:
    """The domain's patches as counter-clockwise rings of corner indices, by planar face walk.

    The unbounded face (by the full point ring's signed area) and holes are dropped.

    Parameters
    ----------
    vertices, edges : list
        As returned by ``weld_network``.
    holes : list[[x, y, z]], optional
        One point inside each hole of the domain.

    Returns
    -------
    list[list[int]]
        One ring of corner indices per patch, counter-clockwise.
    """
    leaving = {}
    for i, (a, b, points) in enumerate(edges):
        leaving.setdefault(a, []).append((atan2(points[1][1] - points[0][1], points[1][0] - points[0][0]), (i, 1)))
        leaving.setdefault(b, []).append((atan2(points[-2][1] - points[-1][1], points[-2][0] - points[-1][0]), (i, -1)))
    for corner in leaving:
        leaving[corner].sort()

    def ends(halfedge: tuple[int, int]) -> tuple[int, int, list[list[float]]]:
        i, direction = halfedge
        a, b, points = edges[i]
        return (a, b, points) if direction > 0 else (b, a, points[::-1])

    faces = []
    seen = set()
    for i in range(len(edges)):
        for direction in (1, -1):
            start = (i, direction)
            if start in seen:
                continue

            cycle = []
            halfedge = start
            while halfedge not in seen:
                seen.add(halfedge)
                cycle.append(halfedge)
                _a, b, _points = ends(halfedge)
                ring = leaving[b]
                reverse = (halfedge[0], -halfedge[1])
                at = next(k for k, (_angle, h) in enumerate(ring) if h == reverse)
                halfedge = ring[(at - 1) % len(ring)][1]

            ring = []
            corners = []
            for halfedge in cycle:
                a, _b, points = ends(halfedge)
                corners.append(a)
                ring.extend(points[:-1])

            if len(corners) < 3 or _shoelace(ring) <= 0.0:
                continue
            if holes and any(_point_in_ring(hole, ring) for hole in holes):
                continue

            faces.append(corners)

    return faces


# ----------------------------------------------------------------------------
# 4. the promise
# ----------------------------------------------------------------------------


def check_faces(faces: list[list[int]], vertices: list[list[float]], sides: tuple[int, ...] = (3, 4)) -> None:
    """Raise unless every recovered patch has an allowed number of sides (triangles become poles)."""
    for corners in faces:
        if len(corners) not in sides:
            raise ValueError(
                "a patch with {} sides was recovered, at {}. This constructor takes triangular and quadrilateral patches only -- split it in the drawing.".format(
                    len(corners), ", ".join(_xy(vertices[corner]) for corner in corners)
                )
            )
