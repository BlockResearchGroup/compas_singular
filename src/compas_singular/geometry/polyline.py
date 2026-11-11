from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import ceil

from compas.geometry import distance_point_point
from compas.geometry import distance_point_point_xy
from compas.itertools import pairwise

__all__ = [
    "bounding_box_diagonal",
    "closest_on_polyline",
    "discretise_boundary",
    "discretise_line",
    "distance_to_loop",
    "distance_to_polyline",
    "loop_arc_lengths",
    "loop_parameter",
    "near_loop",
    "point_at_length",
    "points_in_polygon_xy",
    "polyline_length",
    "project_on_polyline",
    "resample_loop",
    "signed_area",
]


def closest_on_polyline(point: list[float], points: list[list[float]], closed: bool = False) -> tuple[int, float, list[float], float]:
    """``(segment index, t, closest point, distance)`` on a polyline, in 3D.

    Parameters
    ----------
    point : [x, y, z]
    points : list[[x, y, z]]
        Three components each; unlike the rest of this module, ``z`` is read.
    closed : bool, optional
        Add the segment from the last point back to the first.

    Returns
    -------
    tuple
        ``(index, t, [x, y, z], distance)``. Distance is ``inf``, and the other
        three meaningless, when no segment has non-zero length.
    """
    # compas has closest_point_on_polyline_xy, but it returns the point ALONE --
    # no index and no t, so a tangent or an arc length cannot be recovered from
    # it -- and it is ~2x slower, sorting a per-segment cloud instead of keeping
    # a running min. Its distance is bit-identical to this one where it applies.
    # Polyline.tangent_at_point and Polyline.parameter_at look like the missing
    # half, but both search back for the point against INFINITE segment lines:
    # the first raises when it misses, the second returns a silently wrong
    # parameter on any outline with two non-contiguous collinear runs -- an
    # ordinary T- or L-plate.
    chain = list(points) + list(points[:1]) if closed else points
    best = (0, 0.0, list(points[0]), float("inf"))
    for i, (a, b) in enumerate(pairwise(chain)):
        abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        length2 = abx * abx + aby * aby + abz * abz
        if length2 == 0.0:
            continue
        t = ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby + (point[2] - a[2]) * abz) / length2
        t = max(0.0, min(1.0, t))
        q = [a[0] + abx * t, a[1] + aby * t, a[2] + abz * t]
        d = distance_point_point(point, q)
        if d < best[3]:
            best = (i, t, q, d)
    return best


def project_on_polyline(point: list[float], points: list[list[float]]) -> tuple[float, float, float]:
    """``(distance, arclength, total length)`` for a point near an open polyline, in XY.

    Parameters
    ----------
    point : [x, y, z]
    points : list[[x, y, z]]
        The open polyline.

    Returns
    -------
    tuple
        ``(distance, arclength of the closest point, total length)``.
    """
    best = (float("inf"), 0.0)
    travelled = 0.0
    for a, b in pairwise(points):
        abx, aby = b[0] - a[0], b[1] - a[1]
        length = (abx * abx + aby * aby) ** 0.5
        if length == 0.0:
            continue
        t = ((point[0] - a[0]) * abx + (point[1] - a[1]) * aby) / (length * length)
        t = max(0.0, min(1.0, t))
        q = [a[0] + abx * t, a[1] + aby * t, 0.0]
        d = distance_point_point_xy(point, q)
        if d < best[0]:
            best = (d, travelled + t * length)
        travelled += length
    return best[0], best[1], travelled


def distance_to_polyline(point: list[float], points: list[list[float]]) -> float:
    """Shortest distance from a point to an OPEN polyline, in XY."""
    return project_on_polyline(point, points)[0]


def distance_to_loop(p: list[float], loop: list[list[float]]) -> float:
    """Shortest distance from a point to a closed polyline (stored open), in XY.

    Parameters
    ----------
    p : [x, y, z]
    loop : list[[x, y, z]]
        Open, the last point is not the first.

    Returns
    -------
    float
        ``inf`` for a loop with no segment of non-zero length.
    """
    # Same answer from compas (import ``closest_point_on_polyline_xy``), but ~3.3x slower -- it sorts a per-segment cloud instead of keeping a running min:
    # return distance_point_point(p, closest_point_on_polyline_xy(p, list(loop) + list(loop[:1])))
    best = float("inf")
    for a, b in pairwise(list(loop) + list(loop[:1])):
        abx, aby = b[0] - a[0], b[1] - a[1]
        length2 = abx * abx + aby * aby
        if length2 == 0.0:
            continue
        t = ((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / length2
        t = max(0.0, min(1.0, t))
        q = [a[0] + abx * t, a[1] + aby * t, 0.0]
        best = min(best, distance_point_point_xy(p, q))
    return best


def polyline_length(points: list[list[float]]) -> float:
    """Total length of an OPEN polyline."""
    return sum(distance_point_point(a, b) for a, b in pairwise(points))


def loop_arc_lengths(loop: list[list[float]]) -> tuple[list[float], list[float]]:
    """``(segment lengths, cumulative arc length)`` of a CLOSED loop given open.

    ``cumulative`` has one entry per vertex plus a last one, the loop's total
    length, so segment ``i`` runs from ``cumulative[i]`` to ``cumulative[i + 1]``.
    """
    ring = list(loop) + list(loop[:1])
    seg = [distance_point_point(a, b) for a, b in pairwise(ring)]
    cum = [0.0]
    for s in seg:
        cum.append(cum[-1] + s)
    return seg, cum


def loop_parameter(point: list[float], loop: list[list[float]], cumulative: list[float] | None = None) -> tuple[float, float]:
    """``(distance, arc length)`` of the point of a CLOSED loop closest to ``point``.

    Parameters
    ----------
    point : [x, y, z]
    loop : list[[x, y, z]]
        Open, the last point is not the first.
    cumulative : list[float], optional
        ``loop_arc_lengths``' second value, when the caller already has it.
    """
    if cumulative is None:
        cumulative = loop_arc_lengths(loop)[1]
    i, t, _q, d = closest_on_polyline(point, loop, closed=True)
    return d, cumulative[i] + t * (cumulative[i + 1] - cumulative[i])


def point_at_length(chain: list[list[float]], cumulative: list[float], s: float, tol: float = 1e-12) -> list[float]:
    """The point at arc length ``s`` along an open chain, by linear interpolation.

    For a closed loop pass the ring (the loop with its first point repeated at
    the end) and wrap ``s`` yourself. Beyond either end, the last point.
    """
    for i, (c0, c1) in enumerate(zip(cumulative, cumulative[1:])):
        if c0 - tol <= s <= c1 + tol:
            a, b = chain[i], chain[i + 1]
            span = c1 - c0
            t = 0.0 if span == 0 else (s - c0) / span
            return [a[k] + (b[k] - a[k]) * t for k in range(3)]
    return [float(c) for c in chain[-1]]


def signed_area(loop: list[list[float]]) -> float:
    """Signed area of a closed polygon in XY: positive when counter-clockwise."""
    return 0.5 * sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(loop, loop[1:] + loop[:1]))


def points_in_polygon_xy(points: list[list[float]], polygon: list[list[float]]) -> list[bool]:
    """``compas.geometry.is_point_in_polygon_xy`` for many points at once.

    The same ray-crossing test with the same arithmetic, so every answer --
    including a point exactly on an edge -- is the one compas gives.
    """
    import numpy as np

    if not len(points):
        return []
    pts = np.asarray([[p[0], p[1]] for p in points], dtype=float)
    poly = [(p[0], p[1]) for p in polygon]
    inside = np.zeros(len(pts), dtype=bool)
    x, y = pts[:, 0], pts[:, 1]
    for i in range(-1, len(poly) - 1):
        x1, y1 = poly[i]
        x2, y2 = poly[i + 1]
        cross = (y > min(y1, y2)) & (y <= max(y1, y2)) & (x <= max(x1, x2))
        if x1 != x2:
            if y1 == y2:
                continue
            xinters = (y - y1) * (x2 - x1) / (y2 - y1) + x1
            cross &= x <= xinters
        inside ^= cross
    return inside.tolist()


def near_loop(points: list[list[float]], loop: list[list[float]], radius: float, chunk: int = 2048) -> list[bool]:
    """``distance_to_loop(p, loop) < radius`` for many points at once, exactly."""
    import numpy as np

    if not len(points):
        return []
    ring = [(p[0], p[1]) for p in loop] + [(loop[0][0], loop[0][1])] if len(loop) else []
    seg = np.array([[a[0], a[1], b[0] - a[0], b[1] - a[1]] for a, b in pairwise(ring)], dtype=float).reshape(-1, 4)
    seg = seg[seg[:, 2] * seg[:, 2] + seg[:, 3] * seg[:, 3] != 0.0]
    if not len(seg):
        return [False] * len(points)
    ax, ay, abx, aby = (seg[:, k][None, :] for k in range(4))
    length2 = abx * abx + aby * aby
    margin = 1e-9 * abs(radius) + 1e-12
    out = []
    for start in range(0, len(points), chunk):
        part = points[start : start + chunk]
        pts = np.asarray([[p[0], p[1]] for p in part], dtype=float)
        px, py = pts[:, 0:1], pts[:, 1:2]
        t = np.clip(((px - ax) * abx + (py - ay) * aby) / length2, 0.0, 1.0)
        dx = ax + abx * t - px
        dy = ay + aby * t - py
        d = np.sqrt(dx * dx + dy * dy).min(axis=1)
        for p, dist in zip(part, d.tolist()):
            if dist < radius - margin:
                out.append(True)
            elif dist > radius + margin:
                out.append(False)
            else:
                out.append(distance_to_loop(p, loop) < radius)
    return out


def bounding_box_diagonal(*loops: list[list[float]]) -> float:
    """The diagonal of the total XY bounding box of every loop given.

    Parameters
    ----------
    *loops : list[[x, y, z]]
        Any number of point lists. Empty ones are ignored.

    Returns
    -------
    float
        ``0.0`` if no points were given.
    """
    points = [p for loop in loops for p in (loop or [])]
    if not points:
        return 0.0
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5


def _clean_loop(points: list[list[float]]) -> list[list[float]]:
    """A loop as ``[x, y, z]`` floats, with consecutive duplicates and a closing point dropped."""
    pts = [[float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0] for p in points]
    if not pts:
        return []
    out = [pts[0]]
    for p in pts[1:]:
        if distance_point_point(p, out[-1]) > 1e-9:
            out.append(p)
    if len(out) > 1 and distance_point_point(out[0], out[-1]) < 1e-9:
        out.pop()
    return out


def _subdivide(loop: list[list[float]], spacing: float) -> list[list[float]]:
    """Subdivide EACH segment of a closed loop so that none exceeds ``spacing``."""
    out = []
    for a, b in pairwise(loop + loop[:1]):
        n = max(1, int(ceil(distance_point_point(a, b) / spacing)))
        # the end point is dropped -- the next segment starts there, and the
        # last segment's end is the loop's first point
        for i in range(n):
            t = float(i) / n
            out.append([a[k] + (b[k] - a[k]) * t for k in range(3)])
    return out


def _subdivide_chain(chain: list[list[float]], spacing: float) -> list[list[float]]:
    """Subdivide each segment of an open chain so none exceeds ``spacing``, keeping both ends."""
    out = []
    for a, b in pairwise(chain):
        n = max(1, int(ceil(distance_point_point(a, b) / spacing)))
        for i in range(n):
            t = float(i) / n
            out.append([a[k] + (b[k] - a[k]) * t for k in range(3)])
    out.append([float(c) for c in chain[-1]])
    return out


def _clean_chain(points: list[list[float]]) -> list[list[float]]:
    """An open chain as ``[x, y, z]`` floats, consecutive duplicates dropped, ends kept."""
    pts = [[float(p[0]), float(p[1]), float(p[2]) if len(p) > 2 else 0.0] for p in points]
    if not pts:
        return []
    out = [pts[0]]
    for p in pts[1:]:
        if distance_point_point(p, out[-1]) > 1e-9:
            out.append(p)
    return out


def discretise_line(line: list[list[float]], spacing: float | None) -> list[list[float]]:
    """Discretise one open curve at ``spacing`` (thesis eq. 4.1), keeping its extremities.

    Pass the wall spacing, or a 2-point guide never becomes a Delaunay edge.

    Parameters
    ----------
    line : list[[x, y, z]]
        The open curve.
    spacing : float
        Maximum segment length. ``None`` returns the curve unchanged apart from
        dropping consecutive duplicates.

    Returns
    -------
    list[[x, y, z]]
    """
    points = _clean_chain(line)
    if spacing is None or len(points) < 2:
        return points
    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError("spacing must be positive, got {}".format(spacing))
    return _subdivide_chain(points, spacing)


def discretise_boundary(
    outer: list[list[float]],
    inners: list[list[list[float]]] | None = None,
    alpha: float | None = 0.02,
    d_min: int | None = 5,
    spacing: float | None = None,
) -> tuple[list[list[float]], list[list[list[float]]]]:
    """Discretise closed boundary loops per thesis eq. 4.1, ``d_i = max(ceil(l_i / (alpha D)), d_min)``.

    Every input point survives; each segment is subdivided on its own.

    Parameters
    ----------
    outer : list[[x, y, z]]
        The outer loop, stored OPEN -- the last point is not the first. A
        repeated closing point is dropped, so a closed list is accepted too.
    inners : list[list[[x, y, z]]], optional
        The holes, same convention. They are discretised against the SAME ``D``
        as the outer loop, which is the point of taking them here rather than
        loop by loop: a small hole measured against its own bounding box would
        come back far denser than the wall beside it.
    alpha : float, optional
        Fraction of ``D`` to use as the spacing when ``spacing`` is ``None``.
        ``None`` for both returns the loops unchanged -- the explicit opt-out,
        for a caller who has already sampled its walls.
    d_min : int, optional
        Fewest points per loop, whatever the target length says. ``None`` or
        ``0`` disables the floor.
    spacing : float, optional
        Maximum segment length, given directly. Overrides ``alpha`` when not
        ``None``.

    Returns
    -------
    tuple
        ``(outer, inners)``, both open, with the closing segment subdivided like
        any other.
    """
    outer = _clean_loop(outer)
    inners = [_clean_loop(loop) for loop in (inners or [])]

    if spacing is None and alpha is None:
        return outer, inners

    if spacing is None:
        diagonal = bounding_box_diagonal(outer, *inners)
        if diagonal <= 0.0:
            return outer, inners
        spacing = alpha * diagonal

    spacing = float(spacing)
    if spacing <= 0.0:
        raise ValueError("spacing must be positive, got {}".format(spacing))

    def one(loop: list[list[float]]) -> list[list[float]]:
        if len(loop) < 2:
            return loop
        # ``d_min`` as a spacing rather than a count: the subdivision is
        # per-segment, so the floor has to reach the segments to be felt.
        # ``sum(ceil(l_j / s)) >= ceil(perimeter / s) = d_min`` at this spacing.
        perimeter = sum(distance_point_point(a, b) for a, b in pairwise(loop + loop[:1]))
        # Its own name: assigning to ``spacing`` in here would make it local to
        # ``one`` and unbound on this very line.
        step = spacing
        if d_min and perimeter > 0.0:
            step = min(spacing, perimeter / float(d_min))
        return _subdivide(loop, step)

    return one(outer), [one(loop) for loop in inners]


def resample_loop(points: list[list[float]], spacing: float | None = None) -> list[list[float]]:
    """One loop through ``discretise_boundary``, with no scale rule."""
    loop, _ = discretise_boundary(points, spacing=spacing, alpha=None, d_min=None)
    return loop
