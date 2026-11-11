"""Plain-Python planar geometry the symmetry package needs. No numpy, no shapely:
everything here must run in Rhino 8's interpreter as well."""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import acos
from math import ceil
from math import floor
from math import hypot
from math import pi
from typing import Any
from typing import Iterable
from typing import Sequence


def open_loop(points: Iterable[Sequence[float]], tol: float = 1e-12) -> list[list[float]]:
    """A closed point list with its repeated closing point dropped."""
    pts = [[float(p[0]), float(p[1]), 0.0] for p in points]
    while len(pts) > 1 and abs(pts[0][0] - pts[-1][0]) <= tol and abs(pts[0][1] - pts[-1][1]) <= tol:
        pts.pop()
    return pts


def as_points(curve: Any) -> list[list[float]]:
    pts = getattr(curve, "points", curve)
    return [[float(p[0]), float(p[1]), 0.0] for p in pts]


def signed_area(loop: Sequence[Sequence[float]]) -> float:
    a = 0.0
    n = len(loop)
    for i in range(n):
        x1, y1 = loop[i][0], loop[i][1]
        x2, y2 = loop[(i + 1) % n][0], loop[(i + 1) % n][1]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def area_centroid(loop: Sequence[Sequence[float]]) -> tuple[float, list[float]]:
    """``(area, [cx, cy])`` of a simple polygon, area absolute."""
    a = 0.0
    cx = cy = 0.0
    n = len(loop)
    for i in range(n):
        x1, y1 = loop[i][0], loop[i][1]
        x2, y2 = loop[(i + 1) % n][0], loop[(i + 1) % n][1]
        w = x1 * y2 - x2 * y1
        a += w
        cx += (x1 + x2) * w
        cy += (y1 + y2) * w
    a *= 0.5
    if abs(a) < 1e-300:
        xs = [p[0] for p in loop]
        ys = [p[1] for p in loop]
        return 0.0, [sum(xs) / len(xs), sum(ys) / len(ys)]
    return abs(a), [cx / (6.0 * a), cy / (6.0 * a)]


def oriented(loop: Sequence[Sequence[float]], ccw: bool = True) -> list[Sequence[float]]:
    return list(loop) if (signed_area(loop) > 0) == ccw else list(reversed(loop))


def bbox_diagonal(point_lists: Sequence[Sequence[Sequence[float]]]) -> float:
    xs = [p[0] for pts in point_lists for p in pts]
    ys = [p[1] for pts in point_lists for p in pts]
    if not xs:
        return 0.0
    return hypot(max(xs) - min(xs), max(ys) - min(ys))


def point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    ll = dx * dx + dy * dy
    if ll <= 0.0:
        return hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / ll
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    return hypot(px - (ax + t * dx), py - (ay + t * dy))


#: A vertex that turns by more than this is a CORNER, not a sample of a curve,
#: and gets no chord-sag slack. Rhino closes a curve with at least 28 points
#: (``helpers.MIN_CLOSED_CURVE_POINTS``), which turns 12.9 degrees per vertex, so
#: every sampled closed curve is below it; a regular 16-gon drawn as a polygon is
#: exactly at it and stays strict.
CORNER_TURN = pi / 8.0


def chord_sags(points: Sequence[Sequence[float]], closed: bool) -> list[float]:
    """Per segment, how far it may lie from the smooth curve it samples (zero at corners)."""
    n = len(points)
    last = n if closed else n - 1
    turns = [0.0] * n
    for i in range(n):
        if not closed and (i == 0 or i == n - 1):
            continue
        a, b, c = points[i - 1], points[i], points[(i + 1) % n]
        ux, uy = b[0] - a[0], b[1] - a[1]
        vx, vy = c[0] - b[0], c[1] - b[1]
        lu, lv = hypot(ux, uy), hypot(vx, vy)
        if lu <= 0.0 or lv <= 0.0:
            continue
        cosang = max(-1.0, min(1.0, (ux * vx + uy * vy) / (lu * lv)))
        turns[i] = acos(cosang)
    out = []
    for i in range(last):
        j = (i + 1) % n
        a, b = points[i], points[j]
        length = hypot(b[0] - a[0], b[1] - a[1])
        ends = [turns[i], turns[j]]
        if max(ends) >= CORNER_TURN - 1e-12:
            out.append(0.0)
        else:
            out.append(length * max(ends) / 8.0)
    return out


def point_in_polygon(x: float, y: float, loop: Sequence[Sequence[float]]) -> bool:
    inside = False
    n = len(loop)
    j = n - 1
    for i in range(n):
        xi, yi = loop[i][0], loop[i][1]
        xj, yj = loop[j][0], loop[j][1]
        if (yi > y) != (yj > y):
            xc = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < xc:
                inside = not inside
        j = i
    return inside


def distance_to_loop(x: float, y: float, loop: Sequence[Sequence[float]], closed: bool = True) -> float:
    n = len(loop)
    best = float("inf")
    last = n if closed else n - 1
    for i in range(last):
        a, b = loop[i], loop[(i + 1) % n]
        d = point_segment_distance(x, y, a[0], a[1], b[0], b[1])
        if d < best:
            best = d
    return best


class SegmentHash(object):
    """Segments and points in a uniform grid, for bounded exact nearest-distance queries."""

    def __init__(self, cell: float) -> None:
        self.cell = float(cell)
        self.grid: dict[tuple[int, int], list[int]] = {}
        self.items: list[tuple[float, float, float, float, Any, float]] = []

    def _key(self, x: float, y: float) -> tuple[int, int]:
        return (int(floor(x / self.cell)), int(floor(y / self.cell)))

    def _insert(self, index: int, ax: float, ay: float, bx: float, by: float) -> None:
        length = hypot(bx - ax, by - ay)
        steps = max(1, int(ceil(length / (0.5 * self.cell))))
        seen = set()
        for s in range(steps + 1):
            t = s / float(steps)
            k = self._key(ax + t * (bx - ax), ay + t * (by - ay))
            if k not in seen:
                seen.add(k)
                self.grid.setdefault(k, []).append(index)

    def add_polyline(
        self,
        points: Sequence[Sequence[float]],
        closed: bool,
        tag: Any,
        slacks: Sequence[float] | None = None,
    ) -> None:
        """Add each segment; ``slacks[i]`` is how far segment ``i`` may lie from
        the curve it samples (see ``chord_sags``)."""
        n = len(points)
        if n == 1:
            self.add_point(points[0], tag)
            return
        last = n if closed else n - 1
        for i in range(last):
            a, b = points[i], points[(i + 1) % n]
            index = len(self.items)
            slack = slacks[i] if slacks else 0.0
            self.items.append((a[0], a[1], b[0], b[1], tag, slack))
            self._insert(index, a[0], a[1], b[0], b[1])

    def add_point(self, point: Sequence[float], tag: Any) -> None:
        index = len(self.items)
        self.items.append((point[0], point[1], point[0], point[1], tag, 0.0))
        self._insert(index, point[0], point[1], point[0], point[1])

    def nearest(self, x: float, y: float, limit: float) -> tuple[float, Any]:
        """``(excess, tag)`` of the best item within ``limit``, else ``(inf, None)``; excess is distance minus slack."""
        if not self.items:
            return float("inf"), None
        cx, cy = self._key(x, y)
        reach = int(ceil(limit / self.cell)) + 1
        best, best_tag = float("inf"), None
        for i in range(cx - reach, cx + reach + 1):
            for j in range(cy - reach, cy + reach + 1):
                for index in self.grid.get((i, j), ()):
                    ax, ay, bx, by, tag, slack = self.items[index]
                    d = max(0.0, point_segment_distance(x, y, ax, ay, bx, by) - slack)
                    if d < best:
                        best, best_tag = d, tag
        if best > limit:
            return float("inf"), None
        return best, best_tag
