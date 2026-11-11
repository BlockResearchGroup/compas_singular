"""Cut the fundamental unit (a wedge between two seams) out of a domain by half-plane clips.

Plain Python, no shapely. Design notes: ``design_notes/symmetry.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import atan2
from math import cos
from math import degrees
from math import hypot
from math import pi
from math import sin
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas_singular.symmetry._geometry import area_centroid
from compas_singular.symmetry._geometry import open_loop
from compas_singular.symmetry._geometry import oriented
from compas_singular.symmetry._geometry import point_in_polygon
from compas_singular.symmetry._geometry import signed_area
from compas_singular.symmetry.group import _mirror
from compas_singular.symmetry.group import _rotation

if TYPE_CHECKING:
    from compas_singular.symmetry.domain import Domain
    from compas_singular.symmetry.group import SymmetryGroup


__all__ = ["cut_unit", "UnitDomain", "clip_halfplane"]


# ----------------------------------------------------------------------
# half-plane clipping
# ----------------------------------------------------------------------


def _interp(p: Sequence[float], q: Sequence[float], sp: float, sq: float) -> list[float]:
    t = sp / (sp - sq)
    return [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), 0.0]


def _dedupe(points: Sequence[Sequence[float]], eps: float, closed: bool = True) -> list[Sequence[float]]:
    out = []
    for p in points:
        if out and abs(p[0] - out[-1][0]) <= eps and abs(p[1] - out[-1][1]) <= eps:
            continue
        out.append(p)
    if closed:
        while len(out) > 1 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
            out.pop()
    return out


def _remove_spikes(loop: list[Sequence[float]], eps: float) -> list[Sequence[float]]:
    """Drop vertices where the loop doubles straight back on itself."""
    changed = True
    while changed and len(loop) > 3:
        changed = False
        n = len(loop)
        for i in range(n):
            a, b, c = loop[i - 1], loop[i], loop[(i + 1) % n]
            ux, uy = b[0] - a[0], b[1] - a[1]
            vx, vy = c[0] - b[0], c[1] - b[1]
            cross = ux * vy - uy * vx
            dot = ux * vx + uy * vy
            if abs(cross) <= eps * (hypot(ux, uy) + hypot(vx, vy)) and dot < 0:
                del loop[i]
                changed = True
                break
    return loop


def clip_halfplane(loops: Sequence[Sequence[Sequence[float]]], point: Sequence[float], direction: Sequence[float], eps: float) -> list[list[Sequence[float]]]:
    """The part of the region bounded by ``loops`` LEFT of the directed line.

    Parameters
    ----------
    loops : list of open point lists
        Outer loops anticlockwise, holes clockwise.
    point, direction
        The line.
    eps : float
        Distance within which a vertex counts as ON the line.

    Returns
    -------
    list
        Open point lists, oriented the same way.
    """
    cx, cy = point[0], point[1]
    dx, dy = direction[0], direction[1]

    def side(p: Sequence[float]) -> float:
        return dx * (p[1] - cy) - dy * (p[0] - cx)

    def along(p: Sequence[float]) -> float:
        return dx * (p[0] - cx) + dy * (p[1] - cy)

    kept = []
    chains = []
    for loop in loops:
        n = len(loop)
        if n < 3:
            continue
        s = [side(p) for p in loop]
        s = [0.0 if abs(v) <= eps else v for v in s]
        pos = [v >= 0.0 for v in s]
        if all(pos):
            if any(v > 0.0 for v in s):
                kept.append(loop)
            continue
        if not any(pos):
            continue
        start = [i for i in range(n) if pos[i] and not pos[i - 1]][0]
        chain = None
        for k in range(n):
            i = (start + k) % n
            prev, nxt = (i - 1) % n, (i + 1) % n
            if pos[i] and not pos[prev]:
                chain = [loop[i]] if s[i] == 0.0 else [_interp(loop[prev], loop[i], s[prev], s[i]), loop[i]]
            elif pos[i]:
                chain.append(loop[i])
            if pos[i] and not pos[nxt]:
                if s[i] != 0.0:
                    chain.append(_interp(loop[i], loop[nxt], s[i], s[nxt]))
                chains.append(chain)
                chain = None

    if chains:
        events = []
        for index, chain in enumerate(chains):
            events.append((along(chain[0]), 0, index))  # entry
            events.append((along(chain[-1]), 1, index))  # exit
        events.sort()
        # The interior along the line is exit -> next entry. With ties, an entry
        # sorts before an exit, which keeps a loop that TOUCHES the line from
        # outside as a zero-length chain inside the interval it touches.
        order = [e for e in events]
        successor = {}
        k = 0
        while k < len(order):
            u, kind, index = order[k]
            if kind == 1:
                j = k + 1
                while j < len(order) and not (order[j][1] == 0 and order[j][2] not in successor.values()):
                    j += 1
                if j == len(order):
                    raise ValueError("half-plane clip failed: an exit at {:.6g} has no entry after it".format(u))
                successor[index] = order[j][2]
            k += 1
        visited = set()
        for index in range(len(chains)):
            if index in visited:
                continue
            loop = []
            j = index
            guard = 0
            while j not in visited:
                visited.add(j)
                loop.extend(chains[j])
                j = successor[j]
                guard += 1
                if guard > len(chains) + 1:
                    raise ValueError("half-plane clip failed: chains do not close")
            loop = _remove_spikes(_dedupe(loop, eps), eps)
            if len(loop) >= 3 and abs(signed_area(loop)) > eps * eps:
                kept.append(loop)
    return kept


def clip_polyline(points: Sequence[Sequence[float]], point: Sequence[float], direction: Sequence[float], eps: float) -> list[list[Sequence[float]]]:
    """Pieces of an open polyline left of the line (on the line counts as in).
    Pieces lying entirely ON the line are dropped: they are seam, not guide."""
    cx, cy = point[0], point[1]
    dx, dy = direction[0], direction[1]
    s = [dx * (p[1] - cy) - dy * (p[0] - cx) for p in points]
    s = [0.0 if abs(v) <= eps else v for v in s]
    pieces, cur = [], None
    for i, p in enumerate(points):
        if i > 0:
            sp, si = s[i - 1], s[i]
            if sp > 0.0 and si < 0.0:
                cur.append(_interp(points[i - 1], p, sp, si))
                pieces.append(cur)
                cur = None
            elif sp < 0.0 and si > 0.0:
                cur = [_interp(points[i - 1], p, sp, si)]
        if s[i] >= 0.0:
            cur = [p] if cur is None else cur + [p]
        elif cur is not None:
            pieces.append(cur)
            cur = None
    if cur is not None:
        pieces.append(cur)
    out = []
    for piece in pieces:
        piece = _dedupe(piece, eps, closed=False)
        if len(piece) < 2:
            continue
        on_line = all(abs(dx * (q[1] - cy) - dy * (q[0] - cx)) <= eps for q in piece)
        if on_line:
            continue
        out.append(piece)
    return out


# ----------------------------------------------------------------------
# the unit
# ----------------------------------------------------------------------


class Seam(object):
    """One boundary ray of the wedge.

    ``kind`` is ``'mirror'`` (``element`` is the mirror fixing it) or
    ``'rotation'`` (``element`` is the rotation taking seam A onto seam B).
    """

    def __init__(self, name: str, angle: float, kind: str, element_key: str, centre: Sequence[float]) -> None:
        self.name = name
        self.angle = angle
        self.kind = kind
        self.element = element_key
        self.centre = centre
        self.direction = (cos(angle), sin(angle))

    def locate(self, point: Sequence[float], eps: float) -> float | None:
        """Distance ``t`` from the centre along the seam, or ``None`` if off it."""
        dx, dy = self.direction
        px, py = point[0] - self.centre[0], point[1] - self.centre[1]
        if abs(dx * py - dy * px) > eps:
            return None
        t = dx * px + dy * py
        if t < -eps:
            return None
        return max(t, 0.0)

    def point_at(self, t: float) -> list[float]:
        return [self.centre[0] + t * self.direction[0], self.centre[1] + t * self.direction[1], 0.0]

    def to_data(self) -> dict[str, Any]:
        return {"name": self.name, "angle": degrees(self.angle), "kind": self.kind, "element": self.element}


class UnitDomain(object):
    """The cut-out unit: one or more components, plus what was clipped into it.

    Attributes
    ----------
    group : SymmetryGroup
        The group the unit is a fundamental region of.
    seams : list[Seam]
        Empty for the trivial group; otherwise ``[A, B]``.
    components : list[(outer, holes)]
        Usually one. A domain that is not star-shaped about the centre can fall
        apart into several pieces inside the wedge; each is meshed on its own.
    guides : list of open polylines
    poles : list of points
    poles_on_seams : list of points
        Poles exactly on a seam. Not handed to the route (a pole on a wall is not
        something either route accepts); reported so a caller can decide.
    eps : float
    """

    def __init__(
        self,
        group: SymmetryGroup,
        seams: list[Seam],
        components: list[tuple[list[Sequence[float]], list[list[Sequence[float]]]]],
        guides: list[Sequence[Any]],
        poles: list[Sequence[float]],
        poles_on_seams: list[Sequence[float]],
        eps: float,
        wedge: float,
    ) -> None:
        self.group = group
        self.seams = seams
        self.components = components
        self.guides = guides
        self.poles = poles
        self.poles_on_seams = poles_on_seams
        self.eps = eps
        self.wedge = wedge

    @property
    def centre(self) -> list[float]:
        return self.group.centre

    def seam(self, name: str) -> Seam:
        for s in self.seams:
            if s.name == name:
                return s
        raise KeyError(name)

    def locate(self, point: Sequence[float], eps: float | None = None) -> list[tuple[str, float]]:
        """``[(seam name, t), ...]`` for every seam ``point`` lies on."""
        eps = self.eps if eps is None else eps
        out = []
        for s in self.seams:
            t = s.locate(point, eps)
            if t is not None:
                out.append((s.name, t))
        return out

    def _line_of(self, a: Sequence[float], b: Sequence[float]) -> int | None:
        """Index of the seam line both points lie on, or ``None``."""
        lines = self._lines()
        for index, (dx, dy) in enumerate(lines):
            ok = True
            for p in (a, b):
                px, py = p[0] - self.centre[0], p[1] - self.centre[1]
                if abs(dx * py - dy * px) > self.eps:
                    ok = False
                    break
            if ok:
                return index
        return None

    def _lines(self) -> list[tuple[float, float]]:
        out = []
        for s in self.seams:
            dx, dy = s.direction
            if any(abs(dx * ey - dy * ex) <= 1e-12 for ex, ey in out):
                continue
            out.append((dx, dy))
        return out

    def junctions(self) -> list[Sequence[float]]:
        """Corners of the unit on a seam -- where a seam meets a wall, and the apex.

        Every unit mesh must have a vertex at each: the copies are glued along the
        seams, and a seam that ends in the middle of a coarse edge cannot be glued.
        """
        out = []
        for outer, holes in self.components:
            for loop in [outer] + holes:
                n = len(loop)
                for i in range(n):
                    if not self.locate(loop[i]):
                        continue
                    before = self._line_of(loop[i - 1], loop[i])
                    after = self._line_of(loop[i], loop[(i + 1) % n])
                    if before != after:
                        out.append(loop[i])
        return out

    def seam_runs(self) -> dict[str, list[list[float]]]:
        """``{seam name: [[t_lo, t_hi], ...]}`` -- where each seam bounds the unit."""
        out = {}
        for s in self.seams:
            out[s.name] = []
            for lo, hi in self._runs(s):
                out[s.name].append([lo, hi])
        return out

    def _runs(self, s: Seam) -> list[list[float]]:
        segments = []
        for outer, holes in self.components:
            for loop in [outer] + holes:
                n = len(loop)
                for i in range(n):
                    a, b = loop[i], loop[(i + 1) % n]
                    if self._line_of(a, b) is None:
                        continue
                    dx, dy = s.direction
                    if max(abs(dx * (q[1] - self.centre[1]) - dy * (q[0] - self.centre[0])) for q in (a, b)) > self.eps:
                        continue
                    ta = dx * (a[0] - self.centre[0]) + dy * (a[1] - self.centre[1])
                    tb = dx * (b[0] - self.centre[0]) + dy * (b[1] - self.centre[1])
                    lo, hi = max(0.0, min(ta, tb)), max(ta, tb)
                    if hi - lo > self.eps:
                        segments.append((lo, hi))
        segments.sort()
        merged = []
        for lo, hi in segments:
            if merged and lo <= merged[-1][1] + self.eps:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        return merged

    def seam_segments(self) -> list[list[list[float]]]:
        """The seams as polylines, where they bound the unit."""
        runs = []
        for s in self.seams:
            segments = []
            for outer, holes in self.components:
                for loop in [outer] + holes:
                    n = len(loop)
                    for i in range(n):
                        a, b = loop[i], loop[(i + 1) % n]
                        if self._line_of(a, b) is None:
                            continue
                        dx, dy = s.direction
                        ta = dx * (a[0] - self.centre[0]) + dy * (a[1] - self.centre[1])
                        tb = dx * (b[0] - self.centre[0]) + dy * (b[1] - self.centre[1])
                        if max(abs(dx * (q[1] - self.centre[1]) - dy * (q[0] - self.centre[0])) for q in (a, b)) > self.eps:
                            continue
                        lo, hi = max(0.0, min(ta, tb)), max(ta, tb)
                        if hi - lo > self.eps:
                            segments.append((lo, hi))
            segments.sort()
            merged = []
            for lo, hi in segments:
                if merged and lo <= merged[-1][1] + self.eps:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            runs += [[s.point_at(lo), s.point_at(hi)] for lo, hi in merged]
        return runs

    def to_data(self) -> dict[str, Any]:
        return {
            "group": self.group.to_data(),
            "seams": [s.to_data() for s in self.seams],
            "wedge": degrees(self.wedge),
            "eps": self.eps,
            "components": [[outer, holes] for outer, holes in self.components],
            "guides": self.guides,
            "poles": self.poles,
            "poles_on_seams": self.poles_on_seams,
        }


def _choose_rotation_seam(domain: Domain, group: SymmetryGroup, eps: float, samples: int = 180) -> float:
    return rotation_seam_candidates(domain, group, eps, samples)[0]


def rotation_seam_candidates(domain: Domain, group: SymmetryGroup, eps: float, samples: int = 180, count: int | None = None) -> list[float]:
    """Seam angles for a rotation-only group, best first. See the scoring below.

    Nearly equal angles are collapsed so the list offers genuinely different
    seams: a caller retrying after a failed match should not retry a hair away.
    """
    scored = _score_rotation_seams(domain, group, eps, samples)
    span = 2 * pi / group.n
    out = []
    for score, theta in sorted(scored):
        if all(min(abs(theta - o), span - abs(theta - o)) > span / 12.0 for o in out):
            out.append(theta)
        if count and len(out) >= count:
            break
    return out


def _score_rotation_seams(domain: Domain, group: SymmetryGroup, eps: float, samples: int = 180) -> list[tuple[float, float]]:
    """``[(score, angle)]`` for every sampled seam angle of a rotation-only group; lower is better."""
    c = group.centre
    span = 2 * pi / group.n
    radius = 2.0 * (domain.diagonal or 1.0)
    scored = []
    for k in range(samples):
        theta = span * k / samples
        ux, uy = cos(theta), sin(theta)
        far = [c[0] + radius * ux, c[1] + radius * uy]
        score = 0.0
        for loop, weight, closed in [(domain.outer, 0.0, True)] + [(h, 1.0, True) for h in domain.inners] + [(g, 1.0, False) for g in domain.guides]:
            crossings = _count_crossings(c, far, loop, closed)
            if loop is domain.outer:
                score += 2.0 * max(0, crossings - 1)
            else:
                score += weight * crossings
        for p in domain.poles:
            px, py = p[0] - c[0], p[1] - c[1]
            if abs(ux * py - uy * px) < 10 * eps and ux * px + uy * py > -eps:
                score += 5.0
        for p in domain.outer:
            px, py = p[0] - c[0], p[1] - c[1]
            if ux * px + uy * py > 0 and abs(ux * py - uy * px) < 0.02 * (domain.diagonal or 1.0):
                score += 0.5
        scored.append((round(score, 9), theta))
    return scored


def _count_crossings(a: Sequence[float], b: Sequence[float], loop: Sequence[Sequence[float]], closed: bool) -> int:
    n = len(loop)
    last = n if closed else n - 1
    count = 0
    for i in range(last):
        p, q = loop[i], loop[(i + 1) % n]
        d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        d2 = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
        d3 = (q[0] - p[0]) * (a[1] - p[1]) - (q[1] - p[1]) * (a[0] - p[0])
        d4 = (q[0] - p[0]) * (b[1] - p[1]) - (q[1] - p[1]) * (b[0] - p[0])
        if (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0):
            count += 1
    return count


def cut_unit(
    domain: Domain,
    group: SymmetryGroup,
    centre: str = "route",
    seam: float | None = None,
    eps: float | None = None,
) -> UnitDomain:
    """Cut one fundamental region of ``group`` out of ``domain``.

    Parameters
    ----------
    domain : .domain.Domain
    group : .group.SymmetryGroup
        The group to enforce -- usually ``report.group.subgroup(keys)``.
    centre : {'route'}
        How the centre is treated. Only ``'route'`` (the apex is left to the
        route) exists yet.
    seam : float, optional
        Rotation-only groups: the angle of seam A, in radians. Chosen
        automatically when omitted. Ignored for groups with mirrors, whose seams
        are their axes.
    eps : float, optional
        On-line tolerance. Default ``1e-7`` times the domain diagonal.

    Returns
    -------
    UnitDomain
    """
    if centre not in ("route", None):
        raise NotImplementedError("centre={!r} is not implemented yet -- only 'route'".format(centre))
    diagonal = domain.diagonal or 1.0
    eps = 1e-7 * diagonal if eps is None else eps
    c = group.centre

    outer = oriented(open_loop(domain.outer), ccw=True)
    holes = [oriented(open_loop(h), ccw=False) for h in domain.inners if len(h) >= 3]

    if group.is_trivial:
        return UnitDomain(group, [], [(outer, holes)], [list(g) for g in domain.guides], [list(p) for p in domain.poles], [], eps, 2 * pi)

    if group.mirrors:
        a = group.axis
        wedge = pi / group.n
        b = a + wedge
        seam_a = Seam("A", a, "mirror", _mirror(a).key, c)
        seam_b = Seam("B", b, "mirror", _mirror(b).key, c)
    else:
        a = _choose_rotation_seam(domain, group, eps) if seam is None else float(seam)
        wedge = 2 * pi / group.n
        b = a + wedge
        key = _rotation(wedge).key
        seam_a = Seam("A", a, "rotation", key, c)
        seam_b = Seam("B", b, "rotation", key, c)

    # Left of A (angles a .. a + pi), then -- unless the wedge IS that half-plane --
    # left of B reversed (angles b - pi .. b).
    clips = [((cos(a), sin(a)))]
    if abs(wedge - pi) > 1e-12:
        clips.append((-cos(b), -sin(b)))

    loops = [outer] + holes
    guides = [list(g) for g in domain.guides]
    for direction in clips:
        loops = clip_halfplane(loops, c, direction, eps)
        guides = [piece for g in guides for piece in clip_polyline(g, c, direction, eps)]

    outers = [loop for loop in loops if signed_area(loop) > 0]
    inner_loops = [loop for loop in loops if signed_area(loop) < 0]
    components = [(o, []) for o in outers]
    for h in inner_loops:
        tx = sum(p[0] for p in h) / len(h)
        ty = sum(p[1] for p in h) / len(h)
        owners = [i for i, (o, _) in enumerate(components) if point_in_polygon(tx, ty, o)]
        if not owners:
            raise ValueError("a hole of the unit is not inside any of its outer loops")
        owner = min(owners, key=lambda i: area_centroid(components[i][0])[0])
        components[owner][1].append(h)
    if not components:
        raise ValueError("the unit is empty -- is the centre {} inside the domain?".format(c))

    unit = UnitDomain(group, [seam_a, seam_b], components, guides, [], [], eps, wedge)
    for p in domain.poles:
        on = unit.locate(p, eps=max(eps, 1e-9))
        inside = all(dx * (p[1] - c[1]) - dy * (p[0] - c[0]) > -eps for dx, dy in clips)
        if on:
            unit.poles_on_seams.append(list(p))
        elif inside:
            unit.poles.append(list(p))
    return unit


def subgroups_avoiding_poles(domain: Domain, group: SymmetryGroup, eps: float | None = None) -> list[SymmetryGroup]:
    """Subgroups of ``group``, largest first, whose seams do not run through a pole.

    What to suggest when a pole sits on a mirror axis. At equal order a group with
    mirrors is preferred -- its seams are forced, so the suggestion is stable.
    """
    out = []
    for g in sorted(group.subgroups(), key=lambda g: (-g.order, not g.mirrors)):
        if g.is_trivial:
            continue
        try:
            unit = cut_unit(domain, g, eps=eps)
        except ValueError:
            continue
        if not unit.poles_on_seams:
            out.append(g)
    return out


def wedge_angle_of(point: Sequence[float], centre: Sequence[float]) -> float:
    return atan2(point[1] - centre[1], point[0] - centre[0])
