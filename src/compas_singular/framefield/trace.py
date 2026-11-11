"""Separatrix tracing from singularities, reflex corners and uncut holes, with guarded stopping rules."""

from __future__ import annotations

from collections import namedtuple
from math import atan2
from math import cos
from math import pi
from math import sin
from typing import Any

from compas.geometry import distance_point_point
from compas.geometry import intersection_segment_segment_xy
from compas.itertools import pairwise
from compas_singular.framefield.constraints import PERIOD
from compas_singular.framefield.field import CrossField
from compas_singular.framefield.field import wrap_to_period
from compas_singular.framefield.locator import AMBIGUOUS
from compas_singular.framefield.locator import PointLocator
from compas_singular.geometry.polyline import closest_on_polyline
from compas_singular.geometry.polyline import loop_arc_lengths
from compas_singular.geometry.polyline import signed_area

__all__ = ["Tracer", "Separatrix"]


#: ``reason`` is one of 'boundary', 'singularity', 'cycle', 'length', 'lost'.
Separatrix = namedtuple("Separatrix", "points source sink reason")

#: Two cut sites whose field alignment differs by less than this are equally
#: aligned, and ``Tracer._cut_launches_from`` picks between them on arc
#: length instead.
ALIGN_TIE = 0.02

#: Probe circles ``Tracer.launch_directions`` tries, as ``(radius as a
#: multiple of the background spacing, samples per lap)``, in order. One circle
#: is not reliable: too small and it sits where ``|u| -> 0`` and the angle is
#: noise, too large and it leaves the domain or swallows a neighbour, and too
#: few samples alias a period away.
PROBES = [(s, n) for n in (360, 720, 1440) for s in (1.2, 0.9, 1.6, 0.6, 2.0, 0.45)]


class _Trail(object):
    """The points a trace has passed, bucketed, for the limit-cycle test."""

    def __init__(self, tol: float) -> None:
        self.tol = tol
        # a cell a little wider than ``tol``: two points closer than ``tol``
        # are then never more than one cell apart
        self.cell = tol * 1.01
        self.grid = {}
        self.added = 0

    def _key(self, p: list[float]) -> tuple[int, int]:
        return int(p[0] // self.cell), int(p[1] // self.cell)

    def revisits(self, points: list[list[float]], skip: int = 20) -> bool:
        """Whether ``points[-1]`` is within ``tol`` of any of ``points[:-skip]``."""
        for p in points[self.added : len(points) - skip]:
            self.grid.setdefault(self._key(p), []).append(p)
        self.added = max(self.added, len(points) - skip)
        head = points[-1]
        i, j = self._key(head)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for p in self.grid.get((i + di, j + dj), ()):
                    if distance_point_point(head, p) < self.tol:
                        return True
        return False


class Tracer(object):
    """Traces the separatrices of a ``CrossField``.

    Parameters
    ----------
    field : CrossField
    step : float, optional
        Integration step. Defaults to a third of the background spacing, so a
        step never crosses more than one triangle.
    max_length : float, optional
        Arc-length cap per trace. Defaults to 40x the domain diagonal.
    singularity_points : dict[int, [x, y, z]], optional
        Where to launch from, per singular face. Defaults to the face centroid.
        ``symmetry.snap_singularities`` sets it to the field's own ``|u|``
        minimum, which -- unlike the centroids -- is an exact orbit on a
        symmetric domain.
    """

    def __init__(
        self,
        field: CrossField,
        step: float | None = None,
        max_length: float | None = None,
        singularity_points: dict[int, list[float]] | None = None,
    ) -> None:
        self.field = field
        self.background = field.background
        self.mesh = field.background.mesh
        self.locator: PointLocator = field.locator()
        self.step = step or self.background.target_length / 3.0
        self.singularity_points = dict(singularity_points or {})

        self.diagonal = self.background.diagonal
        self.max_length = max_length or 40.0 * self.diagonal

    def _loops(self) -> list[list[list[float]]]:
        return self.background.loops

    # ------------------------------------------------------------------
    # field sampling
    # ------------------------------------------------------------------

    def locate(self, point: list[float], hint: int | None = None) -> tuple[int, tuple[float, float, float]] | None:
        """Containing face and barycentric coordinates, or ``None``. See
        ``PointLocator.locate``."""
        return self.locator.locate(point, hint)

    def direction_at(
        self,
        point: list[float],
        reference: list[float],
        hint: int | None = None,
    ) -> tuple[list[float] | None, int | None]:
        """Unit field direction at ``point`` on the arm closest to ``reference``. ``(direction, fkey)`` or ``(None, None)``."""
        found = self.locator.locate(point, hint)
        if found is None:
            return None, None
        fkey, bary = found
        ref_angle = atan2(reference[1], reference[0])
        theta = self.field.angle_in_face(fkey, bary, reference=ref_angle)
        theta += PERIOD * round((ref_angle - theta) / PERIOD)
        return [cos(theta), sin(theta), 0.0], fkey

    # ------------------------------------------------------------------
    # launch directions
    # ------------------------------------------------------------------

    def singularity_positions(self) -> dict[int, list[float]]:
        """Where each singularity is: ``singularity_points`` where given, else
        the face centroid. What traces launch from and snap back onto."""
        return {fkey: self.singularity_points.get(fkey) or self.mesh.face_centroid(fkey) for fkey, _ in self.field.singularities()}

    #: Old name, kept for callers outside the package.
    _singularity_points = singularity_positions

    def launch_directions(self, centre: list[float], index: int) -> list[list[float]]:
        """Directions in which separatrices leave a singularity: ``4 - index`` arms, found on probe circles."""
        target = 4 - index
        best = []
        for scale, samples in PROBES:
            radius = self.background.target_length * scale
            phis = [2.0 * pi * i / samples for i in range(samples + 1)]  # closes the lap
            points = [[centre[0] + cos(phi) * radius, centre[1] + sin(phi) * radius, 0.0] for phi in phis]
            found = self.locator.locate_many(points)
            if any(f is None for f in found):
                continue  # the circle left the domain
            residual = []
            theta = None
            for point, phi, hit in zip(points, phis, found):
                if hit is AMBIGUOUS:
                    hit = self.locator.locate(point)
                raw = self.field.angle_in_face(hit[0], hit[1], reference=phi)
                theta = raw if theta is None else theta + wrap_to_period(raw - theta)
                residual.append(phi - theta)

            found_dirs = self._level_crossings(phis, residual, target)
            if len(found_dirs) == target:
                return [[cos(a), sin(a), 0.0] for a in found_dirs]
            if len(found_dirs) > len(best):
                best = found_dirs

        return [[cos(a), sin(a), 0.0] for a in best]

    @staticmethod
    def _level_crossings(phis: list[float], residual: list[float], target: int) -> list[float]:
        """Where the residual's running maximum first reaches each of the next ``target`` period multiples."""
        out = []
        start = residual[0]
        levels = []
        k = int(start // PERIOD) + 1
        while len(levels) < target:
            levels.append(k * PERIOD)
            k += 1

        running = start
        li = 0
        for i in range(1, len(residual)):
            previous = running
            running = max(running, residual[i])
            while li < len(levels) and levels[li] <= running:
                level = levels[li]
                span = running - previous
                t = 0.0 if span <= 0 else max(0.0, min(1.0, (level - previous) / span))
                out.append(phis[i - 1] + t * (phis[i] - phis[i - 1]))
                li += 1
        return out

    def _inward_normal(self, loop: list[list[float]], i: int) -> list[float] | None:
        """Unit normal at ``loop[i]`` pointing into the domain, or ``None``.

        Taken from the neighbours' chord, and its sign decided by probing which
        side is inside -- so it is right on a hole that is not convex.
        """
        n = len(loop)
        a, b = loop[i - 1], loop[(i + 1) % n]
        tx, ty = b[0] - a[0], b[1] - a[1]
        length = (tx * tx + ty * ty) ** 0.5
        if length < 1e-12:
            return None
        for sign in (1.0, -1.0):
            normal = [-ty / length * sign, tx / length * sign, 0.0]
            probe = [loop[i][0] + normal[0] * self.background.target_length * 0.6, loop[i][1] + normal[1] * self.background.target_length * 0.6, 0.0]
            if self.locator.locate(probe) is not None:
                return normal
        return None

    def _cut_launches_from(self, loop: list[list[float]], count: int) -> list[tuple[list[float], list[float]]]:
        """``count`` launch sites spread by arc length around ``loop``, each where the field is most normal to it."""
        _seg, cum = loop_arc_lengths(loop)
        total = cum[-1]
        if total <= 0.0:
            return []

        step = self.background.target_length * 0.8
        window = total / (2.0 * count)
        launches = []
        for k in range(count):
            target = total * k / count
            candidates = []
            for i in range(len(loop)):
                offset = abs(cum[i] - target)
                offset = min(offset, total - offset)
                if offset > window:
                    continue
                normal = self._inward_normal(loop, i)
                if normal is None:
                    continue
                probe = [loop[i][0] + normal[0] * step, loop[i][1] + normal[1] * step, 0.0]
                found = self.locator.locate(probe)
                if found is None:
                    continue
                fkey, bary = found
                theta = self.field.angle_in_face(fkey, bary)
                arm, score = None, -2.0
                for a in range(4):
                    angle = theta + a * PERIOD
                    d = [cos(angle), sin(angle), 0.0]
                    dot = d[0] * normal[0] + d[1] * normal[1]
                    if dot > score:
                        arm, score = d, dot
                if arm is not None:
                    candidates.append((score, offset, list(loop[i]), arm))
            if not candidates:
                continue
            top = max(c[0] for c in candidates)
            _, _, point, arm = min((c for c in candidates if c[0] > top - ALIGN_TIE), key=lambda c: c[1])
            launches.append((point, arm))
        return launches

    def hole_launches(
        self,
        corner_limit: float = pi / 12.0,
        count: int = 4,
    ) -> list[tuple[list[float], list[float]]]:
        """Cuts opening a hole that emits no separatrix of its own, at least two per hole.

        Parameters
        ----------
        corner_limit : float, optional
            A hole with a corner above this turn is left alone; its corners
            already supply the cuts.
        count : int, optional
            Cuts per hole.

        Returns
        -------
        list[(point, direction)]
        """
        from compas_singular.framefield.separatrix_network import boundary_corners

        launches = []
        for loop in self.background.inners:
            if boundary_corners(loop, corner_limit):
                continue
            launches.extend(self._cut_launches_from(loop, count))
        return launches

    def corner_points(self, corner_limit: float = pi / 12.0) -> list[list[float]]:
        """Every boundary corner, as a flat list of points."""
        from compas_singular.framefield.separatrix_network import boundary_corners

        out = []
        for loop in self._loops():
            for i in boundary_corners(loop, corner_limit):
                out.append(list(loop[i]))
        return out

    # ------------------------------------------------------------------
    # integration
    # ------------------------------------------------------------------

    def trace(
        self,
        start: list[float],
        direction: list[float],
        source: int | str | None = None,
        singular_points: dict[int, list[float]] | None = None,
    ) -> Separatrix:
        """Integrate one separatrix from ``start`` along ``direction`` (RK2) until a stopping rule fires.

        Parameters
        ----------
        source : int or str, optional
            The singular face launched from (never counted as a hit early on),
            ``'hole'`` for a hole cut, or ``None`` for a corner launch.
        singular_points : dict[int, [x, y, z]], optional
            Singularities a trace may end on.
        """
        singular_points = singular_points or {}
        stop_radius = self.background.target_length * 1.2
        h = self.step
        trail = _Trail(stop_radius * 0.5)

        points = [list(start)]
        d = list(direction)
        fkey = None
        length = 0.0
        reason = "length"
        sink = None

        while length < self.max_length:
            d1, f1 = self.direction_at(points[-1], d, fkey)
            if d1 is None:
                reason = "lost"
                break
            mid = [points[-1][k] + d1[k] * h * 0.5 for k in range(3)]
            d2, _ = self.direction_at(mid, d1, f1)
            if d2 is None:
                d2 = d1
            nxt = [points[-1][k] + d2[k] * h for k in range(3)]

            found = self.locator.locate(nxt, f1)
            if found is None:
                clipped = self._clip_to_boundary(points[-1], nxt)
                if clipped is not None:
                    points.append(clipped)
                    length += distance_point_point(points[-2], clipped)
                reason = "boundary"
                break

            length += distance_point_point(points[-1], nxt)
            points.append(nxt)
            d, fkey = d2, found[0]

            hit = None
            for skey, centre in singular_points.items():
                if skey == source and length < stop_radius * 3.0:
                    continue  # our own launch point
                if distance_point_point(nxt, centre) < stop_radius:
                    hit = skey
                    break
            if hit is not None:
                points.append(list(singular_points[hit]))
                reason, sink = "singularity", hit
                break

            # A trace can run ALONG a wall without ever leaving the domain, and
            # a line lying on a wall is something the face search cannot
            # separate from the wall. Stop on proximity to the wall -- not to a
            # corner, which would stop traces that merely pass one.
            if length > stop_radius * 2.0:
                near = self._project_to_boundary(nxt)
                if near is not None and distance_point_point(nxt, near) < h * 0.75:
                    points.append(near)
                    reason = "boundary"
                    break

            if len(points) > 40 and trail.revisits(points):
                reason = "cycle"
                break

        return Separatrix(points, source, sink, reason)

    def _project_to_boundary(self, point: list[float]) -> list[float] | None:
        """Closest point on any boundary loop."""
        best, best_d = None, float("inf")
        for loop in self._loops():
            _i, _t, q, d = closest_on_polyline(point, loop, closed=True)
            if d < best_d:
                best, best_d = q, d
        return best

    def _clip_to_boundary(self, inside: list[float], outside: list[float]) -> list[float] | None:
        """Where the segment ``inside -> outside`` leaves the domain."""
        best, best_d = None, float("inf")
        for loop in self._loops():
            for a, b in pairwise(loop + loop[:1]):
                x = intersection_segment_segment_xy((inside, outside), (a, b))
                if x is None:
                    continue
                x = [x[0], x[1], 0.0]
                d = distance_point_point(inside, x)
                if d < best_d:
                    best, best_d = x, d
        return best

    # ------------------------------------------------------------------
    # the whole set
    # ------------------------------------------------------------------

    def corner_launches(self, corner_limit: float = pi / 12.0) -> list[tuple[list[float], list[float]]]:
        """Separatrices emitted by the domain's corners: ``round(alpha / 90) - 1`` per corner.

        Returns
        -------
        list[(point, direction)]
        """
        from compas_singular.framefield.separatrix_network import boundary_corners

        launches = []
        for loop_index, loop in enumerate(self._loops()):
            n = len(loop)
            is_hole = loop_index > 0
            ccw_loop = signed_area(loop) > 0
            for i in boundary_corners(loop, corner_limit):
                here = loop[i]
                prev, nxt = loop[(i - 1) % n], loop[(i + 1) % n]
                back = [prev[0] - here[0], prev[1] - here[1], 0.0]
                fwd = [nxt[0] - here[0], nxt[1] - here[1], 0.0]
                interior_angle, ccw = self._interior_angle(back, fwd, ccw_loop)
                if is_hole:
                    # a hole's own inside is outside the domain
                    interior_angle = 2.0 * pi - interior_angle
                    ccw = not ccw
                k = int(round(interior_angle / PERIOD))
                if k < 2:
                    continue

                # Into the domain is the outgoing wall rotated by HALF the
                # interior angle; the bisector of the two wall vectors points
                # the wrong way at a reflex corner.
                a_fwd = atan2(fwd[1], fwd[0])
                turn = 1.0 if ccw else -1.0
                a_in = a_fwd + turn * interior_angle * 0.5
                inset = self.background.target_length * 0.8
                probe = [here[0] + cos(a_in) * inset, here[1] + sin(a_in) * inset, 0.0]
                found = self.locator.locate(probe)
                if found is None:
                    continue
                fkey, bary = found
                theta = self.field.angle_in_face(fkey, bary)

                # the arms lying strictly inside the interior sector
                margin = 0.15
                candidates = []
                for arm in range(4):
                    a = theta + arm * PERIOD
                    rel = turn * (a - a_fwd) % (2.0 * pi)
                    if margin < rel < interior_angle - margin:
                        candidates.append(([cos(a), sin(a), 0.0], rel))
                candidates.sort(key=lambda c: c[1])

                # launched from the corner itself, stepped along the arm -- not
                # from the probe, which would shift the whole trace sideways
                step = self.background.target_length * 0.5
                launches.extend(([here[0] + d[0] * step, here[1] + d[1] * step, 0.0], d) for d, _ in candidates[: k - 1])
        return launches

    @staticmethod
    def _interior_angle(back: list[float], fwd: list[float], ccw: bool) -> tuple[float, bool]:
        """Interior angle at a corner, on the domain side of a loop wound ``ccw``.

        On a counter-clockwise loop the interior is on the left: the sweep from
        the OUTGOING edge round to the INCOMING one.
        """
        a_back = atan2(back[1], back[0])
        a_fwd = atan2(fwd[1], fwd[0])
        sweep = (a_back - a_fwd) % (2.0 * pi)
        return (sweep if ccw else 2.0 * pi - sweep), ccw

    def separatrices(self) -> tuple[list[Separatrix], dict[str, Any]]:
        """Every separatrix of the field: from singularities, corners and holes.

        Returns
        -------
        (list[Separatrix], dict)
            The traces, and a report: ``count``, ``reasons`` (per stop reason),
            ``arm_mismatch`` (``(fkey, found, expected)`` for every singularity
            whose launch count is wrong), ``hole_cuts`` and ``ok``.
        """
        singular_points = self.singularity_positions()
        indices = dict(self.field.singularities())

        out = []
        arm_mismatch = []
        for fkey, centre in singular_points.items():
            dirs = self.launch_directions(centre, indices[fkey])
            expected = 4 - indices[fkey]
            if len(dirs) != expected:
                arm_mismatch.append((fkey, len(dirs), expected))
            for d in dirs:
                start = [centre[0] + d[0] * self.background.target_length * 0.5, centre[1] + d[1] * self.background.target_length * 0.5, 0.0]
                out.append(self.trace(start, d, source=fkey, singular_points=singular_points))

        for start, d in self.corner_launches():
            out.append(self.trace(start, d, source=None, singular_points=singular_points))

        cuts = 0
        for start, d in self.hole_launches():
            # ``'hole'`` tells ``build_network`` this start sits on a smooth wall
            # and is a split point of it, where a corner launch (``None``) must
            # be snapped back onto its corner
            trace = self.trace(start, d, source="hole", singular_points=singular_points)
            # a cut that does not land on another wall is a dangling edge
            if trace.reason == "boundary":
                out.append(trace)
                cuts += 1

        reasons = {}
        for s in out:
            reasons[s.reason] = reasons.get(s.reason, 0) + 1

        report = {
            "count": len(out),
            "reasons": reasons,
            "arm_mismatch": arm_mismatch,
            "hole_cuts": cuts,
            "ok": not arm_mismatch and not reasons.get("lost") and not reasons.get("cycle"),
        }
        return out, report
