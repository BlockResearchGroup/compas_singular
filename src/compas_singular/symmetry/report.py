"""What ``.detect.find_symmetry`` found, and the geometry to show it with."""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import cos
from math import pi
from math import sin
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas_singular.symmetry._geometry import point_in_polygon

if TYPE_CHECKING:
    from compas.geometry import Polyline
    from compas_singular.symmetry.domain import Domain
    from compas_singular.symmetry.group import SymmetryGroup


__all__ = ["SymmetryReport"]


class SymmetryReport(object):
    """The detected symmetry of a domain.

    Attributes
    ----------
    group : .group.SymmetryGroup
        The largest group whose every element passed.
    centre : [x, y, z]
    keys : list[str]
        The names a script picks a unit by -- ``R90``, ``M45`` ...
    deviations : dict
        ``key -> (worst distance, (tag, point))`` for every element of
        ``group``. How far from exact the drawing is.
    near_misses : list
        ``(key, worst distance, (tag, point))`` for elements that did NOT pass
        but came within the near-miss limit -- almost always a drawing slightly
        off, and the tag says which curve.
    tol, near : float
    circle_like : bool
        Every tested rotation passed: the domain is (sampled from) a circle, and
        the reported order is only the highest one tested.
    domain : .domain.Domain
    """

    def __init__(
        self,
        domain: Domain,
        group: SymmetryGroup,
        tol: float,
        near: float,
        include: Sequence[str],
        deviations: dict[str, tuple[float, Any]],
        near_misses: list[tuple[str, float, Any]],
        circle_like: bool,
        max_order: int,
    ) -> None:
        self.domain = domain
        self.group = group
        self.tol = tol
        self.near = near
        self.include = tuple(include)
        self.deviations = deviations
        self.near_misses = near_misses
        self.circle_like = circle_like
        self.max_order = max_order

    # ------------------------------------------------------------------

    @property
    def centre(self) -> list[float]:
        return list(self.group.centre)

    @property
    def keys(self) -> list[str]:
        return self.group.keys()

    def __repr__(self) -> str:
        return "<Symmetry {} about ({:.3f}, {:.3f}): {}>".format(self.group.name, self.centre[0], self.centre[1], " ".join(self.keys) or "-")

    def centre_inside(self) -> bool:
        return self.domain.contains(self.centre)

    def notes(self) -> list[str]:
        """Plain-language remarks a user should read before choosing a unit."""
        out = []
        g = self.group
        if g.is_trivial:
            out.append("no symmetry detected within tol {:.3g}".format(self.tol))
        if self.circle_like:
            out.append("every rotation up to order {} passed: the domain is circle-like, choose the symmetry you want by keys".format(self.max_order))
        if g.n in (3, 6) and self.centre_inside():
            out.append("{}-fold symmetry about a centre inside the domain: any quad mesh with this symmetry has a singularity at the centre".format(g.n))
        worst = max([d for d, _ in self.deviations.values()] or [0.0])
        if worst > 0.0:
            out.append("worst deviation of the detected symmetry: {:.3g} (tol {:.3g})".format(worst, self.tol))
        for key, dev, where in self.near_misses[:5]:
            tag = where[0] if where else "?"
            out.append("near miss {}: {:.3g} off, worst on {}".format(key, dev, tag))
        return out

    def summary(self) -> str:
        lines = [repr(self)]
        lines += ["  " + note for note in self.notes()]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # per feature class
    # ------------------------------------------------------------------

    def by_feature(self) -> list[tuple[str, SymmetryReport]]:
        """The group with walls only, then adding holes, guides and poles in turn.

        Returns
        -------
        list[(label, SymmetryReport)]
            Only the stages that add something present in the domain. Comparing
            consecutive groups says which kind of feature breaks the symmetry.
        """
        from compas_singular.symmetry.detect import find_symmetry

        stages = []
        include = []
        present = {"walls": bool(self.domain.outer), "holes": bool(self.domain.inners), "guides": bool(self.domain.guides), "poles": bool(self.domain.poles)}
        for kind, label in (("walls", "walls"), ("holes", "+holes"), ("guides", "+guides"), ("poles", "+poles")):
            if kind not in self.include:
                continue
            include.append(kind)
            if not present[kind] and kind != "walls":
                continue
            report = find_symmetry(domain=self.domain, tol=self.tol, near=self.near, include=tuple(include), max_order=self.max_order, centre=self.centre)
            stages.append((label, report))
        return stages

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------

    def geometry(self, arc_radius: float | None = None) -> dict[str, Any]:
        """Simple compas geometry that shows the symmetry.

        Returns
        -------
        dict
            ``'centre'``: a ``Point``. ``'mirrors'``: one ``Line`` per mirror
            axis, clipped to the outer wall. ``'rotations'``: a ``Polyline`` arc
            from the first mirror axis (or +x) through the smallest rotation
            angle, with a short tick at its end. ``'near_misses'``: a ``Point``
            at the worst sample of each near miss.
        """
        from compas.geometry import Line
        from compas.geometry import Point
        from compas.geometry import Polyline

        g = self.group
        c = self.centre
        diagonal = self.domain.diagonal or 1.0
        out = {"centre": Point(*c), "mirrors": [], "rotations": [], "near_misses": []}

        for e in g.mirror_elements():
            a, b = self._clip_line(e.angle)
            out["mirrors"].append(Line(a, b))

        if g.n > 1:
            radius = arc_radius or 0.12 * diagonal
            start = g.axis if g.mirrors else 0.0
            sweep = 2 * pi / g.n
            steps = max(8, int(32 * sweep / (2 * pi)) + 4)
            arc = [[c[0] + radius * cos(start + sweep * i / steps), c[1] + radius * sin(start + sweep * i / steps), 0.0] for i in range(steps + 1)]
            out["rotations"].append(Polyline(arc))
            end = start + sweep
            tick = [[c[0] + 0.8 * radius * cos(end), c[1] + 0.8 * radius * sin(end), 0.0], [c[0] + 1.2 * radius * cos(end), c[1] + 1.2 * radius * sin(end), 0.0]]
            out["rotations"].append(Polyline(tick))

        for key, dev, where in self.near_misses:
            if where:
                out["near_misses"].append(Point(*where[1]))
        return out

    def _clip_line(self, angle: float) -> tuple[list[float], list[float]]:
        """The axis through the centre at ``angle``, clipped to the outer wall."""
        c = self.centre
        ux, uy = cos(angle), sin(angle)
        loop = self.domain.outer
        diagonal = self.domain.diagonal or 1.0
        ts = []
        n = len(loop)
        for i in range(n):
            ax, ay = loop[i][0], loop[i][1]
            bx, by = loop[(i + 1) % n][0], loop[(i + 1) % n][1]
            ex, ey = bx - ax, by - ay
            den = ux * ey - uy * ex
            if abs(den) < 1e-15:
                continue
            wx, wy = ax - c[0], ay - c[1]
            t = (wx * ey - wy * ex) / den
            s = (wx * uy - wy * ux) / den
            if -1e-12 <= s <= 1 + 1e-12:
                ts.append(t)
        if len(ts) < 2:
            ts = [-0.5 * diagonal, 0.5 * diagonal]
        lo, hi = min(ts), max(ts)
        return [c[0] + lo * ux, c[1] + lo * uy, 0.0], [c[0] + hi * ux, c[1] + hi * uy, 0.0]

    def unit_outline(self, keys: Sequence[str] | None = None, centre: str = "route", seam: Any = None) -> dict[str, list[Polyline]]:
        """The unit region for ``keys`` as closed ``Polyline``s, before any meshing.

        Returns
        -------
        dict
            ``'loops'``: the unit's walls (outer loops first), ``'seams'``: the
            seam segments, as ``Polyline``s.
        """
        from compas.geometry import Polyline
        from compas_singular.symmetry.cut import cut_unit

        unit = cut_unit(self.domain, self.group.subgroup(keys), centre=centre, seam=seam)
        loops = []
        for outer, holes in unit.components:
            loops.append(Polyline(outer + [outer[0]]))
            for hole in holes:
                loops.append(Polyline(hole + [hole[0]]))
        seams = [Polyline(s) for s in unit.seam_segments()]
        return {"loops": loops, "seams": seams}

    # ------------------------------------------------------------------

    def to_data(self) -> dict[str, Any]:
        return {
            "group": self.group.to_data(),
            "tol": self.tol,
            "near": self.near,
            "include": list(self.include),
            "circle_like": self.circle_like,
            "deviations": {k: v[0] for k, v in self.deviations.items()},
            "near_misses": [[k, d, w[0] if w else None] for k, d, w in self.near_misses],
            "domain": self.domain.to_data(),
        }


def _inside(point: Sequence[float], loop: Sequence[Sequence[float]]) -> bool:
    return point_in_polygon(point[0], point[1], loop)
