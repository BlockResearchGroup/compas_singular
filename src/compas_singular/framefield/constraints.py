"""Directional constraints on the field, pinning ``exp(4i theta)`` at background vertices."""

from __future__ import annotations

from cmath import exp as cexp
from cmath import phase
from collections import namedtuple
from math import atan2
from math import cos
from math import pi
from math import sin
from numbers import Number
from typing import TYPE_CHECKING
from typing import Any

from compas.geometry import normalize_vector
from compas.geometry import subtract_vectors
from compas_singular.geometry.polyline import closest_on_polyline

if TYPE_CHECKING:
    from compas_singular.framefield.background import BackgroundMesh


__all__ = ["Constraint", "PERIOD", "as_curve_list", "from_boundary", "from_curves", "representation"]


#: A cross's period: the angle it can be rotated by without changing.
PERIOD = pi / 2.0

#: ``weight`` of ``None`` means hard (eliminated from the solve); a float means a
#: least-squares penalty, so several constraints can disagree gracefully.
Constraint = namedtuple("Constraint", "vkey direction weight")

MODES = ("perpendicular", "tangent")


def representation(direction: list[float]) -> complex:
    """The 4-fold-symmetric complex representation of a direction,
    ``exp(i * 4 * theta)``. A direction and its 90 degree rotation have the same
    one -- that identity IS the cross field."""
    theta = atan2(direction[1], direction[0])
    return cexp(4j * theta)


def as_curve_list(curves: Any) -> list[Any]:
    """A list of curves, whether one curve or several were passed, decided by nesting depth."""
    if curves and isinstance(curves[0][0], Number):
        return [curves]
    return list(curves or [])


def from_boundary(
    background: BackgroundMesh,
    weight: float | None = None,
    corner_tolerance: float = 0.1,
) -> list[Constraint]:
    """Pin the field tangent to the wall at every boundary vertex; 45/135-degree corners are left free.

    Parameters
    ----------
    background : BackgroundMesh
    weight : float, optional
        ``None`` (default) for hard constraints.
    corner_tolerance : float, optional
        Averaged representations shorter than this leave the vertex free.

    Returns
    -------
    list[Constraint]
    """
    out = []
    for vkey, tangents in background.boundary_tangents().items():
        acc = sum((representation(t) for t in tangents), 0j) / len(tangents)
        if abs(acc) < corner_tolerance:
            continue
        theta = phase(acc) / 4.0
        out.append(Constraint(vkey, [cos(theta), sin(theta), 0.0], weight))
    return out


def from_curves(
    background: BackgroundMesh,
    curves: Any,
    mode: str = "perpendicular",
    band: float | None = None,
    weight: float | None = 1.0,
    skip_boundary: bool = True,
) -> list[Constraint]:
    """Pin the field along guide curves at every vertex within ``band``, independent of curve order.

    Parameters
    ----------
    background : BackgroundMesh
    curves : list[list[[x, y, z]]]
        The guide curves; a single curve may also be passed. Points may be
        lists, tuples or compas ``Point``s.
    mode : {'perpendicular', 'tangent'}, optional
        Whether mesh edges should run across the curve or along it. For a cross
        field the two are the same constraint; the parameter matters once the
        frame is not orthogonal.
    band : float, optional
        Only vertices within this distance of a curve are constrained. Defaults
        to ``background.target_length``.
    weight : float, optional
        Least-squares weight, or ``None`` for hard. Soft by default: a guide
        that disagrees with a wall should bend, not break the solve.
    skip_boundary : bool, optional
        Leave boundary vertices to ``from_boundary``; the wall must win.

    Returns
    -------
    list[Constraint]
    """
    if mode not in MODES:
        raise ValueError("mode must be 'perpendicular' or 'tangent', not {!r}".format(mode))
    curves = as_curve_list(curves)

    radius = background.target_length if band is None else band
    boundary = background.boundary_vertices() if skip_boundary else set()
    mesh = background.mesh
    reach = radius * (1.0 + 1e-9) + 1e-12

    out = []
    for curve in curves:
        pts = [[float(p[0]), float(p[1]), 0.0] for p in curve]
        if not pts:
            continue
        lo_x = min(p[0] for p in pts) - reach
        hi_x = max(p[0] for p in pts) + reach
        lo_y = min(p[1] for p in pts) - reach
        hi_y = max(p[1] for p in pts) + reach
        for vkey in mesh.vertices():
            if vkey in boundary:
                continue
            p = mesh.vertex_coordinates(vkey)
            if not (lo_x <= p[0] <= hi_x and lo_y <= p[1] <= hi_y):
                continue  # further than ``band`` from the whole curve
            i, _t, _q, d = closest_on_polyline(p, pts)
            if d > radius:
                continue
            tangent = normalize_vector(subtract_vectors(pts[i + 1], pts[i]))
            direction = [-tangent[1], tangent[0], 0.0] if mode == "perpendicular" else tangent
            out.append(Constraint(vkey, direction, weight))
    return _merge_hard(out) if weight is None else out


def _merge_hard(constraints: list[Constraint]) -> list[Constraint]:
    """One hard constraint per vertex, averaged in the 4th-power representation."""
    grouped = {}
    for c in constraints:
        grouped.setdefault(c.vkey, []).append(c)

    out = []
    for group in grouped.values():
        if len(group) == 1:
            out.append(group[0])
            continue
        acc = sum((representation(c.direction) for c in group), 0j) / len(group)
        if abs(acc) < 1e-9:
            continue
        theta = phase(acc) / 4.0
        out.append(Constraint(group[0].vkey, [cos(theta), sin(theta), 0.0], None))
    return out


def _assert_mode_identity() -> bool:
    """The 4-fold identity the ``mode`` parameter relies on. Called by the tests."""
    d = normalize_vector([0.37, 0.93, 0.0])
    perp = [-d[1], d[0], 0.0]
    assert abs(representation(d) - representation(perp)) < 1e-12
    assert abs(representation(d) - representation([-d[0], -d[1], 0.0])) < 1e-12
    return True
