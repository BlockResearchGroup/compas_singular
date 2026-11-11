"""Directional guide metrics per face the guide crosses: along, across and off-centre.

Compare with a no-guide control.
"""

from __future__ import annotations

from math import acos
from math import degrees
from typing import TYPE_CHECKING
from typing import Any

from compas_singular.geometry.polyline import closest_on_polyline

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh


__all__ = ["guide_metrics", "format_guide_metrics"]

#: A guide station is "tilted" past this many degrees off horizontal. See the
#: design_notes/framefield.md (guides.py) on why this is a proxy and the control is the real defence.
STEEP = 12.0

#: ``across`` counts as on target within this many degrees of 90.
ACROSS_TOLERANCE = 15.0


def _median(values: list[float]) -> float | None:
    s = sorted(values)
    k = len(s)
    if not k:
        return None
    return s[k // 2] if k % 2 else 0.5 * (s[k // 2 - 1] + s[k // 2])


def _acute(u: tuple[float, float], v: tuple[float, float]) -> float:
    """Angle between two directions ignoring sense, in degrees: 0 to 90."""
    d = abs(u[0] * v[0] + u[1] * v[1])
    return degrees(acos(max(-1.0, min(1.0, d))))


def _nearest_on_polyline(
    point: list[float],
    polyline: list[list[float]],
) -> tuple[float, tuple[float, float] | None]:
    """``(distance, unit tangent)`` of the closest point of a polyline, or ``(inf, None)``."""
    index, _t, _q, distance = closest_on_polyline(point, polyline)
    if distance == float("inf"):
        return distance, None
    a, b = polyline[index], polyline[index + 1]
    abx, aby = b[0] - a[0], b[1] - a[1]
    length = (abx * abx + aby * aby) ** 0.5
    return distance, (abx / length, aby / length)


def _families(points: list[list[float]]) -> list[tuple[float, float]] | None:
    """Mean direction of a quad's two opposite edge families, or ``None``."""
    if len(points) != 4:
        return None

    out = []
    for start in (0, 1):
        vectors = []
        for k in (start, start + 2):
            a, b = points[k], points[(k + 1) % 4]
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = (dx * dx + dy * dy) ** 0.5
            if length <= 1e-12:
                continue
            vectors.append((dx / length, dy / length))
        if not vectors:
            return None
        ux, uy = vectors[0]
        for vx, vy in vectors[1:]:
            if ux * vx + uy * vy < 0.0:  # opposite sense round the face
                vx, vy = -vx, -vy
            ux, uy = ux + vx, uy + vy
        length = (ux * ux + uy * uy) ** 0.5
        if length <= 1e-12:
            return None
        out.append((ux / length, uy / length))
    return out


def guide_metrics(mesh: Mesh, guides: list[list[Any]], steep: float = STEEP) -> dict[str, Any]:
    """**How well a mesh carries its guides**, as numbers rather than a verdict.

    Parameters
    ----------
    mesh : Mesh
        A quad or pseudo-quad mesh.
    guides : sequence[sequence[point]]
        The guide curves, as polylines -- the same objects fed to the field.
    steep : float, optional
        Degrees off horizontal past which a guide station counts as tilted.

    Returns
    -------
    dict
        ``faces`` -- faces the guides pass through, the population everything else is
        measured over. **Zero means the metric could not see the guide at all**, and
        every other value is None.

        ``along`` -- median angle between the guide and the face family nearer to it,
        in degrees. 0 is a course running along the guide.

        ``across`` / ``across_within`` / ``across_tilted`` -- median angle of the other
        family, the share of it within ``ACROSS_TOLERANCE`` of 90 degrees, and the
        median over tilted stations only. 90 is an interface meeting the guide square.

        ``off_centre`` / ``off_centre_worst`` -- the guide's offset from the face centre
        in units of half the face width across it. 0 is mid-block, 1 is on the joint.

        ``poles`` -- pseudo-quad faces skipped, having no two families to compare.
    """
    empty = {"faces": 0, "along": None, "across": None, "across_within": None, "across_tilted": None, "off_centre": None, "off_centre_worst": None, "poles": 0}
    if not guides:
        return empty

    along, across, ratios, poles = [], [], [], 0

    for guide in guides:
        points = [p for p in guide]
        if len(points) < 2:
            continue

        for face in mesh.faces():
            corners = [mesh.vertex_coordinates(v) for v in mesh.face_vertices(face)]
            if len(corners) < 3:
                continue
            centre = [sum(p[i] for p in corners) / len(corners) for i in range(3)]

            distance, tangent = _nearest_on_polyline(centre, points)
            if tangent is None:
                continue
            # only the faces the guide actually passes through
            if distance >= max(((centre[0] - p[0]) ** 2 + (centre[1] - p[1]) ** 2) ** 0.5 for p in corners):
                continue

            # off-centre: the guide's offset in units of half the face width across it
            normal = (-tangent[1], tangent[0])
            projections = [(p[0] - centre[0]) * normal[0] + (p[1] - centre[1]) * normal[1] for p in corners]
            half = (max(projections) - min(projections)) / 2.0
            if half > 1e-9:
                ratios.append(distance / half)

            families = _families(corners)
            if families is None:
                poles += 1
                continue

            angles = sorted(_acute(f, tangent) for f in families)
            along.append(angles[0])
            tilt = _acute(tangent, (1.0, 0.0))
            across.append((tilt, angles[1]))

    if not along and not ratios:
        return dict(empty, poles=poles)

    angles = [a for _, a in across]
    tilted = [a for t, a in across if t >= steep]
    return {
        "faces": len(along),
        "along": _median(along),
        "across": _median(angles),
        "across_within": (100.0 * sum(1 for a in angles if a >= 90.0 - ACROSS_TOLERANCE) / len(angles)) if angles else None,
        "across_tilted": _median(tilted),
        "off_centre": _median(ratios),
        "off_centre_worst": max(ratios) if ratios else None,
        "poles": poles,
    }


def format_guide_metrics(metrics: dict[str, Any]) -> str:
    """One line of the numbers from ``guide_metrics``, for a harness to print."""
    if not metrics["faces"]:
        return "guide unseen (0 faces{})".format(", {} poles".format(metrics["poles"]) if metrics["poles"] else "")

    def num(value: float | None, fmt: str = "{:5.1f}") -> str:
        return "  -- " if value is None else fmt.format(value)

    out = "faces {:3d} | along {} deg | across {} deg, {}% within {:.0f}".format(
        metrics["faces"], num(metrics["along"]), num(metrics["across"]), num(metrics["across_within"], "{:3.0f}"), ACROSS_TOLERANCE
    )
    out += " | tilted {} deg".format(num(metrics["across_tilted"]))
    out += " | off-centre {} med, {} worst".format(num(metrics["off_centre"], "{:4.2f}"), num(metrics["off_centre_worst"], "{:4.2f}"))
    if metrics["poles"]:
        out += " | {} poles skipped".format(metrics["poles"])
    return out
