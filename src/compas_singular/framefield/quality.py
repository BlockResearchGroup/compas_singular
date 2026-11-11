"""Element quality: an absolute hard floor here, the per-domain regression check in ``15_baseline.py``.

Pseudo-quads are measured on the three corners they actually have.
"""

from __future__ import annotations

from math import acos
from math import atan2
from math import degrees
from math import pi
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh


__all__ = ["mesh_quality", "hard_floor", "face_angles", "curve_alignment", "curve_alignment_profile", "HARD_MIN_ANGLE", "HARD_MAX_ANGLE", "LOW_ANGLE"]


#: Below this, an angle is a degeneracy rather than a poor element -- an order
#: of magnitude under the worst legitimate angle in the suite.
HARD_MIN_ANGLE = 0.5

#: Likewise at the top.
HARD_MAX_ANGLE = 179.5

#: Default "low angle" for the share-of-angles metric. Reported, never gated on.
LOW_ANGLE = 20.0


def face_angles(points: list[list[float]]) -> list[float]:
    """Interior angles of a polygon in degrees; a coincident corner gives 0.0 rather than raising."""
    n = len(points)
    out = []
    for i in range(n):
        a, b, c = points[i - 1], points[i], points[(i + 1) % n]
        ux, uy = a[0] - b[0], a[1] - b[1]
        vx, vy = c[0] - b[0], c[1] - b[1]
        lu = (ux * ux + uy * uy) ** 0.5
        lv = (vx * vx + vy * vy) ** 0.5
        if lu < 1e-12 or lv < 1e-12:
            out.append(0.0)
            continue
        out.append(degrees(acos(max(-1.0, min(1.0, (ux * vx + uy * vy) / (lu * lv))))))
    return out


def _face_edges(points: list[list[float]]) -> list[float]:
    """Edge lengths of a polygon, in the order its corners were given."""
    n = len(points)
    return [((points[i][0] - points[(i + 1) % n][0]) ** 2 + (points[i][1] - points[(i + 1) % n][1]) ** 2) ** 0.5 for i in range(n)]


def mesh_quality(mesh: Mesh, low_angle: float = LOW_ANGLE) -> dict[str, Any]:
    """Element quality of a quad (or pseudo-quad) mesh, as plain numbers.

    Parameters
    ----------
    mesh : Mesh
    low_angle : float, optional
        Threshold for ``share_below``, in degrees.

    Returns
    -------
    dict
        ``faces``, ``poles``, ``min_angle``, ``max_angle``, ``aspect_max``,
        ``share_below``, ``low_angle``, ``irregular_interior``, and
        ``worst_face`` -- the key of the face carrying ``min_angle``, so a
        failure can be looked at rather than merely counted.

    Notes
    -----
    Aspect ratio is **longest edge over shortest edge**, per face, worst over
    the mesh. A perfect grid reads 1.00. It is ``inf`` when a face has a
    zero-length edge, which is what ``hard_floor`` rejects on.
    """
    face_pole = mesh.attributes.get("face_pole") or {}

    lo, hi = 180.0, 0.0
    aspect = 0.0
    total = 0
    below = 0
    worst_face = None

    for fkey in mesh.faces():
        # The corners the face HAS. Never (p, a, b, p) -- see the module
        # docstring; that is the whole pseudo-quad trap.
        points = [mesh.vertex_coordinates(v) for v in mesh.face_vertices(fkey)]
        if len(points) < 3:
            lo, worst_face = 0.0, fkey
            aspect = float("inf")
            continue

        angles = face_angles(points)
        total += len(angles)
        below += sum(1 for a in angles if a < low_angle)
        if min(angles) < lo:
            lo, worst_face = min(angles), fkey
        hi = max(hi, max(angles))

        edges = _face_edges(points)
        shortest = min(edges)
        aspect = float("inf") if shortest <= 0.0 else max(aspect, max(edges) / shortest)

    irregular = 0
    for vkey in mesh.vertices():
        if mesh.is_vertex_on_boundary(vkey):
            continue
        if len(mesh.vertex_neighbors(vkey)) != 4:
            irregular += 1

    return {
        "faces": mesh.number_of_faces(),
        "poles": len(face_pole),
        "min_angle": lo,
        "max_angle": hi,
        "aspect_max": aspect,
        "share_below": (below / float(total)) if total else 0.0,
        "low_angle": low_angle,
        "irregular_interior": irregular,
        "worst_face": worst_face,
    }


#: Radius, as a multiple of the sampling spacing, within which a dense mesh
#: edge counts as being "at" a point on the curve.
ALIGNMENT_RADIUS = 1.0


def _curve_samples(curve: list[list[float]], spacing: float) -> list[tuple[tuple[float, float], float]]:
    """``(point, tangent)`` every ``spacing`` units along a polyline."""
    out = []
    for a, b in zip(curve, curve[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-12:
            continue
        steps = max(1, int(round(length / spacing)))
        tangent = atan2(dy, dx)
        for k in range(steps):
            t = (k + 0.5) / steps
            out.append(((a[0] + dx * t, a[1] + dy * t), tangent))
    return out


def curve_alignment_profile(
    mesh: Mesh,
    curve: list[list[float]],
    radius: float = ALIGNMENT_RADIUS,
    spacing: float = 0.5,
) -> list[tuple[tuple[float, float], float]]:
    """Per sample along a curve, the mean angle to nearby mesh edges folded into the cross period (0 to 45).

    Returns
    -------
    list[((float, float), float)]
        Sample point and its mean offset in degrees, in order along the curve.
    """
    edges = []
    for u, v in mesh.edges():
        a = mesh.vertex_coordinates(u)
        b = mesh.vertex_coordinates(v)
        if abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9:
            continue  # a pole's collapsed edge
        edges.append((((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0), atan2(b[1] - a[1], b[0] - a[0])))

    out = []
    for mid, tangent in _curve_samples(curve, spacing):
        vals = []
        for centre, ang in edges:
            if (centre[0] - mid[0]) ** 2 + (centre[1] - mid[1]) ** 2 <= radius * radius:
                d = (tangent - ang) % (pi / 2)
                vals.append(degrees(min(d, pi / 2 - d)))
        if vals:
            out.append((mid, sum(vals) / len(vals)))
    return out


def curve_alignment(
    mesh: Mesh,
    curve: list[list[float]],
    radius: float = ALIGNMENT_RADIUS,
    spacing: float = 0.5,
) -> float:
    """Mean of ``curve_alignment_profile``, in degrees; check the profile too."""
    profile = curve_alignment_profile(mesh, curve, radius, spacing)
    if not profile:
        return float("nan")
    return sum(v for _, v in profile) / len(profile)


def hard_floor(metrics: dict[str, Any]) -> tuple[bool, str]:
    """TIER 1. Is anything in this mesh degenerate rather than merely poor?

    Absolute: an angle at or near 0 or 180 degrees, or a non-finite aspect
    ratio. Everything softer is the regression check's business.

    Returns
    -------
    (bool, str)
        Verdict, and when false the offending metric named with its value.
    """
    if not metrics.get("faces"):
        return False, "no faces"

    aspect = metrics["aspect_max"]
    if aspect != aspect or aspect in (float("inf"), float("-inf")):
        return False, ("aspect_max is {} -- a face has a zero-length edge (face {})".format(aspect, metrics.get("worst_face")))

    if metrics["min_angle"] < HARD_MIN_ANGLE:
        return False, ("min_angle {:.3f} deg is below the hard floor of {} deg (face {})".format(metrics["min_angle"], HARD_MIN_ANGLE, metrics.get("worst_face")))

    if metrics["max_angle"] > HARD_MAX_ANGLE:
        return False, ("max_angle {:.3f} deg is above the hard floor of {} deg".format(metrics["max_angle"], HARD_MAX_ANGLE))

    return True, ""
