"""Turn a metric dict into a reading: the worst element as a handle, how concentrated, and a verdict.

Thresholds come from ``library/thresholds.json``; it reports, it never gates.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

from compas_singular.mcp.handle import vertex_handle

if TYPE_CHECKING:
    from compas.datastructures import Mesh


__all__ = ["DEFAULT_THRESHOLDS", "band", "describe", "format_triple"]


#: Fallback bands. ``min_angle`` and the rest are read as: at least ``good`` is
#: good, at least ``usable`` is usable, at least ``poor`` is poor, below that
#: unusable. ``max_angle`` and ``aspect_max`` run the other way.
DEFAULT_THRESHOLDS = {
    "min_angle": {"good": 45.0, "usable": 25.0, "poor": 15.0, "higher_is_better": True},
    "max_angle": {"good": 135.0, "usable": 155.0, "poor": 168.0, "higher_is_better": False},
    "aspect_max": {"good": 2.0, "usable": 4.0, "poor": 8.0, "higher_is_better": False},
    "share_below": {"good": 0.0, "usable": 0.02, "poor": 0.10, "higher_is_better": False},
}

BANDS = ("good", "usable", "poor", "unusable")


def band(value: Any, limits: dict[str, Any]) -> str:
    """Which band a value falls in. ``'unknown'`` when it cannot be judged."""
    if value is None:
        return "unknown"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if value != value:  # NaN
        return "unknown"
    higher_is_better = limits.get("higher_is_better", True)
    for name in ("good", "usable", "poor"):
        edge = limits.get(name)
        if edge is None:
            continue
        if higher_is_better and value >= edge:
            return name
        if not higher_is_better and value <= edge:
            return name
    return "unusable"


def _worst(bands: Iterable[str]) -> str:
    """The least good of the bands seen."""
    for name in reversed(BANDS):
        if name in bands:
            return name
    return "unknown"


def format_triple(triple: tuple[Any, Any, Any] | None) -> str:
    """``(min, max, aspect)`` as a short string, or ``'-'``."""
    if not triple:
        return "-"
    lo, hi, aspect = triple
    return "{:.1f} / {:.1f} deg, aspect {:.2f}".format(
        lo if lo is not None else float("nan"), hi if hi is not None else float("nan"), aspect if aspect is not None else float("nan")
    )


def _face_location(mesh: Mesh | None, fkey: Any) -> tuple[str | None, bool]:
    """A handle for where a face is, and whether it touches a boundary.

    The face's own key is useless to a caller -- it renumbers, and it is not a
    place. Its centroid, named the way a vertex is named, is both.
    """
    if fkey is None or mesh is None:
        return None, False
    try:
        vertices = mesh.face_vertices(fkey)
    except Exception:
        return None, False
    if not vertices:
        return None, False
    points = [mesh.vertex_coordinates(key) for key in vertices]
    centroid = [sum(p[i] for p in points) / float(len(points)) for i in range(3)]
    on_boundary = any(mesh.is_vertex_on_boundary(key) for key in vertices)
    return vertex_handle(centroid), on_boundary


def describe(mesh: Mesh | None, metrics: dict[str, Any], limits: dict[str, Any] | None = None) -> dict[str, Any]:
    """A prose reading of a quality dict.

    Parameters
    ----------
    mesh : Mesh or None
        Used only to locate the worst face. The reading degrades gracefully
        without it.
    metrics : dict
        From ``framefield.quality.mesh_quality``.
    limits : dict, optional
        Thresholds. Defaults to ``DEFAULT_THRESHOLDS``.

    Returns
    -------
    dict
        ``reading`` (a paragraph), ``verdict`` (one of ``BANDS``),
        ``bands`` (per metric), and ``worst_face_at`` (a handle, or None).
    """
    if not metrics:
        return {"reading": "Nothing is loaded.", "verdict": "unknown", "bands": {}, "worst_face_at": None}

    limits = limits or DEFAULT_THRESHOLDS
    bands = dict((name, band(metrics.get(name), limits[name])) for name in limits if name in limits)

    where, on_boundary = _face_location(mesh, metrics.get("worst_face"))
    sentences = []

    min_angle = metrics.get("min_angle")
    if min_angle is not None:
        place = " at {}".format(where) if where else ""
        edge = " on the boundary" if on_boundary else " in the interior"
        sentences.append("Worst element is a {:.1f} degree corner{}{}.".format(min_angle, edge, place))

    max_angle = metrics.get("max_angle")
    if max_angle is not None:
        sentences.append("Widest corner is {:.1f} degrees.".format(max_angle))

    aspect = metrics.get("aspect_max")
    if aspect is not None:
        if aspect == float("inf"):
            sentences.append("Aspect ratio is infinite: a face has a zero-length edge, which is degenerate rather than merely poor.")
        else:
            sentences.append("Worst aspect ratio is {:.2f} (a perfect grid reads 1.00).".format(aspect))

    # How WIDESPREAD the low angles are is the part that decides whether to
    # smooth locally or globally, so it is stated in those terms.
    share = metrics.get("share_below")
    low = metrics.get("low_angle")
    if share is not None and low is not None:
        if share > 0.0:
            sentences.append(
                "{:.1f}% of all corners are below {:.0f} degrees, so the problem is {} rather than a single face.".format(
                    100.0 * share, low, "widespread" if share > 0.05 else "in more than one place"
                )
            )
        elif bands.get("min_angle") in ("poor", "unusable", "usable"):
            # Nothing is below the low-angle line, yet the minimum is still not
            # good: the figure is one face, not a general condition.
            sentences.append("No corner is below {:.0f} degrees, so that minimum is one local defect rather than a general one.".format(low))
        else:
            sentences.append("No corner is below {:.0f} degrees.".format(low))

    faces = metrics.get("faces")
    irregular = metrics.get("irregular_interior")
    poles = metrics.get("poles")
    if faces is not None:
        tail = "{} faces".format(faces)
        if irregular:
            tail += ", {} irregular interior vertices".format(irregular)
        if poles:
            tail += ", {} poles".format(poles)
        # NOT ``capitalize()``: it lowercases everything after the first
        # character, which would turn "12 faces, 4 Poles" into mush.
        sentences.append("Mesh has " + tail + ".")

    verdict = _worst([b for b in bands.values() if b != "unknown"])
    return {"reading": " ".join(sentences), "verdict": verdict, "bands": bands, "worst_face_at": where}
