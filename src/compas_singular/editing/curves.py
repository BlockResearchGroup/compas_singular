"""Give a moved coarse edge back the curve it was traced with, by warping its separatrix. No field needed.

The match is anchored on the end that did not move.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING

from compas.geometry import distance_point_point
from compas.itertools import pairwise
from compas_singular.editing.rebuild import warp_polyline

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarseQuadMesh


__all__ = ["warp_edge_curve", "warp_chorded_edges"]


def warp_edge_curve(
    polylines: list[list[list[float]]],
    pa: list[float],
    pb: list[float],
    scale: float,
    claimed: set[int] | None = None,
) -> list[list[float]] | None:
    """The traced separatrix this edge came from, warped onto its current endpoints, or ``None``.

    Parameters
    ----------
    polylines : list[list[[x, y, z]]]
        The traced separatrix network, plus any curve the user drew.
    pa, pb : [x, y, z]
        Where the edge's two ends are NOW.
    scale : float
        The length the two tolerances are expressed in -- the background spacing
        the separatrices were traced at, where that is known. ``0.1 * scale`` is
        how close an end has to be to count as not having moved; ``scale`` is how
        far both ends may be when neither is anchored.
    claimed : set, optional
        Indices already taken by another edge. A polyline is only reused when
        nothing unclaimed fits -- the case where one patch was split in two and
        both halves genuinely lie along one separatrix. Mutated in place.

    Returns
    -------
    list[[x, y, z]] or None
        ``None`` when nothing fits, so the caller keeps the straight chord.
    """
    if not polylines:
        return None
    if claimed is None:
        claimed = set()

    anchor = scale * 0.1
    loose = scale
    chord = distance_point_point(pa, pb)

    best = None
    for i, polyline in enumerate(polylines):
        points = [list(p)[:3] for p in polyline]
        if len(points) < 2:
            continue
        for candidate in (points, list(reversed(points))):
            da = distance_point_point(candidate[0], pa)
            db = distance_point_point(candidate[-1], pb)

            if da <= anchor or db <= anchor:
                # anchored: one end never moved, so trust the other freely
                tier, free = 0, (db if da <= anchor else da)
            elif da <= loose and db <= loose:
                # neither end is anchored -- both corners of this edge moved. The
                # correspondence really is a guess here, so it is a timid one.
                tier, free = 1, da + db
            else:
                continue

            length = sum(distance_point_point(a, b) for a, b in pairwise(candidate))
            if chord > 1e-9 and not (0.25 * chord <= length <= 4.0 * chord):
                continue

            score = (i in claimed, tier, free)
            if best is None or score < best[0]:
                best = (score, i, candidate)

    if best is None:
        return None
    _score, index, points = best
    curve = warp_polyline(points, pa, pb)
    if curve is None:
        return None
    claimed.add(index)
    return curve


def warp_chorded_edges(
    mapping: dict[tuple[int, int], list[list[float]]],
    coarse: "CoarseQuadMesh",
    polylines: list[list[list[float]]],
    scale: float,
) -> tuple[dict[tuple[int, int], list[list[float]]], int]:
    """Give every two-point (chord) entry of ``mapping`` a warped curve where one fits. ``(mapping, warped)``."""
    claimed = set()
    warped = 0
    for edge, curve in list(mapping.items()):
        if len(curve) > 2:
            continue  # already a wall arc or an exact match
        u, v = edge
        pa = coarse.vertex_coordinates(u)
        pb = coarse.vertex_coordinates(v)
        out = warp_edge_curve(polylines, pa, pb, scale, claimed)
        if out is not None:
            mapping[edge] = out
            warped += 1
    return mapping, warped
