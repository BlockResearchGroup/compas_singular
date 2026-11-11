"""**Independent symmetry measurements** of point sets and meshes.

Independent of how anything was built: these search positions, they never read
the orbit labels an expansion records. That is what makes them a check.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import hypot
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas_singular.symmetry._geometry import SegmentHash

if TYPE_CHECKING:
    from compas.datastructures import Mesh
    from compas_singular.symmetry.group import SymmetryGroup


__all__ = ["point_invariance", "mesh_invariance"]


def point_invariance(points: Sequence[Sequence[float]], group: SymmetryGroup, tol: float | None = None) -> dict[str, Any]:
    """How symmetric a point set is under every element of ``group``.

    Returns
    -------
    dict
        ``share``: the smallest, over the elements, share of points whose image
        is within ``tol`` of a point. ``worst``: the largest distance from an
        image to its nearest point. ``by_element``: ``key -> (share, worst)``.
    """
    pts = [[p[0], p[1], 0.0] for p in points]
    if not pts:
        return {"share": 1.0, "worst": 0.0, "by_element": {}}
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    diagonal = hypot(max(xs) - min(xs), max(ys) - min(ys)) or 1.0
    tol = 1e-6 * diagonal if tol is None else tol
    limit = 0.05 * diagonal
    index = SegmentHash(max(limit / 4.0, 1e-9))
    for i, p in enumerate(pts):
        index.add_point(p, i)
    by_element = {}
    for e in group.elements:
        hit, worst = 0, 0.0
        for p in pts:
            q = group.apply(e, p)
            d, _ = index.nearest(q[0], q[1], limit)
            if d <= tol:
                hit += 1
            worst = max(worst, d)
        by_element[e.key] = (hit / float(len(pts)), worst)
    return {"share": min(s for s, _ in by_element.values()), "worst": max(w for _, w in by_element.values()), "by_element": by_element}


def mesh_invariance(mesh: Mesh, group: SymmetryGroup, tol: float | None = None) -> dict[str, Any]:
    """``point_invariance`` of a mesh's vertices."""
    return point_invariance([mesh.vertex_coordinates(v) for v in mesh.vertices()], group, tol)
