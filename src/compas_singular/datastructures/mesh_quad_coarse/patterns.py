"""Patch patterns for coarse quad mesh densification: a template on the unit square plus a Coons morph.

``ortho`` reproduces ``discrete_coons_patch`` exactly. Design notes: ``design_notes/datastructures.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas.itertools import pairwise

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarseQuadMesh


__all__ = [
    "PATTERNS",
    "create_pattern",
    "pattern_morph",
    "pattern_morph_triangle",
    "patch_divisions",
    "reconcile_strip_densities",
]


# ==============================================================================
# create patterns
# ==============================================================================


PATTERNS = [
    "ortho",
    "diagonal",
    "fan",
]


def create_pattern(type: str, nu: int, nw: int) -> tuple[list[tuple[float, float]], list[list[int]]]:
    """Build the template of a pattern at a given number of divisions.

    Parameters
    ----------
    kind : str
        A key of ``PATTERNS``.
    nu, nw : int
        Divisions along u (the a->b side) and along w (the a->d side).

    Returns
    -------
    list[(float, float)]
        Template vertices as parameter pairs on the unit square.
    list[list[int]]
        Template faces, indexing into those.

    Raises
    ------
    ValueError
        If ``kind`` is not a known pattern, or the divisions do not suit it.
    """
    PATTERNS = {
        "ortho": _pattern_ortho,
        "diagonal": _pattern_diagonal,
        "fan": _pattern_fan,
        # internal: what 'fan' becomes on a triangular patch. Its parameters
        # are barycentric, so it pairs with pattern_morph_triangle, not
        # pattern_morph -- which is why it is not in the public PATTERNS.
        "fan_triangle": _pattern_fan_triangle,
    }

    if type not in PATTERNS:
        raise ValueError("unknown pattern {!r} -- pick one of {}".format(type, sorted(PATTERNS)))
    return PATTERNS[type](max(1, nu), max(1, nw))


# ==============================================================================
# patterns
# ==============================================================================


def _pattern_ortho(nu: int, nw: int) -> tuple[list[tuple[float, float]], list[list[int]]]:
    """The plain grid: ``nu`` by ``nw`` quads on the unit square, ordered as ``discrete_coons_patch``.

    Parameters
    ----------
    nu, nw : int
        Divisions along u and along w.

    Returns
    -------
    list[(float, float)], list[list[int]]
    """
    uw = [(i / nu, j / nw) for i in range(nu + 1) for j in range(nw + 1)]

    m = nw + 1
    faces = [[i * m + j, i * m + j + 1, (i + 1) * m + j + 1, (i + 1) * m + j] for i in range(nu) for j in range(nw)]
    return uw, faces


def _pattern_diagonal(nu: int, nw: int) -> tuple[list[tuple[float, float]], list[list[int]]]:
    """The grid with both patch diagonals drawn through it as chains of cut cells.

    Needs ``nu == nw`` and even, so both diagonals are unbroken lines.

    Parameters
    ----------
    nu, nw : int
        Divisions along u and along w. Must be equal and even.

    Returns
    -------
    list[(float, float)], list[list[int]]

    Raises
    ------
    ValueError
        If ``nu`` and ``nw`` are not equal, or not even.
    """
    if nu != nw or nu % 2:
        raise ValueError(
            "the diagonal pattern needs the same even number of divisions on all four sides of a patch, got {} by {} -- run reconcile_strip_densities() first".format(nu, nw)
        )

    uw, quads = _pattern_ortho(nu, nw)

    n = nu
    faces = []
    for idx, (i, j) in enumerate((i, j) for i in range(n) for j in range(n)):
        # v0 = (i, j), v1 = (i, j+1), v2 = (i+1, j+1), v3 = (i+1, j)
        v0, v1, v2, v3 = quads[idx]
        if i == j:
            # a -> c, cut v0-v2
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
        elif i + j == n - 1:
            # b -> d, cut v1-v3; never the same cell as a -> c, since n is even
            faces.append([v1, v2, v3])
            faces.append([v1, v3, v0])
        else:
            faces.append(quads[idx])
    return uw, faces


# the four quadrants of a fan, as (pole, first mid-edge, second mid-edge) in
# (u, w). Ordered so the angular sweep runs the same way around every pole,
# which keeps the face winding consistent with the grid templates.
_FAN_QUADRANTS = [
    ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5)),  # at a, between sides ab and ad
    ((1.0, 0.0), (1.0, 0.5), (0.5, 0.0)),  # at b, between sides bc and ab
    ((1.0, 1.0), (0.5, 1.0), (1.0, 0.5)),  # at c, between sides dc and bc
    ((0.0, 1.0), (0.0, 0.5), (0.5, 1.0)),  # at d, between sides ad and dc
]

_FAN_CENTRE = (0.5, 0.5)


def _pattern_fan(nu: int, nw: int) -> tuple[list[tuple[float, float]], list[list[int]]]:
    """Four polar fans, one per corner, meeting at the patch centre.

    Needs ``nu == nw`` and even, so neighbouring quadrants and patches weld.

    Parameters
    ----------
    nu, nw : int
        Divisions along u and along w. Must be equal and even.

    Returns
    -------
    list[(float, float)], list[list[int]]

    Raises
    ------
    ValueError
        If ``nu`` and ``nw`` are not equal, or not even.
    """
    if nu != nw or nu % 2:
        raise ValueError(
            "the fan pattern needs the same even number of divisions on all four sides of a patch, got {} by {} -- run reconcile_strip_densities() first".format(nu, nw)
        )

    rings = nu // 2  # radial steps, pole -> L
    rays = nu  # angular steps across the whole L
    half = rays // 2  # angular steps per leg of the L

    uw = []
    index = {}

    def add(p: tuple[float, float]) -> int:
        # one vertex per parameter pair, so the four quadrants share their
        # seams, their poles and the centre instead of duplicating them
        key = (round(p[0], 12), round(p[1], 12))
        if key not in index:
            index[key] = len(uw)
            uw.append((p[0], p[1]))
        return index[key]

    faces = []
    for pole, m1, m2 in _FAN_QUADRANTS:
        grid = {}
        for i in range(rays + 1):
            # angular: walk the far boundary m1 -> centre -> m2, half the rays
            # on each leg, so the sweep is at constant speed
            if i <= half:
                far = _lerp(m1, _FAN_CENTRE, i, half)
            else:
                far = _lerp(_FAN_CENTRE, m2, i - half, half)
            for j in range(rings + 1):
                # radial: slide from the pole out to the far boundary
                grid[i, j] = add(_lerp(pole, far, j, rings))

        for i in range(rays):
            for j in range(rings):
                faces.append([grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])

    return uw, faces


# the three corner fans of a triangular patch, as (corner, first mid-edge,
# second mid-edge) in barycentric coordinates on the corners (V0, V1, V2),
# counter-clockwise, V0 the pole. A cyclic shift of the corners maps the table
# onto itself -- that is what makes every corner, the pole included, the same.
_FAN_TRIANGLE_REGIONS = [
    ((1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.5, 0.0, 0.5)),  # at V0, between sides V0V1 and V2V0
    ((0.0, 1.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.5, 0.0)),  # at V1, between sides V1V2 and V0V1
    ((0.0, 0.0, 1.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)),  # at V2, between sides V2V0 and V1V2
]

_FAN_TRIANGLE_CENTRE = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def _pattern_fan_triangle(nu: int, nw: int) -> tuple[list[tuple[float, float, float]], list[list[int]]]:
    """Three polar fans, one per corner, meeting at the centre of a triangular patch.

    Vertices are barycentric on (pole, V1, V2); map them with ``pattern_morph_triangle``.

    Parameters
    ----------
    nu, nw : int
        Divisions along the sides. Must be equal and even, as for
        ``_pattern_fan``.

    Returns
    -------
    list[(float, float, float)], list[list[int]]
        Template vertices as barycentric coordinates on (pole, V1, V2), and
        template faces.

    Raises
    ------
    ValueError
        If ``nu`` and ``nw`` are not equal, or not even.
    """
    if nu != nw or nu % 2:
        raise ValueError(
            "the fan pattern needs the same even number of divisions on all three sides of a triangular patch, got {} by {} -- run reconcile_strip_densities() first".format(
                nu, nw
            )
        )

    rings = nu // 2  # radial steps, corner -> L
    rays = nu  # angular steps across the whole L
    half = rays // 2  # angular steps per leg of the L

    lam = []
    index = {}

    def add(key: Any, p: tuple[float, float, float]) -> int:
        # one vertex per key, so the three corners share their seams and the
        # centre -- and V1 and V2 their innermost ring -- instead of
        # duplicating them
        if key not in index:
            index[key] = len(lam)
            lam.append(tuple(p))
        return index[key]

    faces = []
    for n, (corner, m1, m2) in enumerate(_FAN_TRIANGLE_REGIONS):
        grid = {}
        for i in range(rays + 1):
            if i <= half:
                far = _lerp(m1, _FAN_TRIANGLE_CENTRE, i, half)
            else:
                far = _lerp(_FAN_TRIANGLE_CENTRE, m2, i - half, half)
            for j in range(rings + 1):
                p = _lerp(corner, far, j, rings)
                if n == 0 and j == 0:
                    key = ("pole", i)
                else:
                    key = tuple(round(x, 12) for x in p)
                grid[i, j] = add(key, p)

        for i in range(rays):
            for j in range(rings):
                faces.append([grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])

    return lam, faces


def _lerp(p: Sequence[float], q: Sequence[float], k: int, n: int) -> list[float]:
    """Point ``k/n`` of the way from ``p`` to ``q``, symmetric so forward and backward walks agree bit for bit.

    Parameters
    ----------
    p, q : sequence[float]
        The two ends.
    k, n : int
        Step index and step count.

    Returns
    -------
    list[float]
    """
    if k == 0:
        return [float(a) for a in p]
    if k == n:
        return [float(b) for b in q]
    return [(a * (n - k) + b * k) / n for a, b in zip(p, q)]


# ==============================================================================
# morph
# ==============================================================================


def pattern_morph(uw: list[tuple[float, float]], faces: list[list[int]], sides: list[list[list[float]] | None]) -> tuple[list[list[float]], list[list[int]]]:
    """Map a template onto a coarse face through its four side polylines.

    Parameters
    ----------
    uw : list[(float, float)]
        Template vertices, as parameters on the unit square.
    faces : list[list[int]]
        Template faces.
    sides : list
        The four side polylines ``[ab, bc, dc, ad]``, each a list of points, in
        the orientation ``discrete_coons_patch`` documents. A side that is
        ``None`` is a collapsed one -- the patch has a pole at that corner.

    Returns
    -------
    list[[float, float, float]]
        The morphed vertices.
    list[list[int]]
        The faces, with vertices repeated inside a face dropped -- that is what
        turns the fan's innermost ring of quads into triangles.
    """
    ab, bc, dc, ad = sides

    # a missing side is the corner it collapsed to, repeated (as in compas)
    if not ab:
        ab = [ad[0]] * len(dc)
    if not bc:
        bc = [ab[-1]] * len(ad)
    if not dc:
        dc = [bc[-1]] * len(ab)
    if not ad:
        ad = [dc[0]] * len(bc)

    a, b, c, d = ab[0], bc[0], dc[-1], ad[-1]

    vertices = []
    for u, w in uw:
        p_ab = _polyline_point_at(ab, u)
        p_dc = _polyline_point_at(dc, u)
        p_ad = _polyline_point_at(ad, w)
        p_bc = _polyline_point_at(bc, w)

        point = []
        for k in range(3):
            # blend the two curves running along u, then the two along w ...
            along_u = p_ab[k] * (1 - w) + p_dc[k] * w
            along_w = p_ad[k] * (1 - u) + p_bc[k] * u
            # ... and take out the bilinear corner patch, which both blends
            # already contain.
            #
            # Summed with the builtin, not with a running total: compas adds
            # the four corner terms through ``sum_vectors``, and since 3.12
            # CPython's ``sum`` compensates the rounding of a float series. A
            # hand-rolled accumulation lands a unit in the last place away
            # from it, which is enough to shift a welded vertex.
            corner = sum(
                (
                    a[k] * ((1 - u) * (1 - w)),
                    b[k] * (u * (1 - w)),
                    c[k] * (u * w),
                    d[k] * ((1 - u) * w),
                )
            )
            point.append(along_u + along_w - corner)
        vertices.append(point)

    faces = [[p for p, q in pairwise(face + face[:1]) if p != q] for face in faces]
    return vertices, faces


def pattern_morph_triangle(lam: list[tuple[float, float, float]], faces: list[list[int]], sides: list[list[list[float]] | None]) -> tuple[list[list[float]], list[list[int]]]:
    """Map a barycentric template onto a triangular patch through its three sides (Nielson side-vertex).

    Symmetric in the three corners, so which one is the pole does not change the geometry.

    Parameters
    ----------
    lam : list[(float, float, float)]
        Template vertices, as barycentric coordinates on (pole, V1, V2).
    faces : list[list[int]]
        Template faces.
    sides : list
        The four side polylines ``[ab, bc, dc, ad]`` of a pseudo-quad, as for
        ``pattern_morph``, exactly one of them ``None``.

    Returns
    -------
    list[[float, float, float]]
        The morphed vertices.
    list[list[int]]
        The faces, with vertices repeated inside a face dropped.
    """
    ab, bc, dc, ad = sides

    # back to one loop a -> b -> c -> d -> a; the collapsed side names the pole,
    # and the three sides after it run V0 -> V1 -> V2 -> V0
    loop = [ab, bc, dc[::-1] if dc else None, ad[::-1] if ad else None]
    k = [side is None for side in loop].index(True)
    t0, t1, t2 = loop[(k + 1) % 4], loop[(k + 2) % 4], loop[(k + 3) % 4]
    corners = [t0[0], t1[0], t2[0]]
    opposite = [t1, t2, t0]  # E_i, running V(i+1) -> V(i+2)

    vertices = []
    for bary in lam:
        # on a corner or a side, return the input point untouched rather than
        # the blend, which would divide by zero on a corner and could come back
        # a unit in the last place away from the side polyline on a side --
        # and side points are what the neighbouring patch welds to
        zero = [i for i in range(3) if bary[i] == 0.0]
        if len(zero) == 2:
            vertices.append(list(corners[3 - sum(zero)]))
            continue
        if zero:
            i = zero[0]
            j, h = (i + 1) % 3, (i + 2) % 3
            vertices.append(_polyline_point_at(opposite[i], bary[h] / (bary[j] + bary[h])))
            continue

        spokes, weights = [], []
        for i in range(3):
            j, h = (i + 1) % 3, (i + 2) % 3
            e = _polyline_point_at(opposite[i], bary[h] / (bary[j] + bary[h]))
            spokes.append([bary[i] * v + (1.0 - bary[i]) * x for v, x in zip(corners[i], e)])
            weights.append(bary[j] * bary[h])
        total = sum(weights)
        vertices.append([sum(weights[i] * spokes[i][c] for i in range(3)) / total for c in range(3)])

    faces = [[p for p, q in pairwise(face + face[:1]) if p != q] for face in faces]
    return vertices, faces


def _polyline_point_at(points: list[list[float]], t: float) -> list[float]:
    """Evaluate a polyline at ``t`` in [0, 1], parametrised by point index as ``discrete_coons_patch`` is.

    Parameters
    ----------
    points : list[[float, float, float]]
        The polyline points.
    t : float
        Parameter in [0, 1].

    Returns
    -------
    [float, float, float]
    """
    n = len(points) - 1
    if n <= 0:
        return list(points[0])

    s = t * n
    node = int(round(s))
    if abs(s - node) < 1e-9:
        return list(points[min(max(node, 0), n)])

    i = int(s)
    i = min(max(i, 0), n - 1)
    f = s - i
    p, q = points[i], points[i + 1]
    return [a + f * (b - a) for a, b in zip(p, q)]


# ==============================================================================
# divisions
# ==============================================================================


def patch_divisions(sides: list[list[list[float]] | None]) -> tuple[int, int]:
    """The number of divisions a set of side polylines asks for.

    Parameters
    ----------
    sides : list
        ``[ab, bc, dc, ad]``, any one of which may be ``None``.

    Returns
    -------
    (int, int)
        Divisions along u and along w.
    """
    ab, bc, dc, ad = sides
    return len(ab if ab else dc) - 1, len(bc if bc else ad) - 1


# the patterns whose template needs a patch to be d by d with d even -- see
# ``_pattern_diagonal`` and ``_pattern_fan`` for why
_SQUARE_EVEN = ("diagonal", "fan")


def reconcile_strip_densities(coarse: "CoarseQuadMesh", patterns: "str | dict[int, str]") -> dict[int, tuple[int, int]]:
    """Force strip densities to values the patterns can tile. Returns ``{skey: (old, new)}``.

    Strips tied by ``diagonal``/``fan`` patches share the group's largest density, rounded up to even.

    Parameters
    ----------
    coarse : CoarseQuadMesh
        The layout. Its strip densities are modified in place.
    patterns : str or dict
        A key of ``PATTERNS`` for every face, or one per face as
        ``{fkey: pattern}``.

    Returns
    -------
    dict
        ``{skey: (old, new)}`` for every strip whose density was changed.
    """
    if not isinstance(patterns, dict):
        patterns = {fkey: patterns for fkey in coarse.faces()}

    edge_strip = {}
    for skey, edges in coarse.strips(data=True):
        for u, v in edges:
            edge_strip[u, v] = skey
            edge_strip[v, u] = skey

    # union-find over the strips; a square patch merges the two crossing it
    parent = {skey: skey for skey in coarse.strips()}

    def find(skey: int) -> int:
        while parent[skey] != skey:
            parent[skey] = parent[parent[skey]]
            skey = parent[skey]
        return skey

    tied = []
    for fkey in coarse.faces():
        if patterns.get(fkey) not in _SQUARE_EVEN:
            continue
        # the strips ``_patch_sides`` reads nu and nw from: two, or one if a
        # strip crosses itself here. Read off the halfedges rather than
        # ``face_strips`` so a pseudo-quad needs no special case.
        roots = {find(edge_strip[u, v]) for u, v in coarse.face_halfedges(fkey)}
        root = roots.pop()
        for other in roots:
            parent[other] = root
        tied.append(root)

    groups = {}
    for skey in parent:
        groups.setdefault(find(skey), []).append(skey)

    changed = {}
    for root in {find(skey) for skey in tied}:
        d = max(coarse.get_strip_density(skey) for skey in groups[root])
        d = max(2, d + d % 2)
        for skey in groups[root]:
            old = coarse.get_strip_density(skey)
            if old != d:
                coarse.set_strip_density(skey, d)
                changed[skey] = (old, d)

    if changed:
        print(
            "reconcile_strip_densities: {} strip(s) raised to fit the diagonal/fan pattern(s): {}".format(
                len(changed), ", ".join("{} {}->{}".format(skey, old, new) for skey, (old, new) in sorted(changed.items()))
            )
        )

    return changed
