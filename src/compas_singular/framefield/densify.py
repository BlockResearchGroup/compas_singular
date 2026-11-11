"""Field-aware densification: patch interiors relaxed towards the field, patch boundaries fixed.

Each round is one sparse least-squares solve; quality is spent only where a guide asked for it.
"""

from __future__ import annotations

from math import atan2
from math import cos
from math import sin
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from compas.geometry import discrete_coons_patch
from compas.itertools import pairwise
from compas.tolerance import TOL
from compas_singular.datastructures import PseudoQuadMesh
from compas_singular.datastructures import meshes_join_and_weld
from compas_singular.framefield.constraints import PERIOD

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarsePseudoQuadMesh
    from compas_singular.framefield.field import CrossField


__all__ = ["field_densification", "FieldSampler", "relax_patch", "PATCH_MIN_ANGLE", "PATCH_MAX_ANGLE", "PATCH_MAX_ASPECT"]


#: How far a patch may be spent to satisfy a GUIDE curve. Floors, not targets:
#: a Coons patch already worse than these keeps its own values as the limit.
PATCH_MIN_ANGLE = 30.0
PATCH_MAX_ANGLE = 150.0
PATCH_MAX_ASPECT = 5.0

#: Solve / re-arm rounds per stiffness. The angles stop changing well before the
#: positions converge, so more rounds cost time without changing the mesh.
ITERATIONS = 12

#: Stiffnesses tried, weakest first. The first that settles and is accepted
#: wins, so a patch buys as much alignment as its own boundary can carry.
#: ``16_field_densify.py --tradeoff`` prints what each one costs.
STIFFNESS = (0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 64.0)

#: Pull back towards the Coons position, relative to the mean alignment weight.
TIKHONOV = 1e-3

#: Stop a stiffness early once no node moves further than this fraction of the
#: patch's mean edge length.
TOLERANCE = 1e-4

#: A stiffness only counts as settled once its per-round movement has shrunk to
#: this fraction of the first round's -- an oscillation never contracts.
CONTRACTION = 0.25


# ------------------------------------------------------------------
# sampling the field at arbitrary points
# ------------------------------------------------------------------


class FieldSampler(object):
    """The field's angle at points that are not background vertices.

    Parameters
    ----------
    field : CrossField
    locator : PointLocator, optional
        Defaults to ``field.locator()``. A ``Tracer`` is accepted too.
    """

    def __init__(self, field: CrossField, locator: Any = None) -> None:
        self.field = field
        locator = locator if locator is not None else field.locator()
        self.locator = getattr(locator, "locator", locator)
        u = field.u
        faces = self.locator.fkeys
        corners = [[u[v] for v in self.locator.face_vertices[f]] for f in faces]
        self._re = np.array([[z.real for z in row] for row in corners], dtype=float)
        self._im = np.array([[z.imag for z in row] for row in corners], dtype=float)

    def theta(self, point: Any, hint: Any = None) -> tuple[float | None, int | None]:
        """``(theta, fkey)`` at one point, or ``(None, None)`` if it is unreachable.

        A point just outside the domain -- a Coons node bulging past a concave
        wall -- takes the nearest triangle's centre rather than dropping out.
        """
        found = self.locator.locate(point, hint)
        if found is None:
            found = self.locator.nearest(point)
        if found is None:
            return None, None
        fkey, bary = found
        return self.field.angle_in_face(fkey, bary), fkey

    def thetas(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``theta`` along a sequence of points, each hinted by the one before.

        Exactly the values a loop over ``theta`` gives, computed in bulk.

        Returns
        -------
        (ndarray, ndarray)
            The angles, folded into the cross's period, and whether each point
            was reachable at all.
        """
        count = len(points)
        locator = self.locator
        status, faces, bary = locator.locate_arrays(points)
        ok = status == 1

        # Only a point no single face owns depends on the hint -- the face of
        # the point before it -- so only those are walked in order.
        for k in np.flatnonzero(~ok).tolist():
            hit = None
            if status[k] == 2:
                hint = locator.fkeys[faces[k - 1]] if k and ok[k - 1] else None
                hit = locator.locate(points[k], hint)
            if hit is None:
                hit = locator.nearest(points[k])
            if hit is not None:
                faces[k] = locator.index[hit[0]]
                bary[k] = hit[1]
                ok[k] = True

        # ``angle_in_face`` in bulk. Python's ``float * complex`` is written out
        # as real arithmetic so every value, and every signed zero, matches.
        faces = np.where(ok, faces, 0)
        re = np.zeros(count)
        im = np.zeros(count)
        for c in range(3):
            w = bary[:, c]
            ur, ui = self._re[faces, c], self._im[faces, c]
            re = re + (w * ur - 0.0 * ui)
            im = im + (w * ui + 0.0 * ur)

        out = np.zeros(count)
        for k, y, x in zip(np.flatnonzero(ok).tolist(), im[ok].tolist(), re[ok].tolist()):
            if x == 0 and y == 0:
                out[k] = self.field.angle_in_face(locator.fkeys[faces[k]], tuple(bary[k]))
            else:
                out[k] = atan2(y, x) / 4.0
        return out, ok


# ------------------------------------------------------------------
# the patch solver
# ------------------------------------------------------------------


def _unwrap(grid: np.ndarray, sampler: FieldSampler) -> np.ndarray | None:
    """A continuous ``i``-family angle over the patch, or ``None``."""
    n, m, _ = grid.shape
    raw, ok = sampler.thetas(grid.reshape(n * m, 3))
    if not ok.any():
        return None
    raw = raw.reshape(n, m)
    ok = ok.reshape(n, m)

    def branch(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
        return value + PERIOD * np.round((reference - value) / PERIOD)

    theta = np.zeros((n, m))
    seed = atan2(grid[1, 0][1] - grid[0, 0][1], grid[1, 0][0] - grid[0, 0][0])
    theta[0, 0] = branch(raw[0, 0], seed) if ok[0, 0] else seed
    for i in range(1, n):
        theta[i, 0] = branch(raw[i, 0], theta[i - 1, 0]) if ok[i, 0] else theta[i - 1, 0]
    for j in range(1, m):
        theta[:, j] = np.where(ok[:, j], branch(raw[:, j], theta[:, j - 1]), theta[:, j - 1])
    return theta


def _root(values: np.ndarray) -> np.ndarray:
    """Elementwise ``x ** 0.5`` with Python's rounding, which is not ``np.sqrt``'s."""
    return np.array([x**0.5 for x in values.tolist()], dtype=float)


class _PatchSystem(object):
    """The least-squares system of one patch (alignment, smoothing and Tikhonov rows), structure built once."""

    def __init__(self, coons: np.ndarray) -> None:
        n, m, _ = coons.shape
        self.coons = coons
        self.shape = (n, m)
        fi, fj = np.meshgrid(np.arange(1, n - 1), np.arange(1, m - 1), indexing="ij")
        self.fi, self.fj = fi.ravel(), fj.ravel()
        self.nfree = len(self.fi)
        column = -np.ones((n, m), dtype=int)
        column[self.fi, self.fj] = 2 * np.arange(self.nfree)
        self.column = column

        # every grid edge in the order i-steps then j-steps, kept only if it
        # touches a free node
        ia, ja, ib, jb, istep = [], [], [], [], []
        for i in range(n - 1):
            for j in range(m):
                ia.append(i), ja.append(j), ib.append(i + 1), jb.append(j), istep.append(True)
        for i in range(n):
            for j in range(m - 1):
                ia.append(i), ja.append(j), ib.append(i), jb.append(j + 1), istep.append(False)
        ia, ja, ib, jb = (np.array(x, dtype=int) for x in (ia, ja, ib, jb))
        keep = (column[ia, ja] >= 0) | (column[ib, jb] >= 0)
        self.ia, self.ja, self.ib, self.jb = ia[keep], ja[keep], ib[keep], jb[keep]
        self.istep = np.array(istep)[keep]
        self.ka = column[self.ia, self.ja]
        self.kb = column[self.ib, self.jb]

        # the four Laplacian neighbours, in a fixed order
        self.neighbours = []
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            k = column[self.fi + di, self.fj + dj]
            self.neighbours.append((k >= 0, k, self.fi + di, self.fj + dj))

    def solve(self, grid: np.ndarray, theta: np.ndarray, stiffness: float, tikhonov: float) -> np.ndarray | None:
        """One least-squares solve for the interior. A new grid, or ``None`` if
        the system is singular."""
        if not self.nfree:
            return grid
        rows_n = len(self.ia)
        if not rows_n:
            return grid

        ang = 0.5 * (theta[self.ia, self.ja] + theta[self.ib, self.jb])
        s = np.array([sin(a) for a in ang.tolist()])
        c = np.array([cos(a) for a in ang.tolist()])
        # an i step follows e1 = theta, so its residual is along perp(e1);
        # a j step follows e2, whose perpendicular is e1 itself
        nx = np.where(self.istep, -s, c)
        ny = np.where(self.istep, c, s)
        d = grid[self.ib, self.jb] - grid[self.ia, self.ja]
        weight = 1.0 / np.maximum(_root(d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]), 1e-9)

        rows, cols, vals = [], [], []
        rhs = np.zeros(rows_n)
        row = np.arange(rows_n)
        for k, gi, gj, sign in ((self.kb, self.ib, self.jb, 1.0), (self.ka, self.ia, self.ja, -1.0)):
            free = k >= 0
            rows += [row[free], row[free]]
            cols += [k[free], k[free] + 1]
            vals += [(sign * nx[free]) * weight[free], (sign * ny[free]) * weight[free]]
            g = grid[gi, gj]
            fixed_term = (sign * weight) * (nx * g[:, 0] + ny * g[:, 1])
            rhs = rhs - np.where(free, 0.0, fixed_term)

        # a plain left-to-right sum: ``sum()`` over Python floats is compensated
        # on 3.12+, and would not reproduce the same mean on every interpreter
        mean_w = float(np.cumsum(weight)[-1]) / rows_n

        # smoothing: a fixed neighbour sits at its Coons position, so its
        # correction is zero and it drops out of the row altogether
        mu = stiffness * mean_w
        share = mu / 4.0
        coons = self.coons
        base = rows_n
        smooth_rhs = []
        for axis in (0, 1):
            r = base + 2 * np.arange(self.nfree) + axis
            k = 2 * np.arange(self.nfree) + axis
            rows.append(r)
            cols.append(k)
            vals.append(np.full(self.nfree, mu))
            target = mu * coons[self.fi, self.fj, axis]
            for movable, nk, ni, nj in self.neighbours:
                rows.append(r[movable])
                cols.append(nk[movable] + axis)
                vals.append(np.full(int(movable.sum()), -share))
                target = target - np.where(movable, share * coons[ni, nj, axis], 0.0)
            smooth_rhs.append(target)
        smoothing = np.empty(2 * self.nfree)
        smoothing[0::2], smoothing[1::2] = smooth_rhs

        lam = tikhonov * mean_w
        base2 = base + 2 * self.nfree
        tik_rhs = np.empty(2 * self.nfree)
        for axis in (0, 1):
            k = 2 * np.arange(self.nfree) + axis
            rows.append(base2 + k)
            cols.append(k)
            vals.append(np.full(self.nfree, lam))
            tik_rhs[axis::2] = lam * coons[self.fi, self.fj, axis]

        total = base2 + 2 * self.nfree
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(total, 2 * self.nfree)).tocsr()
        b = np.concatenate([rhs, smoothing, tik_rhs])
        normal = (A.T @ A).tocsc()
        try:
            solution = spsolve(normal, A.T @ b)
        except Exception:  # noqa: BLE001 -- any failure of the solve means "no solution"
            return None
        if solution is None or not np.all(np.isfinite(solution)):
            return None

        out = grid.copy()
        out[self.fi, self.fj, 0] = solution[0::2]
        out[self.fi, self.fj, 1] = solution[1::2]
        return out


def _measure(grid: np.ndarray, coons: np.ndarray) -> tuple[float, float, float, int]:
    """``(min_angle, max_angle, aspect_max, folds)`` of a patch grid, folds relative to Coons."""

    def corners(g: np.ndarray) -> list[np.ndarray]:
        return [g[:-1, :-1, :2], g[:-1, 1:, :2], g[1:, 1:, :2], g[1:, :-1, :2]]

    def signed_area(q: list[np.ndarray]) -> np.ndarray:
        total = np.zeros(q[0].shape[:2])
        for a, b in zip(q, q[1:] + q[:1]):
            total = total + (a[..., 0] * b[..., 1] - b[..., 0] * a[..., 1])
        return 0.5 * total

    quad = corners(grid)
    folds = int(np.count_nonzero(signed_area(quad) * signed_area(corners(coons)) <= 0.0))

    lengths = []
    for a, b in zip(quad, quad[1:] + quad[:1]):
        d = (a - b).reshape(-1, 2).tolist()
        # Python's ``x ** 2`` and ``** 0.5``, which numpy does not reproduce exactly
        lengths.append(np.array([(x**2 + y**2) ** 0.5 for x, y in d]))
    angles = []
    for k in range(4):
        a, b, c = quad[k - 1], quad[k], quad[(k + 1) % 4]
        u, v = (a - b).reshape(-1, 2), (c - b).reshape(-1, 2)
        lu = _root(u[:, 0] * u[:, 0] + u[:, 1] * u[:, 1])
        lv = _root(v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1])
        degenerate = (lu < 1e-12) | (lv < 1e-12)
        dot = (u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]) / np.where(degenerate, 1.0, lu * lv)
        angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        angles.append(np.where(degenerate, 0.0, angle))

    angles = np.stack(angles)
    lengths = np.stack(lengths)
    shortest = lengths.min(axis=0)
    if np.any(shortest <= 0.0):
        aspect = float("inf")
    else:
        aspect = max(0.0, float((lengths.max(axis=0) / shortest).max()))
    return float(min(180.0, angles.min())), float(max(0.0, angles.max())), aspect, folds


def _accepts(measured: tuple[float, float, float, int], reference: tuple[float, float, float, int], spend: bool) -> bool:
    """Whether a relaxed patch may replace its Coons interior: never folded, and improving unless ``spend``.

    Parameters
    ----------
    measured, reference : tuple
        ``_measure`` of the relaxed grid and of the Coons grid.
    """
    lo, hi, aspect, folds = measured
    if folds:
        return False
    ref_lo, ref_hi, ref_aspect, _ = reference
    if not spend:
        return lo >= ref_lo and hi <= ref_hi and aspect <= ref_aspect
    return lo >= min(PATCH_MIN_ANGLE, ref_lo) and hi <= max(PATCH_MAX_ANGLE, ref_hi) and aspect <= max(PATCH_MAX_ASPECT, ref_aspect)


def _settle(
    system: _PatchSystem,
    coons: np.ndarray,
    sampler: FieldSampler,
    stiffness: float,
    iterations: int,
    tikhonov: float,
    still: float,
) -> tuple[str, np.ndarray | None]:
    """Solve and re-arm until the grid stops moving, at one stiffness.

    Returns
    -------
    (str, ndarray or None)
        ``'failed'`` (no field under the patch, or a singular system),
        ``'unsettled'`` (it never contracted -- an oscillation, not a
        solution), ``'flat'`` (it settled on the Coons grid) or ``'settled'``,
        with the grid.
    """
    grid = coons.copy()
    first = shift = float("inf")
    for _ in range(iterations):
        theta = _unwrap(grid, sampler)
        if theta is None:
            return "failed", None
        nxt = system.solve(grid, theta, stiffness, tikhonov)
        if nxt is None:
            return "failed", None
        shift = float(np.max(np.abs(nxt - grid)))
        first = shift if first == float("inf") else first
        grid = nxt
        if shift < still:
            break
    if shift > CONTRACTION * first and shift > still:
        return "unsettled", grid
    if float(np.max(np.abs(grid - coons))) < still:
        return "flat", grid
    return "settled", grid


def relax_patch(
    ab: list[list[float]],
    bc: list[list[float]],
    dc: list[list[float]],
    ad: list[list[float]],
    sampler: FieldSampler,
    iterations: int = ITERATIONS,
    tikhonov: float = TIKHONOV,
    stiffness: tuple[float, ...] = STIFFNESS,
    guard: bool = True,
    spend: bool = False,
    early_exit: bool = True,
) -> tuple[list[list[float]], list[list[int]], str, dict[str, Any]]:
    """The field-aware replacement for ``discrete_coons_patch``, same arguments and vertex order.

    Parameters
    ----------
    ab, bc, dc, ad : list[[x, y, z]]
        The four boundary polylines, as ``discrete_coons_patch`` takes them.
        Held fixed.
    sampler : FieldSampler
    iterations, tikhonov, stiffness : optional
        See ``ITERATIONS``, ``TIKHONOV``, ``STIFFNESS``.
    guard : bool, optional
        ``False`` takes the first stiffness that settles without asking whether
        the patch is usable. For measuring the trade-off only.
    spend : bool, optional
        Whether element quality may be spent to buy alignment. See
        ``_accepts``.
    early_exit : bool, optional
        When quality may not be spent, try the STIFFEST setting first -- the one
        closest to Coons, so the likeliest to pass -- and keep the Coons
        interior at once if even that is refused, instead of walking the whole
        schedule to the same answer.

    Returns
    -------
    (list, list, str, dict)
        Vertices, faces, a verdict -- ``'relaxed'``, ``'flat'`` (nothing moved)
        or ``'guarded'`` (no stiffness was acceptable, the Coons interior is
        kept) -- and ``stiffness``, ``min``, ``max``, ``aspect``, ``folds`` of
        the last result measured.
    """
    vertices, faces = discrete_coons_patch(ab, bc, dc, ad)
    n, m = len(ab), len(bc)
    coons = np.array([[v[0], v[1], v[2]] for v in vertices], dtype=float).reshape(n, m, 3)
    info = {"stiffness": None, "min": None, "max": None, "folds": None, "aspect": None}

    if n < 3 or m < 3:
        return vertices, faces, "flat", info

    scale = float(np.mean(np.abs(np.diff(coons[:, :, :2], axis=0)))) or 1.0
    still = TOLERANCE * scale
    system = _PatchSystem(coons)
    reference = _measure(coons, coons)

    def verdict(mu: float, grid: np.ndarray) -> tuple[bool, dict[str, Any]]:
        measured = _measure(grid, coons)
        lo, hi, aspect, folds = measured
        found = {"stiffness": mu, "min": lo, "max": hi, "folds": folds, "aspect": aspect}
        return (not guard or _accepts(measured, reference, spend)), found

    known = {}
    if early_exit and guard and not spend and len(stiffness) > 1:
        mu = stiffness[-1]
        known[mu] = _settle(system, coons, sampler, mu, iterations, tikhonov, still)
        status, grid = known[mu]
        if status == "failed":
            return vertices, faces, "guarded", info
        if status == "unsettled":
            return vertices, faces, "guarded", info
        if status == "settled":
            ok, info = verdict(mu, grid)
            if not ok:
                return vertices, faces, "guarded", info

    for mu in stiffness:
        status, grid = known.get(mu) or _settle(system, coons, sampler, mu, iterations, tikhonov, still)
        if status == "failed":
            return vertices, faces, "guarded", info
        if status == "unsettled":
            continue
        if status == "flat":
            # Coons was already the minimiser; hand the input back untouched
            return vertices, faces, "flat", info
        ok, info = verdict(mu, grid)
        if ok:
            out = [[float(p[0]), float(p[1]), float(p[2])] for p in grid.reshape(n * m, 3)]
            return out, faces, "relaxed", info

    return vertices, faces, "guarded", info


# ------------------------------------------------------------------
# the densification
# ------------------------------------------------------------------


def field_densification(
    coarse: CoarsePseudoQuadMesh,
    field: CrossField,
    locator: Any = None,
    edges_to_curves: dict[tuple[int, int], list[list[float]]] | None = None,
    iterations: int = ITERATIONS,
    tikhonov: float = TIKHONOV,
    stiffness: tuple[float, ...] = STIFFNESS,
    guard: bool = True,
    spend: bool = False,
    field_aware: bool = True,
    early_exit: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Densify a coarse layout with the field steering ordinary patch interiors.

    A drop-in for ``CoarsePseudoQuadMesh.densification``.

    Parameters
    ----------
    coarse : CoarsePseudoQuadMesh
        Strips collected and densities set.
    field : CrossField
    locator : PointLocator, optional
        Defaults to ``field.locator()``. A ``Tracer`` is accepted too.
    edges_to_curves : dict, optional
        ``{(u, v): polyline}``. An edge missing from it densifies as a chord.
    iterations, tikhonov, stiffness, guard, spend, early_exit : optional
        Passed to ``relax_patch``.
    field_aware : bool, optional
        ``False`` reproduces ``discrete_coons_patch`` exactly, for A/B tests.

    Returns
    -------
    (mesh, dict)
        The dense mesh, and counts: ``patches``; ``relaxed``, ``guarded`` and
        ``flat`` (from ``relax_patch``); ``poles`` (pseudo-quads, kept
        Coons); ``coons`` (patches left Coons because ``field_aware`` is off);
        ``stiffness`` (per relaxed patch) and ``folds``. ``seam_edge`` maps the
        geometric key of every patch-boundary node to ``(u, v, index)`` -- the
        coarse edge it lies on and its position along it -- which
        ``relax.relax_mesh`` needs to slide seam nodes without reordering
        them.
    """
    edge_strip = {}
    for strip, edges in coarse.strips(data=True):
        for u, v in edges:
            edge_strip[u, v] = strip
            edge_strip[v, u] = strip

    pole_map = [TOL.geometric_key(coarse.vertex_coordinates(pole)) for pole in coarse.poles()]

    sampler = FieldSampler(field, locator) if field_aware else None
    stats = {"patches": 0, "relaxed": 0, "poles": 0, "guarded": 0, "flat": 0, "coons": 0, "stiffness": [], "folds": 0, "seam_edge": {}}

    meshes = []
    for fkey in coarse.faces():
        stats["patches"] += 1
        polylines = []
        for u, v in coarse.face_halfedges(fkey):
            d = coarse.get_strip_density(edge_strip[u, v])
            polyline = coarse._create_patch_edge(u, v, d, edges_to_curves)
            for index, point in enumerate(polyline):
                stats["seam_edge"].setdefault(TOL.geometric_key(point), (u, v, index))
            polylines.append(polyline)

        pseudo = coarse.is_face_pseudo_quad(fkey)
        if pseudo:
            pole = coarse.attributes["face_pole"][fkey]
            polylines.insert(coarse.face_vertices(fkey).index(pole), None)

        ab, bc, cd, da = polylines
        dc = cd[::-1] if cd else None
        ad = da[::-1] if da else None

        if sampler is None or pseudo:
            vertices, faces = discrete_coons_patch(ab, bc, dc, ad)
            stats["poles" if pseudo else "coons"] += 1
        else:
            vertices, faces, verdict, info = relax_patch(
                ab, bc, dc, ad, sampler, iterations=iterations, tikhonov=tikhonov, stiffness=stiffness, guard=guard, spend=spend, early_exit=early_exit
            )
            stats[verdict] += 1
            if verdict == "relaxed":
                stats["stiffness"].append(info["stiffness"])
                stats["folds"] += info["folds"] or 0

        faces = [[u for u, v in pairwise(face + face[:1]) if u != v] for face in faces]
        meshes.append(PseudoQuadMesh.from_vertices_and_faces_with_face_poles(vertices, faces))

    face_pole_map = {}
    for mesh in meshes:
        for fkey in mesh.faces():
            fv = mesh.face_vertices(fkey)
            for u, v in pairwise(fv + fv[:1]):
                gk = TOL.geometric_key(mesh.vertex_coordinates(u))
                if gk in pole_map and gk == TOL.geometric_key(mesh.vertex_coordinates(v)):
                    face_pole_map[TOL.geometric_key(mesh.face_center(fkey))] = gk
                    break

    coarse.set_quad_mesh(meshes_join_and_weld(meshes))
    dense = coarse.get_quad_mesh()

    face_pole = {}
    for fkey in dense.faces():
        gk = TOL.geometric_key(dense.face_center(fkey))
        if gk in face_pole_map:
            for vkey in dense.face_vertices(fkey):
                if TOL.geometric_key(dense.vertex_coordinates(vkey)) == face_pole_map[gk]:
                    face_pole[fkey] = vkey
                    break
    dense.attributes["face_pole"] = face_pole

    return dense, stats
