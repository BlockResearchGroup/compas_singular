"""Point location on a background triangulation, the hottest code in the field route.

``PointLocator.locate_many`` returns exactly what ``PointLocator.locate`` would.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from compas_singular.framefield.background import BackgroundMesh


__all__ = ["PointLocator", "AMBIGUOUS"]


#: Barycentric tolerance: a point this far outside a triangle still counts as
#: inside it, so a point on a shared edge is found from either side.
INSIDE_TOL = -1e-9

#: Returned by ``PointLocator.locate_many`` for a point that more than one
#: triangle accepts. Which one ``PointLocator.locate`` picks then depends on
#: its ``hint``, so the caller has to ask ``locate`` with the right one.
AMBIGUOUS = object()


def barycentric(
    p: list[float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> tuple[float, float, float] | None:
    """Barycentric coordinates of ``p`` in triangle ``abc``, in XY, or ``None``
    for a degenerate triangle."""
    v0x, v0y = b[0] - a[0], b[1] - a[1]
    v1x, v1y = c[0] - a[0], c[1] - a[1]
    v2x, v2y = p[0] - a[0], p[1] - a[1]
    den = v0x * v1y - v1x * v0y
    if abs(den) < 1e-18:
        return None
    v = (v2x * v1y - v1x * v2y) / den
    w = (v0x * v2y - v2x * v0y) / den
    return (1.0 - v - w, v, w)


class PointLocator(object):
    """Point location on a ``BackgroundMesh``, using a face grid cached as plain arrays.

    Parameters
    ----------
    background : BackgroundMesh
    """

    def __init__(self, background: BackgroundMesh) -> None:
        self.background = background
        mesh = background.mesh
        xs = [p[0] for p in background.outer]
        ys = [p[1] for p in background.outer]
        self.cell = background.target_length * 1.5
        self.origin = (min(xs), min(ys))

        self.fkeys = list(mesh.faces())
        self.index = {fkey: i for i, fkey in enumerate(self.fkeys)}
        self.face_vertices = {}
        self.corners = {}
        for fkey in self.fkeys:
            vkeys = mesh.face_vertices(fkey)
            self.face_vertices[fkey] = vkeys
            self.corners[fkey] = tuple(tuple(mesh.vertex_coordinates(v)[:2]) for v in vkeys)
        self._neighbours = {}
        self._centroids = None

        self.grid = {}
        for fkey in self.fkeys:
            i0, i1, j0, j1 = self._cells(self.corners[fkey], 0.0)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    self.grid.setdefault((i, j), []).append(fkey)

        # The same buckets, with every face's box grown a little: a face that
        # accepts a point within ``INSIDE_TOL`` is always in the point's widened
        # bucket, which is what lets ``locate_many`` prove a match is unique.
        wide = {}
        grow = 1e-6 * self.cell
        for fkey in self.fkeys:
            i0, i1, j0, j1 = self._cells(self.corners[fkey], grow)
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    wide.setdefault((i, j), []).append(fkey)
        # every cell's candidates in one flat table: cell ``c`` owns
        # ``_flat[_start[c]:_start[c + 1]]``, and ``_ordinary`` says whether each
        # candidate is also in that cell's ordinary bucket
        self._cell_id = {}
        flat, ordinary, start = [], [], [0]
        for cell, fkeys in wide.items():
            self._cell_id[cell] = len(start) - 1
            inside = set(self.grid.get(cell, ()))
            flat.extend(self.index[f] for f in fkeys)
            ordinary.extend(f in inside for f in fkeys)
            start.append(len(flat))
        self._flat = np.array(flat, dtype=int)
        self._ordinary = np.array(ordinary, dtype=bool)
        self._start = np.array(start, dtype=int)
        self._xy = np.array([[c for corner in self.corners[f] for c in corner] for f in self.fkeys], dtype=float).reshape(-1, 6)

    def _cells(self, corners: tuple, grow: float) -> tuple[int, int, int, int]:
        ox, oy = self.origin
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        return (int((min(xs) - grow - ox) // self.cell), int((max(xs) + grow - ox) // self.cell), int((min(ys) - grow - oy) // self.cell), int((max(ys) + grow - oy) // self.cell))

    def cell_of(self, point: list[float]) -> tuple[int, int]:
        """Grid cell of a point."""
        return (int((point[0] - self.origin[0]) // self.cell), int((point[1] - self.origin[1]) // self.cell))

    def neighbours(self, fkey: int) -> list[int]:
        """Edge-adjacent faces, in the order ``mesh.face_neighbors`` gives them."""
        found = self._neighbours.get(fkey)
        if found is None:
            found = self._neighbours[fkey] = list(self.background.mesh.face_neighbors(fkey))
        return found

    # ------------------------------------------------------------------
    # one point
    # ------------------------------------------------------------------

    def test(self, fkey: int, point: list[float]) -> tuple[int, tuple[float, float, float]] | None:
        """``(fkey, barycentric)`` if the face contains the point, else ``None``."""
        a, b, c = self.corners[fkey]
        bary = barycentric(point, a, b, c)
        if bary is None:
            return None
        if bary[0] >= INSIDE_TOL and bary[1] >= INSIDE_TOL and bary[2] >= INSIDE_TOL:
            return fkey, bary
        return None

    def locate(self, point: list[float], hint: int | None = None) -> tuple[int, tuple[float, float, float]] | None:
        """Containing face and barycentric coordinates, or ``None`` if outside.

        ``hint`` and its neighbours are tried first: consecutive queries almost
        always land in the same or an adjacent triangle.
        """
        if hint is not None:
            found = self.test(hint, point)
            if found:
                return found
            for fkey in self.neighbours(hint):
                found = self.test(fkey, point)
                if found:
                    return found
        for fkey in self.grid.get(self.cell_of(point), ()):
            found = self.test(fkey, point)
            if found:
                return found
        return None

    def nearest(self, point: list[float]) -> tuple[int, tuple[float, float, float]] | None:
        """The face whose centroid is closest, within the surrounding 3 x 3 cells,
        with barycentric ``(1/3, 1/3, 1/3)``. For a point just outside the domain."""
        if self._centroids is None:
            mesh = self.background.mesh
            self._centroids = {f: mesh.face_centroid(f) for f in self.fkeys}
        i, j = self.cell_of(point)
        best, best_d = None, float("inf")
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for fkey in self.grid.get((i + di, j + dj), ()):
                    c = self._centroids[fkey]
                    d = (c[0] - point[0]) ** 2 + (c[1] - point[1]) ** 2
                    if d < best_d:
                        best, best_d = fkey, d
        if best is None:
            return None
        return best, (1 / 3.0, 1 / 3.0, 1 / 3.0)

    # ------------------------------------------------------------------
    # many points
    # ------------------------------------------------------------------

    def locate_arrays(self, points: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``locate`` for a batch, without a hint, as arrays.

        Returns
        -------
        (ndarray, ndarray, ndarray)
            ``status`` per point -- 0 when no face accepts it, 1 when exactly
            one does, 2 when several do (``AMBIGUOUS``: which one ``locate``
            picks then depends on its hint) -- and, where ``status`` is 1, the
            face's position in ``fkeys`` and the barycentric coordinates.
            Status 0 and 1 are exactly what ``locate`` returns for ANY hint.
        """
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        count = len(pts)
        status = np.zeros(count, dtype=int)
        faces = np.full(count, -1, dtype=int)
        bary = np.zeros((count, 3))
        if not count:
            return status, faces, bary

        ci = np.floor_divide(pts[:, 0] - self.origin[0], self.cell).astype(np.int64)
        cj = np.floor_divide(pts[:, 1] - self.origin[1], self.cell).astype(np.int64)
        cells, inverse = np.unique(np.stack([ci, cj], axis=1), axis=0, return_inverse=True)
        ids = np.array([self._cell_id.get((i, j), -1) for i, j in cells.tolist()], dtype=int)
        cell = ids[inverse.ravel()]
        known = cell >= 0
        begin = np.where(known, self._start[np.maximum(cell, 0)], 0)
        sizes = np.where(known, self._start[np.maximum(cell, 0) + 1] - begin, 0)
        total = int(sizes.sum())
        if not total:
            return status, faces, bary
        k = np.repeat(np.arange(count), sizes)
        offset = np.arange(total) - np.repeat(np.cumsum(sizes) - sizes, sizes)
        slot = np.repeat(begin, sizes) + offset
        f = self._flat[slot]
        in_bucket = self._ordinary[slot]

        ax, ay, bx, by, cx, cy = self._xy[f].T
        px, py = pts[k, 0], pts[k, 1]
        # the same arithmetic, in the same order, as ``barycentric``
        v0x, v0y = bx - ax, by - ay
        v1x, v1y = cx - ax, cy - ay
        v2x, v2y = px - ax, py - ay
        den = v0x * v1y - v1x * v0y
        good = np.abs(den) >= 1e-18
        safe = np.where(good, den, 1.0)
        v = (v2x * v1y - v1x * v2y) / safe
        w = (v0x * v2y - v2x * v0y) / safe
        u = 1.0 - v - w
        inside = np.flatnonzero(good & (u >= INSIDE_TOL) & (v >= INSIDE_TOL) & (w >= INSIDE_TOL))

        hits = np.bincount(k[inside], minlength=count)
        status[hits > 1] = 2
        lone = inside[hits[k[inside]] == 1]
        points_hit = k[lone]
        # a lone match must also be in the point's ordinary bucket, or ``locate``
        # without a hint would not have found it
        status[points_hit] = np.where(in_bucket[lone], 1, 2)
        keep = in_bucket[lone]
        faces[points_hit[keep]] = f[lone[keep]]
        bary[points_hit[keep]] = np.stack([u[lone[keep]], v[lone[keep]], w[lone[keep]]], axis=1)
        return status, faces, bary

    def locate_many(self, points: Any) -> list[Any]:
        """``locate_arrays`` as a list: per point ``(fkey, barycentric)``,
        ``None``, or ``AMBIGUOUS``."""
        status, faces, bary = self.locate_arrays(points)
        out = []
        for s, f, b in zip(status.tolist(), faces.tolist(), bary.tolist()):
            if s == 1:
                out.append((self.fkeys[f], tuple(b)))
            else:
                out.append(None if s == 0 else AMBIGUOUS)
        return out
