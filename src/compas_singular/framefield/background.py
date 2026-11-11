"""The background triangulation the field lives on: walls discretised at ``target_length`` plus an interior grid."""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING
from typing import Any

from compas.geometry import cross_vectors
from compas.geometry import delaunay_triangulation
from compas.geometry import distance_point_point
from compas.geometry import length_vector
from compas.geometry import normalize_vector
from compas.geometry import subtract_vectors
from compas_singular.datastructures import Mesh  # has .boundaries(); compas core does not
from compas_singular.geometry.polyline import bounding_box_diagonal
from compas_singular.geometry.polyline import discretise_boundary
from compas_singular.geometry.polyline import near_loop
from compas_singular.geometry.polyline import points_in_polygon_xy

if TYPE_CHECKING:
    from compas_singular.framefield.symmetry import Symmetry


# ``discretise_boundary`` is shared with the skeleton route, so the two routes
# discretise their walls identically.
__all__ = ["BackgroundMesh", "discretise_boundary", "interior_grid", "inside_domain"]


def _as_open_loop(points: Any) -> list[list[float]]:
    """Drop a repeated closing point and any consecutive duplicates."""
    pts = [[float(p[0]), float(p[1]), 0.0] for p in points]
    out = [pts[0]]
    for p in pts[1:]:
        if distance_point_point(p, out[-1]) > 1e-9:
            out.append(p)
    if len(out) > 1 and distance_point_point(out[0], out[-1]) < 1e-9:
        out.pop()
    return out


def inside_domain(
    points: list[list[float]],
    outer: list[list[float]],
    inners: Any = (),
    clearance: float = 0.0,
) -> list[bool]:
    """Per point: inside ``outer``, outside every hole, and -- when
    ``clearance`` is positive -- at least that far from every wall."""
    inners = list(inners or [])
    keep = points_in_polygon_xy(points, outer)
    for loop in inners:
        keep = [k and not h for k, h in zip(keep, points_in_polygon_xy(points, loop))]
    if clearance > 0.0:
        for loop in [outer] + inners:
            candidates = [p for p, k in zip(points, keep) if k]
            close = iter(near_loop(candidates, loop, clearance))
            keep = [k and not next(close) if k else False for k in keep]
    return keep


def interior_grid(
    outer: list[list[float]],
    inners: Any = (),
    target_length: float | None = None,
    margin: float = 0.45,
    symmetry: Symmetry | None = None,
) -> list[list[float]]:
    """The INTERIOR points of the background mesh, at ``target_length`` spacing.

    A jittered grid over the outline's bounding box, less every point outside
    the domain or within ``margin * target_length`` of a wall.

    Parameters
    ----------
    outer : list[[x, y, z]]
        The domain outline, open and already discretised.
    inners : list[list[[x, y, z]]], optional
        The holes, same convention.
    target_length : float
        Grid spacing -- the spacing the walls were discretised at.
    margin : float, optional
        Clearance from the walls, as a fraction of ``target_length``, so the
        grid does not crowd the wall into slivers.
    symmetry : framefield.symmetry.Symmetry, optional
        When given and non-trivial, a grid exactly invariant under the group
        replaces this one -- see ``symmetry.interior_points``. The plain
        grid is not: its jitter and its anchoring both ignore the symmetry.

    Returns
    -------
    list[[x, y, z]]
    """
    inners = list(inners or [])
    limit = margin * target_length

    if symmetry is not None and symmetry.enabled("background"):
        # imported here: symmetry.py imports _jitter from this module
        from compas_singular.framefield.symmetry import interior_points

        return list(interior_points(symmetry, target_length, outer, inners, margin=margin))

    xs = [p[0] for p in outer]
    ys = [p[1] for p in outer]
    nx = int(ceil((max(xs) - min(xs)) / target_length))
    ny = int(ceil((max(ys) - min(ys)) / target_length))
    grid = [
        [min(xs) + (i + 0.5) * target_length + _jitter(i, j, 0.2 * target_length), min(ys) + (j + 0.5) * target_length + _jitter(j, i, 0.2 * target_length), 0.0]
        for i in range(nx + 1)
        for j in range(ny + 1)
    ]
    return [p for p, keep in zip(grid, inside_domain(grid, outer, inners, limit)) if keep]


def _jitter(i: int, j: int, amount: float) -> float:
    """Deterministic sub-cell offset, so Qhull's diagonal choice on a regular grid is reproducible."""
    h = (i * 73856093) ^ (j * 19349663)
    return (((h >> 8) & 0xFFFF) / 65535.0 - 0.5) * amount


class BackgroundMesh(object):
    """A well-shaped triangulation of a planar domain, with boundary tangents.

    Attributes
    ----------
    mesh : compas_singular.datastructures.Mesh
        The triangulation. All faces have positive area in XY.
    outer : list[[x, y, z]]
        The discretised outer boundary, open (no repeated last point).
    inners : list[list[[x, y, z]]]
        The discretised holes, same convention.
    target_length : float
        The spacing the triangulation was built at.
    """

    def __init__(
        self,
        mesh: Mesh,
        outer: list[list[float]],
        inners: list[list[list[float]]],
        target_length: float,
    ) -> None:
        self.mesh = mesh
        self.outer = outer
        self.inners = inners
        self.target_length = target_length
        self._boundary_tangents = None

    @property
    def loops(self) -> list[list[list[float]]]:
        """Every boundary loop, outer first."""
        return [self.outer] + list(self.inners)

    @property
    def diagonal(self) -> float:
        """Bounding-box diagonal of the outer boundary."""
        return bounding_box_diagonal(self.outer)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_boundary(
        cls,
        outer_boundary: list[list[float]],
        inner_boundaries: list[list[list[float]]] | None = None,
        target_length: float | None = None,
        margin: float = 0.45,
        symmetry: Symmetry | None = None,
        alpha: float = 0.02,
        d_min: int = 5,
    ) -> BackgroundMesh:
        """Triangulate the region inside ``outer_boundary`` and outside the holes.

        Parameters
        ----------
        outer_boundary : list[[x, y, z]]
            The domain outline, closed or open.
        inner_boundaries : list[list[[x, y, z]]], optional
            Holes.
        target_length : float, optional
            Edge length, for the walls and the interior grid alike. Defaults to
            ``alpha`` times the bounding-box diagonal (thesis eq. 4.1, see
            ``discretise_boundary``).
        margin : float, optional
            Interior points' clearance from the walls, as a fraction of
            ``target_length``.
        symmetry : framefield.symmetry.Symmetry, optional
            Passed to ``interior_grid``.
        alpha : float, optional
            ``target_length`` as a fraction of the diagonal, when not given.
        d_min : int, optional
            Fewest points per boundary loop.

        Returns
        -------
        BackgroundMesh
        """
        # Resolved here rather than inside ``discretise_boundary``: the interior
        # grid needs the same spacing, and the instance stores it.
        if target_length is None:
            loops = [_as_open_loop(outer_boundary)]
            loops += [_as_open_loop(loop) for loop in (inner_boundaries or [])]
            target_length = alpha * bounding_box_diagonal(*loops)

        outer, inners = discretise_boundary(outer_boundary, inner_boundaries, spacing=target_length, d_min=d_min)

        points = list(outer)
        for loop in inners:
            points.extend(loop)
        points.extend(interior_grid(outer, inners, target_length, margin=margin, symmetry=symmetry))

        faces = delaunay_triangulation(points)
        mesh = Mesh.from_vertices_and_faces(points, faces)

        # Drop zero-area faces, then faces whose centroid is outside the domain
        # -- the centroid, because a sliver's circumcentre can land far outside
        # a face that is inside.
        kept = []
        for fkey in list(mesh.faces()):
            a, b, c = [mesh.vertex_coordinates(v) for v in mesh.face_vertices(fkey)]
            if length_vector(cross_vectors(subtract_vectors(b, a), subtract_vectors(c, a))) < 1e-12:
                mesh.delete_face(fkey)
            else:
                kept.append(fkey)
        centres = [mesh.face_centroid(fkey) for fkey in kept]
        for fkey, inside in zip(kept, inside_domain(centres, outer, inners)):
            if not inside:
                mesh.delete_face(fkey)

        for vkey in list(mesh.vertices()):
            if not mesh.vertex_faces(vkey):
                mesh.delete_vertex(vkey)

        return cls(mesh, outer, inners, target_length)

    # ------------------------------------------------------------------
    # tangent spaces
    # ------------------------------------------------------------------

    def face_basis(self, fkey: int) -> tuple[list[float], list[float]]:
        """Orthonormal tangent basis of a face: world XY, on a planar domain."""
        return ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    def boundary_tangents(self) -> dict[int, list[list[float]]]:
        """The two adjacent boundary edge directions at every boundary vertex (not their average).

        Returns
        -------
        dict[int, list[[x, y, z]]]
        """
        if self._boundary_tangents is None:
            tangents = {}
            for loop in self.mesh.boundaries():
                n = len(loop)
                for i, vkey in enumerate(loop):
                    here = self.mesh.vertex_coordinates(vkey)
                    prev = self.mesh.vertex_coordinates(loop[(i - 1) % n])
                    nxt = self.mesh.vertex_coordinates(loop[(i + 1) % n])
                    tangents[vkey] = [
                        normalize_vector(subtract_vectors(here, prev)),
                        normalize_vector(subtract_vectors(nxt, here)),
                    ]
            self._boundary_tangents = tangents
        return self._boundary_tangents

    def boundary_vertices(self) -> set[int]:
        """Set of vertex keys on any boundary."""
        return set(self.boundary_tangents())

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def validate(self) -> dict[str, Any]:
        """Check the invariants the field solve relies on.

        Returns
        -------
        dict
            ``ok`` plus the counts behind it: no negative-area face, triangles
            only, one boundary loop per wall, and Euler characteristic
            ``1 - holes`` (which catches a face bridging a hole).
        """
        mesh = self.mesh
        report = {
            "vertices": mesh.number_of_vertices(),
            "faces": mesh.number_of_faces(),
            "edges": mesh.number_of_edges(),
        }

        negative = []
        for fkey in mesh.faces():
            a, b, c = [mesh.vertex_coordinates(v) for v in mesh.face_vertices(fkey)]
            area2 = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
            if area2 <= 0.0:
                negative.append(fkey)
        report["negative_area_faces"] = len(negative)

        report["triangles_only"] = all(len(mesh.face_vertices(f)) == 3 for f in mesh.faces())

        loops = mesh.boundaries()
        report["boundary_loops"] = len(loops)
        report["expected_loops"] = 1 + len(self.inners)

        chi = report["vertices"] - report["edges"] + report["faces"]
        report["euler"] = chi
        report["expected_euler"] = 1 - len(self.inners)

        report["ok"] = (
            report["negative_area_faces"] == 0
            and report["triangles_only"]
            and report["boundary_loops"] == report["expected_loops"]
            and report["euler"] == report["expected_euler"]
        )
        return report
