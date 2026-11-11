from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import floor

# from math import ceil
from math import pi
from operator import itemgetter
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence

from compas.datastructures import Graph
from compas.datastructures.graph.operations.join import graph_polylines
from compas.datastructures.mesh.operations.insert import mesh_insert_vertex_on_edge
from compas.datastructures.mesh.operations.substitute import mesh_substitute_vertex_in_faces
from compas.datastructures.mesh.operations.weld import mesh_unweld_edges
from compas.geometry import Polyline
from compas.geometry import angle_points
from compas.geometry import angle_vectors
from compas.geometry import angle_vectors_signed

# from compas.geometry import cross_vectors
from compas.geometry import centroid_points
from compas.geometry import distance_point_point

# from compas.geometry import length_vector
# from compas.geometry import length_vector_xy
from compas.geometry import subtract_vectors
from compas.itertools import pairwise
from compas.itertools import window
from compas.tolerance import TOL
from compas_singular.algorithms import boundary_triangulation
from compas_singular.algorithms.propagation import quadrangulate_faces
from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.datastructures import Skeleton
from compas_singular.datastructures import mesh_weld
from compas_singular.datastructures import split_quad_in_pseudo_quads
from compas_singular.datastructures import trimesh_face_circle
from compas_singular.geometry import bounding_box_diagonal
from compas_singular.geometry import discretise_boundary
from compas_singular.geometry import discretise_line
from compas_singular.utilities import list_split

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarseQuadMesh
    from compas_singular.datastructures import Mesh
    from compas_singular.symmetry import SymmetryReport
    from compas_singular.symmetry.unit import SymmetricUnit


__all__ = ["SkeletonDecomposition"]


def _wall_spacing(outer: list[list[float]], inners: list[list[list[float]]], target_length: float | None, alpha: float | None) -> float | None:
    """The segment length the walls were discretised at, used for the features too."""
    if target_length is not None:
        return float(target_length)
    if alpha is None:
        return None
    diagonal = bounding_box_diagonal(outer, *inners)
    return alpha * diagonal if diagonal > 0.0 else None


class SkeletonDecomposition(Skeleton):
    """Coarse quad mesh from a topological skeleton and its singularities (Oval's thesis, chapter 4).

    The thesis-to-method map and the known deviations are in ``design_notes/algorithms.md``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(SkeletonDecomposition, self).__init__(*args, **kwargs)
        self.mesh = None
        self.polylines = None
        self.relative_kink_angle_limit = pi / 8.0
        self.flip_angle_limit = pi / 2.0
        #: How far each copy of a duplicated vertex moves toward its own
        #: neighbour centroid, as a fraction of that distance. See
        #: ``solve_triangular_faces``.
        self.collapsed_edge_opening = 0.5
        self.repair_notes = []

        self.origin = None  # skeleton, mesh or boundary
        self.inputs = {}
        #: What ``find_symmetry`` found, or ``None``.
        self.symmetry_report = None
        #: The ``poles`` ``mesh`` was built with, as a rounded tuple. The
        #: cache key of ``coarse_mesh`` -- see there.
        self._mesh_poles = None

    @classmethod
    def from_skeleton(cls, skeleton: Skeleton) -> SkeletonDecomposition:
        """Construct a SkeletonDecomposition object from a Skeleton.

        Returns
        -------
        Skeleton
            A skeleton object.

        """
        return cls.from_vertices_and_faces(*skeleton.to_vertices_and_faces())

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> SkeletonDecomposition:
        """Construct a SkeletonDecomposition object from a Mesh.

        Returns
        -------
        Skeleton
            A skeleton object.

        """
        skeleton = cls.from_vertices_and_faces(*mesh.to_vertices_and_faces())
        skeleton.feature_edges = frozenset(mesh.attributes.get("feature_edges") or ())
        skeleton.feature_points = list(mesh.attributes.get("feature_points") or ())
        return skeleton

    @classmethod
    def from_boundary(
        cls,
        outer_boundary: list[list[float]],
        inner_boundaries: list[list[list[float]]] | None = None,
        polyline_features: list[list[list[float]]] = [],
        point_features: list[list[float]] = [],
        target_length: float | None = None,
        alpha: float | None = 0.02,
        d_min: int | None = 5,
    ) -> SkeletonDecomposition:
        """Triangulate a domain given by its walls, ready to decompose.

        Parameters
        ----------
        outer_boundary : list[[x, y, z]]
            The outer wall, as an open loop -- the last point is not the first.
        inner_boundaries : list[list[[x, y, z]]], optional
            The holes, same convention.
        polyline_features : list[list[[x, y, z]]], optional
            Feature curves the decomposition must follow. See
            ``examples/000_testing.py`` for every benchmark stage by stage, and
            ``markdowns/HOW_IT_WORKS.md`` section 5 for what still does not work.
        point_features : list[[x, y, z]], optional
            Points the decomposition must pass through. They become the POLES of
            the layout, and ``coarse_mesh`` takes them from here.
        target_length : float, optional
            Background spacing the walls are discretised to, not the quad size.
            ``None`` uses ``alpha`` times the bounding-box diagonal; pass
            ``alpha=None`` as well to triangulate the loops as handed in.
        alpha : float, optional
            Fraction of the bounding-box diagonal to use as the target length
            when none is given. Thesis eq. 4.1; see
            ``compas_singular.geometry.discretise_boundary``.
        d_min : int, optional
            Fewest points per boundary loop, whatever the target length says.

        Returns
        -------
        SkeletonDecomposition
        """
        # Discretised for the TRIANGULATION only. What is stored below is what
        # the caller handed in, because that is the most faithful description of
        # the walls there is: ``edges_to_curves`` samples them again, finer, and
        # would otherwise be re-sampling this function's own approximation.
        inner_boundaries = list(inner_boundaries or [])
        outer_dense, inners_dense = discretise_boundary(outer_boundary, inner_boundaries, spacing=target_length, alpha=alpha, d_min=d_min)
        # The features get the SAME sampling as the walls. Thesis eq. 4.1 is
        # stated for every curve, and a feature is cut into the Delaunay along
        # its OWN segments: one longer than the surrounding sampling is not a
        # Delaunay edge, so the cut does not happen there at all. A straight
        # guide drawn in Rhino arrives as TWO points -- ``curve_points`` takes a
        # polyline at its own vertices -- and was triangulated as one long
        # segment the layout ignored completely.
        spacing = _wall_spacing(outer_dense, inners_dense, target_length, alpha)
        features_dense = [discretise_line(feature, spacing) for feature in (polyline_features or [])]
        trimesh = boundary_triangulation(outer_boundary=outer_dense, inner_boundaries=inners_dense, polyline_features=features_dense, point_features=point_features)

        decomposition = cls.from_vertices_and_faces(*trimesh.to_vertices_and_faces())
        decomposition.feature_edges = frozenset(trimesh.attributes.get("feature_edges") or ())
        decomposition.feature_points = list(trimesh.attributes.get("feature_points") or ())
        decomposition.origin = "boundary"
        # ``inputs``, matching ``__init__``. It used to be written as ``input``,
        # so ``coarse_mesh`` read the empty dict and every point feature was
        # silently dropped instead of becoming a pole.
        # The RESOLVED length is stored, not the argument: ``edges_to_curves``
        # derives its wall sampling from it, and a stored ``None`` would leave
        # the walls unsampled for every caller who took the thesis default.
        if target_length is None and alpha is not None:
            diagonal = bounding_box_diagonal(outer_dense, *inners_dense)
            if diagonal > 0.0:
                target_length = alpha * diagonal
        decomposition.inputs = {
            "outer_boundary": outer_boundary,
            "inner_boundaries": inner_boundaries,
            "polyline_features": polyline_features,
            "point_features": point_features,
            "target_length": target_length,
            "alpha": alpha,
            "d_min": d_min,
        }

        return decomposition

    # --------------------------------------------------------------------------
    # symmetry
    # --------------------------------------------------------------------------

    def find_symmetry(self, tol: float | None = None, include: Iterable[str] = ("walls", "holes", "guides", "poles"), max_order: int = 12) -> SymmetryReport:
        """Detect the symmetry of the domain (walls, holes, curve and point features).

        The result is kept on ``symmetry_report``.

        Returns
        -------
        compas_singular.symmetry.SymmetryReport
            ``print`` it for the group and its keys; ``report.geometry()`` gives
            the centre, mirror lines and rotation arcs to draw.
        """
        from compas_singular.symmetry import find_symmetry
        from compas_singular.symmetry.routes import domain_of

        self.symmetry_report = find_symmetry(domain=domain_of(self), tol=tol, include=include, max_order=max_order)
        return self.symmetry_report

    def symmetry_unit(self, keys: list[str] | None = None, centre: str = "route", seam: float | None = None, report: SymmetryReport | None = None) -> SymmetricUnit:
        """Mesh one symmetric unit of the domain with this route, cut along the seams of ``keys``.

        Densify with ``quad_mesh()`` and expand with ``expand_symmetrically()``.

        Parameters
        ----------
        keys : list[str], optional
            Keys from ``find_symmetry`` -- ``['M0', 'M90']``. The subgroup they
            GENERATE is enforced. ``None`` enforces everything detected.
        centre : {'route'}
        seam : float, optional
            Rotation-only groups: the angle of the first seam, radians.
        report : SymmetryReport, optional
            Defaults to ``symmetry_report``, detected now if missing.

        Returns
        -------
        compas_singular.symmetry.SymmetricUnit
        """
        from compas_singular.symmetry import build_unit
        from compas_singular.symmetry.routes import mesher_for

        report = report or self.symmetry_report or self.find_symmetry()
        return build_unit(report, mesher_for(self), keys=keys, centre=centre, seam=seam)

    # --------------------------------------------------------------------------
    # key elements
    # --------------------------------------------------------------------------

    def corner_faces(self) -> list[int]:
        """Get the indices of the corner faces in the Delaunay mesh, i.e. the ones with one neighbour.

        Thesis S4.2.1 calls these END faces: "end faces have one adjacent face".
        Regular faces have two and singular faces three.

        Returns
        -------
        list
            List of face keys.

        """
        return [fkey for fkey in self.faces() if len(self.real_neighbors(fkey)) == 1]

    def corner_vertices(self) -> list[int]:
        """Get the indices of the corner vertices of the topological skeleton, i.e. the two-valent boundary vertices in the Delaunay mesh.

        The "two-valent boundary vertices" the CLOSING operation of thesis
        S4.2.2 splits the boundary at.

        Returns
        -------
        list
            List of vertex keys.

        """
        return [vkey for bdry in self.vertices_on_boundaries() for vkey in bdry if len(self.vertex_neighbors(vkey)) == 2]

    def split_vertices(self) -> list[int]:
        """Get the indices of the boundary split vertices, i.e. the vertices of the singular faces.

        The other half of what CLOSING splits the boundary at (thesis S4.2.2):
        "the vertices of the singular faces".

        Returns
        -------
        list
            List of vertex keys.

        """
        return [vkey for fkey in self.singular_faces() for vkey in self.face_vertices(fkey)]

    def free_tip_vertices(self) -> list[int]:
        """Get the vertices at free extremities of curve features (interior angle 360 degrees).

        Thesis S4.3.2 makes each one a singularity, so it stays a node of the decomposition.

        Returns
        -------
        list
            List of vertex keys.
        """
        return [vkey for bdry in self.vertices_on_boundaries() for vkey in bdry if self.boundary_interior_angle(vkey) > 2 * pi - 0.05]

    # --------------------------------------------------------------------------
    # branches
    # --------------------------------------------------------------------------

    def branches_singularity_to_singularity(self) -> list[list[list[float]]]:
        """Get the skeleton branches between singularities only: PRUNING (thesis S4.2.2).

        Returns
        -------
        list
            List of polylines as list of point XYZ-coordinates.
        """
        map_corners = [TOL.geometric_key(trimesh_face_circle(self, corner)[0]) for corner in self.corner_faces()]
        return [branch for branch in self.branches() if TOL.geometric_key(branch[0]) not in map_corners and TOL.geometric_key(branch[-1]) not in map_corners]

    def branches_singularity_to_boundary(self) -> list[list[list[float]]]:
        """Get new branches from singular faces to their split vertices: GRAFTING (thesis S4.2.2).

        Returns
        -------
        list
            List of polylines as list of point XYZ-coordinates.
        """
        grafts = [[trimesh_face_circle(self, fkey)[0], self.vertex_coordinates(vkey)] for fkey in self.singular_faces() for vkey in self.face_vertices(fkey)]
        return self.merge_graft_targets(grafts)

    def merge_graft_targets(self, grafts: list[list[list[float]]]) -> list[list[list[float]]]:
        """Merge grafts that land on one feature at adjacent samples into one node.

        Otherwise the patch between them comes out a triangle.

        Parameters
        ----------
        grafts : list[[[x, y, z], [x, y, z]]]
            Branches as ``[circumcentre, target]``.

        Returns
        -------
        list[[[x, y, z], [x, y, z]]]
        """
        if not self.feature_points:
            return grafts

        sample_of = {}
        for c, chain in enumerate(self.feature_points):
            for i, sample in enumerate(chain):
                sample_of.setdefault(TOL.geometric_key(sample), (c, i))

        on_feature = {}
        for _, target in grafts:
            key = TOL.geometric_key(target)
            if key in sample_of:
                on_feature[key] = sample_of[key]
        if len(on_feature) < 2:
            return grafts

        # walk each chain in order, grouping runs of samples that touch
        move = {}
        for c in range(len(self.feature_points)):
            hits = sorted(((i, key) for key, (chain, i) in on_feature.items() if chain == c))
            run = []
            for i, key in hits + [(None, None)]:
                if run and (i is None or i - run[-1][0] > 1):
                    if len(run) > 1:
                        # A chain extremity -- free tip, wall landing, junction --
                        # keeps its node. Moving its graft one sample inward, as
                        # chain order would whenever the chain runs towards it,
                        # leaves the extremity with no branch at all.
                        last = len(self.feature_points[c]) - 1
                        ends = [j for j, _ in run if j in (0, last)]
                        shared = self.feature_points[c][ends[0] if ends else run[0][0]]
                        for _, member in run:
                            move[member] = shared
                    run = []
                if i is not None:
                    run.append((i, key))

        if not move:
            return grafts
        return [[centre, move.get(TOL.geometric_key(target), target)] for centre, target in grafts]

    def branches_boundary(self) -> list[list[list[float]]]:
        """Get the boundary branches, split at corner and split vertices: CLOSING (thesis S4.2.2).

        Returns
        -------
        list
            List of polylines as list of point XYZ-coordinates.
        """
        boundaries = [bdry + bdry[0:] for bdry in self.boundaries()]
        splits = self.corner_vertices() + self.split_vertices()
        split_boundaries = [split_boundary for boundary in boundaries for split_boundary in list_split(boundary, [boundary.index(split) for split in splits if split in boundary])]
        return [[self.vertex_coordinates(vkey) for vkey in boundary] for boundary in split_boundaries]

    # --------------------------------------------------------------------------
    # decomposition
    # --------------------------------------------------------------------------

    def decomposition_polylines(self) -> list[list[list[float]]]:
        """Get all branch polylines of the decomposition: pruning, grafting, closing, then the S4.2.3 corrections.

        Returns
        -------
        list
            List of polylines as list of point XYZ-coordinates.
        """
        branches = self.branches_singularity_to_singularity() + self.branches_singularity_to_boundary() + self.branches_boundary()
        # Thesis S4.2.3 orders the corrections, and says why:
        # concavities -> unwanted triangles -> flipped patches -> collapsed.
        # "Adding branches for other corrections can solve collapsed
        # boundaries. Therefore, correcting collapsed boundaries occurs
        # last." The unwanted-triangle correction is not here at all --
        # solve_triangular_faces runs on the built mesh, after this.
        branches += self.branches_splitting_boundary_kinks()
        branches += self.branches_splitting_flipped_faces()
        branches += self.branches_splitting_collapsed_boundaries()
        # Free feature tips are split points like corners. Both sides of the slit
        # join into one segment here, so a grafted tip has only two branches -- the
        # graft and the segment -- and would be merged through, losing the
        # singularity at the extremity (Fig 4.21) and cutting across the feature.
        splits = [self.vertex_coordinates(vkey) for vkey in self.corner_vertices() + self.free_tip_vertices()]
        self.polylines = graph_polylines(Graph.from_lines([(u, v) for polyline in branches for u, v in pairwise(polyline)]), splits=splits)
        return self.polylines

    def decomposition_polyline(self, geom_key_1: str, geom_key_2: str) -> list[list[float]] | None:
        """Retrieve the decomposition polyline with extremities corresponding to two geoemtric keys.

        Parameters
        ----------
        geom_key_1 : float
            Geometric key of one extremity.
        geom_key_2 : float
            Geometric key of the other extremity.

        Returns
        -------
        list, None
            A polyline as a list of point XYZ-coordinates if a polyline corresponds to the geometric keys, None otherwise.
        """
        polylines = {(TOL.geometric_key(polyline[0]), TOL.geometric_key(polyline[-1])): polyline for polyline in self.polylines}
        return polylines.get((geom_key_1, geom_key_2), polylines.get((geom_key_2, geom_key_1), None))

    def decomposition_mesh(self, poles: list[list[float]]) -> CoarsePseudoQuadMesh:
        """Return a quad mesh based on the decomposition polylines.
        Some fixes are added to convert the mesh formed by the decomposition polylines into a (coarse) quad mesh.

        Returns
        -------
        mesh
            A coarse quad mesh based on the topological skeleton from a Delaunay mesh.

        """
        polylines = self.decomposition_polylines()
        boundary_keys = set([TOL.geometric_key(self.vertex_coordinates(vkey)) for bdry in self.vertices_on_boundaries() for vkey in bdry])
        boundary_polylines = [polyline for polyline in polylines if TOL.geometric_key(polyline[0]) in boundary_keys and TOL.geometric_key(polyline[1]) in boundary_keys]
        other_polylines = [polyline for polyline in polylines if TOL.geometric_key(polyline[0]) not in boundary_keys or TOL.geometric_key(polyline[1]) not in boundary_keys]
        self.repair_notes = []
        mesh = CoarsePseudoQuadMesh.from_polylines(boundary_polylines, other_polylines)
        mesh.attributes["decomposition_type"] = "skeleton"
        self.mesh = mesh
        self.solve_triangular_faces()
        self.quadrangulate_polygonal_faces()
        self.repair_polygonal_faces(poles)
        self.split_quads_with_poles(poles)
        self.store_pole_data(poles)
        return self.mesh

    def coarse_mesh(self, poles: list[list[float]] | None = None, force: bool = False) -> CoarsePseudoQuadMesh:
        """The coarse quad layout, built once from the domain's point features and cached.

        Returns the same object on repeat calls, so densities set on it are kept.

        Parameters
        ----------
        poles : list[[x, y, z]], optional
            Preferred pole positions. ``None`` takes them from the
            ``point_features`` of ``from_boundary``, or none at all for a
            decomposition built from a mesh or a skeleton. An explicit list wins.
            Different poles miss the cache.
        force : bool, optional
            Rebuild even on a cache hit -- for a caller that wants a clean layout
            back after mutating the one it was given.

        Returns
        -------
        CoarsePseudoQuadMesh
        """
        if poles is None:
            poles = self.inputs.get("point_features") or []
        poles_key = tuple(tuple(round(float(c), 6) for c in point) for point in poles)
        if not force and self.mesh is not None and self._mesh_poles == poles_key:
            return self.mesh
        mesh = self.decomposition_mesh(poles)
        # Only a decomposition built by ``from_boundary`` has walls to derive
        # curves from -- ``edges_to_curves()`` raises otherwise, and a
        # from_mesh/from_skeleton decomposition is a documented, legitimate way
        # to reach ``coarse_mesh()``, so it must not start raising here too.
        if self.inputs.get("outer_boundary"):
            mapping, _tally = self.edges_to_curves(coarse=mesh)
            mesh.set_edges_to_curves(mapping)
        self._mesh_poles = poles_key
        return mesh

    def edges_to_curves(
        self, coarse: CoarseQuadMesh | CoarsePseudoQuadMesh | None = None, wall_sampling: float | None = None, snap: bool = True
    ) -> tuple[dict[tuple[int, int], list[list[float]]], dict[str, int]]:
        """The shape of every coarse edge, for ``densification``: wall piece, matching branch, or chord.

        Parameters
        ----------
        coarse : CoarseQuadMesh, optional
            The layout to describe. Defaults to ``mesh``.
        wall_sampling : float, optional
            How finely to resample the walls. Defaults to a quarter of the
            background ``target_length``, the ratio ``CMD_coarse_mesh`` uses.
        snap : bool, optional
            Put the layout's boundary corners on the walls first. Mutates the
            layout; it must happen before the arcs are built.

        Returns
        -------
        (dict, dict)
            ``{(u, v): polyline}`` to hand to ``densification``, and the tally.
            A non-zero ``chord`` count on a boundary edge is a layout that
            quietly lost its curvature.

        Raises
        ------
        ValueError
            If this decomposition was not built by ``from_boundary``, so
            there are no walls to derive the curves from.
        """
        from compas_singular.datastructures import coarse_edges_to_curves
        from compas_singular.datastructures import snap_corners_to_walls

        if coarse is None:
            coarse = self.mesh
        if coarse is None:
            raise ValueError("no coarse layout -- call coarse_mesh() first")
        if not self.inputs.get("outer_boundary"):
            raise ValueError(
                "edges_to_curves needs the domain walls, which only "
                "from_boundary records. Call "
                "compas_singular.datastructures.coarse_edges_to_curves directly "
                "with the loops you have."
            )

        if wall_sampling is None:
            target = self.inputs.get("target_length")
            wall_sampling = target * 0.25 if target else None
        # ``alpha``/``d_min`` off: this is output smoothing at an explicitly
        # given sampling, not the thesis discretisation of the input.
        outer_wall, inner_walls = discretise_boundary(self.inputs["outer_boundary"], self.inputs.get("inner_boundaries") or [], spacing=wall_sampling, alpha=None, d_min=None)
        walls = [outer_wall] + inner_walls

        if snap:
            snap_corners_to_walls(coarse, loops=walls)
        return coarse_edges_to_curves(coarse, loops=walls, polylines=self.polylines or [])

    # --------------------------------------------------------------------------
    # corrections
    # --------------------------------------------------------------------------

    def branches_splitting_collapsed_boundaries(self) -> list[list[list[float]]]:
        """Add branches to split boundaries with fewer than three splits (thesis S4.2.3.4).

        Runs last, because the other corrections can already solve it.

        Returns
        -------
        new_branches : list
            List of polylines as list of point XYZ-coordinates.
        """
        new_branches = []

        all_splits = set(list(self.corner_vertices()) + list(self.split_vertices()))

        for polyedge in [bdry + bdry[:1] for bdry in self.boundaries()]:
            # In polyedge ORDER, and unique. It was a set, which is neither:
            # the ``len(splits) == 1`` branch below indexes it, and ``set[0]``
            # raises ``TypeError: 'set' object is not subscriptable`` -- hit on a
            # square with two symmetric curve features, 10 of 48 bulge/spacing
            # combinations. A set would also make the two-split branch's output
            # depend on iteration order.
            splits = []
            for vkey in polyedge:
                if vkey in all_splits and vkey not in splits:
                    splits.append(vkey)
            new_splits = []

            if len(splits) == 0:
                new_splits += [vkey for vkey in list(itemgetter(0, int(floor(len(polyedge) / 3)), int(floor(len(polyedge) * 2 / 3)))(polyedge))]

            elif len(splits) == 1:
                i = polyedge.index(splits[0])
                new_splits += list(itemgetter(i - int(floor(len(polyedge) * 2 / 3)), i - int(floor(len(polyedge) / 3)))(polyedge))

            elif len(splits) == 2:
                one, two = list_split(polyedge, [polyedge.index(vkey) for vkey in splits])
                half = one if len(one) > len(two) else two
                new_splits.append(half[int(floor(len(half) / 2))])

            for vkey in new_splits:
                fkey = list(self.vertex_faces(vkey))[0]
                for edge in self.face_halfedges(fkey):
                    if vkey in edge and not self.is_edge_on_boundary(*edge):
                        new_branches += [[trimesh_face_circle(self, fkey)[0], self.vertex_coordinates(vkey_2)] for vkey_2 in edge]
                        all_splits.update(edge)
                        break

        return new_branches

    def branches_splitting_flipped_faces(self) -> list[list[list[float]]]:
        """Add branches to split patches that would form flipped faces (thesis S4.2.3.3).

        Returns
        -------
        new_branches : list
            List of polylines as list of point XYZ-coordinates.
        """
        new_branches = []
        centre_to_fkey = {TOL.geometric_key(trimesh_face_circle(self, fkey)[0]): fkey for fkey in self.faces()}

        # compute total rotation of polyline
        for polyline in self.branches_singularity_to_singularity():
            angles = [angle_vectors_signed(subtract_vectors(v, u), subtract_vectors(w, v), [0.0, 0.0, 1.0]) for u, v, w in window(polyline, n=3)]
            # subdivide once per angle limit in rotation
            if abs(sum(angles)) > self.flip_angle_limit:
                # the step between subdivision points in polylines (+ 2 for the extremities, which will be discarded)
                alone = len(self.singular_faces()) == 0
                n = floor(abs(sum(angles)) / self.flip_angle_limit) + 1
                step = int(floor(len(polyline) / n))
                # add new branches from corresponding face in Delaunay mesh
                seams = polyline[::step]
                if polyline[-1] != seams[-1]:
                    if len(seams) == n + 1:
                        del seams[-1]
                    seams.append(polyline[-1])
                if alone:
                    seams = seams[0:-1]
                else:
                    seams = seams[1:-1]
                for point in seams:
                    fkey = centre_to_fkey[TOL.geometric_key(point)]
                    for edge in self.face_halfedges(fkey):
                        if not self.is_edge_on_boundary(*edge):
                            new_branches += [[trimesh_face_circle(self, fkey)[0], self.vertex_coordinates(vkey)] for vkey in edge]
                            break

        return new_branches

    def boundary_interior_angle(self, vkey: int) -> float:
        """The interior angle of the domain at a boundary vertex, in radians, from incident face angles.

        A convex corner reads below pi, a reentrant one above, a free curve-feature tip 2 pi.
        """
        total = 0.0
        for fkey in self.vertex_faces(vkey):
            face_vertices = self.face_vertices(fkey)
            i = face_vertices.index(vkey)
            total += angle_points(
                self.vertex_coordinates(vkey), self.vertex_coordinates(face_vertices[i - 1]), self.vertex_coordinates(face_vertices[(i + 1) % len(face_vertices)])
            )
        return total

    def branches_splitting_boundary_kinks(self) -> list[list[list[float]]]:
        """Add branches at concave boundary kinks the skeleton missed (thesis S4.2.3.1).

        Returns
        -------
        new_branches : list
            List of polylines as list of point XYZ-coordinates.
        """
        new_branches = []

        singular_faces = set(self.singular_faces())
        for boundary in self.boundaries():
            angles = {
                (u, v, w): angle_vectors(
                    subtract_vectors(self.vertex_coordinates(v), self.vertex_coordinates(u)), subtract_vectors(self.vertex_coordinates(w), self.vertex_coordinates(v))
                )
                for u, v, w in window(boundary + boundary[:2], n=3)
            }
            for u, v, w, x, y in list(window(boundary + boundary[:4], n=5)):
                # check if not a corner
                if self.vertex_degree(w) == 2:
                    continue

                # Thesis S4.2.3.1: "The skeleton marks convex but not concave
                # kinks". Only a CONCAVITY needs a branch adding; a convex kink
                # is already carried by the skeleton, and correcting one puts a
                # branch point a fraction of the discretisation off the corner.
                #
                # The original test is unsigned, so it cannot tell the two apart.
                # It shows up as soon as a curve feature lands on a wall: the
                # corner stops being two-valent, slips past the guard above, and
                # is corrected as though it were a concavity. Measured on
                # Fig 4.17, a diagonal corner to corner -- both of the square's
                # own corners, interior angle 90 deg, were being corrected.
                if self.boundary_interior_angle(w) <= pi:
                    continue

                angle = angles[(v, w, x)]
                adjacent_angles = (angles[(u, v, w)] + angles[(w, x, y)]) / 2

                if angle - adjacent_angles > self.relative_kink_angle_limit:
                    # check if not already marked via an adjacent singular face
                    if all([fkey not in singular_faces for fkey in self.vertex_faces(w)]):
                        fkeys = list(self.vertex_faces(w, ordered=True))
                        fkey = fkeys[int(floor(len(fkeys) / 2))]
                        for edge in self.face_halfedges(fkey):
                            if w in edge and not self.is_edge_on_boundary(*edge):
                                new_branches += [[trimesh_face_circle(self, fkey)[0], self.vertex_coordinates(vkey)] for vkey in edge]
                                break

        return new_branches

    def solve_triangular_faces(self) -> None:
        """Make the decomposition mesh all-quad by fixing triangles from coinciding singular faces (thesis S4.2.3.2).

        Deviation: does vertex surgery on the built mesh instead of inserting the missing branch.
        """
        mesh = self.mesh

        for fkey in list(mesh.faces()):
            if len(mesh.face_vertices(fkey)) == 3:
                boundary_vertices = [vkey for vkey in mesh.face_vertices(fkey) if mesh.is_vertex_on_boundary(vkey)]
                case = sum(mesh.is_vertex_on_boundary(vkey) for vkey in mesh.face_vertices(fkey))

                if case == 1:
                    # convert triangular face to quad by duplicating the boundary vertex
                    # due to singular face vertices at the same location
                    u = boundary_vertices[0]
                    v = mesh.add_vertex(attr_dict={attr: xyz for attr, xyz in zip(["x", "y", "z"], mesh.vertex_coordinates(u))})

                    # modify adjacent faces
                    vertex_faces = mesh.vertex_faces(u, ordered=True)
                    mesh_substitute_vertex_in_faces(mesh, u, v, vertex_faces[: vertex_faces.index(fkey)])

                    # modify triangular face
                    mesh_insert_vertex_on_edge(mesh, (u, mesh.face_vertex_ancestor(fkey, u)), v)

                elif case == 2:
                    # remove triangular face and merge the two boundary vertices
                    # due to singularities at the same location
                    #
                    # Thesis S4.2.3.2 does the opposite here -- "If two of the
                    # three patch corners are on the boundary, the branch is
                    # inserted at the OTHER corner" -- i.e. split the interior
                    # corner, the mirror of case 1. That was implemented and
                    # measured: across a square, an L-plate, a hole, a point
                    # feature and all three curve-feature benchmarks, **case 2
                    # never fires once** (case 0 fires 12 times, case 1 four
                    # times), so the two rules cannot be told apart. Left as
                    # Robin wrote it rather than shipping an unexercised branch.
                    #
                    # The gap that matters is case 0 -- all three corners
                    # interior -- which neither this code nor S4.2.3.2 covers,
                    # and which is what a free curve extremity produces.
                    # The merge is the point of this case; the decomposition
                    # polyline is only consulted for WHERE to put the merged
                    # vertex. There is not always one joining the pair -- a real
                    # plate raised ``TypeError: 'NoneType' object is not
                    # iterable`` here -- and the two are coincident anyway, so
                    # their centroid is the honest fallback.
                    branch = self.decomposition_polyline(*map(lambda x: TOL.geometric_key(mesh.vertex_coordinates(x)), boundary_vertices))
                    if branch is not None:
                        at = Polyline(branch).point_at(t=0.5, snap=True)
                        point = [at.x, at.y, at.z]
                    else:
                        point = centroid_points([mesh.vertex_coordinates(vkey) for vkey in boundary_vertices])
                    new_vkey = mesh.add_vertex(attr_dict={"x": point[0], "y": point[1], "z": point[2]})

                    # modify triangular face
                    mesh.delete_face(fkey)

                    # modify adjacent faces
                    for old_vkey in boundary_vertices:
                        mesh_substitute_vertex_in_faces(mesh, old_vkey, new_vkey, mesh.vertex_faces(old_vkey))
                        mesh.delete_vertex(old_vkey)

        to_move = {}
        # Give some length to the new edge. Case 1 duplicates a vertex, which
        # leaves a zero-length edge, and densification divides by it -- without
        # this the free-tip case raises ZeroDivisionError.
        #
        # Each copy moves toward its OWN neighbour centroid, so the pair opens up
        # and the step scales with the local patch size. The fraction was 0.1,
        # which is too small to produce an edge: measured on a 10-unit plate with
        # a free guide it left a 0.14 edge, and on a real Rhino plate a 0.019 one
        # -- aspect ratio 376, a 178.6 degree corner, and a singularity that
        # reads as a tiny edge rather than a point. At 0.5 the same case opens to
        # 0.70 with the layout unchanged.
        for edge in mesh.edges():
            threshold = 1e-6
            if mesh.edge_length(*edge) < threshold:
                for vkey in edge:
                    xyz = centroid_points([mesh.vertex_coordinates(nbr) for nbr in mesh.vertex_neighbors(vkey)])
                    xyz0 = mesh.vertex_coordinates(vkey)
                    to_move[vkey] = [self.collapsed_edge_opening * (a - a0) for a, a0 in zip(xyz, xyz0)]

        for vkey, xyz in to_move.items():
            attr = mesh.vertex[vkey]
            attr["x"] += xyz[0]
            attr["y"] += xyz[1]
            attr["z"] += xyz[2]

    def quadrangulate_polygonal_faces(self) -> None:
        """Turn polygonal faces left by a curve feature into quads by seam propagation (thesis Fig 4.20d).

        Triangles are left alone: they are pseudo-quads with a pole.
        """
        supermesh = self.mesh

        # Seam propagation only has something to do where a face is bigger than
        # a quad. Returning early matters: mesh_weld below REBUILDS the mesh and
        # renumbers its keys, and the editing layers address faces by key.
        if not any(len(supermesh.face_vertices(fkey)) > 4 for fkey in supermesh.faces()):
            return

        delaunay_vertex_map = tuple(TOL.geometric_key(self.vertex_coordinates(vkey)) for vkey in self.vertices())
        # newly added vertices in mesh that were not in the Delaunay are missing...

        edges_to_unweld = [edge for edge in supermesh.edges() if sum([TOL.geometric_key(supermesh.vertex_coordinates(i)) in delaunay_vertex_map for i in edge]) == 2]
        mesh_unweld_edges(supermesh, edges_to_unweld)

        # A "discrepancy" is a position whose coincident copies, one per side of
        # the cut, do not agree on their boundary valency: a branch landed on one
        # side only. Collect them over every component BEFORE welding -- welding
        # inside this loop is what used to reduce self.mesh to the last piece.
        #
        # The valencies are gathered over ALL components. The two copies of a
        # position usually sit in different components, and a map rebuilt per
        # component only ever holds one of them -- no source was ever found.
        components = list(supermesh.exploded())
        candidate_map = {}
        for mesh in components:
            for boundary in mesh.vertices_on_boundaries():
                for vkey in boundary:
                    candidate_map.setdefault(TOL.geometric_key(mesh.vertex_coordinates(vkey)), []).append(mesh.vertex_degree(vkey))
        source_map = set(geom_key for geom_key, valencies in candidate_map.items() if len(set(valencies)) > 1)

        # A seam vertex is a source only for the face on the side NO branch landed
        # on -- the copy with the lower valency -- and a genuine corner of the
        # faces on the other side. Record which, per face, keyed by the positions
        # of its vertices so it survives the weld. Per COPY rather than per
        # component: around a free feature both sides are one component.
        flat_by_face = {}
        for mesh in components:
            for fkey in mesh.faces():
                face_vertices = mesh.face_vertices(fkey)
                geom_keys = [TOL.geometric_key(mesh.vertex_coordinates(vkey)) for vkey in face_vertices]
                flat = set(geom_key for vkey, geom_key in zip(face_vertices, geom_keys) if geom_key in source_map and mesh.vertex_degree(vkey) < max(candidate_map[geom_key]))
                if flat:
                    flat_by_face[frozenset(geom_keys)] = flat

        self.mesh = mesh_weld(supermesh)
        mesh = self.mesh

        face_sources = {}
        for fkey in mesh.faces():
            face_vertices = mesh.face_vertices(fkey)
            geom_keys = [TOL.geometric_key(mesh.vertex_coordinates(vkey)) for vkey in face_vertices]
            flat = flat_by_face.get(frozenset(geom_keys))
            if flat:
                face_sources[fkey] = [vkey for vkey, geom_key in zip(face_vertices, geom_keys) if geom_key in flat]

        # A seam strip that closes on itself never ends. Give up on it, keep the
        # layout as it was, and leave its polygons to repair_polygonal_faces.
        before = mesh.copy()
        if not quadrangulate_faces(mesh, face_sources, max_faces=3 * mesh.number_of_faces()):
            self.mesh = before
            self.repair_notes.append("seam propagation did not terminate; polygonal faces left to the fallback repair")

    def quadrangulate_polygonal_faces_wip(self) -> None:
        pass
        # mesh = self.mesh

        # delaunay_vertex_map = tuple(TOL.geometric_key(self.vertex_coordinates(vkey)) for vkey in self.vertices())

        # for fkey in mesh.faces():
        # 	face_vertices = mesh.face_vertices(fkey)
        # 	if len(face_vertices) > 4:

    def repair_polygonal_faces(self, poles: Sequence[list[float]] = ()) -> None:
        """Split any face seam propagation could not quadrangulate along its own diagonals.

        Last resort, inert unless a face larger than a quad survived.

        Parameters
        ----------
        poles : list, optional
            Point features, passed on as preferred pole positions.
        """
        mesh = self.mesh
        if not any(len(mesh.face_vertices(fkey)) > 4 for fkey in mesh.faces()):
            return

        try:
            from compas_singular.editing.repair import solve_non_quad_faces
        except Exception as exc:  # the frame-field extras are optional
            self.repair_notes.append("polygonal faces left and the fallback repair is unavailable ({})".format(exc))
            return

        self.mesh, note = solve_non_quad_faces(mesh, cls=type(mesh), poles=list(poles))
        if note:
            self.repair_notes.append("fallback repair: {}".format(note))

    def split_quads_with_poles(self, poles: list[list[float]]) -> list[list[list[float]]]:
        new_lines = []

        mesh = self.mesh
        pole_map = tuple([TOL.geometric_key(pole) for pole in poles])

        faces = list(mesh.faces())
        for fkey in faces:
            fv = mesh.face_vertices(fkey)
            if len(fv) == 4:
                for vkey in fv:
                    if TOL.geometric_key(mesh.vertex_coordinates(vkey)) in pole_map:
                        idx = fv.index(vkey)
                        xkey = fv[idx + 2 - len(fv)]
                        new_lines.append([mesh.vertex_coordinates(vkey), mesh.vertex_coordinates(xkey)])
                        split_quad_in_pseudo_quads(mesh, fkey, vkey)
                        break

        self.polylines += new_lines
        return new_lines

    def store_pole_data(self, poles: list[list[float]]) -> None:
        """Record, for every triangular face, which corner is the pole.

        Triangles without a point feature at a corner get one chosen by ``_choose_pole``.

        Parameters
        ----------
        poles : list
            Point features, as XYZ coordinates. A triangle with one of these at a
            corner collapses there, exactly as before.
        """
        mesh = self.mesh
        pole_map = tuple([TOL.geometric_key(pole) for pole in poles])

        face_poles = {}
        for fkey in mesh.faces():
            if len(mesh.face_vertices(fkey)) == 3:
                for vkey in mesh.face_vertices(fkey):
                    if TOL.geometric_key(mesh.vertex_coordinates(vkey)) in pole_map:
                        face_poles[fkey] = vkey
                        break
                if fkey not in face_poles:
                    face_poles[fkey] = self._choose_pole(fkey)
                    self.repair_notes.append("face {} is a triangle with no point feature at a corner; pole set to vertex {}".format(fkey, face_poles[fkey]))

        mesh.attributes["face_pole"] = face_poles

    def _choose_pole(self, fkey: int) -> int:
        """Choose the corner of a triangle to collapse at: the one whose two edges are closest in length."""
        mesh = self.mesh
        fv = mesh.face_vertices(fkey)
        best = None
        for i in range(3):
            p, a, b = fv[i], fv[(i + 1) % 3], fv[(i + 2) % 3]
            ra = distance_point_point(mesh.vertex_coordinates(p), mesh.vertex_coordinates(a))
            rb = distance_point_point(mesh.vertex_coordinates(b), mesh.vertex_coordinates(p))
            if ra + rb <= 0.0:
                continue
            imbalance = abs(ra - rb) / (ra + rb)
            if best is None or imbalance < best[0]:
                best = (imbalance, p)
        return best[1] if best else fv[0]


# ==============================================================================
# Main
# ==============================================================================


if __name__ == "__main__":
    pass
