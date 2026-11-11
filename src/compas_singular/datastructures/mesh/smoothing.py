"""Smoothing and relaxation of a mesh, in place, with vertices held on points, curves or surfaces.

Default is ``boundary_constrained_smoothing``; prefer ``'area'`` on a graded mesh.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from math import pi
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence

from compas.datastructures import mesh_smooth_centerofmass
from compas.geometry import Curve
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import angle_vectors
from compas.geometry import closest_point_on_polyline
from compas.geometry import closest_point_on_segment
from compas.geometry import distance_point_point
from compas.geometry import length_vector
from compas.geometry import subtract_vectors
from compas.itertools import flatten
from compas.itertools import pairwise

if TYPE_CHECKING:
    from compas.datastructures import Mesh


__all__ = [
    "closest_point_on_constraint",
    "mesh_boundary_loops",
    "mesh_boundary_polylines",
    "mesh_boundary_corners",
    "constrained_smoothing",
    "automated_boundary_constraints",
    "boundary_constrained_smoothing",
    "boundary_smoothing",
    "region_smoothing",
    "relaxation",
]


#: What a vertex can be constrained to: a point, a polyline, a sequence of points, a curve
#: or a surface. See ``closest_point_on_constraint``.
Constraint = Any

#: The smoothing algorithms accepted by the ``algorithm`` parameters.
ALGORITHMS = ("area", "centroid", "centerofmass")


# ==============================================================================
# Closest point
# ==============================================================================


def _is_xyz(item: Any) -> bool:
    """Return whether an item is a bare sequence of three numbers."""
    return isinstance(item, (list, tuple)) and len(item) == 3 and all(isinstance(i, (int, float)) for i in item)


def closest_point_on_constraint(constraint: Constraint, xyz: list[float], discretisation: int = 128) -> list[float] | None:
    """Project a point onto a constraint object, using COMPAS geometry only.

    Parameters
    ----------
    constraint : None | compas.geometry.Point | [float, float, float] | compas.geometry.Polyline | sequence[point] | compas.geometry.Curve | Any
        The geometry to project onto:

        * ``None`` -- no constraint, None is returned;
        * a ``compas.geometry.Point`` or a bare ``[x, y, z]`` -- the point itself, i.e. a pin;
        * a ``compas.geometry.Polyline`` or any sequence of points -- the closest point
          on that polyline, via ``compas.geometry.closest_point_on_polyline``;
        * any other ``compas.geometry.Curve`` -- a ``compas.geometry.Line``, a
          ``compas.geometry.Circle``, a curve from ``compas_rhino``'s
          ``curve_to_compas`` or from ``compas_occ`` -- the closest point *on the curve*, from
          its own ``closest_point``. See ``_closest_point_on_curve`` for the curves that
          have none and fall back on their ``to_polyline`` discretisation;
        * any other object with a working ``closest_point`` method (a
          ``compas_singular.editing.GuideCurve``, a Rhino or OCC surface, ...) -- the
          result of that method;
        * anything else with a ``to_polyline`` -- the closest point on that discretisation.
    xyz : [float, float, float]
        The coordinates of the point to project.
    discretisation : int, optional
        Number of segments used to discretise a curve that has no analytical closest point.
        Default is ``128``.

    Returns
    -------
    list[float] | None
        The XYZ coordinates of the closest point on the constraint,
        or None if the constraint is None.

    Raises
    ------
    TypeError
        If the closest point on the constraint cannot be computed.

    """
    if constraint is None:
        return None

    # a point constraint pins the vertex
    if isinstance(constraint, Point) or _is_xyz(constraint):
        return [float(constraint[0]), float(constraint[1]), float(constraint[2])]

    # a polyline, or any bare sequence of points read as one. BEFORE the curve branch: a
    # Polyline is a Curve too, and its own closest_point raises NotImplementedError
    if isinstance(constraint, Polyline):
        return closest_point_on_polyline(xyz, constraint)
    if isinstance(constraint, (list, tuple)):
        return closest_point_on_polyline(xyz, [list(point) for point in constraint])

    # a parametric curve: onto the curve itself, not onto a sampling of it
    if isinstance(constraint, Curve):
        return _closest_point_on_curve(constraint, xyz, discretisation)

    # anything else that knows how to project a point: a GuideCurve, Rhino or OCC surfaces, ...
    method = getattr(constraint, "closest_point", None)
    if callable(method):
        try:
            return list(method(Point(*xyz)))[:3]
        except NotImplementedError:
            pass

    # curves without an analytical projection: discretise and project on the polyline
    method = getattr(constraint, "to_polyline", None)
    if callable(method):
        return closest_point_on_polyline(xyz, method(n=discretisation))

    raise TypeError(f"Cannot compute the closest point on a constraint of type {type(constraint)}.")


#: Curve types whose missing ``closest_point`` has already been warned about by
#: ``_closest_point_on_curve``. Once per type, not per call: smoothing projects every
#: constrained vertex at every iteration, and a warning per call would put thousands of
#: identical lines on Rhino's command line.
_EXPLAINED_FALLBACKS = set()


def _closest_point_on_curve(curve: Curve, xyz: list[float], discretisation: int) -> list[float]:
    """The closest point on a compas curve, falling back on its ``to_polyline`` where the curve cannot say."""
    try:
        closest = curve.closest_point(point=Point(*xyz))
    except NotImplementedError:
        if type(curve) not in _EXPLAINED_FALLBACKS:
            _EXPLAINED_FALLBACKS.add(type(curve))
            warnings.warn(
                f"{type(curve).__name__} does not implement closest_point, so points are projected "
                f"onto a {discretisation}-segment polyline of it instead -- up to a sagitta off the "
                "curve itself. Shown once per curve type.",
                stacklevel=2,
            )
        closest = None
    if closest is None:
        return closest_point_on_polyline(xyz, curve.to_polyline(n=discretisation))
    return [float(closest[0]), float(closest[1]), float(closest[2])]


def _is_polyline_like(constraint: Constraint) -> bool:
    """Return whether a constraint is a polyline, or a sequence of points read as one."""
    if isinstance(constraint, Polyline):
        return True
    return isinstance(constraint, (list, tuple)) and not _is_xyz(constraint)


class _PolylineProjector:
    """Project points onto a polyline, searching a window of segments around the previous result first."""

    def __init__(self, polyline: Polyline | Sequence[Sequence[float]], window: int = 10) -> None:
        points = [list(point) for point in polyline]
        if len(points) < 2:
            raise ValueError("A polyline constraint needs at least two points.")
        self.segments = list(pairwise(points))
        self.closed = distance_point_point(points[0], points[-1]) < 1e-9
        self.window = window

    def _search(self, xyz: list[float], indices: Iterable[int]) -> tuple[list[float] | None, int | None]:
        closest, minimum, index = None, None, None
        for i in indices:
            point = closest_point_on_segment(xyz, self.segments[i])
            distance = distance_point_point(xyz, point)
            if minimum is None or distance < minimum:
                closest, minimum, index = point, distance, i
        return closest, index

    def closest_point(self, xyz: list[float], hint: int | None = None) -> tuple[list[float], int]:
        """Return the closest point on the polyline and the index of its segment."""
        count = len(self.segments)

        if hint is None or 2 * self.window + 1 >= count:
            return self._search(xyz, range(count))

        if self.closed:
            offsets = range(-self.window, self.window + 1)
            indices = [(hint + offset) % count for offset in offsets]
        else:
            indices = list(range(max(0, hint - self.window), min(count, hint + self.window + 1)))

        closest, index = self._search(xyz, indices)

        # the window was too small to contain the closest segment: search all of it
        if index == indices[0] or index == indices[-1]:
            return self._search(xyz, range(count))

        return closest, index


# ==============================================================================
# Boundary
# ==============================================================================


def mesh_boundary_loops(mesh: Mesh) -> list[list[int]]:
    """Collect the boundaries of a mesh as ordered, non-repeating loops of vertices.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        A mesh.

    Returns
    -------
    list[list[int]]
        A list of boundaries as ordered lists of vertex keys.
        The first vertex is not repeated at the end, the loops are implicitly closed.

    """
    loops = []
    for boundary in mesh.vertices_on_boundaries():
        loop = list(boundary)
        while len(loop) > 1 and loop[-1] == loop[0]:
            loop.pop()
        if len(loop) > 2:
            loops.append(loop)
    return loops


def _loop_polyline(mesh: Mesh, loop: list[int]) -> Polyline:
    """The closed polyline through a loop of vertices."""
    points = [mesh.vertex_coordinates(vertex) for vertex in loop]
    return Polyline(points + points[:1])


def mesh_boundary_polylines(mesh: Mesh) -> list[Polyline]:
    """Collect the boundaries of a mesh as closed polylines.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        A mesh.

    Returns
    -------
    list[compas.geometry.Polyline]
        A closed polyline per mesh boundary.

    """
    return [_loop_polyline(mesh, loop) for loop in mesh_boundary_loops(mesh)]


def mesh_boundary_corners(mesh: Mesh, corner_angle: float = pi / 6) -> list[int]:
    """Collect the boundary vertices of a mesh that sit on a kink.

    The kink is measured as the deviation from straight between the two boundary
    edges at the vertex: zero along a straight boundary, pi / 2 at a square corner.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        A mesh.
    corner_angle : float, optional
        Threshold deviation angle, in radians. Default is ``pi / 6`` (30 degrees).

    Returns
    -------
    list[int]
        The keys of the boundary vertices at a kink.

    """
    corners = []
    for loop in mesh_boundary_loops(mesh):
        count = len(loop)
        for index, vertex in enumerate(loop):
            a = mesh.vertex_coordinates(loop[index - 1])
            b = mesh.vertex_coordinates(vertex)
            c = mesh.vertex_coordinates(loop[(index + 1) % count])
            u = subtract_vectors(b, a)
            v = subtract_vectors(c, b)
            if length_vector(u) == 0 or length_vector(v) == 0:
                continue
            if angle_vectors(u, v) > corner_angle:
                corners.append(vertex)
    return corners


def split_loop_at_corners(loop: list[int], corners: Sequence[int]) -> list[list[int]]:
    """Split a closed loop of vertices into the segments delimited by its corner vertices.

    The corner vertices are shared by the two segments they delimit. A loop with fewer
    than two corners is returned whole, as a single segment.
    """
    corners = set(corners)
    indices = [index for index, vertex in enumerate(loop) if vertex in corners]
    if len(indices) < 2:
        return [loop]

    count = len(loop)
    segments = []
    for i, start in enumerate(indices):
        end = indices[(i + 1) % len(indices)]
        segment = [loop[start]]
        index = start
        while index != end:
            index = (index + 1) % count
            segment.append(loop[index])
        segments.append(segment)
    return segments


def closest_curve(mesh: Mesh, vertices: list[int], curves: Sequence[Constraint]) -> Constraint:
    """Return the curve that is closest, on average, to a run of mesh vertices.

    The end vertices of the run are left out of the average when there are others, since
    they are corners shared with the neighbouring runs.
    """
    sample = vertices[1:-1] or vertices
    points = [mesh.vertex_coordinates(vertex) for vertex in sample]

    closest, minimum = None, None
    for curve in curves:
        distance = sum(distance_point_point(point, closest_point_on_constraint(curve, point)) for point in points) / len(points)
        if minimum is None or distance < minimum:
            closest, minimum = curve, distance
    return closest


def automated_boundary_constraints(mesh: Mesh, curves: Sequence[Constraint] | None = None, corner_angle: float = pi / 6, fix_corners: bool = True) -> dict[int, Constraint]:
    """Constrain every boundary vertex to the boundary curve it belongs to, pinning kinks.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        The mesh to constrain.
    curves : sequence, optional
        The boundary geometry to snap the boundary vertices to, as COMPAS polylines or
        curves. Each run of boundary vertices between two corners is matched to the curve
        it is closest to. Default is None, in which case the mesh boundary polylines are used.
    corner_angle : float, optional
        Threshold deviation angle for a boundary kink, in radians.
        Default is ``pi / 6`` (30 degrees).
    fix_corners : bool, optional
        If True, pin the boundary vertices at a kink to their current position.
        Default is True.

    Returns
    -------
    dict[int, Constraint]
        A dictionary of mesh constraints for smoothing, as vertex keys pointing to
        point, polyline or curve objects.
    """
    constraints = {}
    corners = set(mesh_boundary_corners(mesh, corner_angle)) if fix_corners else set()

    for loop in mesh_boundary_loops(mesh):
        if not curves:
            polyline = _loop_polyline(mesh, loop)
            for vertex in loop:
                constraints[vertex] = polyline
        else:
            for segment in split_loop_at_corners(loop, corners):
                curve = closest_curve(mesh, segment, curves)
                for vertex in segment:
                    constraints[vertex] = curve

    for vertex in corners:
        constraints[vertex] = Point(*mesh.vertex_coordinates(vertex))

    return constraints


# ==============================================================================
# Smoothing
# ==============================================================================


def _smoothers(algorithm: str | Sequence[str]) -> list[Any]:
    """The smoothing functions for one algorithm name or a sequence of them, in order.

    Raises
    ------
    ValueError
        If no algorithm is given, or one of them is not recognised.
    """
    names = [algorithm] if isinstance(algorithm, str) else list(algorithm)
    if not names:
        raise ValueError(f"No smoothing algorithm given. Pick one or more of {', '.join(ALGORITHMS)}.")
    unknown = [name for name in names if name not in ALGORITHMS]
    if unknown:
        raise ValueError(f"{', '.join(map(repr, unknown))} is not a recognised smoothing algorithm. Pick one or more of {', '.join(ALGORITHMS)}.")

    smoothers = {
        "area": lambda mesh, **kwargs: mesh.smooth_area(**kwargs),
        "centroid": lambda mesh, **kwargs: mesh.smooth_centroid(**kwargs),
        "centerofmass": mesh_smooth_centerofmass,
    }
    return [smoothers[name] for name in names]


def constrained_smoothing(
    mesh: Mesh,
    kmax: int = 100,
    damping: float = 0.5,
    constraints: dict[int, Constraint] | None = None,
    algorithm: str | Sequence[str] = "centroid",
    fixed: Sequence[int] | None = None,
    symmetric: bool | None = None,
) -> None:
    """Smooth a mesh in place, moving each constrained vertex back onto its constraint after every iteration.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        A mesh to smooth, modified in place.
    kmax : int, optional
        Number of iterations for smoothing, per algorithm. Default is ``100``.
    damping : float, optional
        Damping value for smoothing between 0 and 1. Default is ``0.5``.
    constraints : dict[int, Constraint], optional
        Dictionary of constraints as vertex keys pointing to COMPAS geometry
        (points, polylines, curves or surfaces). Empty by default.
        See ``closest_point_on_constraint`` for the accepted geometry.
    algorithm : {'centroid', 'area', 'centerofmass'} | sequence of them, optional
        Smoothing algorithm to apply. Classic centroid by default. A sequence of
        algorithms, such as ``['area', 'centroid']``, runs each in turn for ``kmax``
        iterations, in the order given.
    fixed : sequence[int], optional
        Vertices that the smoothing algorithm must not move at all. Default is None.
    symmetric : bool, optional
        Keep a mesh made by ``expand_symmetrically`` exactly symmetric by averaging
        each vertex over its ``attributes['orbits']`` every iteration. ``None``
        (default) means: when the mesh has orbits.

    Returns
    -------
    None
        The mesh is modified in place.

    Raises
    ------
    ValueError
        If an algorithm is not recognised.
    """
    smoothers = _smoothers(algorithm)
    constraints = constraints or {}

    if symmetric is None:
        symmetric = bool(mesh.attributes.get("orbits"))
    orbit_maps = None
    if symmetric:
        from compas_singular.symmetry.replicate import orbit_maps as mesh_orbit_maps
        from compas_singular.symmetry.replicate import symmetrise_positions

        orbit_maps = mesh_orbit_maps(mesh)

    # polylines get a projector that only searches around the previous result
    projectors = {}
    for constraint in constraints.values():
        if _is_polyline_like(constraint) and id(constraint) not in projectors:
            projectors[id(constraint)] = _PolylineProjector(constraint)
    hints = {}

    def project(vertex: int, constraint: Constraint, xyz: list[float]) -> list[float] | None:
        projector = projectors.get(id(constraint))
        if projector is None:
            return closest_point_on_constraint(constraint, xyz)
        point, hints[vertex] = projector.closest_point(xyz, hints.get(vertex))
        return point

    def callback(k: int, args: tuple[Mesh, dict[int, Constraint]]) -> None:
        mesh, constraints = args
        if orbit_maps is not None:
            symmetrise_positions(mesh, orbit_maps)
        for vertex, constraint in constraints.items():
            if constraint is None:
                continue
            xyz = project(vertex, constraint, mesh.vertex_coordinates(vertex))
            if xyz is None:
                continue
            mesh.vertex_attributes(vertex, "xyz", xyz)

    for smooth in smoothers:
        smooth(mesh, fixed=fixed, kmax=kmax, damping=damping, callback=callback, callback_args=[mesh, constraints])


def boundary_constrained_smoothing(
    mesh: Mesh,
    curves: Sequence[Constraint] | None = None,
    kmax: int = 100,
    damping: float = 0.5,
    algorithm: str | Sequence[str] = "centroid",
    corner_angle: float = pi / 6,
    fix_corners: bool = True,
    constraints: dict[int, Constraint] | None = None,
) -> dict[int, Constraint]:
    """Smooth a mesh with its boundary vertices sliding along the boundary and corners fixed.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        A mesh to smooth, modified in place.
    curves : sequence, optional
        The boundary geometry to snap the boundary vertices to, as COMPAS polylines or
        curves. Default is None, in which case the mesh boundary polylines are used.
    kmax : int, optional
        Number of iterations for smoothing, per algorithm. Default is ``100``.
    damping : float, optional
        Damping value for smoothing between 0 and 1. Default is ``0.5``.
    algorithm : {'centroid', 'area', 'centerofmass'} | sequence of them, optional
        Smoothing algorithm to apply, or several to run in turn.
        Classic centroid by default. See ``constrained_smoothing``.
    corner_angle : float, optional
        Threshold deviation angle for a boundary kink, in radians.
        Default is ``pi / 6`` (30 degrees).
    fix_corners : bool, optional
        If True, pin the boundary vertices at a kink to their current position.
        Default is True.
    constraints : dict[int, Constraint], optional
        Additional constraints, as vertex keys pointing to COMPAS geometry, applied on
        top of the automated ones. Default is None.

    Returns
    -------
    dict[int, Constraint]
        The constraints that were applied, as vertex keys pointing to point, polyline or
        curve objects. The mesh itself is modified in place.
    """
    boundary_constraints = automated_boundary_constraints(mesh, curves=curves, corner_angle=corner_angle, fix_corners=fix_corners)

    if constraints:
        boundary_constraints.update(constraints)

    constrained_smoothing(mesh, kmax=kmax, damping=damping, constraints=boundary_constraints, algorithm=algorithm)

    return boundary_constraints


def boundary_smoothing(
    mesh: Mesh,
    curves: Sequence[Constraint] | None = None,
    kmax: int = 100,
    damping: float = 0.5,
    algorithm: str | Sequence[str] = "area",
    corner_angle: float = pi / 6,
    fix_corners: bool = True,
) -> dict[int, Constraint]:
    """Smooth only the boundary of a mesh, sliding its vertices along the boundary.

    Unlike ``boundary_constrained_smoothing``, every interior vertex is fixed: the
    boundary vertices are redistributed along the boundary and nothing else moves.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        A mesh to smooth, modified in place.
    curves : sequence, optional
        The boundary geometry to slide the boundary vertices along, as COMPAS polylines or
        curves. Default is None, in which case the mesh boundary polylines are used.
    kmax : int, optional
        Number of iterations for smoothing, per algorithm. Default is ``100``.
    damping : float, optional
        Damping value for smoothing between 0 and 1. Default is ``0.5``.
    algorithm : {'area', 'centroid', 'centerofmass'} | sequence of them, optional
        Smoothing algorithm to apply, or several to run in turn.
        Area by default. See ``constrained_smoothing``.
    corner_angle : float, optional
        Threshold deviation angle for a boundary kink, in radians.
        Default is ``pi / 6`` (30 degrees).
    fix_corners : bool, optional
        If True, pin the boundary vertices at a kink to their current position.
        Default is True.

    Returns
    -------
    dict[int, Constraint]
        The constraints that were applied to the boundary vertices.
        The mesh itself is modified in place.

    """
    boundary = set(flatten(mesh.vertices_on_boundaries()))
    fixed = [vertex for vertex in mesh.vertices() if vertex not in boundary]

    constraints = automated_boundary_constraints(mesh, curves=curves, corner_angle=corner_angle, fix_corners=fix_corners)

    constrained_smoothing(mesh, kmax=kmax, damping=damping, constraints=constraints, algorithm=algorithm, fixed=fixed)

    return constraints


def region_smoothing(
    mesh: Mesh, vertices: Sequence[int] | dict[int, float], kmax: int = 50, damping: float = 0.5, blend: int = 3, constraints: dict[int, Constraint] | None = None
) -> dict[int, float]:
    """Area-weighted smoothing of one region, with damping tapered to zero over ``blend`` rings.

    Everything outside the region is fixed; the taper avoids a crease at its edge.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        A mesh to smooth, modified in place.
    vertices : sequence[int] | dict[int, float]
        The region to smooth, either as the vertex keys of its core -- which are then
        tapered over ``blend`` rings -- or as a ready-made dictionary of vertex keys
        pointing to damping weights between 0 and 1, in which case ``blend`` is ignored.
        A vertex that is absent, or whose weight is not positive, is fixed.
    kmax : int, optional
        Number of iterations for smoothing. Default is ``50``, fewer than the whole-mesh
        functions, since a region is small and needs fewer to settle.
    damping : float, optional
        Damping value for smoothing between 0 and 1, scaled per vertex by its weight.
        Default is ``0.5``.
    blend : int, optional
        Rings of vertices outside the core over which the damping falls to zero.
        Default is ``3``. Ignored if ``vertices`` is a dictionary of weights.
    constraints : dict[int, Constraint], optional
        Dictionary of constraints as vertex keys pointing to COMPAS geometry, applied to
        the vertices of the region after every iteration. Constraints on vertices outside
        the region are ignored, since those vertices do not move. Default is None, in
        which case ``automated_boundary_constraints`` is used: a region touching the
        boundary then slides along it instead of being dragged inward.
        See ``closest_point_on_constraint`` for the accepted geometry.

    Returns
    -------
    dict[int, float]
        The damping weights that were applied, as vertex keys pointing to a weight
        between 0 and 1. The mesh itself is modified in place.
    """

    def taper(core: set[int], rings: int) -> dict[int, float]:
        """Per-vertex damping weight: 1 in the core, falling to 0 outside the blend.

        A vertex with no weight is fixed. The ring just beyond the blend is therefore
        held and the ones just inside it barely move, so there is no step in the result.
        """
        weights = {vertex: 1.0 for vertex in core}
        frontier = set(core)
        for ring in range(1, rings + 1):
            frontier = {n for vertex in frontier for n in mesh.vertex_neighbors(vertex)} - set(weights)
            if not frontier:
                break
            weight = 1.0 - ring / float(rings + 1)
            for vertex in frontier:
                weights[vertex] = weight
        return weights

    def area_target(vertex: int) -> list[float] | None:
        """The area-weighted centroid of the faces around a vertex, or None."""
        areas, centroids = [], []
        for face in mesh.vertex_faces(vertex, ordered=True):
            if face is None:
                continue
            areas.append(mesh.face_area(face))
            centroids.append(mesh.face_centroid(face))
        total = sum(areas)
        if not areas or total <= 0.0:
            return None
        return [sum(a * c[i] for a, c in zip(areas, centroids)) / total for i in range(3)]

    weights = dict(vertices) if isinstance(vertices, dict) else taper(set(vertices), blend)

    if constraints is None:
        constraints = automated_boundary_constraints(mesh)
    constraints = {vertex: constraint for vertex, constraint in constraints.items() if weights.get(vertex, 0.0) > 0.0}

    for _ in range(kmax):
        # every target is collected before any vertex moves, so the result does not
        # depend on the order the region is walked in
        targets = {}
        for vertex, weight in weights.items():
            if weight <= 0.0:
                continue
            target = area_target(vertex)
            if target is not None:
                targets[vertex] = target

        for vertex, target in targets.items():
            xyz = mesh.vertex_coordinates(vertex)
            step = damping * weights[vertex]
            mesh.vertex_attributes(vertex, "xyz", [xyz[i] + step * (target[i] - xyz[i]) for i in range(3)])

        # a region touching the outline must slide along it, not be dragged inward
        for vertex, constraint in constraints.items():
            moved = closest_point_on_constraint(constraint, mesh.vertex_coordinates(vertex))
            if moved is not None:
                mesh.vertex_attributes(vertex, "xyz", moved)

    return weights


# ==============================================================================
# Relaxation
# ==============================================================================


@lru_cache(maxsize=None)
def _fd_polyline_constraint_class() -> type:
    """A ``compas_fd`` constraint that holds a vertex on a polyline, built on first use."""
    from compas_fd.constraints import Constraint as FDConstraint

    from compas.geometry import Vector
    from compas.geometry import vector_component

    class PolylineConstraint(FDConstraint):
        """Constraint for limiting the movement of a vertex to a polyline."""

        def __new__(cls, *args, **kwargs):
            # FDConstraint.__new__ picks the class from the type of the geometry, and has
            # none registered for a polyline
            return object.__new__(cls)

        def __init__(self, polyline: Polyline, name: str | None = None) -> None:
            super().__init__(polyline, name=name)
            self._projector = _PolylineProjector(polyline)
            self._segment = None

        def project(self) -> None:
            point, self._segment = self._projector.closest_point(list(self._location), self._segment)
            self._location = Point(*point)

        def compute_tangent(self) -> None:
            start, end = self._projector.segments[self._segment]
            self._tangent = Vector(*vector_component(self.residual, subtract_vectors(end, start)))

        def compute_normal(self) -> None:
            self._normal = self.residual - self.tangent

        def update(self, damping: float = 0.1) -> None:
            self._location = self.location + self.tangent * damping
            self.project()

    return PolylineConstraint


def _fd_constraint(constraint: Constraint) -> Any:
    """Wrap a constraint geometry in the ``compas_fd`` constraint that holds a vertex on it."""
    from compas_fd.constraints import Constraint as FDConstraint

    if isinstance(constraint, FDConstraint):
        return constraint
    if _is_polyline_like(constraint):
        polyline = constraint if isinstance(constraint, Polyline) else Polyline(constraint)
        return _fd_polyline_constraint_class()(polyline)
    try:
        return FDConstraint(constraint)
    except Exception as exc:
        raise TypeError(
            f"relaxation cannot hold a vertex on a {type(constraint).__name__}. Use a point, a "
            "polyline, a compas Vector, Frame, Line, Plane, Circle, NurbsCurve or NurbsSurface, "
            "or a compas_fd Constraint."
        ) from exc


def relaxation(mesh: Mesh, fixed: str | Sequence[int] = "corners", constraints: dict[int, Constraint] | None = None, q_factor: float = 100.0) -> None:
    """Force-density relaxation of a mesh with ``compas_fd``, boundary edges ``q_factor`` times stiffer.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        A mesh to relax, modified in place.
    fixed : {'corners', 'boundary'} | sequence[int], optional
        The vertices that do not move: ``'corners'`` (default) for the boundary vertices at
        a kink, ``'boundary'`` for every boundary vertex, or the vertex keys themselves.
    constraints : dict[int, Constraint], optional
        Dictionary of constraints as vertex keys pointing to geometry the vertex must stay
        on. A point pins the vertex there, like a fixed vertex. A polyline or a sequence of
        points, a compas Vector, Frame, Line, Plane, Circle, NurbsCurve or NurbsSurface, or
        a ``compas_fd`` Constraint lets it slide along. Constraints on fixed vertices are
        ignored. Default is None.
    q_factor : float, optional
        How much larger the force density of a boundary edge is than that of an interior
        edge. Default is ``100``.

    Returns
    -------
    None
        The mesh is modified in place.

    Raises
    ------
    ValueError
        If ``fixed`` is an unrecognised string.
    TypeError
        If a constraint cannot be turned into a ``compas_fd`` constraint.
    ImportError
        If ``compas_fd`` is not installed.
    """
    from compas_fd.solvers import fd_constrained_numpy

    if fixed == "corners":
        fixed = mesh_boundary_corners(mesh)
    elif fixed == "boundary":
        fixed = list(flatten(mesh_boundary_loops(mesh)))
    elif isinstance(fixed, str):
        raise ValueError(f"{fixed!r} is not a recognised set of fixed vertices. Pick 'corners', 'boundary', or give the vertex keys.")
    fixed = set(fixed)

    # compas_fd works on indices, which match the keys only on a mesh that never lost a vertex
    vertex_index = mesh.vertex_index()
    vertices = [mesh.vertex_coordinates(vertex) for vertex in mesh.vertices()]

    fd_constraints = [None] * len(vertices)
    for vertex, constraint in (constraints or {}).items():
        if constraint is None or vertex in fixed:
            continue
        if isinstance(constraint, Point) or _is_xyz(constraint):
            # a point pins the vertex there
            vertices[vertex_index[vertex]] = [float(constraint[0]), float(constraint[1]), float(constraint[2])]
            fixed.add(vertex)
            continue
        fd_constraints[vertex_index[vertex]] = _fd_constraint(constraint)

    edges = list(mesh.edges())
    forcedensities = [q_factor if mesh.is_edge_on_boundary(edge) else 1.0 for edge in edges]

    result = fd_constrained_numpy(
        vertices=vertices,
        fixed=[vertex_index[vertex] for vertex in fixed],
        edges=[(vertex_index[u], vertex_index[v]) for u, v in edges],
        forcedensities=forcedensities,
        loads=[[0.0, 0.0, 0.0] for _ in vertices],
        constraints=fd_constraints,
    )

    for vertex, index in vertex_index.items():
        mesh.vertex_attributes(vertex, "xyz", [float(x) for x in result.vertices[index]])
