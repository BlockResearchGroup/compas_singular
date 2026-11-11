"""Smoothing a mesh on a surface: interior vertices on the surface, boundary vertices on its borders.

Nothing here imports Rhino; surfaces and curves are anything with ``closest_point``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas.geometry import Point
from compas.geometry import closest_point_in_cloud
from compas.geometry import distance_point_point
from compas_singular.datastructures.mesh.smoothing import closest_curve
from compas_singular.datastructures.mesh.smoothing import closest_point_on_constraint
from compas_singular.datastructures.mesh.smoothing import constrained_smoothing
from compas_singular.datastructures.mesh.smoothing import mesh_boundary_loops
from compas_singular.datastructures.mesh.smoothing import split_loop_at_corners

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh


__all__ = [
    "automated_smoothing_surface_constraints",
    "automated_smoothing_constraints",
    "surface_constrained_smoothing",
]


class _NearestOf(object):
    """Several curves read as ONE constraint: a point goes to whichever is nearest.

    Re-chosen on every projection, so a boundary vertex that slides past the end
    of one border is picked up by the next.
    """

    def __init__(self, curves: Sequence[Any]) -> None:
        self.curves = list(curves)

    def closest_point(self, point: Any) -> list[float] | None:
        xyz = list(point)[:3]
        best, minimum = None, None
        for curve in self.curves:
            projected = closest_point_on_constraint(curve, xyz)
            distance = distance_point_point(xyz, projected)
            if minimum is None or distance < minimum:
                best, minimum = projected, distance
        return best


def automated_smoothing_surface_constraints(mesh: Mesh, surface: Any, borders: Sequence[Any], kinks: Sequence[list[float]] | None = None) -> dict[int, Any]:
    """Constrain every vertex to a surface, each boundary vertex to its nearest border, and kinks to points.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        The mesh to constrain.
    surface : Any
        The surface, as anything with a ``closest_point`` method.
    borders : sequence
        The border curves of the surface, as polylines or curves.
    kinks : sequence[[float, float, float]], optional
        The points along the borders where the tangent jumps.

    Returns
    -------
    dict
        Vertex keys pointing to the surface, a border curve or a ``compas.geometry.Point``.
    """
    constraints = {vertex: surface for vertex in mesh.vertices()}

    boundary = [vertex for loop in mesh_boundary_loops(mesh) for vertex in loop]
    for vertex in boundary:
        xyz = mesh.vertex_coordinates(vertex)
        constraints[vertex] = min(borders, key=lambda curve: distance_point_point(xyz, closest_point_on_constraint(curve, xyz)))

    if kinks and boundary:
        cloud = [mesh.vertex_coordinates(vertex) for vertex in boundary]
        for xyz in kinks:
            index = closest_point_in_cloud(xyz, cloud)[2]
            constraints[boundary[index]] = Point(*list(xyz)[:3])

    return constraints


def automated_smoothing_constraints(mesh: Mesh, points: Sequence[list[float]] | None = None, curves: Sequence[Any] | None = None, surface: Any | None = None) -> dict[int, Any]:
    """Constrain the vertices of a mesh to pins, boundary curves and a surface; pins win.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        The mesh to constrain.
    points : sequence[[float, float, float]], optional
        Each pins the mesh vertex nearest to it, anywhere in the mesh.
    curves : sequence, optional
        Curves for the boundary vertices, as polylines or curves.
    surface : Any, optional
        A surface -- or a mesh -- for every vertex, as anything with a
        ``closest_point`` method.

    Returns
    -------
    dict
        Vertex keys pointing to a point, curve or surface.
    """
    constraints = {}

    pinned = {}
    if points:
        vertices = list(mesh.vertices())
        cloud = [mesh.vertex_coordinates(vertex) for vertex in vertices]
        for xyz in points:
            pinned[vertices[closest_point_in_cloud(xyz, cloud)[2]]] = Point(*list(xyz)[:3])

    if surface is not None:
        constraints.update({vertex: surface for vertex in mesh.vertices()})

    if curves:
        for loop in mesh_boundary_loops(mesh):
            for segment in split_loop_at_corners(loop, pinned):
                curve = closest_curve(mesh, segment, curves)
                for vertex in segment:
                    constraints[vertex] = curve

    constraints.update(pinned)
    return constraints


def surface_constrained_smoothing(mesh: Mesh, surface: Any, borders: Sequence[Any], kmax: int = 100, damping: float = 0.5, algorithm: str = "centroid") -> dict[int, Any]:
    """Smooth a mesh on a surface, re-projecting boundary vertices onto the nearest border every iteration.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        A mesh to smooth, modified in place.
    surface : Any
        The surface, as anything with a ``closest_point`` method.
    borders : sequence
        The border curves of the surface, as polylines or curves.
    kmax : int, optional
        Number of iterations for smoothing. Default is ``100``.
    damping : float, optional
        Damping value for smoothing between 0 and 1. Default is ``0.5``.
    algorithm : {'centroid', 'area', 'centerofmass'}, optional
        Type of smoothing algorithm to apply. Classic centroid by default.

    Returns
    -------
    dict
        The constraints that were applied. The mesh itself is modified in place.
    """
    edge = _NearestOf(borders)
    constraints = {}
    for vertex in mesh.vertices():
        constraints[vertex] = edge if mesh.is_vertex_on_boundary(vertex) else surface

    fixed = [vertex for loop in mesh_boundary_loops(mesh) for vertex in loop if mesh.vertex_degree(vertex) == 2]

    constrained_smoothing(mesh, kmax=kmax, damping=damping, constraints=constraints, algorithm=algorithm, fixed=fixed)

    return constraints
