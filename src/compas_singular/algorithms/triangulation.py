from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any
from typing import Callable

from compas.datastructures import Graph
from compas.datastructures.graph.operations.join import graph_polylines
from compas.datastructures.mesh.operations.weld import mesh_unweld_edges
from compas.geometry import Polyline
from compas.geometry import delaunay_triangulation as delaunay_from_points
from compas.geometry import distance_point_point
from compas.geometry import intersection_segment_segment_xy
from compas.geometry import is_point_in_polygon_xy
from compas.itertools import pairwise
from compas.tolerance import TOL
from compas_singular.datastructures import Mesh
from compas_singular.datastructures import is_face_degenerate
from compas_singular.datastructures import trimesh_face_circle

__all__ = [
    "boundary_triangulation",
    "weld_polyline_features",
    "arrange_polyline_features",
    "as_points",
    "as_curves",
]


def as_points(curve: Polyline | list, close: bool | None = None) -> list[list[float]]:
    """Coerce one curve (``Polyline``, anything with ``.points``, or a point list) to a list of ``[x, y, z]``.

    Parameters
    ----------
    curve : Polyline | list
        The curve.
    close : bool, optional
        ``False`` drops a final point coincident with the first, which is the
        convention for a boundary loop. ``True`` adds one, which is the
        convention for a closed curve feature -- the cut has to come back round
        to where it started. ``None`` leaves the ends alone.

    Returns
    -------
    list[list[float]]
    """
    if hasattr(curve, "points"):
        curve = list(curve.points)

    points = curve
    if any(not isinstance(point, list) for point in curve):
        points = [[float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0] for point in curve]

    if close is not None and len(points) > 1:
        closed = TOL.geometric_key(points[0]) == TOL.geometric_key(points[-1])
        if close and not closed:
            points = list(points) + [list(points[0])]
        elif closed and not close:
            points = list(points[:-1])
    return points


def as_curves(curves: Polyline | list | None, close: bool | None = None) -> list[list[list[float]]]:
    """Coerce one curve, or a list of curves, to a list of lists of ``[x, y, z]``.

    A single curve is wrapped in a list, so ``polyline_features=my_polyline``
    means the same as ``polyline_features=[my_polyline]``.

    Parameters
    ----------
    curves : Polyline | list
        One curve or a list of curves, in any of the forms ``as_points``
        accepts.
    close : bool, optional
        Passed through to ``as_points`` for every curve.

    Returns
    -------
    list[list[list[float]]]

    """
    if curves is None:
        return []
    if hasattr(curves, "points"):  # a single Polyline
        return [as_points(curves, close)]
    curves = list(curves)
    if not curves:
        return []
    if hasattr(curves[0], "points"):  # a list of Polylines
        return [as_points(curve, close) for curve in curves]
    try:  # a single curve of points
        float(curves[0][0])
        float(curves[0][1])
        return [as_points(curves, close)]
    except (TypeError, IndexError, ValueError):
        return [as_points(curve, close) for curve in curves]


def weld_polyline_features(polyline_features: list[list[list[float]]]) -> list[list[list[float]]]:
    """Weld feature polylines that share a point into chains split at every non-two-valent junction.

    Crossings without a shared point are left alone; see ``arrange_polyline_features``.

    Parameters
    ----------
    polyline_features : list
        List of planar polylines as lists of vertex coordinates.

    Returns
    -------
    list
        The welded polylines. A single polyline, or an empty list, is returned
        unchanged.
    """
    # One chain has no junction to weld, and round-tripping it through the
    # network would renumber its points and move the Delaunay's tie-breaks.
    if len(polyline_features) < 2:
        return polyline_features
    return graph_polylines(Graph.from_lines([(u, v) for polyline in polyline_features for u, v in pairwise(polyline)]))


def arrange_polyline_features(polyline_features: list[list[list[float]]], tol: float = 1e-9) -> list[list[list[float]]]:
    """Put a vertex where two curve features cross, so the Delaunay can hold both cuts.

    Parameters
    ----------
    polyline_features : list
        Planar polylines as lists of vertex coordinates.
    tol : float, optional
        Distance below which a crossing counts as an existing vertex.

    Returns
    -------
    list
        The polylines, with a vertex inserted at every crossing.
    """
    if len(polyline_features) < 2:
        return polyline_features

    chains = [[list(point) for point in polyline] for polyline in polyline_features]
    cuts = [[] for _ in chains]
    for i, one in enumerate(chains):
        for j, two in enumerate(chains):
            if j <= i:
                continue
            for a, (p1, p2) in enumerate(pairwise(one)):
                for b, (p3, p4) in enumerate(pairwise(two)):
                    point = intersection_segment_segment_xy((p1, p2), (p3, p4))
                    if point is None:
                        continue
                    point = [point[0], point[1], 0.0]
                    # a crossing AT a point one of them already has is a shared
                    # node, which weld_polyline_features handles on its own
                    for index, seg, ends in ((i, a, (p1, p2)), (j, b, (p3, p4))):
                        if all(distance_point_point(point, end) > tol for end in ends):
                            cuts[index].append((seg, point))

    if not any(cuts):
        return polyline_features

    out = []
    for chain, chain_cuts in zip(chains, cuts):
        if not chain_cuts:
            out.append(chain)
            continue
        by_segment = {}
        for seg, point in chain_cuts:
            by_segment.setdefault(seg, []).append(point)
        points = [chain[0]]
        for index, (a, b) in enumerate(pairwise(chain)):
            for point in sorted(by_segment.get(index, []), key=lambda q: distance_point_point(a, q)):
                if distance_point_point(points[-1], point) > tol:
                    points.append(point)
            points.append(b)
        out.append(points)
    return out


def boundary_triangulation(
    outer_boundary: Polyline | list,
    inner_boundaries: list,
    polyline_features: Polyline | list = [],
    point_features: list = [],
    delaunay: Callable[..., Any] | None = None,
) -> Mesh:
    """Delaunay triangulation of boundary (and feature) points, cut open along curve features (thesis S4.2.1, S4.3.2).

    The caller does the sampling; the constrained Delaunay of Fig 4.20 is not implemented.

    Parameters
    ----------
    outer_boundary : Polyline | list
        Planar outer boundary, as a list of vertex coordinates or a
        ``compas.geometry.Polyline``. A final point coincident with the
        first is dropped.
    inner_boundaries : list
        Planar inner boundaries, as a list of the above.
    polyline_features : Polyline | list
        Planar curve features, as a list of the above -- or one curve on its
        own. A closed feature must repeat its first point at the end, and one
        given as a closed ``Polyline`` already does.
    point_features : list
        Planar point features, as a flat list of ``[x, y, z]`` or of
        ``compas.geometry.Point``. NOT a list of lists of points.
    delaunay : callable or proxy
        Delaunay triangulation function.

    Returns
    -------
    delaunay_mesh : Mesh
        The Delaunay mesh.
    """
    if not delaunay:
        delaunay = delaunay_from_points

    # accept Polylines and Points as readily as lists of [x, y, z]
    outer_boundary = as_points(outer_boundary, close=False)
    inner_boundaries = as_curves(inner_boundaries, close=False)
    polyline_features = as_curves(polyline_features)
    point_features = [as_points([point])[0] for point in (point_features or [])]

    # a planar arrangement of the features -- a vertex at every crossing --
    # then the chains split at every junction, as the Rhino front end does
    polyline_features = arrange_polyline_features(polyline_features)
    polyline_features = weld_polyline_features(polyline_features)

    # generate planar Delaunay triangulation
    vertices = [pt for boundary in [outer_boundary] + inner_boundaries + polyline_features for pt in boundary] + point_features
    faces = delaunay(vertices)

    delaunay_mesh = Mesh.from_vertices_and_faces(vertices, faces)

    # delete false faces with aligned vertices.
    #
    # Qhull emits flat simplices wherever the input points are collinear, and
    # the discretisation above puts a RUN of collinear points along every
    # straight stretch of wall -- so a polygonal domain makes them by the
    # dozen. The test has to be scale-relative rather than an exact zero:
    # a flat triangle's doubled area comes out as 0.0 from one pair of its edge
    # vectors and as 8e-17 from another, and the next loop divides by the second
    # formulation. An exact test on the first let those through to a
    # ZeroDivisionError in ``trimesh_face_circle`` -- seen on a 12-gon wall
    # around a hexagonal hole, where a third of the flat faces landed on the
    # wrong side of the disagreement.
    #
    # Note what that makes the failure look like from the front end: REDRAWING
    # THE SAME SHAPE FIXES IT. The corners land on slightly different floats,
    # the flat faces come out of Qhull in a different vertex order, and the two
    # formulations happen to agree that time. Nothing about the geometry
    # changed, so an intermittent crash here is not evidence that the drawing
    # was at fault.
    #
    # The exact test this replaced, kept for reference -- it needs
    # ``length_vector``, ``subtract_vectors`` and ``cross_vectors`` from
    # ``compas.geometry`` back at the top of the module to run:
    #
    # for fkey in list(delaunay_mesh.faces()):
    #     a, b, c = [delaunay_mesh.vertex_coordinates(vkey) for vkey in delaunay_mesh.face_vertices(fkey)]
    #     ab = subtract_vectors(b, a)
    #     ac = subtract_vectors(c, a)
    #     if length_vector(cross_vectors(ab, ac)) == 0:
    #         delaunay_mesh.delete_face(fkey)
    for fkey in list(delaunay_mesh.faces()):
        a, b, c = [delaunay_mesh.vertex_coordinates(vkey) for vkey in delaunay_mesh.face_vertices(fkey)]
        if is_face_degenerate(a, b, c):
            delaunay_mesh.delete_face(fkey)

    # delete faces outisde the borders
    for fkey in list(delaunay_mesh.faces()):
        centre = trimesh_face_circle(delaunay_mesh, fkey)[0]
        if not is_point_in_polygon_xy(centre, outer_boundary) or any([is_point_in_polygon_xy(centre, inner_boundary) for inner_boundary in inner_boundaries]):
            delaunay_mesh.delete_face(fkey)

    # topological cut along the feature polylines through unwelding
    #
    # A feature point that is also a wall point, or the end of another chain, is
    # in ``vertices`` twice, and only one copy is used by the triangulation. The
    # map must point at THAT copy: pointing at the unused one makes the cut edge
    # a non-edge, so a feature landing on a wall was never cut through there --
    # the line stayed a slit, its end segments uncut and its corners lost.
    vertex_map = {}
    for vkey in delaunay_mesh.vertices():
        key = TOL.geometric_key(delaunay_mesh.vertex_coordinates(vkey))
        if key not in vertex_map or delaunay_mesh.vertex_faces(vkey):
            vertex_map[key] = vkey
    edges = [edge for polyline in polyline_features for edge in pairwise([vertex_map[TOL.geometric_key(point)] for point in polyline])]
    mesh_unweld_edges(delaunay_mesh, edges)

    # Record the cut, for ``Skeleton.real_neighbors``.
    # The features as point chains. The grafting needs to know which samples
    # belong to the same curve, and in what order -- see
    # ``SkeletonDecomposition.branches_singularity_to_boundary``.
    delaunay_mesh.attributes["feature_points"] = [[list(point) for point in polyline] for polyline in polyline_features]
    delaunay_mesh.attributes["feature_edges"] = frozenset((TOL.geometric_key(a), TOL.geometric_key(b)) for polyline in polyline_features for a, b in pairwise(polyline))

    return delaunay_mesh


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass
