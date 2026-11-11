from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any
from typing import Sequence

import compas_rhino
import rhinoscriptsyntax as rs  # type: ignore  # noqa: I001
from compas_rhino.geometry import RhinoSurface

from compas.datastructures import Mesh
from compas.datastructures import Network
from compas.datastructures.graph.operations.join import graph_polylines
from compas.geometry import angle_vectors
from compas.geometry import distance_point_point
from compas.itertools import pairwise
from compas_singular.rhino.geometry.curve import RhinoCurve

__all__ = ["RhinoSurface"]


class RhinoSurface(RhinoSurface):
    def __init__(self) -> None:
        super(RhinoSurface, self).__init__()

    def bounding_box(self) -> Any:
        return rs.BoundingBox([self.guid])

    def borders(self, border_type: int = 0) -> list[Any]:
        """Duplicate the borders of the surface.

        Parameters
        ----------
        border_type : {0, 1, 2}
            The type of border.

            * 0: All borders
            * 1: The exterior borders.
            * 2: The interior borders.

        Returns
        -------
        list
            The GUIDs of the extracted border curves.

        """
        curves = rs.DuplicateSurfaceBorder(self.guid, type=border_type)
        exploded_curves = rs.ExplodeCurves(curves, delete_input=False)
        if len(exploded_curves) == 0:
            return curves
        rs.DeleteObjects(curves)
        return exploded_curves

    def kinks(self, threshold: float = 1e-3) -> list[Sequence[float]]:
        """Return the XYZ coordinates of kinks, i.e. tangency discontinuities, along the surface's boundaries.

        Returns
        -------
        list
            The list of XYZ coordinates of surface boundary kinks.

        """
        kinks = []
        borders = self.borders(border_type=0)

        for border_guid in borders:
            extremities = map(
                lambda x: rs.EvaluateCurve(border_guid, rs.CurveParameter(border_guid, x)),
                [0.0, 1.0],
            )

            if rs.IsCurveClosed(border_guid):
                start_tgt = rs.CurveTangent(border_guid, rs.CurveParameter(border_guid, 0.0))
                end_tgt = rs.CurveTangent(border_guid, rs.CurveParameter(border_guid, 1.0))
                if angle_vectors(start_tgt, end_tgt) > threshold:
                    kinks += extremities

            else:
                kinks += extremities

        rs.DeleteObjects(borders)
        return list(set(kinks))

    def closest_point_on_boundaries(self, xyz: Sequence[float]) -> Sequence[float]:
        """Return the XYZ coordinates of the closest point on the boundaries of the surface from input XYZ-coordinates.

        Parameters
        ----------
        xyz : list
            XYZ coordinates.

        Returns
        -------
        list
            The XYZ coordinates of the closest point on the boundaries of the surface.

        """
        borders = self.borders(border_type=0)
        proj_dist = {tuple(proj_xyz): distance_point_point(xyz, proj_xyz) for proj_xyz in [RhinoCurve(border).closest_point(xyz) for border in borders]}
        compas_rhino.delete_objects(borders)
        return min(proj_dist, key=proj_dist.get)

    def closest_points_on_boundaries(self, points: Sequence[Sequence[float]]) -> list[Sequence[float]]:
        return [self.closest_point_on_boundaries(point) for point in points]

    # --------------------------------------------------------------------------
    # mapping
    # --------------------------------------------------------------------------

    def point_xyz_to_uv(self, xyz: Sequence[float]) -> Sequence[float]:
        """Return the UV point from the mapping of a XYZ point based on the UV parameterisation of the surface.

        Parameters
        ----------
        xyz : list
            (x, y, z) coordinates.

        Returns
        -------
        list
            The (u, v) coordinates of the mapped point.

        """
        return rs.SurfaceClosestPoint(self.guid, xyz)

    def point_uv_to_xyz(self, uv: Sequence[float]) -> tuple[float, ...]:
        """Return the XYZ point from the inverse mapping of a UV point based on the UV parameterisation of the surface.

        Parameters
        ----------
        uv : list
            (u, v) coordinates.

        Returns
        -------
        list
            The (x, y, z) coordinates of the inverse-mapped point.

        """
        return tuple(rs.EvaluateSurface(self.guid, *uv))

    def line_uv_to_xyz(self, line: Sequence[Sequence[float]]) -> tuple[Any, Any]:
        """Return the XYZ points from the inverse mapping of a UV line based on the UV parameterisation of the surface.

        Parameters
        ----------
        uv : list
            List of (u, v) coordinates.

        Returns
        -------
        list
            The list of XYZ coordinates of the inverse-mapped line.

        """
        return (self.point_uv_to_xyz(line[0]), self.point_uv_to_xyz(line[1]))

    def polyline_uv_to_xyz(self, polyline: Sequence[Sequence[float]]) -> list[Any]:
        """Return the XYZ points from the inverse mapping of a UV polyline based on the UV parameterisation of the surface.

        Parameters
        ----------
        uv : list
            List of (u, v) coordinates.

        Returns
        -------
        list
            The list of (x, y, z) coordinates of the inverse-mapped polyline.

        """
        return [self.point_uv_to_xyz(vertex) for vertex in polyline]

    def mesh_uv_to_xyz(self, mesh: Mesh) -> None:
        """Return the mesh from the inverse mapping of a UV mesh based on the UV parameterisation of the surface.
        The third coordinate of the mesh vertices is discarded.

        Parameters
        ----------
        mesh : Mesh
            A mesh.

        Returns
        -------
        None
            The mesh is modified in-place.

        """
        for vertex in mesh.vertices():
            xyz = self.point_uv_to_xyz(mesh.vertex_coordinates(vertex)[:2])
            mesh.vertex_attributes(vertex, "xyz", xyz)

    def discrete_mapping(
        self,
        segment_length: float,
        minimum_discretisation: int = 5,
        crv_guids: Sequence[Any] = [],
        pt_guids: Sequence[Any] = [],
    ) -> tuple[Any, list[Any], list[Any], list[Any]]:
        """Map the boundaries of a Rhino NURBS surface to planar poylines dicretised within some discretisation
        using the surface UV parameterisation. Curve and point feautres on the surface can be included.

        Parameters
        ----------
        segment_length : float
            The target discretisation length of the surface boundaries.
        minimum_discretisation : int
            The minimum discretisation of the surface boundaries.
        crv_guids : list
            List of guids of curves on the surface.
        pt_guids : list
            List of guids of points on the surface.

        Returns
        -------
        tuple
            Tuple of the mapped objects: outer boundary, inner boundaries, polyline_features, point_features.
        """
        # a boundary may be made of multiple boundary components and therefore checking for closeness and joining are necessary
        borders = []
        for btype in (1, 2):
            border = []
            for guid in self.borders(border_type=btype):
                L = rs.CurveLength(guid)
                N = max(int(L / segment_length) + 1, minimum_discretisation)
                points = []
                for point in rs.DivideCurve(guid, N):
                    points.append(list(self.point_xyz_to_uv(point)) + [0.0])
                if rs.IsCurveClosed(guid):
                    points.append(points[0])
                border.append(points)
                rs.DeleteObject(guid)
            borders.append(border)
        outer_boundaries = graph_polylines(Network.from_lines([(u, v) for border in borders[0] for u, v in pairwise(border)]))
        inner_boundaries = graph_polylines(Network.from_lines([(u, v) for border in borders[1] for u, v in pairwise(border)]))

        # mapping of the curve features on the surface
        curves = []
        for guid in crv_guids:
            L = rs.CurveLength(guid)
            N = max(int(L / segment_length) + 1, minimum_discretisation)
            points = []
            for point in rs.DivideCurve(guid, N):
                points.append(list(self.point_xyz_to_uv(point)) + [0.0])
            if rs.IsCurveClosed(guid):
                points.append(points[0])
            curves.append(points)
        polyline_features = graph_polylines(Network.from_lines([(u, v) for curve in curves for u, v in pairwise(curve)]))

        # mapping of the point features onthe surface
        point_features = [list(self.point_xyz_to_uv(rs.PointCoordinates(guid))) + [0.0] for guid in pt_guids]

        return outer_boundaries[0], inner_boundaries, polyline_features, point_features
