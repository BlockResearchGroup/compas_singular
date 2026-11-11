from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any

from compas.datastructures import Graph
from compas.datastructures.graph.operations.join import graph_polylines
from compas.tolerance import TOL
from compas_singular.datastructures.mesh import Mesh
from compas_singular.datastructures.mesh import trimesh_face_circle

__all__ = ["Skeleton"]


class Skeleton(Mesh):
    """Skeleton class for the generation of the topological skeleton or medial axis from a Delaunay mesh.

    References
    ----------
    1. Harry Blum. 1967. *A transformation for extracting new descriptors of shape*.
           Models for Perception of Speech and Visual Forms, pages 362--380.
           Available at http://pageperso.lif.univ-mrs.fr/~edouard.thiel/rech/1967-blum.pdf.
    2. Punam K. Saha, Gunilla Borgefors, and Gabriella Sanniti di Baja. 2016. *A survey on skeletonization algorithms and their applications*.
           Pattern Recognition Letters, volume 76, pages 3--12.
           Available at https://www.sciencedirect.com/science/article/abs/pii/S0167865515001233.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(Skeleton, self).__init__(*args, **kwargs)
        #: Curve-feature edges as pairs of ``TOL.geometric_key``. An adjacency
        #: across one of these is not a real adjacency -- see
        #: ``real_neighbors``. Empty unless set by
        #: ``compas_singular.algorithms.boundary_triangulation``.
        self.feature_edges = frozenset()
        #: The curve features as ordered point chains. Grafting uses the
        #: ordering to tell whether two branches land at the same place.
        self.feature_points = []

    def real_neighbors(self, fkey: int) -> list[int]:
        """The adjacent faces of ``fkey``, excluding any across a curve feature.

        A safeguard: with a complete topological cut it equals ``face_neighbors``.

        Returns
        -------
        list[int]
        """
        neighbors = self.face_neighbors(fkey)
        if not self.feature_edges:
            return neighbors
        return [nbr for nbr in neighbors if not self._adjacent_across_feature(fkey, nbr)]

    def _adjacent_across_feature(self, fkey: int, nbr: int) -> bool:
        """Do these two faces share an edge that lies on a curve feature?"""
        shared = set(self.face_vertices(fkey)) & set(self.face_vertices(nbr))
        if len(shared) != 2:
            return False
        u, v = (TOL.geometric_key(self.vertex_coordinates(vkey)) for vkey in shared)
        return (u, v) in self.feature_edges or (v, u) in self.feature_edges

    @classmethod
    def from_mesh(cls, mesh: Mesh) -> "Skeleton":
        """Construct a Skeleton object from a Mesh.

        Returns
        -------
        Skeleton
            A skeleton object.

        """
        skeleton = cls.from_vertices_and_faces(*mesh.to_vertices_and_faces())
        skeleton.feature_edges = frozenset(mesh.attributes.get("feature_edges") or ())
        return skeleton

    def singular_faces(self) -> list[int]:
        """Get the indices of the singular faces in the Delaunay mesh, i.e. the ones with three neighbours.

        Returns
        -------
        list
            List of face keys.

        """
        return [fkey for fkey in self.faces() if len(self.real_neighbors(fkey)) == 3]

    def singular_points(self) -> list[list[float]]:
        """Get the XYZ-coordinates of the singular points of the topological skeleton, i.e. the face circumcentre of the singular faces.

        Returns
        -------
        list
            List of point XYZ-coordinates.

        """
        return [trimesh_face_circle(self, fkey)[0] for fkey in self.singular_faces()]

    def lines(self) -> list[tuple[list[float], list[float]]]:
        """Get the lines forming the topological skeleton, i.e. the lines connecting the circumcentres of adjacent faces.

        Returns
        -------
        list
            List of lines as tuples of pairs XYZ-coordinates.

        """
        return [
            (trimesh_face_circle(self, fkey)[0], trimesh_face_circle(self, nbr)[0])
            for fkey in self.faces()
            for nbr in self.real_neighbors(fkey)
            if fkey < nbr and TOL.geometric_key(trimesh_face_circle(self, fkey)[0]) != TOL.geometric_key(trimesh_face_circle(self, nbr)[0])
        ]

    def branches(self) -> list[list[list[float]]]:
        """Get the branch polylines of the topological skeleton as polylines connecting singular points.

        Returns
        -------
        list
            List of polylines as tuples of XYZ-coordinates.

        """
        return graph_polylines(Graph.from_lines(self.lines()))


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass
