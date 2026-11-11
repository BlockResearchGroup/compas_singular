from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING

from compas.topology import vertex_adjacency_from_edges
from compas.topology import vertex_coloring
from compas_singular.topology import is_adjacency_two_colorable

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh


__all__ = ["mesh_vertex_2_coloring", "mesh_vertex_n_coloring", "mesh_face_2_coloring", "mesh_face_n_coloring"]


def mesh_vertex_2_coloring(mesh: Mesh) -> dict[int, int] | None:
    """Try to color the vertices of a mesh with two colors only without adjacent vertices with the same color.

    Parameters
    ----------
    mesh : Mesh
        A mesh.

    Returns
    -------
    dict, None
        A dictionary with vertex keys pointing to colors, if two-colorable.
        None if not two-colorable.

    """

    return is_adjacency_two_colorable(mesh.adjacency)


def mesh_vertex_n_coloring(mesh: Mesh) -> dict[int, int]:
    """Color the vertices of a mesh with a minimum number of colors without adjacent vertices with the same color.

    Parameters
    ----------
    mesh : Mesh
        A mesh.

    Returns
    -------
    dict
        A dictionary with vertex keys pointing to colors.

    """

    return vertex_coloring(mesh.adjacency)


def mesh_face_2_coloring(mesh: Mesh) -> dict[int, int] | None:
    """Try to color the faces of a mesh with two colors only without adjacent faces with the same color.

    Parameters
    ----------
    mesh : Mesh
        A mesh.

    Returns
    -------
    dict, None
        A dictionary with face keys pointing to colors, if two-colorable.
        None if not two-colorable.

    """

    edges = [(mesh.halfedge[u][v], mesh.halfedge[v][u]) for u, v in mesh.edges() if not mesh.is_edge_on_boundary(u, v)]
    return is_adjacency_two_colorable(vertex_adjacency_from_edges(edges))


def mesh_face_n_coloring(mesh: Mesh) -> dict[int, int]:
    """Color the faces of a mesh with a minimum number of colors without adjacent faces with the same color.

    Parameters
    ----------
    mesh : Mesh
        A mesh.

    Returns
    -------
    dict
        A dictionary with face keys pointing to colors.

    """

    edges = [(mesh.halfedge[u][v], mesh.halfedge[v][u]) for u, v in mesh.edges() if not mesh.is_edge_on_boundary(u, v)]
    return vertex_coloring(vertex_adjacency_from_edges(edges))


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass

    # import compas
    # from compas.datastructures import Mesh

    # mesh = Mesh.from_obj(compas.get('faces.obj'))
    # print(mesh_vertex_2_coloring(mesh))
    # print(mesh_vertex_n_coloring(mesh))
    # print(mesh_face_2_coloring(mesh))
    # print(mesh_face_n_coloring(mesh))
