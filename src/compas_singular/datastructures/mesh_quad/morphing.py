from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Callable

# from .coloring import quad_mesh_polyedge_2_coloring
from compas.itertools import pairwise
from compas_singular.datastructures.mesh.operations import mesh_move_vertices_by

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = ["fold", "fold_vertex_group"]


def fold_vertex_group(quad_mesh: QuadMesh, polyedges: list[list[int]]) -> dict[int, int]:
    # is it always possible? a type of vertex 2-coloring?

    is_edge_in_polyedges = {edge: False for edge in quad_mesh.edges()}
    for polyedge in polyedges:
        for u, v in pairwise(polyedge):
            edge = (u, v) if (u, v) in is_edge_in_polyedges else (v, u)
            is_edge_in_polyedges[edge] = True

    n = quad_mesh.number_of_vertices() * 2
    vkey0 = next(iter(quad_mesh.vertices()))
    vkey_to_group = {vkey0: 0}
    to_visit = quad_mesh.vertex_neighbors(vkey0)

    while len(to_visit) > 0 and n:
        n -= 1
        vkey = to_visit.pop()

        for nbr in quad_mesh.vertex_neighbors(vkey):
            if (vkey, nbr) in is_edge_in_polyedges:
                in_polyedge = is_edge_in_polyedges[(vkey, nbr)]
            elif (nbr, vkey) in is_edge_in_polyedges:
                in_polyedge = is_edge_in_polyedges[(nbr, vkey)]

            if nbr in vkey_to_group:
                vkey_to_group[vkey] = vkey_to_group[nbr] if not in_polyedge else 1 - vkey_to_group[nbr]

        for nbr in quad_mesh.vertex_neighbors(vkey):
            if (vkey, nbr) in is_edge_in_polyedges:
                in_polyedge = is_edge_in_polyedges[(vkey, nbr)]
            elif (nbr, vkey) in is_edge_in_polyedges:
                in_polyedge = is_edge_in_polyedges[(nbr, vkey)]

            if nbr in vkey_to_group:
                if (vkey_to_group[vkey] == vkey_to_group[nbr]) == in_polyedge:
                    print("!")

        to_visit += [nbr for nbr in quad_mesh.vertex_neighbors(vkey) if nbr not in vkey_to_group]

    return vkey_to_group


def fold(
    quad_mesh: QuadMesh, vkey_to_group: dict[int, int], func0: Callable[[QuadMesh, int], list[float]], func1: Callable[[QuadMesh, int], list[float]]
) -> dict[int, list[float]]:

    moves = {}
    for vkey, group in vkey_to_group.items():
        if group == 0:
            vector = func0(quad_mesh, vkey)
        elif group == 1:
            vector = func1(quad_mesh, vkey)
        moves[vkey] = vector
    mesh_move_vertices_by(quad_mesh, moves)
    return moves


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass

    # import compas
    # from compas_singular.datastructures.mesh_quad.mesh_quad import QuadMesh
    # from compas_singular.datastructures.mesh_quad_coarse.coloring import dense_quad_mesh_polyedge_2_coloring
    # from compas.geometry import scale_vector
    # from compas_plotters.meshplotter import MeshPlotter

    # def func0(mesh, vkey):
    #     normal = mesh.vertex_normal(vkey)
    #     return scale_vector(normal, 1)

    # def func1(mesh, vkey):
    #     normal = mesh.vertex_normal(vkey)
    #     return scale_vector(normal, -1)

    # mesh = QuadMesh.from_json('/Users/Robin/Desktop/json/debug.json')
    # mesh.collect_strips()
    # mesh.collect_polyedges()

    # polyedge_key_to_colour = dense_quad_mesh_polyedge_2_coloring(mesh)
    # polyedges_0 = [mesh.attributes['polyedges'][key] for key, colour in polyedge_key_to_colour.items() if colour == 0]
    # polyedges_1 = [mesh.attributes['polyedges'][key] for key, colour in polyedge_key_to_colour.items() if colour == 1]

    # vkey_to_group = fold_vertex_group(mesh, polyedges_1)
    # # fold(mesh, vkey_to_group, func0, func1)

    # plotter = MeshPlotter(mesh, figsize=(20, 20))
    # plotter.draw_vertices(radius=0.4, text=vkey_to_group)
    # plotter.draw_edges()
    # plotter.draw_faces()
    # plotter.show()
