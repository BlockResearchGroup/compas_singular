from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING

from compas.itertools import pairwise
from compas_singular.datastructures.mesh_quad.coloring import quad_mesh_strip_2_coloring
from compas_singular.datastructures.mesh_quad_coarse.mesh_quad_coarse import CoarseQuadMesh

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = ["dense_quad_mesh_polyedge_2_coloring"]


def dense_quad_mesh_polyedge_2_coloring(quad_mesh: "QuadMesh") -> dict[int, int | None]:
    # assume that strips and polyedges are collected in the quad mesh

    # get coarse quad mesh
    coarse_quad_mesh = CoarseQuadMesh.from_quad_mesh(quad_mesh)

    # get coarse strip color
    coarse_skey_to_color = quad_mesh_strip_2_coloring(coarse_quad_mesh)

    # get coarse edge color
    coarse_edge_to_color = {edge: coarse_skey_to_color[skey] for skey in coarse_quad_mesh.strips() for edge in coarse_quad_mesh.strip_edges(skey)}

    # get dense polyedge color
    dense_polyedge_to_color = {tuple(coarse_quad_mesh.attributes["edge_coarse_to_dense"][u][v]): color for (u, v), color in coarse_edge_to_color.items()}

    # get some dense edge color
    some_dense_edge_to_color = {edge: color for polyedge, color in dense_polyedge_to_color.items() for edge in pairwise(polyedge)}

    # get strip color
    dense_strip_to_color = {}
    for skey in quad_mesh.strips():
        for u, v in quad_mesh.strip_edges(skey):
            color = some_dense_edge_to_color.get((u, v), some_dense_edge_to_color.get((v, u), None))
            if color is not None:
                dense_strip_to_color[skey] = color
                break

    # get edge color
    all_dense_edge_to_color = {edge: color for skey, color in dense_strip_to_color.items() for edge in quad_mesh.strip_edges(skey)}

    # get polyedge color
    dense_polyedge_to_color = {}
    for pkey, polyedge in quad_mesh.polyedges(data=True):
        u, v = polyedge[:2]
        color = all_dense_edge_to_color.get((u, v), all_dense_edge_to_color.get((v, u), None))
        dense_polyedge_to_color[pkey] = color

    return dense_polyedge_to_color


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass

    # import time
    # import compas
    # from compas_singular.datastructures.mesh_quad.mesh_quad import QuadMesh

    # mesh = QuadMesh.from_json('/Users/Robin/Desktop/json/debug.json')

    # t0 = time.time()
    # mesh.collect_strips()
    # mesh.collect_polyedges()
    # t1 = time.time()
    # dense_polyedge_to_color = dense_quad_mesh_polyedge_2_coloring(mesh)
    # t2 = time.time()
    # print(t2 - t1, t1 - t0)
    # print(dense_polyedge_to_color)
