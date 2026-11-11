from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any

__all__ = ["is_adjacency_two_colorable"]


def is_adjacency_two_colorable(adjacency: dict[Any, list[Any]]) -> dict[Any, int] | None:
    """Try to color a data of adjacency with two colors only withtout any element adjacent to each other having the same colour.

    Parameters
    ----------
    adjacency : dict
        Dictionary of adjacency between elements, each elements points to the list of adjacent elements.

    Returns
    -------
    key_to_color : dict, None
        A dictionary with vertex keys pointing to colors, if two-colorable.
        None if not two-colorable.

    """

    # store color status of network vertices (-1 means no color)
    key_to_color = {vkey: -1 for vkey in adjacency.keys()}

    count = len(adjacency.keys()) * 2

    # seed every connected component in turn, or the ones not reached from the first stay -1
    for key_0 in adjacency.keys():
        if key_to_color[key_0] != -1:
            continue
        sources = [key_0]

        # propagate until all vertices are colored or two adjacent vertices have the same color
        while count > 0 and sources:
            count -= 1
            key = sources.pop()
            nbr_colors = set([key_to_color[nbr] for nbr in adjacency[key]])

            # if two colors already exist in the neighbors, the network is not two-colourable
            if 0 in nbr_colors and 1 in nbr_colors:
                return None
            # otherwise, color with an available color
            else:
                if 0 not in nbr_colors:
                    key_to_color[key] = 0
                elif 1 not in nbr_colors:
                    key_to_color[key] = 1
                # add uncolored neighbors to sources
                sources += [nbr for nbr in adjacency[key] if key_to_color[nbr] == -1 and nbr not in sources]

    return key_to_color


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass

    # import time
    # import compas
    # from compas_singular.datastructures.mesh.mesh import Mesh
    # from compas.topology import adjacency_from_edges
    # from compas.topology import vertex_coloring
    # # edges = [
    # # 	(0, 1),
    # # 	(1, 2),
    # # 	(2, 3),
    # # 	(3, 4),
    # # 	(4, 5),
    # # 	(5, 0),
    # # 	(0, 3),
    # # 	(2, 5)
    # # ]

    # # adjacency = adjacency_from_edges(edges)
    # # print(adjacency)

    # # t0 = time.time()
    # # print(is_adjacency_two_colorable(adjacency))
    # # t1 = time.time()
    # # print(t1 - t0)

    # mesh = Mesh.from_obj(compas.get('faces.obj'))

    # t0 = time.time()
    # print(is_adjacency_two_colorable(mesh.adjacency))
    # t1 = time.time()
    # print(t1 - t0)

    # t0 = time.time()
    # print(vertex_coloring(mesh.adjacency))
    # t1 = time.time()
    # print(t1 - t0)
