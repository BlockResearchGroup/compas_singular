from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any

import networkx as nx

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = [
    "strip_graph",
    "are_strip_graphs_isomorphic",
    "are_strips_isomorphic",
    "mesh_graph",
    "are_mesh_graphs_isomorphic",
    "are_meshes_isomorphic",
    "matches_between_ismorphic_meshes",
]


# --------------------------------------------------------------------------
# strip isomorphism
# --------------------------------------------------------------------------


def strip_graph(mesh: QuadMesh, close_strip_data: bool = False) -> nx.MultiGraph:
    # graph of quad mesh strips: one graph vertex <-> one mesh strip and one graph <-> edge one mesh face
    # graph vertices have an attribute whether the corresponding strip is closed or not
    if mesh.attributes["strips"] is None or mesh.attributes["strips"] == {}:
        mesh.collect_strips()
    graph = nx.MultiGraph([tuple(mesh.face_strips(fkey)) for fkey in mesh.faces()])
    if close_strip_data:
        nx.set_node_attributes(graph, {skey: {"closed": mesh.is_strip_closed(skey)} for skey in mesh.strips()})
    return graph


def are_strip_graphs_isomorphic(strip_graph_i: nx.MultiGraph, strip_graph_j: nx.MultiGraph) -> bool:
    # check if two strip graphs are isomorphic, including closeness data
    return nx.is_isomorphic(strip_graph_i, strip_graph_j, node_match=nx.isomorphism.categorical_node_match("closed", None))


def are_strips_isomorphic(mesh_i: QuadMesh, mesh_j: QuadMesh, close_strip_data: bool = False) -> bool:
    strip_graph_i = strip_graph(mesh_i, close_strip_data=close_strip_data)
    strip_graph_j = strip_graph(mesh_j, close_strip_data=close_strip_data)
    return are_strip_graphs_isomorphic(strip_graph_i, strip_graph_j)


# --------------------------------------------------------------------------
# mesh isomorphism
# --------------------------------------------------------------------------


def mesh_graph(mesh: QuadMesh, boundary_edge_data: bool = False) -> nx.MultiGraph:
    # graph of meshes with edges only
    # edges have an attribute whether they are on the boundary or not (vertex attributes would not be sufficient)
    graph = nx.MultiGraph(mesh.edges())
    if boundary_edge_data:
        nx.set_edge_attributes(graph, {(*edge, i): {"boundary": mesh.is_edge_on_boundary(*edge)} for i, edge in enumerate(mesh.edges())})
    return graph


def are_mesh_graphs_isomorphic(mesh_graph_i: nx.MultiGraph, mesh_graph_j: nx.MultiGraph) -> bool:
    # check if two mesh graphs are isomorphic, including boundary data
    return nx.is_isomorphic(mesh_graph_i, mesh_graph_j, edge_match=nx.isomorphism.categorical_edge_match("boundary", None))


def are_meshes_isomorphic(mesh_i: QuadMesh, mesh_j: QuadMesh, boundary_edge_data: bool = False) -> bool:
    mesh_graph_i = mesh_graph(mesh_i, boundary_edge_data=boundary_edge_data)
    mesh_graph_j = mesh_graph(mesh_j, boundary_edge_data=boundary_edge_data)
    return are_mesh_graphs_isomorphic(mesh_graph_i, mesh_graph_j)


def matches_between_ismorphic_meshes(mesh_i: QuadMesh, mesh_j: QuadMesh, boundary_edge_data: bool = False) -> Any:
    mesh_graph_i = mesh_graph(mesh_i, boundary_edge_data=boundary_edge_data)
    mesh_graph_j = mesh_graph(mesh_j, boundary_edge_data=boundary_edge_data)
    matcher = nx.isomorphism.GraphMatcher(mesh_graph_i, mesh_graph_j)
    return matcher.match()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass
