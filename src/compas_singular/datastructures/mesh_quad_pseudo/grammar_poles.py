from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from compas_singular.datastructures import PseudoQuadMesh

__all__ = ["split_quad_in_pseudo_quads", "merge_pseudo_quads_in_quad"]


def split_quad_in_pseudo_quads(mesh: "PseudoQuadMesh", fkey: int, vkey: int) -> dict[int, int] | None:

    if len(mesh.face_vertices(fkey)) != 4:
        return None

    a = vkey
    b = mesh.face_vertex_descendant(fkey, a)
    c = mesh.face_vertex_descendant(fkey, b)
    d = mesh.face_vertex_descendant(fkey, c)

    mesh.delete_face(fkey)

    fkey_1 = mesh.add_face([a, b, c])
    fkey_2 = mesh.add_face([a, c, d])

    return {fkey_1: a, fkey_2: a}


def merge_pseudo_quads_in_quad(mesh: "PseudoQuadMesh", fkey_1: int, fkey_2: int) -> int | None:

    edge = mesh.face_adjacency_halfedge(fkey_1, fkey_2)

    if edge is None:
        return None

    a, c = edge
    b = mesh.face_vertex_descendant(fkey_2, a)
    d = mesh.face_vertex_descendant(fkey_1, c)

    mesh.delete_face(fkey_1)
    mesh.delete_face(fkey_2)

    return mesh.add_face([a, b, c, d])


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass

    # import compas
    # from compas_singular.datastructures.mesh_quad_pseudo.mesh_quad_pseudo import PseudoQuadMesh

    # vertices = [
    # [0.0, 0.0, 0.0],
    # [1.0, 0.0, 0.0],
    # [1.0, 1.0, 0.0],
    # [0.0, 1.0, 0.0]
    # ]

    # faces = [
    # [0, 1, 2, 3]
    # ]

    # mesh = PseudoQuadMesh.from_vertices_and_faces(vertices, faces)
    # print(split_quad_in_pseudo_quads(mesh, 0, 0))
    # for fkey in mesh.faces():
    #     print(fkey, mesh.face_vertices(fkey))
    # print(merge_pseudo_quads_in_quad(mesh, 1, 2))
    # for fkey in mesh.faces():
    #     print(fkey, mesh.face_vertices(fkey))
