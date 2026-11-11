from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING

from compas.geometry import centroid_points
from compas.geometry import distance_point_point

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh
    from compas_singular.datastructures import QuadMesh


__all__ = [
    "add_opening",
    "add_handle",
    # 'close_opening',
    # 'close_handle'
]


def add_opening(mesh: Mesh, fkey: int) -> list[int]:
    """Add an opening to a mesh face.

    Parameters
    ----------
    mesh : Mesh
        A mesh.
    fkey : hashable
        A face key.

    Returns
    -------
    list
        List of new vertex keys around the opening oriented towards the outside.
    """

    initial_vertices = mesh.face_vertices(fkey)
    new_vertices = [
        mesh.add_vertex(attr_dict={i: xyz for i, xyz in zip(["x", "y", "z"], centroid_points([mesh.face_centroid(fkey), mesh.vertex_coordinates(vkey)]))})
        for vkey in initial_vertices
    ]
    mesh.delete_face(fkey)
    _ = [mesh.add_face([initial_vertices[i - 1], initial_vertices[i], new_vertices[i], new_vertices[i - 1]]) for i in range(len(initial_vertices))]
    return new_vertices


def add_handle(mesh: QuadMesh, fkey_1: int, fkey_2: int) -> list[int]:
    """Add a handle between two quad mesh faces.

    Parameters
    ----------
    mesh : QuadMesh
        A quad mesh.
    fkey_1 : hashable
        A face key.
    fkey_2 : hashable
        A face key.

    Returns
    -------
    list
        List of new face keys around the opening
    """

    # add two openings
    new_vertices_1 = add_opening(mesh, fkey_1)
    new_vertices_2 = add_opening(mesh, fkey_2)

    # get offset between new openings as to minimise the twist of the handle
    offset_distance = {
        k: sum([distance_point_point(mesh.vertex_coordinates(new_vertices_1[(i - 1) % 4]), mesh.vertex_coordinates(new_vertices_2[(-i - k) % 4])) for i in range(4)])
        for k in range(4)
    }
    k = min(offset_distance, key=offset_distance.get)

    # add handle
    return [mesh.add_face([new_vertices_1[(i - 1) % 4], new_vertices_1[i], new_vertices_2[(-i - 1 - k) % 4], new_vertices_2[(-i - k) % 4]]) for i in range(4)]


# def close_opening(mesh, polyedge):
#     return 0


# def close_handle(mesh, fkeys):
#     # remove handle and close openings
#     # fkeys: closed face strip

#     if fkeys[0] == fkeys[-1]:
#         del fkeys[-1]

#     vertices = []
#     key_to_index = {}
#     for i, vkey in enumerate(mesh.vertices()):
#         vertices.append(mesh.vertex_coordinates(vkey))
#         key_to_index[vkey] = i
#     faces = [[key_to_index[vkey] for vkey in mesh.face_vertices(fkey)] for fkey in fkeys]
#     strip_mesh = Mesh.from_vertices_and_faces(vertices, faces)

#     boundaries = strip_mesh.polyedge_boundaries(strip_mesh)

#     for fkey in fkeys:
#         mesh.delete_face(fkey)
#     new_fkeys = []
#     for bdry in boundaries:
#         new_fkeys += close_opening(mesh, list(reversed(bdry)))

#     return new_fkeys


# def close_handle_2(mesh, edge_path_1, edge_path_2):
#     # two closed edge paths

#     # unweld
#     unweld_mesh_along_edge_path(mesh, edge_path_1)
#     unweld_mesh_along_edge_path(mesh, edge_path_2)

#     # explode
#     parts = mesh_disjointed_parts(mesh)
#     meshes = unjoin_mesh_parts(mesh, parts)

#     # find parts with the topolog of a strip: two boundary components and an EUler characteristic of 0
#     # if there are several, select the topologically smallest one (lowest number of faces)
#     index = -1
#     size = -1
#     for i, submesh in enumerate(meshes):
#         B = len(mesh.polyedge_boundaries())
#         X = submesh.mesh_euler()
#         if B == 2 and X == 0:
#             n = submesh.number_of_faces()
#             if index < 0 or n < size:
#                 index = i
#                 size = n

#     # collect the boundaries of the strip, oriented towards the outside of the strip
#     vertices = []
#     key_to_index = {}
#     for i, vkey in enumerate(mesh.vertices()):
#         vertices.append(mesh.vertex_coordinates(vkey))
#         key_to_index[vkey] = i
#     faces = [[key_to_index[vkey] for vkey in mesh.face_vertices(fkey)] for fkey in parts[index]]
#     strip_mesh = Mesh.from_vertices_and_faces(vertices, faces)

#     boundaries = strip_mesh.polyedge_boundaries()

#     # remove faces of the selected band
#     for fkey in parts[index]:
#         mesh.delete_face(fkey)

#     # close the two boundaries
#     new_fkeys = []
#     for bdry in boundaries:
#         new_fkeys += close_opening(mesh, list(reversed(bdry)))

#     return new_fkeys


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass
