from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import pi
from math import radians
from typing import TYPE_CHECKING

from compas.geometry import angle_vectors
from compas.geometry import discrete_coons_patch
from compas.geometry import length_vector
from compas.geometry import subtract_vectors
from compas_singular.utilities import list_split

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh


__all__ = ["quadrangulate_mesh", "quadrangulate_faces", "quadrangulate_face", "discrete_coons_patch_mesh", "update_adjacent_face"]


def quadrangulate_mesh(mesh: Mesh, sources: list[int]) -> None:
    """Quadrangulate the faces of a mesh by adding edges from vertex sources.

    Parameters
    ----------
    mesh : Mesh
        A mesh to quadrangulate.
    sources : list
        A list of vertex keys to use as sources to add edges for quandrangulation.

    References
    ----------
    1. Oval et al., *Feature-based Topology Finding of Patterns for Shell Structures*. Automation in Construction. 2019.

    """
    sources_to_visit = sources[:]

    count = 1000
    while sources_to_visit and count:
        count -= 1

        vkey = sources_to_visit.pop()
        for fkey in mesh.vertex_faces(vkey):
            face_vertices = mesh.face_vertices(fkey)[:]
            if len(face_vertices) != 4:
                new_sources = quadrangulate_face(mesh, fkey, sources)
                for vkey in face_vertices:
                    if vkey in sources_to_visit:
                        sources_to_visit.remove(vkey)
                sources += new_sources
                sources_to_visit += new_sources


def quadrangulate_faces(mesh: Mesh, face_sources: dict[int, list[int]], max_faces: int | None = None) -> bool:
    """Quadrangulate the polygonal faces of a mesh, each from its own sources, in place.

    Returns False if stopped at ``max_faces``; the mesh should then be discarded.

    Parameters
    ----------
    mesh : Mesh
        The mesh to quadrangulate, modified in place.
    face_sources : dict
        Face keys pointing to the vertex keys that are sources for that face.
    max_faces : int, optional
        Stop once the mesh has more faces than this.

    Returns
    -------
    bool
        False if the propagation was stopped at ``max_faces``. The mesh is then
        left part-way and should be discarded.
    """
    created = set()
    todo = [fkey for fkey in mesh.faces() if len(mesh.face_vertices(fkey)) > 4]

    count = 1000
    while todo and count:
        count -= 1
        if max_faces is not None and mesh.number_of_faces() > max_faces:
            return False

        fkey = todo.pop()
        if not mesh.has_face(fkey) or len(mesh.face_vertices(fkey)) == 4:
            continue

        sources = [vkey for vkey in mesh.face_vertices(fkey) if vkey in face_sources.get(fkey, ()) or (vkey in created and is_straight_through(mesh, fkey, vkey))]
        new_sources = quadrangulate_face(mesh, fkey, sources)
        created.update(new_sources)
        todo += [nbr for vkey in new_sources for nbr in mesh.vertex_faces(vkey) if len(mesh.face_vertices(nbr)) > 4]

    return True


def is_straight_through(mesh: Mesh, fkey: int, vkey: int, tol: float = 0.5) -> bool:
    """Does face ``fkey`` pass straight through vertex ``vkey``, within ``tol`` degrees?"""
    face_vertices = mesh.face_vertices(fkey)
    i = face_vertices.index(vkey)
    xyz = mesh.vertex_coordinates(vkey)
    u = subtract_vectors(mesh.vertex_coordinates(face_vertices[i - 1]), xyz)
    v = subtract_vectors(mesh.vertex_coordinates(face_vertices[(i + 1) % len(face_vertices)]), xyz)
    if not length_vector(u) or not length_vector(v):
        return False
    return angle_vectors(u, v) > pi - radians(tol)


def quadrangulate_face(mesh: Mesh, fkey: int, sources: list[int]) -> list[int]:

    face_vertices = mesh.face_vertices(fkey)[:]

    # a face that passes through a vertex twice cannot be split into its sides
    if len(set(face_vertices)) != len(face_vertices):
        return []

    # differentiate sources and non sources
    sources = [vkey for vkey in face_vertices if vkey in sources]
    non_sources = [vkey for vkey in face_vertices if vkey not in sources]
    new_sources = []

    if len(non_sources) == 4:
        a, b, c, d = non_sources
        ab, bc, cd, da = list_split(face_vertices + face_vertices[:1], [face_vertices.index(vkey) for vkey in non_sources])
        # add missing vertices
        for i, edges in enumerate([[ab, cd], [bc, da]]):
            uv, wx = edges
            # all cases

            if len(uv) == len(wx):
                # no subdivision needed
                continue

            elif len(uv) == 2 and len(wx) != 2:
                # subdivide uv
                n = len(wx) - len(uv) + 1
                new_points = [mesh.edge_point(uv[0], uv[1], float(k) / float(n)) for k in range(n + 1)][1:-1]
                new_vertices = [mesh.add_vertex(attr_dict={xyz: value for xyz, value in zip(["x", "y", "z"], point)}) for point in new_points]
                new_sources += new_vertices
                if i == 0:
                    ab = [uv[0]] + new_vertices + [uv[-1]]
                elif i == 1:
                    bc = [uv[0]] + new_vertices + [uv[-1]]
                update_adjacent_face(mesh, uv[1], uv[0], list(reversed(new_vertices)))
            elif len(uv) != 2 and len(wx) == 2:
                # subdivide wx
                n = len(uv) - len(wx) + 1
                new_points = [mesh.edge_point(wx[0], wx[1], float(k) / float(n)) for k in range(n + 1)][1:-1]
                new_vertices = [mesh.add_vertex(attr_dict={xyz: value for xyz, value in zip(["x", "y", "z"], point)}) for point in new_points]
                new_sources += new_vertices
                if i == 0:
                    cd = [wx[0]] + new_vertices + [wx[-1]]
                elif i == 1:
                    da = [wx[0]] + new_vertices + [wx[-1]]
                # update adjacent faces
                update_adjacent_face(mesh, wx[1], wx[0], list(reversed(new_vertices)))
            elif len(uv) != 2 and len(wx) != 2 and len(uv) != len(wx):
                pass
                # apply Takayama's work
                # print('not implemented yet')

        mesh.delete_face(fkey)

        discrete_coons_patch_mesh(mesh, ab, bc, list(reversed(cd)), list(reversed(da)))

    else:
        pass

    return new_sources


def discrete_coons_patch_mesh(mesh: Mesh, ab: list[int], bc: list[int], dc: list[int], ad: list[int]) -> None:

    ab_xyz = [mesh.vertex_coordinates(vkey) for vkey in ab]
    bc_xyz = [mesh.vertex_coordinates(vkey) for vkey in bc]
    dc_xyz = [mesh.vertex_coordinates(vkey) for vkey in dc]
    ad_xyz = [mesh.vertex_coordinates(vkey) for vkey in ad]

    coons_vertices, coons_face_vertices = discrete_coons_patch(ab_xyz, bc_xyz, dc_xyz, ad_xyz)

    n = len(ab)
    m = len(bc)

    vertex_index_map = {}

    for i, vkey in enumerate(ad):
        vertex_index_map[i] = vkey

    for i, vkey in enumerate(bc):
        vertex_index_map[m * (n - 1) + i] = vkey

    for i, vkey in enumerate(ab):
        vertex_index_map[i * m] = vkey

    for i, vkey in enumerate(dc):
        vertex_index_map[m - 1 + i * m] = vkey

    max_vkey = max(list(mesh.vertices()))

    for i, vertex in enumerate(coons_vertices):
        if i not in vertex_index_map:
            max_vkey += 1
            vertex_index_map[i] = max_vkey
            mesh.add_vertex(max_vkey, attr_dict={xyz: value for xyz, value in zip(["x", "y", "z"], vertex)})

    for face in coons_face_vertices:
        mesh.add_face(list(reversed([vertex_index_map[vkey] for vkey in face])))


def update_adjacent_face(mesh: Mesh, u: int, v: int, vertices_uv: list[int]) -> None:
    fkey = mesh.halfedge[u][v]
    if fkey is not None:
        face_vertices = mesh.face_vertices(fkey)[:]
        i = face_vertices.index(v)
        for vkey in reversed(vertices_uv):
            face_vertices.insert(i, vkey)
        mesh.delete_face(fkey)
        mesh.add_face(face_vertices, fkey)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    from compas_viewer import Viewer

    from compas_singular.datastructures.mesh.mesh import Mesh

    viewer = Viewer()

    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
        [3, 0, 0],
        [3, 1, 0],
        [0, 1, 0],
        [0, 0.5, 0],
        [0, 0.25, 0],
        [4, 0, 0],
        [4, 1, 0],
    ]

    faces = [[0, 1, 2, 3, 4, 5, 6, 7], [3, 8, 9, 4]]

    sources = [1, 2, 6, 7]

    mesh = Mesh.from_vertices_and_faces(vertices, faces)

    viewer.scene.add(mesh.copy())

    quadrangulate_face(mesh, 0, sources)
    quadrangulate_mesh(mesh, sources)

    viewer.scene.add(mesh)
    viewer.show()
