from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any

from compas.geometry import cross_vectors
from compas.geometry import dot_vectors
from compas.geometry import length_vector
from compas.geometry import length_vector_sqrd
from compas.geometry import normalize_vector
from compas.geometry import scale_vector
from compas.geometry import subtract_vectors
from compas.geometry import sum_vectors
from compas.itertools import pairwise
from compas.tolerance import TOL

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh

__all__ = [
    "mesh_move_by",
    "mesh_move_vertices_by",
    "mesh_move_vertex_to",
    "is_face_degenerate",
    "trimesh_face_circle",
    "mesh_weld",
    "meshes_join",
    "meshes_join_and_weld",
]


def mesh_move_by(mesh: Mesh, vector: list[float]) -> None:
    """Move a mesh by a vector.

    Parameters
    ----------
    mesh : Mesh
        A mesh.
    vector : list
        An XYZ vector.
    """

    for vkey in mesh.vertices():
        mesh.vertex[vkey]["x"] += vector[0]
        mesh.vertex[vkey]["y"] += vector[1]
        mesh.vertex[vkey]["z"] += vector[2]


def mesh_move_vertices_by(mesh: Mesh, key_to_vector: dict[int, list[float]]) -> None:
    """Move mesh vertices by different vectors.

    Parameters
    ----------
    mesh : Mesh
        A mesh.
    key_to_vector : dict
        A dictionary of vertex keys pointing to vectors.
    """

    for vkey, vector in key_to_vector.items():
        mesh.vertex[vkey]["x"] += vector[0]
        mesh.vertex[vkey]["y"] += vector[1]
        mesh.vertex[vkey]["z"] += vector[2]


def mesh_move_vertex_to(mesh: Mesh, point: list[float], vkey: int) -> None:
    """Move a mesh vertex to a point.

    Parameters
    ----------
    mesh : Mesh
        A mesh.
    point : list
        An XYZ point.
    vkey : hashable
        A vertex key.
    """

    mesh.vertex[vkey]["x"] = point[0]
    mesh.vertex[vkey]["y"] = point[1]
    mesh.vertex[vkey]["z"] = point[2]


#: Relative height below which a triangle counts as flat. The test compares the
#: doubled area against the longest edge SQUARED, so the quantity is the
#: triangle's height as a fraction of its own size -- dimensionless, and the same
#: verdict whether the model is drawn in millimetres or in metres. Double
#: precision carries ~16 digits, so 1e-12 is four orders of magnitude above the
#: rounding noise it is meant to catch and many orders below any sliver a
#: Delaunay triangulation produces on purpose.
FLATNESS = 1e-12


def is_face_degenerate(a: list[float], b: list[float], c: list[float], tol: float = FLATNESS) -> bool:
    """Whether a triangle is flat (collinear or repeated corners), by a scale-relative height test.

    Parameters
    ----------
    a, b, c : list
        The three corners, as ``[x, y, z]``.
    tol : float, optional
        Relative height below which the triangle counts as flat.

    Returns
    -------
    bool
    """
    ab = subtract_vectors(b, a)
    ac = subtract_vectors(c, a)
    bc = subtract_vectors(c, b)
    longest = max(length_vector(ab), length_vector(ac), length_vector(bc))
    return length_vector(cross_vectors(ab, ac)) <= tol * longest**2


def trimesh_face_circle(mesh: Mesh, fkey: int) -> tuple[list[float], float, list[float]] | None:
    """Circumcircle of a triangular face, in closed form so the centre is bit-reproducible.

    Parameters
    ----------
    mesh : Mesh
        A mesh.
    fkey : hashable
        A face key.

    Returns
    -------
    tuple or None
        ``(center, radius, normal)`` -- centre first, which is the ordering
        compas_singular relies on -- or None if the face is not a triangle.

    Raises
    ------
    ValueError
        If the triangle is flat. A flat triangle has no circumcircle, and the
        closed form below divides by its doubled squared area.
        ``boundary_triangulation`` deletes flat faces before any caller here
        gets to walk them, so reaching this means one was built some other way.
    """
    vertices = mesh.face_vertices(fkey)
    if len(vertices) != 3:
        return None

    u, v, w = vertices
    a = mesh.vertex_coordinates(u)
    b = mesh.vertex_coordinates(v)
    c = mesh.vertex_coordinates(w)

    ab = subtract_vectors(b, a)
    cb = subtract_vectors(b, c)
    ba = subtract_vectors(a, b)
    ca = subtract_vectors(a, c)
    ac = subtract_vectors(c, a)
    bc = subtract_vectors(c, b)

    normal = normalize_vector(cross_vectors(ab, ac))
    d = 2 * length_vector_sqrd(cross_vectors(ba, cb))
    if d == 0.0 or is_face_degenerate(a, b, c):
        raise ValueError("face {} of the mesh is flat -- its three vertices {}, {} and {} are collinear, so it has no circumcircle".format(fkey, a, b, c))
    A = length_vector_sqrd(cb) * dot_vectors(ba, ca) / d
    B = length_vector_sqrd(ca) * dot_vectors(ab, cb) / d
    C = length_vector_sqrd(ba) * dot_vectors(ac, bc) / d
    center = sum_vectors([scale_vector(a, A), scale_vector(b, B), scale_vector(c, C)])
    radius = length_vector(subtract_vectors(a, center))

    return center, radius, normal


def mesh_weld(mesh: Mesh, precision: int | None = None, cls: type | None = None) -> Any:
    """Weld vertices within a precision into a new mesh, dropping faces that collapse below three vertices.

    Parameters
    ----------
    mesh : Mesh
        A mesh.
    precision : int, optional
        Decimals of the geometric key used to match vertices.
        Defaults to the COMPAS tolerance precision.
    cls : type, optional
        The class of the resulting mesh. Defaults to the class of ``mesh``.

    Returns
    -------
    Mesh
        A new welded mesh.
    """
    if cls is None:
        cls = type(mesh)

    key_xyz = {key: mesh.vertex_coordinates(key) for key in mesh.vertices()}
    gkey_key = {TOL.geometric_key(xyz, precision): key for key, xyz in key_xyz.items()}
    gkey_index = {gkey: index for index, gkey in enumerate(gkey_key)}

    vertices = [key_xyz[key] for gkey, key in gkey_key.items()]
    faces = [[gkey_index[TOL.geometric_key(key_xyz[key], precision)] for key in mesh.face_vertices(fkey)] for fkey in mesh.faces()]

    faces[:] = [[u for u, v in pairwise(face + face[:1]) if u != v] for face in faces]
    faces[:] = [face for face in faces if len(face) > 2]  # discard degenerate faces

    return cls.from_vertices_and_faces(vertices, faces)


def meshes_join(meshes: list[Mesh], cls: type | None = None) -> Any:
    """Join meshes without welding, returning a new mesh.

    Parameters
    ----------
    meshes : list
        A list of meshes.
    cls : type, optional
        The class of the resulting mesh. Defaults to the class of the first mesh.

    Returns
    -------
    Mesh
        A new joined mesh.
    """
    if cls is None:
        cls = type(meshes[0])

    vertices = []
    faces = []
    for mesh in meshes:
        key_index = {key: len(vertices) + i for i, key in enumerate(mesh.vertices())}
        vertices += [mesh.vertex_coordinates(key) for key in mesh.vertices()]
        faces += [[key_index[key] for key in mesh.face_vertices(fkey)] for fkey in mesh.faces()]

    return cls.from_vertices_and_faces(vertices, faces)


def meshes_join_and_weld(meshes: list[Mesh], precision: int | None = None, cls: type | None = None) -> Any:
    """Join and weld meshes within a precision distance, returning a new mesh.

    Parameters
    ----------
    meshes : list
        A list of meshes.
    precision : int, optional
        Decimals of the geometric key used to match vertices.
        Defaults to the COMPAS tolerance precision.
    cls : type, optional
        The class of the resulting mesh. Defaults to the class of the first mesh.

    Returns
    -------
    Mesh
        A new joined and welded mesh.
    """
    return mesh_weld(meshes_join(meshes, cls=cls), precision=precision)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass
