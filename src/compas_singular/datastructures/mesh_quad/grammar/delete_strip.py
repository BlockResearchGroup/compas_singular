"""Delete a strip, welding its two sides and placing merged vertices by a fixed rule; plus the queries that price one.

Pre-splitting to save a boundary is the caller's policy, not done here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from compas.datastructures.mesh.operations.substitute import mesh_substitute_vertex_in_faces
from compas.geometry import centroid_points
from compas.itertools import pairwise
from compas.topology import connected_components

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = [
    "delete_strip",
    "delete_strips",
    "collateral_strip_deletions",
    "total_boundary_deletions",
    "strips_to_split_to_prevent_boundary_collapse",
]


def delete_strips(mesh: QuadMesh, skeys: list[int]) -> None:
    """Delete several strips.

    Strip keys are re-checked as the deletions go, because deleting one strip can
    delete others with it -- see ``collateral_strip_deletions``.
    """
    for skey in skeys:
        if skey in list(mesh.strips()):
            delete_strip(mesh, skey)


def delete_strip(mesh: QuadMesh, skey: int) -> dict[int, int]:
    """Delete the strip ``skey``, welding the faces either side together.

    Consumed strips go with it and ``face_pole`` is repointed; vertices move.

    Parameters
    ----------
    mesh : QuadMesh
        A quad mesh.
    skey : hashable
        A strip key.

    Returns
    -------
    dict
        ``{old vertex: the vertex it was merged into}``. Empty if there was no
        such strip.
    """
    if skey not in list(mesh.strips()):
        return {}

    old_boundary_vertices = list(mesh.vertices_on_boundaries())

    # get strip data
    strip_edges = mesh.strip_edges(skey)
    strip_faces = mesh.strip_faces(skey)

    # collateral strip deletions
    collateral_deleted_strips = []
    for skey_2 in mesh.strips():
        if skey_2 == skey:
            continue
        faces_2 = mesh.strip_faces(skey_2)
        # ``faces_2`` non-empty: ``all([])`` is True, so a strip with no faces -- one
        # edge between two polygons -- would be "deleted" by every deletion.
        if faces_2 and all([fkey in strip_faces for fkey in faces_2]):
            collateral_deleted_strips.append(skey_2)

    # connect the vertices of the edges of the strip to delete to
    # get the disconnect parts of vertices to merge
    vertices = set([i for edge in strip_edges for i in edge])
    # maps between old and new indices
    old_to_new = {vkey: i for i, vkey in enumerate(vertices)}
    new_to_old = {i: vkey for i, vkey in enumerate(vertices)}
    # adjacency
    adjacency = {i: [] for i in new_to_old}
    for u, v in strip_edges:
        adjacency[old_to_new[u]].append(old_to_new[v])
        adjacency[old_to_new[v]].append(old_to_new[u])
    # disconnected parts
    parts = connected_components(adjacency)

    # delete strip faces
    for fkey in strip_faces:
        mesh.delete_face_in_strips(fkey)
    for fkey in strip_faces:
        mesh.delete_face(fkey)

    old_vkeys_to_new_vkeys = {}

    # merge strip edge vertices that are connected
    for part in parts:
        # move back from part vertices to mesh vertices
        vertices = [new_to_old[vkey] for vkey in part]

        # skip adding a vertex if all vertices of the part are disconnected
        if any(mesh.is_vertex_connected(vkey) for vkey in vertices):
            # get position based on disconnected vertices that used to be on
            # the boundary if any
            if any(not mesh.is_vertex_connected(vkey) for vkey in vertices):
                points = [mesh.vertex_coordinates(vkey) for vkey in vertices if not mesh.is_vertex_connected(vkey)]
            # or based on old boundary vertices if any
            elif any(vkey in old_boundary_vertices for vkey in vertices):
                points = [mesh.vertex_coordinates(vkey) for vkey in vertices if vkey in old_boundary_vertices]
            else:
                points = [mesh.vertex_coordinates(vkey) for vkey in vertices]

            # new vertex
            x, y, z = centroid_points(points)
            new_vkey = mesh.add_vertex(attr_dict={"x": x, "y": y, "z": z})
            old_vkeys_to_new_vkeys.update({old_vkey: new_vkey for old_vkey in vertices})

            # replace the old vertices
            for old_vkey in vertices:
                mesh.substitute_vertex_in_strips(old_vkey, new_vkey)
                mesh_substitute_vertex_in_faces(mesh, old_vkey, new_vkey, mesh.vertex_faces(old_vkey))

        # delete the old vertices
        for old_vkey in vertices:
            mesh.delete_vertex(old_vkey)

    # delete data of deleted strip and collateral deleted strips
    del mesh.attributes["strips"][skey]
    for skey_2 in collateral_deleted_strips:
        del mesh.attributes["strips"][skey_2]

    if "face_pole" in mesh.attributes:
        for fkey, pole in mesh.attributes["face_pole"].items():
            if fkey in mesh.attributes["face_pole"]:
                if pole == mesh.attributes["face_pole"][fkey]:
                    if pole in old_vkeys_to_new_vkeys:
                        mesh.attributes["face_pole"][fkey] = old_vkeys_to_new_vkeys[pole]

    return old_vkeys_to_new_vkeys


def collateral_strip_deletions(mesh: QuadMesh, skeys: list[int]) -> list[int]:
    """The strips that deleting ``skeys`` would delete as well. Mutates nothing."""
    deleted_fkeys = [fkey for skey in skeys for fkey in mesh.strip_faces(skey)]
    # A strip with no faces at all is not collateral -- see ``delete_strip``.
    return [skey for skey in mesh.strips() if skey not in skeys and mesh.strip_faces(skey) and all([fkey in deleted_fkeys for fkey in mesh.strip_faces(skey)])]


def total_boundary_deletions(mesh: QuadMesh, skeys: list[int]) -> list[list[int]]:
    """The boundaries that deleting ``skeys`` would collapse. Mutates nothing."""
    deleted_strips = list(skeys) + list(collateral_strip_deletions(mesh, skeys))
    deleted_boundaries = []
    deleted_edges = set([edge for skey in deleted_strips for edge in mesh.strip_edges(skey)])
    for boundary in mesh.boundaries():
        edges = [(u, v) for u, v in pairwise(boundary + boundary[:1])]
        if all([(u, v) in deleted_edges or (v, u) in deleted_edges for u, v in edges]):
            deleted_boundaries.append(boundary)
    return deleted_boundaries


def strips_to_split_to_prevent_boundary_collapse(mesh: QuadMesh, skeys: list[int]) -> dict[int, int] | None:
    """What to split before deleting ``skeys`` so every boundary keeps three edges (thesis 5.3.2). Mutates nothing.

    Returns
    -------
    dict or None
        ``None`` if the collapse cannot be prevented, ``{}`` if nothing is at
        risk, else ``{skey: n}`` to refine first. Do not conflate ``None`` and ``{}``.
    """
    to_split = {}
    for boundary in mesh.boundaries():
        non_deleted_strips = [mesh.edge_strip((u, v)) for u, v in pairwise(boundary + boundary[:1]) if mesh.edge_strip((u, v)) not in skeys]

        if len(non_deleted_strips) == 0:
            # Nothing survives on this boundary, so nothing can be refined to
            # save it. ``None``, NOT ``{}`` -- see the tri-state above.
            return None

        elif len(non_deleted_strips) == 1:
            skey = non_deleted_strips[0]
            if skey in to_split:
                if to_split[skey] == 3:
                    break
                if to_split[skey] == 2:
                    to_split[skey] = 3
            else:
                to_split[skey] = 3

        elif len(non_deleted_strips) == 2:
            for skey in non_deleted_strips:
                if skey in to_split:
                    break
            to_split.update({skey: 2 for skey in non_deleted_strips})

    return to_split
