from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import pi
from typing import TYPE_CHECKING

from compas.geometry import centroid_points_weighted
from compas.geometry import subtract_vectors
from compas_singular.datastructures import mesh_move_by
from compas_singular.geometry import circle_evaluate

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = ["interpolation_layout_two_meshes", "interpolation_layout_primary", "interpolation_layout_secondary"]


def interpolation_layout_two_meshes(interpolated_meshes: dict[QuadMesh, tuple[float, float]], dx: float, dy: float) -> None:
    position_to_meshes = {}
    for mesh, (a, b) in interpolated_meshes.items():
        d = a - b
        if d in position_to_meshes:
            position_to_meshes[d].append(mesh)
        else:
            position_to_meshes[d] = [mesh]

    for d, meshes in position_to_meshes.items():
        for j, mesh in enumerate(meshes):
            centre = mesh.centroid() if mesh.area() != 0 else mesh.vertex_centroid()
            mesh_move_by(mesh, subtract_vectors([d * dx, -j * dy, 0.0], centre))


def interpolation_layout_primary(meshes: list[QuadMesh], interpolated_meshes: dict[QuadMesh, list[float]], radius: float) -> None:
    # sort input meshes and interpolating meshes
    ext_meshes = meshes
    int_meshes = []
    for mesh, distances in interpolated_meshes.items():
        if mesh not in ext_meshes:
            int_meshes.append(mesh)

    mesh_to_xyz = {mesh: [0.0, 0.0, 0.0] for mesh in ext_meshes + int_meshes}

    # map input meshes along a circle
    n = len(ext_meshes)
    for i, mesh in enumerate(ext_meshes):
        mesh_to_xyz[mesh] = circle_evaluate(2.0 * pi * i / n, radius / 2.0)
    ext_points = [mesh_to_xyz[mesh] for mesh in ext_meshes]

    # map interpolating meshes in the circle
    for mesh in int_meshes:
        weights = [ext_meshes[i].number_of_strips() - d for i, d in enumerate(interpolated_meshes[mesh])]
        mesh_to_xyz[mesh] = centroid_points_weighted(ext_points, weights)

    # move meshes
    for mesh in interpolated_meshes.keys():
        centre = mesh.centroid() if mesh.area() != 0 else mesh.vertex_centroid()
        mesh_move_by(mesh, subtract_vectors(mesh_to_xyz[mesh], centre))


def interpolation_layout_secondary(interpolated_meshes: dict[QuadMesh, list[float]], radius: float) -> None:
    cluster_meshes = {tuple(distance): [] for distance in interpolated_meshes.values()}
    for mesh, distance in interpolated_meshes.items():
        cluster_meshes[tuple(distance)].append(mesh)

    for meshes in cluster_meshes.values():
        n = len(meshes)
        if n > 1:
            for i, mesh in enumerate(meshes):
                xyz = circle_evaluate(2.0 * pi * i / n + pi / 2.0, radius / 2.0)
                centre = mesh.centroid() if mesh.area() != 0 else mesh.vertex_centroid()
                mesh_move_by(mesh, subtract_vectors(xyz, centre))


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass
