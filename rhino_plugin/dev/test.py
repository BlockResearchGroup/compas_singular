#! python 3
# venv: brg-csd
# r: compas_masonry>=0.4.1

# Offsets every face of a Rhino mesh into a block and saves them as a
# compas_dem BlockModel, which COMPAS-Masonry loads with
# CM_Model_blocks > Json. The env is the plugin's own (brg-csd) so the
# JSON is written by the same compas_dem that reads it.

import pathlib

from compas.geometry import Point
from compas.datastructures import Mesh
from compas_dem.models import BlockModel
from compas_rhino.conversions import mesh_to_compas
import rhinoscriptsyntax as rs
import compas_rhino as cr

from compas.scene import Scene
scene = Scene()

guid = rs.GetObject(message="Select a mesh.", filter=rs.filter.mesh)
if guid:
    thickness = rs.GetReal("Block thickness", 1.0, 0.0)
if guid and thickness:
    mesh = mesh_to_compas(cr.objects.find_object(guid).Geometry)

    vertices, faces = mesh.to_vertices_and_faces()
    mesh = Mesh.from_vertices_and_faces(vertices, faces)
    mesh.weld()
    mesh.remove_duplicate_vertices()

    model = BlockModel()
    for fkey in mesh.faces():
        vkeys = mesh.face_vertices(fkey)

        top_vertices = []
        bottom_vertices = []
        for vkey in vkeys:
            normal = mesh.vertex_normal(vkey)
            offset = [axis * 0.5 * thickness for axis in normal]
            vertex = Point(*mesh.vertex_coordinates(vkey))
            top_vertices.append(vertex.translated(offset))
            bottom_vertices.append(vertex.translated([-axis for axis in offset]))

        n = len(vkeys)
        top_face = list(range(n))
        # Reversed so the bottom face points out of the block, like the top and sides.
        bottom_face = [n + i for i in reversed(range(n))]
        side_faces = [[i, n + i, n + (i + 1) % n, (i + 1) % n] for i in range(n)]

        block = Mesh.from_vertices_and_faces(top_vertices + bottom_vertices, [top_face, bottom_face] + side_faces)
        model.add_block_from_mesh(block)
        scene.add(block)

    scene.draw()

    # Same default path as CM_Model_blocks > Json, so Enter there picks this file up.
    default = pathlib.Path.home() / "blockmodel.json"
    path = rs.SaveFileName("Save blocks", "JSON (*.json)|*.json||", str(default.parent), default.name)
    if path:
        model.to_json(path)
        print(f"Saved {len(list(model.blocks()))} blocks to {path}")
