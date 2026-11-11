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

guids = rs.GetObjects(message="Select a mesh.", filter=rs.filter.mesh)

if guids:
    model = BlockModel()
    for guid in guids:
        mesh = mesh_to_compas(cr.objects.find_object(guid).Geometry)
        model.add_block_from_mesh(mesh)
        scene.add(mesh)

    scene.draw()

    # Same default path as CM_Model_blocks > Json, so Enter there picks this file up.
    default = pathlib.Path.home() / "blockmodel.json"
    path = rs.SaveFileName("Save blocks", "JSON (*.json)|*.json||", str(default.parent), default.name)
    if path:
        model.to_json(path)
        print(f"Saved {len(list(model.blocks()))} blocks to {path}")
