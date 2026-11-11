#! python3

# r: compas
# r: pydantic
# r: compas_singular

"""Utility -- export one picked mesh to a JSON file.

Input: any mesh object in the document. Output: a JSON file ``Mesh.load_from_json`` reads back, session attributes included.
"""
import os

import compas_rhino as cr
import rhinoscriptsyntax as rs

from compas_singular.datastructures import Mesh
from compas_singular.rhino.helpers import mesh_from_rhino
from compas_singular.rhino.session import RhinoSession


def main():
    guid = rs.GetObject("Pick a mesh to export", filter=rs.filter.mesh, preselect=True)
    if not guid:
        print("Cancelled -- nothing exported.")
        return

    session = RhinoSession.current()
    found = session.item_of(guid)
    if found:
        mesh, source = found[1], "the session's {} mesh, attributes included".format(found[0])
    else:
        mesh = mesh_from_rhino(cr.objects.find_object(guid).Geometry, cls=Mesh)
        source = "its Rhino geometry"

    doc = session.doc
    folder = os.path.dirname(doc.Path) if doc.Path else None
    default = (rs.ObjectName(guid) or os.path.splitext(doc.Name or "mesh")[0] + "_mesh") + ".json"
    path = rs.SaveFileName("Export the mesh", "JSON (*.json)|*.json||", folder, default)
    if not path:
        print("Cancelled -- nothing exported.")
        return

    mesh.save_to_json(path)
    print("{} ({} vertices, {} faces) exported from {} to {}".format(
        type(mesh).__name__, mesh.number_of_vertices(), mesh.number_of_faces(), source, path))
    print("load it in a script with: Mesh.load_from_json(r'{}')".format(path))


if __name__ == "__main__":
    main()
