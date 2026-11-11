#! python3
# r: compas
# r: pydantic
# r: compas_singular

"""Step 8 -- the dual of a dense quad mesh.

Input: a mesh picked from TopologyProblem::QuadMesh or a sublayer. Output: its dual on QuadMesh::Dual.
"""
import compas_rhino as cr
import rhinoscriptsyntax as rs

from compas_singular.algorithms.dual_mesh import dual_mesh
from compas_singular.rhino.helpers import bake_mesh
from compas_singular.rhino.helpers import clear_layer
from compas_singular.rhino.helpers import mesh_from_rhino


def quad_mesh_filter(rhobj, geometry, component_index):
    """Pick filter: only objects on QuadMesh or its sublayers."""
    layer = rs.ObjectLayer(rhobj)
    quad_layer = rs.LayerName("QuadMesh", fullpath=True)
    return layer == quad_layer or rs.IsLayerChildOf(quad_layer, layer)


def main():
    if not rs.IsLayer("QuadMesh"):
        raise RuntimeError("No dense quad mesh has been generated yet. The layer does not exist yet.")

    mesh_id = rs.GetObject(message="Pick a dense quad mesh to smoothen from the layer QuadMesh and its sublayers",
                           filter=rs.filter.mesh, preselect=False, select=False,
                           custom_filter=quad_mesh_filter, subobjects=False)
    if not mesh_id:
        print("Cancelled -- nothing was drawn or changed.")
        return
    mesh = mesh_from_rhino(cr.objects.find_object(mesh_id).Geometry)
    dual = dual_mesh(mesh, redistribute=True)

    clear_layer("Dual", clean_sublayers=True)

    rs.AddLayer("Dual", parent="QuadMesh")
    bake_mesh(dual, "Dual")

    print("dual: {} faces from {} primal vertices, baked to 'QuadMesh::Dual'".format(
        dual.number_of_faces(), mesh.number_of_vertices()))
    print("next: this is the end of the pipeline for this mesh -- re-run CMD_quad_mesh "
          "or edit in CMD_edit_quad_mesh if further changes are needed, then dual again.")


if __name__ == "__main__":
    main()
