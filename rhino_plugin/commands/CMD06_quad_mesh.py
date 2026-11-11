#! python3

# r: compas
# r: pydantic
# r: compas_singular

"""Step 6 -- densify the coarse layout into the quad mesh.

Input: the session's layout (and field), the boundaries on InputBoundaries. Output: the dense mesh in the session, drawn on QuadMesh::Dense.
"""
import rhinoscriptsyntax as rs

from compas_singular.datastructures import coarse_edges_to_curves
from compas_singular.datastructures import snap_corners_to_walls
from compas_singular.framefield.quality import mesh_quality
from compas_singular.rhino.helpers import clear_layer
from compas_singular.rhino.helpers import read_boundaries
from compas_singular.rhino.helpers import read_boundary_loops
from compas_singular.rhino.project import ROOT
from compas_singular.rhino.project import get_settings
from compas_singular.rhino.project import layer_path
from compas_singular.rhino.project import read_layout
from compas_singular.rhino.project import resolve_densities
from compas_singular.rhino.project import resolve_field_symmetry
from compas_singular.rhino.project import resolve_relax
from compas_singular.rhino.project import resolve_spacing
from compas_singular.rhino.project import set_settings
from compas_singular.rhino.session import RhinoSession

#: Wall sampling, as a multiple of the background spacing.
WALL_SAMPLING_FACTOR = 0.25


def main():
    settings = get_settings()
    coarse = read_layout()
    # None on the skeleton route.
    field = RhinoSession.current().field

    spacing = resolve_spacing(settings)
    outer, inners, guides, _point_features = read_boundaries(spacing=spacing)
    if field is not None:
        why = field.mismatch(outer, inners, guides=guides,
                             mode=settings["guide_alignment"],
                             target_length=settings["triangulation_spacing"],
                             relax=resolve_relax(settings, guides),
                             symmetry=resolve_field_symmetry(settings))
        if why:
            print("field ignored: {} since it was solved. Re-run step 3.".format(why))
            field = None

    options = (
        ("Field", "Off", "On"),
        ("Boundary_curvature", "False", "True"),
        ("Skeleton_curvature", "False", "True")
    )

    option_defaults = [False, True, True]

    if coarse.attributes['decomposition_type'] == 'field':
        option_defaults[0] = True

    options = rs.GetBoolean("Densify the coarse layout.", options, option_defaults)
    if options is None:
        print("Command cancelled. Nothing changed.")
        return
    field_aware, boundary_curvature, skeleton_curvature = options

    settings["field_aware"] = field_aware
    set_settings(settings)

    if field is None and settings["field_aware"]:
        print("no field available -- patch interiors will be plain Coons. This "
              "is expected on the skeleton route.")

    wall_sampling = spacing * WALL_SAMPLING_FACTOR
    outer_loop, inner_loops = read_boundary_loops(wall_sampling)
    loops = [outer_loop] + inner_loops

    snapped, worst = snap_corners_to_walls(coarse, loops=loops)
    if snapped:
        print("snapped {} boundary corner(s) onto their wall, worst {:.4f}".format(
            snapped, worst))

    edges_to_curves, tally = coarse_edges_to_curves(
        coarse, loops=loops, polylines=coarse.shape_polylines())

    resolve_densities(coarse, settings)

    if settings["field_aware"] and field is not None:
        dense = coarse.quad_mesh(boundary_curvature=boundary_curvature, skeleton_curvature=skeleton_curvature, field=field)
    else:
        dense = coarse.quad_mesh(boundary_curvature=boundary_curvature, skeleton_curvature=skeleton_curvature)

    clear_layer(rs.AddLayer("QuadMesh", parent=ROOT), clean_sublayers=True)
    layer = layer_path("Dense")
    session = RhinoSession.current()
    session.coarse = coarse
    session.dense = dense

    session.record("Quad mesh")

    print("mesh: {} faces from {} patch(es), field_aware={}".format(
        dense.number_of_faces(), coarse.number_of_faces(),
        settings["field_aware"] and field is not None))

    quality = mesh_quality(dense)
    print("quality: min angle {:.1f}, max angle {:.1f}, aspect {:.2f}, "
          "{:.1%} of corners below {:.0f} deg".format(
              quality["min_angle"], quality["max_angle"], quality["aspect_max"],
              quality["share_below"], quality["low_angle"]))

    print("coarse edges: {}".format(dict(tally)))
    if tally.get("wall_missed"):
        print("  {} boundary edge(s) did NOT get their wall arc -- on a hole "
              "this is what makes one circle come out round and the next a "
              "polygon".format(tally["wall_missed"]))

    print("drawn on '{}'".format(layer))
    print("note: re-running this step regenerates from the coarse layout and "
          "overwrites any hand edits made in CMD_edit_quad_mesh.")
    print("next: CMD_smooth to relax the mesh, CMD_dual for "
          "the dual mesh, or CMD_edit_quad_mesh to hand-edit it.")


if __name__ == "__main__":
    main()
