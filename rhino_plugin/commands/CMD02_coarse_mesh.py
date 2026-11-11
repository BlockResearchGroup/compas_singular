#! python3

# r: compas
# r: pydantic
# r: compas_singular

"""Step 3 -- generate the coarse layout, by the skeleton or the frame-field route.

Input: the boundaries on InputBoundaries and the settings. Output: the layout (and field) in the session, drawn on Skeleton.
"""
import rhinoscriptsyntax as rs

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.datastructures.mesh_quad_coarse.coarse_curves import coarse_edges_to_curves
from compas_singular.datastructures.mesh_quad_coarse.coarse_curves import snap_corners_to_walls
from compas_singular.framefield.field_decomposition import FieldDecomposition
from compas_singular.rhino.helpers import clear_layer
from compas_singular.rhino.helpers import read_boundaries
from compas_singular.rhino.helpers import read_boundary_loops
from compas_singular.rhino.project import get_settings
from compas_singular.rhino.project import layer_path
from compas_singular.rhino.project import resolve_field_symmetry
from compas_singular.rhino.project import resolve_relax
from compas_singular.rhino.project import resolve_spacing
from compas_singular.rhino.session import RhinoSession
from compas_singular.symmetry import Domain

#: Wall sampling, as a multiple of the background spacing.
WALL_SAMPLING_FACTOR = 0.25

#: Where point features go on the frame-field route, which does not use them.
IGNORED_POINTS_LAYER = "PointFeaturesIgnored"


def coarse_from_field(settings, outer, inners, guides):
    """Frame-field decomposition. ``(coarse_mesh, skeleton polylines, field)``."""
    decomposition = FieldDecomposition.from_boundary(
        outer, inner_boundaries=inners, guides=guides,
        mode=settings["guide_alignment"],
        target_length=settings["triangulation_spacing"],
        relax=resolve_relax(settings, guides),
        symmetry=resolve_field_symmetry(settings))
    print(decomposition.field.report())
    group = decomposition.symmetry
    if group is None or group.trivial:
        print("symmetry: none detected -- the layout is under no obligation to have any")
    else:
        print("symmetry: order {} about ({:.3f}, {:.3f}) -- {}".format(
            len(group), group.centre[0], group.centre[1], ", ".join(group.names())))

    skeleton = decomposition.decomposition_polylines()
    coarse_mesh = decomposition.coarse_mesh()
    coarse_mesh.attributes["route"] = "field"

    return coarse_mesh, skeleton, decomposition.get_field()


def coarse_from_skeleton(settings, outer, inners, line_features, point_features):
    """Medial-axis decomposition. ``(coarse_mesh, skeleton polylines, None)``.

    Note: guides are passed as polyline features, so here they cut the domain.
    """
    decomposition = SkeletonDecomposition.from_boundary(
        outer,
        inner_boundaries=inners,
        polyline_features=line_features,
        point_features=point_features,
        target_length=settings["triangulation_spacing"])

    coarse_mesh = decomposition.coarse_mesh()
    skeleton = decomposition.polylines

    coarse_mesh.attributes["route"] = "skeleton"

    return coarse_mesh, skeleton, None


def remove_point_features():
    """Move every point feature off ``PointFeatures``, and say so. Returns ``[]``."""
    ids = rs.ObjectsByLayer(layer_path("PointFeatures")) or []
    if not ids:
        return []
    layer = layer_path("Problem") + "::" + IGNORED_POINTS_LAYER
    if not rs.IsLayer(layer):
        rs.AddLayer(IGNORED_POINTS_LAYER, color=(90, 90, 90), parent=layer_path("Problem"))
    rs.ObjectLayer(ids, layer)
    print("point features are NOT an input of the FrameField method: {} point(s) "
          "moved from '{}' to '{}', not deleted. Move them back to use them on "
          "the Skeleton route.".format(len(ids), layer_path("PointFeatures"), layer))
    return []


def main():
    settings = get_settings()
    spacing = resolve_spacing(settings)
    outer, inners, guides, point_features = read_boundaries(spacing=spacing)
    wall_sampling = spacing * WALL_SAMPLING_FACTOR

    coarse_mesh, skeleton, field = None, None, None
    while True:
        mode = rs.GetString(message="Choose coarse mesh generation", defaultString="Skeleton", strings=["FrameField", "Skeleton", "Exit"])
        mode = (mode or "Exit").lower()

        if mode == "skeleton":
            coarse_mesh, skeleton, field = coarse_from_skeleton(
                settings, outer, inners, guides, point_features)
            break
        elif mode == "framefield":
            if point_features:
                point_features = remove_point_features()
            coarse_mesh, skeleton, field = coarse_from_field(settings, outer, inners, guides)
            break
        elif mode == "exit":
            break
        else:
            print("Unrecognised mode: {}".format(mode))

    if coarse_mesh and skeleton:
        rs.EnableRedraw(False)
        clear_layer("Skeleton", clean_sublayers=True)

        outer_loop, inner_loops = read_boundary_loops(wall_sampling)
        snapped, worst = snap_corners_to_walls(
            coarse_mesh, loops=[outer_loop] + inner_loops)
        if snapped:
            print("snapped {} boundary corner(s) onto their wall, worst {:.4f}".format(
                snapped, worst))

        curves, tally = coarse_edges_to_curves(
            coarse_mesh, loops=[outer_loop] + inner_loops, polylines=skeleton)
        print("coarse edges: {}".format(tally))
        if tally["chord"]:
            print("  {} edge(s) densify as a straight chord -- interior edges with "
                  "no traced branch, which is expected, or a boundary corner that "
                  "is not on a wall, which is not".format(tally["chord"]))
        if tally.get("wall_missed"):
            print("  {} boundary edge(s) did NOT get their wall arc -- on a hole "
                  "this is what makes one circle come out round and the next a "
                  "polygon".format(tally["wall_missed"]))

        coarse_mesh.set_edges_to_curves(curves)
        coarse_mesh.set_shape_polylines(skeleton)
        coarse_mesh.set_global_face_pattern("ortho")
        session = RhinoSession.current()
        session.domain = Domain(outer, inners, guides, point_features)
        session.coarse = coarse_mesh
        session.field = field               # None on the skeleton route
        session.record("Coarse mesh")
        print("layout stored in the session{}, and drawn".format(
            ", with its field" if field is not None else ""))
        print("note: this replaces any previous layout -- densities (CMD_densities) and "
              "patterns (CMD_dense_pattern) set on it do not carry over.")
        print("next: CMD_edit_coarse_mesh to hand-edit the layout, then CMD_densities and "
              "CMD_quad_mesh.")

        rs.EnableRedraw(True)


if __name__ == "__main__":
    main()
