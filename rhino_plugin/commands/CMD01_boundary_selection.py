#! python3

# r: compas
# r: pydantic
# r: compas_singular

"""Step 2 -- select the domain.

Input: closed curves, curves and points picked in the document. Output: copies on TopologyProblem::InputBoundaries.
"""
import Rhino
import rhinoscriptsyntax as rs

from compas_singular.rhino.helpers import clear_layer
from compas_singular.rhino.helpers import read_boundaries
from compas_singular.rhino.project import ROOT
from compas_singular.rhino.project import get_settings
from compas_singular.rhino.project import layer_path
from compas_singular.rhino.project import resolve_spacing
from compas_singular.rhino.project import set_settings


def closed_filter(rhobj, geometry, component_index):
    """Pick filter: only closed curves, surfaces and meshes."""
    if isinstance(geometry, Rhino.Geometry.Curve):
        return rs.IsCurveClosed(rhobj)
    elif isinstance(geometry, Rhino.Geometry.Surface):
        return rs.IsSurfaceClosed(rhobj)
    elif isinstance(geometry, Rhino.Geometry.Mesh):
        return rs.IsMeshClosed(rhobj)

    return False


def select_boundary():
    curve = rs.GetObject(message="Pick a closed polyline as boundary", filter=rs.filter.curve,
                         preselect=False, select=True, subobjects=False, custom_filter=closed_filter)
    if curve:
        curve = rs.CopyObject(curve)
    return curve


def select_boundaries():
    curves = rs.GetObjects(message="Pick closed polylines as boundaries", filter=rs.filter.curve, group=True,
                           preselect=False, select=True, minimum_count=0, maximum_count=10, custom_filter=closed_filter)
    if not curves:
        return []
    return rs.CopyObjects(curves) or []


def edit_outer():
    existing = rs.ObjectsByLayer("Outer")
    rs.ObjectLayer(existing, "TrashBin")
    outer_boundary = select_boundary()
    if outer_boundary:
        rs.ObjectLayer(outer_boundary, layer="Outer")


def edit_inner():
    mode = rs.GetString(message="Inner boundary selection mode", defaultString="Add", strings=["Add", "Delete", "New"])
    if mode is None:
        return
    mode = mode.lower()

    if mode == "new":
        new_inner_boundaries()
    elif mode == "delete":
        delete_inner_boundaries()
    elif mode == "add":
        add_inner_boundaries()
    else:
        print("Unrecognised mode '{}', defaulting to Add".format(mode))
        add_inner_boundaries()


def new_inner_boundaries():
    existing = rs.ObjectsByLayer("Inner")
    rs.ObjectLayer(existing, "TrashBin")
    for boundary in select_boundaries():
        rs.ObjectLayer(boundary, layer="Inner")


def add_inner_boundaries():
    for boundary in select_boundaries():
        rs.ObjectLayer(boundary, layer="Inner")


def delete_inner_boundaries():
    def inner_filter(rhobj, geometry, component_index):
        return rs.ObjectLayer(rhobj) == layer_path("Inner")

    to_delete = rs.GetObjects(message="Select inner boundaries to delete", filter=rs.filter.curve, group=True,
                              preselect=False, select=True, minimum_count=0, custom_filter=inner_filter)
    if to_delete:
        rs.ObjectLayer(to_delete, "TrashBin")
    else:
        print("No boundaries selected for deletion.")


def edit_guides():
    mode = rs.GetString(message="Guide selection mode", defaultString="Add", strings=["Add", "Delete", "New"])
    if mode is None:
        return
    mode = mode.lower()

    if mode == "new":
        new_guides()
    elif mode == "delete":
        delete_guides()
    elif mode == "add":
        add_guides()
    else:
        print("Unrecognised mode '{}', defaulting to Add".format(mode))
        add_guides()


def new_guides():
    existing = rs.ObjectsByLayer("Guides")
    rs.ObjectLayer(existing, "TrashBin")

    curves = rs.GetObjects(message="Pick polylines as guides.", filter=rs.filter.curve, group=True, preselect=False, select=True, minimum_count=0)
    curves = rs.CopyObjects(curves) if curves else []
    for guide in curves:
        rs.ObjectLayer(guide, layer="Guides")


def add_guides():
    curves = rs.GetObjects(message="Pick polylines as guides.", filter=rs.filter.curve, group=True, preselect=False, select=True, minimum_count=0)
    curves = rs.CopyObjects(curves) if curves else []
    for guide in curves:
        rs.ObjectLayer(guide, layer="Guides")


def delete_guides():
    def guide_filter(rhobj, geometry, component_index):
        return rs.ObjectLayer(rhobj) == layer_path("Guides")

    to_delete = rs.GetObjects(message="Select guides to delete", filter=rs.filter.curve, group=True,
                              preselect=False, select=True, minimum_count=0, custom_filter=guide_filter)
    if to_delete:
        rs.ObjectLayer(to_delete, "TrashBin")
    else:
        print("No guides selected for deletion.")


def edit_point_features():
    mode = rs.GetString(message="Point feature selection mode", defaultString="Add", strings=["Add", "Delete", "New"])
    if mode is None:
        return
    mode = mode.lower()

    if mode == "new":
        new_pts()
    elif mode == "delete":
        delete_pts()
    elif mode == "add":
        add_pts()
    else:
        print("Unrecognised mode '{}', defaulting to Add".format(mode))
        add_pts()


def new_pts():
    existing = rs.ObjectsByLayer("PointFeatures")
    rs.ObjectLayer(existing, "TrashBin")

    curves = rs.GetObjects(message="Pick points as point features.", filter=rs.filter.point, group=True, preselect=False, select=True, minimum_count=0)
    curves = rs.CopyObjects(curves) if curves else []
    for guide in curves:
        rs.ObjectLayer(guide, layer="PointFeatures")


def add_pts():
    curves = rs.GetObjects(message="Pick points as point features.", filter=rs.filter.point, group=True, preselect=False, select=True, minimum_count=0)
    curves = rs.CopyObjects(curves) if curves else []
    for guide in curves:
        rs.ObjectLayer(guide, layer="PointFeatures")


def delete_pts():
    def guide_filter(rhobj, geometry, component_index):
        return rs.ObjectLayer(rhobj) == layer_path("PointFeatures")

    to_delete = rs.GetObjects(message="Select point features to delete", filter=rs.filter.point, group=True,
                              preselect=False, select=True, minimum_count=0, custom_filter=guide_filter)
    if to_delete:
        rs.ObjectLayer(to_delete, "TrashBin")
    else:
        print("No point features selected for deletion.")


def main():
    settings = get_settings()

    rs.AddLayer(name="Outer", parent="InputBoundaries", color=(255, 0, 0))
    rs.AddLayer(name="Inner", parent="InputBoundaries", color=(0, 255, 0))
    rs.AddLayer(name="Guides", parent="InputBoundaries", color=(255, 127, 0))
    rs.AddLayer(name="PointFeatures", parent="InputBoundaries", color=(0, 255, 0))
    rs.AddLayer(name="TrashBin", parent=ROOT, color=(90, 90, 90), visible=False)

    while True:
        section = rs.GetString(message="Edit boundary", defaultString="Continue",
                               strings=["Outer", "Inner", "Background_Triangulation",
                                        "Point_Features", "Guides", "Guide_Alignment",
                                        "Field_Solver", "Clear", "Continue"])
        # GetString answers are compared lower-case.
        section = (section or "Continue").lower()
        if section == "Outer".lower():
            edit_outer()
        elif section == "Inner".lower():
            edit_inner()
        elif section == "guides":
            edit_guides()
        elif section == "Guide_Alignment".lower():
            answer = rs.GetString(message="Elements run ALONG or ACROSS the guides?",
                                  defaultString=settings["guide_alignment"],
                                  strings=["tangent", "perpendicular"])
            if answer:
                settings["guide_alignment"] = answer.lower()
                set_settings(settings)
        elif section == "Field_Solver".lower():
            current = settings.get("relax", "auto")
            default = "Auto" if str(current).lower() == "auto" else ("On" if current else "Off")
            answer = rs.GetString(message="Relax the field solve? Auto = on when guides exist",
                                  defaultString=default, strings=["Auto", "On", "Off"])
            if answer:
                answer = answer.lower()
                settings["relax"] = "auto" if answer == "auto" else (answer == "on")
                set_settings(settings)
        elif section == "Background_Triangulation".lower():
            # 0 = the thesis value (eq. 4.1, from the domain's size).
            value = rs.GetReal("Background spacing (NOT the quad size), 0 = thesis value",
                               settings["triangulation_spacing"] or 0.0, 0.0)
            if value is not None:
                settings["triangulation_spacing"] = value or None
                set_settings(settings)
        elif section == "Point_Features".lower():
            edit_point_features()
        elif section == "Clear".lower():
            clear_layer("InputBoundaries", True)
        elif section == "Continue".lower():
            break
        else:
            print("Unrecognised option '{}'".format(section))

    outer, inner, guides, point_features = read_boundaries(spacing=resolve_spacing(settings))

    print("Boundary selection completed.")
    print("1 outer boundary curve")
    print(f"{len(inner)} inner boundary curves")
    print(f"{len(guides)} guide curves")
    print(f"{len(point_features)} point features")
    print("next: CMD_coarse_mesh to generate a coarse layout, or CMD_read_coarse_mesh to read a hand-drawn one.")


if __name__ == "__main__":
    main()
