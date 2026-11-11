#! python3
# r: compas
# r: pydantic
# r: compas_singular

from compas_singular.rhino.project import get_settings

import rhinoscriptsyntax as rs
import compas_rhino as cr
from compas_singular.rhino.helpers import mesh_from_rhino

from compas_singular.datastructures.mesh.smoothing import automated_boundary_constraints
from compas_singular.datastructures.mesh.smoothing import constrained_smoothing
from compas_singular.datastructures.mesh.smoothing import boundary_smoothing
from compas_singular.datastructures.mesh.smoothing import region_smoothing
from compas_singular.datastructures.mesh.smoothing import relaxation
from compas_singular.rhino.helpers import bake_mesh, clear_layer

REGION_LAYER = "RelaxRegion"

#: Rings of vertices outside the core over which the damping falls to zero.
#: A region smoothed at full strength up to a hard edge leaves a CREASE at that
#: edge -- a fixed ring next to a fully-relaxed one is a kink, which is the sort of
#: defect this is meant to remove. The taper is what makes it blend.
DEFAULT_BLEND = 3

if not rs.IsLayer("QuadMesh"):
    raise RuntimeError("No dense quad mesh has been generated yet -- the layer does not exist.")


def quad_mesh_filter(rhobj, geometry, component_index):
    layer = rs.ObjectLayer(rhobj)
    quad_layer = rs.LayerName("QuadMesh", fullpath=True)
    return layer == quad_layer or rs.IsLayerChildOf(quad_layer, layer)


mesh_id = rs.GetObject(
    message="Pick a dense quad mesh to smoothen from the layer QuadMesh and its sublayers",
    filter=rs.filter.mesh, preselect=False, select=False,
    custom_filter=quad_mesh_filter, subobjects=False)
if not mesh_id:
    raise RuntimeError("No mesh picked.")

mesh = mesh_from_rhino(cr.objects.find_object(mesh_id).Geometry)
rhino_vertices = rs.MeshVertices(mesh_id)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _distance2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def nearest_vertex(xyz):
    """The compas vertex closest to a point.

    Matched by POSITION, not by index. mesh_to_compas and the bake round-trip both
    go through float conversions, so an index that looks like it should line up is
    not something to bet a selection on.
    """
    best, best_d = None, None
    for vertex in mesh.vertices():
        d = _distance2(mesh.vertex_coordinates(vertex), xyz)
        if best_d is None or d < best_d:
            best, best_d = vertex, d
    return best


def pick_vertices(message):
    picked = rs.GetMeshVertices(mesh_id, message)
    if not picked:
        return []
    return [nearest_vertex(rhino_vertices[i]) for i in picked]


def all_boundary_vertices():
    # NOT vertices_on_boundary(): in COMPAS 2 that returns the LONGEST boundary only,
    # so on a mesh with a hole the hole is left out -- and left free to blow up.
    return [v for loop in mesh.vertices_on_boundaries() for v in loop]


def vertices_in_curve(curve_id):
    """Every vertex inside a closed planar curve. 1 = inside, 2 = on it."""
    inside = []
    for vertex in mesh.vertices():
        if rs.PointInPlanarClosedCurve(mesh.vertex_coordinates(vertex), curve_id) in (1, 2):
            inside.append(vertex)
    return inside


def vertices_in_curves(curve_ids):
    """The union of the vertices inside any of several closed curves.

    The zones do not have to be disjoint -- all four corners of a slab are one
    selection here, and a vertex caught by two overlapping curves stays a single
    core vertex rather than being counted twice.
    """
    inside = []
    seen = set()
    for curve_id in curve_ids:
        for vertex in vertices_in_curve(curve_id):
            if vertex not in seen:
                seen.add(vertex)
                inside.append(vertex)
    return inside


def grow(core, rings):
    """The core plus `rings` of neighbours around it."""
    out = set(core)
    frontier = set(core)
    for _ in range(rings):
        frontier = {n for v in frontier for n in mesh.vertex_neighbors(v)} - out
        out |= frontier
    return out


def taper(core, blend):
    """Per-vertex damping weight: 1 in the core, falling to 0 outside the blend.

    A vertex with no weight is fixed. The ring just beyond the blend is therefore
    held and the ones just inside it barely move, so there is no step in the result.
    """
    weight = {v: 1.0 for v in core}
    frontier = set(core)
    for ring in range(1, blend + 1):
        frontier = {n for v in frontier for n in mesh.vertex_neighbors(v)} - set(weight)
        if not frontier:
            break
        w = 1.0 - ring / float(blend + 1)
        for vertex in frontier:
            weight[vertex] = w
    return weight


def show_region(weight):
    """Core red, blend orange, so the taper is visible before committing to it."""
    if not rs.IsLayer(REGION_LAYER):
        rs.AddLayer(REGION_LAYER, (255, 0, 0))
    clear_layer(REGION_LAYER)
    if not weight:
        return
    for vertex, w in weight.items():
        guid = rs.AddPoint(mesh.vertex_coordinates(vertex))
        rs.ObjectLayer(guid, layer=REGION_LAYER)
        rs.ObjectColor(guid, (255, 0, 0) if w >= 1.0 else (255, 160, 0))
    rs.Redraw()


# ---------------------------------------------------------------------------
# mode
# ---------------------------------------------------------------------------
rs.AddLayer("Smoothed", parent="QuadMesh")

mode = rs.GetString(message="Smooth what?", defaultString="Whole",
                    strings=["Whole", "Region"])
mode = (mode or "Whole").lower()


# ---------------------------------------------------------------------------
# WHOLE
# ---------------------------------------------------------------------------
if mode == "whole":
    boundary_mode = rs.GetString(message="Boundary treatment", defaultString="Sliding",
                                 strings=["Sliding", "Fixed"])
    boundary_mode = (boundary_mode or "Sliding").lower()

    point_mode = rs.GetString(message="Point_feature treatment", defaultString="Free",
                                 strings=["Free", "Fixed"])
    point_mode = (point_mode or "Free").lower()

    algorithm = rs.GetString(message="Select smoothing method", defaultString="Area",
                             strings=["Area", "Centroid", "CenterOfMass", "ForceDensity"])
    algorithm = (algorithm or "Area").lower()

    if algorithm=="forcedensity":
        fixed = rs.GetString(message="Vertices to fix", defaultString="Corners",
                                 strings=["Corners", "Boundary", "Manual"])
        fixed = (fixed or "Corners").lower()
        
        fixed_vertices = []
        if fixed == "manual":    
            fixed_vertices = pick_vertices("Pick vertices to constrain.")

        relaxation(mesh, fixed=fixed_vertices if fixed == "manual" else fixed, q_factor=10)

    else:
        kmax = rs.GetInteger(message="Iterations", number=100, minimum=1)
        damping = rs.GetReal(message="Damping", number=0.5, minimum=0.0, maximum=1.0)

        if boundary_mode == "sliding":
            constraints = automated_boundary_constraints(mesh)    # slides, corners pinned
            fixed = None
        else:
            constraints = {}
            fixed = all_boundary_vertices()

        if point_mode == "fixed":
            pass #add possibility to fix point_features and perhaps also guides
        constrained_smoothing(mesh, kmax=kmax, damping=damping,
                            constraints=constraints, algorithm=algorithm, fixed=fixed)

    layer = {"area": "Area", "centroid": "Centroid",
             "centerofmass": "CenterOfMass", "forcedensity":"ForceDensity"}[algorithm]
    rs.AddLayer(layer, parent="Smoothed")
    clear_layer(layer, clean_sublayers=True)
    bake_mesh(mesh, layer)
    print("smoothed: {}, boundary {}, {} iterations at damping {}.".format(
        layer, boundary_mode, kmax, damping))
    print("baked to 'QuadMesh::Smoothed::{}' -- the original 'QuadMesh' is untouched.".format(layer))
    print("next: CMD_dual for the dual mesh, or run this again on the result if it still needs work.")


# ---------------------------------------------------------------------------
# REGION
# ---------------------------------------------------------------------------
else:
    how = rs.GetString(message="Select the problem zone by", defaultString="Curve",
                       strings=["Curve", "Vertices"])
    how = (how or "Curve").lower()

    core = []
    if how == "curve":
        curve_ids = rs.GetObjects(message="Pick one or more CLOSED curves around the zones",
                                  filter=rs.filter.curve, preselect=False, select=False,
                                  minimum_count=1) or []
        closed = [cid for cid in curve_ids if rs.IsCurveClosed(cid)]
        skipped = len(curve_ids) - len(closed)
        if skipped:
            print("{} of the picked curves {} not closed -- skipped.".format(
                skipped, "is" if skipped == 1 else "are"))
        if closed:
            core = vertices_in_curves(closed)
            print("{} closed curve{} -> {} core vertices.".format(
                len(closed), "" if len(closed) == 1 else "s", len(core)))
        elif curve_ids:
            print("none of the picked curves are closed -- pick vertices instead.")
    else:
        core = pick_vertices("Pick vertices in the zone")

    if not core:
        raise RuntimeError("No region selected.")

    blend = rs.GetInteger(message="Blend rings outside the zone",
                          number=DEFAULT_BLEND, minimum=0)
    weight = taper(set(core), blend)
    show_region(weight)
    print("region: {} core vertices, {} including the blend.".format(len(core), len(weight)))

    try:
        while True:
            action = rs.GetString(message="Region: {} core".format(len(core)),
                                  defaultString="Done",
                                  strings=["Done", "Add", "Remove", "Grow"])
            action = (action or "Done").lower()
            if action == "done":
                break
            elif action == "add":
                for vertex in pick_vertices("Pick vertices to ADD"):
                    if vertex not in core:
                        core.append(vertex)
            elif action == "remove":
                drop = set(pick_vertices("Pick vertices to REMOVE"))
                core = [v for v in core if v not in drop]
            elif action == "grow":
                core = list(grow(set(core), 1))
            weight = taper(set(core), blend)
            show_region(weight)

        if not core:
            raise RuntimeError("Region emptied -- nothing to relax.")

        kmax = rs.GetInteger(message="Iterations", number=50, minimum=1)
        damping = rs.GetReal(message="Damping", number=0.5, minimum=0.0, maximum=1.0)

        # the previewed weights are the ones that get relaxed, so the zone that moves
        # is exactly the zone that was shown; the boundary constraints are assembled
        # inside, and let a zone touching the outline slide along it
        region_smoothing(mesh, weight, kmax=kmax, damping=damping)
    finally:
        if rs.IsLayer(REGION_LAYER):
            rs.PurgeLayer(REGION_LAYER)

    rs.AddLayer("Relaxed", parent="Smoothed")
    bake_mesh(mesh, "Relaxed")
    print("relaxed {} vertices ({} core, blend {} rings).".format(
        len(weight), len(core), blend))
    print("baked to 'QuadMesh::Smoothed::Relaxed' -- the original 'QuadMesh' is untouched.")
    print("next: CMD_dual for the dual mesh, or run this again on another zone.")
