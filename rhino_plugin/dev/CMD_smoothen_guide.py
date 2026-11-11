#! python3
# r: compas
# r: pydantic
# r: compas_singular

"""Smooth a dense quad mesh with chains of vertices attached to guide curves.

Two ways to constrain, asked for first:

* **Auto** -- give the two limits once and every curve on the Guides layer gets its chain
  selected on those limits. Nothing is picked and nothing is edited by hand: the guides
  that shaped the layout are the guides the mesh is held to. Every one is selected BEFORE
  anything is attached, because attaching moves vertices and a guide processed second
  would otherwise be choosing on a mesh the first had already pulled about. Guides that
  get nothing are named, with the reason.
* **Hand** -- one guide at a time: the same automatic proposal, then Add / Remove / Clear
  before it is attached. Use it for the guides Auto refused, or where the proposal is
  right about the course but wrong about how much of it you want.

**The proposal is a piece of a POLYEDGE.** ``QuadMesh.collect_polyedges`` already has the
long lines of the mesh -- each one running from boundary to boundary or singularity to
singularity across four-valent vertices -- so the chain is not built up vertex by vertex,
it is CHOSEN: the longest run of one polyedge that still follows the guide. Building it up
was tried three ways and all three failed. A radius round the curve collects a ragged band
from both sides and folds faces when it is projected (max angle 180.00 on square+ring
cable); the nearest vertex per station comes out a zigzag; and a walk that turns where it
cannot go straight turns onto a DIFFERENT polyedge. Choosing a run of one polyedge cannot
do any of those, by construction.

Two gates decide how much of the polyedge is taken. DISTANCE asks whether a vertex is near
the guide; ANGLE asks whether its line is still going the guide's way, which is what cuts
the chain where the polyedge veers off near the ends.

**A boundary vertex is never moved onto the guide -- at most it SLIDES along the outline.**
Moving it takes the outline of the building with it, since attaching a vertex overrides
the boundary constraint it would otherwise have had.

The whole algorithm lives in ``compas_singular.editing.guide_chain`` and is measured,
without Rhino, in ``examples/dev_examples_tests/guide_chain_tests``. This file is the picks, the prompts and
the bake.
"""

from compas_singular.rhino.project import LAYER_DATA

import rhinoscriptsyntax as rs
import compas_rhino as cr
from compas_singular.rhino.helpers import mesh_from_rhino

from compas_singular.datastructures import QuadMesh
from compas_singular.datastructures import automated_boundary_constraints
from compas_singular.datastructures import constrained_smoothing
from compas_singular.editing import GuideCurve
from compas_singular.editing import attach_chain
from compas_singular.editing import chain_quality
from compas_singular.editing import collect_polyedges
from compas_singular.editing import guide_chain
from compas_singular.editing import mean_edge_length
from compas_singular.editing.guide_chain import DEFAULT_MAX_ANGLE
from compas_singular.editing.guide_chain import DEFAULT_TOLERANCE_FACTOR
from compas_singular.rhino.helpers import bake_mesh, clear_layer, curve_points

HIGHLIGHT_LAYER = "GuideSelection"

#: Where Auto looks for its guides -- the layer CMD_boundary_selection puts them on, i.e.
#: the same curves the field solve was steered by. Sublayers are read too.
#:
#: The full path, from the one place the layer map is written down. rhinoscriptsyntax
#: resolves a bare name with FindName, which returns the FIRST layer of that name anywhere
#: in the document -- fine until a second "Guides" exists somewhere else.
GUIDE_LAYER = LAYER_DATA.get("Guides", ("Guides", None))[0]

if not rs.IsLayer("QuadMesh"):
    raise RuntimeError("No dense quad mesh yet -- the QuadMesh layer does not exist.")


def quad_mesh_filter(rhobj, geometry, component_index):
    layer = rs.ObjectLayer(rhobj)
    quad_layer = rs.LayerName("QuadMesh", fullpath=True)
    return layer == quad_layer or rs.IsLayerChildOf(quad_layer, layer)


mesh_id = rs.GetObject(message="Pick the dense quad mesh", filter=rs.filter.mesh,
                       preselect=False, select=False, custom_filter=quad_mesh_filter)
if not mesh_id:
    raise RuntimeError("No mesh picked.")

# QuadMesh, not a plain Mesh: the polyedges are what the selection chooses from.
_tmp = mesh_from_rhino(cr.objects.find_object(mesh_id).Geometry)
mesh = QuadMesh.from_vertices_and_faces(*_tmp.to_vertices_and_faces())
rhino_vertices = rs.MeshVertices(mesh_id)

_nonquad = [f for f in mesh.faces() if len(mesh.face_vertices(f)) != 4]
if _nonquad:
    print("note: {} non-quad face(s) -- polyedges stop at those.".format(len(_nonquad)))

# collected ONCE: it costs O(edges squared) and does not depend on the guide
polyedges = collect_polyedges(mesh)
average = mean_edge_length(mesh)
print("{} vertices, {} polyedges, mean edge {:.3f}.".format(
    mesh.number_of_vertices(), len(polyedges), average))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _distance2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def nearest_vertex(xyz):
    """The compas vertex closest to a point.

    Matched by POSITION, not by index. mesh_to_compas and the bake round-trip both
    go through float conversions, so an index that looks like it should line up is
    not something to bet a constraint on.
    """
    best, best_d = None, None
    for vertex in mesh.vertices():
        d = _distance2(mesh.vertex_coordinates(vertex), xyz)
        if best_d is None or d < best_d:
            best, best_d = vertex, d
    return best


def show_selection(vertices):
    """Redraw the highlight points. Guarded: rs.AddPoints([]) returns None."""
    if not rs.IsLayer(HIGHLIGHT_LAYER):
        rs.AddLayer(HIGHLIGHT_LAYER, (255, 0, 0))
    clear_layer(HIGHLIGHT_LAYER)
    if not vertices:
        return
    guids = rs.AddPoints([mesh.vertex_coordinates(v) for v in vertices]) or []
    for guid in guids:
        rs.ObjectLayer(guid, layer=HIGHLIGHT_LAYER)
        rs.ObjectColor(guid, (255, 0, 0))
    rs.Redraw()


def pick_vertices(message):
    picked = rs.GetMeshVertices(mesh_id, message)
    if not picked:
        return []
    return [nearest_vertex(rhino_vertices[i]) for i in picked]


# ---------------------------------------------------------------------------
# Constrain: automatically for every guide, or one guide at a time by hand
# ---------------------------------------------------------------------------
mode = rs.GetString(message="Constrain the guides how?", defaultString="Auto",
                    strings=["Auto", "Hand"])
mode = (mode or "Auto").lower()

# Both modes use the same two gates and ask for them the same way, because they are what
# the automatic selection is. They do different jobs. DISTANCE decides how far off the
# guide a vertex may sit -- and a vertex is pulled onto the guide by exactly that far, so
# it sets both coverage and damage. ANGLE decides whether the polyedge is still running
# the guide's way, which is what cuts the chain where the line veers off near the ends.
# Measured over 14 guides: the angle gate takes the worst face angle from 15.9 to 31.6
# degrees at distance 0.8 and covers MORE of the guide, which is what lets the distance
# default be as wide as two target edge lengths. At that width a guide drawn ON a wall is
# no longer refused and will pull the row next to it onto the wall -- drop the distance to
# 0.8 if that matters. Set the angle to 90 to turn the angle gate off.
factor = rs.GetReal(message="Distance tolerance, in target edge lengths",
                    number=DEFAULT_TOLERANCE_FACTOR, minimum=0.05, maximum=5.0)
tolerance = (factor or DEFAULT_TOLERANCE_FACTOR) * average
max_angle = rs.GetReal(message="Angle tolerance in degrees (90 = off)",
                       number=DEFAULT_MAX_ANGLE, minimum=1.0, maximum=90.0)
max_angle = max_angle or DEFAULT_MAX_ANGLE

attached = {}
claimed = {}
moved = set()


def guide_curves():
    """Every curve on the Guides layer and its sublayers.

    Auto does not ask which guides to use: these ARE the guides. They are the curves
    CMD_boundary_selection collected and the field was steered by, so holding the dense
    mesh to them is holding it to what it was generated from.
    """
    if not rs.IsLayer(GUIDE_LAYER):
        return []
    layers = [GUIDE_LAYER] + (rs.LayerChildren(GUIDE_LAYER) or [])
    found = []
    for layer in layers:
        for guid in rs.ObjectsByLayer(layer) or []:
            if rs.IsCurve(guid) and guid not in found:
                found.append(guid)
    return found


def build_guide(curve_id):
    """A picked curve as a GuideCurve."""
    # max_edge only sets how finely a curved guide is sampled; a polyline guide
    # is taken at its own vertices either way.
    points = curve_points(curve_id, 0.25)
    # curve_points DROPS the closing point -- a loop is closed by convention there.
    # A GuideCurve reads its own points, so a ring handed over like that is an arc
    # with a gap in it, and the arc-length parameter cannot wrap across the seam.
    if rs.IsCurveClosed(curve_id) and len(points) > 2:
        points = points + points[:1]
    return GuideCurve(points)


def propose(guide):
    """Select a chain for one guide and say what it is. Returns (chain, info)."""
    selected, info = guide_chain(mesh, guide, tolerance=tolerance,
                                 max_angle=max_angle, polyedges=polyedges)
    if not selected:
        return selected, info
    quality = chain_quality(mesh, selected, guide)
    # coverage says how much of the guide it got; alignment says whether the mesh has a
    # course along this guide at all -- a low alignment means it does not, and no
    # selection will fix that
    print("  {} vertices: {:.0f}% of the guide, alignment {:.2f}, worst face angle if "
          "attached {:.0f}.".format(len(selected), 100 * quality['coverage'],
                                    quality['alignment'], quality['min_face_angle']))
    if quality['alignment'] < 0.8:
        print("  note: this guide does not follow a course of the mesh.")
    if quality['folded_faces']:
        print("  WARNING: attaching this one turns {} face(s) inside out. It is reaching "
              "{:.1f} edge lengths to do it -- lower the distance tolerance to stop "
              "it.".format(quality['folded_faces'], quality['off_max']))
    return selected, info


def attach(selected, guide, hold, label):
    """Move what may move, record what holds each vertex, and report."""
    # A BOUNDARY vertex is never moved onto the guide -- at most it slides along the
    # outline. Moving it takes the outline of the building with it, since attaching
    # overrides the boundary constraint it would otherwise have had.
    # Fixed pins an interior vertex where it lands; Sliding re-projects it every
    # iteration, so it may travel ALONG the curve but never leave it.
    moves, constraints = attach_chain(mesh, selected, guide, hold=hold)
    for vertex, xyz in moves.items():
        mesh.vertex_attributes(vertex, "xyz", xyz)
    # a vertex that two guides both want ends up held by whichever came last, and nothing
    # on screen shows that happening -- so say it
    overlap = [vertex for vertex in constraints if vertex in claimed]
    for vertex in constraints:
        claimed[vertex] = label
    attached.update(constraints)
    moved.update(moves)
    print("  {} attached ({}); {} moved onto the guide, {} left sliding on the "
          "boundary.".format(len(selected), hold, len(moves), len(selected) - len(moves)))
    if overlap:
        print("  note: {} of them were already claimed by an earlier guide and are now "
              "held by this one.".format(len(overlap)))


try:
    # -----------------------------------------------------------------------
    # AUTO -- every guide at once, on the limits just given
    # -----------------------------------------------------------------------
    if mode == "auto":
        curve_ids = guide_curves()
        if not curve_ids:
            raise RuntimeError(
                "No curves on the {} layer -- nothing to constrain to. Draw the guides "
                "there (CMD_boundary_selection does), or run this again in Hand mode and "
                "pick them.".format(GUIDE_LAYER))

        print("auto-constraining every guide on the {} layer: {} curve(s), distance "
              "{:.3f} ({:.2f} edge lengths), angle {:.0f} degrees.".format(
                  GUIDE_LAYER, len(curve_ids), tolerance, tolerance / average, max_angle))

        # SELECT EVERYTHING FIRST, then attach. Attaching moves vertices, so a guide
        # processed after another would be choosing on a mesh the first one had already
        # pulled about -- the result would depend on the order the curves were picked in.
        proposals, refused = [], []
        for index, curve_id in enumerate(curve_ids):
            print("guide {}:".format(index + 1))
            guide = build_guide(curve_id)
            selected, info = propose(guide)
            if not selected:
                print("  skipped -- {}.".format(info['reason']))
                refused.append(index + 1)
                continue
            proposals.append((index + 1, guide, selected))

        if not proposals:
            raise RuntimeError("No guide got a chain -- widen the tolerances.")

        # highlight everything before asking anything, so the answer is given with the
        # whole proposal on screen. This is the only look at it there is: Auto does not
        # stop to be edited.
        chosen = [vertex for _, _, selected in proposals for vertex in selected]
        show_selection(chosen)

        hold = rs.GetString(message="Hold the {} selected vertices how?".format(
            len(set(chosen))), defaultString="Fixed", strings=["Fixed", "Sliding"])
        hold = (hold or "Fixed").lower()

        for number, guide, selected in proposals:
            print("guide {}:".format(number))
            attach(selected, guide, hold, "guide {}".format(number))

        if refused:
            print("{} guide(s) got nothing: {}. Widen the tolerances, or run again in "
                  "Hand mode to pick those by hand.".format(
                      len(refused), ", ".join(str(i) for i in refused)))

    # -----------------------------------------------------------------------
    # HAND -- one guide at a time, proposal edited before it is attached
    # -----------------------------------------------------------------------
    else:
        while True:
            curve_id = rs.GetObject(message="Pick a guide curve (Enter to stop)",
                                    filter=rs.filter.curve, preselect=False, select=False)
            if not curve_id:
                break

            guide = build_guide(curve_id)
            selected, info = propose(guide)
            show_selection(selected)
            if not selected:
                print("  {} -- use Add to pick the chain by hand.".format(info['reason']))

            while True:
                action = rs.GetString(
                    message="Selection: {} vertices".format(len(selected)),
                    defaultString="Done", strings=["Done", "Add", "Remove", "Clear"])
                action = (action or "Done").lower()

                if action == "done":
                    break
                elif action == "add":
                    for vertex in pick_vertices("Pick vertices to ADD"):
                        if vertex not in selected:
                            selected.append(vertex)
                elif action == "remove":
                    drop = set(pick_vertices("Pick vertices to REMOVE"))
                    selected = [v for v in selected if v not in drop]
                elif action == "clear":
                    selected = []
                show_selection(selected)

            if not selected:
                print("  nothing selected -- guide skipped.")
                continue

            hold = rs.GetString(message="Hold those vertices how?", defaultString="Fixed",
                                strings=["Fixed", "Sliding"])
            hold = (hold or "Fixed").lower()

            attach(selected, guide, hold, "this guide")
            show_selection([])
finally:
    if rs.IsLayer(HIGHLIGHT_LAYER):
        rs.PurgeLayer(HIGHLIGHT_LAYER)

if not attached:
    raise RuntimeError("No vertices attached to any guide -- nothing to do.")

# ---------------------------------------------------------------------------
# Smooth
# ---------------------------------------------------------------------------
boundary_mode = rs.GetString(message="Boundary treatment", defaultString="Sliding",
                             strings=["Sliding", "Fixed"])
boundary_mode = (boundary_mode or "Sliding").lower()

kmax = rs.GetInteger(message="Iterations", number=100, minimum=1)
damping = rs.GetReal(message="Damping", number=0.5, minimum=0.0, maximum=1.0)

if boundary_mode == "sliding":
    constraints = automated_boundary_constraints(mesh)       # slides, corners pinned
    fixed = None
else:
    constraints = {}
    fixed = [v for loop in mesh.vertices_on_boundaries() for v in loop]

# An attached INTERIOR vertex is held on its guide; an attached BOUNDARY vertex is held on
# its own outline by attach_chain, which is the same kind of constraint the sliding
# boundary treatment would have given it. So this update never takes a vertex off a wall.
constraints.update(attached)
if fixed:
    fixed = [v for v in fixed if v not in attached]

constrained_smoothing(mesh, kmax=kmax, damping=damping,
                      constraints=constraints, algorithm="area", fixed=fixed)

clear_layer("Smoothed", clean_sublayers=True)
rs.AddLayer("Smoothed", parent="QuadMesh")
rs.AddLayer("Guided", parent="Smoothed")
bake_mesh(mesh, "Guided")

print("smoothed: {} vertices moved onto guides, {} constrained in total, boundary {}.".format(
    len(moved), len(constraints), boundary_mode))
print("baked to 'QuadMesh::Smoothed::Guided' -- the original 'QuadMesh' is untouched.")
print("next: CMD_dual for the dual mesh, or run this again to attach more guides.")