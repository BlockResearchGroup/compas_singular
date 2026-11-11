#! python3

# r: compas
# r: pydantic
# r: compas_rui
# r: compas_singular

"""Step 4 -- edit the coarse layout by hand.

Input: the session's layout and field, the boundaries on InputBoundaries. Output: the edited layout in the session, on commit.
"""
import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
from System.Drawing import Color

from compas_singular.datastructures.mesh_quad_coarse.coarse_curves import coarse_edges_to_curves
from compas_singular.editing import CoarseEditor
from compas_singular.rhino import mesh_ui
from compas_singular.rhino.helpers import read_boundaries
from compas_singular.rhino.helpers import read_boundary_loops
from compas_singular.rhino.mesh_ui import FINISH
from compas_singular.rhino.project import get_settings
from compas_singular.rhino.project import read_layout
from compas_singular.rhino.project import resolve_spacing
from compas_singular.rhino.session import RhinoSession

#: Wall sampling, as a multiple of the background spacing.
WALL_SAMPLING_FACTOR = 0.25

#: Short layer name of the baked layout.
MESH_LAYER = "Mesh"


def redraw(editor, layout, special=None):
    """Draw the layout as pickable corners and edges, poles (or ``special``) highlighted."""
    layout.mesh = editor.mesh
    layout.edge_shape = editor.edge_shape
    if special is None:
        special = editor.mesh.poles() if hasattr(editor.mesh, "poles") else ()
    layout.special = special
    layout.redraw()


def draw_strips(editor, layout):
    """Every strip of the layout being edited, as a pickable ribbon."""
    layout.mesh = editor.mesh
    layout.draw_strips()


def edit(editor, mutate):
    """Run one mutating ``editor`` call with undo bookkeeping. ``(ok, notes)``.

    ``mutate`` is a no-argument callable returning ``(ok, notes)``; a refused call leaves no undo step.
    """
    editor.push_undo()
    ok, notes = mutate()
    if not ok:
        editor.discard_last_undo()
    return ok, notes


def undo(editor, layout):
    """Put back the layout as it was before the last edit. Prints the result."""
    ok, notes = editor.undo()
    if not ok:
        print("Nothing to undo.")
        return False
    redraw(editor, layout)
    print("Undone -- layout now has {} patch(es). {} more undo(s) available."
          .format(notes["faces"], notes["remaining"]))
    return True


def move_vertices(editor, layout):
    """Drag corners until Esc. ``True`` if any moved."""
    changed = False
    while True:
        vkey = layout.pick_vertex("Select a coarse corner")
        if vkey is None:
            return changed
        on_boundary = editor.is_vertex_on_boundary(vkey)
        neighbours = [editor.mesh.vertex_coordinates(nbr)
                      for nbr in editor.mesh.vertex_neighbors(vkey)]
        xyz = mesh_ui.drag_point(
            "New position for corner {}{}".format(
                vkey, " (held on the wall)" if on_boundary else ""),
            editor.mesh.vertex_coordinates(vkey),
            neighbours=neighbours,
            project=lambda point: editor.project(point, on_boundary))
        if xyz is None:
            return changed
        moved, _notes = edit(editor, lambda: editor.move_vertex(vkey, xyz))
        if moved:
            changed = True
            redraw(editor, layout)


def move_pole(editor, layout):
    """Pick a pole, then the corner it moves to, until Esc. ``True`` if any moved.

    Only to a corner every patch of the pole shares.
    """
    changed = False
    while True:
        poles = editor.mesh.poles() if hasattr(editor.mesh, "poles") else []
        if not poles:
            print("No poles on this layout.")
            return changed
        redraw(editor, layout)
        sc.doc.Views.Redraw()
        pkey = layout.pick_vertex("Select a pole to move (highlighted)")
        if pkey is None:
            return changed
        if pkey not in poles:
            print("  corner {} is not a pole -- pick one of the highlighted "
                  "corners.".format(pkey))
            continue

        targets = editor.pole_targets(pkey)
        if not targets:
            mesh_ui.refuse(
                "Move pole",
                "Pole {} cannot move.\n\nIts patches share no other corner, and a "
                "pole can only move to a corner all of its patches share (a pole "
                "with one or two patches, to a neighbouring corner).".format(pkey))
            continue

        redraw(editor, layout, special=targets)
        sc.doc.Views.Redraw()
        print("  pole {} can move to corner(s) {}".format(pkey, targets))
        vkey = layout.pick_vertex("New corner for pole {} (highlighted, Esc to "
                                 "pick another pole)".format(pkey))
        if vkey is None:
            continue

        ok, notes = edit(editor, lambda: editor.move_pole(pkey, vkey))
        if not ok:
            mesh_ui.refuse("Move pole",
                           "The pole was NOT moved.\n\n{}.\n\nThe layout is "
                           "unchanged.".format(editor.last_reason))
            continue
        print("Pole moved: corner {} -> {} ({} patch(es) relabelled).".format(
            notes["pole"], notes["to"], notes["faces"]))
        changed = True


def choose_mode():
    """Polyline / Arc, or ``None`` if the user cancelled (Esc)."""
    go = Rhino.Input.Custom.GetOption()
    go.SetCommandPrompt("Choose path type")
    go.AddOption("Polyline")
    go.AddOption("Arc")
    if go.Get() != Rhino.Input.GetResult.Option:
        return None
    return go.Option().EnglishName


def pick_second_edge(layout):
    """The edge to end ON. ``(guid, geometry)``, or ``(None, None)``."""
    picked = layout.pick_edge("Select the coarse edge to end ON")
    if picked is None:
        return None, None
    _edge, guid = picked
    rs.SelectObject(guid)
    return guid, rs.coercecurve(guid)


def draw_polyline(layout, start_pt):
    """Polyline mode: free points, ending the moment one lands on the layout.

    Note: one GetPoint is reused across picks, so Backspace (undo) keeps working.
    """
    points = [start_pt]
    draw_color = Color.FromArgb(40, 120, 220)

    def _draw(sender, e):
        if len(points) >= 2:
            e.Display.DrawPolyline(points, draw_color, 2)
        e.Display.DrawLine(points[-1], e.CurrentPoint, draw_color, 2)
        for point in points:
            e.Display.DrawPoint(point, draw_color)

    print("Pick points freely - Backspace removes the last point.")
    print("Placing a point ON a layout edge finishes the path (Esc to change path type).")

    gp = Rhino.Input.Custom.GetPoint()
    gp.DynamicDraw += _draw
    gp.Constrain(Rhino.Geometry.Plane.WorldXY, False)
    try:
        while True:
            gp.SetCommandPrompt("Pick next point (Backspace = undo)")
            gp.AcceptUndo(True)
            result = gp.Get()

            if result == Rhino.Input.GetResult.Undo:
                if len(points) > 1:
                    points.pop()
                else:
                    print("Nothing left to undo.")
                continue

            if result != Rhino.Input.GetResult.Point:
                return None

            point = gp.Point()
            snapped = layout.closest_edge_point(point)
            if snapped:
                points.append(snapped)
                break
            points.append(point)
    finally:
        gp.DynamicDraw -= _draw

    if len(points) < 2:
        return None
    return rs.AddPolyline(points)


def draw_arc(layout, start_pt):
    """Arc mode: second edge, end point on it, then a point the arc passes through."""
    guid, geometry = pick_second_edge(layout)
    if not guid:
        return None
    try:
        gp = Rhino.Input.Custom.GetPoint()
        gp.SetCommandPrompt("Pick end point on second curve (Esc to change path type)")
        gp.Constrain(geometry, False)
        gp.DrawLineFromPoint(start_pt, True)
        gp.Get()
        if gp.CommandResult() != Rhino.Commands.Result.Success:
            return None
        end_pt = gp.Point()

        while True:
            gp2 = Rhino.Input.Custom.GetPoint()
            gp2.SetCommandPrompt("Pick a point on the arc (Esc to change path type)")

            def _preview(sender, e):
                try:
                    arc = Rhino.Geometry.Arc(start_pt, e.CurrentPoint, end_pt)
                    if arc.IsValid:
                        e.Display.DrawCurve(Rhino.Geometry.ArcCurve(arc), Color.Blue, 2)
                except Exception:
                    pass

            gp2.DynamicDraw += _preview
            gp2.Get()
            gp2.DynamicDraw -= _preview

            if gp2.CommandResult() != Rhino.Commands.Result.Success:
                return None

            arc_id = rs.AddArc3Pt(start_pt, end_pt, gp2.Point())
            if arc_id:
                return arc_id
            print("Those three points do not define a valid arc - pick again.")
    finally:
        rs.UnselectObject(guid)


def sample_drawn_curve(editor, guid):
    """The drawn curve as a point list: a polyline's vertices, or 8-64 samples of an arc."""
    curve = rs.coercecurve(guid)
    ok, polyline = curve.TryGetPolyline()
    if ok:
        return [[p.X, p.Y, 0.0] for p in polyline]

    length = rs.CurveLength(guid) or 0.0
    step = editor.mean_edge() * 0.1
    count = int(length / step) if step > 0 else 12
    count = max(8, min(64, count))
    return [[p.X, p.Y, 0.0] for p in rs.DivideCurve(guid, count)]


def note_extend(count):
    """Say that a short run is being carried on to the wall. Always ``True``."""
    print("  the run stopped on an interior patch edge ({} end(s)) -- carried on "
          "along the strip to the wall, because a line has to reach a boundary at "
          "both ends or close on itself.".format(count))
    return True


def add_polyedge(editor, layout):
    """Draw curves across the layout and cut each in as a new polyedge, until Esc. ``True`` if any was."""
    changed = False
    while True:
        first_guid = None
        temp_point = None
        obj_id = None
        try:
            picked = layout.pick_edge("Select the coarse edge to start FROM")
            if picked is None:
                return changed
            _edge, first_guid = picked
            rs.SelectObject(first_guid)

            start_pt = rs.GetPointOnCurve(first_guid, "Pick start point on first curve")
            if not start_pt:
                return changed
            temp_point = rs.AddPoint(start_pt)

            while True:
                mode = choose_mode()
                if mode is None:
                    return changed
                if mode == "Arc":
                    obj_id = draw_arc(layout, start_pt)
                else:
                    obj_id = draw_polyline(layout, start_pt)
                if obj_id:
                    break
                print("Cancelled - choose the path type again.")
        finally:
            if first_guid:
                rs.UnselectObject(first_guid)
            if temp_point:
                rs.DeleteObject(temp_point)

        points = sample_drawn_curve(editor, obj_id)
        rs.DeleteObject(obj_id)

        cut, _notes = edit(editor, lambda: editor.divide(points, extend=note_extend))
        if cut:
            note = editor.last_cut
            print("Line added: {} patch(es) cut, {} corner(s) inserted, layout "
                  "now has {} patch(es).".format(
                      note.get("faces"), note.get("corners"), note.get("faces_out")))
            changed = True
            redraw(editor, layout)
        else:
            mesh_ui.refuse("Add line",
                           "The line was NOT added.\n\n{}.\n\nThe layout is "
                           "unchanged.".format(editor.last_reason))
        sc.doc.Views.Redraw()


def remove_strip(editor, layout):
    """Delete picked strips, until Esc. ``True`` if any went."""
    changed = False
    while True:
        draw_strips(editor, layout)
        sc.doc.Views.Redraw()
        skey = layout.pick_strip("Pick a strip to remove (Esc to return to the menu)")
        if skey is None:
            layout.clear_strips()
            sc.doc.Views.Redraw()
            return changed

        removed, notes = edit(editor, lambda: editor.remove_strip(skey=skey))
        if not removed:
            mesh_ui.refuse(
                "Remove line",
                "Nothing was removed.\n\n{}.\n\nThe layout is unchanged."
                .format(notes.get("error", editor.last_reason)))
            continue

        print("strip {} removed: {} patch(es){}. The two sides welded "
              "together, so surrounding corners moved.".format(
                  notes["skey"], notes["faces"],
                  ", plus {} collateral strip(s) whose patches all lay inside "
                  "it".format(notes["collateral"]) if notes["collateral"] else ""))
        if notes["boundaries_lost"]:
            print("  NOTE: this COLLAPSED {} boundary/boundaries -- a hole of "
                  "the domain is no longer a hole of the layout. Pick 'undo' "
                  "from the menu above if that was not intended."
                  .format(notes["boundaries_lost"]))
        changed = True
        redraw(editor, layout)
        layout.clear_strips()


def add_strip(editor, layout):
    """Pick a run of existing corners and unzip it into a strip. ``True`` if added.

    Esc cancels the whole run; Enter adds it from the corners picked so far.
    """
    print("Pick corners in order along the line to unzip. Enter when done, Esc to cancel.")
    print("Both ends must be on the layout boundary, or the run must close.")
    polyedge = []
    cancelled = False
    try:
        while True:
            picked = layout.pick_vertex_or_finish(
                "Corner {} of the new line (Enter when done, Esc to cancel)"
                .format(len(polyedge) + 1))
            if picked is None:
                cancelled = True
                break
            if picked is FINISH:
                break
            vkey = picked
            if polyedge and vkey == polyedge[-1]:
                print("  that is the same corner again -- skipped.")
                continue
            if polyedge and vkey not in editor.mesh.halfedge[polyedge[-1]]:
                print("  corner {} is not joined to corner {} by an edge -- "
                      "skipped. Pick the next corner ALONG an edge.".format(
                          vkey, polyedge[-1]))
                continue
            polyedge.append(vkey)
            layout.show_path(polyedge)
            print("  {} corner(s): {}".format(len(polyedge), polyedge))
    finally:
        layout.clear_path()
        sc.doc.Views.Redraw()

    if cancelled:
        print("Add strip cancelled -- the layout is unchanged.")
        return False

    if len(polyedge) < 3:
        if polyedge:
            mesh_ui.refuse("Add strip",
                           "A strip needs more than two corners to run along.\n\n"
                           "The layout is unchanged.")
        return False

    ok, notes = edit(editor, lambda: editor.add_strip(polyedge))
    if not ok:
        mesh_ui.refuse("Add strip",
                       "The strip was NOT added.\n\n{}.\n\nThe layout is "
                       "unchanged.".format(editor.last_reason))
        return False

    print("Strip added: {} corner(s) unzipped, {} opened, layout now has {} "
          "patch(es).".format(notes.get("corners"), notes.get("opened"),
                              notes.get("faces_out")))
    redraw(editor, layout)
    return True


def divide_strip(editor, layout):
    """Pick strips by their ribbon and split each down the middle, until Esc. ``True`` if any was."""
    changed = False
    while True:
        draw_strips(editor, layout)
        sc.doc.Views.Redraw()
        skey = layout.pick_strip("Pick a strip to divide in two (Esc to return to the menu)")
        if skey is None:
            layout.clear_strips()
            sc.doc.Views.Redraw()
            return changed

        ok, notes = edit(editor, lambda: editor.divide(skey=skey))
        if not ok:
            mesh_ui.refuse("Divide strip",
                           "Nothing was divided.\n\n{}.\n\nThe layout is "
                           "unchanged.".format(editor.last_reason))
            continue
        print("Strip {} divided: {} patch(es) split, layout now has {} "
              "patch(es).".format(notes.get("skey"), notes.get("faces"),
                                  notes.get("faces_out")))
        changed = True
        redraw(editor, layout)
        layout.clear_strips()


def commit(editor, layout):
    """Hand the layout back. ``(committed, notes)``, with ``committed`` None if refused."""
    committed, notes = editor.commit()
    if committed is None:
        mesh_ui.refuse(
            "Commit layout",
            "The edited layout was NOT adopted.\n\n{}\n\nThe layout is left as "
            "you edited it, so the offending corner can be moved again."
            .format(notes.get("error", editor.last_reason)))
        return None, notes

    print("Layout adopted: {} patch(es) in, {} out, {} corner(s) snapped".format(
        notes.get("faces_in"), notes.get("faces_out"), notes.get("snapped")))
    if notes.get("off_wall"):
        print("  {} corner(s) sit on no wall -- expected only where a patch was "
              "deleted".format(notes["off_wall"]))
    odd = {n: count for n, count in (notes.get("sides") or {}).items()
           if n not in (3, 4)}
    if odd:
        print("  patches with {} side(s) -- one point per CORNER, not per "
              "sample".format(sorted(odd)))
    if notes.get("lost_curves"):
        print("  {} drawn curve(s) no longer match a coarse edge and were "
              "dropped -- those edges densify straight".format(notes["lost_curves"]))
    print("  commit path: {}".format(notes.get("path")))
    redraw(editor, layout)
    return committed, notes


def write_back(session, editor, committed, notes, wall_sampling):
    """Store the committed layout, with the shape of every edge, in the session."""
    outer_loop, inner_loops = read_boundary_loops(wall_sampling)
    curves, tally = coarse_edges_to_curves(
        committed,
        loops=[outer_loop] + inner_loops,
        polylines=editor.all_polylines)
    print("coarse edges: {}".format(tally))
    if tally.get("wall_missed"):
        print("  {} boundary edge(s) did NOT get their wall arc -- on a hole "
              "this is what makes one circle come out round and the next a "
              "polygon".format(tally["wall_missed"]))

    print("committed: {} patch(es) in, {} out, {} corner(s) snapped to a wall"
          .format(notes.get("faces_in"), notes.get("faces_out"),
                  notes.get("snapped")))
    if notes.get("off_wall"):
        print("  {} corner(s) sit on no wall -- expected only where a patch was "
              "deleted".format(notes["off_wall"]))
    print("note: any per-strip densities from CMD_densities and patterns from "
          "CMD_dense_pattern survive a move alone, but are dropped by a "
          "topology change (move_pole, add_polyedge, add_strip, remove_strip, "
          "divide_strip) -- "
          "CMD_densities re-derives them from the target in that case.")

    committed.set_edges_to_curves(curves)
    committed.set_shape_polylines(list(curves.values()))
    committed.attributes["edited"] = True
    session.coarse = committed
    session.record("Edit coarse mesh")
    print("layout written back to the session, and drawn")


def main():
    settings = get_settings()
    spacing = resolve_spacing(settings)
    wall_sampling = spacing * WALL_SAMPLING_FACTOR

    edit_layer = rs.AddLayer(name="TempEdit", parent="Skeleton", color=(0, 120, 200))
    rs.AddLayer(name="Polylines", parent="Skeleton")
    rs.AddLayer(name="EdgeCurves", parent="Skeleton", color=(0, 120, 200))

    outer, inners, guides, point_features = read_boundaries(spacing=spacing)
    coarse = read_layout()
    print("layout: {} patch(es), {} corner(s), {} pole(s)".format(
        coarse.number_of_faces(), coarse.number_of_vertices(), len(coarse.poles())))

    session = RhinoSession.current()
    field = session.field
    polylines = coarse.shape_polylines()
    print("separatrices: {} on the layout".format(len(polylines)))

    editor = CoarseEditor(coarse, field=field, loops=[outer] + inners,
                          polylines=polylines)
    layout = session.scene.add(editor.mesh, layer=edit_layer, show_faces=False,
                               show_vertices=True, show_edges=True)

    mesh_guids = rs.ObjectsByLayer(MESH_LAYER)
    rs.HideObjects(mesh_guids)

    redraw(editor, layout)
    lock_state = layout.unlock()
    rs.LayerVisible("Poles", False)
    rs.LayerVisible("Polylines", False)
    rs.LayerVisible("EdgeCurves", False)
    rs.LayerVisible("QuadMesh", False)

    committed, notes = None, {}
    try:
        while True:
            operation = mesh_ui.ask(
                "next", ["move_vertices", "move_pole", "add_polyedge", "add_strip",
                         "divide_strip", "remove_strip", "undo", "reset", "commit",
                         "exit"], "exit")

            if operation == "move_vertices":
                move_vertices(editor, layout)
            elif operation == "move_pole":
                move_pole(editor, layout)
                redraw(editor, layout)
            elif operation == "add_polyedge":
                add_polyedge(editor, layout)
            elif operation == "add_strip":
                add_strip(editor, layout)
            elif operation == "divide_strip":
                divide_strip(editor, layout)
            elif operation == "remove_strip":
                remove_strip(editor, layout)
            elif operation == "undo":
                undo(editor, layout)
            elif operation == "reset":
                editor.reset()
                redraw(editor, layout)
                print("Back to the layout this session started with.")
            elif operation == "commit":
                committed, notes = commit(editor, layout)
                if committed is not None:
                    break
            else:
                break
    finally:
        layout.relock(lock_state)
        layout.clear()
        session.scene.remove(layout)
        rs.ShowObjects([guid for guid in mesh_guids if rs.IsObject(guid)])
        sc.doc.Views.Redraw()

    if committed is None:
        print("nothing committed -- the layout is unchanged.")
    else:
        write_back(session, editor, committed, notes, wall_sampling)

    rs.LayerVisible("Poles", True)
    rs.LayerVisible("Polylines", True)
    rs.LayerVisible("EdgeCurves", True)
    rs.LayerVisible("QuadMesh", True)

    print("next: CMD_densities, then CMD_quad_mesh.")


if __name__ == "__main__":
    main()
