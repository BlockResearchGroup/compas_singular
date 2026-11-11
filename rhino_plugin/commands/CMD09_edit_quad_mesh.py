#! python3

# r: compas
# r: pydantic
# r: compas_rui
# r: compas_singular

"""Optional, after step 6 -- edit the final mesh by hand.

Input: a mesh picked from TopologyProblem::QuadMesh or a sublayer. Output: the edited mesh on QuadMesh::Edited, on save.
"""
import rhinoscriptsyntax as rs
import scriptcontext as sc

from compas_singular.datastructures import QuadMesh
from compas_singular.editing import DenseMeshEditor
from compas_singular.rhino import mesh_ui
from compas_singular.rhino.helpers import bake_mesh
from compas_singular.rhino.helpers import mesh_from_rhino
from compas_singular.rhino.helpers import read_boundary_loops
from compas_singular.rhino.project import get_settings
from compas_singular.rhino.project import layer_path
from compas_singular.rhino.project import resolve_spacing
from compas_singular.rhino.session import RhinoSession

QUADMESH_LAYER = layer_path("QuadMesh")
EDIT_LAYER = QUADMESH_LAYER + "::Edit"
UNEDITED_LAYER = QUADMESH_LAYER + "::Unedited"
EDITED_LAYER = QUADMESH_LAYER + "::Edited"

#: Edge count above which the user is asked before the mesh is drawn as pickable objects.
DRAW_WARN_EDGES = 4000

OPERATIONS = ["move_vertex", "remove_vertex", "remove_edge", "remove_face",
              "draw_edges", "add_line", "remove_line", "relax", "undo", "reset",
              "save", "exit"]


def load_mesh():
    """The PICKED mesh, n-gons intact, with the object and layer it came from."""
    if not rs.IsLayer(QUADMESH_LAYER):
        raise RuntimeError(
            "No layer '{}' -- run CMD_quad_mesh first.".format(QUADMESH_LAYER))
    quad_layer = rs.LayerName(QUADMESH_LAYER, fullpath=True)

    def on_quad_layer(rhobj, geometry, component_index):
        layer = rs.ObjectLayer(rhobj)
        return layer == quad_layer or rs.IsLayerChildOf(quad_layer, layer)

    guid = rs.GetObject(
        message="Pick the mesh to edit, from 'QuadMesh' or a sublayer",
        filter=rs.filter.mesh, preselect=False, select=False,
        custom_filter=on_quad_layer, subobjects=False)
    if not guid:
        return None, None, None
    return (mesh_from_rhino(rs.coercemesh(guid), cls=QuadMesh),
            guid, rs.ObjectLayer(guid))


def load_walls():
    """The domain walls a moved boundary vertex is held on; ``[]`` if they cannot be read."""
    try:
        spacing = resolve_spacing(get_settings())
        outer_loop, inner_loops = read_boundary_loops(spacing * 0.25)
    except Exception as e:
        print("boundary walls unavailable ({}) -- a moved boundary vertex will "
              "NOT be held on the outline.".format(e))
        return []
    return [loop for loop in [outer_loop] + inner_loops if len(loop) >= 3]


def face_degrees(mesh):
    """``{number of sides: number of faces}``."""
    out = {}
    for fkey in mesh.faces():
        n = len(mesh.face_vertices(fkey))
        out[n] = out.get(n, 0) + 1
    return dict(sorted(out.items()))


def special_vertices(editor):
    """Singularities, and any vertex where faces only touch at a corner."""
    try:
        special = set(editor.mesh.singularities())
    except Exception:
        special = set()
    special.update(editor.non_manifold_vertices())
    return special


def redraw(editor, mesh_object):
    """Redraw only what changed since the last draw."""
    mesh_object.mesh = editor.mesh
    mesh_object.special = special_vertices(editor)
    return mesh_object.sync()


def edit(editor, mutate):
    """Run one ``editor`` call with undo bookkeeping. ``(ok, notes)``."""
    editor.push_undo()
    ok, notes = mutate()
    if not ok:
        editor.discard_last_undo()
    return ok, notes


def warn_non_manifold(notes):
    count = len(notes.get("non_manifold") or [])
    if count:
        print("  {} vertex/vertices now join faces that only touch at a corner "
              "(drawn in the singularity colour). add_line and remove_line are "
              "refused until that is fixed -- 'undo' takes it back.".format(count))


def move_vertex(editor, mesh_object):
    changed = False
    while True:
        vkey = mesh_object.pick_vertex("Select a vertex to move (Esc to return to the menu)")
        if vkey is None:
            return changed
        start = editor.mesh.vertex_coordinates(vkey)
        on_boundary = editor.is_vertex_on_boundary(vkey)
        held = on_boundary and bool(editor.walls)

        def project(point, start=start, held=held):
            # The vertex keeps its own z.
            point = [point[0], point[1], start[2]]
            return editor.project_to_wall(point) if held else point

        xyz = mesh_ui.drag_point(
            "New position for vertex {}{}".format(vkey, " (held on the wall)" if held else ""),
            start,
            neighbours=[editor.mesh.vertex_coordinates(n) for n in editor.mesh.vertex_neighbors(vkey)],
            project=project)
        if xyz is None:
            return changed
        ok, _notes = edit(editor, lambda: editor.move_vertex(vkey, xyz, project=False))
        if ok:
            changed = True
            redraw(editor, mesh_object)


def remove_vertex(editor, mesh_object):
    changed = False
    while True:
        vkey = mesh_object.pick_vertex("Select a vertex to remove with its faces (Esc to return)")
        if vkey is None:
            return changed
        ok, notes = edit(editor, lambda: editor.remove_vertex(vkey))
        if not ok:
            mesh_ui.refuse("Remove vertex", "Nothing was removed.\n\n{}.".format(notes["error"]))
            continue
        print("vertex {} removed with {} face(s); mesh now has {} face(s).".format(
            vkey, notes["faces_removed"], notes["faces"]))
        warn_non_manifold(notes)
        changed = True
        redraw(editor, mesh_object)


def remove_edge(editor, mesh_object):
    changed = False
    while True:
        picked = mesh_object.pick_edge("Select an edge to remove (Esc to return to the menu)")
        if picked is None:
            return changed
        edge, _guid = picked
        ok, notes = edit(editor, lambda: editor.remove_edge(edge))
        if not ok:
            mesh_ui.refuse("Remove edge", "Nothing was removed.\n\n{}.".format(notes["error"]))
            continue
        if notes["merged"] is None:
            print("boundary edge {}: its one face was removed; {} face(s) left.".format(
                edge, notes["faces"]))
        else:
            print("edge {} removed: faces of {} and {} sides merged into one of {}.".format(
                edge, notes["degrees"][0], notes["degrees"][1], notes["degree"]))
        warn_non_manifold(notes)
        changed = True
        redraw(editor, mesh_object)


def remove_face(editor, mesh_object):
    changed = False
    while True:
        point = rs.GetPoint("Click inside a face to remove it (Esc to return to the menu)")
        if point is None:
            return changed
        fkey = editor.face_at([point.X, point.Y, point.Z])
        if fkey is None:
            print("That point is not inside a face of the mesh -- click inside one.")
            continue
        ok, notes = edit(editor, lambda: editor.remove_face(fkey))
        if not ok:
            mesh_ui.refuse("Remove face", "Nothing was removed.\n\n{}.".format(notes["error"]))
            continue
        print("face with {} sides removed; {} face(s) left.".format(notes["degree"], notes["faces"]))
        warn_non_manifold(notes)
        changed = True
        redraw(editor, mesh_object)


def draw_edges(editor, mesh_object):
    changed = False
    print("Draw a polyline across the mesh; Enter to finish it. Snap to the "
          "points on '{}' to go through existing vertices.".format(EDIT_LAYER))
    while True:
        points = rs.GetPolyline(
            message1="First point of the line (Esc to return to the menu)",
            message2="Next point",
            message3="Next point, Enter to cut it in")
        if not points:
            return changed
        stroke = [[p.X, p.Y, p.Z] for p in points]
        ok, notes = edit(editor, lambda: editor.draw_edges(stroke))
        if not ok:
            mesh_ui.refuse("Draw edges", "Nothing was added.\n\n{}.".format(notes["error"]))
            continue
        print("line cut in: {} edge(s) split, {} face(s) split, {} new vertex/vertices; "
              "faces now {}.".format(notes["split_edges"], notes["split_faces"],
                                     notes["new_vertices"], face_degrees(editor.mesh)))
        left_out = []
        if notes["trimmed"]:
            left_out.append("{} loose end(s) inside a face or off the mesh".format(notes["trimmed"]))
        if notes["outside"]:
            left_out.append("{} stretch(es) outside the mesh".format(notes["outside"]))
        if notes["along_existing"]:
            left_out.append("{} stretch(es) along existing edges".format(notes["along_existing"]))
        if left_out:
            print("  left out: " + "; ".join(left_out))
        changed = True
        redraw(editor, mesh_object)


def add_line(editor, mesh_object):
    changed = False
    while True:
        picked = mesh_object.pick_edge("Pick an edge; the whole line through it gains a strip (Esc to return)")
        if picked is None:
            return changed
        edge, _guid = picked
        ok, notes = edit(editor, lambda: editor.add_line(edge))
        if not ok:
            mesh_ui.refuse("Add line", "No strip was added.\n\n{}".format(notes["error"]))
            continue
        print("strip added along a {} line of {} vertices: faces {} -> {}.".format(
            "closed" if notes["closed"] else "wall-to-wall", notes["polyedge_vertices"],
            notes["faces_before"], notes["faces_after"]))
        changed = True
        redraw(editor, mesh_object)


def remove_line(editor, mesh_object):
    changed = False
    while True:
        picked = mesh_object.pick_edge("Pick an edge; the whole strip through it is removed (Esc to return)")
        if picked is None:
            return changed
        edge, _guid = picked
        plan = editor.plan_strip_deletion(edge)
        if not plan["ok"]:
            mesh_ui.refuse("Remove line", "Nothing was removed.\n\n{}".format(plan["reason"]))
            continue

        preserve = False
        if plan["boundaries_lost"]:
            answer = mesh_ui.ask(
                "This strip COLLAPSES {} boundary/boundaries. Remove it?".format(plan["boundaries_lost"]),
                ["Yes", "PreserveBoundaries", "No"], "No")
            if not answer or answer.startswith("n"):
                print("Nothing removed.")
                continue
            preserve = answer.startswith("p")

        ok, notes = edit(editor, lambda: editor.remove_line(edge, preserve_boundaries=preserve))
        if not ok:
            mesh_ui.refuse("Remove line", "Nothing was removed.\n\n{}".format(notes["error"]))
            continue
        print("strip removed: {} face(s){}; faces {} -> {}. The two sides welded "
              "together, so the vertices beside it moved.".format(
                  notes["faces"],
                  ", plus {} collateral strip(s)".format(len(notes["collateral"])) if notes["collateral"] else "",
                  notes["faces_before"], notes["faces_after"]))
        changed = True
        redraw(editor, mesh_object)


def relax(editor, mesh_object):
    value = mesh_ui.ask_integer("Smoothing passes", editor.relax_iterations, 1)
    if not value:
        return False
    editor.push_undo()
    editor.relax(iterations=value)
    redraw(editor, mesh_object)
    print("relaxed: {} pass(es), every boundary held.".format(value))
    return True


def undo(editor, mesh_object):
    ok, notes = editor.undo()
    if not ok:
        print("Nothing to undo.")
        return False
    redraw(editor, mesh_object)
    print("Undone -- {} face(s). {} more undo(s) available.".format(notes["faces"], notes["remaining"]))
    return True


def save(editor):
    """Bake the mesh to ``QuadMesh::Edited``. Its guids, or ``[]`` on failure."""
    layer = mesh_ui.ensure_layer(EDITED_LAYER, (0, 120, 200))
    try:
        bake_mesh(editor.mesh, layer)
    except Exception as e:
        mesh_ui.refuse(
            "Save mesh",
            "The mesh was NOT saved.\n\n{}\n\nThe edit is still in memory, so try "
            "again. If '{}' is now empty, this session's starting mesh is on "
            "'{}'.".format(e, layer, UNEDITED_LAYER))
        return []
    guids = rs.ObjectsByLayer(layer) or []
    if not guids:
        mesh_ui.refuse(
            "Save mesh",
            "The mesh was NOT saved -- nothing landed on '{}'.\n\nThe edit is still "
            "in memory, so try again. This session's starting mesh is on "
            "'{}'.".format(layer, UNEDITED_LAYER))
        return []
    print("Saved to '{}': {} faces {}, {} vertices.".format(
        layer, editor.mesh.number_of_faces(), face_degrees(editor.mesh),
        editor.mesh.number_of_vertices()))
    return guids


def main():
    mesh, source_guid, source_layer = load_mesh()
    if mesh is None:
        print("Cancelled -- nothing was drawn or changed.")
        return

    print("editing the mesh on '{}': {} faces {}, {} vertices, {} edges".format(
        source_layer, mesh.number_of_faces(), face_degrees(mesh),
        mesh.number_of_vertices(), mesh.number_of_edges()))

    if mesh.number_of_edges() > DRAW_WARN_EDGES:
        answer = mesh_ui.ask(
            "This mesh has {} edges. Drawing them all as pickable objects will make "
            "Rhino slow. Continue?".format(mesh.number_of_edges()), ["Yes", "No"], "Yes")
        if not answer.startswith("y"):
            print("Cancelled -- nothing was drawn or changed.")
            return

    walls = load_walls()
    if not walls:
        print("  no domain walls loaded -- a moved boundary vertex will not be held "
              "on the outline.")

    editor = DenseMeshEditor(mesh, walls=walls)
    session = RhinoSession.current()
    mesh_object = session.scene.add(mesh, layer=EDIT_LAYER)

    mesh_ui.ensure_layer(UNEDITED_LAYER, (170, 170, 170))
    bake_mesh(mesh, UNEDITED_LAYER)
    rs.LayerVisible(UNEDITED_LAYER, False)
    print("this session's starting mesh kept on '{}'".format(UNEDITED_LAYER))

    rs.HideObjects([source_guid])

    try:
        redraw(editor, mesh_object)
        lock_state = mesh_object.unlock()
        try:
            while True:
                operation = mesh_ui.ask("next", OPERATIONS, "exit")
                if operation == "move_vertex":
                    move_vertex(editor, mesh_object)
                elif operation == "remove_vertex":
                    remove_vertex(editor, mesh_object)
                elif operation == "remove_edge":
                    remove_edge(editor, mesh_object)
                elif operation == "remove_face":
                    remove_face(editor, mesh_object)
                elif operation == "draw_edges":
                    draw_edges(editor, mesh_object)
                elif operation == "add_line":
                    add_line(editor, mesh_object)
                elif operation == "remove_line":
                    remove_line(editor, mesh_object)
                elif operation == "relax":
                    relax(editor, mesh_object)
                elif operation == "undo":
                    undo(editor, mesh_object)
                elif operation == "reset":
                    editor.reset()
                    redraw(editor, mesh_object)
                    print("Back to this session's starting mesh.")
                elif operation == "save":
                    if save(editor):
                        print("  CMD_quad_mesh regenerates from the coarse layout and "
                              "would discard this. Commands reading the mesh with "
                              "mesh_to_compas see polygons as triangle fans.")
                        break
                else:
                    if editor.edited:
                        answer = mesh_ui.ask("Unsaved edits. Save before leaving?", ["Yes", "No"], "Yes")
                        if answer.startswith("y"):
                            if not save(editor):
                                print("Leaving WITHOUT saving -- the save failed above.")
                        else:
                            print("Left unsaved -- '{}' is unchanged.".format(source_layer))
                    break
        finally:
            mesh_object.relock(lock_state)
    finally:
        mesh_object.clear()
        session.scene.remove(mesh_object)
        if rs.IsObject(source_guid):
            rs.ShowObjects([source_guid])
        sc.doc.Views.Redraw()


if __name__ == "__main__":
    main()
