#! python3

# r: compas
# r: pydantic
# r: compas_rui
# r: compas_singular

"""Step 5 -- set the dense mesh pattern of each coarse patch.

Input: the session's layout. Output: the layout with its patch patterns, in the session.
"""
import rhinoscriptsyntax as rs

from compas_singular.datastructures.mesh_quad_coarse.patterns import PATTERNS
from compas_singular.rhino.project import layer_path
from compas_singular.rhino.project import read_layout
from compas_singular.rhino.session import RhinoSession

PATTERN_LAYER = layer_path("Patterns")
HIDDEN_WHILE_PICKING = ("Skeleton", "QuadMesh")


def set_patterns(layout, pattern, faces):
    """``pattern`` on ``faces``; redrawn if anything changed. Whether it did."""
    changed = [face for face in faces if layout.mesh.get_face_pattern(face) != pattern]
    for face in changed:
        layout.mesh.set_face_pattern(face, pattern)
    if changed:
        layout.redraw()
        print("'{}' set on {} patch(es): {}".format(
            pattern, len(changed), ", ".join(str(face) for face in changed)))
    return bool(changed)


def main():
    session = RhinoSession.current()
    coarse = read_layout()
    layout = session.scene.add(coarse, layer=PATTERN_LAYER, show_patterns=True)

    hidden = [layer_path(name) for name in HIDDEN_WHILE_PICKING if rs.IsLayer(layer_path(name))]
    for layer in hidden:
        rs.LayerVisible(layer, False)

    changed = False
    try:
        layout.redraw()
        while True:
            pattern = rs.GetString("Pattern to apply", "Finish", PATTERNS + ["Finish"])
            pattern = (pattern or "Finish").lower()
            if pattern == "finish":
                break

            mode = (rs.GetString("Apply '{}' to".format(pattern), "Pick",
                                 ["Pick", "Pick_multiple", "All"]) or "").lower()
            if mode == "all":
                changed |= set_patterns(layout, pattern, list(coarse.faces()))
            elif mode == "pick_multiple":
                changed |= set_patterns(layout, pattern, layout.pick_faces() or [])
            elif mode == "pick":
                while True:
                    face = layout.pick_face("Pick a patch for '{}' (Esc when done)".format(pattern))
                    if face is None:
                        break
                    changed |= set_patterns(layout, pattern, [face])
    finally:
        layout.clear()
        session.scene.remove(layout)
        for layer in hidden:
            rs.LayerVisible(layer, True)

    if not changed:
        print("patterns: nothing changed.")
        return
    session.coarse = coarse
    session.record("Dense patterns")
    custom = sum(1 for p in coarse.dense_patterns().values() if p != "ortho")
    print("patterns: {} patch(es) set away from the default 'ortho'".format(custom))
    print("note: re-running CMD_coarse_mesh, or a topology-changing edit in "
          "CMD_edit_coarse_mesh, replaces the layout and drops these patterns.")
    print("next: CMD_quad_mesh to densify.")


if __name__ == "__main__":
    main()
