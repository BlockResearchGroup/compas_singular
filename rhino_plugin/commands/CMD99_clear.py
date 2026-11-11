#! python3
# r: compas
# r: pydantic
# r: compas_singular

"""Utility -- clear chosen project layers.

Input: the layers ticked in the prompt. Output: those layers emptied, and their items dropped from the session.
"""
import rhinoscriptsyntax as rs

from compas_singular.rhino.helpers import clear_layer
from compas_singular.rhino.session import RhinoSession

items = ("InputBoundaries", "No", "Yes"), ("FrameField", "No", "Yes"), ("Skeleton", "No", "Yes"), ("QuadMesh", "No", "Yes"), ("TrashBin", "No", "Yes")
defaults = (False, True, True, True, True)
#: The session item cleared with each layer.
SESSION_ITEMS = {"InputBoundaries": "domain", "FrameField": "field",
                 "Skeleton": "coarse", "QuadMesh": "dense"}


def main():
    to_clear = rs.GetBoolean("Clear?", items, defaults) or [False] * len(items)

    objects_deleted_count = 0
    cleared = []
    for i in range(len(items)):
        if to_clear[i]:
            objects_deleted_count += clear_layer(items[i][0], True)
            cleared.append(items[i][0])

    names = [SESSION_ITEMS[layer] for layer in cleared if layer in SESSION_ITEMS]
    if names:
        session = RhinoSession.current()
        session.clear(*names)
        session.record("Clear")

    print(f"Deleted {objects_deleted_count} object(s) from: {', '.join(cleared) or 'nothing'}.")
    print("note: clearing an earlier layer (e.g. InputBoundaries or Skeleton) leaves any "
          "later layer (Skeleton, QuadMesh) describing a domain or layout that no longer exists.")
    print("next: CMD_boundary_selection to start a new domain, or CMD_coarse_mesh / "
          "CMD_read_coarse_mesh if the domain is untouched.")


if __name__ == "__main__":
    main()
