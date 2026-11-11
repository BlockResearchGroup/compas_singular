#! python3

# r: compas
# r: pydantic
# r: compas_singular

"""Utility -- open a project JSON into this document.

Input: a JSON file written by CMD_session_save. Output: its settings, layout, field and mesh in the session, drawn.
"""
import rhinoscriptsyntax as rs
from Rhino.Geometry import Point3d

from compas_singular.rhino.helpers import bake_polylines
from compas_singular.rhino.project import ensure_layers
from compas_singular.rhino.project import layer_path
from compas_singular.rhino.session import RhinoSession
from compas_singular.session import SingularSession

INPUT_LAYERS = ("Outer", "Inner", "Guides", "PointFeatures")


def draw_domain(domain):
    """The inputs onto their layers, if every one of them is empty. Whether it drew."""
    if any(rs.ObjectsByLayer(layer_path(name)) for name in INPUT_LAYERS):
        return False
    closed = [loop + loop[:1] for loop in [domain.outer] + domain.inners if loop]
    bake_polylines(closed[:1], layer_path("Outer"), clear_existing=False)
    bake_polylines(closed[1:], layer_path("Inner"), clear_existing=False)
    bake_polylines(domain.guides, layer_path("Guides"), clear_existing=False)
    for pole in domain.poles:
        guid = rs.AddPoint(Point3d(*pole))
        if guid:
            rs.ObjectLayer(guid, layer_path("PointFeatures"))
    return True


def main():
    path = rs.OpenFileName("Open a project", "JSON (*.json)|*.json||")
    if not path:
        print("Cancelled -- nothing opened.")
        return
    try:
        loaded = SingularSession.load(path)
    except TypeError as error:
        rs.MessageBox("{}\n\nis not a compas_singular project:\n{}".format(path, error), 0,
                      "Open project")
        return

    session = RhinoSession.current()
    session.take(loaded)
    ensure_layers()
    if session.domain is not None:
        if draw_domain(session.domain):
            print("domain drawn onto the input layers")
        else:
            print("this document has its own inputs -- they are kept, and the "
                  "project's domain is not drawn")
    session.record("Open project")
    items = [item for item in session.ITEMS if getattr(session, item) is not None]
    print("project opened from {}: settings, {}".format(path, ", ".join(items) or "no items"))


if __name__ == "__main__":
    main()
