#! python3

# r: compas
# r: pydantic
# r: compas_singular

"""Utility -- save the whole project to one JSON file.

Input: the session. Output: a JSON file with settings, domain, layout, field and dense mesh.
"""
import os

import rhinoscriptsyntax as rs

from compas_singular.rhino.session import RhinoSession


def main():
    session = RhinoSession.current()
    doc = session.doc
    folder = os.path.dirname(doc.Path) if doc.Path else None
    name = (os.path.splitext(doc.Name)[0] if doc.Name else "project") + ".json"
    path = rs.SaveFileName("Save the project", "JSON (*.json)|*.json||", folder, name)
    if not path:
        print("Cancelled -- nothing saved.")
        return
    session.dump(path)
    items = [item for item in session.ITEMS if getattr(session, item) is not None]
    print("project saved to {}: settings, {}".format(path, ", ".join(items) or "no items yet"))


if __name__ == "__main__":
    main()
