#! python3

# r: compas
# r: pydantic
# r: compas_singular

"""Step 1 -- reset the project.

Input: nothing. Output: empty TopologyProblem layers and an empty session with the settings kept.
"""
import rhinoscriptsyntax as rs

from compas_singular.rhino.project import LAYER_DATA
from compas_singular.rhino.project import ROOT
from compas_singular.rhino.session import RhinoSession


def reset_project():
    """Empty the project layers and recreate them, and reset the session. Returns the root layer."""
    # Read before the purge, which deletes the session's anchor object.
    settings = RhinoSession.current().settings

    project_folder = ROOT
    if rs.IsLayer(project_folder):
        if rs.IsLayer("Default"):
            rs.CurrentLayer("Default")
        else:
            print("Will not be able to clear problem folder. Set another layer as active.")

        sublayers = rs.LayerChildren(project_folder)

        for layer in sublayers:
            rs.PurgeLayer(layer)
        objects = rs.ObjectsByLayer(project_folder)
        rs.DeleteObjects(objects)
    else:
        rs.AddLayer(name=project_folder)

    for name, color in LAYER_DATA.values():
        rs.AddLayer(name=name, color=color)

    session = RhinoSession.current()
    session.clear(*session.ITEMS)
    session.settings = settings
    session.record("Start")
    print(settings.model_dump())

    print("CMD_start: project '{}' reset.".format(project_folder))
    print("note: this deletes every input, layout and mesh under '{}' -- nothing "
          "from an earlier session survives.".format(project_folder))
    print("next: CMD_boundary_selection to define a new domain.")
    return project_folder


if __name__ == "__main__":
    reset_project()
