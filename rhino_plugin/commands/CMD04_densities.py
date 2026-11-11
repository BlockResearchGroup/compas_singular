#! python3

# r: compas
# r: pydantic
# r: compas_rui
# r: compas_singular

"""Step 5 -- set the density of each strip.

Input: the session's layout and the density target. Output: the layout with its strip densities, in the session.
"""
import rhinoscriptsyntax as rs

from compas_singular.rhino import mesh_ui
from compas_singular.rhino.project import get_settings
from compas_singular.rhino.project import layer_path
from compas_singular.rhino.project import read_layout
from compas_singular.rhino.project import resolve_densities
from compas_singular.rhino.project import set_settings
from compas_singular.rhino.session import RhinoSession

DENSITY_LAYER = layer_path("Densities")


def draw(layout):
    """Every strip as a pickable ribbon, shaded by density, with its number on it.

    Note: strips are not re-collected, so a pick returns the key the density is stored under.
    """
    strip_densities = layout.mesh.get_strip_densities()
    layout.draw_strips(collect=False,
                       strip_colors=mesh_ui.density_colors(strip_densities),
                       labels={skey: d for skey, d in strip_densities.items()})


def main():
    session = RhinoSession.current()
    settings = get_settings()
    coarse = read_layout()
    resolve_densities(coarse, settings)

    print("layout: {} patch(es), {} strip(s); target {} ({})".format(
        coarse.number_of_faces(), len(list(coarse.strips())),
        settings[target_setting(settings)], settings["density_mode"]))

    layout = session.scene.add(coarse, layer=DENSITY_LAYER, show_faces=False)

    if rs.IsLayer(layer_path("Mesh")):
        rs.LayerVisible(layer_path("Mesh"), False)
    if rs.IsLayer(layer_path("QuadMesh")):
        rs.LayerVisible(layer_path("QuadMesh"), False)

    draw(layout)

    try:
        while True:
            option = (rs.GetString(message="Densities", defaultString="Finish",
                                   strings=["Pick", "Target_length", "Target_density", "Clear", "Finish"])
                      or "finish").lower()
            if option == "pick":
                while True:
                    skey = layout.pick_strip("Pick a strip to set its density (Esc to finish)")
                    if skey is None:
                        break
                    current = coarse.get_strip_density(skey)
                    value = rs.GetInteger("Elements across strip {}".format(skey),
                                          current, 1)
                    if value is None:
                        continue
                    coarse.set_strip_density(skey, int(value))
                    draw(layout)

            elif option == "target_length":
                value = rs.GetReal("Target quad edge length",
                                   settings["target_length"], 1e-3)
                if value:
                    settings["target_length"] = value
                    settings["density_mode"] = "length"
                    set_settings(settings)
                    coarse.set_strips_density_target(value)
                    draw(layout)

            elif option == "target_density":
                value = rs.GetInteger("Elements across every strip",
                                      settings["target_density"], 1)
                if value:
                    settings["target_density"] = value
                    settings["density_mode"] = "density"
                    set_settings(settings)
                    coarse.set_strips_density(int(value))
                    draw(layout)

            elif option == "clear":
                coarse.attributes['strips_density'] = {}
                resolve_densities(coarse, settings, verbose=False)
                draw(layout)
                print("picks cleared -- every strip follows the target.")

            else:
                break
    finally:
        layout.clear()
        session.scene.remove(layout)
        if rs.IsLayer(layer_path("Mesh")):
            rs.LayerVisible(layer_path("Mesh"), True)
        if rs.IsLayer(layer_path("QuadMesh")):
            rs.LayerVisible(layer_path("QuadMesh"), True)

    session.coarse = coarse
    session.record("Densities")
    print("densities: saved on the layout ({} strip(s))".format(
        len(coarse.get_strip_densities())))
    print("note: a topology-changing edit in CMD_edit_coarse_mesh (add_polyedge, "
          "add_strip, remove_strip, divide_strip) drops these -- run this step "
          "again afterward.")
    print("next: CMD_dense_pattern to set per-patch mesh patterns, or CMD_quad_mesh "
          "to densify.")


def target_setting(settings):
    """The settings key of the global rule currently in force."""
    return "target_density" if settings["density_mode"] == "density" else "target_length"


if __name__ == "__main__":
    main()
