#! python3
# r: compas
# r: pydantic
# r: compas_singular

"""Step 1 -- show and edit this document's settings.

Input: the settings stored in the document. Output: the changed settings, stored on every change.
"""
import Rhino
import rhinoscriptsyntax as rs

from compas_singular.rhino.project import DEFAULT_SETTINGS
from compas_singular.rhino.project import get_settings
from compas_singular.rhino.project import set_settings


def show(settings):
    print("settings:")
    for key in sorted(settings):
        marker = "" if settings[key] == DEFAULT_SETTINGS.get(key) else "   (changed)"
        print("    {:<22} {!r}{}".format(key, settings[key], marker))


def on_off(value):
    return "On" if value else "Off"


def option_values(settings):
    """``(option, value shown after its '=')`` for the menu, in menu order."""
    spacing = settings["triangulation_spacing"]
    relax = settings.get("relax", "auto")
    return [
        ("Background_Spacing", "Thesis" if spacing is None else "{:g}".format(spacing)),
        ("Guide_Alignment", settings["guide_alignment"].capitalize()),
        ("Density_Mode", settings["density_mode"].capitalize()),
        ("Target_Length", "{:g}".format(settings["target_length"])),
        ("Target_Density", str(int(settings["target_density"]))),
        ("Field_Aware", on_off(settings["field_aware"])),
        ("Field_Symmetry", "Off" if settings.get("field_symmetry") is None else "Auto"),
        ("Relax", "Auto" if str(relax).lower() == "auto" else on_off(relax)),
        ("Defaults", None),
        ("Done", None),
    ]


def pick_option(settings):
    """The option clicked, lower-cased; ``"done"`` on Enter or Esc.

    Note: an option Rhino refuses with its value is added as the bare name.
    """
    go = Rhino.Input.Custom.GetOption()
    go.SetCommandPrompt("Edit settings")
    go.AcceptNothing(True)
    for name, value in option_values(settings):
        if value is None or go.AddOption(name, value) <= 0:
            go.AddOption(name)
    if go.Get() != Rhino.Input.GetResult.Option:
        return "done"
    return go.Option().EnglishName.lower()


def choose(message, current, strings):
    """One of ``strings``, lower-cased, or None on Esc."""
    answer = rs.GetString(message=message, defaultString=current, strings=strings)
    return answer.lower() if answer else None


def main():
    settings = get_settings()
    show(settings)

    while True:
        option = pick_option(settings)
        changed = True

        if option == "background_spacing":
            # 0 = the thesis value (eq. 4.1, from the domain's size).
            value = rs.GetReal("Background triangulation spacing (NOT the quad size), 0 = thesis value",
                               settings["triangulation_spacing"] or 0.0, 0.0)
            if value is not None:
                settings["triangulation_spacing"] = value or None
        elif option == "guide_alignment":
            answer = choose("Elements run ALONG (tangent) or ACROSS (perpendicular) the guides?",
                            settings["guide_alignment"], ["tangent", "perpendicular"])
            if answer:
                settings["guide_alignment"] = answer
        elif option == "density_mode":
            answer = choose("Strips with no picked density: by target LENGTH or fixed DENSITY?",
                            settings["density_mode"], ["length", "density"])
            if answer:
                settings["density_mode"] = answer
        elif option == "target_length":
            value = rs.GetReal("Target quad edge length", settings["target_length"], 1e-3)
            if value:
                settings["target_length"] = value
        elif option == "target_density":
            value = rs.GetInteger("Elements across every strip", int(settings["target_density"]), 1)
            if value:
                settings["target_density"] = value
        elif option == "field_aware":
            answer = choose("Integrate patch interiors from the field?",
                            "On" if settings["field_aware"] else "Off", ["On", "Off"])
            if answer:
                settings["field_aware"] = answer == "on"
        elif option == "field_symmetry":
            # None = off.
            current = "Off" if settings.get("field_symmetry") is None else "Auto"
            answer = choose("Detect and use symmetry in the frame field?", current, ["Auto", "Off"])
            if answer:
                settings["field_symmetry"] = "auto" if answer == "auto" else None
        elif option == "relax":
            # Stored as "auto", True or False.
            current = settings.get("relax", "auto")
            default = "Auto" if str(current).lower() == "auto" else ("On" if current else "Off")
            answer = choose("Relax the field solve? Auto = on when guides exist",
                            default, ["Auto", "On", "Off"])
            if answer:
                settings["relax"] = "auto" if answer == "auto" else (answer == "on")
        elif option == "defaults":
            if rs.GetString("Reset every setting to its default?", "No", ["Yes", "No"]) == "Yes":
                settings = dict(DEFAULT_SETTINGS)
            else:
                changed = False
        else:
            break

        if changed:
            set_settings(settings)
            show(settings)


if __name__ == "__main__":
    main()
