"""The session: one serialisable object holding a whole project, no Rhino needed.

What these pin, in the order it matters:

* every item comes back, of the right class, with the attributes a bake drops;
* each item comes back ONCE -- the dense mesh is not also inside the layout;
* a mesh decoded as part of a session gets the same integer keys as one read
  with ``load_from_json``;
* undo / redo step through recorded states;
* a file written before the ``guide_allignment`` rename still loads.
"""

import compas
import pytest

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.framefield.field import CrossField
from compas_singular.session import SingularSession
from compas_singular.settings import Settings
from compas_singular.symmetry import Domain


PLATE = [[-6.0, -4.0, 0.0], [6.0, -4.0, 0.0], [6.0, 4.0, 0.0], [-6.0, 4.0, 0.0]]
CABLE = [[[-6.0, -2.0, 0.0], [-1.0, -1.0, 0.0], [3.0, 2.5, 0.0], [6.0, 3.0, 0.0]]]
POLE = [-3.5, 0.0, 0.0]


@pytest.fixture(scope="module")
def project():
    d = SkeletonDecomposition.from_boundary(PLATE, point_features=[POLE], target_length=0.5)
    coarse = d.coarse_mesh()
    coarse.collect_polyedges()
    coarse.set_strips_density_target(t=0.5)
    coarse.set_strip_density(sorted(coarse.strips())[0], 7)
    dense = coarse.densification()
    field = CrossField.from_boundary(PLATE, guides=CABLE, target_length=0.8)
    domain = Domain(PLATE, guides=CABLE, poles=[POLE])
    return SingularSession(domain=domain, coarse=coarse, field=field, dense=dense)


def test_every_item_comes_back(project, tmp_path):
    back = SingularSession.load(project.dump(str(tmp_path / "project.json")))

    assert type(back.coarse) is CoarsePseudoQuadMesh
    assert back.coarse.number_of_faces() == project.coarse.number_of_faces()
    assert back.coarse.attributes["strips_density"] == project.coarse.attributes["strips_density"]
    assert len(back.coarse.attributes["face_pole"]) == len(project.coarse.attributes["face_pole"])

    assert back.dense.number_of_faces() == project.dense.number_of_faces()

    assert back.field.u == project.field.u  # exactly, as field.json does
    assert back.field.singularities() == project.field.singularities()

    assert isinstance(back.domain, Domain)
    assert back.domain.to_data() == project.domain.to_data()
    assert back.settings == project.settings


def test_each_item_is_stored_once(project):
    """compas writes a Data object every time it meets it. The layout used to
    carry the dense mesh it produced, so the session wrote it twice and a load
    gave back two meshes with one guid."""
    assert project.coarse.get_quad_mesh() is project.dense  # the live link exists...
    text = compas.json_dumps(project)
    assert text.count(str(project.dense.guid)) == 1  # ...but is written once

    back = compas.json_loads(text)
    assert back.coarse.get_quad_mesh() is None  # derived; densification rebuilds it


def test_a_mesh_inside_a_session_gets_integer_keys(project):
    """Decoded without load_from_json, which used to be the only place the
    keys were repaired."""
    back = compas.json_loads(compas.json_dumps(project))
    attributes = back.coarse.attributes
    for name in ("strips", "strips_density", "polyedges"):
        assert attributes[name], name  # an empty table proves nothing
        assert all(isinstance(key, int) for key in attributes[name]), name
    assert back.coarse.get_strip_density(sorted(back.coarse.strips())[0]) == 7


def test_the_layout_carries_its_shape_polylines(project):
    """What the edge curves are rebuilt from travels with the layout."""
    coarse = project.coarse.copy()
    traced = [[[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 0.0]]]
    coarse.set_shape_polylines(traced)
    back = compas.json_loads(compas.json_dumps(SingularSession(coarse=coarse)))
    assert back.coarse.shape_polylines() == traced
    assert back.coarse.copy().shape_polylines() == traced
    coarse.set_shape_polylines(None)
    assert coarse.shape_polylines() == []


def test_a_session_file_is_always_a_singular_session(project):
    """A subclass (RhinoSession) dumps as the base class, so a script with no
    Rhino can load what Rhino wrote."""

    class Subclass(SingularSession):
        pass

    text = compas.json_dumps(Subclass(settings=Settings(target_length=0.3)))
    back = compas.json_loads(text)
    assert type(back) is SingularSession
    assert back.settings.target_length == 0.3


def test_undo_and_redo_step_through_recorded_states(project):
    session = SingularSession(coarse=project.coarse)
    session.record("start")
    faces = session.coarse.number_of_faces()

    session.settings.target_length = 0.25
    session.coarse = None
    session.record("clear")

    assert session.undo()
    assert session.coarse.number_of_faces() == faces
    assert session.settings.target_length == 0.5
    assert not session.undo()  # stops at the first record

    assert session.redo()
    assert session.coarse is None
    assert session.settings.target_length == 0.25
    assert not session.redo()

    assert session.history == ["start", "clear"]


def test_recording_after_an_undo_drops_the_redo(project):
    session = SingularSession()
    session.record("a")
    session.record("b")
    session.undo()
    session.record("c")
    assert session.history == ["a", "c"]
    assert not session.redo()


def test_undo_keeps_the_session_object(project):
    """A Rhino scene holds on to the session; undo must not replace it."""
    session = SingularSession(coarse=project.coarse)
    session.record("start")
    session.coarse = None
    session.record("clear")
    before = id(session)
    session.undo()
    assert id(session) == before


def test_clear_empties_items_and_keeps_settings(project):
    session = SingularSession(settings=Settings(target_length=0.3), coarse=project.coarse, dense=project.dense)
    session.clear("dense")
    assert session.dense is None and session.coarse is project.coarse
    session.clear()  # names nothing, clears nothing
    assert session.coarse is project.coarse
    session.clear(*session.ITEMS)
    assert all(getattr(session, item) is None for item in SingularSession.ITEMS)
    assert session.settings.target_length == 0.3
    with pytest.raises(ValueError):
        session.clear("settings")


def test_old_settings_keys_still_load():
    old = {
        "triangulation_spacing": 0.3,
        "guide_allignment": "perpendicular",
        "target_length": 0.5,
        "target_density": 5,
        "density_mode": "length",
        "field_aware": True,
        "density_key": "density_key",
        "field_symmetry": None,
        "relax": "auto",
    }
    settings = Settings.model_validate(old)
    assert settings.guide_alignment == "perpendicular"
    assert settings.field_symmetry is None
    assert "density_key" not in settings.model_dump()
    assert "guide_allignment" not in settings.model_dump()


def test_background_spacing_defaults_to_the_thesis_value():
    from compas_singular.geometry.polyline import bounding_box_diagonal
    from compas_singular.rhino.project import resolve_spacing

    settings = Settings().model_dump()
    assert settings["triangulation_spacing"] is None
    assert resolve_spacing(settings, loops=[PLATE]) == pytest.approx(0.02 * bounding_box_diagonal(PLATE))
    settings["triangulation_spacing"] = 0.3
    assert resolve_spacing(settings, loops=[PLATE]) == 0.3


def test_loading_something_else_is_refused(tmp_path):
    path = str(tmp_path / "domain.json")
    compas.json_dump(Domain(PLATE), path)
    with pytest.raises(TypeError):
        SingularSession.load(path)
