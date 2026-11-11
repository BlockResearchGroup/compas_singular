"""RhinoSession's logic, with the document replaced by a dictionary.

The one thing that needs a real Rhino -- that ``ModifyAttributes`` puts the
snapshot inside the command's undo record -- is what
``rhino_plugin/dev/CMD_probe_undo_storage.py`` checks. Everything around it is
here: what ``current()`` hands out and when it reloads.

Rhino's Ctrl+Z is simulated the only way it reaches us: the document holds an
earlier snapshot than the one this session wrote.
"""

import types

import compas
import pytest

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.rhino import document
from compas_singular.rhino import session as session_module
from compas_singular.rhino.session import RhinoSession
from compas_singular.session import SingularSession


PLATE = [[-6.0, -4.0, 0.0], [6.0, -4.0, 0.0], [6.0, 4.0, 0.0], [-6.0, 4.0, 0.0]]


class FakeDoc(object):
    """What RhinoSession reads from a RhinoDoc."""

    serial = 0

    def __init__(self):
        FakeDoc.serial += 1
        self.RuntimeSerialNumber = FakeDoc.serial
        self.stored = None  # (revision, packed snapshot), as the anchor holds it
        self.history = []  # every state it has held, for "Ctrl+Z"
        self.Views = types.SimpleNamespace(Redraw=lambda: None)

    def undo(self):
        self.history.pop()
        self.stored = self.history[-1] if self.history else None


@pytest.fixture(autouse=True)
def fake_document(monkeypatch):
    """Route document.py's three functions to FakeDoc.stored."""

    def read_revision(doc):
        return doc.stored[0] if doc.stored else None

    def read_snapshot(doc):
        return document.unpack(doc.stored[1]) if doc.stored else None

    def write_snapshot(doc, revision, text):
        doc.stored = (revision, document.pack(text))
        doc.history.append(doc.stored)

    monkeypatch.setattr(document, "read_revision", read_revision)
    monkeypatch.setattr(document, "read_snapshot", read_snapshot)
    monkeypatch.setattr(document, "write_snapshot", write_snapshot)
    monkeypatch.setattr(RhinoSession, "_sessions", {})


DRAWN = []
USER_TEXT = {}  # (guid, key) -> value, as Rhino's object user text


@pytest.fixture(autouse=True)
def fake_display(monkeypatch):
    """The display needs Rhino; record what it would have drawn instead."""
    del DRAWN[:]
    USER_TEXT.clear()
    monkeypatch.setattr(RhinoSession, "_draw_item", lambda self, name, item, options: DRAWN.append((options["layer"], item)))
    monkeypatch.setattr(
        session_module,
        "rs",
        types.SimpleNamespace(
            EnableRedraw=lambda flag: True,
            GetUserText=lambda guid, key: USER_TEXT.get((guid, key)),
            SetUserText=lambda guid, key, value: USER_TEXT.__setitem__((guid, key), value),
        ),
    )


@pytest.fixture(scope="module")
def coarse():
    return SkeletonDecomposition.from_boundary(PLATE, target_length=0.5).coarse_mesh()


def test_pack_round_trips_and_compresses():
    text = compas.json_dumps({"u": list(range(5000))})
    packed = document.pack(text)
    assert document.unpack(packed) == text
    assert packed.isascii() and len(packed) < len(text)


def test_a_new_document_gets_an_empty_session():
    doc = FakeDoc()
    session = RhinoSession.current(doc)
    assert session.revision is None and session.doc is doc
    assert all(getattr(session, item) is None for item in session.ITEMS)
    assert RhinoSession.current(doc) is session  # cached while nothing changes


def test_record_writes_into_the_document_and_keeps_the_session(coarse):
    doc = FakeDoc()
    session = RhinoSession.current(doc)
    session.coarse = coarse
    session.record("Coarse mesh")
    assert doc.stored[0] == session.revision
    assert RhinoSession.current(doc) is session  # its own write is not a change
    assert session.undo() is False  # undo is Rhino's here


def test_ctrl_z_is_picked_up_by_the_next_command(coarse):
    doc = FakeDoc()
    session = RhinoSession.current(doc)
    session.coarse = coarse
    session.record("Coarse mesh")
    session.settings.target_length = 0.25
    session.record("Settings")

    doc.undo()  # Rhino's Ctrl+Z
    after = RhinoSession.current(doc)
    assert after is not session  # reloaded, not the stale copy
    assert after.settings.target_length == 0.5
    assert after.coarse.number_of_faces() == coarse.number_of_faces()

    doc.undo()  # back past the first record
    empty = RhinoSession.current(doc)
    assert empty.coarse is None and empty.revision is None


def test_documents_do_not_share_a_session(coarse):
    first, second = FakeDoc(), FakeDoc()
    one = RhinoSession.current(first)
    one.coarse = coarse
    one.record("Coarse mesh")
    assert RhinoSession.current(second).coarse is None


def test_what_rhino_stores_loads_in_a_script(coarse):
    doc = FakeDoc()
    session = RhinoSession.current(doc)
    session.coarse = coarse
    session.record("Coarse mesh")
    loaded = compas.json_loads(document.unpack(doc.stored[1]))
    assert type(loaded) is SingularSession
    assert loaded.coarse.number_of_faces() == coarse.number_of_faces()


def test_record_redraws_only_what_was_replaced(coarse):
    doc = FakeDoc()
    session = RhinoSession.current(doc)
    session.coarse = coarse
    session.record("Coarse mesh")
    assert [layer for layer, _ in DRAWN] == ["TopologyProblem::Skeleton::Mesh"]

    del DRAWN[:]
    session.settings.target_length = 0.3
    session.record("Settings")
    assert DRAWN == []  # settings draw nothing

    del DRAWN[:]
    session.coarse = coarse.copy()  # the copy rule: a new object
    session.dense = None
    session.record("Densities")
    assert [layer for layer, _ in DRAWN] == ["TopologyProblem::Skeleton::Mesh"]

    del DRAWN[:]
    session.coarse = None
    session.record("Clear")
    assert DRAWN == [("TopologyProblem::Skeleton::Mesh", None)]  # None clears


def test_a_reload_trusts_what_rhino_restored(coarse):
    """After Ctrl+Z Rhino has already put back the display; drawing it again
    would only cost time."""
    doc = FakeDoc()
    session = RhinoSession.current(doc)
    session.coarse = coarse
    session.record("Coarse mesh")
    session.coarse = coarse.copy()
    session.record("Densities")
    doc.undo()

    del DRAWN[:]
    after = RhinoSession.current(doc)
    after.record("Settings")
    assert DRAWN == []


def test_a_drawn_object_leads_back_to_its_item(coarse):
    """What CMD_mesh_export picks by: the tag on the object, not its layer."""
    doc = FakeDoc()
    session = RhinoSession.current(doc)
    session.coarse = coarse
    USER_TEXT[("mesh-guid", RhinoSession.ITEM_KEY)] = "coarse"
    USER_TEXT[("dense-guid", RhinoSession.ITEM_KEY)] = "dense"  # but no dense mesh yet
    USER_TEXT[("odd-guid", RhinoSession.ITEM_KEY)] = "settings"  # not a displayed item
    assert session.item_of("mesh-guid") == ("coarse", coarse)
    assert session.item_of("dense-guid") is None
    assert session.item_of("odd-guid") is None
    assert session.item_of("untagged-guid") is None
