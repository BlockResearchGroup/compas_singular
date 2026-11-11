"""Tests for the shared editing base, :class:`MeshEditor`.

Pins the four things the two hand-editors had duplicated and that a subclass is
now entitled to rely on: working-copy semantics, the boundary rule (holes
included), the strip-deletion template with its subclass gate, and the in-place
transplant.
"""

import pytest

from compas_singular.datastructures import QuadMesh
from compas_singular.editing.editor import MeshEditor


def _grid(n):
    """An ``n`` x ``n`` quad grid, plus ``{(i, j): vertex key}``."""
    vertices, index = [], {}
    for i in range(n + 1):
        for j in range(n + 1):
            index[i, j] = len(vertices)
            vertices.append([float(i), float(j), 0.0])
    faces = [[index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]] for i in range(n) for j in range(n)]
    return QuadMesh.from_vertices_and_faces(vertices, faces), index


def _annulus():
    """3 x 3 grid with the centre face removed: an outer wall and a hole."""
    vertices, index = [], {}
    for i in range(4):
        for j in range(4):
            index[i, j] = len(vertices)
            vertices.append([float(i), float(j), 0.0])
    faces = [[index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]] for i in range(3) for j in range(3) if not (i == 1 and j == 1)]
    return QuadMesh.from_vertices_and_faces(vertices, faces), index


@pytest.fixture
def grid4():
    return _grid(4)


# ==============================================================================
# working-copy semantics
# ==============================================================================


def test_the_callers_mesh_is_not_touched_until_a_transplant(grid4):
    mesh, index = grid4
    vkey = index[2, 2]
    before = mesh.vertex_coordinates(vkey)

    editor = MeshEditor(mesh)
    ok, _notes = editor.move_vertex(vkey, [9.0, 9.0, 0.0])

    assert ok
    assert editor.mesh.vertex_coordinates(vkey) == [9.0, 9.0, 0.0]
    assert mesh.vertex_coordinates(vkey) == before, "the edit leaked into the target"
    assert editor.target is mesh


def test_work_on_copy_false_edits_in_place(grid4):
    mesh, index = grid4
    editor = MeshEditor(mesh, work_on_copy=False)
    assert editor.mesh is mesh


# ==============================================================================
# moving, and the boundary rule
# ==============================================================================


def test_move_vertex_returns_ok_and_notes(grid4):
    mesh, index = grid4
    editor = MeshEditor(mesh)
    ok, notes = editor.move_vertex(index[2, 2], [2.5, 2.5, 0.0])
    assert ok and notes["moved"] == index[2, 2] and notes["projected"] is False


def test_a_refusal_reports_the_same_string_twice(grid4):
    """``last_reason`` and ``notes['error']`` must not be able to disagree."""
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    ok, notes = editor.move_vertex(9999, [0.0, 0.0, 0.0])
    assert not ok
    assert notes["error"] == editor.last_reason
    assert "9999" in editor.last_reason


def test_an_interior_vertex_is_never_projected(grid4):
    mesh, index = grid4
    wall = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]
    editor = MeshEditor(mesh, walls=[wall])
    _ok, notes = editor.move_vertex(index[2, 2], [2.5, 2.5, 0.0])
    assert notes["projected"] is False
    assert editor.mesh.vertex_coordinates(index[2, 2]) == [2.5, 2.5, 0.0]


def test_a_boundary_vertex_is_pulled_back_onto_its_wall(grid4):
    mesh, index = grid4
    wall = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]
    editor = MeshEditor(mesh, walls=[wall])
    _ok, notes = editor.move_vertex(index[2, 0], [2.5, -3.0, 0.0])
    assert notes["projected"] is True
    x, y, _z = editor.mesh.vertex_coordinates(index[2, 0])
    assert y == pytest.approx(0.0), "it left the wall"
    assert x == pytest.approx(2.5)


def test_a_hole_vertex_counts_as_a_boundary_vertex():
    """The plural accessor. The singular form misses every hole.

    Measured on this annulus: ``vertices_on_boundary()`` returns 12 of the 16
    boundary vertices and none of the hole's four.
    """
    mesh, index = _annulus()
    editor = MeshEditor(mesh)
    hole = [index[1, 1], index[2, 1], index[2, 2], index[1, 2]]

    assert set(hole).isdisjoint(set(mesh.vertices_on_boundary())), "the singular accessor unexpectedly sees the hole -- this test is stale"
    for vkey in hole:
        assert editor.is_vertex_on_boundary(vkey)
    assert len(editor.boundary_vertices()) == 16


# ==============================================================================
# undo
# ==============================================================================


def test_reset_restores_topology_not_just_coordinates(grid4):
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    faces = editor.mesh.number_of_faces()

    ok, _ = editor.remove_strip(edge=list(editor.mesh.edges())[0])
    assert ok
    assert editor.mesh.number_of_faces() < faces

    ok, notes = editor.reset()
    assert ok and notes["faces"] == faces
    assert editor.mesh.number_of_faces() == faces


def test_snapshot_moves_the_point_reset_returns_to(grid4):
    mesh, index = grid4
    editor = MeshEditor(mesh)
    editor.move_vertex(index[2, 2], [2.5, 2.5, 0.0])
    editor.snapshot()
    editor.move_vertex(index[2, 2], [3.5, 3.5, 0.0])
    editor.reset()
    assert editor.mesh.vertex_coordinates(index[2, 2]) == [2.5, 2.5, 0.0]


# ==============================================================================
# thesis Eq 5.6
# ==============================================================================


def test_check_strip_count_holds_on_a_grid(grid4):
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    ok, counted, expected = editor.check_strip_count()
    assert ok, "S_open=%d but E-2F=%d" % (counted, expected)
    assert counted == 8


def test_check_strip_count_does_not_leave_strip_data_behind(grid4):
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    editor.check_strip_count()
    assert not editor.mesh.attributes["strips"], "it collected onto the live mesh"


# ==============================================================================
# the strip-deletion template
# ==============================================================================


def test_plan_and_delete_agree(grid4):
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    edge = list(editor.mesh.edges())[0]

    plan = editor.plan_strip_deletion(edge=edge)
    assert plan["ok"] and plan["skey"] is not None and plan["faces"] > 0

    ok, notes = editor.remove_strip(edge=edge)
    assert ok
    assert notes["skey"] == plan["skey"]
    assert notes["faces"] == plan["faces"]
    assert editor.mesh.number_of_faces() == plan["faces_after"]


def test_a_strip_can_be_addressed_by_key_as_well_as_by_edge(grid4):
    """A front end that lets the user pick the strip has a key already."""
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    skey = editor.strip_through(list(editor.mesh.edges())[0])
    ok, notes = editor.remove_strip(skey=skey)
    assert ok and notes["skey"] == skey


def test_an_unknown_edge_is_refused_without_changing_anything(grid4):
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    faces = editor.mesh.number_of_faces()
    ok, notes = editor.remove_strip(edge=(9998, 9999))
    assert not ok
    assert notes["error"] == editor.last_reason
    assert editor.mesh.number_of_faces() == faces


def test_the_subclass_gate_can_refuse_and_costs_nothing(grid4):
    """``_gate`` is the whole reason this is a template rather than copy-paste."""

    class Fussy(MeshEditor):
        def _gate(self, work):
            return False, "no strip shall pass"

    mesh, _ = grid4
    editor = Fussy(mesh)
    faces = editor.mesh.number_of_faces()

    plan = editor.plan_strip_deletion(edge=list(editor.mesh.edges())[0])
    assert not plan["ok"] and plan["reason"] == "no strip shall pass"

    ok, notes = editor.remove_strip(edge=list(editor.mesh.edges())[0])
    assert not ok and notes["error"] == "no strip shall pass"
    assert editor.mesh.number_of_faces() == faces, "a refusal changed the mesh"


def test_planning_never_mutates(grid4):
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    faces = editor.mesh.number_of_faces()
    vertices = editor.mesh.number_of_vertices()
    for edge in list(editor.mesh.edges()):
        editor.plan_strip_deletion(edge=edge)
    assert editor.mesh.number_of_faces() == faces
    assert editor.mesh.number_of_vertices() == vertices


# ==============================================================================
# the transplant
# ==============================================================================


def test_transplant_keeps_the_targets_identity(grid4):
    mesh, index = grid4
    editor = MeshEditor(mesh)
    editor.move_vertex(index[2, 2], [2.5, 2.5, 0.0])

    out = editor._transplant(editor.mesh)

    assert out is mesh, "the caller holds a different object"
    assert mesh.vertex_coordinates(index[2, 2]) == [2.5, 2.5, 0.0]


def test_transplant_carries_vertex_and_face_keys(grid4):
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    keys = set(editor.mesh.vertices())
    fkeys = set(editor.mesh.faces())
    editor._transplant(editor.mesh)
    assert set(mesh.vertices()) == keys
    assert set(mesh.faces()) == fkeys


def test_transplant_resets_the_derived_dense_mesh(grid4):
    """A dense mesh built from the pre-edit layout must not survive the edit."""
    mesh, _ = grid4
    mesh.attributes["quad_mesh"] = "STALE"
    mesh.attributes["vertex_coarse_to_dense"] = {0: 7}
    editor = MeshEditor(mesh)
    editor._transplant(editor.mesh)
    assert mesh.attributes["quad_mesh"] is None
    assert mesh.attributes["vertex_coarse_to_dense"] == {}


def test_transplant_drops_strip_data_by_default_and_carries_it_on_request(grid4):
    """Thesis 5.3.3: the grammar preserves strip labels, a rebuild does not."""
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    editor.mesh.collect_strips()
    editor.mesh.attributes["strips_density"] = {0: 9}

    editor._transplant(editor.mesh, carry_strip_data=True)
    assert mesh.attributes["strips_density"] == {0: 9}
    assert mesh.attributes["strips"]

    editor._transplant(editor.mesh, carry_strip_data=False)
    assert mesh.attributes["strips_density"] == {}
    assert mesh.attributes["strips"] == {}


def test_transplant_keeps_the_route_of_the_target(grid4):
    mesh, _ = grid4
    mesh.attributes["route"] = "skeleton"
    editor = MeshEditor(mesh)
    editor._transplant(editor.mesh)
    assert mesh.attributes["route"] == "skeleton"


# ==============================================================================
# snap_to_loops -- the same boundary rule, on the commit side
# ==============================================================================
#
# The editor's preview and the commit's snap must agree about which vertices are
# on a boundary, or a dragged corner is shown in one place and written in another.
# Both used the SINGULAR accessor, so they agreed with each other while both
# skipped every hole. The base class fixed the preview half; this is the other.

from compas_singular.editing.rebuild import snap_to_loops  # noqa: E402


def test_snap_to_loops_pulls_a_hole_corner_back_onto_its_wall():
    mesh, index = _annulus()
    outer = [[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0]]
    hole = [[1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0]]

    vkey = index[1, 1]
    mesh.vertex_attributes(vkey, "xyz", [1.3, 1.3, 0.0])  # drifted into the hole

    moved, off = snap_to_loops(mesh, [outer, hole], 1.0)

    assert moved >= 1, "the hole corner was treated as interior and skipped"
    assert off == 0
    x, y, _z = mesh.vertex_coordinates(vkey)
    # back on the hole's rim: one of the two coordinates is 1.0 again
    assert min(abs(x - 1.0), abs(y - 1.0)) == pytest.approx(0.0, abs=1e-9)


def test_snap_to_loops_still_leaves_interior_vertices_alone():
    """Eligibility stays TOPOLOGICAL -- proximity is never enough."""
    mesh, index = _grid(4)
    wall = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]
    vkey = index[1, 1]
    mesh.vertex_attributes(vkey, "xyz", [0.1, 0.1, 0.0])  # very close to the wall
    before = mesh.vertex_coordinates(vkey)

    snap_to_loops(mesh, [wall], 1.0)

    assert mesh.vertex_coordinates(vkey) == before, "an interior vertex was dragged"


# ==============================================================================
# preserve_boundaries only acts when a boundary is actually at risk
# ==============================================================================
#
# ``strips_to_split_to_prevent_boundary_collapse`` returns an EMPTY dict when
# nothing needs splitting and ``None`` when a collapse cannot be prevented.
# Treating the two the same way meant asking to preserve boundaries REFUSED a
# deletion that would not have cost a boundary -- with a message saying one would
# collapse. Nothing exercised it; the agent's remove_line tool is the one caller
# that can reach it.


def test_preserve_boundaries_does_not_refuse_a_harmless_deletion(grid4):
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    edge = list(editor.mesh.edges())[0]

    plain = editor.plan_strip_deletion(edge=edge)
    assert plain["ok"] and not plain["boundaries_lost"], "this fixture no longer deletes without losing a boundary"

    guarded = editor.plan_strip_deletion(edge=edge, preserve_boundaries=True)
    assert guarded["ok"], guarded["reason"]
    assert guarded["to_split"] == {}, "it pre-split when nothing was at risk"


def test_the_plan_reports_what_it_would_have_to_split_either_way(grid4):
    """Informational: a front end says what saving the hole would cost first."""
    mesh, _ = grid4
    editor = MeshEditor(mesh)
    for edge in list(editor.mesh.edges()):
        plan = editor.plan_strip_deletion(edge=edge)
        if plan["boundaries_lost"]:
            assert "to_split" in plan
            break
