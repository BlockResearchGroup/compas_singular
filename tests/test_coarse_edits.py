"""The coarse layout's grammar operations: adding a strip, and dividing one.

Most of this file builds layouts by hand so the geometry is known exactly and the
opening rule can be measured rather than eyeballed; the last section runs the
same editor against real field decompositions.
"""

import subprocess
import sys
from math import cos
from math import pi
from math import sin

import pytest

from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.editing import CoarseEditor


SIZE = 3.0
WALL = [[0.0, 0.0, 0.0], [SIZE, 0.0, 0.0], [SIZE, SIZE, 0.0], [0.0, SIZE, 0.0]]


def _grid(n, size=SIZE):
    """``n`` x ``n`` coarse quads over a ``size`` square, plus ``{(i, j): key}``."""
    vertices, index = [], {}
    step = size / n
    for i in range(n + 1):
        for j in range(n + 1):
            index[i, j] = len(vertices)
            vertices.append([i * step, j * step, 0.0])
    faces = [[index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]] for i in range(n) for j in range(n)]
    return CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, faces), index


@pytest.fixture
def grid3():
    return _grid(3)


def _columns(mesh):
    return sorted(set(round(mesh.vertex_coordinates(v)[0], 6) for v in mesh.vertices()))


def _all_quad(mesh):
    return [f for f in mesh.faces() if len(mesh.face_vertices(f)) != 4]


# ==============================================================================
# add_strip -- thesis 5.3.1
# ==============================================================================


def test_add_strip_unzips_a_wall_to_wall_polyedge(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    polyedge = [index[1, 0], index[1, 1], index[1, 2], index[1, 3]]

    ok, notes = editor.add_strip(polyedge)

    assert ok, editor.last_reason
    assert notes["skey"] is not None
    assert notes["opened"] == 4
    assert editor.mesh.number_of_faces() == 12  # 9 + one quad per span
    assert editor.mesh.number_of_vertices() == 20  # 16 + one copy per corner
    assert not _all_quad(editor.mesh)


def test_the_new_strip_redivides_the_two_quads_in_three(grid3):
    """The opening rule, measured. Zero width would leave 3 columns, not 5."""
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    editor.add_strip([index[1, 0], index[1, 1], index[1, 2], index[1, 3]])

    # the span 0..2 was two quads of width 1; it now carries three edges
    assert _columns(editor.mesh) == pytest.approx([0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0, 3.0], abs=1e-6)


def test_add_strip_keeps_the_strip_data_valid(grid3):
    """Thesis Eq 5.6 as a post-condition, and 5.3.3 as the reason it holds."""
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    editor.add_strip([index[1, 0], index[1, 1], index[1, 2], index[1, 3]])

    ok, counted, expected = editor.check_strip_count()
    assert ok, "S_open=%d but E-2F=%d" % (counted, expected)


def test_a_commit_after_add_strip_keeps_the_densities(grid3):
    """The grammar preserves strip labels, so step 5's numbers survive."""
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    editor.mesh.collect_strips()
    editor.mesh.attributes["strips_density"] = {0: 7}

    editor.add_strip([index[1, 0], index[1, 1], index[1, 2], index[1, 3]])
    layout, notes = editor.commit()

    assert layout is mesh
    assert notes["path"] == "moves", "a grammar edit should not need the rebuild"
    assert mesh.attributes["strips_density"] == {0: 7}


# ------------------------------------------------------------------ refusals


def test_a_polyedge_with_an_interior_end_is_refused(grid3):
    """A strip runs the full width: wall to wall, or all the way round."""
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    ok, notes = editor.add_strip([index[1, 0], index[1, 1], index[1, 2]])
    assert not ok
    assert "full width" in notes["error"]
    assert notes["error"] == editor.last_reason


def test_a_list_of_picks_that_is_not_a_chain_is_refused(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    ok, notes = editor.add_strip([index[0, 0], index[2, 2], index[3, 3]])
    assert not ok
    assert "not joined by a coarse edge" in notes["error"]


def test_two_corners_are_not_a_polyedge(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    ok, notes = editor.add_strip([index[0, 0], index[1, 0]])
    assert not ok
    assert "more than two corners" in notes["error"]


def test_an_unknown_corner_is_refused(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    ok, notes = editor.add_strip([9999, index[1, 1], index[1, 2]])
    assert not ok
    assert "9999" in notes["error"]


def test_a_refusal_leaves_the_layout_untouched(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    faces = editor.mesh.number_of_faces()
    vertices = editor.mesh.number_of_vertices()
    editor.add_strip([index[1, 0], index[1, 1], index[1, 2]])
    assert editor.mesh.number_of_faces() == faces
    assert editor.mesh.number_of_vertices() == vertices


def test_a_polyedge_through_a_pole_is_refused_rather_than_guessed():
    """Thesis Fig 5.15 allows a pole END; crossing a pole would tear it in two."""
    editor = _two_triangle_pole()
    assert 2 in editor.mesh.poles(), "this fixture no longer has its pole at 2"
    ok, notes = editor.add_strip([1, 2, 3])
    assert not ok
    assert "through the pole" in notes["error"]


def _square_centre_pole():
    """Four pseudo-quads around a pole at the centre of a 10 x 10 square."""
    vertices = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0], [5.0, 5.0, 0.0]]
    faces = [[4, 0, 1], [4, 1, 2], [4, 2, 3], [4, 3, 0]]
    mesh = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(vertices, faces, [[5.0, 5.0, 0.0]])
    mesh.collect_strips()
    return CoarseEditor(mesh, loops=[[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]])


def _assert_pseudo_quad_layout(mesh):
    face_pole = mesh.attributes["face_pole"]
    for fkey in mesh.faces():
        vertices = mesh.face_vertices(fkey)
        assert len(vertices) == 4 or face_pole.get(fkey) in vertices, (fkey, vertices)
    assert all(mesh.has_face(fkey) for fkey in face_pole)


def test_dividing_a_strip_that_ends_at_a_pole_runs_the_line_into_the_pole():
    """The pole is not split: its fan gains one pseudo-quad."""
    editor = _square_centre_pole()
    radial = [skey for skey in editor.mesh.strips() if editor.mesh.has_strip_poles(skey)][0]
    ok, notes = editor.divide(skey=radial, t=0.5)
    assert ok, notes
    assert editor.mesh.poles() == [4]
    assert len(editor.mesh.vertex_pole_faces(4)) == 5
    _assert_pseudo_quad_layout(editor.mesh)

    layout, notes = editor.commit()
    assert layout is not None, notes
    layout.set_strips_density(3)
    layout.densification()
    assert len(layout.get_quad_mesh().poles()) == 1


def test_dividing_the_ring_around_a_pole_keeps_the_pole_fan():
    editor = _square_centre_pole()
    ring = [skey for skey in editor.mesh.strips() if editor.mesh.is_strip_closed(skey)][0]
    ok, notes = editor.divide(skey=ring, t=0.5)
    assert ok, notes
    assert len(editor.mesh.vertex_pole_faces(4)) == 4
    assert editor.mesh.number_of_faces() == 8
    _assert_pseudo_quad_layout(editor.mesh)


# ==============================================================================
# move_pole -- a relabel of the collapsed corner
# ==============================================================================


def _one_triangle_pole():
    """A quad and a pseudo-quad ``[1, 2, 4]`` collapsed at corner 2."""
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    faces = [[0, 1, 4, 3], [1, 2, 4]]
    mesh = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(vertices, faces, [[2.0, 0.0, 0.0]])
    return CoarseEditor(mesh, loops=[[[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0]]])


def _two_triangle_pole():
    """Two quads and two pseudo-quads collapsed at corner 2, sharing edge 2-5."""
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.5, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.5, 0.0], [0.0, 0.5, 0.0]]
    faces = [[0, 1, 5, 6], [6, 5, 3, 4], [1, 2, 5], [5, 2, 3]]
    mesh = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(vertices, faces, [[2.0, 0.5, 0.0]])
    return CoarseEditor(mesh, loops=[[[0, 0, 0], [1, 0, 0], [2, 0.5, 0], [1, 1, 0], [0, 1, 0]]])


def _pole_digest(mesh):
    return (sorted(mesh.attributes["face_pole"].items()), [mesh.face_vertices(f) for f in mesh.faces()], [mesh.vertex_coordinates(v) for v in mesh.vertices()])


def test_pole_targets_are_the_corners_every_pole_patch_shares():
    assert _one_triangle_pole().pole_targets(2) == [1, 4]
    assert _two_triangle_pole().pole_targets(2) == [5]
    assert _one_triangle_pole().pole_targets(0) == []


def test_move_pole_relabels_and_moves_nothing_else():
    editor = _two_triangle_pole()
    faces = [editor.mesh.face_vertices(f) for f in editor.mesh.faces()]
    xyz = [editor.mesh.vertex_coordinates(v) for v in editor.mesh.vertices()]
    ok, notes = editor.move_pole(2, 5)
    assert ok, notes
    assert notes["faces"] == 2
    assert editor.mesh.poles() == [5]
    assert [editor.mesh.face_vertices(f) for f in editor.mesh.faces()] == faces
    assert [editor.mesh.vertex_coordinates(v) for v in editor.mesh.vertices()] == xyz


@pytest.mark.parametrize(
    "pkey, vkey, word",
    [
        (2, 0, "not a corner of the pole patch"),
        (0, 1, "not a pole"),
        (2, 99, "not part of the layout"),
        (2, 2, "already"),
    ],
)
def test_an_illegal_pole_move_is_refused_and_changes_nothing(pkey, vkey, word):
    editor = _one_triangle_pole()
    before = _pole_digest(editor.mesh)
    ok, notes = editor.move_pole(pkey, vkey)
    assert not ok
    assert word in notes["error"]
    assert _pole_digest(editor.mesh) == before


def test_undo_puts_the_pole_back():
    editor = _one_triangle_pole()
    editor.push_undo()
    assert editor.move_pole(2, 4)[0]
    assert editor.mesh.poles() == [4]
    assert editor.undo()[0]
    assert editor.mesh.poles() == [2]
    assert not editor._strips_stale


def test_reset_puts_the_preferred_poles_back():
    fixture = _two_triangle_pole()
    editor = CoarseEditor(fixture.target, loops=fixture.loops, poles=[[2.0, 0.5, 0.0]])
    assert editor.move_pole(2, 5)[0]
    assert editor.poles == [[1.0, 0.5, 0.0]]
    editor.reset()
    assert editor.poles == [[2.0, 0.5, 0.0]]


def test_unify_counts_the_faces_it_flipped():
    from compas_singular.editing.repair import _unify

    vertices = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0]]
    faces = [[0, 1, 5, 4], [5, 6, 2, 1], [2, 3, 7, 6]]  # the middle one clockwise
    mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, faces)
    _out, flipped = _unify(mesh, CoarsePseudoQuadMesh)
    assert flipped == 1


def test_a_commit_after_move_pole_keeps_keys_and_drops_the_strip_data():
    # Not the one-triangle fixture: its corner 5 belongs to no patch, which
    # ``densifiable`` rightly calls non-manifold.
    editor = _two_triangle_pole()
    target = editor.target
    target.collect_strips()
    target.attributes["strips_density"] = {skey: 3 for skey in target.attributes["strips"]}
    editor = CoarseEditor(target, loops=editor.loops)
    assert editor.move_pole(2, 5)[0]
    layout, notes = editor.commit()
    assert layout is target, notes
    assert notes["path"] == "moves+pole"
    assert target.poles() == [5]
    assert not target.attributes.get("strips_density")


# ==============================================================================
# divide -- strip-traced
# ==============================================================================
#
# A cut must enter and leave every patch through OPPOSITE sides, and that IS the
# strip relation, so a legal cut crosses exactly one strip's patches. Measured on
# this 3x3 layout: of five drawn cuts, the four that are accepted match a strip
# exactly and the fifth -- a diagonal -- is refused.


def test_divide_by_strip_needs_no_drawing(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    skey = editor.strip_through((index[0, 1], index[1, 1]))

    ok, notes = editor.divide(skey=skey)

    assert ok, editor.last_reason
    assert notes["skey"] == skey
    assert editor.mesh.number_of_faces() == 12
    assert not _all_quad(editor.mesh)
    assert editor.check_strip_count()[0]


def test_divide_splits_every_rung_at_the_same_parameter(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    skey = editor.strip_through((index[0, 1], index[1, 1]))

    editor.divide(skey=skey, t=0.25)

    # the band spanning x 0..1, split a quarter of the way across
    assert _columns(editor.mesh) == pytest.approx([0.0, 0.25, 1.0, 2.0, 3.0], abs=1e-6)


def test_divide_defaults_to_the_middle(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    editor.divide(skey=editor.strip_through((index[0, 1], index[1, 1])))
    assert _columns(editor.mesh) == pytest.approx([0.0, 0.5, 1.0, 2.0, 3.0], abs=1e-6)


def test_a_drawn_cut_reports_the_strip_it_followed(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    ok, notes = editor.divide([[0.0, 1.5, 0.0], [SIZE, 1.5, 0.0]], extend=True)
    assert ok
    assert notes["skey"] is not None
    assert notes["faces"] == 3


def test_strip_of_cut_previews_the_band_before_the_curve_is_finished(grid3):
    """A front end can highlight what would be divided from the first point."""
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    drawn = editor.strip_of_cut([[0.0, 1.5, 0.0], [SIZE, 1.5, 0.0]])
    ok, notes = editor.divide([[0.0, 1.5, 0.0], [SIZE, 1.5, 0.0]], extend=True)
    assert drawn is not None and drawn == notes["skey"]


def test_a_diagonal_is_refused_structurally_not_by_a_side_count(grid3):
    """It used to come back as 'that patch has five sides', after mutating a copy."""
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    ok, notes = editor.divide([[0.0, 0.5, 0.0], [SIZE, 2.5, 0.0]], extend=True)
    assert not ok
    assert "does not follow a single line of patches" in notes["error"]
    assert editor.mesh.number_of_faces() == 9, "a refusal changed the layout"


def test_divide_needs_something_to_go_on(grid3):
    mesh, _ = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    ok, notes = editor.divide()
    assert not ok and "either a drawn curve or a strip" in notes["error"]


def test_an_unknown_strip_and_a_degenerate_parameter_are_refused(grid3):
    mesh, _ = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    assert not editor.divide(skey=9999)[0]
    assert "no strip" in editor.last_reason
    assert not editor.divide(skey=0, t=0.0)[0]
    assert "between 0 and 1" in editor.last_reason


def test_divide_takes_the_rebuild_path_at_commit(grid3):
    """Unlike add_strip: a divide inserts NEW corners, so the strip data goes."""
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    editor.divide(skey=editor.strip_through((index[0, 1], index[1, 1])))
    layout, notes = editor.commit()
    assert layout is mesh
    assert notes["path"] == "rebuild"


# ==============================================================================
# preserve_boundaries -- thesis 5.3.2 / Fig 5.18
# ==============================================================================
#
# A deletion that would leave a boundary with fewer than three edges is preceded
# by REFINING the strips that survive on it. The refinement is itself a strip
# addition, so it inherits the grammar's rule: topology only, and the new strip
# has ZERO WIDTH until the caller opens it. The base therefore leaves the split
# strips closed, and each editor opens them its own way -- exact thirds here,
# relaxation on a dense mesh.
#
# Until 2026-09-18 the base instead went through a second ``add_strip`` that
# ended in ``func_1``, a 20-iteration constrained smooth; on a COARSE layout that
# slid corners along the chorded boundary, which is what the override below was
# originally written to avoid. The override survives that implementation for the
# opposite reason: the base no longer moves anything, and now something has to.

from compas_singular.editing.editor import MeshEditor  # noqa: E402


class _Unopened(CoarseEditor):
    """The base behaviour, for comparison: topology only, nothing opened."""

    def _split_strips(self, work, to_split):
        return MeshEditor._split_strips(self, work, to_split)


def _coincident_corner_pairs(mesh):
    """Pairs of corners of one face sitting on top of each other."""
    n = 0
    for fkey in mesh.faces():
        points = [mesh.vertex_coordinates(v) for v in mesh.face_vertices(fkey)]
        for a in range(len(points)):
            for b in range(a + 1, len(points)):
                if sum((x - y) ** 2 for x, y in zip(points[a], points[b])) ** 0.5 < 1e-9:
                    n += 1
    return n


def _split_and_measure(cls, mesh):
    editor = cls(mesh, loops=[WALL])
    work = editor.mesh.copy()
    work.collect_strips()
    before = {v: work.vertex_coordinates(v) for v in work.vertices()}
    skey = sorted(work.attributes["strips"])[0]

    editor._split_strips(work, {skey: 2})

    drift = 0.0
    moved = 0
    for vkey, was in before.items():
        if vkey not in work.vertex:
            continue
        now = work.vertex_coordinates(vkey)
        d = sum((a - b) ** 2 for a, b in zip(was, now)) ** 0.5
        if d > 1e-9:
            moved += 1
            drift = max(drift, d)
    return work, moved, drift


def test_the_coarse_pre_split_moves_no_existing_corner(grid3):
    """Refining to save a boundary must not disturb the layout it is saving."""
    mesh, _ = grid3
    work, moved, drift = _split_and_measure(CoarseEditor, mesh)

    assert moved == 0, "%d corner(s) moved, worst %.4f" % (moved, drift)
    assert work.number_of_faces() == 12, "the strip was not actually split"


def test_the_base_pre_split_leaves_the_new_strip_closed(grid3):
    """Pins WHY the override exists. If this ever stops being true the override
    is dead weight and should go.

    The base is topology only, so every rung of the refined strip comes out with
    its two sides on top of each other. The override opens them, and the measure
    of that is strictly fewer coincident corners -- not zero, because
    ``_open_strip`` still leaves the rungs at the two ends closed on this layout.
    """
    mesh, _ = grid3
    base, _moved, _drift = _split_and_measure(_Unopened, mesh)
    opened, _moved, _drift = _split_and_measure(CoarseEditor, mesh)

    base_pairs = _coincident_corner_pairs(base)
    opened_pairs = _coincident_corner_pairs(opened)

    assert base_pairs > 0, "the base pre-split no longer leaves the strip closed -- re-check whether the coarse override is still needed"
    assert opened_pairs < base_pairs, "the override opened nothing (base %d, override %d)" % (base_pairs, opened_pairs)


def test_the_plan_reports_whether_a_boundary_can_be_saved(grid3):
    """Tri-state: ``None`` from the helper means it cannot be prevented, ``{}``
    means nothing is at risk. They must not be flattened together."""
    mesh, _ = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    plan = editor.plan_strip_deletion(edge=list(editor.mesh.edges())[0])

    assert plan["can_preserve"] is True
    assert plan["to_split"] == {}, "nothing is at risk on a plain grid"


def test_preserve_boundaries_is_a_no_op_when_nothing_is_at_risk(grid3):
    mesh, _ = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    edge = list(editor.mesh.edges())[0]
    faces = editor.mesh.number_of_faces()

    ok, notes = editor.remove_strip(edge=edge, preserve_boundaries=True)

    assert ok, editor.last_reason
    assert notes["to_split"] == {}
    assert editor.mesh.number_of_faces() < faces


# ==============================================================================
# drawn cuts through pseudo-quads: along the pole side, or into the pole
# ==============================================================================


def _ringed_pole():
    """A pole at (5, 5) in four pseudo-quads, inside a ring of four quads."""
    vertices = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0], [3.0, 3.0, 0.0], [7.0, 3.0, 0.0], [7.0, 7.0, 0.0], [3.0, 7.0, 0.0], [5.0, 5.0, 0.0]]
    faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7], [8, 4, 5], [8, 5, 6], [8, 6, 7], [8, 7, 4]]
    mesh = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(vertices, faces, [[5.0, 5.0, 0.0]])
    return CoarseEditor(mesh, loops=[[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]])


def _corner_pole():
    """Four pseudo-quads fanned from a pole at corner (0, 0) of a 10 x 10 square."""
    vertices = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 5.0, 0.0], [10.0, 10.0, 0.0], [5.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]
    mesh = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(vertices, faces, [[0.0, 0.0, 0.0]])
    return CoarseEditor(mesh, loops=[[[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]])


def _commits_and_densifies(editor):
    layout, notes = editor.commit()
    assert layout is not None, notes
    layout.set_strips_density(2)
    layout.densification()
    assert len(layout.get_quad_mesh().poles()) == 1


@pytest.mark.parametrize(
    "points",
    [
        [[5.0, 0.0, 0.0], [5.0, 5.0, 0.0]],  # drawn to the pole
        [[5.0, 0.0, 0.0], [5.0, 2.0, 0.0], [5.3, 4.0, 0.0]],  # stops loosely inside the pseudo-quad
    ],
)
def test_a_cut_across_the_pole_side_ends_at_the_pole(points):
    editor = _ringed_pole()
    ok, notes = editor.divide(points=points)
    assert ok, notes
    assert editor.mesh.poles() == [8]
    assert len(editor.mesh.vertex_pole_faces(8)) == 5
    _assert_pseudo_quad_layout(editor.mesh)
    _commits_and_densifies(editor)


def test_a_cut_that_passes_the_pole_divides_the_strips_on_both_sides():
    editor = _ringed_pole()
    ok, notes = editor.divide(points=[[4.0, 0.0, 0.0], [4.4, 10.0, 0.0]])
    assert ok, notes
    assert len(notes["skey"]) == 2
    assert len(editor.mesh.vertex_pole_faces(8)) == 6
    _assert_pseudo_quad_layout(editor.mesh)
    _commits_and_densifies(editor)


def test_an_extended_cut_stops_at_the_pole():
    editor = _ringed_pole()
    ok, notes = editor.divide(points=[[5.0, 0.0, 0.0], [5.0, 3.0, 0.0]], extend=True)
    assert ok, notes
    assert len(editor.mesh.vertex_pole_faces(8)) == 5


def test_a_cut_along_the_pole_side_splits_each_pseudo_quad_in_a_quad_and_a_pseudo_quad():
    import math

    arc = [[5.0 * math.cos(t), 5.0 * math.sin(t), 0.0] for t in [i * math.pi / 24 for i in range(13)]]
    editor = _corner_pole()
    ok, notes = editor.divide(points=arc)
    assert ok, notes
    mesh = editor.mesh
    assert len(mesh.vertex_pole_faces(0)) == 4
    assert sum(1 for f in mesh.faces() if len(mesh.face_vertices(f)) == 4) == 4
    _assert_pseudo_quad_layout(mesh)
    _commits_and_densifies(editor)


def test_a_cut_through_two_sides_of_a_pseudo_quad_that_meet_is_refused():
    editor = _corner_pole()
    ok, notes = editor.divide(points=[[5.0, 0.0, 0.0], [10.0, 2.5, 0.0]])
    assert not ok
    assert editor.mesh.number_of_faces() == 4


# ==============================================================================
# add_strip along a wall, and through itself -- the thesis's strip-addition figures
# ==============================================================================


def _area(mesh, fkey):
    points = [mesh.vertex_coordinates(vkey) for vkey in mesh.face_vertices(fkey)]
    return 0.5 * sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(points, points[1:] + points[:1]))


def _assert_flat_and_consistent(mesh):
    """Every face turns the same way with area, all quads, and the strip table equals a fresh collection."""
    assert min(_area(mesh, f) for f in mesh.faces()) > 1e-9
    assert not _all_quad(mesh)
    assert mesh.is_manifold()
    fresh = mesh.copy()
    fresh.collect_strips()

    def table(m):
        return sorted(sorted(tuple(sorted(e)) for e in edges) for edges in m.attributes["strips"].values())

    assert table(mesh) == table(fresh)


def _points(mesh):
    return set((round(x, 9), round(y, 9)) for x, y, _ in (mesh.vertex_coordinates(v) for v in mesh.vertices()))


def test_a_ring_along_the_whole_boundary_of_one_quad():
    """Thesis A-B-C-D-A: the outer copies stay on the corners, the inner ones go a third inwards."""
    mesh, index = _grid(1)
    editor = CoarseEditor(mesh, loops=[WALL])
    ok, notes = editor.add_strip([index[0, 1], index[1, 1], index[1, 0], index[0, 0], index[0, 1]])
    assert ok, editor.last_reason
    assert notes["opened"] == 4
    assert editor.mesh.number_of_faces() == 5
    assert _points(editor.mesh) == {(0, 0), (3, 0), (3, 3), (0, 3), (1, 1), (2, 1), (2, 2), (1, 2)}
    _assert_flat_and_consistent(editor.mesh)


def test_a_ring_along_the_outer_wall_of_a_grid(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    ring = [index[0, j] for j in (3, 2, 1, 0)] + [index[i, 0] for i in (1, 2, 3)] + [index[3, j] for j in (1, 2, 3)] + [index[i, 3] for i in (2, 1, 0)]
    ok, notes = editor.add_strip(ring)
    assert ok, editor.last_reason
    assert notes["opened"] == 12
    _assert_flat_and_consistent(editor.mesh)


def test_a_strip_that_turns_onto_the_wall(grid3):
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    ok, _ = editor.add_strip([index[1, 3], index[1, 2], index[1, 1], index[1, 0], index[2, 0], index[3, 0]])
    assert ok, editor.last_reason
    _assert_flat_and_consistent(editor.mesh)


def test_a_self_crossing_polyedge_adds_one_strip_through_itself():
    """Thesis A-B-C-D-E-B-F: B is split twice, into four copies around the crossing."""
    mesh, index = _grid(2)
    editor = CoarseEditor(mesh, loops=[WALL])
    polyedge = [index[0, 1], index[1, 1], index[2, 1], index[2, 2], index[1, 2], index[1, 1], index[1, 0]]
    ok, notes = editor.add_strip(polyedge)
    assert ok, editor.last_reason
    assert editor.mesh.number_of_vertices() == 17
    assert editor.mesh.number_of_faces() == 11
    assert len(editor.mesh.attributes["strips"]) == 5
    # one strip of seven faces, walked through its middle face twice: nine rungs
    assert len(editor.mesh.strip_edges(notes["skey"])) == 9
    assert {(1, 1), (2, 1), (2, 2), (1, 2)} <= _points(editor.mesh)
    _assert_flat_and_consistent(editor.mesh)


def test_a_loop_that_closes_on_a_vertex_it_passed(grid3):
    """A closed polyedge whose last vertex repeats is re-started where it passes once."""
    mesh, index = _grid(4, size=4.0)
    editor = CoarseEditor(mesh, loops=[[[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]])
    loop = [index[2, 0], index[1, 0], index[1, 1], index[2, 1], index[3, 1], index[3, 2], index[2, 2], index[2, 1], index[2, 0]]
    ok, _ = editor.add_strip(loop)
    assert ok, editor.last_reason
    _assert_flat_and_consistent(editor.mesh)


def test_a_polyedge_that_ends_where_it_passed():
    mesh, index = _grid(4, size=4.0)
    editor = CoarseEditor(mesh, loops=[[[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]])
    lasso = [index[4, 2], index[3, 2], index[2, 2], index[2, 3], index[3, 3], index[3, 2], index[3, 1], index[3, 0]]
    ok, _ = editor.add_strip(lasso)
    assert ok, editor.last_reason
    _assert_flat_and_consistent(editor.mesh)


def test_a_polyedge_that_runs_next_to_itself():
    """A U round one face: each end's side neighbour is the other end, split by the same walk."""
    mesh, index = _grid(4, size=4.0)
    editor = CoarseEditor(mesh, loops=[[[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]]])
    ok, _ = editor.add_strip([index[2, 0], index[2, 1], index[1, 1], index[1, 0]])
    assert ok, editor.last_reason
    _assert_flat_and_consistent(editor.mesh)


def test_a_strip_starting_at_a_hole_corner():
    mesh, index = _grid(4, size=4.0)
    mesh.delete_face(next(f for f in mesh.faces() if set(mesh.face_vertices(f)) == {index[2, 2], index[3, 2], index[3, 3], index[2, 3]}))
    hole = [[2, 2, 0], [3, 2, 0], [3, 3, 0], [2, 3, 0]]
    editor = CoarseEditor(mesh, loops=[[[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0]], hole])
    ok, _ = editor.add_strip([index[2, 2], index[1, 2], index[1, 3], index[1, 4]])
    assert ok, editor.last_reason
    _assert_flat_and_consistent(editor.mesh)


def test_a_strip_that_would_fold_a_face_is_refused(grid3):
    """The grammar raises rather than hand back a folded face; the editor turns that into a refusal."""
    mesh, index = grid3
    editor = CoarseEditor(mesh, loops=[WALL])
    faces, vertices = editor.mesh.number_of_faces(), editor.mesh.number_of_vertices()
    ok, notes = editor.add_strip([9, 10, 6, 7, 3, 2, 6, 5, 9])
    assert not ok
    assert "opened flat" in notes["error"]
    assert editor.mesh.number_of_faces() == faces
    assert editor.mesh.number_of_vertices() == vertices


# ==============================================================================
# the editor on real field decompositions -- every operation the command offers
# ==============================================================================

from compas_singular.framefield.field_decomposition import FieldDecomposition  # noqa: E402


def _square_wall(size=10.0):
    return [[0.0, 0.0, 0.0], [size, 0.0, 0.0], [size, size, 0.0], [0.0, size, 0.0], [0.0, 0.0, 0.0]]


def _circle(cx, cy, r, n=24):
    pts = [[cx + r * cos(2 * pi * i / n), cy + r * sin(2 * pi * i / n), 0.0] for i in range(n)]
    return pts + pts[:1]


def _field_editor(inners=None):
    """A CoarseEditor anchored on a field layout: walls, separatrices and field handed over."""
    d = FieldDecomposition.from_boundary(_square_wall(), inners, target_length=1.0)
    d.decomposition_mesh()
    return CoarseEditor(d.mesh, field=d.get_field(), loops=d._loops(), polylines=d.polylines or ())


def _bad_patches(mesh):
    """Faces that are neither a quad nor a 3-sided pseudo-quad with a pole."""
    face_pole = mesh.attributes.get("face_pole") or {}
    return [f for f in mesh.faces() if not ((len(mesh.face_vertices(f)) == 4 and f not in face_pole) or (len(mesh.face_vertices(f)) == 3 and f in face_pole))]


def _half_square():
    """A square cut in half -- the smallest field layout with an INTERIOR edge."""
    editor = _field_editor()
    assert editor.divide([[0.0, 5.0, 0.0], [10.0, 5.0, 0.0]], extend=True)[0]
    return editor


def _mesh_digest(mesh):
    """Enough of a mesh to prove a refusal changed nothing."""
    return (mesh.number_of_vertices(), mesh.number_of_faces(), sorted(tuple(round(c, 9) for c in mesh.vertex_coordinates(v)) for v in mesh.vertices()))


def _stop_on_interior_edge(editor):
    u, v = next((u, v) for u, v in editor.mesh.edges() if not editor.mesh.is_edge_on_boundary((u, v)))
    pu, pv = editor.mesh.vertex_coordinates(u), editor.mesh.vertex_coordinates(v)
    mid = [(pu[0] + pv[0]) / 2.0, (pu[1] + pv[1]) / 2.0, 0.0]
    return [[mid[0], 0.0, 0.0], mid]  # from the bottom wall straight up to that edge


def test_a_boundary_corner_moved_off_its_wall_lands_back_on_it():
    editor = _field_editor()
    vkey = next(v for v in editor.mesh.vertices() if editor.is_vertex_on_boundary(v))
    before = editor.mesh.vertex_coordinates(vkey)
    editor.move_vertex(vkey, [before[0] + 2.0, before[1] + 2.0, 0.0])
    after = editor.mesh.vertex_coordinates(vkey)
    assert min(w.project(after)[0] for w in editor.walls) < 1e-9
    assert sum((a - b) ** 2 for a, b in zip(after, before)) > 1e-9
    editor.reset()
    assert editor.mesh.vertex_coordinates(vkey) == before


def test_a_wall_to_wall_cut_on_a_field_layout():
    editor = _field_editor()
    n_in = editor.mesh.number_of_faces()
    ok, _ = editor.divide([[0.0, 5.0, 0.0], [10.0, 5.0, 0.0]], extend=True)
    assert ok, editor.last_reason
    assert editor.mesh.number_of_faces() > n_in
    assert not _bad_patches(editor.mesh)


def test_a_cut_stopping_on_an_interior_edge_is_refused_without_extend():
    # Aimed at a REAL interior edge: a one-patch square has none, and a test using it
    # measures the wrong refusal -- not on the layout, rather than stops short.
    editor = _half_square()
    digest = _mesh_digest(editor.mesh)
    ok, _ = editor.divide(_stop_on_interior_edge(editor), extend=None)
    assert not ok
    assert "five sides" in editor.last_reason
    assert _mesh_digest(editor.mesh) == digest


def test_a_cut_stopping_on_an_interior_edge_asks_and_extends():
    editor = _half_square()
    points = _stop_on_interior_edge(editor)
    n_in = editor.mesh.number_of_faces()
    asked = []
    ok, _ = editor.divide(points, extend=lambda n: asked.append(n) or True)
    assert ok, editor.last_reason
    assert asked == [1]
    assert editor.mesh.number_of_faces() > n_in
    assert not _bad_patches(editor.mesh)


def test_a_cut_that_starts_nowhere_is_refused():
    editor = _field_editor()
    digest = _mesh_digest(editor.mesh)
    ok, _ = editor.divide([[-5.0, 5.0, 0.0], [-5.0, 6.0, 0.0]], extend=True)
    assert not ok
    assert _mesh_digest(editor.mesh) == digest


def test_a_committed_cut_densifies():
    editor = _field_editor()
    editor.divide([[0.0, 5.0, 0.0], [10.0, 5.0, 0.0]], extend=True)
    faces_before = editor.mesh.number_of_faces()
    layout, notes = editor.commit()
    assert layout is not None, notes.get("error")
    assert notes.get("faces_in") == faces_before
    assert notes.get("faces_out") is not None
    assert "lost_curves" in notes
    editor.mesh.collect_strips()
    edges, _ = editor.edge_curves()
    editor.mesh.collect_strips()
    editor.mesh.set_strips_density_target(1.0)
    dense = editor.mesh.densification(overwrite_edges_to_curves=edges)
    assert dense is not None and dense.number_of_faces() > 0


def test_strip_deletion_on_a_holed_field_layout_is_planned_then_performed():
    editor = _field_editor([_circle(5.0, 5.0, 1.5)])
    editor.mesh.collect_strips()
    plans = [(edge, editor.plan_strip_deletion(edge)) for edge in editor.mesh.edges()]
    assert all(plan["skey"] is not None for _, plan in plans)  # every edge resolves to a strip
    refused = [p for p in plans if not p[1]["ok"]]
    allowed = [p for p in plans if p[1]["ok"]]
    assert allowed and refused
    assert all(plan["reason"] for _, plan in refused)

    edge, plan = refused[0]
    digest = _mesh_digest(editor.mesh)
    ok, _ = editor.remove_strip(edge)
    assert not ok
    assert _mesh_digest(editor.mesh) == digest
    assert editor.last_reason == plan["reason"]  # plan and delete agree

    edge, _ = allowed[0]
    n_in = editor.mesh.number_of_faces()
    ok, _ = editor.remove_strip(edge)
    assert ok, editor.last_reason
    assert editor.mesh.number_of_faces() < n_in
    assert not _bad_patches(editor.mesh)
    assert editor.mesh.is_manifold()
    assert editor.last_deletion.get("skey") is not None
    layout, notes = editor.commit()
    assert layout is not None, notes.get("error")


def test_reset_undoes_a_strip_deletion_topology_and_all():
    editor = _field_editor([_circle(5.0, 5.0, 1.5)])
    editor.mesh.collect_strips()
    n_in = editor.mesh.number_of_faces()
    for edge in editor.mesh.edges():
        if editor.plan_strip_deletion(edge)["ok"] and editor.remove_strip(edge)[0]:
            break
    assert editor.mesh.number_of_faces() != n_in
    editor.reset()
    assert editor.mesh.number_of_faces() == n_in


def test_the_editor_imports_no_rhino():
    # in a fresh interpreter: another test in this session may have stubbed Rhino
    code = "import sys; import compas_singular.editing; print([m for m in sys.modules if m.split('.')[0] in ('rhinoscriptsyntax', 'Rhino', 'scriptcontext', 'System')])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
