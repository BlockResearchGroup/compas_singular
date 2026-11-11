"""The curvature branch: giving a moved coarse edge back its traced curve.

A coarse edge is a straight chord; curvature reaches the mesh through a separate
one-polyline-per-edge mapping. An edit breaks the exact match between an edge and
the separatrix it was traced from, and this is what puts it back.
"""

import pytest

from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.editing import CoarseEditor
from compas_singular.editing.curves import warp_chorded_edges
from compas_singular.editing.curves import warp_edge_curve


def _bump(a, b, height=0.5, n=9):
    """A curved polyline from ``a`` to ``b``, bulging by ``height``."""
    out = []
    for i in range(n):
        t = i / float(n - 1)
        x = a[0] + t * (b[0] - a[0])
        y = a[1] + t * (b[1] - a[1]) + height * (t * (1.0 - t)) * 4.0
        out.append([x, y, 0.0])
    return out


CURVE = _bump([0.0, 0.0, 0.0], [3.0, 0.0, 0.0])


# ==============================================================================
# warp_edge_curve
# ==============================================================================


def test_an_anchored_end_lets_the_other_move_freely():
    """One corner dragged: the still end identifies the curve, the moved one is
    trusted however far it went."""
    moved = [3.0, 2.0, 0.0]
    out = warp_edge_curve([CURVE], [0.0, 0.0, 0.0], moved, scale=1.0)

    assert out is not None
    assert out[0] == pytest.approx([0.0, 0.0, 0.0])
    assert out[-1] == pytest.approx(moved)
    # it is still curved, not a chord
    assert len(out) == len(CURVE)
    assert max(abs(p[1] - (p[0] / 3.0) * 2.0) for p in out) > 0.1


def test_a_curve_that_is_nowhere_near_is_not_used():
    out = warp_edge_curve([CURVE], [50.0, 50.0, 0.0], [53.0, 50.0, 0.0], scale=1.0)
    assert out is None


def test_a_wildly_disproportionate_candidate_is_refused():
    """The wrong curve warped is worse than no curve: it bulges through a
    neighbour rather than merely running straight."""
    # anchored at one end, but the new chord is a twentieth of the curve's length
    out = warp_edge_curve([CURVE], [0.0, 0.0, 0.0], [0.15, 0.0, 0.0], scale=1.0)
    assert out is None


def test_a_polyline_is_only_reused_when_nothing_unclaimed_fits():
    claimed = set()
    first = warp_edge_curve([CURVE], [0.0, 0.0, 0.0], [3.0, 0.5, 0.0], scale=1.0, claimed=claimed)
    assert first is not None and claimed == {0}

    # a second edge can still have it, but only because there is nothing else
    second = warp_edge_curve([CURVE], [0.0, 0.0, 0.0], [3.0, -0.5, 0.0], scale=1.0, claimed=claimed)
    assert second is not None


def test_the_curve_may_be_matched_either_way_round():
    out = warp_edge_curve([CURVE], [3.0, 0.0, 0.0], [0.0, 0.6, 0.0], scale=1.0)
    assert out is not None
    assert out[0] == pytest.approx([3.0, 0.0, 0.0])
    assert out[-1] == pytest.approx([0.0, 0.6, 0.0])


# ==============================================================================
# warp_chorded_edges
# ==============================================================================


def _two_quads():
    """Two quads side by side, so edge (0, 3) spans the curve above."""
    vertices = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.5, 2.0, 0.0], [3.0, 2.0, 0.0]]
    faces = [[0, 1, 4, 3], [1, 2, 5, 4]]
    return CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, faces)


def test_only_straight_chords_are_touched():
    mesh = _two_quads()
    mapping = {edge: [mesh.vertex_coordinates(edge[0]), mesh.vertex_coordinates(edge[1])] for edge in mesh.edges()}
    # one entry already has a shape and must be left exactly as it is
    shaped = list(mapping)[0]
    mapping[shaped] = [[0.0, 0.0, 0.0], [0.1, 0.1, 0.0], [0.2, 0.0, 0.0]]
    before = list(mapping[shaped])

    mapping, warped = warp_chorded_edges(mapping, mesh, [CURVE], scale=1.0)

    assert mapping[shaped] == before, "an edge that already had a shape was rewritten"
    assert warped >= 1
    assert all(len(c) >= 2 for c in mapping.values())


# ==============================================================================
# through the editor
# ==============================================================================

WALL = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 2.0, 0.0], [0.0, 2.0, 0.0]]


#: The INTERIOR separatrix of the two-quad layout: the edge (1, 4) up the middle,
#: as it would have been traced -- bowed rather than straight. A curve lying on a
#: WALL is no use here: those edges take the boundary-arc branch and never come
#: back as chords, so there would be nothing for the warp to fix.
SEPARATRIX = [[1.5 + 0.3 * (t * (1.0 - t)) * 4.0, 2.0 * t, 0.0] for t in [i / 8.0 for i in range(9)]]


def test_an_untouched_layout_is_not_warped():
    """Every edge still matches exactly, so the branch has nothing to do -- and
    running it could only risk a wrong match."""
    mesh = _two_quads()
    editor = CoarseEditor(mesh, loops=[WALL], polylines=[SEPARATRIX])
    assert editor.edited is False
    _mapping, tally = editor.edge_curves()
    assert tally["warped"] == 0
    assert tally["traced"] >= 1, "the separatrix should match its edge exactly"


def test_a_move_turns_the_branch_on_and_saves_chords():
    """Slide the bottom end of the middle edge: it stops matching, and the warp
    puts the traced shape back on the corners as they are now."""
    mesh = _two_quads()
    editor = CoarseEditor(mesh, loops=[WALL], polylines=[SEPARATRIX])
    _m, before = editor.edge_curves()

    # far enough to defeat ``coarse_edges_to_curves``'s own fuzzy fallback, which
    # tolerates about a quarter of a mean edge to absorb the single-precision
    # round trip through Rhino. Past that the edge really does come back a chord,
    # which is the case this branch exists for.
    editor.move_vertex(1, [2.6, 0.0, 0.0])

    assert editor.edited is True
    _m, without = editor.edge_curves(warp=False)
    _m, with_warp = editor.edge_curves(warp=True)

    assert without["warped"] == 0
    assert without["chord"] > before["chord"], "the move should have cost a curve"
    assert with_warp["warped"] >= 1
    assert with_warp["chord"] < without["chord"]


def test_the_warped_curve_ends_exactly_on_the_moved_corner():
    mesh = _two_quads()
    editor = CoarseEditor(mesh, loops=[WALL], polylines=[SEPARATRIX])
    editor.move_vertex(1, [2.6, 0.0, 0.0])
    mapping, _tally = editor.edge_curves()

    edge = (1, 4) if (1, 4) in mapping else (4, 1)
    curve = mapping[edge]
    assert len(curve) > 2, "the middle edge came back a chord"
    ends = {tuple(round(c, 6) for c in curve[0]), tuple(round(c, 6) for c in curve[-1])}
    for vkey in edge:
        assert tuple(round(c, 6) for c in editor.mesh.vertex_coordinates(vkey)) in ends


def test_reset_turns_the_branch_off_again():
    mesh = _two_quads()
    editor = CoarseEditor(mesh, loops=[WALL], polylines=[SEPARATRIX])
    editor.move_vertex(1, [2.6, 0.0, 0.0])
    assert editor.edited is True
    editor.reset()
    assert editor.edited is False
    _m, tally = editor.edge_curves()
    assert tally["warped"] == 0
