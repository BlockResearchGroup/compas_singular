"""Tests for the QuadMesh polyedge/strip machinery.

Covers the batch 1-2 port of compas_quad: the null guards, the accessors, and the
opt-in flags (``strict``, ``oriented``, ``both_sides``, ``legacy``) that carry the
new behaviour without moving any default.
"""

import pytest

from compas_singular.datastructures import QuadMesh


GRID_VERTICES = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]]
GRID_FACES = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]


@pytest.fixture
def grid():
    """Standard 2 x 2 quad grid."""
    return QuadMesh.from_vertices_and_faces(GRID_VERTICES, GRID_FACES)


@pytest.fixture
def annulus():
    """Four quads around a square hole -- two nested boundaries."""
    vertices = [[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0], [1, 1, 0], [2, 1, 0], [2, 2, 0], [1, 2, 0]]
    faces = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return QuadMesh.from_vertices_and_faces(vertices, faces)


@pytest.fixture
def kinked_grid():
    """2 x 2 grid with an extra quad hanging off vertex 1.

    Vertex 1 is then a degree-4 vertex *on the boundary*, i.e. singular -- the
    case a naive ``len(nbrs) == 4`` rule walks straight through.
    """
    vertices = GRID_VERTICES + [[1.0, -1.0, 0.0], [2.0, -1.0, 0.0]]
    faces = GRID_FACES + [[1, 9, 10, 2]]
    return QuadMesh.from_vertices_and_faces(vertices, faces)


# ----------------------------------------------------------------------------
# null guards
# ----------------------------------------------------------------------------


def test_face_opposite_edge_returns_none_on_a_boundary_halfedge(grid):
    u, v = 1, 0  # halfedge on the outside of the boundary edge (0, 1)
    assert grid.halfedge[u][v] is None
    assert grid.face_opposite_edge(u, v) is None


def test_face_opposite_edge_still_works_inside_a_face(grid):
    assert grid.face_opposite_edge(0, 1) == (4, 3)


def test_collect_strip_survives_a_boundary_halfedge_seed(grid):
    # would previously raise while unpacking the None
    assert grid.collect_strip(1, 0)


# ----------------------------------------------------------------------------
# accessors
# ----------------------------------------------------------------------------


def test_polyedge_accessors(grid):
    grid.collect_polyedges()
    pkey = next(iter(grid.polyedges()))

    assert grid.number_of_polyedges() == len(list(grid.polyedges()))
    assert grid.polyedge_vertices(pkey) == grid.attributes["polyedges"][pkey]
    assert grid.polyline(pkey) == [grid.vertex_coordinates(v) for v in grid.polyedge_vertices(pkey)]

    edges = grid.polyedge_edges(pkey)
    assert isinstance(edges, list)  # not a one-shot iterator
    assert len(edges) == len(grid.polyedge_vertices(pkey)) - 1
    assert list(edges) == list(edges)  # re-iterable

    assert grid.polyedge_length(pkey) == pytest.approx(2.0)
    assert len(grid.polyedge_midpoint(pkey)) == 3


# ----------------------------------------------------------------------------
# collect_polyedge: both_sides / oriented
# ----------------------------------------------------------------------------


def test_collect_polyedge_default_loses_the_seed_direction(grid):
    # the walk hits an extremity on the first side and returns reversed
    assert grid.collect_polyedge(2, 1) == [0, 1, 2]


def test_collect_polyedge_oriented_keeps_the_seed_direction(grid):
    assert grid.collect_polyedge(2, 1, oriented=True) == [2, 1, 0]


def test_collect_polyedge_one_sided_stops_at_the_first_extremity(grid):
    assert grid.collect_polyedge(2, 1, both_sides=False) == [2, 1, 0]


def test_collect_strip_one_sided_returns_a_list_of_edges(grid):
    strip = grid.collect_strip(1, 0, both_sides=False)
    assert strip == [(1, 0)]


# ----------------------------------------------------------------------------
# vertex_opposite_vertex: strict
# ----------------------------------------------------------------------------


def test_strict_does_not_turn_an_interior_polyedge_onto_a_boundary(annulus):
    # vertex 0 is on the outer boundary, 4 on the inner one, and (0, 4) is an
    # interior edge; the default rule only tests the vertices and turns onto the
    # hole boundary
    assert annulus.vertex_opposite_vertex(0, 4) == 5
    assert annulus.vertex_opposite_vertex(0, 4, strict=True) is None


def test_strict_separates_the_two_boundary_loops(annulus):
    annulus.collect_polyedges(strict=True)
    # each ring closes on itself, joined by four clean radial polyedges
    assert sorted(annulus.attributes["polyedges"].values()) == [
        [2, 3, 0, 1, 2],  # outer ring
        [4, 0],
        [5, 1],
        [6, 2],  # radials
        [6, 7, 4, 5, 6],  # inner ring
        [7, 3],  # radial
    ]


def test_default_walk_mixes_the_two_boundary_loops(annulus):
    """What strict=True fixes -- kept as the record of the default behaviour."""
    annulus.collect_polyedges()
    polyedges = list(annulus.attributes["polyedges"].values())
    # every polyedge drags vertices of both rings along
    inner, outer = {4, 5, 6, 7}, {0, 1, 2, 3}
    assert any(set(p) & inner and set(p) & outer for p in polyedges)


def test_strict_does_not_walk_through_a_four_valent_boundary_vertex(kinked_grid):
    # vertex 1 has four neighbours but sits on the boundary, so it is singular;
    # an unguarded `len(nbrs) == 4` rule would return 2 here, and 9 from vertex 4
    assert kinked_grid.vertex_neighbors(1, ordered=True) == [0, 4, 2, 9]
    assert kinked_grid.vertex_opposite_vertex(0, 1, strict=True) is None
    assert kinked_grid.vertex_opposite_vertex(4, 1, strict=True) is None


def test_strict_defaults_off(annulus):
    """The default walk must keep its historical (wrong) answer."""
    assert annulus.collect_polyedge(0, 4) == [0, 4, 5, 1, 0]


# ----------------------------------------------------------------------------
# polyedge_graph: legacy
# ----------------------------------------------------------------------------


def test_legacy_graph_is_the_default_and_emits_self_loops(grid):
    grid.collect_polyedges()
    _, edges = grid.polyedge_graph()
    assert [e for e in edges if e[0] == e[1]]


def test_new_graph_has_no_self_loops_and_fewer_edges(grid):
    grid.collect_polyedges()
    _, legacy = grid.polyedge_graph(legacy=True)
    _, new = grid.polyedge_graph(legacy=False)
    assert [e for e in new if e[0] == e[1]] == []
    assert len(new) < len(legacy)


def test_new_graph_pairs_the_two_polyedges_crossing_at_a_vertex(grid):
    grid.collect_polyedges()
    nodes, edges = grid.polyedge_graph(legacy=False)
    vertex_to_polyedges = {}
    for pkey, polyedge in grid.polyedges(data=True):
        for vkey in polyedge:
            vertex_to_polyedges.setdefault(vkey, set()).add(pkey)
    expected = sum(1 for vkey, pkeys in vertex_to_polyedges.items() if len(pkeys) == 2 and not grid.is_vertex_singular(vkey))
    assert len(edges) == expected
    assert set(nodes) == set(grid.polyedges())


# ==============================================================================
# Strip data hygiene -- collect_strips and the grammar's new-key choice
# ==============================================================================
#
# Two defects that compounded into a third. ``collect_strips`` used ``update``
# without clearing, so a strip key that had been DELETED and was then re-assigned
# landed at the END of the dict's insertion order. That alone leaves stale entries
# (read as real by ``is_strip_closed``, which only looks at ``strips[skey][0]``),
# and it also breaks an ``add_strip`` that names the new strip ``list(mesh.strips())[-1] + 1``,
# which then names a strip that already exists and silently overwrites its edges.
#
# Measured on the 4 x 4 grid below, before the fix:
#     after delete_strip(3) + collect_strips() -> [0, 1, 2, 4, 5, 6, 7, 3]
#     last = 3, max = 7, and ``last + 1`` = 4, which was a live strip.

from compas_singular.datastructures.mesh_quad.grammar.add_strip import (  # noqa: E402
    add_strip as pattern_add_strip,
)
from compas_singular.datastructures.mesh_quad.grammar.delete_strip import (  # noqa: E402
    delete_strip as pattern_delete_strip,
)


def _square_grid(n):
    """An ``n`` x ``n`` quad grid, plus ``{(i, j): vertex key}``."""
    vertices, index = [], {}
    for i in range(n + 1):
        for j in range(n + 1):
            index[i, j] = len(vertices)
            vertices.append([float(i), float(j), 0.0])
    faces = [[index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1]] for i in range(n) for j in range(n)]
    return QuadMesh.from_vertices_and_faces(vertices, faces), index


@pytest.fixture
def grid4():
    """4 x 4 grid: 8 strips, enough to delete one from the middle."""
    return _square_grid(4)


def test_collect_strips_drops_keys_that_no_longer_exist(grid4):
    """Re-collecting must not leave the old strip count behind."""
    mesh, _ = grid4
    mesh.collect_strips()
    assert len(mesh.attributes["strips"]) == 8

    pattern_delete_strip(mesh, 3)
    mesh.collect_strips()

    # 4 x 4 minus one strip is a 4 x 3 grid: 4 + 3 = 7 strips, not 8.
    assert len(mesh.attributes["strips"]) == 7
    assert sorted(mesh.attributes["strips"]) == list(range(7))


def test_collect_strips_leaves_insertion_order_tracking_the_maximum(grid4):
    """``list(strips())[-1]`` must still be the largest key after a deletion."""
    mesh, _ = grid4
    mesh.collect_strips()
    pattern_delete_strip(mesh, 3)
    mesh.collect_strips()

    keys = list(mesh.attributes["strips"])
    assert keys[-1] == max(keys), "insertion order stopped tracking the maximum: %s" % keys


def test_a_new_strip_key_does_not_name_a_live_strip(grid4):
    """The compound failure: the key the grammar would pick was already taken."""
    mesh, _ = grid4
    mesh.collect_strips()
    pattern_delete_strip(mesh, 3)
    mesh.collect_strips()

    for candidate in (list(mesh.strips())[-1] + 1, max(mesh.strips()) + 1):
        assert candidate not in mesh.attributes["strips"], "new strip key %r already exists -- adding a strip would overwrite it" % candidate


def test_add_strip_along_a_closed_polyedge_leaves_the_strip_closed(grid4):
    """Thesis 5.3.1: a closed polyedge unzips into a CLOSED strip."""
    mesh, index = grid4
    mesh.collect_strips()
    before = dict(mesh.attributes["strips"])
    faces_before = mesh.number_of_faces()

    ring = [index[1, 1], index[2, 1], index[3, 1], index[3, 2], index[3, 3], index[2, 3], index[1, 3], index[1, 2]]
    skey = pattern_add_strip(mesh, ring + [ring[0]])[0]

    assert skey not in before, "the new strip overwrote an existing one"
    assert mesh.is_strip_closed(skey)
    # eight rungs, so eight new quads
    assert len(mesh.attributes["strips"][skey]) == 8
    assert mesh.number_of_faces() == faces_before + 8
    assert all(len(mesh.face_vertices(f)) == 4 for f in mesh.faces())


def test_open_strip_count_matches_the_euler_style_relation(grid4):
    """Thesis Eq 5.6: S_open == E - 2F. A free check that the strip data is sane."""
    mesh, _ = grid4
    mesh.collect_strips()
    open_strips = [s for s in mesh.attributes["strips"] if not mesh.is_strip_closed(s)]
    assert len(open_strips) == mesh.number_of_edges() - 2 * mesh.number_of_faces()


# ==============================================================================
# strips with a pole end (thesis Fig 5.15): the pole is never copied
# ==============================================================================

from compas_singular.datastructures import CoarsePseudoQuadMesh  # noqa: E402
from compas_singular.datastructures.mesh_quad.grammar.add_strip import (  # noqa: E402
    pole_ends,
    split_strip,
)


def _centre_pole(density=None):
    """Four pseudo-quads around a pole at (5, 5); densified if ``density`` is given."""
    vertices = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0], [5.0, 5.0, 0.0]]
    faces = [[4, 0, 1], [4, 1, 2], [4, 2, 3], [4, 3, 0]]
    mesh = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(vertices, faces, poles=[[5.0, 5.0, 0.0]])
    mesh.collect_strips()
    if density is None:
        return mesh
    mesh.set_strips_density(density)
    mesh.densification()
    dense = mesh.get_quad_mesh()
    dense.collect_strips()
    return dense


def _strip_face_sets(mesh):
    return sorted(tuple(sorted(mesh.strip_faces(skey))) for skey in mesh.strips())


def _assert_consistent(mesh):
    face_pole = mesh.attributes["face_pole"]
    for fkey in mesh.faces():
        vertices = mesh.face_vertices(fkey)
        assert len(vertices) == 4 or face_pole.get(fkey) in vertices, (fkey, vertices)
    assert mesh.is_manifold()
    fresh = mesh.copy()
    fresh.collect_strips()
    assert _strip_face_sets(mesh) == _strip_face_sets(fresh), "stored strip data went stale"


def test_splitting_a_pole_strip_leaves_the_pole_whole():
    mesh = _centre_pole(density=4)
    pole = mesh.poles()[0]
    fan = len(mesh.vertex_pole_faces(pole))
    skey = [s for s in mesh.strips() if mesh.has_strip_poles(s)][0]

    split_strip(mesh, skey, 3)

    assert mesh.poles() == [pole]
    assert len(mesh.vertex_pole_faces(pole)) == fan + 2
    _assert_consistent(mesh)


def test_both_sides_of_a_new_pole_strip_end_at_the_pole():
    mesh = _centre_pole(density=4)
    skey = [s for s in mesh.strips() if mesh.has_strip_poles(s)][0]
    new = split_strip(mesh, skey, 2)[1]
    left, right = mesh.strip_side_polyedges(new)
    pole = mesh.poles()[0]
    assert pole in (left[0], left[-1]) and pole in (right[0], right[-1])


def _two_poles():
    """A 20 x 10 rectangle with poles at (5, 5) and (15, 5), densified once."""
    vertices = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, 10.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0], [5.0, 5.0, 0.0], [15.0, 5.0, 0.0]]
    faces = [[6, 0, 1], [6, 4, 5], [6, 5, 0], [6, 1, 7, 4], [7, 1, 2], [7, 2, 3], [7, 3, 4]]
    coarse = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(vertices, faces, poles=[[5.0, 5.0, 0.0], [15.0, 5.0, 0.0]])
    coarse.collect_strips()
    coarse.set_strips_density(3)
    coarse.densification()
    mesh = coarse.get_quad_mesh()
    mesh.collect_strips()
    return mesh


def test_a_strip_between_two_poles():
    from compas.topology import shortest_path

    mesh = _two_poles()
    p, q = mesh.poles()
    fans = {pole: len(mesh.vertex_pole_faces(pole)) for pole in (p, q)}
    interior = {v: [n for n in mesh.vertex_neighbors(v) if not mesh.is_vertex_on_boundary(n)] for v in mesh.vertices()}
    polyedge = shortest_path(interior, p, q)
    assert pole_ends(mesh, polyedge) == (True, True)

    pattern_add_strip(mesh, list(polyedge))

    assert sorted(mesh.poles()) == sorted([p, q])
    assert all(len(mesh.vertex_pole_faces(pole)) == fans[pole] + 1 for pole in (p, q))
    _assert_consistent(mesh)


def test_pole_ends_are_off_on_a_plain_quad_mesh(grid4):
    mesh, index = grid4
    assert pole_ends(mesh, [index[0, 1], index[1, 1], index[2, 1]]) == (False, False)


def test_splitting_at_a_corner_keeps_both_copies_on_the_wall():
    """The thirds rule would put both copies on the diagonal across the corner."""
    mesh = _centre_pole()
    skey = [s for s in mesh.strips() if mesh.has_strip_poles(s)][0]
    split_strip(mesh, skey, 2)
    for vkey in mesh.vertices():
        if vkey == 4:
            continue
        x, y, _ = mesh.vertex_coordinates(vkey)
        assert min(abs(x), abs(x - 10.0), abs(y), abs(y - 10.0)) < 1e-9, (vkey, x, y)
    _assert_consistent(mesh)
