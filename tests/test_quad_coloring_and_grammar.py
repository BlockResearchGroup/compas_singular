"""Tests for the strip/polyedge colourings and the small topology grammars around QuadMesh.

Covers ``topology.coloring``, ``mesh_quad.coloring``, ``mesh_quad.morphing``,
``mesh_quad.grammar_shape``, ``mesh_quad_pseudo.grammar_poles``, the singularity
decompositions and strip graph on ``QuadMesh``, and ``utilities.lists``.
"""

from math import cos
from math import pi
from math import sin

import pytest

from compas_singular.datastructures import PseudoQuadMesh
from compas_singular.datastructures import QuadMesh
from compas_singular.datastructures.mesh_quad.coloring import quad_mesh_polyedge_2_coloring
from compas_singular.datastructures.mesh_quad.coloring import quad_mesh_polyedge_n_coloring
from compas_singular.datastructures.mesh_quad.coloring import quad_mesh_strip_2_coloring
from compas_singular.datastructures.mesh_quad.coloring import quad_mesh_strip_n_coloring
from compas_singular.datastructures.mesh_quad.grammar_shape import add_handle
from compas_singular.datastructures.mesh_quad.grammar_shape import add_opening
from compas_singular.datastructures.mesh_quad.morphing import fold
from compas_singular.datastructures.mesh_quad.morphing import fold_vertex_group
from compas_singular.datastructures.mesh_quad_pseudo.grammar_poles import merge_pseudo_quads_in_quad
from compas_singular.datastructures.mesh_quad_pseudo.grammar_poles import split_quad_in_pseudo_quads
from compas_singular.topology import is_adjacency_two_colorable
from compas_singular.utilities import are_items_in_list
from compas_singular.utilities import common_items
from compas_singular.utilities import list_split
from compas_singular.utilities.lists import remove_isomorphism_in_integer_list
from compas_singular.utilities.lists import sublist_from_to_items_in_closed_list


def grid(n, m):
    vertices = [[float(i), float(j), 0.0] for j in range(m + 1) for i in range(n + 1)]
    faces = [[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i] for j in range(m) for i in range(n)]
    return QuadMesh.from_vertices_and_faces(vertices, faces)


def star(k):
    """k quads round an interior vertex 0 of valence k: a singularity unless k == 4."""
    vertices = [[0.0, 0.0, 0.0]]
    for i in range(k):
        vertices.append([cos(2 * pi * i / k), sin(2 * pi * i / k), 0.0])
        vertices.append([1.5 * cos(2 * pi * (i + 0.5) / k), 1.5 * sin(2 * pi * (i + 0.5) / k), 0.0])
    faces = [[0, 1 + 2 * i, 2 + 2 * i, 1 + (2 * i + 2) % (2 * k)] for i in range(k)]
    mesh = QuadMesh.from_vertices_and_faces(vertices, faces)
    mesh.collect_strips()
    mesh.collect_polyedges(strict=True)
    return mesh


def is_proper(adjacency, colors):
    return all(colors[u] != colors[v] for u in adjacency for v in adjacency[u] if u != v)


# ----------------------------------------------------------------------------
# is_adjacency_two_colorable
# ----------------------------------------------------------------------------


def test_an_even_cycle_is_two_colourable():
    adjacency = {0: [1, 3], 1: [0, 2], 2: [1, 3], 3: [2, 0]}
    colors = is_adjacency_two_colorable(adjacency)
    assert set(colors.values()) == {0, 1}
    assert is_proper(adjacency, colors)


@pytest.mark.parametrize("n", [3, 5])
def test_an_odd_cycle_is_not(n):
    adjacency = {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}
    assert is_adjacency_two_colorable(adjacency) is None


def test_every_component_is_coloured():
    # a disconnected graph used to come back with -1 for every component but the first
    adjacency = {0: [1], 1: [0], 2: [3], 3: [2], 4: []}
    colors = is_adjacency_two_colorable(adjacency)
    assert -1 not in colors.values()
    assert is_proper(adjacency, colors)


def test_an_odd_cycle_in_a_later_component_is_still_found():
    adjacency = {0: [1], 1: [0], 2: [3, 4], 3: [2, 4], 4: [2, 3]}
    assert is_adjacency_two_colorable(adjacency) is None


def test_a_self_loop_is_ignored():
    # polyedge_graph(legacy=True) emits self-loops by design
    assert is_adjacency_two_colorable({0: [0, 1], 1: [0]}) == {0: 0, 1: 1}


# ----------------------------------------------------------------------------
# strip and polyedge colourings
# ----------------------------------------------------------------------------


def test_a_regular_grid_colours_its_strips_by_direction():
    mesh = grid(3, 2)
    mesh.collect_strips()
    colors = quad_mesh_strip_2_coloring(mesh)
    by_direction = {True: set(), False: set()}
    for skey, color in colors.items():
        u, v = mesh.strip_edges(skey)[0]
        by_direction[mesh.vertex_coordinates(u)[0] == mesh.vertex_coordinates(v)[0]].add(color)
    assert len(by_direction[True]) == len(by_direction[False]) == 1
    assert by_direction[True] != by_direction[False]


@pytest.mark.parametrize("k, colourable", [(3, False), (4, True), (5, False)])
def test_strips_are_two_colourable_only_round_an_even_valence(k, colourable):
    assert (quad_mesh_strip_2_coloring(star(k)) is not None) == colourable


@pytest.mark.parametrize("k", [3, 4, 5])
def test_strip_n_colouring_never_gives_crossing_strips_one_colour(k):
    mesh = star(k)
    colors = quad_mesh_strip_n_coloring(mesh)
    for fkey in mesh.faces():
        a, b = mesh.face_strips(fkey)
        assert colors[a] != colors[b]


@pytest.mark.parametrize("k", [3, 5])
def test_polyedges_round_an_odd_singularity_are_two_colourable(k):
    # the spokes only meet at the singularity, which does not count as crossing,
    # so the polyedge graph splits into components -- each must be coloured
    mesh = star(k)
    colors = quad_mesh_polyedge_2_coloring(mesh)
    assert set(colors.values()) == {0, 1}
    spokes = {pkey for pkey, polyedge in mesh.polyedges(data=True) if 0 in polyedge}
    assert len({colors[pkey] for pkey in spokes}) == 1


def test_polyedge_colouring_by_edge_matches_the_grid_directions():
    mesh = grid(3, 3)
    mesh.collect_polyedges(strict=True)
    colors = quad_mesh_polyedge_2_coloring(mesh, edge_output=True)
    horizontal = {colors[(u, v)] for u, v in mesh.edges() if mesh.vertex_coordinates(u)[1] == mesh.vertex_coordinates(v)[1]}
    vertical = {colors[(u, v)] for u, v in mesh.edges() if mesh.vertex_coordinates(u)[0] == mesh.vertex_coordinates(v)[0]}
    assert len(horizontal) == len(vertical) == 1
    assert horizontal != vertical
    assert all(colors[(u, v)] == colors[(v, u)] for u, v in mesh.edges())


def test_polyedge_n_colouring_of_a_grid_needs_two_colours():
    mesh = grid(3, 3)
    mesh.collect_polyedges(strict=True)
    assert set(quad_mesh_polyedge_n_coloring(mesh).values()) == {0, 1}


# ----------------------------------------------------------------------------
# morphing: fold groups
# ----------------------------------------------------------------------------


def test_fold_groups_alternate_across_the_selected_polyedges():
    # crossing an edge of a selected polyedge flips the group, any other edge keeps it;
    # selecting every vertical line therefore alternates the groups row by row
    mesh = grid(3, 3)
    mesh.collect_polyedges(strict=True)
    verticals = [polyedge for _, polyedge in mesh.polyedges(data=True) if mesh.vertex_coordinates(polyedge[0])[0] == mesh.vertex_coordinates(polyedge[-1])[0]]
    groups = fold_vertex_group(mesh, verticals)
    assert set(groups) == set(mesh.vertices())
    for vkey, group in groups.items():
        row = int(mesh.vertex_coordinates(vkey)[1])
        assert group == (row + groups[0]) % 2


def test_fold_moves_each_group_by_its_own_function():
    mesh = grid(1, 1)
    groups = {0: 0, 1: 1, 2: 1, 3: 0}
    moves = fold(mesh, groups, lambda m, v: [0.0, 0.0, 1.0], lambda m, v: [0.0, 0.0, -1.0])
    assert moves[0] == [0.0, 0.0, 1.0] and moves[1] == [0.0, 0.0, -1.0]
    assert [mesh.vertex_coordinates(v)[2] for v in range(4)] == [1.0, -1.0, -1.0, 1.0]


# ----------------------------------------------------------------------------
# shape grammar: openings and handles
# ----------------------------------------------------------------------------


def test_add_opening_punches_a_hole_ringed_by_quads():
    mesh = grid(2, 2)
    new = add_opening(mesh, 0)
    assert len(new) == 4
    assert mesh.number_of_faces() == 4 - 1 + 4
    assert len(mesh.vertices_on_boundaries()) == 2
    assert mesh.euler() == 0  # a disc with a hole: an annulus
    assert all(len(mesh.face_vertices(f)) == 4 for f in mesh.faces())


def test_the_opening_sits_halfway_to_the_face_centre():
    mesh = grid(1, 1)
    new = add_opening(mesh, 0)
    xs = sorted(round(mesh.vertex_coordinates(v)[0], 9) for v in new)
    assert xs == [0.25, 0.25, 0.75, 0.75]


def test_add_handle_adds_one_genus():
    mesh = grid(3, 1)
    euler = mesh.euler()
    handle = add_handle(mesh, 0, 2)
    assert len(handle) == 4
    assert mesh.euler() == euler - 2
    assert mesh.is_manifold()
    assert all(len(mesh.face_vertices(f)) == 4 for f in mesh.faces())


# ----------------------------------------------------------------------------
# pseudo-quad grammar: split and merge
# ----------------------------------------------------------------------------


def test_split_and_merge_a_quad_round_trip():
    mesh = PseudoQuadMesh.from_vertices_and_faces([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], [[0, 1, 2, 3]])
    poles = split_quad_in_pseudo_quads(mesh, 0, 0)
    assert set(poles.values()) == {0}  # both triangles keep their pole at the cut vertex
    assert sorted(len(mesh.face_vertices(f)) for f in mesh.faces()) == [3, 3]

    a, b = poles
    fkey = merge_pseudo_quads_in_quad(mesh, a, b)
    assert mesh.number_of_faces() == 1
    face = mesh.face_vertices(fkey)
    i = face.index(0)
    assert face[i:] + face[:i] == [0, 1, 2, 3]  # same winding as the original


def test_split_refuses_a_non_quad():
    mesh = PseudoQuadMesh.from_vertices_and_faces([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], [[0, 1, 2]])
    assert split_quad_in_pseudo_quads(mesh, 0, 0) is None


def test_merge_refuses_faces_that_do_not_touch():
    mesh = PseudoQuadMesh.from_vertices_and_faces([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0]], [[0, 1, 2], [3, 4, 5]])
    assert merge_pseudo_quads_in_quad(mesh, 0, 1) is None


# ----------------------------------------------------------------------------
# QuadMesh: singularity decomposition and the strip graph
# ----------------------------------------------------------------------------


def test_a_regular_grid_decomposes_into_its_four_sides():
    mesh = grid(2, 2)
    mesh.collect_polyedges(strict=True)
    patches = mesh.singularity_polyedge_decomposition(strict=True)
    assert sorted(sorted(p) for p in patches) == [[0, 1, 2], [0, 3, 6], [2, 5, 8], [6, 7, 8]]


def test_a_singularity_splits_the_decomposition_at_it():
    mesh = star(3)
    polyedges = mesh.singularity_polyedges()
    assert all(polyedge[0] == 0 or polyedge[-1] == 0 or mesh.is_edge_on_boundary(polyedge[0], polyedge[1]) for polyedge in polyedges)
    assert sum(1 for p in polyedges if 0 in (p[0], p[-1])) == 3  # one per spoke


def test_strip_side_polyedges_bound_the_strip():
    mesh = grid(2, 2)
    mesh.collect_strips()
    for skey in mesh.strips():
        left, right = mesh.strip_side_polyedges(skey)
        assert len(left) == len(right) == 3
        assert list(zip(left, right)) == [tuple(e) for e in mesh.strip_edges(skey)]


def test_the_strip_graph_has_one_edge_per_face():
    mesh = grid(2, 2)
    mesh.collect_strips()
    vertices, edges = mesh.strip_graph()
    assert set(vertices) == set(mesh.strips())
    assert len(edges) == mesh.number_of_faces()
    assert all(a != b for a, b in edges)  # a face is never crossed twice by one strip here


# ----------------------------------------------------------------------------
# utilities.lists
# ----------------------------------------------------------------------------


def test_list_split_open_and_closed():
    assert list_split([0, 1, 2, 3, 4], [2]) == [[0, 1, 2], [2, 3, 4]]
    assert list_split([0, 1, 2, 3, 0], [0, 2]) == [[0, 1, 2], [2, 3, 0]]
    assert list_split([0, 1, 2, 3, 0], [2]) == [[2, 3, 0, 1, 2]]  # one cut of a loop re-roots it


def test_sublist_in_a_closed_list_follows_the_loop():
    loop = [0, 1, 2, 3, 4, 0]
    assert sublist_from_to_items_in_closed_list(loop, 1, 3) == [1, 2, 3]
    assert sublist_from_to_items_in_closed_list(loop, 3, 1) == [3, 4, 0, 1]


def test_membership_helpers():
    assert are_items_in_list([1, 2], [2, 3, 1])
    assert not are_items_in_list([5], [1])
    assert common_items([1, 2, 3], [3, 1]) == [1, 3]


@pytest.mark.parametrize("a, b", [([5, 1, 4], [4, 1, 5]), ([3, 1, 2, 3], [2, 3, 1, 2]), ([2, 0, 1, 2], [1, 2, 0, 1])])
def test_isomorphic_polyedges_share_one_canonical_form(a, b):
    assert remove_isomorphism_in_integer_list(a) == remove_isomorphism_in_integer_list(b)
