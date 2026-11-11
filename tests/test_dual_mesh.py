"""Tests for ``algorithms.dual_mesh``, the library half of CMD08_dual.

The figures pinned here are the ones the redistribution was designed against: the
dual's boundary sits on the primal's edge midpoints, so its outer blocks are half
size (block area max/min 4.00) until ``redistribute_blocks`` evens them out.
"""

from math import cos
from math import pi
from math import sin

import pytest
from compas.datastructures import Mesh

from compas_singular.algorithms.dual_mesh import _is_oriented
from compas_singular.algorithms.dual_mesh import _respaced
from compas_singular.algorithms.dual_mesh import dual_mesh
from compas_singular.datastructures import QuadMesh


def grid(n, m, fold=False):
    """An n x m grid of unit quads; ``fold`` bends it into a roof along its middle."""
    vertices = [[float(i), float(j), abs(i - n / 2.0) if fold else 0.0] for j in range(m + 1) for i in range(n + 1)]
    faces = [[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i] for j in range(m) for i in range(n)]
    return vertices, faces


def annulus(n=16, rings=3, r0=1.0, r1=3.0):
    """Rings of quads round a circular hole: two smooth boundaries, no corners."""
    vertices = [[(r0 + (r1 - r0) * k / rings) * cos(2 * pi * i / n), (r0 + (r1 - r0) * k / rings) * sin(2 * pi * i / n), 0.0] for k in range(rings + 1) for i in range(n)]
    faces = [[k * n + i, k * n + (i + 1) % n, (k + 1) * n + (i + 1) % n, (k + 1) * n + i] for k in range(rings) for i in range(n)]
    return vertices, faces


def spread(mesh):
    areas = [mesh.face_area(fkey) for fkey in mesh.faces()]
    return max(areas) / min(areas)


def coordinates(mesh):
    return sorted(tuple(round(c, 9) for c in mesh.vertex_coordinates(v)) for v in mesh.vertices())


@pytest.fixture
def plain_grid():
    return Mesh.from_vertices_and_faces(*grid(6, 4))


# ----------------------------------------------------------------------------
# topology
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [grid(6, 4), grid(6, 4, fold=True), annulus()])
def test_the_dual_is_all_quad(shape):
    dual = dual_mesh(Mesh.from_vertices_and_faces(*shape))
    assert {len(dual.face_vertices(fkey)) for fkey in dual.faces()} == {4}


def test_a_grid_dual_has_one_block_per_primal_vertex(plain_grid):
    # interior vertex -> conway dual face, boundary vertex -> one closing quad
    dual = dual_mesh(plain_grid)
    assert dual.number_of_faces() == plain_grid.number_of_vertices() == 35


def test_an_annulus_dual_keeps_both_boundaries():
    primal = Mesh.from_vertices_and_faces(*annulus())
    dual = dual_mesh(primal)
    assert dual.number_of_faces() == primal.number_of_vertices()
    assert len(dual.vertices_on_boundaries()) == 2


def test_the_dual_keeps_the_primal_orientation(plain_grid):
    dual = dual_mesh(plain_grid)
    up = plain_grid.face_normal(next(iter(plain_grid.faces())))
    for fkey in dual.faces():
        normal = dual.face_normal(fkey)
        assert sum(a * b for a, b in zip(up, normal)) == pytest.approx(1.0)


def test_the_dual_faces_are_consistently_wound(plain_grid):
    dual = dual_mesh(plain_grid)
    assert _is_oriented([dual.face_vertices(fkey) for fkey in dual.faces()])


def test_is_oriented_catches_a_reversed_face():
    assert _is_oriented([[0, 1, 2, 3], [1, 4, 5, 2]])
    assert not _is_oriented([[0, 1, 2, 3], [2, 5, 4, 1]])  # claims halfedge (1, 2) twice


# ----------------------------------------------------------------------------
# geometry: the domain is kept, the half blocks are evened out
# ----------------------------------------------------------------------------


def test_a_planar_dual_covers_the_primal_exactly(plain_grid):
    assert dual_mesh(plain_grid, redistribute=False).area() == pytest.approx(plain_grid.area())
    assert dual_mesh(plain_grid).area() == pytest.approx(plain_grid.area())


def test_unredistributed_boundary_blocks_are_half_and_quarter_size(plain_grid):
    assert spread(dual_mesh(plain_grid, redistribute=False)) == pytest.approx(4.0)


def test_redistribution_makes_a_grid_dual_uniform(plain_grid):
    assert spread(dual_mesh(plain_grid)) == pytest.approx(1.0)


def test_redistribution_on_a_folded_slab():
    primal = Mesh.from_vertices_and_faces(*grid(6, 4, fold=True))
    assert spread(dual_mesh(primal, redistribute=False)) == pytest.approx(4.0)
    assert spread(dual_mesh(primal)) == pytest.approx(1.354, abs=1e-3)


def test_redistributed_dual_stays_on_a_folded_slab():
    # every dual vertex must lie on one of the two roof planes, z = |x - 3|
    dual = dual_mesh(Mesh.from_vertices_and_faces(*grid(6, 4, fold=True)))
    for vkey in dual.vertices():
        x, _, z = dual.vertex_coordinates(vkey)
        assert z == pytest.approx(abs(x - 3.0), abs=1e-9)


def test_redistribution_on_an_annulus_does_not_shrink_it():
    # the smooth rings have no corner, so only the radial strips are respaced;
    # the remaining spread is the annulus's own (its primal spread is 2.0)
    primal = Mesh.from_vertices_and_faces(*annulus())
    raw = dual_mesh(primal, redistribute=False)
    even = dual_mesh(primal)
    assert spread(raw) == pytest.approx(4.0)
    assert spread(even) < 2.25
    assert even.area() == pytest.approx(raw.area())


def test_respaced_doubles_the_end_gaps_and_keeps_the_span():
    # gaps 0.5, 1, 1, 0.5 -> doubled ends give four equal gaps of 0.75
    assert _respaced([0.0, 0.5, 1.5, 2.5, 3.0]) == pytest.approx([0.75, 1.5, 2.25])


# ----------------------------------------------------------------------------
# input types
# ----------------------------------------------------------------------------


def test_a_quad_mesh_dualises_like_a_plain_mesh():
    # compas_singular's Mesh.to_vertices_and_faces returns dicts by default,
    # which used to break dual_mesh three frames deep with a TypeError
    vertices, faces = grid(3, 3)
    plain = dual_mesh(Mesh.from_vertices_and_faces(vertices, faces))
    quad = dual_mesh(QuadMesh.from_vertices_and_faces(vertices, faces))
    assert isinstance(quad, QuadMesh)
    assert coordinates(quad) == coordinates(plain)


def test_dual_mesh_does_not_move_the_primal(plain_grid):
    before = coordinates(plain_grid)
    dual_mesh(plain_grid)
    assert coordinates(plain_grid) == before
