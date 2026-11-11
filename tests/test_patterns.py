"""Patch patterns -- the diagonal's density check.

The diagonal pattern draws both diagonals of every patch as chains of cell
diagonals. A chain is unbroken only when the patch is ``d`` by ``d`` with ``d``
even, and ``d`` comes from the two strips crossing the patch, so
``reconcile_strip_densities`` has to tie those strips together before any patch
is built. Measured on the template before that check existed: 4x2, 6x3 and 8x5
broke BOTH diagonals, 3x3, 5x5 and 7x7 broke b->d at the centre cell.
"""

import math

import pytest

from compas.geometry import Polygon
from compas.tolerance import TOL

from compas_singular.algorithms.skeleton_decomposition import SkeletonDecomposition
from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.datastructures.mesh_quad_coarse.patterns import create_pattern
from compas_singular.datastructures.mesh_quad_coarse.patterns import patch_divisions
from compas_singular.datastructures.mesh_quad_coarse.patterns import pattern_morph
from compas_singular.datastructures.mesh_quad_coarse.patterns import reconcile_strip_densities


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def connected(adjacency, start, end):
    seen, stack = {start}, [start]
    while stack:
        x = stack.pop()
        if x == end:
            return True
        for y in adjacency.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return False


def template_cuts(uw, faces):
    """The template edges that are not grid edges -- the cell diagonals."""
    adjacency = {}
    for face in faces:
        for a, b in zip(face, face[1:] + face[:1]):
            (ua, wa), (ub, wb) = uw[a], uw[b]
            if ua != ub and wa != wb:
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
    return adjacency


def diagonals_are_whole(coarse, dense, fkey):
    """Do both diagonals of one patch run corner to corner in the dense mesh?

    Walks cell diagonals only -- dense edges from grid point ``(i, j)`` to
    ``(i+1, j+-1)``. "An edge with a triangle on both sides" is NOT enough: in a
    staircase two neighbouring cut cells share a grid edge that has exactly
    that, and connects the staircase up. The grid points come from morphing the
    plain grid onto the patch's own sides, which is where the pattern put them.

    Returns
    -------
    (bool, bool)
        a -> c, and b -> d.
    """
    edge_strip = {}
    for skey, edges in coarse.strips(data=True):
        for u, v in edges:
            edge_strip[u, v] = edge_strip[v, u] = skey
    sides = coarse._patch_sides(fkey, edge_strip)
    nu, nw = patch_divisions(sides)
    uw, faces = create_pattern("ortho", nu, nw)
    points, _ = pattern_morph(uw, faces, sides)

    vkey = {TOL.geometric_key(dense.vertex_coordinates(v)): v for v in dense.vertices()}
    grid = {(i, j): vkey[TOL.geometric_key(points[i * (nw + 1) + j])] for i in range(nu + 1) for j in range(nw + 1)}

    # ``halfedge``, not ``Mesh.has_edge``: that one is ``key in set(edges())``,
    # so it answers False for an edge stored the other way round
    adjacency = {}
    for (i, j), p in grid.items():
        for q in (grid.get((i + 1, j + 1)), grid.get((i + 1, j - 1))):
            if q is not None and q != p and q in dense.halfedge[p]:
                adjacency.setdefault(p, set()).add(q)
                adjacency.setdefault(q, set()).add(p)

    return (connected(adjacency, grid[0, 0], grid[nu, nw]), connected(adjacency, grid[nu, 0], grid[0, nw]))


def rectangle(t=0.5):
    """A 5 x 10 rectangle. Target-length densities differ between its strips."""
    outer = Polygon([[0, 0, 0], [5, 0, 0], [5, 10, 0], [0, 10, 0]])
    coarse = SkeletonDecomposition.from_boundary(outer_boundary=outer).coarse_mesh()
    coarse.set_strips_density_target(t=t)
    return coarse


def square_and_triangle(shared=3, across=4, alone=5):
    """A 10 x 10 square with a triangle on top of it, the apex its pole.

    Three strips, set to the given densities: ``shared`` runs bottom -> top ->
    pole through both patches, ``across`` runs left-right through the square
    only, ``alone`` runs through the triangle only.

    Returns
    -------
    coarse, quad, triangle, {name: skey}
    """
    P = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 18, 0]]
    polylines = [[P[0], P[1]], [P[1], P[2]], [P[2], P[3]], [P[3], P[0]], [P[2], P[4]], [P[4], P[3]]]
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(polylines, poles=[P[4]])

    quad = next(f for f in coarse.faces() if not coarse.is_face_pseudo_quad(f))
    triangle = next(f for f in coarse.faces() if coarse.is_face_pseudo_quad(f))
    in_quad, in_triangle = set(coarse.face_strips(quad)), set(coarse.face_strips(triangle))
    strips = {"shared": (in_quad & in_triangle).pop(), "across": (in_quad - in_triangle).pop(), "alone": (in_triangle - in_quad).pop()}
    for name, d in (("shared", shared), ("across", across), ("alone", alone)):
        coarse.set_strip_density(strips[name], d)
    return coarse, quad, triangle, strips


def densities(coarse, strips):
    return {name: coarse.get_strip_density(skey) for name, skey in strips.items()}


# ----------------------------------------------------------------------------
# the template
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 4, 6, 8, 12])
def test_both_diagonals_are_one_straight_chain(n):
    uw, faces = create_pattern("diagonal", n, n)
    cuts = template_cuts(uw, faces)
    index = {p: i for i, p in enumerate(uw)}

    assert connected(cuts, index[0.0, 0.0], index[1.0, 1.0])  # a -> c
    assert connected(cuts, index[1.0, 0.0], index[0.0, 1.0])  # b -> d

    # straight: every cut lies on one of the two patch diagonals
    for a, others in cuts.items():
        for b in others:
            (ua, wa), (ub, wb) = uw[a], uw[b]
            on_main = abs(ua - wa) < 1e-12 and abs(ub - wb) < 1e-12
            on_anti = abs(ua + wa - 1) < 1e-12 and abs(ub + wb - 1) < 1e-12
            assert on_main or on_anti


@pytest.mark.parametrize("nu, nw", [(4, 2), (6, 3), (8, 5), (3, 3), (5, 5)])
def test_a_size_that_would_break_a_diagonal_is_refused(nu, nw):
    with pytest.raises(ValueError, match="reconcile_strip_densities"):
        create_pattern("diagonal", nu, nw)


# ----------------------------------------------------------------------------
# the density check
# ----------------------------------------------------------------------------


def test_every_diagonal_patch_is_square_and_even():
    coarse = rectangle()
    before = dict(coarse.get_strip_densities())
    assert any(len({before[s] for s in coarse.face_strips(f)}) > 1 for f in coarse.faces())

    changed = reconcile_strip_densities(coarse, "diagonal")

    after = coarse.get_strip_densities()
    for fkey in coarse.faces():
        nu, nw = (after[s] for s in coarse.face_strips(fkey))
        assert nu == nw and nu % 2 == 0
    # refined, never coarsened, and the report is exactly what moved
    assert all(after[s] >= before[s] for s in before)
    assert changed == {s: (before[s], after[s]) for s in before if before[s] != after[s]}


def test_a_second_pass_changes_nothing():
    coarse = rectangle()
    reconcile_strip_densities(coarse, "diagonal")
    assert reconcile_strip_densities(coarse, "diagonal") == {}


def test_ortho_patches_tie_nothing():
    coarse, quad, triangle, strips = square_and_triangle(shared=3, across=4, alone=5)

    reconcile_strip_densities(coarse, {quad: "diagonal", triangle: "ortho"})

    # the square's two strips meet at the larger, already even; the strip that
    # crosses only the ortho triangle keeps its odd density
    assert densities(coarse, strips) == {"shared": 4, "across": 4, "alone": 5}


def test_a_pseudo_quad_ties_its_two_strips_too():
    coarse, _quad, _triangle, strips = square_and_triangle(shared=3, across=4, alone=5)

    reconcile_strip_densities(coarse, "diagonal")

    # the square ties shared-across, the triangle shared-alone: one group,
    # max 5, rounded up
    assert densities(coarse, strips) == {"shared": 6, "across": 6, "alone": 6}


def test_fan_still_takes_one_even_density_everywhere():
    coarse = rectangle()
    top = max(coarse.get_strip_densities().values())

    reconcile_strip_densities(coarse, "fan")

    assert set(coarse.get_strip_densities().values()) == {top + top % 2}


# ----------------------------------------------------------------------------
# end to end
# ----------------------------------------------------------------------------


def test_densified_diagonals_run_corner_to_corner_in_every_patch():
    """Target-length densities in, the Rhino workflow's own path."""
    coarse = rectangle()
    dense = coarse.quad_mesh(pattern_overwrite="diagonal")

    for fkey in coarse.faces():
        assert diagonals_are_whole(coarse, dense, fkey) == (True, True), fkey


def test_diagonal_on_a_pseudo_quad_densifies():
    coarse, quad, triangle, _strips = square_and_triangle(shared=3, across=4, alone=5)

    dense = coarse.quad_mesh(pattern_overwrite="diagonal")

    # the cut cells on the collapsed side lose one triangle to the weld and
    # keep the other -- the one carrying the cut -- so nothing degenerate
    # survives and the diagonal still reaches the pole
    assert all(len(dense.face_vertices(f)) >= 3 for f in dense.faces())
    assert dense.is_manifold()
    assert diagonals_are_whole(coarse, dense, quad) == (True, True)
    assert diagonals_are_whole(coarse, dense, triangle) == (True, True)


# ----------------------------------------------------------------------------
# the triangular fan
# ----------------------------------------------------------------------------
#
# 'fan' on a pseudo-quad is three polar fans, one per corner -- the pole
# included -- meeting at the centre. Before it existed the quad fan could not
# be used there (on a zero-length side it left d - 2 zero-area faces and a pole
# of valency 3d - 3), so densify quietly swapped in ortho. Patterns are set on
# the faces, not through ``pattern_overwrite``, which only reaches the
# densities.

S3 = math.sqrt(3.0)
EQUILATERAL = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, S3, 0.0]]


def signed_area(points):
    return 0.5 * sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(points, points[1:] + points[:1]))


def hausdorff(a, b):
    def one_way(xs, ys):
        return max(min(math.dist(x, y) for y in ys) for x in xs)

    return max(one_way(a, b), one_way(b, a))


def arc(p, q, bulge, n=24):
    """``p`` to ``q``, bowed ``bulge`` to the right of the chord at its middle."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    length = math.hypot(dx, dy)
    nx, ny = dy / length, -dx / length
    return [[p[0] + dx * t + nx * bulge * 4 * t * (1 - t), p[1] + dy * t + ny * bulge * 4 * t * (1 - t), 0.0] for t in (i / n for i in range(n + 1))]


def triangle(pole, d, pattern="fan", bulge=None):
    """One triangular patch on the equilateral triangle, the pole on corner ``pole``.

    Returns the dense mesh. ``bulge`` bows all three sides alike, so the input
    keeps the triangle's threefold symmetry.
    """
    if bulge is None:
        coarse = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(EQUILATERAL, [[0, 1, 2]], [EQUILATERAL[pole]])
        coarse.collect_strips()
        curves = None
    else:
        a, b, c = EQUILATERAL
        coarse = CoarsePseudoQuadMesh.from_coarse_polylines([arc(a, b, bulge), arc(b, c, bulge), arc(c, a, bulge)], poles=[EQUILATERAL[pole]])
        curves = coarse.edges_to_curves()
    coarse.set_strips_density(d)
    coarse.set_global_face_pattern(pattern)
    return coarse.densify(overwrite_edges_to_curves=curves)


def vertex_at(mesh, xyz):
    return min(mesh.vertices(), key=lambda v: math.dist(mesh.vertex_coordinates(v), xyz))


def assert_well_formed(dense):
    areas = [signed_area(dense.face_coordinates(f)) for f in dense.faces()]
    assert all(abs(a) > 1e-9 for a in areas), "a zero-area face"
    assert all(a > 0 for a in areas) or all(a < 0 for a in areas), "faces wound both ways"
    assert dense.is_manifold()
    assert len(dense.vertices_on_boundaries()) == 1
    return sum(abs(a) for a in areas)


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_triangular_fan_template_is_three_corner_fans(n):
    lam, faces = create_pattern("fan_triangle", n, n)

    # three corners, each rays x rings = n x n/2 cells
    assert len(faces) == 3 * n * n // 2
    # every side carries exactly the n + 1 points an ortho neighbour puts on it
    for i in range(3):
        on_side = {round(p[(i + 2) % 3], 12) for p in lam if p[i] == 0.0}
        assert on_side == {round(k / n, 12) for k in range(n + 1)}


@pytest.mark.parametrize("n", [2, 4, 6, 8])
def test_triangular_fan_template_is_the_same_at_every_corner(n):
    lam, _faces = create_pattern("fan_triangle", n, n)

    points = {tuple(round(x, 12) for x in p) for p in lam}
    assert {(p[2], p[0], p[1]) for p in points} == points


@pytest.mark.parametrize("nu, nw", [(4, 2), (3, 3), (5, 5)])
def test_a_size_that_would_break_the_triangular_fan_is_refused(nu, nw):
    with pytest.raises(ValueError, match="reconcile_strip_densities"):
        create_pattern("fan_triangle", nu, nw)


@pytest.mark.parametrize("bulge", [None, 0.3, -0.2])
@pytest.mark.parametrize("d", [4, 6, 8])
def test_every_corner_of_a_triangular_fan_is_the_same(d, bulge):
    """The pole is a corner like the other two -- straight sides or curved.

    With the fans laid out in (u, w) and mapped by the pseudo-quad's Coons
    patch instead, this was 0.08 off on straight sides and up to 0.14 on
    curved ones.
    """
    meshes = [triangle(pole, d, bulge=bulge) for pole in range(3)]
    points = [[m.vertex_coordinates(v) for v in m.vertices()] for m in meshes]

    cx, cy = 1.0, S3 / 3.0
    turn = 2.0 * math.pi / 3.0
    turned = [[cx + (x - cx) * math.cos(turn) - (y - cy) * math.sin(turn), cy + (x - cx) * math.sin(turn) + (y - cy) * math.cos(turn), z] for x, y, z in points[0]]

    assert hausdorff(turned, points[0]) < 1e-9
    assert hausdorff(points[0], points[1]) < 1e-9
    assert hausdorff(points[0], points[2]) < 1e-9
    for corner in EQUILATERAL:
        assert meshes[0].vertex_degree(vertex_at(meshes[0], corner)) == d + 1


@pytest.mark.parametrize("d", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("pole", [0, 1, 2])
def test_a_triangular_fan_is_well_formed(pole, d):
    dense = triangle(pole, d)
    area = assert_well_formed(dense)

    assert dense.number_of_faces() == 3 * d * d // 2  # not an ortho fallback
    assert abs(area - assert_well_formed(triangle(pole, d, "ortho"))) < 1e-9


@pytest.mark.parametrize("pattern", ["ortho", "fan"])
@pytest.mark.parametrize("d", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("pole", [0, 1, 2])
def test_every_pole_face_is_registered(pole, d, pattern):
    """Pole faces were matched across the weld by their centre, rounded.

    A centre on a rounding digit came out 0.313 before the weld and 0.312 after
    and dropped its face from ``face_pole`` -- the fan at pole 1, d 4.
    """
    dense = triangle(pole, d, pattern)
    pole_key = vertex_at(dense, EQUILATERAL[pole])

    at_pole = {f for f in dense.vertex_faces(pole_key) if len(dense.face_vertices(f)) == 3}
    assert len(at_pole) == d
    assert dense.attributes["face_pole"] == {f: pole_key for f in at_pole}


def test_the_spokes_of_a_straight_triangular_fan_are_straight():
    d = 8
    dense = triangle(0, d)
    points = [dense.vertex_coordinates(v) for v in dense.vertices()]
    centre = [1.0, S3 / 3.0, 0.0]
    a, b, c = EQUILATERAL

    def on_segment(p, q):
        return sum(1 for x in points if abs(math.dist(p, x) + math.dist(x, q) - math.dist(p, q)) < 1e-9)

    # a corner's middle ray, and a seam between two corners: rings + 1 each
    for start in (a, b, c, [(a[i] + b[i]) / 2 for i in range(3)], [(b[i] + c[i]) / 2 for i in range(3)], [(c[i] + a[i]) / 2 for i in range(3)]):
        assert on_segment(start, centre) == d // 2 + 1


@pytest.mark.parametrize("pole", ["apex", "shared"])
@pytest.mark.parametrize("quad_pattern", ["ortho", "diagonal", "fan"])
def test_a_triangular_fan_welds_to_any_neighbour(quad_pattern, pole):
    """A square under a triangle, the edge between them curved."""
    P = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [5, 18, 0]]
    polylines = [[P[0], P[1]], [P[1], P[2]], arc(P[2], P[3], -1.0), [P[3], P[0]], arc(P[2], P[4], -1.5), arc(P[4], P[3], -1.5)]
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(polylines, poles=[P[4] if pole == "apex" else P[2]])
    quad = next(f for f in coarse.faces() if not coarse.is_face_pseudo_quad(f))
    coarse.set_strips_density(4)
    coarse.set_global_face_pattern("fan")
    coarse.set_face_pattern(quad, quad_pattern)

    dense = coarse.densify(overwrite_edges_to_curves=coarse.edges_to_curves())

    assert_well_formed(dense)
    # the curved edge between the two is interior: one boundary loop carrying
    # exactly the coarse boundary's divisions, so nothing along it failed to weld
    edge_strip = {}
    for skey, edges in coarse.strips(data=True):
        for u, v in edges:
            edge_strip[u, v] = edge_strip[v, u] = skey
    expected = sum(coarse.get_strip_density(edge_strip[edge]) for edge in coarse.edges_on_boundary())
    # a boundary loop repeats its first vertex at the end
    assert len(set(dense.vertices_on_boundaries()[0])) == expected


def test_a_full_pole_of_triangular_fans():
    """Four triangles meeting at a centre pole, every one a fan."""
    d = 4
    vertices = [[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0], [2, 2, 0]]
    faces = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
    coarse = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(vertices, faces, [vertices[4]])
    coarse.collect_strips()
    coarse.set_strips_density(d)
    coarse.set_global_face_pattern("fan")

    dense = coarse.densify()

    assert_well_formed(dense)
    pole = vertex_at(dense, vertices[4])
    assert dense.vertex_degree(pole) == 4 * d
    assert sorted(dense.attributes["face_pole"]) == sorted(dense.vertex_faces(pole))
