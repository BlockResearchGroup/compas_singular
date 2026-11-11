"""``CoarsePseudoQuadMesh.from_coarse_polylines`` -- a drawn network in, a layout out.

The numbers in the first three cases are measured against
``Mesh.from_polylines`` on the same input, which is what the workflow used before:
it returns ZERO faces for anything whose cuts all run wall to wall, because it keeps
a face only when a corner is off the boundary.
"""

from math import cos, pi, sin

import pytest

from compas.geometry import distance_point_point
from compas.itertools import pairwise

from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.datastructures import split_at_corners
from compas_singular.datastructures import split_at_junctions


# ----------------------------------------------------------------------------
# domains
# ----------------------------------------------------------------------------


def square_edges():
    """The four sides of a 10 x 10 square, one polyline each."""
    corners = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]
    return [[corners[i], corners[(i + 1) % 4]] for i in range(4)]


def arc(cx, cy, r, start, end, n=12):
    """An arc sampled at ``n`` segments, angles in degrees."""
    return [[cx + r * cos(pi * (start + (end - start) * k / n) / 180.0), cy + r * sin(pi * (start + (end - start) * k / n) / 180.0), 0.0] for k in range(n + 1)]


def at(r, deg):
    """The point at radius ``r`` and angle ``deg``, degrees."""
    return [r * cos(pi * deg / 180.0), r * sin(pi * deg / 180.0), 0.0]


def signed_area(ring):
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in pairwise(list(ring) + list(ring[:1])))


def sides(mesh):
    out = {}
    for fkey in mesh.faces():
        n = len(mesh.face_vertices(fkey))
        out[n] = out.get(n, 0) + 1
    return out


# ----------------------------------------------------------------------------
# the layouts that work
# ----------------------------------------------------------------------------


def test_square_with_four_arm_cross():
    """The one case ``from_polylines`` also handles -- same answer, 4 quads."""
    polylines = square_edges() + [[[5, 5, 0], [5, 0, 0]], [[5, 5, 0], [10, 5, 0]], [[5, 5, 0], [5, 10, 0]], [[5, 5, 0], [0, 5, 0]]]
    # the walls have to be split at the cross's landings, as they are on screen
    polylines = polylines[4:] + [
        [[0, 0, 0], [5, 0, 0]],
        [[5, 0, 0], [10, 0, 0]],
        [[10, 0, 0], [10, 5, 0]],
        [[10, 5, 0], [10, 10, 0]],
        [[10, 10, 0], [5, 10, 0]],
        [[5, 10, 0], [0, 10, 0]],
        [[0, 10, 0], [0, 5, 0]],
        [[0, 5, 0], [0, 0, 0]],
    ]

    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(polylines)

    assert coarse.number_of_faces() == 4
    assert coarse.number_of_vertices() == 9
    assert sides(coarse) == {4: 4}


def test_square_with_one_cut_wall_to_wall():
    """``from_polylines`` gives 0 faces here: every corner is on a boundary."""
    polylines = [
        [[0, 0, 0], [5, 0, 0]],
        [[5, 0, 0], [10, 0, 0]],
        [[10, 0, 0], [10, 10, 0]],
        [[10, 10, 0], [5, 10, 0]],
        [[5, 10, 0], [0, 10, 0]],
        [[0, 10, 0], [0, 0, 0]],
        [[5, 0, 0], [5, 10, 0]],
    ]

    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(polylines)

    assert sides(coarse) == {4: 2}


def test_square_with_hash_grid():
    """Four strokes drawn as a ``#``, pre-split at their crossings: 9 quads."""
    xs = [0, 3, 7, 10]
    polylines = []
    for j in (0, 3):  # bottom and top walls
        for i in range(3):
            polylines.append([[xs[i], j and 10 or 0, 0], [xs[i + 1], j and 10 or 0, 0]])
    for i in (0, 3):  # left and right walls
        for j in range(3):
            polylines.append([[i and 10 or 0, xs[j], 0], [i and 10 or 0, xs[j + 1], 0]])
    for y in (3, 7):  # horizontals
        for i in range(3):
            polylines.append([[xs[i], y, 0], [xs[i + 1], y, 0]])
    for x in (3, 7):  # verticals
        for j in range(3):
            polylines.append([[x, xs[j], 0], [x, xs[j + 1], 0]])

    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(polylines)

    assert sides(coarse) == {4: 9}
    assert coarse.number_of_vertices() == 16


def annulus_polylines():
    """r = 5 outside, r = 2 inside, cut at 0, 120 and 240 degrees. Curved edges."""
    polylines = []
    for a, b in ((0, 120), (120, 240), (240, 360)):
        polylines.append(arc(0, 0, 5, a, b))
        polylines.append(arc(0, 0, 2, a, b))
    for a in (0, 120, 240):
        polylines.append([at(2, a), at(5, a)])
    return polylines


def test_annulus_hole_is_not_filled():
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(annulus_polylines(), holes=[[0, 0, 0]])

    assert sides(coarse) == {4: 3}
    assert coarse.number_of_vertices() == 6


def test_annulus_without_holes_fills_the_hole():
    """The hole is a bounded cycle like any other; only ``holes`` tells it apart."""
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(annulus_polylines())

    assert coarse.number_of_faces() == 4  # three patches plus the hole


def test_every_face_winds_counter_clockwise():
    """The regression test for orienting on corners instead of on the arcs.

    A curved patch's CORNER polygon can wind the other way to the patch itself, so
    a walk that decides the unbounded face from the corners hands the mesh an
    inverted face. Measured on this annulus before the fix.
    """
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(annulus_polylines(), holes=[[0, 0, 0]])

    for fkey in coarse.faces():
        ring = [coarse.vertex_coordinates(v) for v in coarse.face_vertices(fkey)]
        assert signed_area(ring) > 0.0


def test_triangle_is_a_pseudo_quad():
    polylines = [[[0, 0, 0], [10, 0, 0]], [[10, 0, 0], [5, 8, 0]], [[5, 8, 0], [0, 0, 0]]]

    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(polylines)

    assert coarse.number_of_faces() == 1
    fkey = list(coarse.faces())[0]
    assert coarse.is_face_pseudo_quad(fkey)
    assert len(coarse.poles()) == 1


def test_poles_choose_which_corner_collapses():
    polylines = [[[0, 0, 0], [10, 0, 0]], [[10, 0, 0], [5, 8, 0]], [[5, 8, 0], [0, 0, 0]]]

    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(polylines, poles=[[5, 8, 0]])

    pole = coarse.poles()[0]
    assert distance_point_point(coarse.vertex_coordinates(pole), [5, 8, 0]) < 1e-9


# ----------------------------------------------------------------------------
# the drawings that are refused
# ----------------------------------------------------------------------------


def split_square_edges():
    """The square's walls, already split at the four mid-side landings."""
    ring = [[0, 0, 0], [5, 0, 0], [10, 0, 0], [10, 5, 0], [10, 10, 0], [5, 10, 0], [0, 10, 0], [0, 5, 0]]
    return [[ring[i], ring[(i + 1) % 8]] for i in range(8)]


def test_dangling_curve_raises():
    """A stub running from a wall corner into the middle of nowhere."""
    polylines = split_square_edges() + [[[5, 0, 0], [5, 5, 0]]]

    with pytest.raises(ValueError, match="dangling"):
        CoarsePseudoQuadMesh.from_coarse_polylines(polylines)


def test_t_junction_raises():
    """The common mistake: a wall drawn as one curve with a cut landing halfway."""
    polylines = square_edges() + [[[5, 0, 0], [5, 10, 0]], [[10, 5, 0], [5, 5, 0]], [[0, 5, 0], [5, 5, 0]]]

    with pytest.raises(ValueError, match="interior of polyline"):
        CoarsePseudoQuadMesh.from_coarse_polylines(polylines)


def test_crossing_curves_raise():
    """Both cuts land on proper corners, so only the crossing is wrong."""
    polylines = split_square_edges() + [[[5, 0, 0], [5, 10, 0]], [[0, 5, 0], [10, 5, 0]]]

    with pytest.raises(ValueError, match="cross at"):
        CoarsePseudoQuadMesh.from_coarse_polylines(polylines)


def test_five_sided_patch_raises():
    """A pentagon: perfectly drawable, and it has no densification."""
    polylines = []
    corners = [[cos(2 * pi * k / 5) * 5, sin(2 * pi * k / 5) * 5, 0.0] for k in range(5)]
    for k in range(5):
        polylines.append([corners[k], corners[(k + 1) % 5]])

    with pytest.raises(ValueError, match="5 sides"):
        CoarsePseudoQuadMesh.from_coarse_polylines(polylines)


def test_two_edges_between_the_same_corners_raise():
    """An annulus cut TWICE: a mesh cannot key two half-edges to one corner pair."""
    polylines = [arc(0, 0, 5, 0, 180), arc(0, 0, 5, 180, 360), arc(0, 0, 2, 0, 180), arc(0, 0, 2, 180, 360), [[2, 0, 0], [5, 0, 0]], [[-2, 0, 0], [-5, 0, 0]]]

    with pytest.raises(ValueError, match="both corners"):
        CoarsePseudoQuadMesh.from_coarse_polylines(polylines)


def test_closed_curve_is_not_an_edge():
    polylines = square_edges() + [[[3, 3, 0], [7, 3, 0], [7, 7, 0], [3, 7, 0], [3, 3, 0]]]

    with pytest.raises(ValueError, match="same corner"):
        CoarsePseudoQuadMesh.from_coarse_polylines(polylines)


# ----------------------------------------------------------------------------
# the curvature, and what carries it
# ----------------------------------------------------------------------------


def test_densification_follows_the_drawn_curves():
    """A bare ``densification()`` reproduces the arcs, with nothing handed back."""
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(annulus_polylines(), holes=[[0, 0, 0]])
    coarse.set_strips_density(4)
    dense = coarse.densification()

    on_circle = 0
    for vkey in dense.vertices():
        x, y, _z = dense.vertex_coordinates(vkey)
        r = (x * x + y * y) ** 0.5
        if abs(r - 5.0) < 1e-6 or abs(r - 2.0) < 1e-6:
            on_circle += 1
    # every vertex of both rings, sampled along the arcs rather than their chords
    assert on_circle >= 24


def test_cleared_curves_fall_back_to_chords():
    """The unchanged branch: no stored curves means straight coarse edges."""
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(annulus_polylines(), holes=[[0, 0, 0]])
    coarse.set_edges_to_curves(None)
    coarse.set_strips_density(4)
    dense = coarse.densification()

    assert coarse.edges_to_curves() == {}
    off_circle = [v for v in dense.vertices() if abs((sum(c * c for c in dense.vertex_coordinates(v)[:2])) ** 0.5 - 5.0) > 1e-6]
    assert off_circle  # the outer ring is now chorded


def test_layouts_from_other_constructors_are_untouched():
    """The fallback must be a no-op for every mesh built any other way."""
    vertices = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0], [0, 1, 0]]
    faces = [[0, 1, 4, 5], [1, 2, 3, 4]]
    coarse = CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, faces)

    assert coarse.edges_to_curves() == {}
    coarse.collect_strips()
    coarse.set_strips_density(2)
    assert coarse.densification().number_of_faces() == 8


def test_json_round_trip_keeps_the_curves():
    import json
    import os
    import tempfile

    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(annulus_polylines(), holes=[[0, 0, 0]])
    path = os.path.join(tempfile.mkdtemp(), "coarse.json")
    coarse.save_to_json(path)

    reloaded = CoarsePseudoQuadMesh.from_json(path)

    assert reloaded.number_of_faces() == coarse.number_of_faces()
    assert reloaded.edges_to_curves().keys() == coarse.edges_to_curves().keys()
    with open(path) as f:
        json.load(f)  # it is real JSON, tuple keys and all


def test_a_layout_saved_before_this_change_still_loads():
    """``attributes`` from an older JSON has no ``edges_to_curves`` key at all."""
    vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    coarse = CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, [[0, 1, 2, 3]])
    del coarse.attributes["edges_to_curves"]

    assert coarse.edges_to_curves() == {}
    coarse.collect_strips()
    coarse.set_strips_density(2)
    assert coarse.densification().number_of_faces() == 4


# ----------------------------------------------------------------------------
# the round trip this constructor exists for
# ----------------------------------------------------------------------------


def test_generated_layout_survives_a_bake_and_a_read_back():
    """A layout out as one curve per edge, and back in, is the same layout.

    This is the whole point: ``coarse_edges_to_curves`` is what bakes
    ``Skeleton::EdgeCurves``, and this constructor is its inverse. A plate with a
    square hole, so the read-back also has to reject the hole.
    """
    from compas.geometry import Polyline
    from compas_singular.algorithms import SkeletonDecomposition
    from compas_singular.datastructures import coarse_edges_to_curves

    outer = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 0]]
    inner = [[6, 4, 0], [4, 4, 0], [4, 6, 0], [6, 6, 0], [6, 4, 0]]

    decomposition = SkeletonDecomposition.from_boundary(Polyline(outer), inner_boundaries=[Polyline(inner)], target_length=0.5)
    coarse = decomposition.coarse_mesh()
    curves, _tally = coarse_edges_to_curves(coarse, loops=[outer[:-1], inner[:-1]], polylines=decomposition.polylines)

    back = CoarsePseudoQuadMesh.from_coarse_polylines(list(curves.values()), holes=[[5, 5, 0]])

    assert back.number_of_faces() == coarse.number_of_faces()
    assert back.number_of_vertices() == coarse.number_of_vertices()
    assert len(back.edges_to_curves()) == back.number_of_edges()

    back.set_strips_density_target(t=0.5)
    assert back.densification().is_manifold()


# ----------------------------------------------------------------------------
# reading a CAD polyline as edges
# ----------------------------------------------------------------------------


def test_closed_rectangle_splits_into_its_four_sides():
    rectangle = [[0, 0, 0], [10, 0, 0], [10, 6, 0], [0, 6, 0], [0, 0, 0]]

    pieces = split_at_corners(rectangle)

    assert len(pieces) == 4
    assert all(len(piece) == 2 for piece in pieces)
    # the four pieces tile the loop: each ends where the next begins
    for a, b in zip(pieces, pieces[1:] + pieces[:1]):
        assert distance_point_point(a[-1], b[0]) < 1e-12


def test_seam_halfway_along_a_side_is_not_a_corner():
    """A loop started mid-side: the pieces either side of the seam are rejoined."""
    loop = [[5, 0, 0], [10, 0, 0], [10, 6, 0], [0, 6, 0], [0, 0, 0], [5, 0, 0]]

    pieces = split_at_corners(loop)

    assert len(pieces) == 4
    bottom = [piece for piece in pieces if all(abs(p[1]) < 1e-12 for p in piece)]
    assert len(bottom) == 1 and len(bottom[0]) == 3  # 0 -> 5 -> 10, one edge


def test_collinear_vertex_is_not_a_corner():
    wall = [[0, 0, 0], [4, 0, 0], [10, 0, 0]]

    assert split_at_corners(wall) == [[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [10.0, 0.0, 0.0]]]


def test_open_polyline_splits_at_its_bends():
    zigzag = [[0, 0, 0], [5, 0, 0], [5, 5, 0], [10, 5, 0]]

    assert len(split_at_corners(zigzag)) == 3


def test_the_frame_drawn_in_rhino():
    """The drawing that first hit the command: two closed polylines and 4 lines.

    Outer rectangle and inner quad each drawn as ONE closed polyline, corners
    joined by four diagonals -- coordinates from the screenshot, y flipped. The
    command used a LOOP reader here, which drops a closed curve's closing point:
    the rectangle came through as three sides, and the diagonals landing on its
    remaining corners were reported as T-junctions.
    """
    outer = [[15, -20, 0], [487, -20, 0], [487, -410, 0], [15, -410, 0], [15, -20, 0]]
    inner = [[153, -123, 0], [360, -118, 0], [347, -282, 0], [152, -282, 0], [153, -123, 0]]
    diagonals = [[outer[k], inner[k]] for k in range(4)]

    polylines = split_at_corners(outer) + split_at_corners(inner) + diagonals
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(polylines)

    assert sides(coarse) == {4: 5}
    assert coarse.number_of_vertices() == 8
    coarse.set_strips_density(3)
    assert coarse.densification().is_manifold()


# ----------------------------------------------------------------------------
# splitting a drawing where its curves meet
# ----------------------------------------------------------------------------


def drawn(*curves):
    """What the Rhino command does: polylines at their corners, then junctions."""
    edges = [piece for curve in curves for piece in split_at_corners(curve)]
    return split_at_junctions(edges)


RECTANGLE = [[0, 0, 0], [20, 0, 0], [20, 10, 0], [0, 10, 0], [0, 0, 0]]


def test_rectangle_cut_in_half():
    """The drawing that hit the strict rule: one closed polyline, one line across."""
    pieces, _origin = drawn(RECTANGLE, [[0, 5, 0], [20, 5, 0]])

    assert len(pieces) == 7  # 4 sides, 2 of them split, + the cut
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(pieces)
    assert sides(coarse) == {4: 2}


def test_plus_drawn_as_two_crossing_strokes():
    pieces, _origin = drawn(RECTANGLE, [[0, 5, 0], [20, 5, 0]], [[10, 0, 0], [10, 10, 0]])

    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(pieces)
    assert sides(coarse) == {4: 4}
    assert coarse.number_of_vertices() == 9


def test_hash_drawn_as_four_strokes():
    square = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0], [0, 0, 0]]
    strokes = [[[0, 3, 0], [10, 3, 0]], [[0, 7, 0], [10, 7, 0]], [[3, 0, 0], [3, 10, 0]], [[7, 0, 0], [7, 10, 0]]]
    pieces, _origin = drawn(square, *strokes)

    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(pieces)
    assert sides(coarse) == {4: 9}


def test_annulus_drawn_as_two_full_circles():
    """Closed curves are cut at every landing -- including one AT the seam.

    A circle's seam sits at angle 0, and a cut drawn at 0 degrees lands exactly
    there. That cut has to split the circle like any other.
    """
    outer = arc(0, 0, 5, 0, 360, 48)
    inner = arc(0, 0, 2, 0, 360, 24)
    cuts = [[at(2, a), at(5, a)] for a in (0, 120, 240)]
    pieces, origin = split_at_junctions([outer, inner] + cuts)

    assert origin.count(0) == 3 and origin.count(1) == 3
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(pieces, holes=[[0, 0, 0]])
    assert sides(coarse) == {4: 3}


def test_origin_maps_every_piece_back_to_its_curve():
    curves = [[[0, 5, 0], [20, 5, 0]]] + split_at_corners(RECTANGLE)
    pieces, origin = split_at_junctions(curves)

    assert len(origin) == len(pieces)
    assert origin.count(0) == 1  # the cut itself is not split
    for piece, i in zip(pieces, origin):  # every piece lies along its source
        source = curves[i]
        for point in piece:
            assert min(distance_point_point(point, q) for q in source) <= 20.0


def test_a_curve_stopping_short_is_still_refused():
    """Short of the wall is not ON it: nothing to split at, still dangling."""
    pieces, _origin = drawn(RECTANGLE, [[0, 5, 0], [19.5, 5, 0]])

    with pytest.raises(ValueError, match="dangling"):
        CoarsePseudoQuadMesh.from_coarse_polylines(pieces)


def test_a_circle_with_nothing_landing_on_it_stays_whole():
    circle = arc(0, 0, 5, 0, 360, 48)
    pieces, _origin = split_at_junctions([circle])

    assert len(pieces) == 1
    with pytest.raises(ValueError, match="same corner"):
        CoarsePseudoQuadMesh.from_coarse_polylines(pieces)


# ----------------------------------------------------------------------------
# one connected drawing, and corners marked by points
# ----------------------------------------------------------------------------


def test_a_loose_piece_is_refused():
    """A hole no division line reaches used to be kept INSIDE a patch, silently.

    The square around it came back as one quad covering the hole, and
    ``densifiable`` accepted it. Now it is refused, naming a curve of the loop.
    """
    square = [[[0, 0, 0], [10, 0, 0]], [[10, 0, 0], [10, 10, 0]], [[10, 10, 0], [0, 10, 0]], [[0, 10, 0], [0, 0, 0]]]
    loop = [[[4, 4, 0], [6, 4, 0]], [[6, 4, 0], [6, 6, 0]], [[6, 6, 0], [4, 6, 0]], [[4, 6, 0], [4, 4, 0]]]

    with pytest.raises(ValueError, match=r"2 separate pieces: polyline [4-7]"):
        CoarsePseudoQuadMesh.from_coarse_polylines(square + loop, holes=[[5, 5, 0]])


def test_hole_reached_by_divisions_is_kept_open():
    """The same square hole, now connected to the walls at its four corners."""
    outer = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]
    inner = [[4, 4, 0], [6, 4, 0], [6, 6, 0], [4, 6, 0]]
    walls = [[ring[k], ring[(k + 1) % 4]] for ring in (outer, inner) for k in range(4)]
    divisions = [[outer[k], inner[k]] for k in range(4)]

    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(walls + divisions, holes=[[5, 5, 0]])

    assert sides(coarse) == {4: 4}


def test_extra_points_cut_like_a_landing():
    """A corner on a wall with nothing ending there -- a disc as ONE patch."""
    circle = arc(0, 0, 5, 0, 360, 48)
    corners = [at(5, a) for a in (0, 90, 180, 270)]

    pieces, origin = split_at_junctions([circle], points=corners)

    assert len(pieces) == 4 and origin == [0, 0, 0, 0]
    coarse = CoarsePseudoQuadMesh.from_coarse_polylines(pieces)
    assert sides(coarse) == {4: 1}
