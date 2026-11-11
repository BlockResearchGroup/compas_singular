"""The automatic guide selection: a guide is matched to one run of one polyedge.

Everything asserted here was measured first (``examples/dev_examples_tests/guide_chain_tests``);
these are the properties that must not regress, not a wish list.
"""

from math import cos
from math import pi
from math import sin

import pytest
from compas.geometry import Circle
from compas.geometry import Frame
from compas.geometry import Line
from compas.geometry import Point
from compas.geometry import distance_point_point

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.algorithms import boundary_triangulation
from compas_singular.datastructures import QuadMesh
from compas_singular.editing import GuideCurve
from compas_singular.editing import attach_chain
from compas_singular.editing import chain_quality
from compas_singular.editing import collect_polyedges
from compas_singular.editing import guide_chain
from compas_singular.editing import mean_edge_length

# ==============================================================================
# meshes
# ==============================================================================

SQUARE = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]
HOLE = [[4, 4, 0], [6, 4, 0], [6, 6, 0], [4, 6, 0]]
L_PLATE = [[0, 0, 0], [10, 0, 0], [10, 4, 0], [4, 4, 0], [4, 10, 0], [0, 10, 0]]


def _grid(n=6):
    """A plain n x n grid: the one mesh whose right answer can be written down."""
    vertices = [[i, j, 0] for j in range(n + 1) for i in range(n + 1)]
    faces = [[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i] for j in range(n) for i in range(n)]
    return QuadMesh.from_vertices_and_faces(vertices, faces)


def _densify(points, n=5):
    """A closed polygon as a densified point loop, corners kept."""
    out = []
    loop = list(points) + [points[0]]
    for a, b in zip(loop[:-1], loop[1:]):
        out.extend([list(point)[:3] for point in Line(a, b).to_polyline(n=n).points[:-1]])
    return out


def _build(outer, holes=(), density=4):
    """Outer boundary (+ holes) -> skeleton decomposition -> dense quad mesh, as the Rhino commands do."""
    trimesh = boundary_triangulation(outer_boundary=_densify(outer), inner_boundaries=[_densify(hole) for hole in holes], point_features=[])
    coarse = SkeletonDecomposition.from_mesh(trimesh).decomposition_mesh([])
    coarse.collect_strips()
    coarse.set_strips_density(d=density)
    coarse.densification()
    return coarse.get_quad_mesh()


def _arc(centre, radius, start, end, count=24):
    """A circular arc as points. ``start``/``end`` in degrees."""
    return [[centre[0] + radius * cos(a), centre[1] + radius * sin(a), 0.0] for a in ((start + (end - start) * i / float(count)) * pi / 180.0 for i in range(count + 1))]


@pytest.fixture(scope="module")
def cases():
    """[(mesh name, mesh, [(guide name, points)])] -- every guide with a known right answer.

    ``axis`` runs the way a course does; ``short`` must be TRIMMED; ``diagonal`` is 45 degrees
    to both families; ``arc`` is curved; ``hug`` is one row in from the wall; ``wall`` is ON
    the wall; ``ring`` wraps around the hole; ``through`` crosses it; ``elbow`` has a corner.
    """
    return [
        (
            "square",
            _build(SQUARE),
            [
                ("axis", [[0.2, 5.0, 0.0], [9.8, 5.0, 0.0]]),
                ("short", [[3.0, 5.0, 0.0], [7.0, 5.0, 0.0]]),
                ("diagonal", [[0.5, 0.5, 0.0], [9.5, 9.5, 0.0]]),
                ("arc", _arc([5.0, -3.0], 8.0, 60.0, 120.0)),
                ("hug", [[0.5, 1.1, 0.0], [9.5, 1.1, 0.0]]),
                ("wall", [[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]]),
            ],
        ),
        (
            "square+hole",
            _build(SQUARE, [HOLE]),
            [
                ("axis", [[0.2, 2.0, 0.0], [9.8, 2.0, 0.0]]),
                ("through", [[0.2, 5.0, 0.0], [9.8, 5.0, 0.0]]),
                ("ring", _arc([5.0, 5.0], 3.5, 0.0, 360.0, count=48)),
                ("hug", [[0.5, 0.7, 0.0], [9.5, 0.7, 0.0]]),
            ],
        ),
        (
            "L-plate",
            _build(L_PLATE),
            [
                ("axis", [[0.2, 2.0, 0.0], [9.8, 2.0, 0.0]]),
                ("elbow", [[0.5, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 9.5, 0.0]]),
                ("diagonal", [[0.5, 0.5, 0.0], [3.5, 3.5, 0.0]]),
                ("hug", [[0.5, 0.6, 0.0], [9.5, 0.6, 0.0]]),
            ],
        ),
    ]


def _sweep(cases, tolerance_factor=None):
    """Every mesh, every guide, both boundary modes -> the quality of each chain."""
    out = []
    for _, mesh, guides in cases:
        polyedges = collect_polyedges(mesh)
        tolerance = None if tolerance_factor is None else tolerance_factor * mean_edge_length(mesh)
        for _, points in guides:
            guide = GuideCurve(points)
            for mode in ("anchor", "exclude"):
                chain, _ = guide_chain(mesh, guide, tolerance=tolerance, boundary=mode, polyedges=polyedges)
                if chain:
                    out.append(chain_quality(mesh, chain, guide))
    return out


# ==============================================================================
# on a grid, where the answer is known by hand
# ==============================================================================


def test_on_a_grid_the_answer_is_known_by_hand():
    mesh = _grid()
    polyedges = collect_polyedges(mesh)

    along = GuideCurve([[0, 3, 0], [6, 3, 0]])
    chain, info = guide_chain(mesh, along, polyedges=polyedges)
    assert chain == [21, 22, 23, 24, 25, 26, 27], "a guide down a row selects that whole row"
    assert info["coverage"] == 1.0
    quality = chain_quality(mesh, chain, along)
    assert quality["purity"] == 1.0 and quality["runs"] == 1, "as one straight run"
    assert quality["off_max"] == 0.0
    assert quality["min_face_angle"] == 90.0, "projecting it changes nothing"

    across = GuideCurve([[3, 0, 0], [3, 6, 0]])
    chain, _ = guide_chain(mesh, across, polyedges=polyedges)
    assert chain == [3, 10, 17, 24, 31, 38, 45], "the other family selects the column, not the row"
    assert chain_quality(mesh, chain, across)["alignment"] > 0.99

    chain, _ = guide_chain(mesh, GuideCurve([[1.4, 3.0, 0], [4.6, 3.0, 0]]), polyedges=polyedges)
    assert chain == [22, 23, 24, 25, 26], "a short guide selects a TRIMMED run, not the row"

    chain, info = guide_chain(mesh, GuideCurve([[0, 50, 0], [6, 50, 0]]), polyedges=polyedges)
    assert chain == [] and info["reason"], "a guide nowhere near the mesh returns a reason"

    diagonal = GuideCurve([[0, 0, 0], [6, 6, 0]])
    chain, _ = guide_chain(mesh, diagonal, polyedges=polyedges)
    assert not chain or chain_quality(mesh, chain, diagonal)["alignment"] < 0.8, "a 45 degree guide is reported as poorly aligned, not forced into a staircase"


# ==============================================================================
# the boundary rule
# ==============================================================================


def test_the_boundary_rule():
    mesh = _grid()
    polyedges = collect_polyedges(mesh)
    average = mean_edge_length(mesh)
    hugging = GuideCurve([[0.2, 0.4, 0], [5.8, 0.4, 0]])  # nearer the wall than the row

    chain, _ = guide_chain(mesh, hugging, tolerance=average, boundary="free", polyedges=polyedges)
    assert chain_quality(mesh, chain, hugging)["boundary_interior"] > 0, "without the rule the outline itself is selected ('free' is the old behaviour)"

    chain, _ = guide_chain(mesh, hugging, tolerance=average, boundary="anchor", polyedges=polyedges)
    quality = chain_quality(mesh, chain, hugging)
    assert quality["boundary_interior"] == 0, "no boundary vertex in the middle"
    assert quality["boundary_ends"] == 2, "but the two ends may anchor on the wall"

    chain, _ = guide_chain(mesh, hugging, tolerance=average, boundary="exclude", polyedges=polyedges)
    quality = chain_quality(mesh, chain, hugging)
    assert quality["boundary_interior"] == 0 and quality["boundary_ends"] == 0, "'exclude' drops the anchors too"


def test_a_selected_boundary_vertex_slides_it_does_not_move(cases):
    mesh = _grid()
    hugging = GuideCurve([[0.2, 0.4, 0], [5.8, 0.4, 0]])
    chain, _ = guide_chain(mesh, hugging, tolerance=mean_edge_length(mesh), polyedges=collect_polyedges(mesh))
    boundary = [vertex for vertex in chain if mesh.is_vertex_on_boundary(vertex)]
    assert len(boundary) == 2

    moves, constraints = attach_chain(mesh, chain, hugging)
    assert all(vertex not in moves for vertex in boundary), "no boundary vertex is given a new position"
    assert all(vertex in moves for vertex in chain if vertex not in boundary), "every interior vertex is"
    assert all(not isinstance(constraints[vertex], Point) for vertex in boundary), "a boundary vertex is constrained to a curve -- it slides, it is not pinned"
    assert all(isinstance(constraints[vertex], Point) for vertex in chain if vertex not in boundary), "an interior one is pinned under hold='fixed'"
    _, sliding = attach_chain(mesh, chain, hugging, hold="sliding")
    assert all(not isinstance(sliding[vertex], Point) for vertex in chain), "under hold='sliding' nothing is pinned"

    drift = 0.0
    for _, case_mesh, guides in cases:
        polyedges = collect_polyedges(case_mesh)
        for _, points in guides:
            guide = GuideCurve(points)
            chain, _ = guide_chain(case_mesh, guide, polyedges=polyedges)
            if chain:
                drift = max(drift, chain_quality(case_mesh, chain, guide)["end_drift"])
    assert drift == 0.0, "over every mesh and guide, no attached vertex ends up off its outline"


# ==============================================================================
# every guide on every mesh
# ==============================================================================


def test_every_chain_on_every_mesh_is_one_clean_run(cases):
    sweep = _sweep(cases)
    assert len(sweep) >= 20, "the sweep actually ran"
    assert all(q["boundary_interior"] == 0 for q in sweep), "no boundary vertex anywhere but at an end"
    assert all(q["valid_path"] for q in sweep), "every chain is a connected path of real edges"
    assert all(q["purity"] == 1.0 and q["runs"] == 1 for q in sweep), "exactly one run of one polyedge"


def test_folding_is_the_price_of_reach(cases):
    """A wide tolerance folds faces by pulling far vertices onto the guide -- not a selection fault.

    Casper set the default at 2.0 knowing this.
    """
    assert all(q["folded_faces"] == 0 for q in _sweep(cases, 0.8)), "at 0.8 edge lengths nothing folds"
    folded = [q for q in _sweep(cases, 2.0) if q["folded_faces"]]
    assert folded, "at 2.0 some chains do"
    assert all(q["off_max"] > 1.0 for q in folded), "every one of them was reaching more than an edge length"
    assert all(q["purity"] == 1.0 and q["valid_path"] for q in folded), "the chains themselves are still clean"


def test_the_angle_gate_drops_what_the_distance_gate_keeps(cases):
    mesh = _grid()
    polyedges = collect_polyedges(mesh)
    diagonal = GuideCurve([[0, 0, 0], [6, 6, 0]])
    chain, info = guide_chain(mesh, diagonal, tolerance=3.0, max_angle=30.0, polyedges=polyedges)
    assert not chain and "degrees" in info["reason"], "a 45 degree guide is refused by angle"
    chain, _ = guide_chain(mesh, diagonal, tolerance=3.0, max_angle=90.0, polyedges=polyedges)
    assert chain, "and selected again with the gate off"
    chain, _ = guide_chain(mesh, GuideCurve([[3, 0, 0], [3, 6, 0]]), polyedges=polyedges)
    assert chain == [3, 10, 17, 24, 31, 38, 45], "a perpendicular course is still selected for a guide along it"

    # on the L-plate the elbow guide keeps a vertex whose polyedge has already
    # turned away, and that vertex is the one that squashes a face
    _, plate, guides = [case for case in cases if case[0] == "L-plate"][0]
    polyedges = collect_polyedges(plate)
    elbow = GuideCurve(dict(guides)["elbow"])
    average = mean_edge_length(plate)
    loose, _ = guide_chain(plate, elbow, tolerance=average, max_angle=90.0, polyedges=polyedges)
    gated, _ = guide_chain(plate, elbow, tolerance=average, max_angle=30.0, polyedges=polyedges)
    assert len(gated) < len(loose), "the gate shortens a chain that veers off at the end"
    assert chain_quality(plate, gated, elbow)["min_face_angle"] > chain_quality(plate, loose, elbow)["min_face_angle"]


def test_a_guide_shorter_than_its_polyedge_collapses_no_edge():
    # everything past the end of a guide projects onto the SAME endpoint, so two such
    # vertices would land on one point and the edge between them would vanish
    mesh = _grid()
    guide = GuideCurve([[1.5, 3.0, 0], [4.5, 3.0, 0]])
    chain, _ = guide_chain(mesh, guide, tolerance=10.0, polyedges=collect_polyedges(mesh))
    projected = [tuple(round(value, 6) for value in guide.closest_point(mesh.vertex_coordinates(vertex))) for vertex in chain]
    assert len(set(projected)) == len(projected)
    assert chain_quality(mesh, chain, guide)["min_face_angle"] > 0.0


def test_a_guide_that_carries_its_curve_lands_vertices_on_it(cases):
    _, mesh, guides = [case for case in cases if case[0] == "square+hole"][0]
    points = dict(guides)["ring"]
    circle = Circle(3.5, Frame([5.0, 5.0, 0.0], [1, 0, 0], [0, 1, 0]))
    sampled, exact = GuideCurve(points), GuideCurve(points, curve=circle)
    polyedges = collect_polyedges(mesh)

    chain, info = guide_chain(mesh, sampled, polyedges=polyedges)
    chain_exact, info_exact = guide_chain(mesh, exact, polyedges=polyedges)
    assert chain and chain == chain_exact and info == info_exact, "the chain is chosen exactly as before"

    def off(moves):
        return max(abs(distance_point_point(xyz, [5.0, 5.0, 0.0]) - 3.5) for xyz in moves.values())

    moves, _ = attach_chain(mesh, chain, sampled)
    moves_exact, constraints = attach_chain(mesh, chain, exact, hold="sliding")
    assert off(moves) > 1e-3, "on the 48 samples a vertex lands a chord off the circle"
    assert off(moves_exact) < 1e-12, "with the curve every vertex lands on it"
    assert all(constraints[vertex] is exact for vertex in moves_exact), "a sliding hold re-projects through the same GuideCurve"
