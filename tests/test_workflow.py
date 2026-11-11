"""The shared front-end workflow: ``from_boundary`` -> ``coarse_mesh`` ->
densities -> ``densification``, for both decomposition classes.

The point of these is the LAST step. A coarse layout is a coarse layout whatever
produced it, so a skeleton layout and a cross field solved on the same walls have
to work together -- that is what ``densification(field=...)`` exists for and what
nothing else covers.
"""

import math

import pytest

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.framefield.field import CrossField
from compas_singular.geometry import bounding_box_diagonal
from compas_singular.geometry import discretise_boundary
from compas_singular.geometry import resample_loop


PLATE = [[-6.0, -4.0, 0.0], [6.0, -4.0, 0.0], [6.0, 4.0, 0.0], [-6.0, 4.0, 0.0]]
CABLE = [[[-6.0, -2.0, 0.0], [-1.0, -1.0, 0.0], [3.0, 2.5, 0.0], [6.0, 3.0, 0.0]]]


def circle(centre, radius, n):
    return [[centre[0] + radius * math.cos(2.0 * math.pi * i / n), centre[1] + radius * math.sin(2.0 * math.pi * i / n), 0.0] for i in range(n)]


def max_segment(loop):
    return max(math.dist(a, b) for a, b in zip(loop, loop[1:] + loop[:1]))


def is_all_quad(mesh):
    poles = mesh.attributes.get("face_pole") or {}
    return all(len(mesh.face_vertices(f)) == 4 or f in poles for f in mesh.faces())


# ----------------------------------------------------------------------------
# discretise_boundary -- thesis eq. 4.1
# ----------------------------------------------------------------------------

L_SHAPE = [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 2.0, 0.0], [2.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 4.0, 0.0]]


def test_discretise_boundary_keeps_corners_and_is_idempotent():
    dense, inners = discretise_boundary(PLATE, spacing=0.5)
    assert len(dense) == 80  # perimeter 40, at 0.5
    assert inners == []
    for corner in PLATE:  # every input point survives
        assert corner in dense
    assert discretise_boundary(dense, spacing=0.5)[0] == dense
    # a closing point is dropped rather than leaving a zero-length segment
    assert discretise_boundary(PLATE + PLATE[:1])[0] == discretise_boundary(PLATE)[0]


@pytest.mark.parametrize("spacing", [0.5, 1.7, 3.0, 100.0])
def test_discretise_boundary_keeps_a_reflex_corner_at_any_spacing(spacing):
    """Each SEGMENT is subdivided on its own, so no corner is ever rounded off --
    including at a spacing 25x coarser than the shape."""
    dense, _ = discretise_boundary(L_SHAPE, spacing=spacing)
    for corner in L_SHAPE:
        assert any(math.dist(corner, p) < 1e-12 for p in dense)


@pytest.mark.parametrize("spacing", [0.5, 0.7, 0.9, 1.3])
def test_discretise_boundary_target_is_a_bound_not_an_average(spacing):
    """``ceil``, not ``round``: rounding to nearest left segments up to 1.5x the
    target -- measured 1.12x on this disc at 0.5."""
    for loop in (PLATE, L_SHAPE, circle([0, 0, 0], 5.0, 28)):
        dense, _ = discretise_boundary(loop, spacing=spacing)
        assert max_segment(dense) <= spacing + 1e-12


def test_discretise_boundary_d_min_is_a_floor_only():
    tiny = circle([0, 0, 0], 0.1, 4)
    assert len(discretise_boundary(tiny, spacing=10.0, d_min=5)[0]) >= 5
    assert len(discretise_boundary(tiny, spacing=10.0, d_min=10)[0]) >= 10
    # it never coarsens a loop that is already fine enough
    assert len(discretise_boundary(PLATE, spacing=0.5, d_min=10)[0]) == 80


def test_discretise_boundary_alpha_is_the_thesis_scale_rule():
    D = bounding_box_diagonal(PLATE)
    assert D == pytest.approx(math.hypot(12.0, 8.0))
    dense, _ = discretise_boundary(PLATE, alpha=0.04)
    assert max_segment(dense) <= 0.04 * D + 1e-12
    # d_i = ceil(l_i / (alpha D)) per SEGMENT, summed
    assert len(dense) == sum(math.ceil(math.dist(a, b) / (0.04 * D)) for a, b in zip(PLATE, PLATE[1:] + PLATE[:1]))
    # a smaller alpha is a finer boundary
    assert len(discretise_boundary(PLATE, alpha=0.01)[0]) > len(dense)


def test_discretise_boundary_target_length_overrides_alpha():
    assert discretise_boundary(PLATE, spacing=0.5, alpha=0.01)[0] == discretise_boundary(PLATE, spacing=0.5)[0]


def test_discretise_boundary_alpha_none_is_the_opt_out():
    raw, _ = discretise_boundary(PLATE, spacing=None, alpha=None)
    assert raw == [[float(c) for c in p] for p in PLATE]


def test_discretise_boundary_inners_share_the_total_scale():
    """A hole measured against its OWN bounding box would come back far denser
    than the wall beside it."""
    hole = circle([0.0, 0.0, 0.0], 0.5, 12)
    _, inners = discretise_boundary(PLATE, [hole], alpha=0.04)
    alone, _ = discretise_boundary(hole, alpha=0.04)
    assert len(inners[0]) < len(alone)
    D = bounding_box_diagonal(PLATE, hole)
    assert len(inners[0]) == sum(max(1, math.ceil(math.dist(a, b) / (0.04 * D))) for a, b in zip(hole, hole[1:] + hole[:1]))


def test_resample_loop_alias_still_works():
    """Kept for callers outside this repository."""
    dense = resample_loop(PLATE, 0.5)
    assert len(dense) == 80
    assert resample_loop(dense, 0.5) == dense
    assert resample_loop(PLATE) == [[float(c) for c in p] for p in PLATE]
    assert resample_loop(PLATE + PLATE[:1]) == resample_loop(PLATE)


# ----------------------------------------------------------------------------
# SkeletonDecomposition
# ----------------------------------------------------------------------------


def test_from_boundary_resamples_for_the_triangulation():
    """target_length is the BACKGROUND spacing, and it used to be ignored."""
    default = SkeletonDecomposition.from_boundary(PLATE)
    fine = SkeletonDecomposition.from_boundary(PLATE, target_length=0.5)
    # no target_length is the THESIS default now, not the two-triangle
    # caricature -- the same alpha*D the field route uses
    assert default.number_of_faces() > 50
    assert default.inputs["target_length"] == pytest.approx(0.02 * bounding_box_diagonal(PLATE))
    # the caricature is still reachable, but only by asking for it
    caricature = SkeletonDecomposition.from_boundary(PLATE, alpha=None)
    assert caricature.number_of_faces() == 2
    assert fine.number_of_faces() > 50
    # ...but what is STORED is what the caller handed in, so edges_to_curves
    # samples the walls rather than this function's approximation of them
    assert fine.inputs["outer_boundary"] == PLATE


def test_coarse_mesh_is_idempotent_and_takes_its_poles_from_the_domain():
    d = SkeletonDecomposition.from_boundary(PLATE, point_features=[[-3.5, 0.0, 0.0]], target_length=0.5)
    coarse = d.coarse_mesh()
    assert coarse is d.coarse_mesh()  # same object, not an equal one
    assert d.coarse_mesh(force=True) is not coarse
    # the point feature became a pole without being passed a second time
    assert len(list(d.coarse_mesh().poles())) == 1


def test_workflow_skeleton():
    d = SkeletonDecomposition.from_boundary(PLATE, target_length=0.5)
    coarse = d.coarse_mesh()
    coarse.set_strips_density_target(t=0.5)
    dense = coarse.densification()
    assert dense is coarse.get_quad_mesh()
    assert dense.number_of_faces() > 0
    assert is_all_quad(dense)


def test_edges_to_curves_keeps_a_hole_round():
    hole = circle([2.0, 0.0], 1.6, 80)
    d = SkeletonDecomposition.from_boundary(PLATE, inner_boundaries=[hole], target_length=0.5)
    coarse = d.coarse_mesh()
    coarse.set_strips_density_target(t=0.5)

    curves, tally = d.edges_to_curves()
    assert tally["chord"] == 0
    assert tally["wall_missed"] == 0

    def worst_hole_error(mesh):
        errors = [abs(math.dist(mesh.vertex_coordinates(v), [2.0, 0.0, 0.0]) - 1.6) for ring in mesh.vertices_on_boundaries() for v in ring]
        return max(e for e in errors if e < 1.0)  # the hole ring only

    with_curves = worst_hole_error(coarse.densification(overwrite_edges_to_curves=curves))

    # ``coarse_mesh()`` now stores curves on the layout itself, so a bare
    # ``densification()`` call picks them up by default -- asking for none
    # explicitly is what reproduces the old "no curves" baseline.
    bare = d.coarse_mesh(force=True)
    bare.set_strips_density_target(t=0.5)
    without = worst_hole_error(bare.densification(boundary_curvature=False, skeleton_curvature=False))

    assert with_curves < 0.01 * 1.6  # under 1% of the radius
    assert without > 0.4 * 1.6  # a polygon, not a circle


def test_edges_to_curves_refuses_without_a_domain():
    from compas_singular.algorithms import boundary_triangulation

    trimesh = boundary_triangulation(resample_loop(PLATE, 0.5), [], [], [])
    d = SkeletonDecomposition.from_mesh(trimesh)
    d.coarse_mesh()
    with pytest.raises(ValueError):
        d.edges_to_curves()


# ----------------------------------------------------------------------------
# CrossField on its own, and across the routes
# ----------------------------------------------------------------------------


def test_cross_field_stands_alone():
    field = CrossField.from_boundary(PLATE, guides=CABLE, mode="tangent", target_length=0.8)
    assert field.report()["ok"]
    assert len(field.guides) == 1
    assert field.locator() is field.locator()  # built once

    import pickle

    field.locator().locate([0.0, 0.0, 0.0])  # forces the bucket grid
    revived = pickle.loads(pickle.dumps(field))
    assert revived._locator is None  # grid excluded from the pickle
    assert revived.locator().locate([0.0, 0.0, 0.0]) is not None


def test_skeleton_layout_densified_with_a_field():
    """The capability the whole workflow exists for: a layout from one route,
    a field from the other, on the same walls."""
    d = SkeletonDecomposition.from_boundary(PLATE, target_length=0.5)
    field = CrossField.from_boundary(PLATE, guides=CABLE, mode="tangent", target_length=0.5)

    coarse = d.coarse_mesh()
    coarse.set_strips_density_target(t=0.5)
    coons = coarse.densification()
    coons_points = [coons.vertex_coordinates(v) for v in sorted(coons.vertices())]

    fielded = d.coarse_mesh(force=True)
    fielded.set_strips_density_target(t=0.5)
    mesh = fielded.densification(field=field)

    assert mesh is fielded.get_quad_mesh()
    assert is_all_quad(mesh)
    assert mesh.number_of_faces() == coons.number_of_faces()
    assert len(list(mesh.vertices_on_boundaries())) == 1  # welded, one ring
    # same topology, different interiors -- the field moved them
    points = [mesh.vertex_coordinates(v) for v in sorted(mesh.vertices())]
    assert points != coons_points


def test_field_decomposition_answers_the_same_names():
    from compas_singular.framefield.field_decomposition import FieldDecomposition

    d = FieldDecomposition.from_boundary(PLATE, target_length=0.8)
    coarse = d.coarse_mesh()
    assert coarse is d.decomposition_mesh()
    assert d.get_field() is d.field
    coarse.set_strips_density_target(t=0.8)
    dense = coarse.densification(field=d.get_field())
    assert is_all_quad(dense)


# ----------------------------------------------------------------------------
# JSON round trip -- the state a CAD bake cannot carry
# ----------------------------------------------------------------------------


def test_json_round_trip_carries_the_attributes(tmp_path):
    d = SkeletonDecomposition.from_boundary(PLATE, point_features=[[-3.5, 0.0, 0.0]], target_length=0.5)
    coarse = d.coarse_mesh()
    coarse.set_strips_density_target(t=0.5)
    skey = sorted(coarse.strips())[0]
    coarse.set_strip_density(skey, 9)
    coarse.attributes["route"] = "skeleton"  # arbitrary extra state

    # save_to_json creates the folder; to_json raises FileNotFoundError
    path = coarse.save_to_json(str(tmp_path / "nested" / "coarse.json"))

    back = CoarsePseudoQuadMesh.load_from_json(path)
    assert type(back) is CoarsePseudoQuadMesh  # dtype, not cls
    assert back.attributes["route"] == "skeleton"
    assert len(back.attributes["face_pole"]) == len(coarse.attributes["face_pole"])
    # JSON keys are strings; load_from_json turns the digit ones back into ints,
    # or get_strip_density(0) would raise and sorted() would order '10' before '2'
    assert all(isinstance(k, int) for k in back.attributes["strips"])
    assert back.get_strip_density(skey) == 9
    assert back.densification().number_of_faces() == coarse.densification().number_of_faces()


def test_load_from_json_defaults_and_refuses_the_wrong_class(tmp_path):
    from compas_singular.datastructures import QuadMesh

    assert CoarsePseudoQuadMesh.load_from_json(str(tmp_path / "absent.json"), default="none") == "none"

    plain = QuadMesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]])
    path = plain.save_to_json(str(tmp_path / "plain.json"))
    assert type(QuadMesh.load_from_json(path)) is QuadMesh
    with pytest.raises(TypeError):  # Data.from_json's own check
        CoarsePseudoQuadMesh.load_from_json(path)


# ----------------------------------------------------------------------------
# the field's own JSON -- the other half of the pair above
# ----------------------------------------------------------------------------
#
# A coarse layout has always been serialisable. Until the field was too, half of
# a field-driven job could be saved and the expensive half had to be re-solved
# on every open. These pin the two properties that make the pair usable: the
# field comes back EXACTLY, and it knows whether it still belongs to the domain.


def test_field_json_round_trip_is_exact(tmp_path):
    field = CrossField.from_boundary(PLATE, guides=CABLE, target_length=0.6, guide_weight=5.0)
    path = field.save_to_json(str(tmp_path / "nested" / "field.json"))

    back = CrossField.load_from_json(path)

    # Exactly, not nearly. theta is rebuilt from u by __init__, so any loss in u
    # shows up here -- and 21_edit_coarse asserts a field is unchanged to 0.0.
    assert set(back.theta) == set(field.theta)
    assert max(abs(field.theta[k] - back.theta[k]) for k in field.theta) == 0.0
    assert back.u == field.u

    # JSON object keys are strings. A vertex key left as '17' looks fine until
    # something indexes the background mesh with it.
    assert all(isinstance(k, int) for k in back.u)
    assert all(isinstance(k, int) for k in back.singularity_points)

    # snap_singularities overwrites _singularities and it is not recoverable
    # from u afterwards, so it has to travel.
    assert back.singularities() == field.singularities()
    assert back.background.mesh.number_of_vertices() == field.background.mesh.number_of_vertices()
    assert back.background.target_length == field.background.target_length
    assert back.relaxed == field.relaxed
    assert back.guides == [[list(p) for p in curve] for curve in field.guides]


def test_field_json_carries_the_symmetry_group(tmp_path):
    field = CrossField.from_boundary(PLATE, target_length=0.6)
    path = field.save_to_json(str(tmp_path / "field.json"))
    back = CrossField.load_from_json(path)

    if field.symmetry is None:
        assert back.symmetry is None
    else:
        assert back.symmetry is not None
        assert sorted(back.symmetry.names()) == sorted(field.symmetry.names())
        assert back.symmetry.steps == field.symmetry.steps
        assert back.symmetry.centre == field.symmetry.centre
        assert back.symmetry.trivial == field.symmetry.trivial


def test_a_loaded_field_densifies_identically(tmp_path):
    """The reason any of this exists: save both halves, reopen, densify."""
    field = CrossField.from_boundary(PLATE, guides=CABLE, target_length=0.6, guide_weight=5.0)
    d = SkeletonDecomposition.from_boundary(PLATE, target_length=0.6)

    def dense_from(f, coarse):
        coarse.collect_strips()
        coarse.set_strips_density_target(t=1.0)
        return coarse.densification(field=f)

    live = dense_from(field, d.coarse_mesh())

    field_path = field.save_to_json(str(tmp_path / "field.json"))
    coarse_path = d.coarse_mesh().save_to_json(str(tmp_path / "coarse.json"))
    reopened = dense_from(CrossField.load_from_json(field_path), CoarsePseudoQuadMesh.load_from_json(coarse_path))

    def xyz(mesh):
        return sorted(tuple(round(c, 12) for c in mesh.vertex_coordinates(v)) for v in mesh.vertices())

    assert reopened.number_of_faces() == live.number_of_faces()
    assert xyz(reopened) == xyz(live)


def test_field_knows_when_the_domain_moved(tmp_path):
    kwargs = dict(target_length=0.6, guide_weight=5.0)
    field = CrossField.from_boundary(PLATE, guides=CABLE, **kwargs)
    back = CrossField.load_from_json(field.save_to_json(str(tmp_path / "field.json")))

    # The inputs it was solved from, through a save and a load.
    assert back.mismatch(PLATE, guides=CABLE, **kwargs) is None
    assert back.matches(PLATE, guides=CABLE, **kwargs)

    # A closed loop and the same loop given open are the SAME domain: an
    # invalidation that fires on nothing is as useless as one that never fires.
    assert back.matches(PLATE + [PLATE[0]], guides=CABLE, **kwargs)

    moved = [list(p) for p in PLATE]
    moved[2][0] += 0.001
    hole = [circle([0.0, 0.0], 1.0, 12)]

    assert "outer boundary" in back.mismatch(moved, guides=CABLE, **kwargs)
    assert "holes" in back.mismatch(PLATE, hole, guides=CABLE, **kwargs)
    assert "guide" in back.mismatch(PLATE, **kwargs)
    assert "target_length" in back.mismatch(PLATE, guides=CABLE, target_length=0.5, guide_weight=5.0)
    assert "guide_weight" in back.mismatch(PLATE, guides=CABLE, target_length=0.6)

    # A guide is not a hole. The cache key made exactly this mistake once, by
    # hashing outer + holes + guides into one flat list.
    only_hole = CrossField.from_boundary(PLATE, hole, target_length=0.6)
    assert only_hole.mismatch(PLATE, guides=hole, target_length=0.6) is not None


def test_field_without_provenance_reports_that_it_cannot_tell():
    """A field built through solve() never sees the boundaries."""
    from compas_singular.framefield.background import BackgroundMesh
    from compas_singular.framefield.constraints import from_boundary as constrain

    background = BackgroundMesh.from_boundary(PLATE, target_length=1.0)
    field = CrossField.solve(background, constrain(background))

    assert field.inputs is None
    assert not field.matches(PLATE, target_length=1.0)
    assert "no record of its inputs" in field.mismatch(PLATE, target_length=1.0)


def test_load_from_json_defaults_and_refuses_the_wrong_file(tmp_path):
    assert CrossField.load_from_json(str(tmp_path / "absent.json")) is None
    assert CrossField.load_from_json(str(tmp_path / "absent.json"), default="none") == "none"

    # A coarse mesh's JSON is valid JSON and a valid compas object. Pointing
    # load_from_json at it must say so, not raise a KeyError about a vertex.
    coarse = SkeletonDecomposition.from_boundary(PLATE, target_length=1.0).coarse_mesh()
    wrong = coarse.save_to_json(str(tmp_path / "coarse.json"))
    with pytest.raises(ValueError):
        CrossField.load_from_json(wrong)


def test_load_from_json_refuses_a_newer_version(tmp_path):
    import compas

    from compas_singular.framefield import field as field_module

    field = CrossField.from_boundary(PLATE, target_length=1.0)
    payload = field.__jsondata__()
    payload["version"] = field_module.JSON_VERSION + 1
    path = str(tmp_path / "future.json")
    compas.json_dump(payload, path)

    with pytest.raises(ValueError):
        CrossField.load_from_json(path)
