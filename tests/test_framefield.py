"""The frame-field front end: point location, sampling, storage and routing.

The fast paths here (batched point location, batched field sampling, batched
wall tests, the vectorised patch solve) are required to give EXACTLY what the
one-at-a-time definitions give, so these compare with ``==``, not a tolerance.
"""

import json
import math
import os
import random
import warnings

import pytest

from compas.geometry import is_point_in_polygon_xy

from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.framefield.constraints import from_curves
from compas_singular.framefield.densify import FieldSampler
from compas_singular.framefield.field import CrossField
from compas_singular.framefield.field import FieldInputs
from compas_singular.framefield.field_decomposition import FieldDecomposition
from compas_singular.framefield.locator import AMBIGUOUS
from compas_singular.framefield.relax import _Curve
from compas_singular.geometry.polyline import distance_to_loop
from compas_singular.geometry.polyline import near_loop
from compas_singular.geometry.polyline import points_in_polygon_xy


HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

PLATE = [[-6.0, -4.0, 0.0], [6.0, -4.0, 0.0], [6.0, 4.0, 0.0], [-6.0, 4.0, 0.0]]
CABLE = [[[-6.0, -2.0, 0.0], [-1.0, -1.0, 0.0], [3.0, 2.5, 0.0], [6.0, 3.0, 0.0]]]
DISC = [[5 * math.cos(2 * math.pi * i / 48), 5 * math.sin(2 * math.pi * i / 48), 0.0] for i in range(48)]


@pytest.fixture(scope="module")
def field():
    return CrossField.from_boundary(DISC, target_length=0.5)


def _probe_points(field, count=3000, seed=11):
    """Random points, plus points exactly on background vertices and edges --
    where more than one triangle contains the point."""
    rng = random.Random(seed)
    mesh = field.background.mesh
    points = [[rng.uniform(-5.5, 5.5), rng.uniform(-5.5, 5.5), 0.0] for _ in range(count)]
    for vkey in list(mesh.vertices())[:200]:
        points.append(list(mesh.vertex_coordinates(vkey)))
    for u, v in list(mesh.edges())[:200]:
        a, b = mesh.vertex_coordinates(u), mesh.vertex_coordinates(v)
        points.append([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, 0.0])
    return points


# ----------------------------------------------------------------------------
# point location
# ----------------------------------------------------------------------------


def test_locate_many_is_locate(field):
    locator = field.locator()
    points = _probe_points(field)
    ambiguous = 0
    for point, found in zip(points, locator.locate_many(points)):
        if found is AMBIGUOUS:
            ambiguous += 1
            continue
        assert found == locator.locate(point)
        # the answer does not depend on the hint
        if found is not None:
            assert locator.locate(point, hint=found[0]) == found
    assert ambiguous  # the on-vertex and on-edge probes must exercise the slow path


def test_the_locator_is_shared_and_never_saved(field):
    assert field.locator() is field.locator()
    import pickle

    revived = pickle.loads(pickle.dumps(field))
    assert revived._locator is None
    assert "_locator" not in json.dumps({k: str(v) for k, v in field.__jsondata__().items()})


def test_batched_sampling_is_the_hinted_loop(field):
    sampler = FieldSampler(field)
    points = _probe_points(field, count=1500, seed=3)
    thetas, ok = sampler.thetas(points)
    hint = None
    for point, theta, good in zip(points, thetas, ok):
        expected, hint = sampler.theta(point, hint)
        if expected is None:
            assert not good
        else:
            assert good and theta == expected


# ----------------------------------------------------------------------------
# batched geometry
# ----------------------------------------------------------------------------


def test_batched_wall_tests_match_the_scalar_ones():
    rng = random.Random(5)
    for n in (4, 7, 48):
        loop = [[3 * math.cos(2 * math.pi * i / n) * rng.uniform(0.7, 1.0), 3 * math.sin(2 * math.pi * i / n) * rng.uniform(0.7, 1.0), 0.0] for i in range(n)]
        points = [[rng.uniform(-4, 4), rng.uniform(-4, 4), 0.0] for _ in range(2000)]
        points += [list(p) for p in loop]
        assert points_in_polygon_xy(points, loop) == [is_point_in_polygon_xy(p, loop) for p in points]
        for radius in (0.05, 0.4):
            edge = [[p[0] + radius, p[1], 0.0] for p in loop]
            for batch in (points, edge):
                assert near_loop(batch, loop, radius) == [distance_to_loop(p, loop) < radius for p in batch]


def test_curve_projection_batch_matches_single():
    rng = random.Random(9)
    points = [[math.cos(t) * 4, math.sin(t) * 2, 0.0] for t in [i * 0.05 for i in range(126)]]
    curve = _Curve(points + points[:1], closed=True)
    probes = [[rng.uniform(-5, 5), rng.uniform(-3, 3), 0.0] for _ in range(500)] + points
    assert curve.project_many(probes) == [curve.project(p) for p in probes]


# ----------------------------------------------------------------------------
# a field saved before a change still belongs to its document
# ----------------------------------------------------------------------------


def test_a_field_saved_by_an_earlier_version_still_matches_and_densifies_identically():
    """``tests/data/*_v1.json`` were written before the 2026-09 refactor. A Rhino
    document carries exactly this payload, so if ``mismatch`` stops returning
    ``None`` here every saved document silently loses its field."""
    field = CrossField.load_from_json(os.path.join(DATA, "field_v1.json"))
    assert field.mismatch(PLATE, guides=CABLE, target_length=0.6, guide_weight=5.0) is None
    assert field.locator() is not None

    coarse = CoarsePseudoQuadMesh.load_from_json(os.path.join(DATA, "coarse_v1.json"))
    dense, stats = field.densify(coarse)
    with open(os.path.join(DATA, "dense_v1.json")) as f:
        expected = json.load(f)
    # numpy patch releases move the last bits (2.5.1 -> 2.5.3 moved one by 1e-14), so not ==
    xyz = [c for v in dense.vertices() for c in dense.vertex_coordinates(v)]
    assert xyz == pytest.approx([c for point in expected["vertices"] for c in point], abs=1e-9)
    assert [dense.face_vertices(f) for f in dense.faces()] == expected["faces"]


def test_the_stored_inputs_keep_their_shape():
    field = CrossField.from_boundary(PLATE, guides=CABLE, target_length=0.6, guide_weight=5.0)
    with open(os.path.join(DATA, "field_v1.json")) as f:
        stored = json.load(f)
    assert list(field.inputs) == list(stored["inputs"])
    assert list(field.inputs["geometry"]) == list(stored["inputs"]["geometry"])
    assert list(field.inputs["params"]) == list(stored["inputs"]["params"])
    assert field.inputs == stored["inputs"]


def test_field_inputs_defaults_are_from_boundarys():
    import inspect

    solve = inspect.signature(CrossField.from_boundary).parameters
    record = inspect.signature(FieldInputs).parameters
    assert list(solve) == list(record)
    for name in solve:
        assert solve[name].default == record[name].default, name


# ----------------------------------------------------------------------------
# routing and fallbacks
# ----------------------------------------------------------------------------


def test_route_is_recorded_not_read_off_the_notes():
    d = FieldDecomposition.from_boundary(PLATE, target_length=0.6)
    d.quad_mesh(target_length=1.0)
    assert d.route() == "field"
    d.repair_notes.append("a note that mentions the triangulation and DISCARDED")
    assert d.route() == "field"


def test_a_crash_in_densification_is_reported_or_raised(monkeypatch):
    d = FieldDecomposition.from_boundary(PLATE, target_length=0.6)

    def broken(*args, **kwargs):
        raise TypeError("broken on purpose")

    monkeypatch.setattr(d, "densify", broken)
    with pytest.raises(TypeError):
        d.quad_mesh(target_length=1.0, strict=True)

    mesh = d.quad_mesh(target_length=1.0)
    assert mesh.number_of_faces()
    assert d.route() == "triangulation"
    assert "broken on purpose" in " ".join(d.repair_notes)
    assert "TypeError" in d.last_error


def test_densify_stats_always_carry_every_count():
    d = FieldDecomposition.from_boundary(PLATE, target_length=0.6)
    d.quad_mesh(target_length=1.0)
    for key in ("patches", "relaxed", "guarded", "flat", "poles", "coons", "stiffness", "folds"):
        assert key in d.densify_stats


def test_a_bad_guide_mode_is_refused_even_far_from_the_guide(field):
    far = [[100.0, 100.0, 0.0], [101.0, 100.0, 0.0]]
    with pytest.raises(ValueError):
        from_curves(field.background, [far], mode="sideways")


def test_field_tau_still_works_under_its_old_name():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        d = FieldDecomposition.from_boundary(PLATE, target_length=0.6, relax=True, field_tau=0.5, solve=False)
    assert d.inputs["tau"] == 0.5
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_snapped_count_survives_the_symmetry_projection():
    square = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]
    arcs = []
    for k in range(4):
        c, s = math.cos(k * math.pi / 2), math.sin(k * math.pi / 2)
        arc = [[2 + 6 * i / 10.0, 2 + 1.5 * math.sin(math.pi * i / 10.0), 0.0] for i in range(11)]
        arcs.append([[5 + c * (p[0] - 5) - s * (p[1] - 5), 5 + s * (p[0] - 5) + c * (p[1] - 5), 0.0] for p in arc])
    d = FieldDecomposition.from_boundary(square, guides=arcs, target_length=0.5)
    assert d.symmetry is not None and not d.symmetry.trivial
    _, _, info = d._build()
    # every arm of a singularity shares its start, so ends DO get merged
    assert info["snapped"] > 0
