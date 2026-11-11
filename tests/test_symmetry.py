"""Symmetry: detection, the unit's guards, and the scripting workflow on both routes."""

import random
from math import cos
from math import pi
from math import sin

import pytest
from compas.data import json_dump
from compas.data import json_load

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.datastructures.mesh.smoothing import constrained_smoothing
from compas_singular.datastructures.mesh_quad_coarse.mesh_quad_coarse import CoarseQuadMesh
from compas_singular.editing import CoarseEditor
from compas_singular.framefield.field_decomposition import FieldDecomposition
from compas_singular.symmetry import SymmetryGroup
from compas_singular.symmetry import find_symmetry
from compas_singular.symmetry.measure import mesh_invariance

# ==============================================================================
# domains
# ==============================================================================


def ngon(n, r=5.0, cx=5.0, cy=5.0, phase=0.0):
    return [[cx + r * cos(phase + 2 * pi * i / n), cy + r * sin(phase + 2 * pi * i / n), 0.0] for i in range(n)]


def trefoil(n=120, hole=None):
    outer = [[(7 + 3.5 * cos(3 * t)) * cos(t), (7 + 3.5 * cos(3 * t)) * sin(t), 0.0] for t in (2 * pi * i / n for i in range(n))]
    inners = [ngon(32, hole, 0.0, 0.0)] if hole else []
    return outer, inners


def arc(cx, cy, r, a0, a1, n=20):
    return [[cx + r * cos(a0 + (a1 - a0) * i / n), cy + r * sin(a0 + (a1 - a0) * i / n), 0.0] for i in range(n + 1)]


def turn(points, k, centre=(5.0, 5.0)):
    out = []
    for p in points:
        x, y = p[0] - centre[0], p[1] - centre[1]
        for _ in range(k % 4):
            x, y = -y, x
        out.append([x + centre[0], y + centre[1], 0.0])
    return out


SQUARE = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
RECT = [[0.0, 0.0, 0.0], [14.0, 0.0, 0.0], [14.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
L_SHAPE = [[0, 0, 0], [12, 0, 0], [12, 4, 0], [5, 4, 0], [5, 10, 0], [0, 10, 0]]
PLUS = [[4, 0, 0], [8, 0, 0], [8, 4, 0], [12, 4, 0], [12, 8, 0], [8, 8, 0], [8, 12, 0], [4, 12, 0], [4, 8, 0], [0, 8, 0], [0, 4, 0], [4, 4, 0]]
FOUR_POLES = [[3.0, 3.0, 0.0], [7.0, 3.0, 0.0], [7.0, 7.0, 0.0], [3.0, 7.0, 0.0]]
#: Four arcs that are D4 images of each other.
FOUR_ARCS = [turn(arc(0.0, 0.0, 6.5, 0.35, pi / 2 - 0.35), k) for k in range(4)]
#: Four arcs that are only C4 images: the base arc is NOT symmetric about 45 deg.
PINWHEEL_ARCS = [turn(arc(0.0, 0.0, 6.5, 0.25, pi / 2 - 0.55), k) for k in range(4)]
TREFOIL_OUTER, TREFOIL_HOLES = trefoil(hole=2.0)

# ==============================================================================
# the group model
# ==============================================================================


@pytest.fixture
def d4():
    return SymmetryGroup([0, 0, 0], 4, True, 0.0)


def test_the_d4_group(d4):
    assert d4.order == 8
    assert d4.keys() == ["R90", "R180", "R270", "M0", "M45", "M90", "M135"]
    assert len(d4.subgroups()) == 10
    assert SymmetryGroup.from_data(d4.to_data()) == d4


@pytest.mark.parametrize(
    "keys, name",
    [
        (["M0", "M45"], "D4"),
        (["M0", "M90"], "D2"),
        ("R90", "C4"),
    ],
)
def test_generators_give_the_subgroup(d4, keys, name):
    assert d4.subgroup(keys).name == name


def test_subgroup_edge_cases(d4):
    assert d4.subgroup("M45").keys() == ["M45"]
    assert d4.subgroup([]).is_trivial


def test_composition(d4):
    m = d4.element("M45")
    assert d4.compose(m, m).key == "I"
    assert d4.compose(d4.element("R90"), d4.element("R90")).key == "R180"


# ==============================================================================
# detection
# ==============================================================================


@pytest.mark.parametrize(
    "kw, expected, centre",
    [
        pytest.param(dict(outer=SQUARE), "D4", (5, 5), id="square"),
        pytest.param(dict(outer=RECT), "D2", (7, 5), id="rectangle"),
        pytest.param(dict(outer=ngon(6)), "D6", (5, 5), id="hexagon"),
        pytest.param(dict(outer=ngon(6, phase=pi / 6)), "D6", (5, 5), id="hexagon rotated 30"),
        pytest.param(dict(outer=ngon(3)), "D3", (5, 5), id="triangle"),
        pytest.param(dict(outer=ngon(5)), "D5", (5, 5), id="pentagon"),
        pytest.param(dict(outer=trefoil()[0]), "D3", (0, 0), id="trefoil"),
        pytest.param(dict(outer=TREFOIL_OUTER, inners=TREFOIL_HOLES), "D3", (0, 0), id="trefoil with hole"),
        pytest.param(dict(outer=PLUS), "D4", (6, 6), id="plus plate"),
        pytest.param(dict(outer=L_SHAPE), "C1", None, id="L-shape"),
        pytest.param(dict(outer=SQUARE, inners=[ngon(26, 2.0)]), "D4", (5, 5), id="circle hole sampled 26"),
        pytest.param(dict(outer=SQUARE, inners=[ngon(32, 2.0, phase=5 * pi / 180)]), "D4", (5, 5), id="circle hole sampled 32 rotated 5 deg"),
        pytest.param(dict(outer=SQUARE, guides=FOUR_ARCS), "D4", (5, 5), id="four symmetric arcs"),
        pytest.param(dict(outer=SQUARE, guides=PINWHEEL_ARCS), "C4", (5, 5), id="pinwheel arcs"),
        pytest.param(dict(outer=SQUARE, poles=FOUR_POLES), "D4", (5, 5), id="four poles"),
        pytest.param(dict(outer=SQUARE, guides=[[[2, 2, 0], [2, 6, 0]]]), "C1", None, id="off-centre guide"),
    ],
)
def test_detected_group(kw, expected, centre):
    report = find_symmetry(**kw)
    assert report.group.name == expected
    if centre is not None:
        assert report.centre[0] == pytest.approx(centre[0], abs=1e-6)
        assert report.centre[1] == pytest.approx(centre[1], abs=1e-6)


def test_excluding_guides_restores_the_walls_group():
    report = find_symmetry(outer=SQUARE, guides=[[[2, 2, 0], [2, 6, 0]]], include=("walls", "holes", "poles"))
    assert report.group.name == "D4"


def test_two_guides_mirrored_within_tolerance_give_one_mirror():
    report = find_symmetry(outer=SQUARE, guides=[[[2, 2, 0], [2, 6.001, 0]], [[8, 2, 0], [8, 6, 0]]])
    assert report.keys == ["M90"]
    assert dict((label, rep.group.name) for label, rep in report.by_feature()) == {"walls": "D4", "+guides": "D1"}


def test_a_guide_just_off_is_a_named_near_miss():
    report = find_symmetry(outer=SQUARE, guides=[[[2, 2, 0], [2, 6.05, 0]], [[8, 2, 0], [8, 6, 0]]])
    assert report.group.is_trivial
    assert any(w and w[0].startswith("guide") for _, _, w in report.near_misses)


def test_exact_translated_copies_keep_d4():
    for s in range(40):
        off = s * 1e-7 + 1.5e-6 * (s % 3)
        report = find_symmetry(outer=[[p[0] + off, p[1] + off, 0] for p in SQUARE], inners=[[[p[0] + off, p[1] + off, 0] for p in ngon(24, 2.0)]])
        assert report.group.order == 8, s


@pytest.mark.parametrize("eps", [1e-7, 1e-6, 1e-5])
def test_noise_keeps_d4(eps):
    rng = random.Random(0)
    for _ in range(20):
        loops = [[[p[0] + rng.uniform(-eps, eps), p[1] + rng.uniform(-eps, eps), 0] for p in loop] for loop in (SQUARE, ngon(24, 2.0))]
        assert find_symmetry(outer=loops[0], inners=[loops[1]]).group.order == 8


def test_keys_do_not_depend_on_position():
    moved = [[p[0] + 123.4, p[1] - 56.7, 0] for p in ngon(6)]
    assert find_symmetry(outer=ngon(6)).keys == find_symmetry(outer=moved).keys


def test_geometry_draws_mirrors_clipped_to_the_wall_and_rotation_arcs():
    geometry = find_symmetry(outer=SQUARE).geometry()
    assert len(geometry["mirrors"]) == 4
    assert all(abs(line.length - 10.0) < 1e-6 or abs(line.length - 10.0 * 2**0.5) < 1e-6 for line in geometry["mirrors"])
    assert len(geometry["rotations"]) == 2


def test_a_three_fold_group_notes_the_centre_singularity():
    assert any("singularity at the centre" in n for n in find_symmetry(outer=ngon(3)).notes())


# ==============================================================================
# the unit's guards
# ==============================================================================


def _unit(keys=None, **kw):
    return SkeletonDecomposition.from_boundary(target_length=0.5, **kw).symmetry_unit(keys=keys)


def test_a_fresh_unit_passes_check():
    ok, notes = _unit(outer_boundary=SQUARE).check()
    assert ok, notes


def test_a_seam_corner_moved_off_the_seam_is_caught_and_refused():
    unit = _unit(outer_boundary=SQUARE)
    seam = unit.seams[0]
    on_seam = [v for v in unit.vertices() if seam.locate(unit.vertex_coordinates(v), unit.eps) not in (None, 0.0)]
    broken = unit.copy()
    broken.attributes["symmetry"] = unit.symmetry
    x, y, z = broken.vertex_coordinates(on_seam[0])
    broken.vertex_attributes(on_seam[0], "xyz", [x + 0.3, y + 0.3, z])
    ok, notes = broken.check()
    assert not ok, notes
    with pytest.raises(ValueError, match="cannot densify"):
        broken.quad_mesh()


def test_a_density_changed_after_quad_mesh_is_refused_at_expansion():
    unit = _unit(outer_boundary=SQUARE)
    unit.collect_strips()
    unit.set_strips_density_target(0.5)
    quad_unit = unit.quad_mesh()
    skey = list(unit.strips())[0]
    unit.set_strip_density(skey, unit.get_strip_density(skey) + 2)
    with pytest.raises(ValueError, match="changed since quad_mesh"):
        quad_unit.expand_symmetrically(unit)


def test_poles_on_the_diagonal_mirrors_are_refused_with_d2_suggested():
    with pytest.raises(ValueError) as info:
        _unit(outer_boundary=SQUARE, point_features=FOUR_POLES)
    assert "pole" in str(info.value) and "D2" in str(info.value)


def test_rotation_seams_glue_densities():
    # The FIELD route's C3 unit: its strips reach the two seams as different strips.
    c3 = FieldDecomposition.from_boundary(TREFOIL_OUTER, inner_boundaries=TREFOIL_HOLES, target_length=0.5, solve=False).symmetry_unit(keys=["R120"])
    c3.collect_strips()
    c3.set_strips_density_target(0.5)
    glued = [g for g in c3.glued_strips() if len(g) > 1]
    assert glued, c3.glued_strips()
    group = glued[0]
    c3.set_strip_density(group[0], 7)
    assert all(c3.get_strip_density(k) == 7 for k in group)
    CoarseQuadMesh.set_strip_density(c3, group[0], 3)  # force the pair apart
    ok, notes = c3.check()
    assert not ok and "rotation seam" in " ".join(notes)


@pytest.mark.parametrize("name", ["coarse_mesh", "decomposition_mesh", "get_field", "quad_mesh", "edges_to_curves", "decomposition_polylines", "report"])
def test_an_unsolved_field_decomposition_asks_for_solve(name):
    d = FieldDecomposition.from_boundary(SQUARE, target_length=0.5, solve=False)
    assert d.is_solved is False
    with pytest.raises(ValueError, match=r"solve\(\) first"):
        getattr(d, name)()


def test_solve_later_equals_solving_up_front():
    solved = FieldDecomposition.from_boundary(SQUARE, target_length=0.5, solve=False).solve()
    direct = FieldDecomposition.from_boundary(SQUARE, target_length=0.5)
    a, b = solved.coarse_mesh(), direct.coarse_mesh()
    assert a.number_of_faces() == b.number_of_faces()
    assert sorted(map(tuple, (a.vertex_coordinates(v) for v in a.vertices()))) == sorted(map(tuple, (b.vertex_coordinates(v) for v in b.vertices())))
    assert direct.inputs.get("target_length") == 0.5


def test_a_unit_survives_save_and_load(tmp_path):
    path = str(tmp_path / "unit.json")
    json_dump(_unit(outer_boundary=SQUARE, keys=["M0", "M90"]), path)
    loaded = json_load(path)
    assert type(loaded).__name__ == "SymmetricUnit"
    assert loaded.group.name == "D2" and len(loaded.seams) == 2
    ok, notes = loaded.check()
    assert ok, notes


def test_a_unit_survives_a_committed_move_and_still_expands_exactly():
    unit = _unit(outer_boundary=SQUARE, keys=["M0", "M90"])
    editor = CoarseEditor(unit)
    v = next(v for v in unit.vertices() if not unit.is_vertex_on_boundary(v))
    x, y, z = unit.vertex_coordinates(v)
    editor.move_vertex(v, [x + 0.1, y + 0.05, z])
    editor.commit()
    assert type(unit).__name__ == "SymmetricUnit"
    assert unit.symmetry.get("group", {}).get("name") == "D2"
    unit.collect_strips()
    unit.set_strips_density_target(0.5)
    quad_mesh = unit.quad_mesh().expand_symmetrically(unit)
    assert mesh_invariance(quad_mesh, unit.group)["share"] == 1.0
    # symmetric smoothing stays exactly symmetric
    boundary = set(v for loop in quad_mesh.vertices_on_boundaries() for v in loop)
    constrained_smoothing(quad_mesh, kmax=100, damping=0.5, fixed=list(boundary))
    assert mesh_invariance(quad_mesh, unit.group)["share"] == 1.0


def test_a_unit_survives_a_strip_split_on_the_rebuild_path():
    unit = _unit(outer_boundary=SQUARE, keys=["M0", "M90"])
    unit.collect_strips()
    editor = CoarseEditor(unit)
    ok = editor.divide(skey=list(unit.strips())[0], t=0.5)
    ok = ok[0] if isinstance(ok, tuple) else ok
    layout, notes = editor.commit()
    assert ok and layout is not None and notes.get("path") == "rebuild"
    assert type(unit).__name__ == "SymmetricUnit"
    assert unit.symmetry.get("group", {}).get("name") == "D2"
    ok, notes = unit.check()
    assert ok, notes
    unit.collect_strips()
    unit.set_strips_density_target(0.5)
    assert mesh_invariance(unit.quad_mesh().expand_symmetrically(unit), unit.group)["share"] == 1.0


# ==============================================================================
# the scripting workflow, verbatim, on both routes
# ==============================================================================

# label, skeleton route kwargs, field route kwargs (None: same), keys, expected enforced group
WORKFLOW_CASES = [
    ("square", dict(outer_boundary=SQUARE), None, None, "D4"),
    ("square D2", dict(outer_boundary=SQUARE), None, ["M0", "M90"], "D2"),
    ("square C4", dict(outer_boundary=SQUARE), None, ["R90"], "C4"),
    ("square D1", dict(outer_boundary=SQUARE), None, ["M45"], "D1"),
    ("square C2", dict(outer_boundary=SQUARE), None, ["R180"], "C2"),
    ("plus plate", dict(outer_boundary=PLUS), None, None, "D4"),
    ("hexagon + hole", dict(outer_boundary=ngon(6), inner_boundaries=[ngon(32, 1.5)]), None, None, "D6"),
    ("trefoil + hole", dict(outer_boundary=TREFOIL_OUTER, inner_boundaries=TREFOIL_HOLES), None, None, "D3"),
    ("trefoil + hole C3", dict(outer_boundary=TREFOIL_OUTER, inner_boundaries=TREFOIL_HOLES), "skip", ["R120"], "C3"),
    ("disc 48 as D4", dict(outer_boundary=ngon(48)), None, ["M0", "M45"], "D4"),
    ("4 poles D2", dict(outer_boundary=SQUARE, point_features=FOUR_POLES), "skip", ["M0", "M90"], "D2"),
    ("four arcs", "skip", dict(outer_boundary=SQUARE, guides=FOUR_ARCS), None, "D4"),
    ("pinwheel arcs C4", "skip", dict(outer_boundary=SQUARE, guides=PINWHEEL_ARCS), None, "C4"),
    ("L-shape (no symmetry)", dict(outer_boundary=L_SHAPE), None, None, "C1"),
]


def _workflow_params():
    for route in ("skeleton", "field"):
        for label, skeleton_kw, field_kw, keys, expected in WORKFLOW_CASES:
            kw = skeleton_kw if route == "skeleton" else (field_kw or skeleton_kw)
            if kw == "skip" or (route == "field" and "point_features" in kw):
                continue
            yield pytest.param(route, kw, keys, expected, id="{} {}".format(route, label))


def _all_quad(mesh):
    face_pole = mesh.attributes.get("face_pole") or {}
    return all(len(mesh.face_vertices(f)) == 4 or (len(mesh.face_vertices(f)) == 3 and f in face_pole) for f in mesh.faces())


@pytest.mark.parametrize("route, kw, keys, expected", list(_workflow_params()))
def test_the_workflow_expands_to_an_exactly_symmetric_quad_mesh(route, kw, keys, expected):
    kw = dict(kw)
    if route == "field":
        cls = FieldDecomposition
        kw.setdefault("solve", False)
    else:
        cls = SkeletonDecomposition
    decomposition = cls.from_boundary(target_length=0.5, **kw)
    decomposition.find_symmetry()
    unit = decomposition.symmetry_unit(keys=keys)
    unit.collect_strips()
    unit.set_strips_density_target(0.5)
    layout = unit.expand_symmetrically()
    quad_mesh = unit.quad_mesh().expand_symmetrically(unit)

    assert unit.group.name == expected
    ok, notes = unit.check()
    assert ok, notes
    assert layout.is_manifold() and quad_mesh.is_manifold()
    assert _all_quad(layout)
    assert _all_quad(quad_mesh)
    # measured by nearest-neighbour search on positions, independently of the orbit labels
    assert mesh_invariance(layout, unit.group)["share"] == 1.0
    assert mesh_invariance(quad_mesh, unit.group)["share"] == 1.0
