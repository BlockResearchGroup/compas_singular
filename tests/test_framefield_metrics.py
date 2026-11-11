"""Tests for the frame-field measuring tools: the tier-1 quality gate and the guide metrics.

``hard_floor`` is what ``field_decomposition`` checks before it accepts a field
layout, so each of its refusals is pinned along with the message that names it.
"""

import math

import pytest

from compas_singular.datastructures import QuadMesh
from compas_singular.framefield.guides import format_guide_metrics
from compas_singular.framefield.guides import guide_metrics
from compas_singular.framefield.quality import HARD_MAX_ANGLE
from compas_singular.framefield.quality import HARD_MIN_ANGLE
from compas_singular.framefield.quality import curve_alignment
from compas_singular.framefield.quality import curve_alignment_profile
from compas_singular.framefield.quality import hard_floor
from compas_singular.framefield.quality import mesh_quality


@pytest.fixture
def grid():
    """A 4 x 4 grid of unit quads on [0, 4] x [0, 4]."""
    n = 4
    vertices = [[float(i), float(j), 0.0] for j in range(n + 1) for i in range(n + 1)]
    faces = [[j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i] for j in range(n) for i in range(n)]
    return QuadMesh.from_vertices_and_faces(vertices, faces)


HORIZONTAL_MID_ROW = [[0.0, 2.5, 0.0], [4.0, 2.5, 0.0]]  # through face centres
HORIZONTAL_ON_JOINT = [[0.0, 2.0, 0.0], [4.0, 2.0, 0.0]]  # along a mesh line
DIAGONAL = [[0.0, 0.0, 0.0], [4.0, 4.0, 0.0]]


# ----------------------------------------------------------------------------
# hard_floor: tier 1 of the quality gate
# ----------------------------------------------------------------------------


def test_a_square_grid_passes_the_hard_floor(grid):
    metrics = mesh_quality(grid)
    assert metrics["min_angle"] == pytest.approx(90.0)
    assert metrics["aspect_max"] == pytest.approx(1.0)
    assert hard_floor(metrics) == (True, "")


@pytest.mark.parametrize("metrics", [{}, {"faces": 0}])
def test_the_hard_floor_refuses_an_empty_mesh(metrics):
    assert hard_floor(metrics) == (False, "no faces")


def test_the_hard_floor_refuses_a_near_zero_angle(grid):
    ok, why = hard_floor(dict(mesh_quality(grid), min_angle=HARD_MIN_ANGLE / 5.0))
    assert not ok
    assert "min_angle" in why and "face" in why


def test_the_hard_floor_refuses_a_near_straight_angle(grid):
    ok, why = hard_floor(dict(mesh_quality(grid), max_angle=(HARD_MAX_ANGLE + 180.0) / 2.0))
    assert not ok
    assert "max_angle" in why


@pytest.mark.parametrize("aspect", [float("inf"), float("nan")])
def test_the_hard_floor_refuses_a_non_finite_aspect(grid, aspect):
    ok, why = hard_floor(dict(mesh_quality(grid), aspect_max=aspect))
    assert not ok
    assert "zero-length edge" in why


def test_the_thresholds_sit_just_inside_degenerate(grid):
    metrics = mesh_quality(grid)
    assert hard_floor(dict(metrics, min_angle=HARD_MIN_ANGLE))[0]
    assert hard_floor(dict(metrics, max_angle=HARD_MAX_ANGLE))[0]


# ----------------------------------------------------------------------------
# curve_alignment
# ----------------------------------------------------------------------------


def test_a_curve_along_the_grid_is_aligned(grid):
    assert curve_alignment(grid, HORIZONTAL_ON_JOINT) == pytest.approx(0.0)


def test_a_diagonal_is_as_far_off_as_the_cross_period_allows(grid):
    assert curve_alignment(grid, DIAGONAL) == pytest.approx(45.0)


def test_a_curve_off_the_mesh_has_no_alignment(grid):
    assert curve_alignment_profile(grid, [[10.0, 10.0, 0.0], [11.0, 10.0, 0.0]]) == []
    assert math.isnan(curve_alignment(grid, [[10.0, 10.0, 0.0], [11.0, 10.0, 0.0]]))


def test_the_profile_runs_along_the_curve(grid):
    profile = curve_alignment_profile(grid, HORIZONTAL_ON_JOINT, spacing=1.0)
    xs = [point[0] for point, _ in profile]
    assert xs == sorted(xs)
    assert len(profile) == 4


# ----------------------------------------------------------------------------
# guide_metrics
# ----------------------------------------------------------------------------


def test_a_guide_through_face_centres_is_carried_perfectly(grid):
    metrics = guide_metrics(grid, [HORIZONTAL_MID_ROW])
    assert metrics["faces"] == 4
    assert metrics["along"] == pytest.approx(0.0)
    assert metrics["across"] == pytest.approx(90.0)
    assert metrics["across_within"] == pytest.approx(100.0)
    assert metrics["off_centre"] == pytest.approx(0.0)
    assert metrics["across_tilted"] is None  # a horizontal guide is never tilted
    assert metrics["poles"] == 0


def test_a_guide_on_a_joint_is_off_centre_by_a_half_width(grid):
    # it touches the rows on both sides, and sits on the edge of each
    metrics = guide_metrics(grid, [HORIZONTAL_ON_JOINT])
    assert metrics["faces"] == 8
    assert metrics["off_centre"] == pytest.approx(1.0)
    assert metrics["along"] == pytest.approx(0.0)


def test_a_diagonal_guide_is_tilted_and_at_45_degrees(grid):
    metrics = guide_metrics(grid, [DIAGONAL])
    assert metrics["faces"] == 4
    assert metrics["along"] == pytest.approx(45.0)
    assert metrics["across_tilted"] == pytest.approx(45.0)
    assert metrics["across_within"] == pytest.approx(0.0)


def test_no_guide_means_unseen(grid):
    metrics = guide_metrics(grid, [])
    assert metrics["faces"] == 0
    assert all(value is None for key, value in metrics.items() if key not in ("faces", "poles"))
    assert format_guide_metrics(metrics) == "guide unseen (0 faces)"


def test_a_one_point_guide_is_skipped(grid):
    assert guide_metrics(grid, [[[1.0, 1.0, 0.0]]])["faces"] == 0


def test_pseudo_quad_faces_count_as_poles(grid):
    # collapse face 0 to a triangle: it has no two edge families to compare
    a, b, c, _ = grid.face_vertices(0)
    grid.delete_face(0)
    grid.add_face([a, b, c])
    metrics = guide_metrics(grid, [[[0.0, 0.5, 0.0], [4.0, 0.5, 0.0]]])
    assert metrics["poles"] == 1
    assert metrics["faces"] == 3


def test_format_guide_metrics_prints_one_line(grid):
    line = format_guide_metrics(guide_metrics(grid, [HORIZONTAL_MID_ROW]))
    assert "\n" not in line
    assert line.startswith("faces   4")
    assert "along   0.0 deg" in line
    assert "tilted   --  deg" in line
