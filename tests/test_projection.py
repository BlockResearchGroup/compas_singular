"""Headless tests for smoothing a mesh ON a surface.

No Rhino: the surface is a quarter cylinder that projects analytically, standing
in for anything with a ``closest_point`` method.
"""

import random
from math import atan2
from math import cos
from math import pi
from math import sin

from compas.datastructures import Mesh
from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import distance_point_point

from compas_singular.datastructures import automated_smoothing_constraints
from compas_singular.datastructures import automated_smoothing_surface_constraints
from compas_singular.datastructures import closest_point_on_constraint
from compas_singular.datastructures import surface_constrained_smoothing


RADIUS = 5.0
LENGTH = 10.0
SWEEP = pi / 2


class QuarterCylinder(object):
    """``y = R cos(t)``, ``z = R sin(t)`` for ``x`` in [0, L] and ``t`` in [0, pi/2]."""

    def closest_point(self, point):
        x, y, z = point
        t = min(max(atan2(z, y), 0.0), SWEEP)
        return Point(min(max(x, 0.0), LENGTH), RADIUS * cos(t), RADIUS * sin(t))


def _off_cylinder(xyz):
    """Signed radial distance from the cylinder."""
    return (xyz[1] ** 2 + xyz[2] ** 2) ** 0.5 - RADIUS


def _borders(n=64):
    arc = [[0.0, RADIUS * cos(SWEEP * i / n), RADIUS * sin(SWEEP * i / n)] for i in range(n + 1)]
    return [
        Polyline(arc),
        Polyline([[LENGTH, y, z] for _, y, z in arc]),
        Polyline([[0.0, RADIUS, 0.0], [LENGTH, RADIUS, 0.0]]),
        Polyline([[0.0, 0.0, RADIUS], [LENGTH, 0.0, RADIUS]]),
    ]


KINKS = [[0.0, RADIUS, 0.0], [LENGTH, RADIUS, 0.0], [0.0, 0.0, RADIUS], [LENGTH, 0.0, RADIUS]]


def _draped_grid(nx=8, nt=6, amplitude=0.3, seed=0):
    """A grid on the cylinder whose interior vertices are shaken OFF the surface."""
    mesh = Mesh.from_meshgrid(dx=LENGTH, nx=nx, dy=SWEEP, ny=nt)
    rng = random.Random(seed)
    for vertex in mesh.vertices():
        x, t, _ = mesh.vertex_coordinates(vertex)
        r = RADIUS
        if not mesh.is_vertex_on_boundary(vertex):
            x += rng.uniform(-amplitude, amplitude)
            t += rng.uniform(-amplitude, amplitude) / RADIUS
            r += rng.uniform(-amplitude, amplitude)
        mesh.vertex_attributes(vertex, "xyz", [x, r * cos(t), r * sin(t)])
    return mesh


def _distance_to_borders(xyz, borders):
    return min(distance_point_point(xyz, closest_point_on_constraint(border, xyz)) for border in borders)


def test_surface_constraints_assign_surface_borders_and_kinks():
    mesh = _draped_grid()
    surface, borders = QuarterCylinder(), _borders()
    constraints = automated_smoothing_surface_constraints(mesh, surface, borders, kinks=KINKS)

    assert set(constraints) == set(mesh.vertices())
    pins = [v for v, c in constraints.items() if isinstance(c, Point)]
    assert len(pins) == 4
    for vertex in pins:
        assert min(distance_point_point(mesh.vertex_coordinates(vertex), k) for k in KINKS) < 1e-9
    for vertex, constraint in constraints.items():
        if vertex in pins:
            continue
        if mesh.is_vertex_on_boundary(vertex):
            assert any(constraint is border for border in borders)
        else:
            assert constraint is surface


def test_surface_constrained_smoothing_keeps_the_mesh_on_the_surface():
    mesh = _draped_grid()
    surface, borders = QuarterCylinder(), _borders()
    corners = {v: mesh.vertex_coordinates(v) for v in mesh.vertices() if mesh.vertex_degree(v) == 2}
    before = {v: mesh.vertex_coordinates(v) for v in mesh.vertices()}

    surface_constrained_smoothing(mesh, surface, borders, kmax=30, damping=0.5, algorithm="area")

    for vertex in mesh.vertices():
        xyz = mesh.vertex_coordinates(vertex)
        if mesh.is_vertex_on_boundary(vertex):
            assert _distance_to_borders(xyz, borders) < 1e-9
        else:
            assert abs(_off_cylinder(xyz)) < 1e-9
    # fixed, but still projected onto their border every iteration: float noise only
    for vertex, xyz in corners.items():
        assert distance_point_point(mesh.vertex_coordinates(vertex), xyz) < 1e-12
    assert max(distance_point_point(before[v], mesh.vertex_coordinates(v)) for v in mesh.vertices()) > 0.1


def test_automated_smoothing_constraints_pins_win_over_curves_and_surface():
    mesh = _draped_grid()
    surface, borders = QuarterCylinder(), _borders()
    pin = [LENGTH / 2 + 0.2, RADIUS, 0.0]
    constraints = automated_smoothing_constraints(mesh, points=[pin], curves=borders, surface=surface)

    pinned = [v for v, c in constraints.items() if isinstance(c, Point)]
    assert len(pinned) == 1
    assert list(constraints[pinned[0]]) == pin
    for vertex, constraint in constraints.items():
        if vertex in pinned:
            continue
        if mesh.is_vertex_on_boundary(vertex):
            assert any(constraint is border for border in borders)
        else:
            assert constraint is surface
