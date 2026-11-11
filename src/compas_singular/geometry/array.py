from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import cos
from math import pi
from math import sin

from compas.geometry import add_vectors

__all__ = ["circle_evaluate", "archimedean_spiral_evaluate", "line_array", "rectangular_array", "circular_array", "spiral_array"]


def circle_evaluate(t: float, r: float, z: float = 0) -> list[float]:
    """Evaluate a circle of radius ``r`` centred on the origin at angle ``t`` in radians.

    Parameters
    ----------
    t : float
        The angle in radians.
    r : float
        The radius.
    z : float, optional
        The elevation of the circle's plane above the XY plane.

    Returns
    -------
    list
        The XYZ coordinates of the point.
    """
    return [r * cos(t), r * sin(t), z]


def archimedean_spiral_evaluate(t: float, a: float, b: float, z: float = 0) -> list[float]:
    """Evaluate an archimedean spiral ``r = a + b * theta`` at angle ``t`` in radians.

    Parameters
    ----------
    t : float
        The angle in radians.
    a : float
        The offset angle of the spiral.
    b : float
        The radial growth per radian.
    z : float, optional
        The elevation of the spiral's plane above the XY plane.

    Returns
    -------
    list
        The XYZ coordinates of the point.
    """
    return [b * t * cos(t + a), b * t * sin(t + a), z]


def line_array(n: int, d: float, anchor: list[float] = [0.0, 0.0, 0.0]) -> list[list[float]]:
    return [add_vectors(anchor, [i * d, 0.0, 0.0]) for i in range(n)]


def rectangular_array(nx: int, ny: int, dx: float, dy: float, anchor: list[float] = [0.0, 0.0, 0.0]) -> list[list[float]]:
    return [add_vectors(anchor, [x * dx, y * dy, 0.0]) for y in range(ny) for x in range(nx)]


def circular_array(n: int, r: float, anchor: list[float] = [0.0, 0.0, 0.0]) -> list[list[float]]:
    return [add_vectors(anchor, circle_evaluate(2 * pi * float(i) / float(n), r)) for i in range(n)]


def spiral_array(n: int, d: float, anchor: list[float] = [0.0, 0.0, 0.0]) -> list[list[float]]:
    # spiral parameters set to respect d spacing between consecutive points and consecutive spiral elements
    a, b = 0, d / (2 * pi)
    ts = [pi * b]
    for i in range(n):
        ts.append((2 * d / b + ts[-1] ** 2) ** 0.5)
    return [add_vectors(anchor, archimedean_spiral_evaluate(t, a, b, 0)) for t in ts]


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass

    # print(rectangular_array(4, 2, 10.0, 0.5, anchor=[1.0, -1.0, 0.0]))
    # print(spiral_array(15, 2, anchor=[1.0, -1.0, 0.0]))
