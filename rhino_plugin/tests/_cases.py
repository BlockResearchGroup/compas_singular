"""Domains shared by the symmetry tests. Plain point lists, no Rhino."""
import os
import sys
from math import cos, pi, sin

REPO = r"C:\Users\Casper\libraries\carbcomn\compas_singular\compas_singular"
if os.path.join(REPO, "src") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "src"))


FAILURES = []


def check(name, condition, detail=""):
    print("  {} {}{}".format("ok  " if condition else "FAIL", name,
                             (" -- " + str(detail)) if detail else ""))
    if not condition:
        FAILURES.append(name)
    return condition


def finish():
    print()
    if FAILURES:
        print("{} FAILED: {}".format(len(FAILURES), ", ".join(FAILURES)))
        sys.exit(1)
    print("all passed")


def ngon(n, r=5.0, cx=5.0, cy=5.0, phase=0.0):
    return [[cx + r * cos(phase + 2 * pi * i / n), cy + r * sin(phase + 2 * pi * i / n), 0.0]
            for i in range(n)]


SQUARE = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
RECT = [[0.0, 0.0, 0.0], [14.0, 0.0, 0.0], [14.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
L_SHAPE = [[0, 0, 0], [12, 0, 0], [12, 4, 0], [5, 4, 0], [5, 10, 0], [0, 10, 0]]
PLUS = [[4, 0, 0], [8, 0, 0], [8, 4, 0], [12, 4, 0], [12, 8, 0], [8, 8, 0],
        [8, 12, 0], [4, 12, 0], [4, 8, 0], [0, 8, 0], [0, 4, 0], [4, 4, 0]]
FOUR_POLES = [[3.0, 3.0, 0.0], [7.0, 3.0, 0.0], [7.0, 7.0, 0.0], [3.0, 7.0, 0.0]]


def trefoil(n=120, hole=None):
    outer = [[(7 + 3.5 * cos(3 * t)) * cos(t), (7 + 3.5 * cos(3 * t)) * sin(t), 0.0]
             for t in (2 * pi * i / n for i in range(n))]
    inners = [ngon(32, hole, 0.0, 0.0)] if hole else []
    return outer, inners


def arc(cx, cy, r, a0, a1, n=20):
    return [[cx + r * cos(a0 + (a1 - a0) * i / n), cy + r * sin(a0 + (a1 - a0) * i / n), 0.0]
            for i in range(n + 1)]


def turn(points, k, centre=(5.0, 5.0)):
    out = []
    for p in points:
        x, y = p[0] - centre[0], p[1] - centre[1]
        for _ in range(k % 4):
            x, y = -y, x
        out.append([x + centre[0], y + centre[1], 0.0])
    return out


#: Four arcs that are D4 images of each other (from 24_symmetry.py).
FOUR_ARCS = [turn(arc(0.0, 0.0, 6.5, 0.35, pi / 2 - 0.35), k) for k in range(4)]

#: Four arcs that are only C4 images: the base arc is NOT symmetric about 45 deg.
PINWHEEL_ARCS = [turn(arc(0.0, 0.0, 6.5, 0.25, pi / 2 - 0.55), k) for k in range(4)]
