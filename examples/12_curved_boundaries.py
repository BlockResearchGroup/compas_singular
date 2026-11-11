"""12 - Curved boundaries.

A curved boundary is given as a list of points along it. Two things decide
whether the mesh follows the curve:

* the SAMPLING: the points are the boundary. Sampled too coarsely, a circle is a
  polygon, and every point is a corner that the skeleton branches out from.
* the CURVATURE of the coarse edges: a coarse edge is a straight line between
  two corners, but it remembers the part of the boundary it came from.
  densify() uses that curve by default; boundary_curvature=False uses
  the straight line instead.

Three cases, on an ellipse with a circular hole:

* the hole sampled with 8 points,
* the hole sampled with 64 points, densified without the curves,
* the hole sampled with 64 points, densified with the curves.
"""
import math

from compas.colors import Color
from compas.geometry import Polyline, Translation
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_singular.algorithms import SkeletonDecomposition

outer = [[6.0 * math.cos(2 * math.pi * i / 96), 4.0 * math.sin(2 * math.pi * i / 96), 0.0] for i in range(96)]
coarse_hole = [[1.6 * math.cos(2 * math.pi * i / 8), 1.6 * math.sin(2 * math.pi * i / 8), 0.0] for i in range(8)]
fine_hole = [[1.6 * math.cos(2 * math.pi * i / 64), 1.6 * math.sin(2 * math.pi * i / 64), 0.0] for i in range(64)]

cases = [
    (coarse_hole, True),
    (fine_hole, False),
    (fine_hole, True),
]

# The exact area of the domain, to compare with.
print('exact area :', round(math.pi * 6.0 * 4.0 - math.pi * 1.6 ** 2, 2))

config = Config()
config.renderer.show_grid = False
viewer = Viewer(config=config)
viewer.renderer.view = 'top'

for i, (hole, curvature) in enumerate(cases):

    decomposition = SkeletonDecomposition.from_boundary(outer, inner_boundaries=[hole], target_length=0.5)
    coarse = decomposition.coarse_mesh()
    coarse.set_strips_density_target(0.4)
    dense = coarse.densify(boundary_curvature=curvature)
    print('case', i + 1, ':', coarse.number_of_faces(), 'patches, area', round(dense.area(), 2))

    # Top: the coarse patches. Bottom: the dense mesh, with the true circle in red.
    group = viewer.scene.add_group(name='case {}'.format(i + 1))
    for polyline in decomposition.polylines:
        group.add(Polyline(polyline).transformed(Translation.from_vector([14 * i, 10, 0])), linecolor=Color.blue(), linewidth=2)
    group.add(dense.transformed(Translation.from_vector([14 * i, 0, 0])), show_points=False)
    group.add(Polyline(fine_hole + fine_hole[:1]).transformed(Translation.from_vector([14 * i, 0, 0.01])), linecolor=Color.red(), linewidth=3)

viewer.renderer.camera.target.set(14, 5, 0)
viewer.renderer.camera.distance = 22
viewer.show()
