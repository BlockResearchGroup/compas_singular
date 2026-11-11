"""08 - Curve features.

A curve feature is a line the quad mesh must follow: a crease, a rib, a joint.
The domain is cut open along the line before the skeleton is computed, so the
line becomes a row of mesh edges.

Three cases on the same square:

* a slanted line from wall to wall,
* two crossing lines,
* a line with one free end. A free end becomes a pole of the mesh, and the
  quads around it are distorted; smoothing (06) is needed there.
"""
from compas.colors import Color
from compas.geometry import Polyline, Translation
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_singular.algorithms import SkeletonDecomposition

square = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]]

cases = [
    [[[0.0, 4.0, 0.0], [10.0, 6.0, 0.0]]],
    [[[0.0, 5.0, 0.0], [10.0, 5.0, 0.0]], [[5.0, 0.0, 0.0], [5.0, 10.0, 0.0]]],
    [[[0.0, 5.0, 0.0], [6.0, 5.0, 0.0]]],
]

config = Config()
config.renderer.show_grid = False
viewer = Viewer(config=config)
viewer.renderer.view = 'top'

for i, lines in enumerate(cases):

    # The lines are passed as polyline_features, a list of point lists.
    # A line only needs its end points: it is resampled like the boundary.
    decomposition = SkeletonDecomposition.from_boundary(square, polyline_features=lines, target_length=0.5)
    coarse = decomposition.coarse_mesh()
    coarse.set_strips_density_target(0.4)
    dense = coarse.densify()
    print('case', i + 1, ':', coarse.number_of_faces(), 'patches,', dense.number_of_faces(), 'quads')

    # Top: the coarse patches. Bottom: the dense mesh. The lines in red.
    group = viewer.scene.add_group(name='case {}'.format(i + 1))
    for polyline in decomposition.polylines:
        group.add(Polyline(polyline).transformed(Translation.from_vector([13 * i, 13, 0])), linecolor=Color.blue(), linewidth=2)
    group.add(dense.transformed(Translation.from_vector([13 * i, 0, 0])), show_points=False)
    for line in lines:
        group.add(Polyline(line).transformed(Translation.from_vector([13 * i, 0, 0.01])), linecolor=Color.red(), linewidth=4)

viewer.renderer.camera.target.set(18, 11.5, 0)
viewer.renderer.camera.distance = 22
viewer.show()
