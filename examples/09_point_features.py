"""09 - Point features.

A point feature is a point the quad mesh must converge to: a POLE. All the mesh
lines around it run into that one point, like the meridians of a sphere into
its pole. Useful for a column head, an oculus or the apex of a dome.

Around a pole the quads are degenerate: each has two corners on the pole, so it
is really a triangle.

Three cases on the same square: one point in the centre, one point off-centre,
and two points.
"""
from compas.colors import Color
from compas.geometry import Point, Polyline, Translation
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_singular.algorithms import SkeletonDecomposition

square = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]]

cases = [
    [[5.0, 5.0, 0.0]],
    [[3.0, 6.0, 0.0]],
    [[3.0, 5.0, 0.0], [7.0, 5.0, 0.0]],
]

config = Config()
config.renderer.show_grid = False
viewer = Viewer(config=config)
viewer.renderer.view = 'top'

for i, points in enumerate(cases):

    # The points are passed as point_features. Each one becomes a pole: the
    # skeleton is forced through it, and the patches around it are triangles.
    decomposition = SkeletonDecomposition.from_boundary(square, point_features=points, target_length=0.5)
    coarse = decomposition.coarse_mesh()
    coarse.set_strips_density_target(0.4)
    dense = coarse.densify()
    print('case', i + 1, ':', coarse.number_of_faces(), 'patches,', dense.number_of_faces(), 'quads, poles at', [coarse.vertex_coordinates(v) for v in coarse.poles()])

    # Top: the coarse patches. Bottom: the dense mesh. The points in red.
    group = viewer.scene.add_group(name='case {}'.format(i + 1))
    for polyline in decomposition.polylines:
        group.add(Polyline(polyline).transformed(Translation.from_vector([13 * i, 13, 0])), linecolor=Color.blue(), linewidth=2)
    group.add(dense.transformed(Translation.from_vector([13 * i, 0, 0])), show_points=False)
    for point in points:
        group.add(Point(*point).transformed(Translation.from_vector([13 * i, 0, 0.01])), pointcolor=Color.red(), pointsize=15)

viewer.renderer.camera.target.set(18, 11.5, 0)
viewer.renderer.camera.distance = 22
viewer.show()
