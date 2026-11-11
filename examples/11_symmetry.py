"""11 - Symmetry.

A symmetric domain does not always give a symmetric mesh: the decomposition can
break the symmetry. To get a mesh that is symmetric by construction:

1. find the symmetry of the input,
2. cut out one symmetric unit and make its coarse mesh,
3. densify only the unit,
4. rotate and mirror the unit into the whole mesh.

This works on both routes. Here: the skeleton route, on a square with four
point features (09).
"""
from compas.colors import Color
from compas.geometry import Point, Polyline, Translation
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.symmetry.measure import mesh_invariance

square = [[-5.0, -5.0, 0.0], [5.0, -5.0, 0.0], [5.0, 5.0, 0.0], [-5.0, 5.0, 0.0]]
poles = [[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [2.0, 2.0, 0.0], [-2.0, 2.0, 0.0]]

# Without symmetry, for comparison.
reference = SkeletonDecomposition.from_boundary(square, point_features=poles, target_length=0.5)
reference_coarse = reference.coarse_mesh()
reference_coarse.set_strips_density_target(0.4)
reference_dense = reference_coarse.densify()

# 1. Find the symmetry of the input: the square with its poles is D4, four
#    rotations and four mirrors.
decomposition = SkeletonDecomposition.from_boundary(square, point_features=poles, target_length=0.5)
report = decomposition.find_symmetry()
print(report.summary())

# 2. Cut out one unit. The poles lie on the diagonal mirrors, and a pole cannot
#    lie on the cut, so only the two axis mirrors M0 and M90 are used: the unit
#    is a quarter of the square. check() says whether the unit can be expanded.
unit = decomposition.symmetry_unit(keys=['M0', 'M90'])
print(unit, 'check:', unit.check())

# 3. Densities and dense mesh of the unit only. The unit uses quad_mesh(), which
#    does the same as densify() but keeps the symmetry data needed to expand it.
unit.collect_strips()
unit.set_strips_density_target(0.4)
dense_unit = unit.quad_mesh()

# 4. Expand the unit, and its dense mesh, into the whole.
coarse = unit.expand_symmetrically()
dense = dense_unit.expand_symmetrically(unit)

# The share of vertices that land on a vertex when mirrored: 1.0 is symmetric.
print('without symmetry :', mesh_invariance(reference_dense, unit.group)['share'])
print('with symmetry    :', mesh_invariance(dense, unit.group)['share'])

# Visualisation: without symmetry | unit | expanded coarse mesh | expanded dense mesh
config = Config()
config.renderer.show_grid = False
viewer = Viewer(config=config)
viewer.renderer.view = 'top'

viewer.scene.add(reference_dense, show_points=False, name='without symmetry')
viewer.scene.add(unit.transformed(Translation.from_vector([13, 0, 0])), facecolor=Color(1.0, 0.85, 0.85), show_points=False, name='unit')
viewer.scene.add(coarse.transformed(Translation.from_vector([26, 0, 0])), show_points=False, name='expanded coarse mesh')
viewer.scene.add(dense.transformed(Translation.from_vector([39, 0, 0])), show_points=False, name='expanded dense mesh')
group = viewer.scene.add_group(name='square and poles')
group.add(Polyline(square + square[:1]).transformed(Translation.from_vector([13, 0, 0])), linecolor=Color.grey(), linewidth=2)
for pole in poles:
    group.add(Point(*pole).transformed(Translation.from_vector([13, 0, 0.01])), pointcolor=Color.red(), pointsize=12)

viewer.renderer.camera.target.set(19.5, 0, 0)
viewer.renderer.camera.distance = 28
viewer.show()
