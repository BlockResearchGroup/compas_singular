"""02 - The basic frame-field workflow.

Boundary -> cross field -> coarse quad layout -> densities -> dense quad mesh.

The twin of 01: the same domain, only the coarse layout is made differently.
Instead of a skeleton, a CROSS FIELD is solved on a triangulation of the domain.
Its singularities become the poles of the quad mesh, and the lines traced out of
them (separatrices) cut the domain into quad patches. The field is used once more
when the patches are filled, so the dense quads follow it everywhere.
"""
import math

from compas.colors import Color
from compas.geometry import Polyline, Translation
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_singular.framefield.field_decomposition import FieldDecomposition
from compas_singular.framefield.viz import add_field, add_singularities

# The domain: a rectangle with a circular hole.
outer = [[-6.0, -4.0, 0.0], [6.0, -4.0, 0.0], [6.0, 4.0, 0.0], [-6.0, 4.0, 0.0]]
hole = [[1.6 * math.cos(2 * math.pi * i / 64), 1.6 * math.sin(2 * math.pi * i / 64), 0.0] for i in range(64)]

# The cross field. target_length is the spacing of the triangulation the field
# is solved on, not the quad size: finer is more accurate and slower.
# relax=True gives more accurate singularity positions than the default solve.
decomposition = FieldDecomposition.from_boundary(outer, inner_boundaries=[hole], target_length=0.5, relax=True)

# The coarse quad mesh: the separatrices cut the domain into quad patches.
coarse = decomposition.coarse_mesh()
print('coarse mesh :', coarse.number_of_faces(), 'patches')

# Always read the warnings: a layout is returned even when something was repaired.
for warning in decomposition.warnings():
    print('warning     :', warning)

# The densities, as in 01.
coarse.collect_strips()
coarse.set_strips_density_target(0.4)

# The dense quad mesh. Passing the field fills each patch interior along the
# field, instead of blending it from its four sides. edges_to_curves gives the
# patch edges the curved shape of the separatrices and walls they came from.
# coarse.quad_mesh() and coarse.densification() take the same arguments.
dense = coarse.densify(overwrite_edges_to_curves=decomposition.edges_to_curves(), field=decomposition.get_field())
print('dense mesh  :', dense.number_of_faces(), 'quads')
print('quality     :', decomposition.quality(dense))

# Visualisation: cross field + singularities | coarse mesh | dense mesh
config = Config()
config.renderer.show_grid = False
viewer = Viewer(config=config)
viewer.renderer.view = 'top'

group = viewer.scene.add_group(name='cross field')
add_field(group, decomposition)
add_singularities(group, decomposition)

move = Translation.from_vector([14, 0, 0])
viewer.scene.add(coarse.transformed(move), show_faces=False, show_lines=False, name='coarse mesh')
group = viewer.scene.add_group(name='coarse patch edges')
for polyline in decomposition.decomposition_polylines():
    group.add(Polyline(polyline).transformed(move), linecolor=Color.blue(), linewidth=2)

viewer.scene.add(dense.transformed(Translation.from_vector([28, 0, 0])), show_points=False, name='dense mesh')

viewer.renderer.camera.target.set(14, 0, 0)
viewer.renderer.camera.distance = 22
viewer.show()
