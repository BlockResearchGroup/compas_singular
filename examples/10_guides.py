"""10 - Guides on the frame-field route.

A guide is a curve the quads should run along: a cable, a force line, a
direction of span. It is only available on the frame-field route (02): the
cross field is aligned with the guide near it, and the dense mesh follows the
field.

Unlike a curve feature (08), a guide does not cut the domain and is not a row of
mesh edges. It steers the directions of the mesh around it.

Three cases:

* a rectangle without a guide,
* the same rectangle with a curved guide,
* the rectangle with a hole, with a ring-shaped guide around the hole.
"""
import math

from compas.colors import Color
from compas.geometry import Polyline, Translation
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_singular.framefield.field_decomposition import FieldDecomposition
from compas_singular.framefield.viz import add_field

outer = [[-6.0, -4.0, 0.0], [6.0, -4.0, 0.0], [6.0, 4.0, 0.0], [-6.0, 4.0, 0.0]]
hole = [[1.6 * math.cos(2 * math.pi * i / 64), 1.6 * math.sin(2 * math.pi * i / 64), 0.0] for i in range(64)]

# A sine-shaped guide from the left wall to the right wall, and a ring around the hole.
guide = [[-6.0 + 0.5 * i, 2.5 * math.sin(math.pi * (-6.0 + 0.5 * i) / 12), 0.0] for i in range(25)]
ring = [[2.8 * math.cos(2 * math.pi * i / 48), 2.8 * math.sin(2 * math.pi * i / 48), 0.0] for i in range(49)]

cases = [
    ([], []),
    ([], [guide]),
    ([hole], [ring]),
]

config = Config()
config.renderer.show_grid = False
viewer = Viewer(config=config)
viewer.renderer.view = 'top'

for i, (holes, guides) in enumerate(cases):

    # The guides are a list of point lists, like the holes.
    decomposition = FieldDecomposition.from_boundary(outer, inner_boundaries=holes, guides=guides, target_length=0.5, relax=True)
    coarse = decomposition.coarse_mesh()
    coarse.collect_strips()
    coarse.set_strips_density_target(0.4)
    dense = coarse.densify(overwrite_edges_to_curves=decomposition.edges_to_curves(), field=decomposition.get_field())
    print('case', i + 1, ':', coarse.number_of_faces(), 'patches,', dense.number_of_faces(), 'quads, warnings', decomposition.warnings())

    # Top: the cross field. Bottom: the dense mesh. The guide in red.
    group = viewer.scene.add_group(name='case {}'.format(i + 1))
    add_field(group, decomposition, 14 * i, 10)
    group.add(dense.transformed(Translation.from_vector([14 * i, 0, 0])), show_points=False)
    for line in guides:
        group.add(Polyline(line).transformed(Translation.from_vector([14 * i, 10, 0.01])), linecolor=Color.red(), linewidth=4)
        group.add(Polyline(line).transformed(Translation.from_vector([14 * i, 0, 0.01])), linecolor=Color.red(), linewidth=4)

viewer.renderer.camera.target.set(14, 5, 0)
viewer.renderer.camera.distance = 22
viewer.show()
