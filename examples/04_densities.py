"""04 - Densities.

The density of a strip is the number of quads across it. A strip runs through
several patches, and the patches on either side of an edge share it, so
setting densities per strip always gives a conforming mesh.

Three ways to set densities:

* a target edge length for every strip,
* an exact number for one strip,
* a target number of faces for the whole mesh.
"""
import math

from compas.colors import Color
from compas.geometry import Polyline, Translation
from compas_viewer import Viewer
from compas_viewer.config import Config
from compas_viewer.scene import Tag

from compas_singular.algorithms import SkeletonDecomposition

# The coarse mesh of 01.
outer = [[-6.0, -4.0, 0.0], [6.0, -4.0, 0.0], [6.0, 4.0, 0.0], [-6.0, 4.0, 0.0]]
hole = [[1.6 * math.cos(2 * math.pi * i / 64), 1.6 * math.sin(2 * math.pi * i / 64), 0.0] for i in range(64)]

decomposition = SkeletonDecomposition.from_boundary(outer, inner_boundaries=[hole], target_length=0.5)
coarse = decomposition.coarse_mesh()

# The strips of the coarse mesh. The picture numbers each one.
for skey in coarse.strips():
    print('strip', skey, 'edges', coarse.strip_edges(skey))

# 1. A target edge length: every strip gets the number of quads that makes its
#    edges closest to 0.5 long.
coarse.set_strips_density_target(0.5)
dense_target = coarse.densify().copy()
print('target length   :', dense_target.number_of_faces(), 'quads')

# 2. An exact number for one strip, on top of that. Strip 7 is the ring of
#    patches around the hole: 8 quads across it refines the mesh at the hole.
coarse.set_strip_density(7, 8)
dense_strip = coarse.densify().copy()
print('strip 7 set to 8:', dense_strip.number_of_faces(), 'quads')

# 3. A target number of faces: one equal density on every strip.
coarse.set_mesh_density_face_target(300)
dense_faces = coarse.densify().copy()
print('face target 300 :', dense_faces.number_of_faces(), 'quads')

# Visualisation: numbered strips | target length | one strip refined | face target
config = Config()
config.renderer.show_grid = False
viewer = Viewer(config=config)
viewer.renderer.view = 'top'

group = viewer.scene.add_group(name='coarse patch edges')
for polyline in decomposition.polylines:
    group.add(Polyline(polyline), linecolor=Color.blue(), linewidth=2)
group = viewer.scene.add_group(name='strip numbers')
for skey in coarse.strips():
    u, v = coarse.strip_edges(skey)[len(coarse.strip_edges(skey)) // 2]
    group.add(Tag(str(skey), coarse.edge_midpoint(u, v), color=Color.red(), height=60, horizontal_align='center', vertical_align='center'))

viewer.scene.add(dense_target.transformed(Translation.from_vector([14, 0, 0])), show_points=False, name='target length 0.5')
viewer.scene.add(dense_strip.transformed(Translation.from_vector([28, 0, 0])), show_points=False, name='strip 7 set to 8')
viewer.scene.add(dense_faces.transformed(Translation.from_vector([42, 0, 0])), show_points=False, name='face target 300')

viewer.renderer.camera.target.set(21, 0, 0)
viewer.renderer.camera.distance = 30
viewer.show()
