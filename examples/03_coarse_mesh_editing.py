"""03 - Editing the coarse quad mesh.

The coarse mesh decides the topology of the final quad mesh: where its poles
are and which way its quads run. It can be edited before it is densified.

All edits work on STRIPS: a strip is a chain of patches connected through
opposite edges. Adding, removing or dividing whole strips is what keeps every
patch a quad. The edits here:

* add a strip along a chain of corners (a polyedge),
* divide a strip along a drawn line,
* remove a strip.
"""
import math

from compas.colors import Color
from compas.geometry import Polyline, Translation
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.editing import CoarseEditor

# The coarse mesh of 01.
outer = [[-6.0, -4.0, 0.0], [6.0, -4.0, 0.0], [6.0, 4.0, 0.0], [-6.0, 4.0, 0.0]]
hole = [[1.6 * math.cos(2 * math.pi * i / 64), 1.6 * math.sin(2 * math.pi * i / 64), 0.0] for i in range(64)]

decomposition = SkeletonDecomposition.from_boundary(outer, inner_boundaries=[hole], target_length=0.5)
coarse = decomposition.coarse_mesh()

# Densify the original once, to compare with at the end.
coarse.set_strips_density_target(0.4)
original_coarse = coarse.copy()
original_dense = coarse.densify()

# The editor works on a copy. It needs the domain walls, so corners on the
# boundary stay on it, and the curved patch edges, so they stay curved.
editor = CoarseEditor(coarse, loops=[outer, hole], polylines=decomposition.polylines)

# Corners and edges are picked by a point near them.
# 1. Add a strip along the chain of corners from the top wall to the bottom wall.
polyedge = [editor.locate(point)[1] for point in [[-3.5, 4.0, 0.0], [-3.63, 1.63, 0.0], [-3.63, -1.63, 0.0], [-3.5, -4.0, 0.0]]]
ok, notes = editor.add_strip(polyedge)
print('add strip    :', ok, notes)
after_add = [Polyline(curve) for curve in editor.edge_curves()[0].values()]

# 2. Divide a strip along a line. The line must start and end on the coarse mesh
#    and cross the patches of one strip, through opposite edges.
line = [[4.75, -4.0, 0.0], [4.75, 4.0, 0.0]]
ok, notes = editor.divide(points=line)
print('divide strip :', ok, notes)
after_divide = [Polyline(curve) for curve in editor.edge_curves()[0].values()]

# 3. Remove the strip through an edge: here the ring of patches around the hole.
#    plan_strip_deletion(edge) reports the same, without changing anything.
kind, edge, point = editor.locate([2.55, -1.12, 0.0])
ok, notes = editor.remove_strip(edge=edge)
print('remove strip :', ok, 'patches', notes['faces_before'], '->', notes['faces_after'])

# A refused edit returns ok=False with the reason, and changes nothing.
# commit() writes the edits back into the coarse mesh.
coarse, notes = editor.commit()
print('commit       :', notes['faces_in'], '->', notes['faces_out'], 'patches')

# The edited mesh has new strips, so densities are set again. The editor keeps
# the curved shape of every patch edge, including the ones that moved.
coarse.collect_strips()
coarse.set_strips_density_target(0.4)
edges_to_curves, tally = editor.edge_curves()
dense = coarse.densify(overwrite_edges_to_curves=edges_to_curves)

# Visualisation. Top: the coarse mesh before and after each edit.
# Bottom: the dense mesh before and after all edits.
config = Config()
config.renderer.show_grid = False
viewer = Viewer(config=config)
viewer.renderer.view = 'top'

group = viewer.scene.add_group(name='original')
for polyline in decomposition.polylines:
    group.add(Polyline(polyline), linecolor=Color.blue(), linewidth=2)

move = Translation.from_vector([14, 0, 0])
group = viewer.scene.add_group(name='1. strip added')
for polyline in after_add:
    group.add(polyline.transformed(move), linecolor=Color.blue(), linewidth=2)
group.add(Polyline(original_coarse.vertices_attributes('xyz', keys=polyedge)).transformed(move), linecolor=Color.red(), linewidth=4)

move = Translation.from_vector([28, 0, 0])
group = viewer.scene.add_group(name='2. strip divided')
for polyline in after_divide:
    group.add(polyline.transformed(move), linecolor=Color.blue(), linewidth=2)
group.add(Polyline(line).transformed(move), linecolor=Color.red(), linewidth=4)

move = Translation.from_vector([42, 0, 0])
group = viewer.scene.add_group(name='3. strip removed')
for curve in edges_to_curves.values():
    group.add(Polyline(curve).transformed(move), linecolor=Color.blue(), linewidth=2)

viewer.scene.add(original_dense.transformed(Translation.from_vector([0, -10, 0])), show_points=False, name='original dense mesh')
viewer.scene.add(dense.transformed(Translation.from_vector([42, -10, 0])), show_points=False, name='edited dense mesh')

viewer.renderer.camera.target.set(21, -5, 0)
viewer.renderer.camera.distance = 30
viewer.show()
