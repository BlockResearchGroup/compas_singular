"""13 - From quad mesh to blocks: the dual mesh.

For masonry or other block structures, the quad mesh is read the other way
round: each vertex of the quad mesh becomes a block, and each edge a joint
between two blocks. That is the DUAL mesh.

A regular vertex (4 edges) becomes a quad block. A singular vertex becomes an
n-gon block: here the four 5-valent vertices of the mesh of 01 become four
pentagonal blocks.

The plain dual stops half-way to the boundary, so dual_mesh() closes it with a
layer of boundary blocks. Those come out half-sized; redistribute=True evens
the blocks along the boundary out.
"""
import math

from compas.colors import Color
from compas.datastructures import Mesh
from compas.geometry import Point, Translation
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.algorithms.dual_mesh import dual_mesh

# The mesh of 01, coarser so the blocks are easy to see.
outer = [[-6.0, -4.0, 0.0], [6.0, -4.0, 0.0], [6.0, 4.0, 0.0], [-6.0, 4.0, 0.0]]
hole = [[1.6 * math.cos(2 * math.pi * i / 64), 1.6 * math.sin(2 * math.pi * i / 64), 0.0] for i in range(64)]

decomposition = SkeletonDecomposition.from_boundary(outer, inner_boundaries=[hole], target_length=0.5)
coarse = decomposition.coarse_mesh()
coarse.set_strips_density_target(0.8)
dense = coarse.densify()

# The singular vertices: interior vertices without 4 edges.
singular = [v for v in dense.vertices() if not dense.is_vertex_on_boundary(v) and dense.vertex_degree(v) != 4]
print('singular vertices :', len(singular), 'with', [dense.vertex_degree(v) for v in singular], 'edges')

# The dual wants a plain COMPAS mesh.
plain = Mesh.from_vertices_and_faces(*dense.to_vertices_and_faces(keep_keys=False))
blocks = dual_mesh(plain, redistribute=False)
even_blocks = dual_mesh(plain, redistribute=True)
for name, mesh in [('dual', blocks), ('redistributed', even_blocks)]:
    boundary = [mesh.face_area(f) for f in mesh.faces() if mesh.is_face_on_boundary(f)]
    interior = [mesh.face_area(f) for f in mesh.faces() if not mesh.is_face_on_boundary(f)]
    print('{:<17} : mean block area {:.3f} on the boundary, {:.3f} inside'.format(name, sum(boundary) / len(boundary), sum(interior) / len(interior)))
ngons = [len(even_blocks.face_vertices(f)) for f in even_blocks.faces() if len(even_blocks.face_vertices(f)) != 4]
print('blocks            :', even_blocks.number_of_faces(), ', of which n-gons:', ngons)

# Visualisation: quad mesh | dual | dual, redistributed. The n-gon blocks in red.
config = Config()
config.renderer.show_grid = False
viewer = Viewer(config=config)
viewer.renderer.view = 'top'

viewer.scene.add(dense, show_points=False, name='quad mesh')
group = viewer.scene.add_group(name='singular vertices')
for v in singular:
    group.add(Point(*dense.vertex_coordinates(v)).transformed(Translation.from_vector([0, 0, 0.01])), pointcolor=Color.red(), pointsize=12)

colors = {f: Color.red() if len(blocks.face_vertices(f)) != 4 else Color.white() for f in blocks.faces()}
viewer.scene.add(blocks.transformed(Translation.from_vector([14, 0, 0])), facecolor=colors, show_points=False, name='dual')

colors = {f: Color.red() if len(even_blocks.face_vertices(f)) != 4 else Color.white() for f in even_blocks.faces()}
viewer.scene.add(even_blocks.transformed(Translation.from_vector([28, 0, 0])), facecolor=colors, show_points=False, name='dual, redistributed')

viewer.renderer.camera.target.set(14, 0, 0)
viewer.renderer.camera.distance = 22
viewer.show()
