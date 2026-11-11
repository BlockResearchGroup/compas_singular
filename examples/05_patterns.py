"""05 - Patterns.

Each patch of the coarse mesh can be filled with a different pattern:

* 'ortho'    - a plain grid,
* 'diagonal' - a grid with both diagonals of the patch cut through it,
* 'fan'      - quads fanning out from the corners of the patch.

Here the same square layout is filled with each pattern, side by side.
"""
from compas.geometry import Translation
from compas_viewer import Viewer
from compas_viewer.config import Config

from compas_singular.algorithms import SkeletonDecomposition

# A 10 x 10 square. Its skeleton cuts it into four square patches.
outer = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0]]
coarse = SkeletonDecomposition.from_boundary(outer, target_length=0.5).coarse_mesh()

config = Config()
config.renderer.show_grid = False
viewer = Viewer(config=config)

for i, pattern in enumerate(['ortho', 'diagonal', 'fan']):

    # Every patch 4 x 4, with one pattern for all of them.
    # set_face_pattern(fkey, pattern) sets the pattern of one patch.
    # 'diagonal' and 'fan' need an even density, and raise it if it is not.
    coarse.set_strips_density(4)
    coarse.set_global_face_pattern(pattern)

    # densify() builds the dense mesh with the patterns.
    dense = coarse.densify()

    viewer.scene.add(dense.transformed(Translation.from_vector([11 * i, 0, 0])), show_points=False, name=pattern)

viewer.renderer.camera.target.set(14, 5, 0)
viewer.renderer.camera.position.set(14, -18, 22)
viewer.show()
