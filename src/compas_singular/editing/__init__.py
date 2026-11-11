"""Hand-editing a coarse layout or a dense mesh, with no CAD in it. Rhino supplies only picks and prompts.

Imports nothing from ``framefield``. Design notes: ``design_notes/editing.md``.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

# The shared base. Imported first: the editors below are its subclasses.
from compas_singular.editing.editor import MeshEditor  # noqa: F401

from compas_singular.editing.coarseeditor import *  # noqa: F401 F403
from compas_singular.editing.denseeditor import *  # noqa: F401 F403
from compas_singular.editing.guide_chain import *  # noqa: F401 F403

# NOT star-imported: ``rebuild`` also defines ``PRECISION``, which would rebind
# ``coarse_layout``'s. Same value today, and that is exactly why a clash here
# would go unnoticed.
from compas_singular.editing.rebuild import coarse_from_skeleton  # noqa: F401
from compas_singular.editing.rebuild import face_polylines  # noqa: F401
from compas_singular.editing.rebuild import faces_from_geometry  # noqa: F401
from compas_singular.editing.rebuild import mesh_from_faces  # noqa: F401
from compas_singular.editing.rebuild import snap_to_loops  # noqa: F401
from compas_singular.editing.rebuild import warp_polyline  # noqa: F401
from compas_singular.editing.repair import densifiable  # noqa: F401
from compas_singular.editing.repair import solve_non_quad_faces  # noqa: F401
from compas_singular.editing.repair import topological_quad_split  # noqa: F401

import types  # noqa: E402

# Never re-export the submodules themselves: ``from .foo import *`` binds ``foo``
# here too, and a module object of the same name shadows a function further up.
__all__ = [name for name, obj in list(globals().items()) if not name.startswith("_") and not isinstance(obj, types.ModuleType)]
