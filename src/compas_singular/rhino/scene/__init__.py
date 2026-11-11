"""compas scene objects that draw and pick the session's coarse layout and dense mesh in Rhino.

``ensure_registered`` fills the scene registry, also after a reload.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

from compas.plugins import plugin


__all__ = ["register_scene_objects_rhino", "ensure_registered"]


@plugin(category="factories", pluggable_name="register_scene_objects", requires=["Rhino"])
def register_scene_objects_rhino() -> None:
    from compas.scene.context import register

    from compas_singular.datastructures import CoarsePseudoQuadMesh
    from compas_singular.datastructures import QuadMesh

    from compas_singular.rhino.scene.coarseobject import RhinoCoarseObject
    from compas_singular.rhino.scene.denseobject import RhinoDenseObject

    # The registry follows the item's class hierarchy, nearest first: a layout
    # is a QuadMesh too, and still gets RhinoCoarseObject.
    register(CoarsePseudoQuadMesh, RhinoCoarseObject, context="Rhino")
    register(QuadMesh, RhinoDenseObject, context="Rhino")


def ensure_registered() -> None:
    """compas's own scene objects, then ours from THIS import. Safe to call every time."""
    from compas.scene.context import ITEM_SCENEOBJECT
    from compas.scene.context import register_scene_objects

    if not ITEM_SCENEOBJECT:
        register_scene_objects()
    register_scene_objects_rhino()
