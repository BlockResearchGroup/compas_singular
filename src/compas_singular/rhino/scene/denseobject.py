"""The dense quad mesh (``QuadMesh``) in Rhino."""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any

from compas_singular.rhino.scene.meshobject import RhinoSingularMeshObject

__all__ = ["RhinoDenseObject"]


class RhinoDenseObject(RhinoSingularMeshObject):
    """A dense mesh as pickable points and lines, updated with ``sync`` rather than a full redraw."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("show_faces", False)
        kwargs.setdefault("show_vertices", True)
        kwargs.setdefault("show_edges", True)
        super(RhinoDenseObject, self).__init__(**kwargs)
