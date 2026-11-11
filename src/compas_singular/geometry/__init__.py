"""Geometry functions on plain point lists: arrays of points, and projection onto and discretisation of polylines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import annotations

from compas_singular.geometry.array import *  # noqa: F401 F403
from compas_singular.geometry.polyline import *  # noqa: F401 F403

import types  # noqa: E402

# Only re-export names bound in this namespace, never the submodules themselves:
# ``from .foo import *`` also binds ``foo`` as an attribute of this package, and
# re-exporting that module object shadows any function or subpackage of the same
# name further up the chain.
__all__ = [name for name, obj in list(globals().items()) if not name.startswith("_") and not isinstance(obj, types.ModuleType)]
