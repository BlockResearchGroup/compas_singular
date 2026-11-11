"""Frame-field front end: solve a cross field, trace separatrices into a coarse layout, densify along the field.

Entry point ``FieldDecomposition``; ``CrossField.from_boundary`` alone gives a field for any layout.
"""

from compas_singular.framefield.background import BackgroundMesh  # noqa: F401
from compas_singular.framefield.constraints import Constraint  # noqa: F401
from compas_singular.framefield.constraints import from_boundary  # noqa: F401
from compas_singular.framefield.constraints import from_curves  # noqa: F401
from compas_singular.framefield.densify import field_densification  # noqa: F401
from compas_singular.framefield.field import CrossField  # noqa: F401
from compas_singular.framefield.field import FieldInputs  # noqa: F401
from compas_singular.framefield.guides import guide_metrics  # noqa: F401
from compas_singular.framefield.locator import PointLocator  # noqa: F401
from compas_singular.framefield.quality import curve_alignment  # noqa: F401
from compas_singular.framefield.quality import hard_floor  # noqa: F401
from compas_singular.framefield.quality import mesh_quality  # noqa: F401
from compas_singular.framefield.relax import relax_mesh  # noqa: F401
from compas_singular.framefield.symmetry import Symmetry  # noqa: F401
from compas_singular.framefield.field_decomposition import FieldDecomposition  # noqa: F401

__all__ = [
    "FieldDecomposition",
    "BackgroundMesh",
    "Constraint",
    "from_boundary",
    "from_curves",
    "CrossField",
    "FieldInputs",
    "PointLocator",
    "guide_metrics",
    "relax_mesh",
    "field_densification",
    "mesh_quality",
    "curve_alignment",
    "hard_floor",
    "Symmetry",
]
