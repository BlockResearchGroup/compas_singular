"""Symmetry for either route: find the symmetry of a domain, mesh one unit, and expand it by the group.

Design notes: ``design_notes/symmetry.md``.
"""

from __future__ import absolute_import
from __future__ import annotations

from typing import Any

from compas_singular.symmetry.group import SymmetryGroup  # noqa: F401
from compas_singular.symmetry.domain import Domain  # noqa: F401
from compas_singular.symmetry.report import SymmetryReport  # noqa: F401
from compas_singular.symmetry.detect import find_symmetry  # noqa: F401


def __getattr__(name: str) -> Any:
    # The unit classes import compas_singular.datastructures, which is heavier and
    # would make ``compas_singular.symmetry`` impossible to import from inside
    # ``compas_singular.datastructures``' own initialisation. Resolve them lazily.
    if name in ("SymmetricUnit", "SymmetricQuadUnit", "build_unit"):
        from compas_singular.symmetry import unit

        return getattr(unit, name)
    raise AttributeError(name)


__all__ = ["find_symmetry", "SymmetryReport", "SymmetryGroup", "Domain", "SymmetricUnit", "SymmetricQuadUnit", "build_unit"]
