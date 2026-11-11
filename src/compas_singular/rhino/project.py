"""What the ``CMD_`` commands share in Rhino: settings, the layer tree and the coarse layout.

Importing it does nothing, and the Rhino-free parts import headless.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas_singular.settings import Settings

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarsePseudoQuadMesh

try:
    import rhinoscriptsyntax as rs  # type: ignore  # noqa: I001
except ImportError:
    rs = None


__all__ = [
    "DEFAULT_SETTINGS",
    "get_settings",
    "set_settings",
    "resolve_relax",
    "resolve_field_symmetry",
    "resolve_spacing",
    "THESIS_ALPHA",
    "resolve_densities",
    "has_closed_guide",
    "ROOT",
    "LAYER_DATA",
    "layer_path",
    "ensure_layers",
    "read_layout",
]


def _require_rhino() -> None:
    if rs is None:
        raise RuntimeError("compas_singular.rhino.project: this function needs Rhino (rhinoscriptsyntax is not importable here).")


# ==============================================================================
# settings
# ==============================================================================

#: Every setting at its default, as the dict ``get_settings`` returns.
DEFAULT_SETTINGS = Settings().model_dump()

#: The ``alpha`` of thesis eq. 4.1, the default of both decompositions'
#: ``from_boundary`` and of ``discretise_boundary``.
THESIS_ALPHA = 0.02


def get_settings() -> dict[str, Any]:
    """The session's settings, as a dict a command can change and hand to ``set_settings``."""
    _require_rhino()
    from compas_singular.rhino.session import RhinoSession

    return RhinoSession.current().settings.model_dump()


def set_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Store ``settings`` (a dict) in the session, and record it. Returns them as stored."""
    _require_rhino()
    from compas_singular.rhino.session import RhinoSession

    session = RhinoSession.current()
    session.settings = Settings.model_validate(settings)
    session.record("Settings")
    return session.settings.model_dump()


def resolve_relax(settings: dict[str, Any], guides: Sequence[Any]) -> bool:
    """Whether to solve with diffusion + normalisation, for THIS input."""
    value = settings.get("relax", "auto")
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "auto":
            return bool(guides)
        return value in ("on", "yes", "true", "1")
    return bool(value)


def resolve_spacing(settings: dict[str, Any], loops: Sequence[Sequence[Sequence[float]]] | None = None) -> float:
    """The background spacing as a number for this document, from settings or thesis eq. 4.1.

    For the Rhino side only; pass the setting itself to the decompositions.
    """
    value = settings.get("triangulation_spacing")
    if value is not None:
        return float(value)
    from compas_singular.geometry.polyline import bounding_box_diagonal

    if loops is None:
        _require_rhino()
        guids = list(rs.ObjectsByLayer("Outer") or []) + list(rs.ObjectsByLayer("Inner") or [])
        box = rs.BoundingBox(guids) if guids else None
        if not box:
            raise RuntimeError("No outer boundary selected -- the thesis spacing is measured on it. Set Background_Spacing instead.")
        loops = [[[p.X, p.Y, p.Z] for p in box]]
    return THESIS_ALPHA * bounding_box_diagonal(*loops)


def has_closed_guide() -> bool:
    """Is any curve on the Guides layer closed? A closed guide is the one case
    measured where relaxation makes the layout WORSE (13 patches -> 1)."""
    _require_rhino()
    for guid in rs.ObjectsByLayer(layer_path("Guides")) or []:
        if rs.IsCurveClosed(guid):
            return True
    return False


def resolve_field_symmetry(settings: dict[str, Any]) -> str | None:
    """The symmetry group to solve the field under for this document (``'auto'`` or disabled)."""
    value = settings.get("field_symmetry", "auto")
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ("", "none", "off", "no", "false", "0"):
            return None
        return "auto"
    return None if not value else "auto"


def resolve_densities(coarse: CoarsePseudoQuadMesh, settings: dict[str, Any], verbose: bool = True) -> bool:
    """Keep the layout's saved densities if every strip has one, else apply the global rule.

    Returns True when the densities had to be re-derived.

    Returns
    -------
    bool
        True when the densities had to be re-derived.
    """
    if coarse.has_densities():
        return False
    coarse.attributes["strips_density"] = {}
    if settings.get("density_mode") == "density":
        coarse.set_strips_density(int(settings["target_density"]))
    else:
        coarse.set_strips_density_target(settings["target_length"])
    if verbose:
        print("densities: none saved on the layout -- every strip set by the target {}".format(settings.get("density_mode", "length")))
    return True


# ==============================================================================
# layers
# ==============================================================================

ROOT = "TopologyProblem"

#: Short name -> (full layer path, colour or None). ``CMD_start`` creates every
#: one of these, parents first, in this order.
LAYER_DATA = {
    "Problem": (ROOT, None),
    "Input": (ROOT + "::InputBoundaries", None),
    "Outer": (ROOT + "::InputBoundaries::Outer", (255, 0, 0)),
    "Inner": (ROOT + "::InputBoundaries::Inner", (0, 255, 0)),
    "Guides": (ROOT + "::InputBoundaries::Guides", (255, 127, 0)),
    "PointFeatures": (ROOT + "::InputBoundaries::PointFeatures", (0, 255, 0)),
    "Skeleton": (ROOT + "::Skeleton", None),
    "Poles": (ROOT + "::Skeleton::Poles", None),
    "Polylines": (ROOT + "::Skeleton::Polylines", None),
    "Mesh": (ROOT + "::Skeleton::Mesh", None),
    "EdgeCurves": (ROOT + "::Skeleton::EdgeCurves", (0, 120, 200)),
    "TempEdit": (ROOT + "::Skeleton::TempEdit", (0, 120, 200)),
    "Densities": (ROOT + "::Attributes::Densities", None),
    "Patterns": (ROOT + "::Attributes::Patterns", None),
    "QuadMesh": (ROOT + "::QuadMesh", None),
    "Dense": (ROOT + "::QuadMesh::Dense", None),
    "Dual": (ROOT + "::QuadMesh::Dual", None),
    "Smoothed": (ROOT + "::QuadMesh::Smoothed", None),
    "Area": (ROOT + "::QuadMesh::Smoothed::Area", None),
}


def layer_path(name: str) -> str:
    """The full ``::`` path of a project layer by its short name; use it with ``rhinoscriptsyntax``."""
    return LAYER_DATA[name][0]


def ensure_layers() -> None:
    """Create every project layer that does not exist yet. Deletes nothing."""
    _require_rhino()
    for name, color in LAYER_DATA.values():
        if not rs.IsLayer(name):
            rs.AddLayer(name=name, color=color)


# ==============================================================================
# the layout
# ==============================================================================


def read_layout(verbose: bool = True) -> CoarsePseudoQuadMesh:
    """A copy of the session's coarse layout; raises if the session holds none.

    Raises
    ------
    RuntimeError
        If the session holds no layout. The mesh drawn on ``Skeleton::Mesh`` is
        NOT read instead: it is vertices and faces only, a view of the layout.
    """
    _require_rhino()
    from compas_singular.rhino.session import RhinoSession

    layout = RhinoSession.current().coarse
    if layout is None:
        raise RuntimeError("This document holds no coarse layout. Run CMD_coarse_mesh or CMD_read_coarse_mesh first.")
    coarse = layout.copy()
    if verbose:
        print("layout: {} patch(es), route {!r}".format(coarse.number_of_faces(), coarse.attributes.get("route", "unknown")))
    return coarse
