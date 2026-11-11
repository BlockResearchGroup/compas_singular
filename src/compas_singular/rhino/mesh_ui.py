"""Rhino-side helpers every editing command shares: prompts, refusals, drags and colours.

Previews show the projected point, and refusals go in a dialog, not a print.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any
from typing import Callable
from typing import Sequence

try:
    import Rhino  # type: ignore  # noqa: I001
    import rhinoscriptsyntax as rs  # type: ignore  # noqa: I001
    from Rhino.Geometry import Line  # type: ignore  # noqa: I001
    from Rhino.Geometry import Point3d  # type: ignore  # noqa: I001

    # ``from System.Drawing.Color import FromArgb`` is an IronPython idiom and
    # raises under Rhino 8's CPython, where Color is a class and not a module.
    # It raised at the END of this block, so everything above it stayed bound
    # and the failure only surfaced later as ``NameError: FromArgb``.
    from System.Drawing import Color  # type: ignore  # noqa: I001

    RHINO = True
except ImportError:  # importable outside Rhino, so the package still imports
    RHINO = False


__all__ = [
    "DEFAULT_COLORS",
    "FINISH",
    "ask",
    "ask_integer",
    "density_colors",
    "drag_point",
    "ensure_layer",
    "refuse",
    "relock",
    "unlock",
]


#: What ``pick_vertex_or_finish`` (on the scene objects) returns on Enter, as
#: distinct from the ``None`` it returns on Esc. A dedicated sentinel rather
#: than a plain string: a vertex key is an int, never this object, so an
#: equality check can never mistake one for the other.
FINISH = object()


#: Colours every hand-edit command draws with. One scheme, so a boundary corner
#: looks the same in the coarse editor and the dense one.
DEFAULT_COLORS = {
    "vertex": (0, 0, 0),
    "vertex.boundary": (0, 120, 200),
    "vertex.special": (200, 0, 120),  # a pole, or a singularity
    "edge": (110, 110, 110),
    "preview": (255, 120, 0),
    "strip": (0, 160, 190),
    "strip.highlight": (255, 120, 0),
    "path": (255, 120, 0),
    "ortho": (255, 100, 100),
    "diagonal": (50, 255, 50),
    "fan": (50, 50, 255),
}

#: The two ends of the density colour scale: the lowest density in the layout is
#: drawn LIGHT, the highest DARK. The ends of matplotlib's 'Blues'.
DENSITY_COLOR_LOW = (198, 219, 239)
DENSITY_COLOR_HIGH = (8, 48, 107)


def density_colors(
    strip_densities: dict[Any, float],
    low: tuple[int, int, int] = DENSITY_COLOR_LOW,
    high: tuple[int, int, int] = DENSITY_COLOR_HIGH,
) -> dict[Any, tuple[int, int, int]]:
    """``{skey: (r, g, b)}``, a blue darkening with density relative to this layout."""
    if not strip_densities:
        return {}
    lo = min(strip_densities.values())
    hi = max(strip_densities.values())
    out = {}
    for skey, d in strip_densities.items():
        t = 0.5 if hi == lo else (d - lo) / float(hi - lo)
        out[skey] = tuple(int(round(low[i] + (high[i] - low[i]) * t)) for i in range(3))
    return out


def _require_rhino() -> None:
    if not RHINO:
        raise RuntimeError("compas_singular.rhino.mesh_ui needs Rhino. The editing operations themselves do not -- see compas_singular.editing.")


# ----------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------


def ensure_layer(path: str, color: tuple[int, int, int] | None = None) -> str:
    """Create a ``::`` layer path, parents first, and return the full path."""
    _require_rhino()
    parts = path.split("::")
    for i in range(len(parts)):
        name = "::".join(parts[: i + 1])
        if not rs.IsLayer(name):
            rs.AddLayer(name=parts[i], parent="::".join(parts[:i]) if i else None, color=color if i == len(parts) - 1 else None)
    return path


def unlock(layer: str, guids: Sequence[Any] = ()) -> dict[str, Any]:
    """Unlock a layer, all its parents, and the given objects; returns what was locked for ``relock``."""
    _require_rhino()
    parts = layer.split("::")
    state = {"layers": {}, "objects": []}
    for i in range(len(parts)):
        name = "::".join(parts[: i + 1])
        if rs.IsLayer(name):
            state["layers"][name] = rs.LayerLocked(name)
            if state["layers"][name]:
                rs.LayerLocked(name, False)
    for guid in guids:
        if rs.IsObject(guid) and rs.IsObjectLocked(guid):
            state["objects"].append(guid)
            rs.UnlockObject(guid)
    return state


def relock(state: dict[str, Any]) -> None:
    """Put back exactly what ``unlock`` took off, and nothing else."""
    _require_rhino()
    for guid in state.get("objects", []):
        if rs.IsObject(guid):
            rs.LockObject(guid)
    for layer, was_locked in state.get("layers", {}).items():
        if was_locked and rs.IsLayer(layer):
            rs.LayerLocked(layer, True)


# ----------------------------------------------------------------------
# asking
# ----------------------------------------------------------------------


def ask(message: str, options: Sequence[str], default: str | None = None) -> str:
    """A command-line menu. Returns the answer lowercased, or ``''`` on Esc.

    Lowercased because every caller compares against a lowercase name, and
    ``rs.GetString`` returns whatever case the option was declared in.
    """
    _require_rhino()
    answer = rs.GetString(message, default if default is not None else options[0], list(options))
    return (answer or "").lower()


def ask_integer(message: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int | None:
    """``rs.GetInteger``, returning ``None`` on Esc rather than a surprise."""
    _require_rhino()
    value = rs.GetInteger(message, default, minimum, maximum)
    return None if value is None else int(value)


def refuse(title: str, message: str) -> None:
    """Tell the user why nothing happened, in a dialog."""
    _require_rhino()
    rs.MessageBox(message, 0, title)


# ----------------------------------------------------------------------
# dragging
# ----------------------------------------------------------------------


def drag_point(
    prompt: str,
    start: Sequence[float],
    neighbours: Sequence[Sequence[float]] = (),
    project: Callable[[Sequence[float]], Sequence[float]] | None = None,
    colors: dict[str, tuple[int, int, int]] | None = None,
) -> list[float] | None:
    """Drag one point with a live preview. ``[x, y, 0.0]``, or ``None`` on Esc.

    Parameters
    ----------
    start : list
        Where the point is now. Used as the drag base point.
    neighbours : list, optional
        Points to rubber-band to, so the mesh is seen deforming rather than one
        dot moving.
    project : callable, optional
        ``xyz -> xyz``, applied to the cursor before anything is drawn. This is
        what makes the preview honest: a boundary vertex is shown where it will
        actually end up, not where the cursor is.

    Constrained to world XY because the pipeline is planar. An unconstrained
    pick lands on whatever construction plane the view happens to have, and the
    point moved would then differ from the one previewed.
    """
    _require_rhino()
    colors = colors or DEFAULT_COLORS
    color = Color.FromArgb(*colors["preview"])
    neighbours = [list(point) for point in neighbours]

    def _at(x: float, y: float) -> Sequence[float]:
        xyz = [x, y, 0.0]
        return project(xyz) if project is not None else xyz

    gp = Rhino.Input.Custom.GetPoint()
    gp.SetCommandPrompt(prompt)
    gp.SetBasePoint(Point3d(*start), True)
    gp.Constrain(Rhino.Geometry.Plane.WorldXY, False)

    def on_dynamic_draw(sender: Any, e: Any) -> None:
        cp = e.CurrentPoint
        point = Point3d(*_at(cp.X, cp.Y))
        for nbr in neighbours:
            e.Display.DrawLine(Line(point, Point3d(*nbr)), color, 2)
        e.Display.DrawPoint(point, color)

    gp.DynamicDraw += on_dynamic_draw
    try:
        gp.Get()
        if gp.CommandResult() != Rhino.Commands.Result.Success:
            return None
        ep = gp.Point()
    finally:
        # Detached explicitly: the handler closes over ``neighbours``, and a
        # GetPoint that outlives the call would keep drawing a stale mesh.
        gp.DynamicDraw -= on_dynamic_draw
    return _at(ep.X, ep.Y)
