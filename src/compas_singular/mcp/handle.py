"""Name mesh places by rounded position (handles) instead of vertex keys, which edits renumber.

Regions are selected by description via ``select``, never enumerated.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence

if TYPE_CHECKING:
    from compas.datastructures import Mesh


__all__ = [
    "PRECISION",
    "vertex_handle",
    "handles_of",
    "resolve",
    "select",
    "describe_selector",
    "SelectorError",
]


#: Decimals kept in a handle. Three is a tenth of a millimetre in metres, and
#: matches the geometric-key precision the rest of the library rounds to.
PRECISION = 3

#: How far from a handle a vertex may be and still answer to it, as a multiple
#: of the rounding step. Slightly over half a step, so a vertex that drifted
#: across a rounding boundary is still found.
RESOLVE_STEPS = 0.75


class SelectorError(ValueError):
    """A region selector this module does not understand."""


def vertex_handle(xyz: Sequence[float], precision: int = PRECISION) -> str:
    """``[x, y, z]`` as a stable, readable name.

    Returns
    -------
    str
    """
    pattern = "v:{0:." + str(precision) + "f},{1:." + str(precision) + "f},{2:." + str(precision) + "f}"
    # ``+ 0.0`` normalises -0.0 to 0.0, so a vertex on an axis does not get two
    # different names depending on which side it was approached from.
    return pattern.format(xyz[0] + 0.0, xyz[1] + 0.0, xyz[2] + 0.0)


def handles_of(mesh: Mesh, vertices: Iterable[Any] | None = None) -> dict[Any, str]:
    """``{vertex key: handle}`` for the given vertices, or for all of them."""
    keys = list(mesh.vertices()) if vertices is None else list(vertices)
    return dict((key, vertex_handle(mesh.vertex_coordinates(key))) for key in keys)


def _parse(handle: Any) -> list[float] | None:
    """A handle back to ``[x, y, z]``. ``None`` if it is not one."""
    if not isinstance(handle, str):
        return None
    text = handle.strip()
    if text.startswith("v:"):
        text = text[2:]
    parts = text.replace(" ", "").split(",")
    if len(parts) != 3:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def resolve(mesh: Mesh, handle: str) -> tuple[Any, str]:
    """Find the vertex a handle names: exact, else nearest within tolerance. ``(key, how)`` or ``(None, reason)``.

    Parameters
    ----------
    mesh : Mesh
    handle : str

    Returns
    -------
    tuple
        ``(vertex key, how)`` where ``how`` is ``'exact'`` or ``'nearest'``, or
        ``(None, reason)`` when nothing is close enough.
    """
    point = _parse(handle)
    if point is None:
        return None, ("{!r} is not a vertex handle -- they look like v:12.500,4.250,0.000".format(handle))

    wanted = vertex_handle(point)
    best, best_distance = None, None
    for key in mesh.vertices():
        xyz = mesh.vertex_coordinates(key)
        if vertex_handle(xyz) == wanted:
            return key, "exact"
        distance = _distance(xyz, point)
        if best_distance is None or distance < best_distance:
            best, best_distance = key, distance

    tolerance = RESOLVE_STEPS * (10.0**-PRECISION)
    if best is not None and best_distance <= tolerance:
        return best, "nearest"
    if best is None:
        return None, "the mesh has no vertices"
    return None, ("no vertex at {} -- the nearest is {:.4f} away, at {}".format(handle, best_distance, vertex_handle(mesh.vertex_coordinates(best))))


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


# ==============================================================================
# regions
# ==============================================================================


def _boundary_vertices(mesh: Mesh) -> set:
    return set(mesh.vertices_on_boundary())


def _singular_vertices(mesh: Mesh) -> set:
    """Irregular interior vertices and poles."""
    test = getattr(mesh, "is_vertex_singular", None)
    if test is not None:
        try:
            return set(key for key in mesh.vertices() if test(key))
        except Exception:
            pass
    boundary = _boundary_vertices(mesh)
    return set(key for key in mesh.vertices() if key not in boundary and len(mesh.vertex_neighbors(key)) != 4)


def _grow(mesh: Mesh, seed: Iterable[Any], rings: int) -> set:
    """``seed`` plus ``rings`` rings of neighbours around it."""
    out = set(seed)
    frontier = set(seed)
    for _ in range(max(0, int(rings))):
        frontier = set(n for key in frontier for n in mesh.vertex_neighbors(key)) - out
        if not frontier:
            break
        out |= frontier
    return out


def _face_min_angle(mesh: Mesh, fkey: Any) -> float:
    from compas_singular.framefield.quality import face_angles

    points = [mesh.vertex_coordinates(key) for key in mesh.face_vertices(fkey)]
    angles = face_angles(points)
    return min(angles) if angles else 180.0


def select(mesh: Mesh, selector: dict[str, Any] | str) -> tuple[set, str]:
    """Resolve a region selector to a set of vertex keys.

    Parameters
    ----------
    mesh : Mesh
    selector : dict or str
        A bare string is shorthand for ``{'kind': <string>}``. Kinds:

        ``all``
            Every vertex.
        ``interior``
            Every vertex not on a boundary.
        ``near_point``
            ``point`` and ``radius``. Everything within the radius.
        ``worst_faces``
            ``count`` (default 1). The corners of the ``count`` faces with the
            smallest corner angle -- where a mesh is worst is where a local
            smoothing pass is worth spending.
        ``singularities``
            ``rings`` (default 2). The irregular vertices and poles, grown.
        ``handles``
            ``handles``, a list of vertex handles. For repeating a selection a
            previous report named.

    Returns
    -------
    tuple
        ``(set of vertex keys, note)`` -- the note says what was selected and is
        meant to be reported back.

    Raises
    ------
    SelectorError
        On an unknown kind or missing argument, so the tool can turn it into a
        refusal that names the mistake.
    """
    if isinstance(selector, str):
        selector = {"kind": selector}
    if not isinstance(selector, dict):
        raise SelectorError("a region must be an object like {'kind': 'worst_faces', 'count': 3}")

    kind = selector.get("kind")
    if not kind:
        raise SelectorError("a region needs a 'kind'; try 'all', 'interior', 'near_point', 'worst_faces', 'singularities' or 'handles'")

    if kind == "all":
        keys = set(mesh.vertices())
        return keys, "every vertex ({})".format(len(keys))

    if kind == "interior":
        keys = set(mesh.vertices()) - _boundary_vertices(mesh)
        return keys, "{} interior vertices".format(len(keys))

    if kind == "near_point":
        point = selector.get("point")
        radius = selector.get("radius")
        if point is None or radius is None:
            raise SelectorError("'near_point' needs 'point' and 'radius'")
        keys = set(key for key in mesh.vertices() if _distance(mesh.vertex_coordinates(key), point) <= radius)
        return keys, "{} vertices within {:g} of {}".format(len(keys), radius, vertex_handle(point))

    if kind == "worst_faces":
        count = int(selector.get("count", 1))
        if count < 1:
            raise SelectorError("'count' must be at least 1")
        ranked = sorted(mesh.faces(), key=lambda f: _face_min_angle(mesh, f))
        chosen = ranked[:count]
        keys = set(key for fkey in chosen for key in mesh.face_vertices(fkey))
        worst = _face_min_angle(mesh, chosen[0]) if chosen else None
        return keys, "{} vertices of the {} worst face(s); worst angle {:.1f} deg".format(len(keys), len(chosen), worst if worst is not None else float("nan"))

    if kind == "singularities":
        rings = int(selector.get("rings", 2))
        seed = _singular_vertices(mesh)
        keys = _grow(mesh, seed, rings)
        return keys, "{} singular vertices grown by {} ring(s) to {}".format(len(seed), rings, len(keys))

    if kind == "handles":
        wanted = selector.get("handles")
        if not wanted:
            raise SelectorError("'handles' needs a non-empty 'handles' list")
        keys, missing = set(), []
        for item in wanted:
            key, how = resolve(mesh, item)
            if key is None:
                missing.append(item)
            else:
                keys.add(key)
        if missing:
            raise SelectorError("could not place {} of {} handles: {}".format(len(missing), len(wanted), ", ".join(str(m) for m in missing[:3])))
        return keys, "{} named vertices".format(len(keys))

    raise SelectorError("unknown region kind {!r}; try 'all', 'interior', 'near_point', 'worst_faces', 'singularities' or 'handles'".format(kind))


def describe_selector(selector: dict[str, Any] | str | Any) -> str:
    """A short phrase for a selector, for the history. Never raises."""
    if isinstance(selector, str):
        return selector
    if not isinstance(selector, dict):
        return str(selector)
    kind = selector.get("kind", "?")
    extra = ", ".join("{}={}".format(k, v) for k, v in sorted(selector.items()) if k != "kind" and k != "handles")
    return "{}({})".format(kind, extra) if extra else kind
