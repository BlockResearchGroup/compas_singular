"""Meshes and curves as plain index-based JSON for the spool, with poles as points, never keys.

Nothing compas-typed crosses the wire, and coordinates are not rounded.
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
    from compas.geometry import Polyline


__all__ = [
    "mesh_to_wire",
    "mesh_from_wire",
    "points_of",
    "curve_to_wire",
    "curves_to_wire",
]


class WireError(ValueError):
    """The wire payload is not a mesh. Raised where the fault is, not later."""


def points_of(curve: Polyline | Sequence[Any]) -> list[list[float]]:
    """Any curve-ish thing as a plain list of ``[x, y, z]``.

    Accepts what the rest of the library hands around: a compas ``Polyline``, a
    sequence of compas ``Point`` objects, or a sequence of bare triples.

    Parameters
    ----------
    curve : Polyline | sequence

    Returns
    -------
    list[list[float]]
    """
    points = getattr(curve, "points", None)
    if points is None:
        points = curve
    return [[float(p[0]), float(p[1]), float(p[2])] for p in points]


def curve_to_wire(curve: Polyline | Sequence[Any]) -> list[list[float]]:
    """One curve as a point list. Alias of ``points_of``, named for the wire."""
    return points_of(curve)


def curves_to_wire(curves: Iterable[Polyline | Sequence[Any]] | None) -> list[list[list[float]]]:
    """A sequence of curves as a list of point lists."""
    return [points_of(curve) for curve in curves or []]


def _pole_points(mesh: Mesh) -> list[list[float]]:
    """The mesh's poles as points, or ``[]`` if it has no notion of one."""
    poles = getattr(mesh, "poles", None)
    if poles is None:
        return []
    try:
        keys = list(poles())
    except Exception:
        return []
    return [list(mesh.vertex_coordinates(vkey)) for vkey in keys]


def mesh_to_wire(mesh: Mesh) -> dict[str, Any]:
    """Encode a mesh for the spool.

    Parameters
    ----------
    mesh : Mesh
        Any compas mesh, including this library's pseudo-quad subclasses.

    Returns
    -------
    dict
        ``vertices``, ``faces`` and ``poles``.
    """
    keys = list(mesh.vertices())
    index = dict((key, i) for i, key in enumerate(keys))
    vertices = [list(mesh.vertex_coordinates(key)) for key in keys]
    faces = [[index[key] for key in mesh.face_vertices(fkey)] for fkey in mesh.faces()]
    return {"vertices": vertices, "faces": faces, "poles": _pole_points(mesh)}


def mesh_from_wire(data: dict[str, Any], cls: type | None = None) -> Mesh:
    """Rebuild a mesh from the spool.

    Parameters
    ----------
    data : dict
        As produced by ``mesh_to_wire``.
    cls : type, optional
        The class to build. Defaults to
        ``compas_singular.datastructures.QuadMesh``, or to
        ``compas_singular.datastructures.CoarsePseudoQuadMesh`` when the
        payload carries poles -- a mesh with poles is not a plain quad mesh and
        building it as one silently discards the collapsed sides.

    Returns
    -------
    Mesh
    """
    if not isinstance(data, dict):
        raise WireError("mesh payload is {}, not a dict".format(type(data).__name__))
    vertices = data.get("vertices")
    faces = data.get("faces")
    if not isinstance(vertices, list) or not isinstance(faces, list):
        raise WireError("mesh payload needs 'vertices' and 'faces' lists; got keys {}".format(sorted(data)))
    if not vertices:
        raise WireError("mesh payload has no vertices")

    count = len(vertices)
    for i, face in enumerate(faces):
        if not isinstance(face, list) or len(face) < 3:
            raise WireError("face {} is not a list of at least 3 indices".format(i))
        for key in face:
            if not isinstance(key, int) or key < 0 or key >= count:
                raise WireError("face {} refers to vertex {}, but the payload has {} vertices".format(i, key, count))

    poles = data.get("poles") or []

    # Imported here, not at module scope: the bridge is imported by the Rhino
    # side too, and keeping it free of package-level imports is what lets it be
    # loaded after ``ensure_paths()`` without dragging the datastructures in.
    if cls is None:
        if poles:
            from compas_singular.datastructures import CoarsePseudoQuadMesh as cls
        else:
            from compas_singular.datastructures import QuadMesh as cls

    builder = getattr(cls, "from_vertices_and_faces_with_poles", None)
    if poles and builder is not None:
        return builder(vertices, faces, poles)
    return cls.from_vertices_and_faces(vertices, faces)
