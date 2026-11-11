"""Expand a unit into the global mesh: replicate by the group and weld along seams topologically.

No vertex is matched by rounded coordinates, so the result is symmetric to floating point.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any

from compas_singular.symmetry.group import SymmetryGroup

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarsePseudoQuadMesh
    from compas_singular.datastructures import PseudoQuadMesh
    from compas_singular.symmetry.cut import Seam


__all__ = ["seam_membership", "rotation_partners", "expand", "orbit_maps"]


def seam_membership(mesh: PseudoQuadMesh | CoarsePseudoQuadMesh, seams: list[Seam], eps: float) -> dict[int, list[tuple[str, float]]]:
    """``{vkey: [(seam name, t), ...]}`` for every vertex on a seam."""
    out = {}
    for vkey in mesh.vertices():
        p = mesh.vertex_coordinates(vkey)
        here = []
        for s in seams:
            t = s.locate(p, eps)
            if t is not None:
                here.append((s.name, t))
        if here:
            out[vkey] = here
    return out


def rotation_partners(membership: dict[int, list[tuple[str, float]]], tol: float) -> tuple[dict[int, int], list[tuple[float, int]], list[tuple[float, int]]]:
    """Match vertices on seam A to vertices on seam B at the same distance ``t``.

    Returns
    -------
    (dict, list, list)
        ``{vertex on A: vertex on B}``, then the unmatched ``(t, vkey)`` on A and
        on B. A vertex on both seams (the centre) is its own partner.
    """
    on_a = sorted((t, v) for v, here in membership.items() for name, t in here if name == "A")
    on_b = sorted((t, v) for v, here in membership.items() for name, t in here if name == "B")
    partners = {}
    used = set()
    unmatched_a = []
    j = 0
    for t, v in on_a:
        while j < len(on_b) and on_b[j][0] < t - tol:
            j += 1
        best = None
        for k in (j, j + 1):
            if k < len(on_b) and on_b[k][1] not in used and abs(on_b[k][0] - t) <= tol:
                if best is None or abs(on_b[k][0] - t) < abs(on_b[best][0] - t):
                    best = k
        if best is None:
            unmatched_a.append((t, v))
        else:
            partners[v] = on_b[best][1]
            used.add(on_b[best][1])
    unmatched_b = [(t, v) for t, v in on_b if v not in used]
    return partners, unmatched_a, unmatched_b


def expand(
    mesh: PseudoQuadMesh | CoarsePseudoQuadMesh,
    group: SymmetryGroup,
    seams: list[Seam],
    eps: float,
    cls: type,
    match_tol: float | None = None,
    curves: dict[tuple[int, int], Any] | None = None,
) -> Any:
    """The global mesh from a unit mesh.

    Parameters
    ----------
    mesh : PseudoQuadMesh or CoarsePseudoQuadMesh
        The unit, with ``attributes['face_pole']`` for its pole faces.
    group : SymmetryGroup
    seams : list[Seam]
        Empty for the trivial group.
    eps : float
        On-seam tolerance.
    cls : type
        The class of the result. Must offer ``from_vertices_and_faces_with_face_poles``.
    match_tol : float, optional
        Rotation seams: how far apart ``t`` may be for two vertices to be glued.
    curves : dict, optional
        ``{(u, v): points}`` edge shapes of the unit, carried to the copies.

    Returns
    -------
    mesh
        With ``attributes['orbits']`` recording, for every ``(unit vertex,
        element)`` pair, its global vertex.
    """
    elements = group.elements
    vkeys = list(mesh.vertices())
    xyz = dict((v, mesh.vertex_coordinates(v)) for v in vkeys)
    membership = seam_membership(mesh, seams, eps) if seams else {}
    match_tol = 1e3 * eps if match_tol is None else match_tol

    parent = {}

    def find(x: tuple[Any, str]) -> tuple[Any, str]:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent[x]
        return root

    def rank(x: tuple[Any, str]) -> tuple[int, str, str]:
        # The identity copy is preferred as a class's representative, so a vertex
        # of the unit keeps the unit's own coordinates in the global mesh.
        return (0 if x[1] == "I" else 1, str(x[0]), x[1])

    def union(x: tuple[Any, str], y: tuple[Any, str]) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            if rank(ry) < rank(rx):
                rx, ry = ry, rx
            parent[ry] = rx

    if seams and seams[0].kind == "rotation":
        partners, unmatched_a, unmatched_b = rotation_partners(membership, match_tol)
        if unmatched_a or unmatched_b:
            raise ValueError(
                "the rotation seams do not match: {} vertex(es) on seam A and {} on seam B "
                "have no partner at the same distance from the centre (A at t={}, B at t={}). "
                "Densities across the seams must agree -- set them on the unit, not per copy.".format(
                    len(unmatched_a), len(unmatched_b), [round(t, 4) for t, _ in unmatched_a[:4]], [round(t, 4) for t, _ in unmatched_b[:4]]
                )
            )
        rotation = group.element(seams[0].element)
        for e in elements:
            er = group.compose(e, rotation)
            for va, vb in partners.items():
                union((vb, e.key), (va, er.key))
    elif seams:
        by_name = dict((s.name, s) for s in seams)
        for v, here in membership.items():
            for name, _ in here:
                mirror = group.element(by_name[name].element)
                for e in elements:
                    union((v, e.key), (v, group.compose(e, mirror).key))

    gid = {}
    root_gid = {}
    vertices = []
    table = []
    for e in elements:
        for v in vkeys:
            root = find((v, e.key))
            if root not in root_gid:
                rv, re_key = root
                root_gid[root] = len(vertices)
                vertices.append(group.apply(group.element(re_key) if re_key != "I" else group.identity, xyz[rv]))
            gid[(v, e.key)] = root_gid[root]
            table.append([v, e.key, root_gid[root]])

    face_pole = mesh.attributes.get("face_pole") or {}
    faces = []
    face_poles = {}
    for e in elements:
        for fkey in mesh.faces():
            face = [gid[(v, e.key)] for v in mesh.face_vertices(fkey)]
            if len(set(face)) != len(face):
                raise ValueError("face {} collapses in copy {} -- a unit face touches a seam at two vertices glued together".format(fkey, e.key))
            if e.reflects:
                face.reverse()
            faces.append(face)
            pole = face_pole.get(fkey)
            if pole is not None:
                face_poles[len(faces) - 1] = gid[(pole, e.key)]

    out = cls.from_vertices_and_faces_with_face_poles(vertices, faces, face_poles)
    out.attributes["orbits"] = {"group": group.to_data(), "table": table}

    if curves and hasattr(out, "set_edges_to_curves"):
        shapes = {}
        for e in elements:
            for (u, v), points in curves.items():
                if (u, e.key) not in gid or (v, e.key) not in gid:
                    continue
                a, b = gid[(u, e.key)], gid[(v, e.key)]
                if (a, b) in shapes or (b, a) in shapes:
                    continue
                shapes[(a, b)] = group.apply_points(e, points)
        out.set_edges_to_curves(shapes)
    return out


def orbit_maps(mesh: PseudoQuadMesh | CoarsePseudoQuadMesh) -> dict[str, dict[int, int]]:
    """``{element key: {global vertex: global vertex}}`` from an expanded mesh.

    Exact: read from the labels ``expand`` recorded, not from positions.
    """
    orbits = mesh.attributes.get("orbits")
    if not orbits:
        raise ValueError("this mesh was not expanded from a symmetric unit -- it has no orbits")
    group = SymmetryGroup.from_data(orbits["group"])
    pair_gid = {}
    rep = {}
    for v, key, g in orbits["table"]:
        pair_gid[(v, key)] = g
        rep.setdefault(g, (v, key))
    maps = {}
    for h in group.elements:
        sigma = {}
        for g, (v, key) in rep.items():
            e = group.element(key) if key != "I" else group.identity
            sigma[g] = pair_gid[(v, group.compose(h, e).key)]
        maps[h.key] = sigma
    return maps


def symmetrise_positions(mesh: PseudoQuadMesh | CoarsePseudoQuadMesh, maps: dict[str, dict[int, int]] | None = None) -> float:
    """Replace every vertex by the average of its orbit, in place. Returns the largest displacement."""
    orbits = mesh.attributes["orbits"]
    group = SymmetryGroup.from_data(orbits["group"])
    maps = orbit_maps(mesh) if maps is None else maps
    new = {}
    for v in mesh.vertices():
        sx = sy = 0.0
        for h in group.elements:
            w = maps[h.key][v]
            p = group.apply(group.inverse(h), mesh.vertex_coordinates(w))
            sx += p[0]
            sy += p[1]
        new[v] = (sx / group.order, sy / group.order)
    worst = 0.0
    for v, (x, y) in new.items():
        p = mesh.vertex_coordinates(v)
        worst = max(worst, abs(p[0] - x), abs(p[1] - y))
        mesh.vertex_attributes(v, "xy", [x, y])
    return worst
