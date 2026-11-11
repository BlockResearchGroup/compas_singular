"""Make a mesh all-quad again after surgery, from the mesh, walls and poles alone (no field).

Main entry points: ``solve_non_quad_faces``, ``topological_quad_split``, ``densifiable``.
"""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING
from typing import Sequence

from compas.geometry import angle_vectors
from compas.geometry import distance_point_point
from compas.geometry import length_vector
from compas.geometry import subtract_vectors
from compas.itertools import pairwise

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = ["densifiable", "solve_non_quad_faces", "topological_quad_split"]


def _arc_midpoint(loops: list[list[list[float]]] | None, a: list[float], b: list[float], tol: float = 1e-6) -> list[float] | None:
    """Midpoint of the boundary arc between two points, if both lie on one loop."""
    for loop in loops or []:
        ring = list(loop) + list(loop[:1])
        best = [None, None]
        for k, p in enumerate((a, b)):
            bd, bs = float("inf"), None
            acc = 0.0
            for u, v in pairwise(ring):
                abx, aby = v[0] - u[0], v[1] - u[1]
                length = (abx * abx + aby * aby) ** 0.5
                if length == 0.0:
                    continue
                t = ((p[0] - u[0]) * abx + (p[1] - u[1]) * aby) / (length * length)
                t = max(0.0, min(1.0, t))
                q = [u[0] + abx * t, u[1] + aby * t, 0.0]
                d = distance_point_point(p, q)
                if d < bd:
                    bd, bs = d, acc + t * length
                acc += length
            if bd > tol:
                best = None
                break
            best[k] = (bs, acc)
        if best is None or best[0] is None or best[1] is None:
            continue

        sa, sb = best[0][0], best[1][0]
        total = best[0][1]
        forward = (sb - sa) % total
        span = min(forward, total - forward)

        # Both endpoints being on the loop is NOT enough: an edge can be a CHORD
        # across a hole, with both ends on the hole's rim and its interior well
        # inside the domain. Snapping such an edge's midpoint onto the rim moves
        # a vertex somewhere it does not belong -- measured on a round-holed
        # plate, that produced 114% coverage and a non-manifold mesh.
        #
        # A genuine boundary edge runs ALONG the wall, so its arc and its chord
        # have nearly the same length. A chord across a hole does not.
        chord = distance_point_point(a, b)
        if span > max(chord * 1.5, 1e-9):
            continue

        s = (sa + forward * 0.5) % total if forward <= total - forward else (sb + (total - forward) * 0.5) % total

        acc = 0.0
        for u, v in pairwise(ring):
            abx, aby = v[0] - u[0], v[1] - u[1]
            length = (abx * abx + aby * aby) ** 0.5
            if length == 0.0:
                continue
            if acc <= s <= acc + length:
                t = (s - acc) / length
                return [u[0] + abx * t, u[1] + aby * t, 0.0]
            acc += length
    return None


def topological_quad_split(
    vertices: list[list[float]],
    faces: list[list[int]],
    loops: list[list[list[float]]] | None = None,
) -> tuple[list[list[float]], list[list[int]]]:
    """Catmull-Clark topological split: every n-gon becomes n quads, no vertex moves.

    Always all-quad and conforming, at the cost of a valence-n vertex per n-gon.

    Returns
    -------
    (list[[x, y, z]], list[list[int]])
    """
    new_vertices = [list(p) for p in vertices]
    edge_mid = {}

    # Two distinct edges can snap to the SAME point on the wall, and a mesh with
    # two vertices at one position is not manifold. Measured on a round-holed
    # plate: 3 duplicated coordinates and a non-manifold result. Keep the
    # occupied positions and fall back to the chord when one is taken.
    def key_of(p: list[float]) -> tuple[float, float]:
        return (round(p[0], 9), round(p[1], 9))

    taken = set(key_of(p) for p in vertices)

    def mid(u: int, v: int) -> int:
        key = (min(u, v), max(u, v))
        if key not in edge_mid:
            edge_mid[key] = len(new_vertices)
            a, b = vertices[u], vertices[v]
            point = _arc_midpoint(loops, a, b)
            if point is None or key_of(point) in taken:
                point = [(a[i] + b[i]) / 2.0 for i in range(3)]
            taken.add(key_of(point))
            new_vertices.append(point)
        return edge_mid[key]

    new_faces = []
    for fv in faces:
        n = len(fv)
        centre = len(new_vertices)
        new_vertices.append([sum(vertices[v][i] for v in fv) / n for i in range(3)])
        mids = [mid(fv[i], fv[(i + 1) % n]) for i in range(n)]
        for i in range(n):
            new_faces.append([fv[i], mids[i], centre, mids[i - 1]])
    return new_vertices, new_faces


def _signed_area(pts: list[list[float]]) -> float:
    return 0.5 * sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))


def _min_angle(pts: list[list[float]]) -> float:
    """Smallest interior angle of a polygon, in radians."""
    n = len(pts)
    worst = pi
    for i in range(n):
        u = subtract_vectors(pts[(i - 1) % n], pts[i])
        v = subtract_vectors(pts[(i + 1) % n], pts[i])
        if length_vector(u) < 1e-12 or length_vector(v) < 1e-12:
            return 0.0
        worst = min(worst, angle_vectors(u, v))
    return worst


def _fan(fv: list[int]) -> list[list[int]]:
    """Cut an n-gon into quads plus at most one triangle along its diagonals, adding no vertex."""
    out = []
    fv = list(fv)
    while len(fv) > 4:
        out.append(fv[0:4])
        fv = fv[3:] + fv[0:1]
    out.append(fv)
    return out


def _best_fan(fv: list[int], coords: list[list[float]]) -> list[list[int]] | None:
    """The valid rotation of ``_fan`` with the largest minimum angle, or ``None``."""
    n = len(fv)
    whole = _signed_area([coords[v] for v in fv])
    best = None
    for start in range(n):
        rotated = fv[start:] + fv[:start]
        pieces = _fan(rotated)
        areas = [_signed_area([coords[v] for v in piece]) for piece in pieces]
        if any(a <= 1e-12 for a in areas):
            continue
        if abs(sum(areas) - whole) > 1e-6 * max(abs(whole), 1.0):
            continue
        score = min(_min_angle([coords[v] for v in piece]) for piece in pieces)
        if best is None or score > best[0]:
            best = (score, pieces)
    return best[1] if best else None


def _choose_pole(fv: list[int], coords: dict[int, list[float]]) -> int:
    """Choose the corner of a triangle to collapse at: the one whose two edges are closest in length."""
    best = None
    for i in range(3):
        p, a, b = fv[i], fv[(i + 1) % 3], fv[(i + 2) % 3]
        ra = distance_point_point(coords[p], coords[a])
        rb = distance_point_point(coords[b], coords[p])
        if ra + rb <= 0.0:
            continue
        imbalance = abs(ra - rb) / (ra + rb)
        if best is None or imbalance < best[0]:
            best = (imbalance, p)
    return best[1] if best else fv[0]


def solve_non_quad_faces(
    mesh: "QuadMesh",
    cls: type | None = None,
    loops: list[list[list[float]]] | None = None,
    poles: Sequence[list[float]] = (),
) -> tuple["QuadMesh", str]:
    """Make ``mesh`` a quad mesh: merge wall triangles, fan n-gons, keep triangles as poles, else quad-split.

    Returns ``(mesh, note)``.

    Parameters
    ----------
    mesh : CoarsePseudoQuadMesh
    cls : type, optional
    loops : list[list[[x, y, z]]], optional
        Boundary loops, so a split vertex lands on the wall and not on a chord.
    poles : list[[x, y, z]], optional
        Preferred pole positions. A triangle with a corner at one of these
        collapses there rather than wherever stage 3 would have chosen.

    Returns
    -------
    (mesh, note)
        The mesh, and a one-line description of what was needed -- empty when
        the input was already all quads.
    """
    cls = cls or type(mesh)
    sides = [len(mesh.face_vertices(f)) for f in mesh.faces()]
    if not sides:
        return mesh, "empty mesh"

    # ``from_polylines`` recovers faces from a planar embedding and does not
    # guarantee they all wind the same way. A backwards face breaks strip
    # propagation, so the two edges of a quad that ought to share a strip end up
    # in different ones with different densities -- and densification then hands
    # ``discrete_coons_patch`` two opposite sides of unequal length and dies with
    # an IndexError far from the cause. Measured on a hexagon: 2 faces wound
    # backwards out of 44, 11 faces left with mismatched opposite densities.
    mesh, flipped = _unify(mesh, cls)

    sides = [len(mesh.face_vertices(f)) for f in mesh.faces()]
    if set(sides) == {4}:
        return mesh, ("unified {} backwards face(s)".format(flipped) if flipped else "")

    before = {}
    for n in sides:
        before[n] = before.get(n, 0) + 1

    # -- stage 1: triangles touching the boundary ---------------------------
    merged = 0
    for fkey in list(mesh.faces()):
        if fkey not in list(mesh.faces()) or len(mesh.face_vertices(fkey)) != 3:
            continue
        on_wall = [v for v in mesh.face_vertices(fkey) if mesh.is_vertex_on_boundary(v)]
        if len(on_wall) != 2:
            continue
        a, b = on_wall
        if distance_point_point(mesh.vertex_coordinates(a), mesh.vertex_coordinates(b)) > 1e-6:
            continue  # genuinely distinct corners, not a doubling
        mesh.delete_face(fkey)
        for other in list(mesh.faces()):
            fv = mesh.face_vertices(other)
            if b in fv:
                mesh.face[other] = [a if v == b else v for v in fv]
        merged += 1
    if merged:
        mesh = _rebuild(mesh, cls)

    sides = [len(mesh.face_vertices(f)) for f in mesh.faces()]
    if set(sides) == {4}:
        return mesh, "merged {} doubled boundary vertex/ices".format(merged)

    stray = {n: c for n, c in before.items() if n != 4}

    # -- stages 2 and 3: local, conforming, and no extra patches -------------
    local, fanned, poled = _local_repair(mesh, cls, poles)
    if local is not None:
        ok, _ = densifiable(local)
        if ok:
            note = []
            if fanned:
                note.append("fanned {} n-gon(s) along their own diagonals".format(fanned))
            if poled:
                note.append("{} triangle(s) kept as pseudo-quad pole(s)".format(poled))
            return local, "resolved {} non-quad patch(es) {}: {}".format(sum(stray.values()), stray, "; ".join(note))

    # -- stage 4: the guaranteed step ---------------------------------------
    index = {vkey: i for i, vkey in enumerate(mesh.vertices())}
    vertices = [mesh.vertex_coordinates(v) for v in mesh.vertices()]
    faces = [[index[v] for v in mesh.face_vertices(f)] for f in mesh.faces()]
    vertices, faces = topological_quad_split(vertices, faces, loops)
    out = cls.from_vertices_and_faces(vertices, faces)

    note = "quad-split {} non-quad patch(es) {} -- the local repair did not leave a densifiable layout, so every n-gon centre is now a valence-n vertex, worst {}".format(
        sum(stray.values()), stray, max(before)
    )
    return out, note


def _local_repair(mesh: "QuadMesh", cls: type, poles: Sequence[list[float]] = ()) -> tuple["QuadMesh | None", int, int]:
    """Stages 2 and 3: fan the n-gons, pole the triangles. No new vertices.

    Returns
    -------
    (mesh or None, int, int)
        The repaired mesh -- ``None`` if some face could not be fanned validly
        -- and how many faces were fanned and how many poles registered.
    """
    index = {vkey: i for i, vkey in enumerate(mesh.vertices())}
    coords = [mesh.vertex_coordinates(v) for v in mesh.vertices()]

    faces = []
    fanned = 0
    for fkey in mesh.faces():
        fv = [index[v] for v in mesh.face_vertices(fkey)]
        if len(fv) <= 4:
            faces.append(fv)
            continue
        pieces = _best_fan(fv, coords)
        if pieces is None:
            return None, 0, 0  # a face no diagonal partitions
        faces.extend(pieces)
        fanned += 1

    out = cls.from_vertices_and_faces(coords, faces)

    pole_keys = set()
    for p in poles or ():
        pole_keys.add(tuple(round(c, 6) for c in p[:2]))

    face_poles = {}
    for fkey in out.faces():
        fv = out.face_vertices(fkey)
        if len(fv) != 3:
            continue
        chosen = None
        for vkey in fv:
            if tuple(round(c, 6) for c in out.vertex_coordinates(vkey)[:2]) in pole_keys:
                chosen = vkey
                break
        if chosen is None:
            chosen = _choose_pole(fv, {v: out.vertex_coordinates(v) for v in fv})
        face_poles[fkey] = chosen
    out.attributes["face_pole"] = face_poles
    return out, fanned, len(face_poles)


def densifiable(mesh: "QuadMesh") -> tuple[bool, str]:
    """Whether ``densification`` will succeed: quads or registered poles, none inverted, opposite edges share a strip.

    Returns
    -------
    (bool, str)
        Verdict and, when false, the first reason.
    """
    faces = list(mesh.faces())
    if not faces:
        return False, "no faces"

    if not mesh.is_manifold():
        return False, "coarse mesh is not manifold"

    face_pole = mesh.attributes.get("face_pole") or {}
    for fkey in faces:
        fv = mesh.face_vertices(fkey)
        if len(fv) == 3 and fkey in face_pole:
            if face_pole[fkey] not in fv:
                return False, "face {} has a pole that is not one of its corners".format(fkey)
        elif len(fv) != 4:
            return False, "face {} has {} sides".format(fkey, len(fv))
        pts = [mesh.vertex_coordinates(v) for v in fv]
        area2 = sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))
        if area2 <= 0.0:
            return False, "face {} is inverted or degenerate".format(fkey)

    try:
        mesh.collect_strips()
        edge_strip = {}
        for skey, edges in mesh.strips(data=True):
            for u, v in edges:
                edge_strip[u, v] = skey
                edge_strip[v, u] = skey
        for fkey in faces:
            he = list(mesh.face_halfedges(fkey))
            if len(he) == 3:
                pole = face_pole[fkey]
                rails = [e for e in he if pole in e]
                if edge_strip[rails[0]] != edge_strip[rails[1]]:
                    return False, "pseudo-quad {} has its two rails in different strips".format(fkey)
                continue
            s = [edge_strip[e] for e in he]
            if s[0] != s[2] or s[1] != s[3]:
                return False, "face {} has opposite edges in different strips".format(fkey)
    except Exception as exc:
        return False, "strip collection failed: {}".format(type(exc).__name__)

    return True, ""


def _unify(mesh: "QuadMesh", cls: type) -> tuple["QuadMesh", int]:
    """Make every face wind counter-clockwise in XY. Returns how many flipped."""
    index = {vkey: i for i, vkey in enumerate(mesh.vertices())}
    vertices = [mesh.vertex_coordinates(v) for v in mesh.vertices()]
    faces = [[index[v] for v in mesh.face_vertices(f)] for f in mesh.faces()]

    def signed(fv: list[int]) -> float:
        pts = [vertices[v] for v in fv]
        return sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))

    if all(signed(fv) > 0 for fv in faces):
        return mesh, 0

    # Reversing individual negative-area faces is WRONG, even though it makes
    # every signed area positive. Orientation is a topological property shared
    # across edges: flipping one face and not its neighbour leaves them
    # traversing their common edge the same way, which is exactly what
    # non-manifold means. Doing that turned a manifold backstop mesh
    # non-manifold while every face reported positive area.
    #
    # So unify TOPOLOGICALLY first -- that makes the winding consistent but
    # picks its seed arbitrarily, so it can settle on all-clockwise -- then flip
    # the WHOLE mesh if the total area came out negative.
    # Written out rather than calling ``Mesh.unify_cycles`` -- that did not
    # return in ten minutes on a few-thousand-face backstop mesh. This is a
    # single breadth-first pass over face adjacency.
    edge_faces = {}
    for i, fv in enumerate(faces):
        for u, v in pairwise(fv + fv[:1]):
            edge_faces.setdefault((min(u, v), max(u, v)), []).append(i)

    seen = [False] * len(faces)
    reversed_ = [False] * len(faces)
    for seed in range(len(faces)):
        if seen[seed]:
            continue
        seen[seed] = True
        stack = [seed]
        while stack:
            i = stack.pop()
            fv = faces[i]
            directed = set(pairwise(fv + fv[:1]))
            for u, v in directed:
                for j in edge_faces.get((min(u, v), max(u, v)), ()):
                    if j == i or seen[j]:
                        continue
                    seen[j] = True
                    # neighbours must traverse the shared edge the OTHER way
                    if (u, v) in set(pairwise(faces[j] + faces[j][:1])):
                        faces[j] = list(reversed(faces[j]))
                        reversed_[j] = True
                    stack.append(j)

    if sum(signed(fv) for fv in faces) < 0:
        faces = [list(reversed(fv)) for fv in faces]
        reversed_ = [not flag for flag in reversed_]

    return cls.from_vertices_and_faces(vertices, faces), sum(reversed_)


def _rebuild(mesh: "QuadMesh", cls: type) -> "QuadMesh":
    """Re-index a mesh after face surgery, dropping orphaned vertices."""
    keep = {v for f in mesh.faces() for v in mesh.face_vertices(f)}
    index = {vkey: i for i, vkey in enumerate(sorted(keep))}
    vertices = [mesh.vertex_coordinates(v) for v in sorted(keep)]
    faces = [[index[v] for v in mesh.face_vertices(f)] for f in mesh.faces()]
    return cls.from_vertices_and_faces(vertices, faces)
