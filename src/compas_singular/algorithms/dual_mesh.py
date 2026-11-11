"""The dual of a quad mesh, closed on the boundary so every face is a quad.

Loses some area on a curved outline; use ``Mesh.dual(include_boundary=True)`` when that matters.
Do not dualise a mesh with poles. Design notes: ``design_notes/algorithms.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Sequence

from compas.datastructures import mesh_conway_dual
from compas.geometry import bestfit_plane
from compas.geometry import closest_point_on_segment
from compas.geometry import cross_vectors
from compas.geometry import distance_point_plane
from compas.geometry import distance_point_point
from compas.geometry import dot_vectors
from compas.geometry import length_vector
from compas.geometry import subtract_vectors
from compas.tolerance import TOL

if TYPE_CHECKING:
    from compas.datastructures import Mesh


__all__ = ["dual_mesh", "redistribute_blocks"]


def dual_mesh(mesh: Mesh, redistribute: bool = True) -> Mesh:
    """The dual of a mesh, with its boundary closed by a layer of quads.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh

    Returns
    -------
    compas.datastructures.Mesh
        The dual, of the same class as ``mesh``.

    """
    dual = mesh_conway_dual(mesh)

    # renumbers the keys 0..n-1, which is what from_vertices_and_faces wants -- done by
    # hand, as compas_singular's Mesh.to_vertices_and_faces returns dicts by default
    keys = list(dual.vertices())
    index_of = {vkey: i for i, vkey in enumerate(keys)}
    vertices = [dual.vertex_coordinates(vkey) for vkey in keys]
    faces = [[index_of[vkey] for vkey in dual.face_vertices(fkey)] for fkey in dual.faces()]

    index = {TOL.geometric_key(xyz): i for i, xyz in enumerate(vertices)}

    # Extent to the boundary
    def add(xyz: Sequence[float]) -> int:
        gkey = TOL.geometric_key(xyz)
        if gkey not in index:
            index[gkey] = len(vertices)
            vertices.append([float(c) for c in xyz])
        return index[gkey]

    for vkey in mesh.vertices():
        if not mesh.is_vertex_on_boundary(vkey):
            continue
        nbrs = mesh.vertex_neighbors(vkey, ordered=True)
        if not nbrs:
            continue
        ring = [add(mesh.edge_midpoint((vkey, nbrs[0])))]
        ring += [add(mesh.face_centroid(fkey)) for fkey in mesh.vertex_faces(vkey, ordered=True)]
        ring.append(add(mesh.edge_midpoint((vkey, nbrs[-1]))))
        if len(nbrs) == 2:
            ring.append(add(mesh.vertex_coordinates(vkey)))
        # mesh_conway_dual winds its faces with reversed(vertex_faces(ordered)),
        # and the ring above follows vertex_faces. A face added the other way
        # round does not raise, it overwrites a halfedge -- so reverse it here
        # rather than leaving unify_cycles to repair a broken map.
        faces.append(ring[::-1])

    out = type(dual).from_vertices_and_faces(vertices, faces)

    # There is already a way to extent to the boundary in compas, but I have noticed that it gives weird corners
    # out = mesh.dual(include_boundary=True)
    # _vertices, faces = out.to_vertices_and_faces()

    # Both halves wind the same way, so the result is already oriented -- and it
    # comes out with the primal's own orientation, normals dotting +1.00 with it.
    # unify_cycles is only the repair for a primal that arrived mis-wound, and it
    # is QUADRATIC: on 2880 primal faces it took 39.8 s, against 0.06 s for the
    # conway dual it repairs. So pay for it only when the check says to.
    if not _is_oriented(faces):
        out.unify_cycles()
    # Redistribute
    if redistribute:
        redistribute_blocks(out, mesh)

    return out


def _is_oriented(faces: Sequence[Sequence[int]]) -> bool:
    """True if no directed halfedge is claimed twice."""
    seen = set()
    for face in faces:
        for u, v in zip(face, face[1:] + face[:1]):
            if (u, v) in seen:
                return False
            seen.add((u, v))
    return True


# ----------------------------------------------------------------------------
# evening out the block sizes
# ----------------------------------------------------------------------------


def redistribute_blocks(dual: Mesh, primal: Mesh, project: bool = True) -> Mesh:
    """Even out the half-size boundary blocks of a dual in place, by respacing along strips and boundary arcs.

    Parameters
    ----------
    dual : compas.datastructures.Mesh
        The dual to modify, as returned by ``dual_mesh``. Modified in place.
    primal : compas.datastructures.Mesh
        The mesh it was built from. Supplies the outline, the strips and the
        surface.
    project : bool, optional
        Pull the moved vertices back onto the primal surface afterwards. A chain
        segment that spans a crease is a chord, so a vertex placed along it
        would otherwise float off a folded slab. Skipped on a planar primal,
        where it can do nothing.

    Returns
    -------
    compas.datastructures.Mesh
        ``dual``, modified in place.
    """
    gkey_vertex = {TOL.geometric_key(dual.vertex_coordinates(v)): v for v in dual.vertices()}

    def at(xyz: Sequence[float]) -> int | None:
        return gkey_vertex.get(TOL.geometric_key(xyz))

    # A boundary chain rides the primal OUTLINE, not its own chords: on a wall
    # that bends, midpoint -> midpoint cuts the corner and the dual boundary
    # would leave the outline it is supposed to trace (measured 3.7e-2 off a
    # folded slab). A strip chain has no such guide and uses its own points --
    # what it chords is the crease, which the projection below pulls back.
    chains = []
    for chain, polyline in _boundary_chains(primal, gkey_vertex):
        lengths = [distance_point_point(a, b) for a, b in zip(polyline, polyline[1:])]
        stops, running = [0.0], 0.0
        for length in lengths:
            stops.append(running + length / 2.0)  # where the edge midpoint sits
            running += length
        stops.append(running)
        chains.append((chain, polyline, lengths, stops))

    for faces, ends in _strips(primal):
        chain = [at(ends[0])] + [at(primal.face_centroid(f)) for f in faces] + [at(ends[1])]
        if any(vkey is None for vkey in chain):
            continue
        points = [dual.vertex_coordinates(v) for v in chain]
        lengths = [distance_point_point(a, b) for a, b in zip(points, points[1:])]
        stops, running = [0.0], 0.0
        for length in lengths:
            running += length
            stops.append(running)
        chains.append((chain, points, lengths, stops))

    # every chain is measured against the ORIGINAL dual and the displacements
    # are summed, never applied in sequence. A boundary midpoint slides ALONG
    # the wall and the strip it caps runs INTO the wall, so the two moves are
    # orthogonal and independent -- but measure the strip after the boundary has
    # already moved and its end segment reads as the diagonal of the two, which
    # skews the whole chain (measured: block spread 1.00 -> 1.07 on a grid).
    moves = {}
    for chain, polyline, lengths, stops in chains:
        if stops[-1] <= 0.0:
            continue
        for vkey, stop in zip(chain[1:-1], _respaced(stops)):
            was = dual.vertex_coordinates(vkey)
            xyz = _point_at(polyline, lengths, stop)
            delta = moves.setdefault(vkey, [0.0, 0.0, 0.0])
            for k in range(3):
                delta[k] += xyz[k] - was[k]

    home = {}
    for fkey in primal.faces():
        vkey = at(primal.face_centroid(fkey))
        if vkey is not None:
            home[vkey] = fkey

    for vkey, delta in moves.items():
        xyz = dual.vertex_coordinates(vkey)
        dual.vertex_attributes(vkey, "xyz", [xyz[k] + delta[k] for k in range(3)])

    # Only a vertex near a CREASE can be off the surface. Everywhere else the
    # mesh is locally flat over the whole half-cell a vertex can travel, the
    # chain segments lie in that plane, and projecting is a no-op that costs
    # nine face tests a vertex. On a folded 8 000-face slab only 160 faces are
    # creased, and skipping the rest took the run from 11.9 s to 1.3 s.
    if project and not _is_planar(primal):
        creased = _creased_faces(primal)
        around = {}
        for vkey in moves:
            fkey = home.get(vkey)
            if fkey is None or fkey not in creased:
                continue
            if fkey not in around:
                around[fkey] = list(set([fkey] + [g for v in primal.face_vertices(fkey) for g in primal.vertex_faces(v)]))
            xyz, _ = _project(dual.vertex_coordinates(vkey), primal, around[fkey])
            dual.vertex_attributes(vkey, "xyz", xyz)

    return dual


def _respaced(stops: Sequence[float]) -> list[float]:
    """New positions for a chain's interior points: double the two end gaps, rescale to the chain length."""
    widths = [b - a for a, b in zip(stops, stops[1:])]
    span = stops[-1] - stops[0]
    widths[0] *= 2.0
    widths[-1] *= 2.0
    scale = span / sum(widths)

    out, running = [], stops[0]
    for width in widths[:-1]:
        running += width * scale
        out.append(running)
    return out


def _point_at(polyline: Sequence[Sequence[float]], lengths: Sequence[float], distance: float) -> list[float]:
    """The point at ``distance`` along a polyline of known segment lengths."""
    for i, length in enumerate(lengths):
        if distance <= length or i == len(lengths) - 1:
            t = distance / length if length else 0.0
            a, b = polyline[i], polyline[i + 1]
            return [a[k] + t * (b[k] - a[k]) for k in range(3)]
        distance -= length
    return list(polyline[-1])


def _boundary_chains(primal: Mesh, gkey_vertex: dict[str, int]) -> list[tuple[list[int], list[Sequence[float]]]]:
    """Each primal boundary arc, as ``(dual vertices, primal outline points)``."""
    chains = []
    for ring in primal.vertices_on_boundaries():
        if ring[0] == ring[-1]:
            ring = ring[:-1]
        n = len(ring)
        if n < 3:
            continue

        # dual_mesh inserts the primal vertex itself only at a valence-2 domain
        # corner, so a primal boundary vertex that IS a dual vertex is a corner
        corner = [gkey_vertex.get(TOL.geometric_key(primal.vertex_coordinates(v))) for v in ring]
        marks = [i for i, vkey in enumerate(corner) if vkey is not None]
        if not marks:
            continue  # a smooth ring has no half block along itself

        mids = [gkey_vertex.get(TOL.geometric_key(primal.edge_midpoint((ring[i], ring[(i + 1) % n])))) for i in range(n)]

        for a, b in zip(marks, marks[1:] + marks[:1]):
            k = (b - a) % n
            chain = [corner[a]] + [mids[(a + j) % n] for j in range(k)] + [corner[b]]
            if len(chain) > 2 and all(vkey is not None for vkey in chain):
                outline = [primal.vertex_coordinates(ring[(a + j) % n]) for j in range(k + 1)]
                chains.append((chain, outline))
    return chains


def _strips(primal: Mesh) -> list[tuple[list[int], tuple[Sequence[float], Sequence[float]]]]:
    """Open primal strips, as ``(faces, (start midpoint, end midpoint))``."""
    strips, seen = [], set()
    # hoisted: Mesh.number_of_faces() is len(list(self.faces())), so leaving it
    # in the walk below makes the whole pass quadratic -- 14 s of a 15 s run on
    # an 8 000-face mesh, in 128 million generator steps
    cap = primal.number_of_faces()
    for ring in primal.edges_on_boundaries():
        for edge in ring:
            if frozenset(edge) in seen:
                continue
            faces, walk, entry, exit_edge = [], _face_of(primal, edge), edge, None
            while walk is not None:
                faces.append(walk)
                exit_edge = _opposite_edge(primal, walk, entry)
                if exit_edge is None:
                    break
                nxt = None
                for candidate in primal.edge_faces(exit_edge):
                    if candidate is not None and candidate != walk:
                        nxt = candidate
                        break
                entry, walk = exit_edge, nxt
                if len(faces) > cap:
                    exit_edge = None  # a strip that will not close: give up
                    break
            if not faces or exit_edge is None:
                continue
            seen.add(frozenset(edge))
            seen.add(frozenset(exit_edge))
            strips.append((faces, (primal.edge_midpoint(edge), primal.edge_midpoint(exit_edge))))
    return strips


def _face_of(mesh: Mesh, edge: tuple[int, int]) -> int | None:
    for fkey in mesh.edge_faces(edge):
        if fkey is not None:
            return fkey
    return None


def _opposite_edge(mesh: Mesh, fkey: int, edge: tuple[int, int]) -> tuple[int, int] | None:
    """The edge of a quad facing ``edge``; None if the face is not a quad."""
    vertices = mesh.face_vertices(fkey)
    if len(vertices) != 4:
        return None
    pair = set(edge)
    for i in range(4):
        if {vertices[i], vertices[(i + 1) % 4]} == pair:
            return (vertices[(i + 2) % 4], vertices[(i + 3) % 4])
    return None


def _creased_faces(mesh: Mesh, tol: float = 1e-9) -> set[int]:
    """Faces with a non-coplanar face in their 1-ring -- where a fold shows."""
    normals = {fkey: mesh.face_normal(fkey) for fkey in mesh.faces()}
    creased = set()
    for fkey in mesh.faces():
        ring = set(g for vkey in mesh.face_vertices(fkey) for g in mesh.vertex_faces(vkey))
        for g in ring:
            if dot_vectors(normals[fkey], normals[g]) < 1.0 - tol:
                creased.add(fkey)
                break
    return creased


def _is_planar(mesh: Mesh, tol: float | None = None) -> bool:
    points = [mesh.vertex_coordinates(v) for v in mesh.vertices()]
    if len(points) < 4:
        return True
    if tol is None:
        lengths = [mesh.edge_length(edge) for edge in mesh.edges()]
        tol = 1e-6 * sum(lengths) / float(len(lengths))
    plane = bestfit_plane(points)
    return all(distance_point_plane(point, plane) <= tol for point in points)


def _project(point: Sequence[float], mesh: Mesh, candidates: Sequence[int]) -> tuple[Sequence[float], int]:
    """Closest point on a local patch of faces; also returns the winning face."""
    best, best_distance, best_face = point, None, candidates[0]
    for fkey in candidates:
        polygon = mesh.face_coordinates(fkey)
        centroid = mesh.face_centroid(fkey)
        for i in range(len(polygon)):
            near = _closest_on_triangle(point, polygon[i], polygon[(i + 1) % len(polygon)], centroid)
            distance = distance_point_point(point, near)
            if best_distance is None or distance < best_distance:
                best, best_distance, best_face = near, distance, fkey
    return best, best_face


def _closest_on_triangle(point: Sequence[float], a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> Sequence[float]:
    """Closest point of triangle abc to point, in its interior or on an edge."""
    ab = subtract_vectors(b, a)
    ac = subtract_vectors(c, a)
    normal = cross_vectors(ab, ac)
    norm = length_vector(normal)
    if norm > 1e-12:
        normal = [n / norm for n in normal]
        ap = subtract_vectors(point, a)
        inplane = [point[k] - dot_vectors(ap, normal) * normal[k] for k in range(3)]
        u, v = _barycentric(inplane, a, ab, ac)
        if u >= 0.0 and v >= 0.0 and u + v <= 1.0:
            return inplane
    best, best_distance = None, None
    for start, end in ((a, b), (b, c), (c, a)):
        near = closest_point_on_segment(point, (start, end))
        distance = distance_point_point(point, near)
        if best_distance is None or distance < best_distance:
            best, best_distance = near, distance
    return best


def _barycentric(point: Sequence[float], a: Sequence[float], ab: Sequence[float], ac: Sequence[float]) -> tuple[float, float]:
    d00 = dot_vectors(ab, ab)
    d01 = dot_vectors(ab, ac)
    d11 = dot_vectors(ac, ac)
    ap = subtract_vectors(point, a)
    d20 = dot_vectors(ap, ab)
    d21 = dot_vectors(ap, ac)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-18:
        return -1.0, -1.0
    return (d11 * d20 - d01 * d21) / denom, (d00 * d21 - d01 * d20) / denom
