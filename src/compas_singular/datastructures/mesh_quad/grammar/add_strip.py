"""Add a strip along a polyedge (thesis 5.3.1) and, by default, open it by an exact rule: thirds across a vertex, a third inwards along a wall.

With ``open_strip=False`` the new vertex pairs stay coincident until the caller separates them. An opened strip that folds a face raises.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable

from compas.datastructures.mesh.operations.substitute import mesh_substitute_vertex_in_faces
from compas.geometry import angle_vectors
from compas.geometry import distance_point_point
from compas.geometry import subtract_vectors
from compas.itertools import pairwise
from compas.topology import breadth_first_paths

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = [
    "add_strip",
    "add_strips",
    "split_strip",
    "split_strips",
    "strip_polyedge_update",
    "is_polyedge_valid_for_strip_addition",
    "pole_ends",
    "polyedge_sides",
    "open_added_strip",
]


def add_strips(mesh: QuadMesh, polyedges: list[list[int]], open_strip: bool = True, project: Callable[[list[float]], list[float]] | None = None) -> list[int]:
    """Add a strip along each polyedge in order, re-deriving the remaining polyedges after each insertion.

    Parameters
    ----------
    mesh : QuadMesh
        A quad mesh, with ``attributes['strips']`` already collected.
    polyedges : list[list[int]]
        Polyedges, each a list of vertex keys.
    open_strip : bool, optional
        Open each new strip by ``open_added_strip``. Default ``True``.
    project : callable, optional
        Passed on to ``open_added_strip``.

    Returns
    -------
    list
        The new strip keys, in the order the strips were added.
    """
    pending = list(polyedges)
    new_skeys = []

    while pending:
        polyedge = pending.pop()
        # ``list(...)``: ``add_strip`` consumes the polyedge it is given.
        new_skey, _old_to_new, copies = _add_strip(mesh, list(polyedge), open_strip=open_strip, project=project)
        new_skeys.append(new_skey)
        pending = [strip_polyedge_update(mesh, pending_polyedge, copies) for pending_polyedge in pending]

    return new_skeys


def add_strip(mesh: QuadMesh, polyedge: list[int], open_strip: bool = True, project: Callable[[list[float]], list[float]] | None = None) -> tuple[int, dict[int, tuple[int, int]]]:
    """Add a strip along ``polyedge``, opened by ``open_added_strip`` by default.

    The polyedge is consumed. It may run along a wall, and pass a vertex twice (a self-crossing strip).

    Parameters
    ----------
    mesh : QuadMesh
        A quad mesh. ``attributes['strips']`` must be non-empty --
        ``update_strip_data`` does ``max(...) + 1`` and raises on an empty dict.
    polyedge : list[int]
        Vertex keys in order, each joined to the next by an edge. Either closed,
        or with both ends on the boundary or at a pole (see ``pole_ends``).
    open_strip : bool, optional
        Give the new strip its width with ``open_added_strip``. Default
        ``True``. ``False`` is topology only: both copies of each vertex stay on
        top of the vertex they replace.
    project : callable, optional
        ``project(xyz) -> xyz`` for the new pair at a BOUNDARY vertex, e.g. back
        onto a curved wall. Without it they stay on the chord across the vertex.
        Ignored when ``open_strip`` is ``False``.

    Returns
    -------
    tuple[int, dict]
        The new strip key, and ``{old vertex: (left copy, right copy)}``. The
        left and right polyedges are the values of that map read in polyedge
        order; the pairs in it are exactly the vertices an opener has to
        separate. A pole end maps to ``(pole, pole)``. A vertex passed twice is
        split twice: the second pass splits the first pass's copies, which then
        key its pairs although they are no longer in the mesh.

    Raises
    ------
    ValueError
        With ``open_strip``, when a face next to the strip is left folded or
        without area. The mesh is left edited: work on a copy.

    Notes
    -----
    An end at an existing pole is not copied: the strip ends there in a new
    pseudo-quad, as in thesis Fig. 5.15. See ``pole_ends``.
    """
    skey, old_to_new, _copies = _add_strip(mesh, polyedge, open_strip=open_strip, project=project)
    return skey, old_to_new


def _add_strip(
    mesh: QuadMesh,
    polyedge: list[int],
    open_strip: bool = True,
    project: Callable[[list[float]], list[float]] | None = None,
) -> tuple[int, dict[int, tuple[int, int]], dict[int, tuple[int, ...]]]:
    """``add_strip``, also returning ``{input vertex: every copy of it left in the mesh}``."""
    if not polyedge:
        raise ValueError("Empty polyedge.")
    start_pole, end_pole = pole_ends(mesh, polyedge)
    if start_pole and end_pole and len(polyedge) < 3:
        raise ValueError("A strip with two pole ends needs a polyedge of at least two edges.")
    face_pole = mesh.attributes.get("face_pole")

    closed = polyedge[0] == polyedge[-1]
    seq = polyedge[:-1] if closed else list(polyedge)
    revisits = len(set(seq)) < len(seq)
    if closed and revisits and seq.count(seq[-1]) > 1:
        # a loop has no fixed start: end it on a vertex it passes once, as the
        # walk cannot re-route a second pass that has nothing after it
        once = [j for j, vkey in enumerate(seq) if seq.count(vkey) == 1]
        if once:
            seq = seq[once[0] + 1 :] + seq[: once[0] + 1]
            polyedge[:] = seq + seq[:1]
    # A polyedge passing a vertex twice, or next to itself, is beyond
    # ``update_strip_data``, which follows each edge through one split only.
    touches = revisits or _touches_itself(mesh, seq, closed)
    by_roots = revisits or (touches and not (start_pole or end_pole))
    plans = None
    if open_strip:
        # Measured BEFORE the walk: it deletes every vertex of the polyedge, so
        # the neighbours that define each side have to be read off the mesh as
        # it still is.
        sides = polyedge_sides(mesh, polyedge)
        boundary = _boundary_vertex_set(mesh)
        on_boundary = set(vkey for vkey in polyedge if vkey in boundary)
        # The thirds rule sees one pair per vertex and two sides with neighbours,
        # read after the walk has deleted any that were on the polyedge too; a wall
        # with nothing beyond it, or a polyedge meeting itself, needs a plan per visit.
        if touches or any(_along_wall(mesh, seq, closed, j, boundary) for j in range(len(seq))):
            plans = [_visit_plan(mesh, seq, closed, j, boundary, project) for j in range(len(seq))]
        sign, tol = _orientation(mesh, seq)
    # every copy's input vertex, and per split: (position in the walk, copies, face roots around each)
    root = {}
    steps = []
    track = by_roots or plans is not None

    full_updated_polyedge = []
    # store data
    left_polyedge = []
    right_polyedge = []
    new_faces = []

    # exception if closed
    is_closed = polyedge[0] == polyedge[-1]
    if is_closed:
        polyedge.pop()

    k = -1
    count = len(polyedge) * 2
    while count and len(polyedge) > 0:
        k += 1
        count -= 1

        # select u, v, w if not closed
        if not is_closed:
            # u
            if len(left_polyedge) != 0:
                u1, u2 = left_polyedge[-1], right_polyedge[-1]
            else:
                u1, u2 = None, None
            # v
            v = polyedge.pop(0)
            full_updated_polyedge.append(v)
            # w
            if len(polyedge) != 0:
                w = polyedge[0]
            else:
                w = None

            # a pole end is not copied: both sides of the strip end at it
            if (start_pole and k == 0) or (end_pole and len(polyedge) == 0):
                left_polyedge.append(v)
                right_polyedge.append(v)
                if len(polyedge) == 0:
                    # the temporary triangle [u1, v, u2] is the closing pseudo-quad
                    face_pole[new_faces[-1]] = v
                continue

        # select u, v, w if closed
        else:
            # u
            if len(new_faces) != 0:
                u1, u2 = left_polyedge[-1], right_polyedge[-1]
            else:
                u1 = polyedge[-1]  # artificial u1
            # v
            v = polyedge.pop(0)
            full_updated_polyedge.append(v)
            # w
            if len(polyedge) != 0:
                w = polyedge[0]
            else:
                w = left_polyedge[0]

        # add new vertices
        faces = sort_faces(mesh, u1, v, w)
        v1, v2 = mesh.add_vertex(attr_dict=mesh.vertex[v]), mesh.add_vertex(attr_dict=mesh.vertex[v])

        if isinstance(faces[0], list):
            faces_1, faces_2 = faces
        else:
            # exception necessary for U-turns
            if faces[0] in mesh.vertex_faces(left_polyedge[-2]):
                faces_1 = faces
                faces_2 = []
            else:
                faces_1 = []
                faces_2 = faces
        mesh_substitute_vertex_in_faces(mesh, v, v1, faces_1)
        mesh_substitute_vertex_in_faces(mesh, v, v2, faces_2)
        if face_pole:
            # a pole crossed by the polyedge splits in two with the faces around it
            for new, fkeys in ((v1, faces_1), (v2, faces_2)):
                for fkey in fkeys:
                    if face_pole.get(fkey) == v:
                        face_pole[fkey] = new
        mesh.delete_vertex(v)
        left_polyedge.append(v1)
        right_polyedge.append(v2)

        # add new faces, different if at the start, end or main part of the polyedge
        if len(new_faces) == 0 and start_pole:
            fkey = mesh.add_face([u1, v1, v2])
            face_pole[fkey] = u1
            new_faces.append(fkey)
            if w is not None:
                new_faces.append(mesh.add_face([v1, w, v2]))
        elif len(new_faces) == 0:
            if not is_closed:
                new_faces.append(mesh.add_face([v1, w, v2]))
            else:
                new_faces.append(mesh.add_face([v1, v2, u1]))
                new_faces.append(mesh.add_face([v1, w, v2]))
        elif len(polyedge) == 0:
            if not is_closed:
                u1, u2 = left_polyedge[-2], right_polyedge[-2]
                face = new_faces.pop()
                mesh.delete_face(face)
                new_faces.append(mesh.add_face([u1, v1, v2, u2]))
            else:
                u1, u2 = left_polyedge[-2], right_polyedge[-2]
                face = new_faces.pop()
                mesh.delete_face(face)
                # ``[u1, v1, v2, u2]``, as every other rung of the strip. This was
                # ``[v1, u1, u2, v2]`` -- the same quad REVERSED -- so it shared a
                # same-direction halfedge with both its neighbours: the mesh still
                # passed ``is_manifold`` but every strip walk through that face went
                # wrong. The closing face below is right as it is: from the last
                # vertex to the first, ``v`` plays the role ``u`` plays here.
                new_faces.append(mesh.add_face([u1, v1, v2, u2]))
                face = new_faces.pop(0)
                mesh.delete_face(face)
                u1, u2 = left_polyedge[0], right_polyedge[0]
                new_faces.append(mesh.add_face([v1, u1, u2, v2]))
                # NOT followed by ``mesh_substitute_vertex_in_faces(mesh, v, v1/v2)``.
                # ``v`` was deleted above, and those calls pass no ``fkeys``, so they
                # default to EVERY face in the mesh and delete-and-re-add each one --
                # twice -- to substitute a vertex that no face references any more.
                # The ring is already closed by the two faces added above.
        else:
            face = new_faces.pop()
            mesh.delete_face(face)
            new_faces.append(mesh.add_face([u1, v1, v2, u2]))
            new_faces.append(mesh.add_face([v1, w, v2]))

        if track:
            root[v1] = root[v2] = root.get(v, v)
            steps.append((len(full_updated_polyedge) - 1, v1, v2, _face_roots(mesh, v1, root), _face_roots(mesh, v2, root)))

        # update
        updated_polyedge = []
        via_vkeys = [v1, v2]
        for i, vkey in enumerate(polyedge):
            if vkey != v:
                updated_polyedge.append(vkey)
            elif i == len(polyedge) - 1 and not is_closed:
                # the polyedge ENDS where it passed before: end on a copy it
                # reaches, on the wall where one is
                from_vkey = polyedge[i - 1]
                ends = [[c] for c in via_vkeys if c in mesh.halfedge[from_vkey]]
                ends += [[c, d] for c, d in (via_vkeys, via_vkeys[::-1]) if c in mesh.halfedge[from_vkey] and d in mesh.halfedge[c]]
                ends.sort(key=lambda end: (not mesh.is_vertex_on_boundary(end[-1]), len(end)))
                updated_polyedge += ends[0]
            else:
                from_vkey = polyedge[i - 1]
                to_vkey = polyedge[i + 1]
                updated_polyedge += polyedge_from_to_via_vertices(mesh, from_vkey, to_vkey, via_vkeys)[1:-1]
        polyedge = updated_polyedge

    old_vkeys_to_new_vkeys = {u0: (u1, u2) for u0, u1, u2 in zip(full_updated_polyedge, left_polyedge, right_polyedge)}

    if by_roots:
        n = _update_strip_data_by_roots(mesh, root, steps)
        # a pair the second pass split again is gone from the mesh
        old_vkeys_to_new_vkeys = {u0: pair for u0, pair in old_vkeys_to_new_vkeys.items() if pair[0] in mesh.vertex and pair[1] in mesh.vertex}
        copies = {}
        for pair in old_vkeys_to_new_vkeys.values():
            for vkey in pair:
                copies.setdefault(root.get(vkey, vkey), []).append(vkey)
        copies = {vkey: tuple(new) for vkey, new in copies.items()}
    else:
        n = update_strip_data(mesh, full_updated_polyedge, old_vkeys_to_new_vkeys, closed=is_closed)
        copies = old_vkeys_to_new_vkeys

    if open_strip:
        if plans is None:
            open_added_strip(mesh, old_vkeys_to_new_vkeys, sides, on_boundary, project=project)
        else:
            _open_by_visits(mesh, seq, full_updated_polyedge, root, steps, plans)
        new_vertices = set(vkey for pair in old_vkeys_to_new_vkeys.values() for vkey in pair)
        folded = _folded_faces(mesh, new_vertices, sign, tol)
        if folded:
            raise ValueError("the new strip could not be opened flat: {} face(s) next to it fold over or have no area ({})".format(len(folded), sorted(folded)))
    return n, old_vkeys_to_new_vkeys, copies


def split_strip(mesh: QuadMesh, skey: int, n: int = 2, open_strip: bool = True, project: Callable[[list[float]], list[float]] | None = None) -> list[int]:
    """Refine a strip into ``n`` strips. ``open_strip``/``project`` as in ``add_strip``.

    Returns
    -------
    list
        The existing strip key, followed by the ``n - 1`` new ones.
    """
    return [skey] + [add_strip(mesh, list(mesh.strip_side_polyedges(skey)[0]), open_strip=open_strip, project=project)[0] for _ in range(n - 1)]


def split_strips(mesh: QuadMesh, skey_to_n: dict[int, int], open_strip: bool = True, project: Callable[[list[float]], list[float]] | None = None) -> dict[int, list[int]]:
    """Refine several strips. ``open_strip``/``project`` as in ``add_strip``.

    Parameters
    ----------
    mesh : QuadMesh
        A quad mesh.
    skey_to_n : dict
        Strip keys pointing to the number of strips to refine each one into.

    Returns
    -------
    dict
        Each split strip key, pointing to the keys of the strips refining it.
    """
    return {skey: split_strip(mesh, skey, n, open_strip=open_strip, project=project) for skey, n in skey_to_n.items()}


def polyedge_sides(mesh: QuadMesh, polyedge: list[int]) -> dict[int, tuple[list[int], list[int]] | None]:
    """``{vertex: (left neighbours, right neighbours)}`` across the polyedge, in XY."""
    closed = polyedge[0] == polyedge[-1]
    seq = polyedge[:-1] if closed else polyedge
    count = len(seq)
    out = {}
    for i, vkey in enumerate(seq):
        if closed:
            prev, nxt = seq[(i - 1) % count], seq[(i + 1) % count]
        else:
            prev = seq[i - 1] if i > 0 else None
            nxt = seq[i + 1] if i < count - 1 else None
        point = mesh.vertex_coordinates(vkey)
        a = mesh.vertex_coordinates(prev) if prev is not None else point
        b = mesh.vertex_coordinates(nxt) if nxt is not None else point
        dx, dy = b[0] - a[0], b[1] - a[1]
        if dx * dx + dy * dy < 1e-18:
            out[vkey] = None
            continue
        left, right = [], []
        for nbr in mesh.vertex_neighbors(vkey):
            if nbr == prev or nbr == nxt:
                continue
            q = mesh.vertex_coordinates(nbr)
            side = dx * (q[1] - point[1]) - dy * (q[0] - point[0])
            (left if side > 0.0 else right).append(nbr)
        out[vkey] = (left, right)
    return out


def open_added_strip(
    mesh: QuadMesh,
    old_to_new: dict[int, tuple[int, int]],
    sides: dict[int, tuple[list[int], list[int]] | None],
    on_boundary: Iterable[int] = (),
    project: Callable[[list[float]], list[float]] | None = None,
) -> int:
    """Give a new strip its width: place each new pair at thirds of the span across the old vertex.

    Only the new pairs move; boundary pairs go through ``project``. Returns the number repositioned.

    Parameters
    ----------
    mesh : QuadMesh
        The mesh after a topology-only ``add_strip``.
    old_to_new : dict
        ``{old vertex: (copy, copy)}``, the second return value of ``add_strip``.
    sides : dict
        ``polyedge_sides`` of the polyedge, measured BEFORE ``add_strip``.
    on_boundary : iterable, optional
        Old vertices that were on the boundary.
    project : callable, optional
        ``project(xyz) -> xyz`` applied to the new pair at a boundary vertex.

    Returns
    -------
    int
        The number of pairs actually repositioned. Planar (xy): z is set to 0.
    """
    on_boundary = set(on_boundary)
    opened = 0
    for old, pair in old_to_new.items():
        info = sides.get(old)
        if info is None or len(pair) != 2 or pair[0] == pair[1]:
            continue
        left_keys, right_keys = info
        first, second = pair

        # Which copy took which side is MEASURED on the result rather than
        # assumed from the walk's left/right convention: the copy adjacent to a
        # left-hand neighbour is the left one.
        if any(k in mesh.halfedge.get(first, {}) for k in left_keys):
            low, high = first, second
        elif any(k in mesh.halfedge.get(second, {}) for k in left_keys):
            low, high = second, first
        elif any(k in mesh.halfedge.get(first, {}) for k in right_keys):
            low, high = second, first
        elif any(k in mesh.halfedge.get(second, {}) for k in right_keys):
            low, high = first, second
        else:
            continue

        here = mesh.vertex_coordinates(low)
        if old in on_boundary and _open_at_corner(mesh, low, high, left_keys, right_keys, project):
            opened += 1
            continue
        a = _centroid([mesh.vertex_coordinates(k) for k in left_keys], here)
        b = _centroid([mesh.vertex_coordinates(k) for k in right_keys], here)
        if distance_point_point(a, b) <= 1e-12:
            continue

        one = [a[0] + (b[0] - a[0]) / 3.0, a[1] + (b[1] - a[1]) / 3.0, 0.0]
        two = [a[0] + 2.0 * (b[0] - a[0]) / 3.0, a[1] + 2.0 * (b[1] - a[1]) / 3.0, 0.0]
        if project is not None and old in on_boundary:
            one, two = project(one), project(two)

        mesh.vertex_attributes(low, "xyz", [one[0], one[1], 0.0])
        mesh.vertex_attributes(high, "xyz", [two[0], two[1], 0.0])
        opened += 1
    return opened


def _open_at_corner(
    mesh: QuadMesh,
    low: int,
    high: int,
    left_keys: list[int],
    right_keys: list[int],
    project: Callable[[list[float]], list[float]] | None = None,
    min_turn: float = 30.0,
) -> bool:
    """At a boundary corner, keep one copy on the corner and slide the other half-way along the longer wall.

    The thirds rule would put both copies on the chord across the corner, inside the domain.
    Returns whether it applied: one wall neighbour per side, turning by more than ``min_turn`` degrees.
    """
    # ``k in halfedge[copy]``: across a hole corner a side can hold a neighbour of the other copy
    walls = [[k for k in keys if k in mesh.halfedge[copy] and mesh.is_edge_on_boundary(copy, k)] for copy, keys in ((low, left_keys), (high, right_keys))]
    if len(walls[0]) != 1 or len(walls[1]) != 1:
        return False
    here = mesh.vertex_coordinates(low)
    move = _corner_move(here, mesh.vertex_coordinates(walls[0][0]), mesh.vertex_coordinates(walls[1][0]), project, min_turn)
    if move is None:
        return False
    side, point = move
    mesh.vertex_attributes((low, high)[side], "xyz", [point[0], point[1], 0.0])
    return True


def _corner_move(
    here: list[float],
    left: list[float],
    right: list[float],
    project: Callable[[list[float]], list[float]] | None = None,
    min_turn: float = 30.0,
) -> tuple[int, list[float]] | None:
    """``(0 for the left copy or 1 for the right, where it goes)`` at a corner, or None where the walls turn too little."""
    if angle_vectors(subtract_vectors(here, left), subtract_vectors(right, here), deg=True) <= min_turn:
        return None
    if distance_point_point(here, left) >= distance_point_point(here, right):
        side, target = 0, left
    else:
        side, target = 1, right
    point = [(here[0] + target[0]) / 2.0, (here[1] + target[1]) / 2.0, 0.0]
    if project is not None:
        point = project(point)
    return side, point


def _centroid(points: list[list[float]], fallback: list[float]) -> list[float]:
    """The average of ``points``, or ``fallback`` when there are none."""
    if not points:
        return list(fallback)
    n = float(len(points))
    return [sum(p[0] for p in points) / n, sum(p[1] for p in points) / n, 0.0]


def _boundary_vertex_set(mesh: QuadMesh) -> set[int]:
    """Every vertex with a faceless halfedge -- holes included, no loop walk.

    The same rule as ``editing.editor.boundary_vertex_set``, repeated here
    because the grammar must not import from ``editing``.
    """
    out = set()
    for u, nbrs in mesh.halfedge.items():
        for v, fkey in nbrs.items():
            if fkey is None:
                out.add(u)
                out.add(v)
    return out


# ==============================================================================
# opening per visit: strips along a wall, and strips that cross themselves
# ==============================================================================


def _legs(seq: list[int], closed: bool, j: int) -> tuple[int | None, int | None]:
    """The vertices before and after visit ``j``; None past an open end."""
    if closed:
        return seq[j - 1], seq[(j + 1) % len(seq)]
    return (seq[j - 1] if j > 0 else None), (seq[j + 1] if j < len(seq) - 1 else None)


def _touches_itself(mesh: QuadMesh, seq: list[int], closed: bool) -> bool:
    """Whether a polyedge vertex has a neighbour on the polyedge other than the ones it runs to."""
    on_polyedge = set(seq)
    for j, vkey in enumerate(seq):
        legs = _legs(seq, closed, j)
        if any(nbr in on_polyedge and nbr not in legs for nbr in mesh.vertex_neighbors(vkey)):
            return True
    return False


def _along_wall(mesh: QuadMesh, seq: list[int], closed: bool, j: int, boundary: set[int]) -> bool:
    """Whether the polyedge arrives at or leaves visit ``j`` along a boundary edge."""
    vkey = seq[j]
    if vkey not in boundary:
        return False
    return any(k is not None and mesh.is_edge_on_boundary(vkey, k) for k in _legs(seq, closed, j))


def _sector(mesh: QuadMesh, vkey: int, start: int | None, stop: int | None) -> dict[str, Any]:
    """Faces, neighbours and outside met turning counter-clockwise about ``vkey`` from leg ``start`` to leg ``stop``.

    A None leg is the gap in the boundary: the turn starts after it, or ends at it.
    """
    faces, nbrs, outside = [], [], False
    if start is None:
        nbr = next(u for u in mesh.halfedge[vkey] if mesh.halfedge[u][vkey] is None)
        if nbr == stop:
            return {"faces": faces, "nbrs": nbrs, "outside": True}
        nbrs.append(nbr)
        outside = True
    else:
        nbr = start
    for _ in range(len(mesh.halfedge[vkey]) + 1):
        fkey = mesh.halfedge[vkey][nbr]
        if fkey is None:
            outside = True
            if stop is None:
                break
            nbr = next(u for u in mesh.halfedge[vkey] if mesh.halfedge[u][vkey] is None)
        else:
            faces.append(fkey)
            nbr = mesh.face_vertex_before(fkey, vkey)
        if nbr == stop:
            break
        nbrs.append(nbr)
    return {"faces": faces, "nbrs": nbrs, "outside": outside}


def _visit_plan(
    mesh: QuadMesh,
    seq: list[int],
    closed: bool,
    j: int,
    boundary: set[int],
    project: Callable[[list[float]], list[float]] | None = None,
) -> dict[str, Any]:
    """Where the left and right copies of visit ``j`` go, and the vertices that tell them apart, on the unsplit mesh."""
    vkey = seq[j]
    prev, nxt = _legs(seq, closed, j)
    here = mesh.vertex_coordinates(vkey)
    if vkey not in boundary and (prev is None or nxt is None):
        # an end at a pole: not copied, so nothing to place
        return {"here": here, "marks": (set(), set()), "targets": None}
    sides = (_sector(mesh, vkey, nxt, prev), _sector(mesh, vkey, prev, nxt))
    legs = {vkey, prev, nxt}
    marks = tuple(set(k for fkey in side["faces"] for k in mesh.face_vertices(fkey)) - legs for side in sides)
    if _along_wall(mesh, seq, closed, j, boundary):
        targets = tuple(_wall_target(mesh, vkey, prev, nxt, side, here, project) for side in sides)
    else:
        targets = _thirds_targets(mesh, vkey, sides, here, vkey in boundary, project)
    return {"here": here, "marks": marks, "targets": targets}


def _thirds_targets(
    mesh: QuadMesh,
    vkey: int,
    sides: tuple[dict[str, Any], dict[str, Any]],
    here: list[float],
    on_boundary: bool,
    project: Callable[[list[float]], list[float]] | None = None,
) -> tuple[list[float], list[float]] | None:
    """``open_added_strip``'s thirds and corner rules for one visit, read off the unsplit mesh."""
    left_keys, right_keys = sides[0]["nbrs"], sides[1]["nbrs"]
    if on_boundary:
        walls = [[k for k in keys if mesh.is_edge_on_boundary(vkey, k)] for keys in (left_keys, right_keys)]
        if len(walls[0]) == 1 and len(walls[1]) == 1:
            move = _corner_move(here, mesh.vertex_coordinates(walls[0][0]), mesh.vertex_coordinates(walls[1][0]), project)
            if move is not None:
                side, point = move
                return (point, list(here)) if side == 0 else (list(here), point)
    a = _centroid([mesh.vertex_coordinates(k) for k in left_keys], here)
    b = _centroid([mesh.vertex_coordinates(k) for k in right_keys], here)
    if distance_point_point(a, b) <= 1e-12:
        return None
    one = [a[0] + (b[0] - a[0]) / 3.0, a[1] + (b[1] - a[1]) / 3.0, 0.0]
    two = [a[0] + 2.0 * (b[0] - a[0]) / 3.0, a[1] + 2.0 * (b[1] - a[1]) / 3.0, 0.0]
    if project is not None and on_boundary:
        one, two = project(one), project(two)
    return one, two


def _wall_target(
    mesh: QuadMesh,
    vkey: int,
    prev: int | None,
    nxt: int | None,
    side: dict[str, Any],
    here: list[float],
    project: Callable[[list[float]], list[float]] | None = None,
) -> list[float]:
    """Where the copy on ``side`` of a visit along a wall goes: a third of the way into its own side.

    Nothing on that side: it stays on the wall. The outside on that side: it slides along the wall.
    """
    if not side["faces"]:
        return list(here)
    if side["outside"]:
        walls = [k for k in side["nbrs"] if mesh.is_edge_on_boundary(vkey, k)]
        if len(walls) != 1:
            return list(here)
        wall = mesh.vertex_coordinates(walls[0])
        point = [here[0] + (wall[0] - here[0]) / 3.0, here[1] + (wall[1] - here[1]) / 3.0, 0.0]
        return project(point) if project is not None else point
    if side["nbrs"]:
        c = _centroid([mesh.vertex_coordinates(k) for k in side["nbrs"]], here)
        return [here[0] + (c[0] - here[0]) / 3.0, here[1] + (c[1] - here[1]) / 3.0, 0.0]
    # one face between the two legs: into it, a third along each leg
    a, b = mesh.vertex_coordinates(prev), mesh.vertex_coordinates(nxt)
    return [here[0] + (a[0] - here[0] + b[0] - here[0]) / 3.0, here[1] + (a[1] - here[1] + b[1] - here[1]) / 3.0, 0.0]


def _face_roots(mesh: QuadMesh, vkey: int, root: dict[int, int]) -> set[int]:
    """The input vertices behind every vertex of the faces around ``vkey``."""
    return set(root.get(k, k) for fkey in mesh.vertex_faces(vkey) for k in mesh.face_vertices(fkey))


def _open_by_visits(
    mesh: QuadMesh,
    seq: list[int],
    walked: list[int],
    root: dict[int, int],
    steps: list[tuple[int, int, int, set[int], set[int]]],
    plans: list[dict[str, Any]],
) -> None:
    """Place every copy left in the mesh by its visits' plans; a copy split twice adds both displacements."""
    # which visit each vertex of the walk belongs to: a revisit may expand into two
    visit, j = [], -1
    for vkey in walked:
        if j < 0 or seq[j] != root.get(vkey, vkey):
            j += 1
        visit.append(j)

    made = {}  # copy -> (step, 0 for the left copy or 1 for the right; None if unknown)
    for s, (at, v1, v2, roots1, roots2) in enumerate(steps):
        left, right = plans[visit[at]]["marks"]
        if roots1 & left or roots2 & right:
            made[v1], made[v2] = (s, 0), (s, 1)
        elif roots2 & left or roots1 & right:
            made[v1], made[v2] = (s, 1), (s, 0)
        else:
            made[v1], made[v2] = (s, None), (s, None)

    for _at, v1, v2, _r1, _r2 in steps:
        for leaf in (v1, v2):
            if leaf not in mesh.vertex:
                continue
            moves, vkey = [], leaf
            while vkey in made:
                s, side = made[vkey]
                at = steps[s][0]
                plan = plans[visit[at]]
                if side is not None and plan["targets"] is not None:
                    moves.append((plan["here"], plan["targets"][side]))
                vkey = walked[at]
            if not moves:
                continue
            if len(moves) == 1:
                point = moves[0][1]
            else:
                here = moves[0][0]
                point = [here[0] + sum(t[0] - h[0] for h, t in moves), here[1] + sum(t[1] - h[1] for h, t in moves)]
            mesh.vertex_attributes(leaf, "xyz", [point[0], point[1], 0.0])


def _update_strip_data_by_roots(mesh: QuadMesh, root: dict[int, int], steps: list[tuple[int, int, int, set[int], set[int]]]) -> int:
    """``update_strip_data`` for a polyedge that passes a vertex twice: re-collect each touched strip from an edge descending from its own."""
    by_roots = {}
    for u, v in mesh.edges():
        ru, rv = root.get(u, u), root.get(v, v)
        by_roots.setdefault((ru, rv), (u, v))
        by_roots.setdefault((rv, ru), (v, u))
    strips = mesh.attributes["strips"]
    for skey, edges in list(strips.items()):
        if all(u in mesh.vertex and v in mesh.vertex for u, v in edges):
            continue
        seed = next((by_roots[(u, v)] for u, v in edges if u != v and (u, v) in by_roots), None)
        if seed is not None:
            strips[skey] = mesh.collect_strip(*seed)
    n = max(strips) + 1
    rung = next((v1, v2) for _at, v1, v2, _r1, _r2 in steps if v1 in mesh.halfedge and v2 in mesh.halfedge[v1])
    strips[n] = mesh.collect_strip(*rung)
    return n


# ==============================================================================
# the flatness check
# ==============================================================================


def _signed_area(mesh: QuadMesh, fkey: int) -> float:
    """The face's area in XY, positive counter-clockwise."""
    points = [mesh.vertex_coordinates(vkey) for vkey in mesh.face_vertices(fkey)]
    return 0.5 * sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(points, points[1:] + points[:1]))


def _orientation(mesh: QuadMesh, seq: list[int]) -> tuple[int, float]:
    """The sign the faces around the polyedge turn with, and the area below which one counts as collapsed."""
    areas = [_signed_area(mesh, fkey) for fkey in set(f for vkey in seq for f in mesh.vertex_faces(vkey))]
    if not areas:
        return 0, 0.0
    total = sum(areas)
    sign = 1 if total > 0.0 else -1 if total < 0.0 else 0
    return sign, 1e-9 * sum(abs(a) for a in areas) / len(areas)


def _folded_faces(mesh: QuadMesh, vertices: Iterable[int], sign: int, tol: float) -> list[int]:
    """The faces around ``vertices`` turning against ``sign`` or with no more area than ``tol``."""
    if not sign:
        return []
    faces = set(f for vkey in vertices if vkey in mesh.vertex for f in mesh.vertex_faces(vkey))
    return [fkey for fkey in faces if sign * _signed_area(mesh, fkey) <= tol]


def update_strip_data(mesh: QuadMesh, full_updated_polyedge: list[int], old_vkeys_to_new_vkeys: dict[int, tuple[int, int]], closed: bool = False) -> int:
    """Bring ``attributes['strips']`` up to date after a strip was added, closing edge included."""
    sequence = full_updated_polyedge + full_updated_polyedge[:1] if closed else full_updated_polyedge

    # orthogonal strips
    orth_to_update = {}
    for old_u, old_v in pairwise(sequence):
        new_u = old_vkeys_to_new_vkeys[old_u][0]
        new_v = old_vkeys_to_new_vkeys[old_v][0]
        skey = mesh.edge_strip((old_u, old_v))
        edges = mesh.collect_strip(new_u, new_v)
        orth_to_update[skey] = edges
    for skey, edges in orth_to_update.items():
        mesh.attributes["strips"][skey] = edges

    # parallel strips; a pole end kept its key, so it needs no update
    copied = {old for old, pair in old_vkeys_to_new_vkeys.items() if pair[0] != old}
    paral_to_update = {}
    for skey, edges in mesh.attributes["strips"].items():
        if skey not in orth_to_update:
            for u, v in edges:
                if u in copied:
                    u, v = v, u
                elif v in copied:
                    u, v = u, v
                else:
                    continue
                new_v = [vkey for vkey in old_vkeys_to_new_vkeys[v] if vkey in mesh.halfedge[u]][0]
                new_edges = mesh.collect_strip(u, new_v)
                paral_to_update[skey] = new_edges
                break
    for skey, edges in paral_to_update.items():
        mesh.attributes["strips"][skey] = edges

    # self strip
    n = max(mesh.attributes["strips"]) + 1
    strip_edges = [tuple(old_vkeys_to_new_vkeys[vkey]) for vkey in full_updated_polyedge]
    if any(u == v for u, v in strip_edges):
        # with a pole end, face each edge forward as ``collect_strip`` does:
        # ``PseudoQuadMesh.strip_faces`` reads the pseudo-quad off that side
        strip_edges = [(v, u) for u, v in strip_edges]
    mesh.attributes["strips"][n] = strip_edges

    return n


def strip_polyedge_update(mesh: QuadMesh, polyedge: list[int], vertex_modifications: dict[int, tuple[int, int]]) -> list[int]:
    """Rewrite ``polyedge`` in terms of the vertices an insertion left behind.

    Parameters
    ----------
    mesh : QuadMesh
        A quad mesh.
    polyedge : list[int]
        A polyedge, as a list of the OLD vertex keys.
    vertex_modifications : dict
        Old vertex keys pointing to the new ones that replaced them -- the
        second return value of ``add_strip``.

    Returns
    -------
    list
        The shortest path that visits the replacements in the original order.
    """
    closed = polyedge[0] == polyedge[-1]

    if closed:
        polyedge = polyedge[:-1]

    # update polyedge with candidate vertices
    polyedge_modifications = {vkey: (vertex_modifications[vkey] if vkey in vertex_modifications else [vkey]) for vkey in polyedge}
    # list all candidate vertices to form new polyedge
    candidate_vertices = tuple(set([vkey for vkeys in polyedge_modifications.values() for vkey in vkeys]))
    # adjacency restricted to candidate vertices
    adjacency = {vkey: [nbr for nbr in mesh.vertex_neighbors(vkey) if nbr in candidate_vertices] for vkey in mesh.vertices() if vkey in candidate_vertices}

    # an extremity that was not copied, like a pole, stays a valid one
    def is_extremity(vkey, candidates):
        return mesh.is_vertex_on_boundary(vkey) or len(set(candidates)) == 1

    # for each combination of boundary vertex extremities, get the shortest
    # valid path through the modified vertices of the polyedges
    shortest_polyedge = None
    # start vertices on boundary
    for vkey_start in polyedge_modifications[polyedge[0]]:
        if is_extremity(vkey_start, polyedge_modifications[polyedge[0]]):
            # end vertices on boundary
            for vkey_end in polyedge_modifications[polyedge[-1]]:
                if is_extremity(vkey_end, polyedge_modifications[polyedge[-1]]):
                    # iterate through all paths between start and end vertices
                    # starting by the shortest
                    for candidate_polyedge in breadth_first_paths(adjacency, vkey_start, vkey_end):
                        is_valid = True
                        # if was initailly closed, make sure that temporary end
                        # is adjacent to start
                        if closed:
                            if candidate_polyedge[0] not in mesh.vertex_neighbors(candidate_polyedge[-1]):
                                continue
                        # check that vertices in path come from the modified
                        # vertices of the polyedge in the same order
                        i = 0
                        for vkey in candidate_polyedge:
                            if vkey in polyedge_modifications[polyedge[i]]:
                                continue
                            elif vkey in polyedge_modifications[polyedge[i + 1]]:
                                i += 1
                            else:
                                is_valid = False
                                break
                        # update if shorter
                        if is_valid:
                            if shortest_polyedge is None or len(shortest_polyedge) > len(candidate_polyedge):
                                shortest_polyedge = candidate_polyedge
                            break

    if closed:
        shortest_polyedge.append(shortest_polyedge[0])

    return shortest_polyedge


def sort_faces(mesh: QuadMesh, u: int | None, v: int, w: int | None) -> list[list[int]] | list[int]:

    sorted_faces = [[], []]
    k = 0
    vertex_faces = mesh.vertex_faces(v, ordered=True, include_none=True)
    f0 = mesh.halfedge[w][v] if w is not None else None
    i0 = vertex_faces.index(f0)
    vertex_faces = vertex_faces[i0:] + vertex_faces[:i0]

    for face in vertex_faces:
        if face is not None:
            sorted_faces[k].append(face)

        if u is None:
            if face is None:
                k = 1 - k
        elif face == mesh.halfedge[v][u]:
            k = 1 - k

    # indeterminate exception if u == w
    if u == w:
        return [face for faces in sorted_faces for face in faces]
    else:
        return sorted_faces


def adjacency_from_to_via_vertices(mesh: QuadMesh, from_vkey: int, to_vkey: int, via_vkeys: list[int]) -> dict[int, dict[int, Any]]:
    # get mesh adjacency constraiend to from_vkey and via_keys, via_keys and via_keys, and via_vkeys and to_vkey

    all_vkeys = set([from_vkey, to_vkey] + via_vkeys)
    adjacency = {}
    for vkey, nbrs in mesh.adjacency.items():
        if vkey not in all_vkeys:
            continue
        else:
            sub_adj = {}
            for nbr, face in nbrs.items():
                if nbr not in all_vkeys or (vkey == from_vkey and nbr == to_vkey) or (vkey == to_vkey and nbr == from_vkey):
                    continue
                else:
                    sub_adj.update({nbr: face})
            adjacency.update({vkey: sub_adj})
    return adjacency


def polyedge_from_to_via_vertices(mesh: QuadMesh, from_vkey: int, to_vkey: int, via_vkeys: list[int]) -> list[int]:
    # return shortest polyedge from_vkey to_vkey via_vkeys

    adjacency = adjacency_from_to_via_vertices(mesh, from_vkey, to_vkey, via_vkeys)
    return next(breadth_first_paths(adjacency, from_vkey, to_vkey))


def is_polyedge_valid_for_strip_addition(mesh: QuadMesh, polyedge: list[int]) -> bool:
    if len(polyedge) > 2:
        if polyedge[0] == polyedge[-1]:
            return True
        start_pole, end_pole = pole_ends(mesh, polyedge)
        if (start_pole or mesh.is_vertex_on_boundary(polyedge[0])) and (end_pole or mesh.is_vertex_on_boundary(polyedge[-1])):
            return True
    return False


def pole_ends(mesh: QuadMesh, polyedge: list[int]) -> tuple[bool, bool]:
    """Whether each end of an open polyedge is an existing pole of a pseudo-quad mesh.

    Note
    ----
    Thesis Sec. 8.2 also turns an interior non-pole end into a new pole; that is not done here.
    """
    face_pole = mesh.attributes.get("face_pole")
    if not face_pole or not hasattr(mesh, "is_face_pseudo_quad") or polyedge[0] == polyedge[-1]:
        return False, False
    poles = set(face_pole.values())
    return polyedge[0] in poles, polyedge[-1] in poles
