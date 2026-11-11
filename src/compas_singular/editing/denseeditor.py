"""Hand-editing the final dense mesh, with no Rhino in it.

``DenseMeshEditor`` moves and removes vertices, faces and edges, draws edges in,
and adds or removes strips where the mesh is quads. Design notes: ``design_notes/editing.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any

# compas 2.x dropped ``mesh_smooth_centroid`` from the ``compas.datastructures``
# namespace, but not from the module it lives in.
from compas.datastructures.mesh.smoothing import mesh_smooth_centroid
from compas.geometry import is_point_in_polygon_xy
from compas.itertools import pairwise
from compas_singular.datastructures.mesh_quad.grammar.add_strip import add_strip as _grammar_add_strip
from compas_singular.datastructures.mesh_quad.grammar.add_strip import is_polyedge_valid_for_strip_addition
from compas_singular.datastructures.mesh_quad.grammar.add_strip import split_strips
from compas_singular.editing.editor import MeshEditor

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = ["DenseMeshEditor", "RELAX_ITERATIONS", "DRAW_SNAP"]


#: Smoothing passes used to open a newly added strip. ``add_strip`` creates its
#: two vertices on top of the one they replace, so without this the new strip
#: has zero width.
RELAX_ITERATIONS = 50

#: How close, as a fraction of the mean edge length, a drawn line has to pass to
#: an existing vertex to go THROUGH it rather than beside it. Past this the line
#: splits the edge instead, however short the piece it cuts off.
DRAW_SNAP = 0.05


class DenseMeshEditor(MeshEditor):
    """A dense mesh being hand-edited. No Rhino, no prompts, no prints.

    Parameters
    ----------
    mesh : compas_singular.datastructures.QuadMesh
        The mesh to edit, polygons allowed. A ``QuadMesh`` rather than a plain
        ``compas.datastructures.Mesh``, because the strip operations need
        its polyedges and strips.
    walls : list, optional
        The domain walls, as point lists, closed polylines or
        ``BoundaryLoop`` instances. A boundary vertex that moves is
        projected onto the nearest one. Without walls it is left where it was put.
    relax_iterations : int, optional
        Default number of smoothing passes for ``relax``. Defaults to
        ``RELAX_ITERATIONS``.

    Attributes
    ----------
    mesh : compas_singular.datastructures.QuadMesh
        The mesh as it currently stands.
    last_reason : str
        Why the last operation was refused. Empty when nothing was.
    last_addition : dict
        What the last successful ``add_line`` did.
    last_deletion : dict
        What the last successful ``remove_line`` did.

    Examples
    --------
    >>> editor = DenseMeshEditor(mesh, walls=loops)
    >>> ok, notes = editor.remove_edge((12, 13))
    >>> ok, notes = editor.draw_edges([[0.0, 2.5, 0.0], [10.0, 2.5, 0.0]])
    >>> editor.undo()
    """

    def __init__(self, mesh: "QuadMesh", walls: list[Any] | None = None, relax_iterations: int | None = None) -> None:
        if not hasattr(mesh, "collect_polyedges"):
            raise TypeError(
                "DenseMeshEditor needs a QuadMesh -- the strip operations select from its polyedges and strips, which a plain Mesh does not have. Got {}.".format(
                    type(mesh).__name__
                )
            )
        # ``work_on_copy=False``: this stage has no commit to transplant a copy
        # back through, so a copy would silently strand the edit.
        super(DenseMeshEditor, self).__init__(mesh, walls=walls, work_on_copy=False)
        self.relax_iterations = RELAX_ITERATIONS if relax_iterations is None else relax_iterations
        self.last_addition = {}

    # ------------------------------------------------------------------
    # queries -- none of these mutate anything
    # ------------------------------------------------------------------

    def non_quad_faces(self, mesh: "QuadMesh | None" = None) -> list[int]:
        """Every face that is not four-sided. Empty on a clean quad mesh."""
        mesh = self.mesh if mesh is None else mesh
        return [fkey for fkey in mesh.faces() if len(mesh.face_vertices(fkey)) != 4]

    def polyedge_through(self, edge: tuple[int, int], mesh: "QuadMesh | None" = None) -> tuple[int | None, list[int] | None]:
        """``(pkey, polyedge)`` for the polyedge containing ``edge``, as a new list.

        Re-collected every time: an added or deleted strip renumbers the keys.
        The list is a COPY because ``add_strip`` pops from the list it is given.
        """
        mesh = self.mesh if mesh is None else mesh
        u, v = edge
        for pkey, polyedge in dict(mesh.collect_polyedges()).items():
            for a, b in pairwise(polyedge):
                if (a, b) == (u, v) or (a, b) == (v, u):
                    return pkey, list(polyedge)
        return None, None

    def non_manifold_vertices(self, mesh: "QuadMesh | None" = None) -> list[int]:
        """Vertices where two separate fans of faces meet at a single point."""
        mesh = self.mesh if mesh is None else mesh
        return [vkey for vkey in mesh.vertices() if sum(1 for fkey in mesh.halfedge[vkey].values() if fkey is None) > 1]

    def face_at(self, point: list[float], mesh: "QuadMesh | None" = None) -> int | None:
        """The face whose outline contains ``point`` in plan, or ``None``.

        Tested in XY, so it also finds a face of a mesh that has been given
        height, as long as the mesh does not fold back over itself in plan.
        """
        mesh = self.mesh if mesh is None else mesh
        for fkey in mesh.faces():
            polygon = [mesh.vertex_coordinates(v) for v in mesh.face_vertices(fkey)]
            if is_point_in_polygon_xy(point, polygon):
                return fkey
        return None

    def _non_quad_on_strip(self, edge: tuple[int, int], mesh: "QuadMesh | None" = None) -> int | None:
        """A non-quad face the strip through ``edge`` crosses or ends at, or ``None`` if it is all quads."""
        mesh = self.mesh if mesh is None else mesh
        for a, b in mesh.collect_strip(*edge):
            for fkey in (mesh.halfedge[a].get(b), mesh.halfedge[b].get(a)):
                if fkey is not None and len(mesh.face_vertices(fkey)) != 4:
                    return fkey
        return None

    def _is_edge(self, edge: tuple[int, int], mesh: "QuadMesh | None" = None) -> bool:
        mesh = self.mesh if mesh is None else mesh
        u, v = edge
        return u in mesh.halfedge and v in mesh.halfedge[u]

    # ------------------------------------------------------------------
    # housekeeping after a topological edit
    # ------------------------------------------------------------------

    @staticmethod
    def _cull(mesh: "QuadMesh") -> list[int]:
        """Remove edges and vertices no face uses any more. Returns the removed vertex keys.

        Edges first: ``Mesh.delete_vertex`` can leave faceless edges behind a deleted n-gon.
        """
        for u in list(mesh.halfedge):
            for v in list(mesh.halfedge[u]):
                if mesh.halfedge[u].get(v, 0) is None and mesh.halfedge.get(v, {}).get(u, 0) is None:
                    del mesh.halfedge[u][v]
                    del mesh.halfedge[v][u]
        unused = [vkey for vkey in mesh.vertices() if not mesh.halfedge[vkey]]
        mesh.remove_unused_vertices()
        return unused

    @staticmethod
    def _forget_stale_poles(mesh: "QuadMesh") -> int:
        """Drop ``face_pole`` entries whose face no longer is that pole's triangle."""
        face_pole = mesh.attributes.get("face_pole")
        if not face_pole:
            return 0
        stale = [fkey for fkey, pole in face_pole.items() if fkey not in mesh.face or len(mesh.face_vertices(fkey)) != 3 or pole not in mesh.face_vertices(fkey)]
        for fkey in stale:
            del face_pole[fkey]
        return len(stale)

    def _adopt(self, work: "QuadMesh") -> None:
        self._forget_stale_poles(work)
        self.mesh = work
        self.edited = True

    # ------------------------------------------------------------------
    # remove a vertex / a face -- the compas operations, as they are
    # ------------------------------------------------------------------

    def remove_vertex(self, vkey: int) -> tuple[bool, dict[str, Any]]:
        """Delete a vertex and every face around it. ``(ok, notes)``.

        Not refused when it leaves a non-manifold vertex; ``notes['non_manifold']`` lists them.
        """
        mesh = self.mesh
        if vkey not in mesh.vertex:
            return self._refuse("vertex {} is not in the mesh".format(vkey))
        faces = [fkey for fkey in mesh.vertex_faces(vkey) if fkey is not None]
        if len(faces) >= mesh.number_of_faces():
            return self._refuse("that would delete every face of the mesh")

        mesh.delete_vertex(vkey)
        culled = self._cull(mesh)
        self._adopt(mesh)
        return self._accept(removed=vkey, faces_removed=len(faces), vertices_culled=len(culled), non_manifold=self.non_manifold_vertices(), faces=mesh.number_of_faces())

    def remove_face(self, fkey: int) -> tuple[bool, dict[str, Any]]:
        """**Delete one face.** ``(ok, notes)``.

        ``Mesh.delete_face``. Refused only for a face that is not there, and for
        the last face of the mesh.
        """
        mesh = self.mesh
        if fkey not in mesh.face:
            return self._refuse("face {} is not in the mesh".format(fkey))
        if mesh.number_of_faces() == 1:
            return self._refuse("that is the last face of the mesh")

        degree = len(mesh.face_vertices(fkey))
        mesh.delete_face(fkey)
        culled = self._cull(mesh)
        self._adopt(mesh)
        return self._accept(removed=fkey, degree=degree, vertices_culled=len(culled), non_manifold=self.non_manifold_vertices(), faces=mesh.number_of_faces())

    # ------------------------------------------------------------------
    # remove an edge -- merge the faces either side
    # ------------------------------------------------------------------

    def remove_edge(self, edge: tuple[int, int]) -> tuple[bool, dict[str, Any]]:
        """Remove one edge by merging the two faces either side of it. ``(ok, notes)``.

        A boundary edge deletes its one face instead, with ``notes['merged']`` set to ``None``.
        """
        if not self._is_edge(edge):
            return self._refuse("{} is not an edge of the mesh".format(tuple(edge)))
        u, v = edge
        mesh = self.mesh
        left, right = mesh.halfedge[u][v], mesh.halfedge[v][u]

        if left is None or right is None:
            fkey = left if left is not None else right
            ok, notes = self.remove_face(fkey)
            if ok:
                notes = dict(notes, edge=(u, v), merged=None, on_boundary=True)
            return ok, notes

        if left == right:
            return self._refuse("the same face lies on both sides of edge {} -- there is nothing to merge".format((u, v)))

        work = mesh.copy()
        degrees = (len(work.face_vertices(left)), len(work.face_vertices(right)))
        merged = work.merge_faces([left, right])
        if merged is None:
            return self._refuse("the faces either side of edge {} could not be merged".format((u, v)))
        corners = work.face_vertices(merged)
        if len(corners) < 3 or len(set(corners)) != len(corners):
            return self._refuse("merging the faces either side of edge {} would make a face that touches itself".format((u, v)))

        culled = self._cull(work)
        self._adopt(work)
        return self._accept(
            edge=(u, v),
            merged=merged,
            degrees=degrees,
            degree=len(corners),
            on_boundary=False,
            vertices_culled=len(culled),
            non_manifold=self.non_manifold_vertices(),
            faces=work.number_of_faces(),
        )

    # ------------------------------------------------------------------
    # draw new edges -- cut a polyline into the mesh
    # ------------------------------------------------------------------

    def draw_edges(self, points: list[list[float]], snap: float | None = None) -> tuple[bool, dict[str, Any]]:
        """Cut a drawn polyline into the mesh, splitting every edge and face it crosses (in XY). ``(ok, notes)``.

        An edge cannot end inside a face, so dangling or outside parts are left out and counted in ``notes``.

        Parameters
        ----------
        points : list
            The polyline, at least two points.
        snap : float, optional
            Distance within which the line goes through an existing vertex or
            lies on an existing edge. Defaults to ``DRAW_SNAP`` times the
            mean edge length.
        """
        points = [list(p) + [0.0] * (3 - len(p)) for p in points]
        if len(points) < 2:
            return self._refuse("a line needs at least two points")
        mesh = self.mesh
        if snap is None:
            snap = DRAW_SNAP * self.mean_edge()

        stations = self._stations(mesh, points, snap)
        runs, notes = self._runs(mesh, stations)
        if not runs:
            reason = "the line did not cut through any face"
            if notes["along_existing"]:
                reason += " -- it runs along edges that are already there"
            elif notes["outside"] or notes["trimmed"]:
                reason += " -- it has to cross a face from one edge or vertex to another"
            return self._refuse(reason)

        work = mesh.copy()
        keys = self._materialise(work, stations, runs, notes)
        for i, j, _fkey in runs:
            interior = [stations[k]["point"] for k in range(i + 1, j)]
            if self._split_face(work, keys[i], keys[j], interior, notes) is None:
                return self._refuse("lost track of the face between {} and {} while cutting -- nothing was changed".format(stations[i]["point"], stations[j]["point"]))

        if self.mesh.is_manifold() and not work.is_manifold():
            return self._refuse("the cut would leave the mesh non-manifold")
        self._adopt(work)
        notes["faces"] = work.number_of_faces()
        return self._accept(**notes)

    @staticmethod
    def _stations(mesh: "QuadMesh", points: list[list[float]], snap: float) -> list[dict[str, Any]]:
        """Where the line meets the mesh, in order: one station per edge crossing and per line point."""
        xyz = dict((v, mesh.vertex_coordinates(v)) for v in mesh.vertices())
        edges = list(mesh.edges())
        snap2 = snap * snap

        def near_vertex(point: list[float], candidates: Any) -> int | None:
            best, best_d = None, snap2
            for v in candidates:
                d = _d2(xyz[v], point)
                if d <= best_d:
                    best, best_d = v, d
            return best

        raw = []
        for i, (p, q) in enumerate(pairwise(points)):
            raw.append((float(i), None, p))
            lo_x, hi_x = min(p[0], q[0]) - snap, max(p[0], q[0]) + snap
            lo_y, hi_y = min(p[1], q[1]) - snap, max(p[1], q[1]) + snap
            for u, v in edges:
                a, b = xyz[u], xyz[v]
                if max(a[0], b[0]) < lo_x or min(a[0], b[0]) > hi_x or max(a[1], b[1]) < lo_y or min(a[1], b[1]) > hi_y:
                    continue
                hit = _segment_intersection_xy(p, q, a, b)
                if hit is not None:
                    raw.append((i + hit[0], (u, v, hit[1]), None))
        raw.append((float(len(points) - 1), None, points[-1]))
        raw.sort(key=lambda item: item[0])

        stations = []
        for s, crossing, point in raw:
            if crossing is not None:
                u, v, t = crossing
                a, b = xyz[u], xyz[v]
                point = [a[k] + t * (b[k] - a[k]) for k in range(3)]
                vkey = near_vertex(point, (u, v))
                if vkey is not None:
                    station = {"kind": "vertex", "vkey": vkey}
                else:
                    station = {"kind": "edge", "edge": (u, v), "t": t}
            else:
                vkey = near_vertex(point, xyz)
                if vkey is not None:
                    station = {"kind": "vertex", "vkey": vkey}
                else:
                    station = {"kind": "free"}
                    for u, v in edges:
                        t, c = _closest_on_segment_xy(point, xyz[u], xyz[v])
                        if 0.0 < t < 1.0 and _d2(c, point) <= snap2:
                            station = {"kind": "edge", "edge": (u, v), "t": t}
                            break
            if station["kind"] == "vertex":
                station["point"] = list(xyz[station["vkey"]])
            elif station["kind"] == "edge":
                u, v = station["edge"]
                t = station["t"]
                station["point"] = [xyz[u][k] + t * (xyz[v][k] - xyz[u][k]) for k in range(3)]
            else:
                station["point"] = list(point)
            station["s"] = s

            if stations and _same_place(stations[-1], station):
                # a stroke point exactly on a crossing, or a crossing at a vertex
                # found once per edge through it: keep the stronger reading
                if stations[-1]["kind"] == "free":
                    stations[-1] = station
                continue
            stations.append(station)

        # a free bend within snap of the station next to it would only add a
        # sliver edge
        cleaned = []
        for k, station in enumerate(stations):
            if station["kind"] == "free":
                nbrs = [stations[k - 1]] if k > 0 else []
                nbrs += [stations[k + 1]] if k + 1 < len(stations) else []
                if any(n["kind"] != "free" and _d2(n["point"], station["point"]) <= snap2 for n in nbrs):
                    continue
            cleaned.append(station)
        return cleaned

    def _runs(self, mesh: "QuadMesh", stations: list[dict[str, Any]]) -> tuple[list[tuple[int, int, int]], dict[str, Any]]:
        """Which stretches of the line split which face, decided before anything changes. ``([(i, j, fkey)], notes)``."""
        notes = {"split_edges": 0, "split_faces": 0, "new_vertices": 0, "along_existing": 0, "trimmed": 0, "outside": 0}
        anchors = [k for k, station in enumerate(stations) if station["kind"] != "free"]
        if not anchors:
            notes["trimmed"] = len(stations)
            return [], notes
        notes["trimmed"] = anchors[0] + (len(stations) - 1 - anchors[-1])

        def faces_of(station: dict[str, Any]) -> set[int]:
            if station["kind"] == "vertex":
                return set(f for f in mesh.vertex_faces(station["vkey"]) if f is not None)
            u, v = station["edge"]
            return set(f for f in (mesh.halfedge[u][v], mesh.halfedge[v][u]) if f is not None)

        def on_edge_of(station: dict[str, Any]) -> set[frozenset]:
            if station["kind"] == "vertex":
                return set(frozenset((station["vkey"], n)) for n in mesh.vertex_neighbors(station["vkey"]))
            return set([frozenset(station["edge"])])

        runs = []
        for i, j in pairwise(anchors):
            a, b = stations[i], stations[j]
            interior = stations[i + 1 : j]
            if a["kind"] == "vertex" and b["kind"] == "vertex" and a["vkey"] == b["vkey"]:
                continue
            if not interior and on_edge_of(a) & on_edge_of(b):
                notes["along_existing"] += 1
                continue
            towards = interior[0]["point"] if interior else b["point"]
            probe = [(a["point"][0] + towards[0]) / 2.0, (a["point"][1] + towards[1]) / 2.0, 0.0]
            face = None
            for fkey in faces_of(a) & faces_of(b):
                polygon = [mesh.vertex_coordinates(v) for v in mesh.face_vertices(fkey)]
                if is_point_in_polygon_xy(probe, polygon):
                    face = fkey
                    break
            if face is None:
                notes["outside"] += 1
                continue
            runs.append((i, j, face))
        return runs, notes

    @staticmethod
    def _materialise(
        work: "QuadMesh",
        stations: list[dict[str, Any]],
        runs: list[tuple[int, int, int]],
        notes: dict[str, Any],
    ) -> dict[int, int]:
        """A vertex key for every station a run uses, splitting edges on ``work``."""
        keys = {}
        splits = {}
        used = sorted(set(k for i, j, _f in runs for k in (i, j)))
        for k in used:
            station = stations[k]
            if station["kind"] == "vertex":
                keys[k] = station["vkey"]
                continue
            u, v = station["edge"]
            t = station["t"]
            # the same crossing twice -- a line that crosses itself on an edge --
            # reuses the vertex rather than splitting a zero-length piece off
            again = [vkey for ts, vkey in splits.get((u, v), []) if abs(ts - t) <= 1e-9]
            if again:
                keys[k] = again[0]
                continue
            chain = [(0.0, u)] + sorted(splits.get((u, v), [])) + [(1.0, v)]
            for (ta, a), (tb, b) in pairwise(chain):
                if ta <= t <= tb:
                    break
            local = (t - ta) / (tb - ta) if tb > ta else 0.5
            pa, pb = work.vertex_coordinates(a), work.vertex_coordinates(b)
            vkey = work.split_edge((a, b), t=local, allow_boundary=True)
            work.vertex_attributes(vkey, "xyz", [pa[i] + local * (pb[i] - pa[i]) for i in range(3)])
            splits.setdefault((u, v), []).append((t, vkey))
            notes["split_edges"] += 1
            notes["new_vertices"] += 1
            keys[k] = vkey
        return keys

    @staticmethod
    def _split_face(
        work: "QuadMesh",
        a: int,
        b: int,
        interior: list[list[float]],
        notes: dict[str, Any],
    ) -> int | None:
        """Split the face holding ``a`` and ``b`` along ``a``, ``interior``, ``b``."""
        towards = interior[0] if interior else work.vertex_coordinates(b)
        pa = work.vertex_coordinates(a)
        probe = [(pa[0] + towards[0]) / 2.0, (pa[1] + towards[1]) / 2.0, 0.0]
        face = None
        for fkey in set(work.vertex_faces(a)) & set(work.vertex_faces(b)):
            if fkey is None:
                continue
            polygon = [work.vertex_coordinates(v) for v in work.face_vertices(fkey)]
            if is_point_in_polygon_xy(probe, polygon):
                face = fkey
                break
        if face is None:
            return None

        corners = work.face_vertices(face)
        new = []
        for point in interior:
            x, y, z = point[0], point[1], _height_in_face(work, face, point)
            new.append(work.add_vertex(x=x, y=y, z=z))
        start = corners.index(a)
        loop = corners[start:] + corners[:start]
        end = loop.index(b)
        first = loop[: end + 1] + new[::-1]
        second = loop[end:] + [a] + new
        work.delete_face(face)
        work.add_face(first)
        work.add_face(second)
        notes["split_faces"] += 1
        notes["new_vertices"] += len(new)
        return face

    # ------------------------------------------------------------------
    # strip operations -- gated per strip
    # ------------------------------------------------------------------

    def _split_strips(self, work: "QuadMesh", to_split: dict[int, int]) -> dict[int, list[int]]:
        """Refine strips to save a boundary, opening each by the grammar's exact rule."""
        return split_strips(work, to_split, open_strip=True, project=self.project_to_wall)

    def _gate(self, work: "QuadMesh") -> tuple[bool, str]:
        """No NEW face that is not a quad, and still manifold if it was."""
        before = len(self.non_quad_faces())
        after = len(self.non_quad_faces(work))
        if after > before:
            return False, ("the result would have {} more face(s) that are not quads".format(after - before))
        if self.mesh.is_manifold() and not work.is_manifold():
            return False, "the result would not be manifold"
        return True, ""

    def add_line(self, edge: tuple[int, int]) -> tuple[bool, dict[str, Any]]:
        """Grow a strip along the polyedge through ``edge``, on a copy adopted only if it passes ``_gate``. ``(ok, notes)``.

        Only the new vertex pairs move; smoothing is a separate step (``relax``).

        Parameters
        ----------
        edge : tuple[int, int]
            Any edge of the polyedge to add a strip along.
        """
        if not self._is_edge(edge):
            return self._refuse("{} is not an edge of the mesh".format(tuple(edge)))
        pkey, polyedge = self.polyedge_through(edge)
        if polyedge is None:
            return self._refuse("edge {} belongs to no polyedge -- nothing to add along".format(tuple(edge)))

        manifold = self._manifold_refusal()
        if manifold:
            return self._refuse(manifold)

        closed = polyedge[0] == polyedge[-1]
        # A strip has to run the full width of the mesh -- anything less leaves
        # a face with five sides.
        if not is_polyedge_valid_for_strip_addition(self.mesh, polyedge):
            return self._refuse(
                "the line through edge {} has {} vertices, is not closed, and "
                "does not end on the boundary at both ends. A strip has to run "
                "the full width of the mesh -- wall to wall, or all the way "
                "round.".format(tuple(edge), len(polyedge))
            )

        blocking = [fkey for vkey in set(polyedge) for fkey in self.mesh.vertex_faces(vkey) if fkey is not None and len(self.mesh.face_vertices(fkey)) != 4]
        if blocking:
            return self._refuse(
                "the line through edge {} touches {} face(s) that are not quads. "
                "A strip is only defined through quads, so a line that runs "
                "past a pole or a polygon cannot gain one.".format(tuple(edge), len(set(blocking)))
            )

        work = self.mesh.copy()
        # A PRECONDITION, not housekeeping: add_strip writes into
        # attributes['strips'] and fails without it.
        work.collect_strips()
        _pkey, work_polyedge = self.polyedge_through(edge, mesh=work)
        if work_polyedge is None:
            return self._refuse("lost track of the line through edge {} on the working copy".format(tuple(edge)))

        try:
            skey, old_to_new = _grammar_add_strip(work, list(work_polyedge), open_strip=True, project=self.project_to_wall)
        except Exception as exc:
            return self._refuse("add_strip failed: {}: {}".format(type(exc).__name__, exc))

        ok, reason = self._gate(work)
        if not ok:
            return self._refuse(reason)

        # old_to_new maps old vkey -> the pair it was split into
        new_vertices = set()
        for pair in old_to_new.values():
            new_vertices.update(pair)

        self.last_addition = {
            "strip": skey,
            "polyedge": pkey,
            "polyedge_vertices": len(work_polyedge),
            "closed": closed,
            "faces_before": self.mesh.number_of_faces(),
            "faces_after": work.number_of_faces(),
            "new_vertices": sorted(new_vertices),
        }
        self._adopt(work)
        return self._accept(**self.last_addition)

    @staticmethod
    def _as_dense_plan(info: dict[str, Any]) -> dict[str, Any]:
        """The base's deletion plan in this class's published shape (``strip``, counts)."""
        plan = dict(info)
        plan["strip"] = info["skey"]
        plan["boundaries_lost"] = len(info["boundaries_lost"])
        plan["boundaries_lost_vertices"] = list(info["boundaries_lost"])
        return plan

    def _manifold_refusal(self) -> str:
        """``''``, or why no strip operation can run on this mesh (a non-manifold vertex)."""
        bowties = self.non_manifold_vertices()
        if not bowties:
            return ""
        return "{} vertex/vertices join faces that only touch at a corner, and the strip grammar cannot walk a boundary through one. Remove or merge a face there first.".format(
            len(bowties)
        )

    def _edge_of_strip(self, skey: int) -> tuple[int, int] | None:
        """One edge of strip ``skey``, resolved on a copy, or ``None`` if there is no such strip."""
        work = self.mesh.copy()
        work.collect_strips()
        if skey not in work.attributes["strips"]:
            return None
        return tuple(work.strip_edges(skey)[0])

    def _resolve_edge(self, edge: tuple[int, int] | None, skey: int | None) -> tuple[tuple[int, int] | None, str]:
        """``(edge, '')`` for a strip given by ``edge`` or ``skey``, or ``(None, reason)``."""
        if edge is not None:
            return tuple(edge), ""
        if skey is None:
            return None, "no edge or strip given"
        edge = self._edge_of_strip(skey)
        if edge is None:
            return None, "there is no strip {!r} on this mesh".format(skey)
        return edge, ""

    def _strip_refusal(self, edge: tuple[int, int]) -> str:
        """Why the strip through ``edge`` cannot be touched, or ``''``."""
        if not self._is_edge(edge):
            return "{} is not an edge of the mesh".format(tuple(edge))
        manifold = self._manifold_refusal()
        if manifold:
            return manifold
        fkey = self._non_quad_on_strip(edge)
        if fkey is None:
            return ""
        return (
            "the strip through edge {} crosses or ends at a face with {} "
            "sides. A strip is only defined through quads, so a strip that "
            "reaches a pole or a polygon cannot be removed as one.".format(tuple(edge), len(self.mesh.face_vertices(fkey)))
        )

    def plan_strip_deletion(
        self,
        edge: tuple[int, int] | None = None,
        preserve_boundaries: bool = False,
        skey: int | None = None,
    ) -> dict[str, Any]:
        """**What deleting the strip through** ``edge`` (or ``skey``) **would cost.** Mutates nothing.

        The deletion is PERFORMED on a copy and what came out is reported,
        because no cheap test predicts every failure.

        Returns
        -------
        dict
            ``ok`` -- whether ``remove_line`` would succeed with these
            arguments -- and ``reason`` when it would not. Plus ``strip``,
            ``faces``, ``collateral``, ``boundaries_lost`` (a COUNT),
            ``boundaries_lost_vertices`` (which ones), ``to_split``,
            ``faces_before`` / ``faces_after`` and
            ``vertices_before`` / ``vertices_after``. The counts after are
            ``None`` when the deletion would not go through.
        """
        edge, reason = self._resolve_edge(edge, skey)
        if not reason:
            reason = self._strip_refusal(edge)
        if reason:
            self.last_reason = reason
            return self._as_dense_plan(self._empty_plan(reason))

        info = super(DenseMeshEditor, self).plan_strip_deletion(edge=edge, preserve_boundaries=preserve_boundaries)
        return self._as_dense_plan(info)

    def remove_strip(
        self,
        edge: tuple[int, int] | None = None,
        preserve_boundaries: bool = False,
        skey: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Same as ``remove_line``, with the strip given by ``edge`` or ``skey``. ``(ok, notes)``."""
        edge, reason = self._resolve_edge(edge, skey)
        if reason:
            return self._refuse(reason)
        return self.remove_line(edge, preserve_boundaries=preserve_boundaries)

    def remove_line(self, edge: tuple[int, int], preserve_boundaries: bool = False) -> tuple[bool, dict[str, Any]]:
        """**Delete the strip through** ``edge``. ``(ok, notes)``.

        Performs the same trial as ``plan_strip_deletion`` and adopts the
        result, so a refusal costs nothing.

        Parameters
        ----------
        edge : tuple[int, int]
            Any edge of the strip to delete.
        preserve_boundaries : bool, optional
            Pre-split the strips that would otherwise leave a boundary with too
            few splits.
        """
        reason = self._strip_refusal(edge)
        if reason:
            return self._refuse(reason)

        faces_before = self.mesh.number_of_faces()
        vertices_before = self.mesh.number_of_vertices()
        ok, notes = super(DenseMeshEditor, self).remove_strip(edge=edge, preserve_boundaries=preserve_boundaries)
        if not ok:
            return ok, notes

        plan = self._as_dense_plan(notes)
        self.last_deletion = {
            "strip": plan["strip"],
            "faces": plan["faces"],
            "collateral": plan["collateral"],
            "boundaries_lost": plan["boundaries_lost"],
            "pre_split": plan["to_split"] if preserve_boundaries else {},
            "faces_before": faces_before,
            "faces_after": self.mesh.number_of_faces(),
            "vertices_before": vertices_before,
            "vertices_after": self.mesh.number_of_vertices(),
        }
        self._forget_stale_poles(self.mesh)
        return self._accept(**self.last_deletion)

    # ------------------------------------------------------------------
    # relaxation
    # ------------------------------------------------------------------

    def relax(self, iterations: int | None = None, mesh: "QuadMesh | None" = None) -> "QuadMesh":
        """Smooth the interior, holding every boundary ring. Returns the mesh."""
        mesh = self.mesh if mesh is None else mesh
        iterations = self.relax_iterations if iterations is None else iterations
        fixed = list(self.boundary_vertices(mesh))
        mesh_smooth_centroid(mesh, kmax=iterations, fixed=fixed)
        if mesh is self.mesh:
            self.edited = True
        return mesh

    # ------------------------------------------------------------------
    # undo
    # ------------------------------------------------------------------

    def _state(self) -> dict[str, Any]:
        state = super(DenseMeshEditor, self)._state()
        state["last_addition"] = dict(self.last_addition)
        return state

    def _restore(self, state: dict[str, Any]) -> None:
        super(DenseMeshEditor, self)._restore(state)
        self.last_addition = state.get("last_addition", {})

    def reset(self) -> tuple[bool, dict[str, Any]]:
        """Put the mesh back as it was at construction, or at the last snapshot."""
        self.last_addition = {}
        return super(DenseMeshEditor, self).reset()


# ----------------------------------------------------------------------
# plan geometry for draw_edges
# ----------------------------------------------------------------------


def _d2(a: list[float], b: list[float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _same_place(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["kind"] == "vertex" and b["kind"] == "vertex":
        return a["vkey"] == b["vkey"]
    return _d2(a["point"], b["point"]) <= 1e-18


def _segment_intersection_xy(p: list[float], q: list[float], a: list[float], b: list[float]) -> tuple[float, float] | None:
    """``(s, t)`` where segment p-q crosses segment a-b in plan, or ``None``.

    Parallel segments do not cross here, even when they overlap: a line drawn
    along an edge is picked up by its own points snapping to that edge instead.
    """
    rx, ry = q[0] - p[0], q[1] - p[1]
    sx, sy = b[0] - a[0], b[1] - a[1]
    den = rx * sy - ry * sx
    scale = (rx * rx + ry * ry) ** 0.5 * (sx * sx + sy * sy) ** 0.5
    if scale == 0.0 or abs(den) <= 1e-12 * scale:
        return None
    wx, wy = a[0] - p[0], a[1] - p[1]
    s = (wx * sy - wy * sx) / den
    t = (wx * ry - wy * rx) / den
    eps = 1e-9
    if -eps <= s <= 1 + eps and -eps <= t <= 1 + eps:
        return min(max(s, 0.0), 1.0), min(max(t, 0.0), 1.0)
    return None


def _closest_on_segment_xy(point: list[float], a: list[float], b: list[float]) -> tuple[float, list[float]]:
    """``(t, closest point)`` on segment a-b, in plan."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return 0.0, list(a)
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2
    t = min(1.0, max(0.0, t))
    return t, [a[0] + t * dx, a[1] + t * dy, a[2] + t * (b[2] - a[2])]


def _height_in_face(mesh: "QuadMesh", fkey: int, point: list[float]) -> float:
    """The height of the face at ``point``, from the fan triangle it lies in."""
    corners = [mesh.vertex_coordinates(v) for v in mesh.face_vertices(fkey)]
    n = len(corners)
    centre = [sum(c[i] for c in corners) / n for i in range(3)]
    for a, b in pairwise(corners + corners[:1]):
        weights = _barycentric_xy(point, centre, a, b)
        if weights is not None and min(weights) >= -1e-9:
            return weights[0] * centre[2] + weights[1] * a[2] + weights[2] * b[2]
    return centre[2]


def _barycentric_xy(p: list[float], a: list[float], b: list[float], c: list[float]) -> tuple[float, float, float] | None:
    den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(den) < 1e-18:
        return None
    wa = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / den
    wb = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / den
    return wa, wb, 1.0 - wa - wb
