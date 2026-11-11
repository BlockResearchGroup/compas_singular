"""Hand-editing a coarse quad layout, with no Rhino in it.

``CoarseEditor`` moves corners and poles, divides strips (by a drawn curve or
a picked strip), adds and removes strips, and commits back into the layout.
Design notes: ``design_notes/editing.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas.geometry import closest_point_on_segment_xy
from compas.geometry import distance_point_point
from compas.geometry import intersection_segment_segment_xy
from compas.geometry import is_point_in_polygon_xy
from compas.itertools import pairwise
from compas_singular.datastructures.mesh_quad.grammar.add_strip import add_strip as _grammar_add_strip
from compas_singular.datastructures.mesh_quad.grammar.add_strip import is_polyedge_valid_for_strip_addition
from compas_singular.datastructures.mesh_quad_coarse.coarse_curves import coarse_edges_to_curves
from compas_singular.editing.curves import warp_chorded_edges
from compas_singular.editing.editor import MeshEditor
from compas_singular.editing.rebuild import coarse_from_skeleton
from compas_singular.editing.rebuild import snap_to_loops
from compas_singular.editing.repair import densifiable

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarsePseudoQuadMesh


__all__ = ["CoarseEditor", "PRECISION"]


#: Decimals the curve map is keyed at. The resolution ``geometric_key`` and
#: ``from_polylines``' endpoint matching already use, so a key made here still
#: matches after a round trip through either.
PRECISION = 3


class CoarseEditor(MeshEditor):
    """A coarse quad layout being hand-edited. No Rhino, no prompts, no prints.

    Parameters
    ----------
    coarse : CoarsePseudoQuadMesh
        The layout to edit. Edited on a COPY; ``commit`` writes back into
        this object, keeping its identity.
    field : CrossField, optional
        Carried so a caller can densify with it afterwards, and used for its
        ``background`` when ``loops`` or ``snap_tol`` are not given. Never
        solved, never consulted by an edit.
    loops : list, optional
        The domain walls, outer first, as point lists. A boundary corner that
        moves is projected onto the nearest one. Resolved from ``field`` when
        omitted, and from the layout's own boundary when there is no field
        either -- in which case the projection is a near no-op and no curvature
        is available.
    polylines : list, optional
        The traced separatrices as plain point lists. Used to give the two
        halves of a curved edge their shape when a cut crosses it, and published
        with the drawn curves through ``all_polylines``.
    poles : list, optional
        Preferred pole positions for a rebuild. Defaults to the layout's own.
    snap_tol : float, optional
        How far a corner ON THE LAYOUT'S BOUNDARY may be projected onto a domain
        wall at commit. Interior corners are never moved, whatever their
        distance.

    Attributes
    ----------
    curves : dict
        ``(rounded pa, rounded pb) -> [[x, y, z], ...]`` -- the SHAPE of a coarse
        edge that has one: a curve the user drew, or a half of a curved edge that
        a cut crossed. Keyed geometrically and never by vertex key, because a
        rebuilding commit renumbers them.
    last_reason : str
    last_cut : dict
    last_deletion : dict

    Examples
    --------
    >>> editor = CoarseEditor(coarse, loops=[outer] + holes)
    >>> ok, notes = editor.divide([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], extend=True)
    >>> ok, notes = editor.move_pole(pkey, editor.pole_targets(pkey)[0])
    >>> layout, notes = editor.commit()
    """

    #: A coarse layout is planar by construction; a moved corner is flattened.
    PLANAR = True

    def __init__(
        self,
        coarse: "CoarsePseudoQuadMesh",
        field: Any = None,
        loops: list[list[list[float]]] | None = None,
        polylines: Sequence[list[list[float]]] = (),
        poles: list[list[float]] | None = None,
        snap_tol: float | None = None,
    ) -> None:
        if coarse is None:
            raise ValueError("no coarse layout to edit -- pass the mesh to edit")
        loops = self._resolve_loops(coarse, field, loops)
        super(CoarseEditor, self).__init__(coarse, walls=loops, work_on_copy=True)

        #: The walls as raw point lists. ``walls`` holds the same loops as
        #: ``BoundaryLoop`` objects for projection; ``coarse_from_skeleton`` and
        #: ``snap_to_loops`` want the points.
        self.loops = loops
        self.field = field
        self.polylines = [[list(point) for point in polyline] for polyline in (polylines or [])]
        self.poles = None if poles is None else [list(p) for p in poles]
        self._snapshot_poles = None if poles is None else [list(p) for p in poles]
        self.snap_tol = self._default_snap_tol(field) if snap_tol is None else snap_tol
        #: The length the warp's two tolerances are expressed in -- the spacing
        #: the separatrices were traced at, where a field says what that was.
        #: Without one there is nothing that knows it, so a quarter of the mean
        #: coarse edge stands in: on the measured documents the background is
        #: 0.5 against coarse edges of 2 to 4, which is the same ratio.
        self.warp_scale = self._default_warp_scale(field)
        self.curves = {}
        self._snapshot_curves = {}
        #: Set by ``divide`` alone. The strip grammar maintains its own strip
        #: data and preserves strip labels (thesis 5.3.3), so an added or deleted
        #: strip does not need the weld-and-repair that a face split does.
        self._topology_dirty = False
        #: Whether ``commit`` has ever adopted a layout from this editor. The
        #: replacement for the decomposition's ``_edited``, which a commit no
        #: longer sets -- and truer than inferring it from the last cut or
        #: deletion, which a moves-only commit leaves empty.
        self.committed = False
        self.last_cut = {}
        #: Set by ``move_pole``. Connectivity is unchanged, but the strips
        #: through the relabelled pseudo-quads are not, so a commit must not
        #: carry the old strip data across.
        self._strips_stale = False
        self.last_pole = {}

    @staticmethod
    def _resolve_loops(
        coarse: "CoarsePseudoQuadMesh",
        field: Any,
        loops: list[list[list[float]]] | None,
    ) -> list[list[list[float]]]:
        """The domain walls, from the caller, the field, or the layout itself."""
        if loops:
            return [[list(point) for point in loop] for loop in loops]
        background = getattr(field, "background", None)
        if background is not None:
            return [[list(p) for p in background.outer]] + [[list(p) for p in inner] for inner in background.inners]
        # No walls given and no field to read them off. The layout's own boundary
        # is the honest fallback: projecting a boundary corner onto it is very
        # nearly a no-op, which is correct -- without the real walls there is
        # nothing better to hold a corner on, and pretending otherwise would move
        # corners onto a shape the domain does not have.
        return [[coarse.vertex_coordinates(vkey) for vkey in ring] for ring in coarse.boundaries()]

    def _default_warp_scale(self, field: Any) -> float:
        background = getattr(field, "background", None)
        target = getattr(background, "target_length", None)
        return target if target else 0.25 * self.mean_edge()

    def _default_snap_tol(self, field: Any) -> float:
        background = getattr(field, "background", None)
        target = getattr(background, "target_length", None)
        if target:
            return target * 0.5
        return 0.5 * self.mean_edge()

    def corner_tol(self) -> float:
        """How close to a corner a point is taken as that corner.

        Keeps edge splits clear of slivers and of the 3-decimal weld at commit.
        """
        return 0.05 * self.mean_edge()

    def on_tol(self) -> float:
        """How far off an edge a point may be and still count as on it.

        Tight, because a point that reaches here was picked with the cursor
        constrained onto a layout edge -- this absorbs float noise, not aim.
        """
        return 1e-3 * self.mean_edge()

    # ------------------------------------------------------------------
    # queries a front end needs
    # ------------------------------------------------------------------

    def locate(self, point: list[float]) -> tuple[str, int] | tuple[str, tuple[int, int], list[float]] | None:
        """Where a point sits on the layout: ``('vertex', vkey)``, ``('edge', (u, v), xyz)`` or ``None``.

        Corners win over edges within ``corner_tol``.
        """
        mesh = self.mesh
        tol = self.corner_tol()
        best = None
        for vkey in mesh.vertices():
            d = distance_point_point(point, mesh.vertex_coordinates(vkey))
            if d <= tol and (best is None or d < best[0]):
                best = (d, vkey)
        if best is not None:
            return ("vertex", best[1])

        tol = self.on_tol()
        best = None
        for u, v in mesh.edges():
            segment = (mesh.vertex_coordinates(u), mesh.vertex_coordinates(v))
            q = closest_point_on_segment_xy(point, segment)
            d = distance_point_point(point, q)
            if d <= tol and (best is None or d < best[0]):
                best = (d, (u, v), [q[0], q[1], 0.0])
        if best is not None:
            return ("edge", best[1], best[2])
        return None

    def project(self, xyz: list[float], boundary: bool) -> list[float]:
        """Project a boundary corner onto the nearest wall; an interior one stays put.

        Eligibility is topological, as in ``rebuild.snap_to_loops``.
        """
        if not boundary:
            return list(xyz)
        return self.project_to_wall(xyz)

    def edge_shape(self, u: int, v: int) -> list[list[float]] | None:
        """The drawn SHAPE this edge carries, or ``None`` if it is a chord.

        For display: an edge that carries an inserted arc should be drawn as
        that arc, or a cut that worked looks like one that snapped straight.
        """
        return self._own_curve(self.mesh.vertex_coordinates(u), self.mesh.vertex_coordinates(v))

    # ------------------------------------------------------------------
    # the curve map
    # ------------------------------------------------------------------

    @staticmethod
    def _curve_key(pa: list[float], pb: list[float]) -> tuple[float, float, float, float]:
        """Geometric key for the curve map. 3 decimals, as everything else."""
        return (round(pa[0], PRECISION), round(pa[1], PRECISION), round(pb[0], PRECISION), round(pb[1], PRECISION))

    def _own_curve(self, pa: list[float], pb: list[float]) -> list[list[float]] | None:
        """The shape registered for this edge, either way round, or ``None``."""
        curve = self.curves.get(self._curve_key(pa, pb))
        if curve is not None:
            return [list(p) for p in curve]
        curve = self.curves.get(self._curve_key(pb, pa))
        if curve is not None:
            return [list(p) for p in reversed(curve)]
        return None

    def _edge_curve(self, pa: list[float], pb: list[float]) -> list[list[float]] | None:
        """The shape this coarse edge follows: a drawn curve, or its traced separatrix."""
        curve = self._own_curve(pa, pb)
        if curve is not None:
            return curve

        tol = 1e-3
        for polyline in self.polylines or []:
            points = [list(p)[:3] for p in polyline]
            # Two points is a chord and carries no shape worth splitting.
            if len(points) < 3:
                continue
            if distance_point_point(points[0], pa) < tol and distance_point_point(points[-1], pb) < tol:
                return points
            if distance_point_point(points[0], pb) < tol and distance_point_point(points[-1], pa) < tol:
                return list(reversed(points))
        return None

    @staticmethod
    def _clean(points: list[list[float]]) -> list[list[float]]:
        """Drop consecutive duplicates -- ``Polyline.point_at`` divides by them."""
        out = [list(points[0])]
        for point in points[1:]:
            if distance_point_point(out[-1], point) > 1e-9:
                out.append(list(point))
        return out

    def _split_curve(self, curve: list[list[float]], point: list[float]) -> tuple[list[float] | None, list[list[float]] | None, list[list[float]] | None]:
        """``(point on the curve, head, tail)`` for the curve cut at ``point``.

        ``(None, None, None)`` when the split leaves a stub.
        """
        best = None
        for i in range(len(curve) - 1):
            q = closest_point_on_segment_xy(point, (curve[i], curve[i + 1]))
            d = distance_point_point(point, q)
            if best is None or d < best[0]:
                best = (d, i, [q[0], q[1], 0.0])
        if best is None:
            return None, None, None

        _d, i, q = best
        head = self._clean([list(p) for p in curve[: i + 1]] + [q])
        tail = self._clean([q] + [list(p) for p in curve[i + 1 :]])
        if len(head) < 2 or len(tail) < 2:
            return None, None, None
        return q, head, tail

    # ------------------------------------------------------------------
    # planning a cut
    # ------------------------------------------------------------------

    def _node_point(self, node: Any) -> list[float]:
        """The xyz a plan node sits at."""
        if node[0] == "vertex":
            return self.mesh.vertex_coordinates(node[1])
        return node[2]

    @staticmethod
    def _direction(points: list[list[float]], index: int, start: list[float]) -> list[float] | None:
        """Curve direction leaving ``start`` on segment ``index``."""
        for i in range(index, len(points) - 1):
            dx = points[i + 1][0] - start[0]
            dy = points[i + 1][1] - start[1]
            if (dx * dx + dy * dy) > 1e-18:
                return [dx, dy, 0.0]
            start = points[i + 1]
        return None

    def _face_across(self, edge: tuple[int, int], direction: list[float]) -> int | None:
        """The face on the side of ``edge`` the curve is heading into, decided by centroid side.

        ``None`` when the curve runs along the edge or that side is open.
        """
        mesh = self.mesh
        u, v = edge
        pu = mesh.vertex_coordinates(u)
        pv = mesh.vertex_coordinates(v)
        ex, ey = pv[0] - pu[0], pv[1] - pu[1]
        side = ex * direction[1] - ey * direction[0]
        if abs(side) <= 1e-12:
            return None
        for fkey in (mesh.halfedge[u][v], mesh.halfedge[v][u]):
            if fkey is None:
                continue
            centroid = mesh.face_centroid(fkey)
            cs = ex * (centroid[1] - pu[1]) - ey * (centroid[0] - pu[0])
            if cs * side > 0.0:
                return fkey
        return None

    def _face_around(self, vkey: int, points: list[list[float]], index: int, start: list[float]) -> int | None:
        """The face at a corner the curve heads into -- containment of a probe."""
        direction = self._direction(points, index, start)
        if direction is None:
            return None
        length = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        step = 0.25 * self.mean_edge() / length
        probe = [start[0] + direction[0] * step, start[1] + direction[1] * step, 0.0]
        for fkey in self.mesh.vertex_faces(vkey):
            polygon = [self.mesh.vertex_coordinates(w) for w in self.mesh.face_vertices(fkey)]
            if is_point_in_polygon_xy(probe, polygon):
                return fkey
        return None

    def _exit_of(self, fkey: int, points: list[list[float]], index: int, start: list[float], end: Any) -> tuple[Any, int, list[list[float]] | None, str]:
        """Where the curve leaves patch ``fkey`` after entering at ``start``.

        Returns ``(node, index, subcurve, reason)``; ``node`` is ``None`` on a refusal.
        """
        mesh = self.mesh
        tol = self.on_tol()
        skip = max(self.corner_tol(), tol)
        halfedges = mesh.face_halfedges(fkey)

        cursor = list(start)
        for i in range(index, len(points) - 1):
            a = cursor if i == index else points[i]
            b = points[i + 1]
            best = None
            for u, v in halfedges:
                segment = (mesh.vertex_coordinates(u), mesh.vertex_coordinates(v))
                x = intersection_segment_segment_xy((a, b), segment, tol)
                if x is None:
                    continue
                # The way IN is an intersection too, and it is not the way out.
                if distance_point_point(x, start) <= skip:
                    continue
                t = distance_point_point(a, x)
                if best is None or t < best[0]:
                    best = (t, [x[0], x[1], 0.0], (u, v))
            if best is None:
                continue
            _t, x, edge = best
            sub = self._clean([list(start)] + [list(p) for p in points[index + 1 : i + 1]] + [x])
            return self._node_at(fkey, edge, x), i, sub, ""

        # No crossing left: the curve stops inside this patch. That is only
        # legal if it stopped ON this patch's own rim -- which the exact
        # arithmetic above can miss by a float when the last point IS the exit.
        point = self._node_point(end)
        corners = set(mesh.face_vertices(fkey))
        on_rim = (end[0] == "vertex" and end[1] in corners) or (end[0] == "edge" and (tuple(end[1]) in halfedges or tuple(reversed(end[1])) in halfedges))
        if on_rim and distance_point_point(point, start) > skip:
            sub = self._clean([list(start)] + [list(p) for p in points[index + 1 : -1]] + [list(point)])
            node = end if end[0] == "vertex" else self._node_at(fkey, end[1], point)
            return node, len(points) - 2, sub, ""

        return None, index, None, ("the line stops inside a patch. It has to reach a patch edge -- and to leave the layout all-quad, a boundary")

    def _pole_ahead(self, fkey: int, entry: Any) -> int | None:
        """The pole of pseudo-quad ``fkey`` if the cut entered it through the side opposite the pole."""
        face_pole = self.mesh.attributes.get("face_pole") or {}
        pole = face_pole.get(fkey)
        if pole is None or entry[0] != "edge" or pole in entry[1]:
            return None
        return pole

    def _pole_under(self, point: list[float]) -> tuple[str, int] | None:
        """``('vertex', pole)`` for a curve end drawn inside a pseudo-quad, else ``None``."""
        face_pole = self.mesh.attributes.get("face_pole") or {}
        for fkey, pole in face_pole.items():
            polygon = [self.mesh.vertex_coordinates(w) for w in self.mesh.face_vertices(fkey)]
            if is_point_in_polygon_xy(point, polygon):
                return ("vertex", pole)
        return None

    def _node_at(self, fkey: int, edge: tuple[int, int], point: list[float]) -> tuple[str, int] | tuple[str, tuple[int, int], list[float]]:
        """An exit node, collapsed onto a corner when it lands on one."""
        for vkey in self.mesh.face_vertices(fkey):
            if distance_point_point(point, self.mesh.vertex_coordinates(vkey)) <= self.corner_tol():
                return ("vertex", vkey)
        return ("edge", tuple(edge), point)

    def _plan_cut(self, points: list[list[float]]) -> tuple[dict[str, Any] | None, str]:
        """Walk the drawn curve across the layout without mutating it.

        Returns ``{'nodes': [...], 'faces': [(fkey, i, j, subcurve), ...]}``.
        """
        points = [[float(p[0]), float(p[1]), 0.0] for p in points]
        points = self._clean(points)
        if len(points) < 2:
            return None, "the drawn curve has no length"

        start = self.locate(points[0]) or self._pole_under(points[0])
        if start is None:
            return None, ("the line does not START on the layout -- pick its first point on a coarse edge")
        end = self.locate(points[-1]) or self._pole_under(points[-1])
        if end is None:
            return None, ("the line does not END on the layout -- pick its last point on a coarse edge")
        points[0] = list(self._node_point(start))
        points[-1] = list(self._node_point(end))
        if distance_point_point(points[0], points[-1]) <= self.corner_tol():
            return None, "the line starts and ends at the same corner"

        nodes = [start]
        faces = []
        index, entry = 0, 0
        while True:
            here = self._node_point(nodes[entry])
            if nodes[entry][0] == "vertex":
                fkey = self._face_around(nodes[entry][1], points, index, here)
            else:
                direction = self._direction(points, index, here)
                fkey = self._face_across(nodes[entry][1], direction) if direction is not None else None
            if fkey is None:
                return None, ("the line leaves the layout, or runs along a patch edge instead of across it")
            if any(item[0] == fkey for item in faces):
                return None, ("the line crosses the same patch twice. Draw it as two cuts")

            pole = self._pole_ahead(fkey, nodes[entry])
            node, index, sub, reason = self._exit_of(fkey, points, index, here, end)
            if pole is not None and node != ("vertex", pole):
                # Entered through the side opposite the pole: the strip ends at the
                # pole, so the cut does too, wherever the curve leaves the patch.
                pole_point = self.mesh.vertex_coordinates(pole)
                nodes.append(("vertex", pole))
                faces.append((fkey, entry, len(nodes) - 1, [list(here), list(pole_point)]))
                entry = len(nodes) - 1
                if (
                    node is None
                    or distance_point_point(self._node_point(node), points[-1]) <= self.on_tol()
                    or distance_point_point(pole_point, points[-1]) <= self.corner_tol()
                    or self._direction(points, index, pole_point) is None
                ):
                    break
                if len(faces) > self.mesh.number_of_faces():
                    return None, "the line does not terminate on the layout"
                # the curve carries on past the pole: carry on from the pole
                continue
            if node is None:
                return None, reason
            nodes.append(node)
            faces.append((fkey, entry, len(nodes) - 1, sub))
            entry = len(nodes) - 1

            if distance_point_point(self._node_point(node), points[-1]) <= self.on_tol():
                break
            if len(faces) > self.mesh.number_of_faces():
                return None, "the line does not terminate on the layout"

        return {"nodes": nodes, "faces": faces}, ""

    def _open_ends(self, plan: dict[str, Any]) -> list[tuple[int, int]]:
        """The ends of a planned cut that stop on an interior edge, as ``(node index, from patch)``."""
        out = []
        if not plan["faces"]:
            return out
        for index, fkey in ((plan["faces"][0][1], plan["faces"][0][0]), (plan["faces"][-1][2], plan["faces"][-1][0])):
            node = plan["nodes"][index]
            if node[0] != "edge":
                continue
            if not self.mesh.is_edge_on_boundary(node[1]):
                out.append((index, fkey))
        return out

    def _extend_plan(self, plan: dict[str, Any], index: int, fkey: int) -> tuple[bool, str]:
        """Carry a cut that stops on an interior edge on to the boundary, along its strip.

        The continuation is straight; the drawn part keeps its shape.
        """
        mesh = self.mesh
        while True:
            node = plan["nodes"][index]
            edge, point = node[1], node[2]
            nxt = None
            for candidate in (mesh.halfedge[edge[0]][edge[1]], mesh.halfedge[edge[1]][edge[0]]):
                if candidate is not None and candidate != fkey:
                    nxt = candidate
            if nxt is None:
                return True, ""
            if any(item[0] == nxt for item in plan["faces"]):
                return False, ("extending the cut runs back into a patch it has already crossed -- draw it as two cuts")
            if not mesh.is_strip_face(nxt):
                return False, ("the cut would have to continue through a patch that is not a quad, and there is no opposite side there to continue to")
            pole = self._pole_ahead(nxt, node)
            if pole is not None:
                # into a pseudo-quad through the side opposite its pole: end there
                plan["nodes"].append(("vertex", pole))
                plan["faces"].append((nxt, index, len(plan["nodes"]) - 1, [list(point), list(mesh.vertex_coordinates(pole))]))
                return True, ""

            # The halfedge as THIS patch sees it, or ``face_opposite_edge``
            # answers for the patch on the other side.
            uv = None
            for u, v in mesh.face_halfedges(nxt):
                if set((u, v)) == set(edge):
                    uv = (u, v)
            if uv is None:
                return False, "the cut lost track of the patch it was crossing"
            u, v = uv
            w, x = mesh.face_opposite_edge(u, v)

            pu, pv = mesh.vertex_coordinates(u), mesh.vertex_coordinates(v)
            span = distance_point_point(pu, pv)
            if span <= 0.0:
                return False, "the cut would continue across a zero-length edge"
            t = distance_point_point(pu, closest_point_on_segment_xy(point, (pu, pv))) / span
            # (u, v, w, x) go round the quad, so u pairs with x and v with w:
            # the point t of the way from u to v is t of the way from x to w.
            pw, px = mesh.vertex_coordinates(w), mesh.vertex_coordinates(x)
            q = [px[0] + t * (pw[0] - px[0]), px[1] + t * (pw[1] - px[1]), 0.0]

            plan["nodes"].append(self._node_at(nxt, (w, x), q))
            plan["faces"].append((nxt, index, len(plan["nodes"]) - 1, [list(point), list(q)]))
            index = len(plan["nodes"]) - 1
            fkey = nxt
            if plan["nodes"][index][0] != "edge":
                return True, ""
            if mesh.is_edge_on_boundary(plan["nodes"][index][1]):
                return True, ""

    # ------------------------------------------------------------------
    # applying a cut
    # ------------------------------------------------------------------

    def _current_edge(self, work: "CoarsePseudoQuadMesh", u: int, v: int, point: list[float]) -> tuple[int, int] | None:
        """The piece of edge ``(u, v)`` that ``point`` is on, after any splits."""
        if work.has_edge((u, v)):
            return (u, v)
        pu = work.vertex_coordinates(u)
        pv = work.vertex_coordinates(v)
        tol = self.on_tol()
        for a, b in work.edges():
            pa = work.vertex_coordinates(a)
            pb = work.vertex_coordinates(b)
            if distance_point_point(pa, closest_point_on_segment_xy(pa, (pu, pv))) > tol or distance_point_point(pb, closest_point_on_segment_xy(pb, (pu, pv))) > tol:
                continue
            if distance_point_point(point, closest_point_on_segment_xy(point, (pa, pb))) <= tol:
                return (a, b)
        return None

    def _split_edge_at(
        self,
        work: "CoarsePseudoQuadMesh",
        edge: tuple[int, int],
        point: list[float],
        pending: dict[tuple[float, float, float, float], list[list[float]]],
    ) -> int | None:
        """Split one coarse edge, placing the new corner on the edge's curve or wall.

        Both halves of a curved edge inherit their piece of the curve.
        """
        found = self._current_edge(work, edge[0], edge[1], point)
        if found is None:
            return None
        u, v = found
        pu = work.vertex_coordinates(u)
        pv = work.vertex_coordinates(v)

        curve = self._edge_curve(pu, pv)
        exact, head, tail = self._split_curve(curve, point) if curve is not None else (None, None, None)
        if exact is None:
            exact = self.project(list(point), work.is_edge_on_boundary((u, v)))
            head = tail = None

        span = distance_point_point(pu, pv)
        if span <= 0.0:
            return None
        q = closest_point_on_segment_xy(point, (pu, pv))
        t = distance_point_point(pu, q) / span
        t = min(max(t, 1e-6), 1.0 - 1e-6)

        w = work.split_edge((u, v), t, allow_boundary=True)
        if w is None or w in (u, v):
            return None
        work.vertex_attributes(w, "xyz", [exact[0], exact[1], 0.0])

        if head is not None and tail is not None:
            pending[self._curve_key(pu, exact)] = head
            pending[self._curve_key(exact, pv)] = tail
        return w

    def _apply_cut(self, work: "CoarsePseudoQuadMesh", plan: dict[str, Any]) -> tuple[bool, str, dict[tuple[float, float, float, float], list[list[float]]]]:
        """Perform a planned cut on ``work``. ``(ok, reason, curves)``."""
        pending = {}
        vertex_of = {}
        for index, node in enumerate(plan["nodes"]):
            if node[0] == "vertex":
                vertex_of[index] = node[1]
                continue
            w = self._split_edge_at(work, node[1], node[2], pending)
            if w is None:
                return False, "a coarse edge could not be split there", {}
            vertex_of[index] = w

        face_pole = work.attributes.get("face_pole")
        for fkey, i, j, _sub in plan["faces"]:
            a, b = vertex_of[i], vertex_of[j]
            if a == b:
                return False, "the line enters and leaves a patch at one corner", {}
            try:
                halves = work.split_face(fkey, a, b)
            except ValueError:
                # ``split_face`` refuses neighbouring corners, which is exactly
                # the cut that would slice a triangle off the patch.
                return False, ("the line cuts a corner off a patch instead of crossing it: it enters and leaves through sides that meet"), {}
            pole = face_pole.pop(fkey, None) if face_pole is not None else None
            if pole is not None:
                # the halves still touching the pole as triangles stay pseudo-quads
                for half in halves:
                    if pole in work.face_vertices(half) and len(work.face_vertices(half)) == 3:
                        face_pole[half] = pole

        ok, reason = self._check_quads(work)
        if not ok:
            return False, reason, {}

        for fkey, i, j, sub in plan["faces"]:
            pa = work.vertex_coordinates(vertex_of[i])
            pb = work.vertex_coordinates(vertex_of[j])
            curve = self._clean([list(pa)] + [list(p) for p in sub[1:-1]] + [list(pb)])
            # Two points IS the chord, which is what an unregistered edge
            # densifies as. Registering it would only add noise to the map.
            if len(curve) > 2:
                pending[self._curve_key(pa, pb)] = curve
        return True, "", pending

    def _check_quads(self, work: "CoarsePseudoQuadMesh") -> tuple[bool, str]:
        """Check that the edit left every patch four-sided (existing pseudo-quads excepted)."""
        face_pole = work.attributes.get("face_pole") or {}
        for fkey in work.faces():
            n = len(work.face_vertices(fkey))
            if n == 4 and fkey not in face_pole:
                continue
            if n == 3 and fkey in face_pole:
                continue
            x, y, _z = work.face_centroid(fkey)
            if fkey in face_pole:
                return False, ("the edit runs into the pseudo-quad near ({:.2f}, {:.2f}) and would break its pole".format(x, y))
            return False, (
                "it would leave the patch near ({:.2f}, {:.2f}) with {} sides. A "
                "cut has to enter and leave each patch through OPPOSITE sides, "
                "and every edge it crosses has to be crossed through -- so both "
                "its ends belong on the layout boundary. Continue the line to a "
                "wall, or start it on one".format(x, y, n)
            )
        return True, ""

    # ------------------------------------------------------------------
    # cutting the layout with a drawn curve
    # ------------------------------------------------------------------

    def divide(
        self,
        points: list[list[float]] | None = None,
        skey: int | None = None,
        t: float | None = None,
        extend: bool | Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Subdivide a strip, by a drawn curve (``points``) or a picked strip (``skey``, ``t``). ``(ok, notes)``.

        A legal cut crosses exactly the patches of one strip; anything else is refused.
        """
        if points is None and skey is None:
            return self._refuse("divide needs either a drawn curve or a strip to divide")
        if points is not None:
            return self._divide_by_curve(points, extend=extend)
        return self._divide_by_strip(skey, 0.5 if t is None else t)

    def _divide_by_strip(self, skey: int, t: float = 0.5) -> tuple[bool, dict[str, Any]]:
        """Split every rung of a strip at the same parameter. ``(ok, notes)``."""
        if not 0.0 < t < 1.0:
            return self._refuse("the split parameter has to be strictly between 0 and 1 -- at 0 or 1 the new corner lands on an existing one")

        work = self.mesh.copy()
        work.collect_strips()
        if skey not in work.attributes["strips"]:
            return self._refuse("there is no strip {!r} on this layout".format(skey))

        # Read BEFORE any split: ``split_edge`` leaves the stored strip data
        # describing a layout that no longer exists.
        rungs = [tuple(edge) for edge in work.strip_edges(skey)]
        closed = work.is_strip_closed(skey)
        strip_faces = set(work.strip_faces(skey))
        face_pole = work.attributes.get("face_pole")

        corners = []
        for u, v in rungs:
            if u == v:
                # a pole end is not split: the division line runs into the pole
                corners.append(u)
                continue
            w = work.split_edge((u, v), t, allow_boundary=True)
            if w is None or w in (u, v):
                return self._refuse("a rung of that strip could not be split")
            corners.append(w)

        pairs = list(zip(corners, corners[1:]))
        if closed and len(corners) > 2:
            pairs.append((corners[-1], corners[0]))

        for a, b in pairs:
            fkey = None
            for candidate in work.vertex_faces(a):
                if candidate in strip_faces and b in work.face_vertices(candidate):
                    fkey = candidate
                    break
            if fkey is None:
                return self._refuse("the split lost track of the patch between two rungs of that strip")
            pole = face_pole.pop(fkey, None) if face_pole is not None else None
            try:
                halves = work.split_face(fkey, a, b)
            except ValueError:
                return self._refuse("the two new corners of a patch are neighbours, so the split would slice a triangle off it rather than cross it")
            if pole is not None:
                # a split pseudo-quad leaves a pseudo-quad on each side still touching the pole
                for half in halves:
                    if pole in work.face_vertices(half) and len(work.face_vertices(half)) == 3:
                        face_pole[half] = pole

        ok, reason = self._check_quads(work)
        if not ok:
            return self._refuse(reason)

        self.mesh = work
        self._topology_dirty = True
        self.edited = True
        self.last_cut = {"skey": skey, "faces": len(pairs), "corners": len(corners), "faces_out": work.number_of_faces()}
        return self._accept(**self.last_cut)

    def strip_of_cut(self, points: list[list[float]]) -> int | None:
        """The strip a drawn curve would divide, or ``None``.

        Resolved from the first coarse edge the curve crosses. A front end can
        highlight the band before the rest of the curve is drawn.
        """
        start = self.locate(points[0]) if points else None
        if start is None or start[0] != "edge":
            return None
        return self.strip_through(start[1])

    def _divide_by_curve(self, points: list[list[float]], extend: bool | Any | None = None) -> tuple[bool, dict[str, Any]]:
        """Cut the layout with a drawn curve, on a copy adopted only if all-quad. ``(ok, notes)``.

        Inserts new corners, so it marks the topology dirty and ``commit`` rebuilds.

        Parameters
        ----------
        points : list[[x, y, z]]
            The curve, already sampled. The samples ARE the shape from here on:
            an arc handed over as two points is a chord.
        extend : bool or callable, optional
            What to do when the curve stops on an interior edge -- the case that
            leaves a five-sided patch behind. ``None`` refuses, ``True`` carries
            the cut on to the boundary (``_extend_plan``), and a callable is
            asked, with the number of ends that need it, and answers.
        """
        plan, reason = self._plan_cut(points)
        if plan is None:
            return self._refuse(reason)

        ends = self._open_ends(plan)
        if ends:
            wanted = extend(len(ends)) if callable(extend) else bool(extend)
            if not wanted:
                return self._refuse(
                    "the line stops on an interior patch edge, which would "
                    "leave the patch behind it with five sides. Either draw it "
                    "out to a boundary, or let the command carry it there"
                )
            for index, fkey in ends:
                ok, reason = self._extend_plan(plan, index, fkey)
                if not ok:
                    return self._refuse(reason)

        # **The structural check, before any geometry is touched.** A legal cut
        # crosses exactly the patches of one strip -- see ``divide``. When the
        # planned run is not a strip the cut is going to be refused anyway, but
        # by ``_check_quads`` counting sides on a mesh it already mutated, which
        # can only say "this patch came out with five sides". Saying it here says
        # what the user actually did wrong.
        skey = self._cut_strip(plan)
        if skey is None:
            return self._refuse(
                "the line does not follow a single line of patches. A cut has to "
                "enter and leave every patch through OPPOSITE sides -- through "
                "sides that meet it would slice a triangle off -- so it runs "
                "along one strip from wall to wall. Draw it across the layout "
                "rather than at an angle to it"
            )

        work = self.mesh.copy()
        ok, reason, curves = self._apply_cut(work, plan)
        if not ok:
            return self._refuse(reason)

        self.mesh = work
        self.curves.update(curves)
        self._topology_dirty = True
        self.edited = True
        self.last_cut = {
            "skey": skey,
            "faces": len(plan["faces"]),
            "corners": sum(1 for node in plan["nodes"] if node[0] == "edge"),
            "faces_out": work.number_of_faces(),
        }
        return self._accept(**self.last_cut)

    def _cut_strip(self, plan: dict[str, Any]) -> int | list[int] | None:
        """The strip a planned cut runs along, or ``None`` if it runs along none.

        A cut through a pole runs along one strip on each side of it; that returns their list.
        """
        poles = set((self.mesh.attributes.get("face_pole") or {}).values())
        runs, run = [], []
        for item in plan["faces"]:
            run.append(item)
            node = plan["nodes"][item[2]]
            if node[0] == "vertex" and node[1] in poles and item is not plan["faces"][-1]:
                runs.append(run)
                run = []
        runs.append(run)

        work = self.mesh.copy()
        work.collect_strips()
        skeys = []
        for run in runs:
            entry = None
            for item in run:
                for node in (plan["nodes"][item[1]], plan["nodes"][item[2]]):
                    if node[0] == "edge":
                        entry = node[1]
                        break
                if entry is not None:
                    break
            if entry is None:
                return None
            skey = work.edge_strip(tuple(entry))
            if skey is None:
                return None
            if set(item[0] for item in run) != set(work.strip_faces(skey)):
                return None
            skeys.append(skey)
        return skeys[0] if len(skeys) == 1 else skeys

    # ------------------------------------------------------------------
    # deleting a strip -- the base's walk, this layout's gate
    # ------------------------------------------------------------------

    def _gate(self, work: "CoarsePseudoQuadMesh") -> tuple[bool, str]:
        """Check the trial deletion is all-quad and manifold.

        Nothing about a strip predicts non-manifoldness, so the gate deletes on a copy and looks.
        """
        ok, reason = self._check_quads(work)
        if not ok:
            return False, reason
        if not work.is_manifold():
            return False, (
                "deleting that strip would leave the layout non-manifold -- "
                "welding its two sides together puts more than two patches on "
                "one edge. Nothing has been changed; try a different strip"
            )
        return True, ""

    @staticmethod
    def _as_coarse_plan(info: dict[str, Any]) -> dict[str, Any]:
        """The base's deletion plan with collateral strips and lost boundaries as counts."""
        plan = dict(info)
        plan["collateral"] = len(info["collateral"])
        plan["boundaries_lost"] = len(info["boundaries_lost"])
        return plan

    def plan_strip_deletion(
        self,
        edge: tuple[int, int] | None = None,
        preserve_boundaries: bool = False,
        skey: int | None = None,
    ) -> dict[str, Any]:
        """What deleting the strip would cost, without mutating anything.

        Returns ``skey``, ``faces``, ``collateral``, ``boundaries_lost`` (counts), ``ok`` and ``reason``.
        """
        info = super(CoarseEditor, self).plan_strip_deletion(edge=edge, preserve_boundaries=preserve_boundaries, skey=skey)
        return self._as_coarse_plan(info)

    def remove_strip(
        self,
        edge: tuple[int, int] | None = None,
        preserve_boundaries: bool = False,
        skey: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """**Delete the strip through** ``edge`` (or ``skey``). ``(ok, notes)``."""
        faces_in = self.mesh.number_of_faces()
        ok, notes = super(CoarseEditor, self).remove_strip(edge=edge, preserve_boundaries=preserve_boundaries, skey=skey)
        if not ok:
            return ok, notes
        notes = self._as_coarse_plan(notes)
        notes["faces_in"] = faces_in
        notes["faces_out"] = self.mesh.number_of_faces()
        self.last_deletion = dict(notes)
        return True, notes

    # ------------------------------------------------------------------
    # adding a strip -- Robin's rule, and the opening it needs
    # ------------------------------------------------------------------

    def add_strip(self, polyedge: list[int]) -> tuple[bool, dict[str, Any]]:
        """Unzip a polyedge of existing corners into a new strip (thesis 5.3.1). ``(ok, notes)``.

        Strip labels and densities survive; the new strip is opened by the grammar's exact rule.

        Parameters
        ----------
        polyedge : list[int]
            Corner keys, in order, each joined to the next by a coarse edge.
            Either closed on itself, or with both ends on the layout boundary or
            at a pole -- a strip has to run the full width of the layout, for the
            same reason a cut has to reach a wall.
        """
        polyedge = list(polyedge)
        if len(polyedge) < 3:
            return self._refuse("a polyedge needs more than two corners to add a strip along")

        for vkey in polyedge:
            if vkey not in self.mesh.vertex:
                return self._refuse("corner {} is not part of the layout".format(vkey))
        for u, v in pairwise(polyedge):
            if u == v or v not in self.mesh.halfedge[u]:
                return self._refuse("corners {} and {} are not joined by a coarse edge -- a polyedge is a chain of edges, not a list of picks".format(u, v))

        # A pole END is the grammar's (thesis Fig 5.15); a pole the line passes
        # THROUGH would be torn in two, so that is refused rather than guessed.
        poles = set(self.mesh.poles()) if hasattr(self.mesh, "poles") else set()
        on_pole = [vkey for vkey in polyedge[1:-1] if vkey in poles]
        if on_pole:
            return self._refuse("the line runs through the pole at corner {}. A strip may end at a pole, but not cross one".format(on_pole[0]))

        if not is_polyedge_valid_for_strip_addition(self.mesh, polyedge):
            return self._refuse(
                "that line has {} corner(s), is not closed, and does not end on "
                "the layout boundary at both ends. A strip has to run the full "
                "width of the layout -- wall to wall, or all the way "
                "round".format(len(polyedge))
            )

        work = self.mesh.copy()
        # A PRECONDITION, not housekeeping: ``update_strip_data`` does
        # ``max(attributes['strips']) + 1`` and raises ValueError on an empty dict.
        work.collect_strips()
        try:
            skey, old_to_new = _grammar_add_strip(work, list(polyedge), open_strip=True, project=self.project_to_wall)
        except Exception as exc:
            return self._refuse("the grammar could not add that strip ({}: {})".format(type(exc).__name__, exc))

        # The grammar opens every pair it can; one it could not stays coincident.
        opened = sum(1 for a, b in old_to_new.values() if distance_point_point(work.vertex_coordinates(a), work.vertex_coordinates(b)) > 1e-12)

        ok, reason = self._gate(work)
        if not ok:
            return self._refuse(reason)

        self.mesh = work
        self.edited = True
        self.last_cut = {"skey": skey, "corners": len(polyedge), "opened": opened, "faces_out": work.number_of_faces()}
        return self._accept(**self.last_cut)

    def _split_strips(self, work: "CoarsePseudoQuadMesh", to_split: dict[int, int]) -> dict[int, list[int]]:
        """Refine strips, opening each new one by the grammar's exact division.

        Not by smoothing: relaxation clusters corners along a chorded boundary.
        """
        out = {}
        for skey, n in (to_split or {}).items():
            keys = [skey]
            for _ in range(int(n) - 1):
                sides = work.strip_side_polyedges(skey)
                if not sides:
                    break
                new_skey, _old_to_new = _grammar_add_strip(work, list(sides[0]), open_strip=True, project=self.project_to_wall)
                keys.append(new_skey)
            out[skey] = keys
        return out

    # ------------------------------------------------------------------
    # moving a pole -- a relabel of the collapsed corner, nothing else
    # ------------------------------------------------------------------

    def pole_targets(self, pkey: int) -> list[int]:
        """The corners the pole at ``pkey`` may move to: those shared by all its triangles."""
        face_pole = self.mesh.attributes.get("face_pole") or {}
        faces = [fkey for fkey, pole in face_pole.items() if pole == pkey]
        if not faces:
            return []
        shared = set(self.mesh.face_vertices(faces[0]))
        for fkey in faces[1:]:
            shared &= set(self.mesh.face_vertices(fkey))
        shared.discard(pkey)
        return sorted(shared)

    def move_pole(self, pkey: int, vkey: int) -> tuple[bool, dict[str, Any]]:
        """Move the pole at ``pkey`` to corner ``vkey`` by relabelling its pseudo-quads. ``(ok, notes)``.

        Strip data is not carried through a commit afterwards; see ``pole_targets``.
        """
        face_pole = self.mesh.attributes.get("face_pole")
        if not face_pole:
            return self._refuse("this layout has no poles")
        faces = [fkey for fkey, pole in face_pole.items() if pole == pkey]
        if not faces:
            return self._refuse("corner {} is not a pole".format(pkey))
        if vkey not in self.mesh.vertex:
            return self._refuse("corner {} is not part of the layout".format(vkey))
        if vkey == pkey:
            return self._refuse("the pole is already at corner {}".format(pkey))
        for fkey in faces:
            if vkey not in self.mesh.face_vertices(fkey):
                x, y, _z = self.mesh.face_centroid(fkey)
                return self._refuse(
                    "corner {} is not a corner of the pole patch near ({:.2f}, "
                    "{:.2f}). A pseudo-quad can only collapse at one of its own "
                    "corners, so a pole moves only to a corner all its patches "
                    "share{}".format(vkey, x, y, " -- here: {}".format(self.pole_targets(pkey)) if self.pole_targets(pkey) else " -- and this pole has none")
                )

        work = self.mesh.copy()
        for fkey in faces:
            work.attributes["face_pole"][fkey] = vkey

        # All-quad only: a relabel changes no connectivity, so it cannot change
        # whether the layout is manifold -- and ``_gate``'s refusal for that
        # talks about deleting a strip.
        ok, reason = self._check_quads(work)
        if not ok:
            return self._refuse(reason)
        try:
            work.collect_strips()
        except Exception as exc:
            return self._refuse("the strips cannot be traced with the pole there ({}: {})".format(type(exc).__name__, exc))

        self.mesh = work
        self.edited = True
        self._strips_stale = True
        if self.poles is not None:
            old = self.target.vertex_coordinates(pkey) if pkey in self.target.vertex else None
            new = list(work.vertex_coordinates(vkey))
            self.poles = [new if old is not None and distance_point_point(p, old) < 1e-6 else p for p in self.poles]
        self.last_pole = {"pole": pkey, "to": vkey, "faces": len(faces)}
        return self._accept(**self.last_pole)

    # ------------------------------------------------------------------
    # undo
    # ------------------------------------------------------------------

    def snapshot(self) -> None:
        """Make the current layout, curves and poles the state ``reset`` returns to."""
        super(CoarseEditor, self).snapshot()
        self._snapshot_curves = dict(self.curves)
        self._snapshot_poles = None if self.poles is None else [list(p) for p in self.poles]

    def reset(self) -> tuple[bool, dict[str, Any]]:
        """Back to the layout this round of edits started from, curves and poles included."""
        ok, notes = super(CoarseEditor, self).reset()
        self.curves = dict(self._snapshot_curves)
        self.poles = None if self._snapshot_poles is None else [list(p) for p in self._snapshot_poles]
        self._topology_dirty = False
        self._strips_stale = False
        self.last_cut = {}
        self.last_pole = {}
        return ok, notes

    def _state(self) -> dict[str, Any]:
        """The base's mesh snapshot plus the curve map and the topology-dirty flag."""
        state = super(CoarseEditor, self)._state()
        state["curves"] = dict(self.curves)
        state["topology_dirty"] = self._topology_dirty
        state["last_cut"] = dict(self.last_cut)
        state["strips_stale"] = self._strips_stale
        state["last_pole"] = dict(self.last_pole)
        state["poles"] = None if self.poles is None else [list(p) for p in self.poles]
        return state

    def _restore(self, state: dict[str, Any]) -> None:
        super(CoarseEditor, self)._restore(state)
        self.curves = state["curves"]
        self._topology_dirty = state["topology_dirty"]
        self.last_cut = state["last_cut"]
        self._strips_stale = state["strips_stale"]
        self.last_pole = state["last_pole"]
        self.poles = state["poles"]

    def _rekey_curves(self, mesh: "CoarsePseudoQuadMesh") -> int:
        """Re-key the curve map onto a renumbered layout by rounded coordinates.

        Curves whose edge did not survive the repair are dropped and counted.
        """
        tol = 1e-3
        kept = {}
        lost = 0
        edges = [(mesh.vertex_coordinates(u), mesh.vertex_coordinates(v)) for u, v in mesh.edges()]
        for curve in self.curves.values():
            match = None
            for pa, pb in edges:
                if distance_point_point(curve[0], pa) < tol and distance_point_point(curve[-1], pb) < tol:
                    match = (pa, pb)
                    break
                if distance_point_point(curve[0], pb) < tol and distance_point_point(curve[-1], pa) < tol:
                    match = (pb, pa)
                    break
            if match is None:
                lost += 1
                continue
            curve = [list(p) for p in curve]
            curve[0], curve[-1] = list(match[0]), list(match[1])
            kept[self._curve_key(curve[0], curve[-1])] = curve
        self.curves = kept
        return lost

    # ------------------------------------------------------------------
    # what the layout densifies along
    # ------------------------------------------------------------------

    @property
    def all_polylines(self) -> list[list[list[float]]]:
        """Traced separatrices plus the user's drawn curves, as published to densification."""
        return [[list(p) for p in polyline] for polyline in self.polylines] + [[list(p) for p in curve] for curve in self.curves.values()]

    def edge_curves(self, loops: list[list[list[float]]] | None = None, warp: bool | None = None) -> tuple[dict[tuple[int, int], list[list[float]]], dict[str, Any]]:
        """``(mapping, tally)``: one polyline per coarse edge, for densification.

        Wall arc, then exact separatrix match, then (if ``warp``) a warped separatrix, else the chord.
        """
        mapping, tally = coarse_edges_to_curves(self.mesh, loops=loops or self.loops, polylines=self.all_polylines)
        tally = dict(tally)
        tally["warped"] = 0
        if self.edited if warp is None else warp:
            mapping, warped = warp_chorded_edges(mapping, self.mesh, self.all_polylines, self.warp_scale)
            tally["warped"] = warped
            tally["chord"] = max(0, tally.get("chord", 0) - warped)
        return mapping, tally

    # ------------------------------------------------------------------
    # commit
    # ------------------------------------------------------------------

    def commit(self) -> tuple["CoarsePseudoQuadMesh | None", dict[str, Any]]:
        """Write the edited layout back into the mesh this editor was given. ``(layout, notes)``.

        Moves only keep keys and strip data; a cut rebuilds via ``coarse_from_skeleton``.
        ``layout`` is ``None`` on refusal, with ``notes['error']``; ``notes['path']`` says which path ran.
        """
        poles = self.poles
        if poles is None:
            poles = [self.mesh.vertex_coordinates(v) for v in self.mesh.poles()] if hasattr(self.mesh, "poles") else []

        if self._topology_dirty:
            mesh, notes = coarse_from_skeleton(self.mesh, loops=self.loops, poles=poles, snap_tol=self.snap_tol)
            if mesh is None:
                return None, self._refuse(
                    "the edited layout has no usable patch ({} face(s) in, {} degenerate). A patch must be a closed polyline with one point per corner.".format(
                        notes["faces_in"], notes["degenerate"]
                    )
                )[1]

            ok, why = densifiable(mesh)
            if not ok:
                return None, self._refuse(
                    "the edited layout will not densify: {}. It is NOT being "
                    "replaced by a fallback -- fix the layout and hand it back. "
                    "({} patch(es) in, {} out, side counts {})".format(why, notes["faces_in"], notes["faces_out"], notes["sides"])
                )[1]

            self.mesh = mesh
            notes["lost_curves"] = self._rekey_curves(mesh)
            notes["path"] = "rebuild"
            self._transplant(mesh, carry_strip_data=False)
        else:
            snapped, off_wall = snap_to_loops(self.mesh, self.loops, self.snap_tol)
            ok, why = densifiable(self.mesh)
            if not ok:
                return None, self._refuse("the edited layout will not densify: {}. It is NOT being replaced by a fallback -- fix the layout and hand it back.".format(why))[1]

            sides = {}
            for fkey in self.mesh.faces():
                n = len(self.mesh.face_vertices(fkey))
                sides[n] = sides.get(n, 0) + 1
            faces = self.mesh.number_of_faces()
            notes = {
                "faces_in": faces,
                "faces_out": faces,
                "vertices": self.mesh.number_of_vertices(),
                "snapped": snapped,
                "off_wall": off_wall,
                "degenerate": 0,
                "sides": sides,
                "repair": "",
                "lost_curves": 0,
                "path": "moves",
            }
            # The grammar keeps its own strip table valid and preserves strip
            # labels, and a move changes no connectivity at all, so the densities
            # set per strip in step 5 survive a corner drag. A moved POLE is the
            # exception: the strips through its pseudo-quads are different ones.
            notes["path"] = "moves+pole" if self._strips_stale else "moves"
            self._transplant(self.mesh, carry_strip_data=not self._strips_stale)

        self.target.attributes["user_curves"] = [[list(p) for p in curve] for curve in self.curves.values()]
        self.snapshot()
        self._topology_dirty = False
        self._strips_stale = False
        self.committed = True
        self.last_reason = ""
        return self.target, notes
