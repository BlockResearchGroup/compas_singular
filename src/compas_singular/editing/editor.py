"""Shared base for the coarse and dense hand-editors, with no CAD in it.

Holds the working copy and the walls, moves a vertex, undoes, and deletes a strip.
Design notes: ``design_notes/editing.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any

from compas_singular.datastructures.mesh_quad.grammar.add_strip import split_strips
from compas_singular.datastructures.mesh_quad.grammar.delete_strip import collateral_strip_deletions
from compas_singular.datastructures.mesh_quad.grammar.delete_strip import delete_strip as _grammar_delete_strip
from compas_singular.datastructures.mesh_quad.grammar.delete_strip import strips_to_split_to_prevent_boundary_collapse
from compas_singular.datastructures.mesh_quad.grammar.delete_strip import total_boundary_deletions
from compas_singular.datastructures.mesh_quad_coarse.coarse_curves import BoundaryLoop
from compas_singular.datastructures.mesh_quad_coarse.coarse_curves import mean_edge_length

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = ["MeshEditor", "boundary_vertex_set"]


def boundary_vertex_set(mesh: "QuadMesh") -> set[int]:
    """Every vertex with a faceless halfedge. See ``MeshEditor.boundary_vertices``."""
    out = set()
    for u, nbrs in mesh.halfedge.items():
        for v, fkey in nbrs.items():
            if fkey is None:
                out.add(u)
                out.add(v)
    return out


class MeshEditor(object):
    """A quad mesh being hand-edited. No CAD, no prompts, no prints.

    Parameters
    ----------
    mesh : compas_singular.datastructures.QuadMesh
        The mesh to edit. Kept as ``target``; the editing happens on a copy.
    walls : list, optional
        The domain walls, as point lists, closed polylines or
        ``BoundaryLoop`` instances. A boundary vertex that moves is projected
        back onto the nearest one, so nudging the edge of the mesh does not eat
        the outline. Without walls a moved boundary vertex is left where it was
        put.
    work_on_copy : bool, optional
        Edit a copy and leave ``mesh`` untouched until a commit transplants into
        it. Default ``True``; ``False`` edits in place, for a caller that has
        already made its own copy.

    Attributes
    ----------
    target : mesh
        The object handed in. Only a commit writes to it.
    mesh : mesh
        The working copy. Replaced wholesale by every topological operation that
        succeeds, so hold the editor, not the mesh.
    walls : list
        The domain walls as ``BoundaryLoop`` instances.
    last_reason : str
        Why the last operation was refused. Empty when nothing was.
    """

    #: Whether a moved vertex is flattened onto z = 0. A coarse layout is planar
    #: by construction and its subclass turns this on; a dense mesh need not be.
    PLANAR = False

    #: ``attributes`` keys that describe the layout as it was BEFORE this edit --
    #: a DENSE mesh built from it, or geometry keyed to its vertices. They go
    #: stale the moment the topology changes, and carrying them across a rebuild
    #: is how a stale dense mesh gets served for an edited layout.
    #:
    #: ``edges_to_curves`` is here for the second reason and is worth spelling
    #: out: it maps ``(u, v)`` to the polyline that edge densifies along, so a
    #: rebuild that renumbers points it at the wrong edges, and a MOVED corner
    #: leaves it anchored where the corner used to be -- the two patches sharing
    #: that edge would then be handed different endpoints and fail to weld.
    #: Empty on any layout that did not come from ``from_coarse_polylines``, so
    #: clearing it is a no-op everywhere else.
    DERIVED_ATTRIBUTES = ("quad_mesh", "polygonal_mesh", "vertex_coarse_to_dense", "edge_coarse_to_dense", "edges_to_curves")

    #: ``attributes`` keys that are keyed BY STRIP INDEX, and so are meaningless
    #: once a rebuild has renumbered the strips.
    STRIP_KEYED_ATTRIBUTES = ("strips", "polyedges", "strips_density")

    def __init__(self, mesh: "QuadMesh", walls: list[Any] | None = None, work_on_copy: bool = True) -> None:
        if mesh is None:
            raise ValueError("no mesh to edit")
        self.target = mesh
        self.mesh = mesh.copy() if work_on_copy else mesh
        self.walls = self._as_walls(walls)
        #: The mesh as it was before the current round of edits. A whole mesh
        #: rather than a coordinate map: the topological operations change the
        #: connectivity, and putting coordinates back would leave the faces they
        #: created in place with nothing to say where they came from.
        self._snapshot = self.mesh.copy()
        self.last_reason = ""
        self.last_deletion = {}
        #: Whether anything has been changed since this editor was built. What
        #: the curvature branch keys off: on an UNEDITED layout every coarse edge
        #: still matches the separatrix it was traced from exactly, so there is
        #: nothing to warp and trying would only risk a wrong match.
        self.edited = False
        #: One entry per completed edit, oldest first -- see ``push_undo``.
        self._undo_stack = []

    # ------------------------------------------------------------------
    # walls
    # ------------------------------------------------------------------

    @staticmethod
    def _as_walls(walls: list[Any] | None) -> list[BoundaryLoop]:
        """``BoundaryLoop`` per wall. Accepts loops already built as one."""
        out = []
        for wall in walls or []:
            if isinstance(wall, BoundaryLoop):
                out.append(wall)
                continue
            points = list(wall)
            if len(points) >= 3:
                out.append(BoundaryLoop(points))
        return out

    def project_to_wall(self, xyz: list[float]) -> list[float]:
        """The nearest point of the nearest wall, or ``xyz`` if there is no wall.

        Only for vertices topologically on the mesh boundary.
        """
        if not self.walls:
            return list(xyz)
        best_d, best_p = float("inf"), None
        for wall in self.walls:
            d, _s, q = wall.project(xyz)
            if d < best_d:
                best_d, best_p = d, q
        return list(best_p) if best_p is not None else list(xyz)

    # ------------------------------------------------------------------
    # measuring the mesh
    # ------------------------------------------------------------------

    def mean_edge(self, mesh: "QuadMesh | None" = None) -> float:
        """Mean edge length -- the scale every tolerance here is expressed in."""
        return mean_edge_length(self.mesh if mesh is None else mesh)

    def boundary_vertices(self, mesh: "QuadMesh | None" = None) -> set[int]:
        """Every vertex on every boundary, holes included, as a set read off the halfedges."""
        mesh = self.mesh if mesh is None else mesh
        return boundary_vertex_set(mesh)

    def is_vertex_on_boundary(self, vkey: int, mesh: "QuadMesh | None" = None) -> bool:
        """Whether a vertex is on ANY boundary of the mesh -- holes included."""
        return vkey in self.boundary_vertices(mesh)

    def check_strip_count(self, mesh: "QuadMesh | None" = None) -> tuple[bool, int, int]:
        """Check thesis Eq 5.6: the number of open strips is ``E - 2F``. ``(ok, counted, E - 2F)``.

        Returns
        -------
        (bool, int, int)
            Whether it holds, the counted number of open strips, and ``E - 2F``.

        Notes
        -----
        Collected on a COPY, so calling it never leaves strip data on a mesh the
        caller did not ask to have it.
        """
        mesh = self.mesh if mesh is None else mesh
        work = mesh.copy()
        work.collect_strips()
        counted = sum(1 for skey in work.attributes["strips"] if not work.is_strip_closed(skey))
        expected = work.number_of_edges() - 2 * work.number_of_faces()
        return counted == expected, counted, expected

    # ------------------------------------------------------------------
    # returning
    # ------------------------------------------------------------------

    def _refuse(self, reason: str) -> tuple[bool, dict[str, str]]:
        """Record why an operation was not performed. Always ``(False, notes)``."""
        self.last_reason = reason
        return False, {"error": reason}

    def _accept(self, **notes: Any) -> tuple[bool, dict[str, Any]]:
        """Record a success. Always ``(True, notes)``."""
        self.last_reason = ""
        return True, notes

    # ------------------------------------------------------------------
    # move one vertex
    # ------------------------------------------------------------------

    def move_vertex(self, vkey: int, xyz: list[float], project: bool = True) -> tuple[bool, dict[str, Any]]:
        """Move one vertex, projecting a boundary vertex back onto its wall unless ``project`` is off. ``(ok, notes)``."""
        if vkey not in self.mesh.vertex:
            return self._refuse("vertex {} is not in the mesh".format(vkey))

        point = self._as_point(xyz)
        projected = False
        if project and self.is_vertex_on_boundary(vkey):
            point = self._as_point(self.project_to_wall(point))
            projected = True

        self.mesh.vertex_attributes(vkey, "xyz", point)
        self.edited = True
        return self._accept(moved=vkey, xyz=point, projected=projected)

    def _as_point(self, xyz: list[float]) -> list[float]:
        """``xyz`` as three floats, flattened onto z = 0 when ``PLANAR``."""
        point = list(xyz)
        while len(point) < 3:
            point.append(0.0)
        return [float(point[0]), float(point[1]), 0.0 if self.PLANAR else float(point[2])]

    # ------------------------------------------------------------------
    # undo
    # ------------------------------------------------------------------

    def snapshot(self) -> None:
        """Make the current mesh the state ``reset`` returns to."""
        self._snapshot = self.mesh.copy()

    def reset(self) -> tuple[bool, dict[str, Any]]:
        """Restore the whole mesh as it was at construction or at the last snapshot."""
        self.mesh = self._snapshot.copy()
        self.last_deletion = {}
        self.edited = False
        return self._accept(faces=self.mesh.number_of_faces())

    # ------------------------------------------------------------------
    # undo -- one step back, not all the way to the start
    # ------------------------------------------------------------------
    #
    # ``reset`` answers "throw away this whole round of edits"; this
    # answers "that last one, not the others". A front end that acts on a pick
    # immediately -- deleting the strip the moment it is clicked, say, rather
    # than asking first -- needs this to make an accidental pick cheap to walk
    # back, without losing everything edited before it.

    def _state(self) -> dict[str, Any]:
        """Everything ``undo`` needs to put back. A subclass with more
        state than the mesh (a curve map, a dirty flag) extends this and
        ``_restore`` together, never one without the other."""
        return {"mesh": self.mesh.copy(), "edited": self.edited, "last_deletion": dict(self.last_deletion)}

    def _restore(self, state: dict[str, Any]) -> None:
        """The inverse of ``_state``."""
        self.mesh = state["mesh"]
        self.edited = state["edited"]
        self.last_deletion = state["last_deletion"]

    def push_undo(self) -> None:
        """Snapshot the mesh before an operation that may mutate it."""
        self._undo_stack.append(self._state())

    def discard_last_undo(self) -> None:
        """Drop the most recent ``push_undo`` snapshot -- nothing changed."""
        if self._undo_stack:
            self._undo_stack.pop()

    def undo(self) -> tuple[bool, dict[str, Any]]:
        """Put back the mesh as it was before the last ``push_undo``. ``(ok, notes)``.

        Refuses with nothing to restore rather than silently doing nothing, so
        a front end can tell "undid something" from "there was nothing left".
        """
        if not self._undo_stack:
            return self._refuse("nothing to undo")
        self._restore(self._undo_stack.pop())
        self.last_reason = ""
        return self._accept(faces=self.mesh.number_of_faces(), remaining=len(self._undo_stack))

    # ------------------------------------------------------------------
    # deleting a strip -- the template
    # ------------------------------------------------------------------

    def _split_strips(self, work: "QuadMesh", to_split: dict[int, int]) -> dict[int, list[int]]:
        """Refine the strips that would otherwise let a boundary collapse (thesis 5.3.2).

        The new strips have zero width; subclasses must open them.
        """
        return split_strips(work, to_split, open_strip=False)

    def _gate(self, work: "QuadMesh") -> tuple[bool, str]:
        """Whether a trial result is acceptable. ``(ok, reason)``. The base accepts anything."""
        return True, ""

    def strip_through(self, edge: tuple[int, int]) -> int | None:
        """The strip key through ``edge``, resolved on a copy. ``None`` if none."""
        work = self.mesh.copy()
        work.collect_strips()
        return work.edge_strip(tuple(edge))

    def _empty_plan(self, reason: str = "") -> dict[str, Any]:
        """A deletion plan with nothing done yet: every key ``plan_strip_deletion`` publishes."""
        return {
            "ok": False,
            "reason": reason,
            "skey": None,
            "faces": 0,
            "collateral": [],
            "boundaries_lost": [],
            "to_split": {},
            "can_preserve": True,
            "faces_before": self.mesh.number_of_faces(),
            "faces_after": None,
            "vertices_before": self.mesh.number_of_vertices(),
            "vertices_after": None,
        }

    def _trial_delete(
        self,
        edge: tuple[int, int] | None = None,
        preserve_boundaries: bool = False,
        skey: int | None = None,
    ) -> tuple["QuadMesh | None", dict[str, Any]]:
        """Delete a strip (by ``edge`` or ``skey``) on a copy. ``(mesh or None, info)``.

        Shared by ``plan_strip_deletion`` and ``remove_strip`` so they cannot disagree.
        """
        info = self._empty_plan()

        work = self.mesh.copy()
        work.collect_strips()

        if skey is None:
            if edge is None:
                info["reason"] = "no edge or strip given"
                return None, info
            skey = work.edge_strip(tuple(edge))
        if skey is None or skey not in work.attributes["strips"]:
            info["reason"] = "that edge belongs to no strip -- nothing to remove"
            return None, info

        info["skey"] = skey
        info["faces"] = len(work.strip_faces(skey))
        info["collateral"] = list(collateral_strip_deletions(work, [skey]))
        # ``total_boundary_deletions`` returns the BOUNDARIES themselves -- a list
        # of vertex rings -- despite the name. Formatting it with ``{}`` prints a
        # list of rings where a number was meant, so callers take ``len``.
        info["boundaries_lost"] = list(total_boundary_deletions(work, [skey]))

        # **Thesis Eq 5.11, and this helper is the only thing that implements
        # it.** A boundary collapses when fewer than THREE edges represent it
        # after the deletion. ``total_boundary_deletions`` above asks a stricter
        # and different question -- whether EVERY one of its edges is consumed --
        # so it cannot stand in for this, and gating the pre-split on it meant
        # never pre-splitting in the cases the thesis is actually about.
        #
        # The helper is tri-state and all three states matter:
        #
        #   ``None``   the collapse cannot be prevented: no strip survives on that
        #              boundary to refine;
        #   ``{}``     nothing is at risk, so there is nothing to do;
        #   ``{...}``  refine these first -- one survivor, split it in three; two
        #              survivors, split each in two, "to avoid any bias" (5.3.2).
        #
        # Flattening ``None`` and ``{}`` together is what made asking to preserve
        # boundaries refuse a deletion that risked none.
        at_risk = strips_to_split_to_prevent_boundary_collapse(work, [skey])
        # Reported whether or not it was asked for: a front end wants to say "this
        # would cost you a hole, and here is what I would have to split to save
        # it" before the user decides.
        info["to_split"] = {} if at_risk is None else dict(at_risk)
        info["can_preserve"] = at_risk is not None

        if preserve_boundaries:
            if at_risk is None:
                info["reason"] = "this strip cannot go without collapsing a boundary, and there is no strip left on that boundary to split to prevent it"
                return None, info
            if at_risk:
                self._split_strips(work, at_risk)
                # ``to_split`` never contains the strip being deleted, but the
                # split rebuilt the strip data, so re-resolve rather than trust
                # the key.
                if edge is not None:
                    skey = work.edge_strip(tuple(edge))
                    if skey is None:
                        info["reason"] = "lost track of the strip after the pre-split"
                        return None, info
                    info["skey"] = skey

        try:
            _grammar_delete_strip(work, skey)
        except Exception as exc:
            # Measured: 1 strip of 24 on a two-hole plate raises ``KeyError`` from
            # inside the grammar. Catching it costs the user a refusal instead of
            # a traceback.
            info["reason"] = "the grammar could not delete that strip ({}: {})".format(type(exc).__name__, exc)
            return None, info

        if work.number_of_faces() == 0:
            info["reason"] = "that would delete every face of the mesh"
            return None, info

        ok, reason = self._gate(work)
        if not ok:
            info["reason"] = reason
            return None, info

        info["ok"] = True
        info["faces_after"] = work.number_of_faces()
        info["vertices_after"] = work.number_of_vertices()
        return work, info

    def plan_strip_deletion(
        self,
        edge: tuple[int, int] | None = None,
        preserve_boundaries: bool = False,
        skey: int | None = None,
    ) -> dict[str, Any]:
        """What deleting the strip would cost, predicted on a copy: moved vertices, collateral strips, lost boundaries.

        Returns
        -------
        dict
            ``ok``, ``reason``, ``skey``, ``faces``, ``collateral`` (a LIST),
            ``boundaries_lost`` (a list of vertex rings), ``to_split``, and the
            before/after face and vertex counts. A subclass that publishes
            different names or shapes adapts this at its own boundary.
        """
        _work, info = self._trial_delete(edge, preserve_boundaries=preserve_boundaries, skey=skey)
        self.last_reason = "" if info["ok"] else info["reason"]
        return info

    def remove_strip(
        self,
        edge: tuple[int, int] | None = None,
        preserve_boundaries: bool = False,
        skey: int | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Delete the strip through ``edge`` (or ``skey``), on a copy adopted only if it passes ``_gate``. ``(ok, notes)``.

        Collapses the band and welds its sides; it does not dissolve the picked line.
        """
        work, info = self._trial_delete(edge, preserve_boundaries=preserve_boundaries, skey=skey)
        if work is None:
            return self._refuse(info["reason"])

        self.mesh = work
        self.edited = True
        self.last_deletion = dict(info)
        return self._accept(**info)

    # ------------------------------------------------------------------
    # handing the result back to the caller's own object
    # ------------------------------------------------------------------

    def _transplant(self, source: "QuadMesh", carry_strip_data: bool = False) -> "QuadMesh":
        """Make ``target`` become ``source`` in place, keeping the caller's reference.

        Strip data is carried only when ``carry_strip_data``; a rebuild renumbers strips.
        """
        target = self.target
        route = target.attributes.get("route")
        # A symmetric unit's seams, group and tolerances (``compas_singular.symmetry``)
        # describe the DOMAIN the layout lives in, like the route -- a rebuild that
        # renumbers the layout does not change them, and dropping them would leave
        # a unit that can no longer be expanded.
        symmetry = target.attributes.get("symmetry")

        target.clear()
        # ``clear()`` deliberately does not do this, and the omission is the trap:
        # a stale 'face_pole' or 'quad_mesh' would survive the transplant.
        target.attributes.clear()
        target.attributes.update(deepcopy(source.attributes))

        for vkey in source.vertices():
            target.add_vertex(key=vkey, attr_dict=dict(source.vertex[vkey]))
        for fkey in source.faces():
            target.add_face(source.face_vertices(fkey), fkey=fkey)
        for fkey, data in getattr(source, "facedata", {}).items():
            target.facedata[fkey] = deepcopy(data)
        for edge, data in getattr(source, "edgedata", {}).items():
            target.edgedata[edge] = deepcopy(data)

        # The dense mesh and the coarse-to-dense maps describe the layout as it
        # was BEFORE this edit. Keeping them is how a stale mesh gets served.
        for key in self.DERIVED_ATTRIBUTES:
            if key in target.attributes:
                target.attributes[key] = None if "mesh" in key else {}
        if not carry_strip_data:
            for key in self.STRIP_KEYED_ATTRIBUTES:
                if key in target.attributes:
                    target.attributes[key] = {}
        # The route is a property of the layout, not of the rebuild that produced
        # it, so it comes from the object being written into rather than the copy.
        if route is not None:
            target.attributes["route"] = route
        if symmetry:
            target.attributes["symmetry"] = deepcopy(symmetry)
        return target
