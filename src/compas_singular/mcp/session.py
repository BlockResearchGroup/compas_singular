"""One mesh being improved, with a position-map undo stack and a history of every step.

Topology edits snapshot the whole mesh; the coarse layout has its own undo stack.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable

from compas_singular.framefield.quality import mesh_quality

if TYPE_CHECKING:
    from compas.datastructures import Mesh
    from compas.geometry import Polyline
    from compas_singular.algorithms.skeleton_decomposition import SkeletonDecomposition
    from compas_singular.datastructures import CoarsePseudoQuadMesh


__all__ = ["MeshSession", "UNDO_DEPTH"]


#: How many snapshots to keep. Position maps are small, but a long session with
#: an undo per step would still grow without bound.
UNDO_DEPTH = 20

#: Actions that change the COARSE layout, and so belong to its undo stack.
#: Everything else in the history belongs to the dense mesh. ``coarse_densify``
#: is dense: it reads the layout and replaces the dense mesh.
COARSE_ACTIONS = frozenset(
    (
        "create_coarse_mesh",
        "coarse_set_density",
        "coarse_set_pattern",
        "coarse_add_strip",
        "coarse_remove_strip",
        "coarse_divide",
        "coarse_move_corner",
        "rhino_pull_coarse",
        "coarse_load",
    )
)


class MeshSession(object):
    """The mesh in hand, the geometry constraining it, and what has been done."""

    def __init__(self) -> None:
        #: The authoritative mesh, in double precision. Rhino only gets a copy.
        self.mesh: Mesh | None = None
        #: Outer and inner boundaries as compas polylines. What the boundary
        #: vertices are allowed to slide along.
        self.walls: list[Polyline] = []
        #: Drawn guide curves, kept for reporting and for a later guide tool.
        self.guides: list[Polyline] = []
        #: Input point features, as [x, y, z]. Each one is a place the mesh is
        #: supposed to carry a pole; whether it does is a thing you see rather
        #: than measure, which is why they are drawn.
        self.points: list[list[float]] = []
        #: Where the mesh came from: ``{'kind': 'rhino'|'file', ...}``.
        self.source: dict[str, Any] | None = None
        #: The coarse layout, once ``create_coarse_mesh`` has built one --
        #: a ``CoarsePseudoQuadMesh``, or ``None``. Independent of ``mesh``:
        #: pulling a new dense mesh does not clear this, because
        #: ``coarse_densify`` adopts ITS OWN output through the same path and
        #: clearing the layout that just produced it would be wrong.
        self.coarse: CoarsePseudoQuadMesh | None = None
        #: The ``SkeletonDecomposition`` that built ``coarse``, kept so
        #: ``coarse_densify`` can ask it for curved-boundary edge shapes.
        self.decomposition: SkeletonDecomposition | None = None
        self.history: list[dict[str, Any]] = []
        self.remarks: list[dict[str, Any]] = []
        self.started = time.time()
        self._undo: list[dict[str, Any]] = []
        #: The coarse layout's own undo stack -- whole-mesh copies, not
        #: positions. See design_notes/mcp.md (session.py) for why.
        self._coarse_undo: list[dict[str, Any]] = []
        #: Monotonic counters behind ``visually_current``. Numbers rather
        #: than a bool because "seen since the last change" is the question, and
        #: a bool would need resetting in every tool that moves a vertex.
        self._changed_at = 0
        self._seen_at = 0
        #: The same pair for the coarse layout, which ``rhino_push_coarse``
        #: gates on exactly as ``rhino_push`` gates on the dense mesh's.
        self._coarse_changed_at = 0
        self._coarse_seen_at = 0

    # --------------------------------------------------------------------
    # loading
    # --------------------------------------------------------------------

    def adopt(
        self, mesh: Mesh, walls: Iterable[Any] | None = None, guides: Iterable[Any] | None = None, points: Iterable[Any] | None = None, source: dict[str, Any] | None = None
    ) -> MeshSession:
        """Take a new mesh, clearing the undo stack, and drop the coarse layout if the domain changed."""
        old_domain = self._domain()
        self.mesh = mesh
        if walls is not None:
            self.walls = _as_polylines(walls)
        if guides is not None:
            self.guides = _as_polylines(guides)
        if points is not None:
            self.points = [list(p) for p in points]
        if self.coarse is not None and self._domain() != old_domain:
            self.coarse = None
            self.decomposition = None
            self._coarse_undo = []
        self.source = source
        self._undo = []
        # A newly adopted mesh has never been looked at.
        self._changed_at = 1
        self._seen_at = 0
        return self

    def _domain(self) -> tuple[Any, Any, list[list[float]]]:
        """Walls, guides and points as plain nested lists, for comparison."""

        def curves(items: Iterable[Any]) -> list[list[list[float]]]:
            return [[list(p) for p in getattr(c, "points", c)] for c in items]

        return (curves(self.walls), curves(self.guides), [list(p) for p in self.points])

    @property
    def loaded(self) -> bool:
        return self.mesh is not None

    # --------------------------------------------------------------------
    # measuring
    # --------------------------------------------------------------------

    def quality(self, low_angle: float | None = None) -> dict[str, Any] | None:
        """``mesh_quality`` of the mesh in hand, or ``None`` if there is none."""
        if self.mesh is None:
            return None
        if low_angle is None:
            return mesh_quality(self.mesh)
        return mesh_quality(self.mesh, low_angle=low_angle)

    def triple(self) -> tuple[Any, Any, Any] | None:
        """``(min angle, max angle, max aspect)`` -- the three a step is judged on.

        The same three ``framefield.relax`` gates on, deliberately: a step that
        improves one and wrecks another has not improved the mesh.
        """
        metrics = self.quality()
        if metrics is None:
            return None
        return (metrics.get("min_angle"), metrics.get("max_angle"), metrics.get("aspect_max"))

    # --------------------------------------------------------------------
    # has anybody actually looked at it
    # --------------------------------------------------------------------

    def mark_changed(self) -> None:
        """Record that the mesh moved. Every tool that moves a vertex calls this."""
        self._changed_at += 1

    def mark_seen(self) -> None:
        """Record that the mesh has been rendered and looked at."""
        self._seen_at = self._changed_at

    @property
    def visually_current(self) -> bool:
        """Whether the mesh has been looked at since it last changed; delivery tools refuse until it has."""
        return self._seen_at >= self._changed_at

    def mark_coarse_seen(self) -> None:
        """Record that the coarse layout has been drawn and looked at."""
        self._coarse_seen_at = self._coarse_changed_at

    @property
    def coarse_visually_current(self) -> bool:
        """Whether the coarse LAYOUT has been looked at since it last changed.

        Bumped by every recorded coarse step and by ``undo_coarse``, so an edit
        that was refused -- and recorded nothing -- does not demand another look.
        """
        return self.coarse is not None and self._coarse_seen_at >= self._coarse_changed_at

    # --------------------------------------------------------------------
    # undo
    # --------------------------------------------------------------------

    def positions(self) -> dict[Any, list[float]]:
        return dict((key, list(self.mesh.vertex_coordinates(key))) for key in self.mesh.vertices())

    def snapshot(self, label: str = "", whole: bool = False) -> str | None:
        """Remember every vertex position, or with ``whole=True`` a copy of the mesh, on the undo stack."""
        if self.mesh is None:
            return None
        entry = {
            "label": label,
            "vertices": self.mesh.number_of_vertices(),
            "faces": self.mesh.number_of_faces(),
            "step": len(self.history),
        }
        if whole:
            entry["mesh"] = self.mesh.copy()
        else:
            entry["positions"] = self.positions()
        self._undo.append(entry)
        while len(self._undo) > UNDO_DEPTH:
            self._undo.pop(0)
        return label

    def can_undo(self) -> bool:
        return bool(self._undo)

    def undo(self) -> tuple[bool, str]:
        """Restore the most recent snapshot.

        Returns
        -------
        tuple
            ``(True, label)``, or ``(False, reason)``.
        """
        if self.mesh is None:
            return False, "nothing is loaded"
        if not self._undo:
            return False, "nothing to undo -- no snapshot has been taken"
        entry = self._undo.pop()
        if "mesh" in entry:
            self.mesh = entry["mesh"]
            self._mark_undone(entry["step"], "dense")
            self.mark_changed()
            return True, entry["label"] or "the last snapshot"
        if entry["vertices"] != self.mesh.number_of_vertices() or entry["faces"] != self.mesh.number_of_faces():
            # Reachable since the dense line tools: a POSITION snapshot taken
            # before a topology edit that was not itself undone first.
            return False, (
                "the mesh has {} vertices and {} faces, but the snapshot was taken at {} and {} -- a position undo cannot restore a changed topology".format(
                    self.mesh.number_of_vertices(), self.mesh.number_of_faces(), entry["vertices"], entry["faces"]
                )
            )
        for key, xyz in entry["positions"].items():
            self.mesh.vertex_attributes(key, "xyz", xyz)
        self._mark_undone(entry["step"], "dense")
        # An undo moves every vertex, so the last picture is out of date too.
        self.mark_changed()
        return True, entry["label"] or "the last snapshot"

    # --------------------------------------------------------------------
    # coarse-layout undo -- a different stack, because a strip edit changes
    # topology and a position map cannot restore that
    # --------------------------------------------------------------------

    def snapshot_coarse(self, label: str = "") -> str | None:
        """Remember the coarse layout whole, so a strip edit can be undone."""
        if self.coarse is None:
            return None
        self._coarse_undo.append({"label": label, "mesh": self.coarse.copy(), "step": len(self.history)})
        while len(self._coarse_undo) > UNDO_DEPTH:
            self._coarse_undo.pop(0)
        return label

    def can_undo_coarse(self) -> bool:
        return bool(self._coarse_undo)

    def undo_coarse(self) -> tuple[bool, str]:
        """Restore the coarse layout to its most recent snapshot.

        Returns
        -------
        tuple
            ``(True, label)``, or ``(False, reason)``.
        """
        if self.coarse is None:
            return False, "no coarse layout is loaded"
        if not self._coarse_undo:
            return False, "nothing to undo -- no coarse snapshot has been taken"
        entry = self._coarse_undo.pop()
        self.coarse = entry["mesh"]
        self._coarse_changed_at += 1
        self._mark_undone(entry["step"], "coarse")
        return True, entry["label"] or "the last coarse snapshot"

    def _mark_undone(self, step: int, layer: str) -> None:
        """Flag, not delete, the history steps an undo took back, for its own layer only."""
        for entry in self.history[step:]:
            if entry.get("layer") == layer:
                entry["undone"] = True

    def live_steps(self) -> list[dict[str, Any]]:
        """The history minus every step an undo took back."""
        return [entry for entry in self.history if not entry.get("undone")]

    # --------------------------------------------------------------------
    # the record
    # --------------------------------------------------------------------

    def record(self, action: str, before: Any = None, after: Any = None, **detail: Any) -> dict[str, Any]:
        """Append a step. Returns the entry, so a tool can report it back."""
        entry = {"step": len(self.history), "action": action, "layer": "coarse" if action in COARSE_ACTIONS else "dense", "at": time.time(), "before": before, "after": after}
        entry.update(detail)
        self.history.append(entry)
        if entry["layer"] == "coarse":
            self._coarse_changed_at += 1
        return entry

    def add_remark(self, text: str, about: Any = None) -> dict[str, Any]:
        remark = {"step": len(self.history), "text": text, "about": about, "at": time.time()}
        self.remarks.append(remark)
        return remark

    def remarks_for(self, step: int) -> list[str]:
        return [r["text"] for r in self.remarks if r["step"] == step]

    def state(self) -> dict[str, Any]:
        """A small summary. The reading in prose is ``describe``'s job."""
        mesh = self.mesh
        coarse = self.coarse
        return {
            "loaded": self.loaded,
            "source": self.source,
            "vertices": mesh.number_of_vertices() if mesh else None,
            "faces": mesh.number_of_faces() if mesh else None,
            "walls": len(self.walls),
            "guides": len(self.guides),
            "points": len(self.points),
            "steps": len(self.live_steps()),
            "undone_steps": len(self.history) - len(self.live_steps()),
            "remarks": len(self.remarks),
            "undo_depth": len(self._undo),
            "visually_current": self.visually_current,
            "coarse_loaded": coarse is not None,
            "coarse_faces": coarse.number_of_faces() if coarse else None,
            "coarse_strips": len(list(coarse.strips())) if coarse else None,
            "coarse_undo_depth": len(self._coarse_undo),
            "coarse_visually_current": self.coarse_visually_current,
        }

    def __repr__(self) -> str:
        state = self.state()
        return "<MeshSession {} faces, {} step(s)>".format(state["faces"], state["steps"])


def _as_polylines(curves: Iterable[Any] | None) -> list[Polyline]:
    """Point lists as compas polylines, which the smoothers want."""
    from compas.geometry import Polyline

    out = []
    for curve in curves or []:
        if isinstance(curve, Polyline):
            out.append(curve)
            continue
        points = getattr(curve, "points", curve)
        points = [list(p) for p in points]
        if len(points) >= 2:
            out.append(Polyline(points))
    return out
