"""``FieldDecomposition``: the frame-field counterpart of ``SkeletonDecomposition``, with the same method names.

Solve the field, trace separatrices, build the planar network, recover and repair the coarse layout, densify.
"""

from __future__ import annotations

import traceback
import warnings
from typing import TYPE_CHECKING
from typing import Any

from compas.geometry import distance_point_point
from compas.itertools import pairwise
from compas.tolerance import TOL
from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.editing.rebuild import coarse_from_skeleton
from compas_singular.editing.rebuild import mesh_from_faces
from compas_singular.editing.rebuild import warp_polyline
from compas_singular.editing.repair import densifiable
from compas_singular.editing.repair import solve_non_quad_faces
from compas_singular.editing.repair import topological_quad_split
from compas_singular.framefield.arrangement import faces_from_arrangement
from compas_singular.framefield.arrangement import faces_with_repeated_vertices
from compas_singular.framefield.arrangement import planar_arrangement
from compas_singular.framefield.background import BackgroundMesh
from compas_singular.framefield.densify import field_densification
from compas_singular.framefield.field import CrossField
from compas_singular.framefield.quality import hard_floor
from compas_singular.framefield.quality import mesh_quality
from compas_singular.framefield.separatrix_network import SHARP_TURN
from compas_singular.framefield.separatrix_network import boundary_corners
from compas_singular.framefield.separatrix_network import build_network
from compas_singular.framefield.trace import Tracer
from compas_singular.geometry.polyline import closest_on_polyline
from compas_singular.geometry.polyline import distance_to_polyline
from compas_singular.geometry.polyline import loop_arc_lengths
from compas_singular.geometry.polyline import loop_parameter
from compas_singular.geometry.polyline import point_at_length
from compas_singular.geometry.polyline import project_on_polyline
from compas_singular.geometry.polyline import signed_area

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh
    from compas_singular.datastructures import QuadMesh
    from compas_singular.framefield.symmetry import Symmetry
    from compas_singular.framefield.trace import Separatrix
    from compas_singular.symmetry.report import SymmetryReport
    from compas_singular.symmetry.unit import SymmetricUnit


__all__ = ["FieldDecomposition"]


#: The routes ``FieldDecomposition.quad_mesh`` can take, best first.
ROUTES = ("field", "polygon", "triangulation")


def _on_loop(point: list[float], loop: list[list[float]], tol: float = 1e-6) -> bool:
    """Whether a point lies within ``tol`` of a closed loop."""
    return closest_on_polyline(point, loop, closed=True)[3] < tol


def _loop_chord_tolerance(loop: list[list[float]]) -> float:
    """How far off a loop the curve it samples can lie (its largest sagitta); ``1e-6`` for a polygon."""
    from math import acos

    pts = list(loop)
    n = len(pts)
    if n < 3:
        return 1e-6

    worst = 0.0
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        ux, uy = b[0] - a[0], b[1] - a[1]
        wx, wy = c[0] - b[0], c[1] - b[1]
        lu = (ux * ux + uy * uy) ** 0.5
        lw = (wx * wx + wy * wy) ** 0.5
        if lu < 1e-12 or lw < 1e-12:
            continue
        cosv = max(-1.0, min(1.0, (ux * wx + uy * wy) / (lu * lw)))
        if acos(cosv) > SHARP_TURN:
            continue
        worst = max(worst, distance_to_polyline(b, [a, c]))
    return max(1e-6, 0.5 * worst)


def _arcs_between(
    loop: list[list[float]],
    pa: list[float],
    pb: list[float],
) -> list[list[list[float]]]:
    """Both ways round a closed loop between two points on it, shorter first. Empty if they coincide."""
    pts = list(loop)
    ring = pts + pts[:1]
    _seg, cum = loop_arc_lengths(pts)
    total = cum[-1]
    if total == 0.0:
        return []

    def point_at(s: float) -> list[float]:
        return point_at_length(ring, cum, s % total, tol=1e-9)

    def span_arc(s0: float, span: float, reverse: bool) -> list[list[float]]:
        arc = [list(pa if not reverse else pb)]
        for i in range(1, len(pts) + 1):
            arc.append(point_at(s0 + span * i / float(len(pts) + 1)))
        arc.append(list(pb if not reverse else pa))
        if reverse:
            arc.reverse()
        arc[0], arc[-1] = list(pa), list(pb)
        return arc

    sa = loop_parameter(pa, pts, cum)[1]
    sb = loop_parameter(pb, pts, cum)[1]
    forward = (sb - sa) % total
    backward = total - forward
    if forward <= 1e-12 or backward <= 1e-12:
        return []

    ahead = (sa, forward, False)
    behind = (sb, backward, True)
    order = (ahead, behind) if forward <= backward else (behind, ahead)
    return [span_arc(*args) for args in order]


def _arc_between(loop: list[list[float]], pa: list[float], pb: list[float]) -> list[list[float]] | None:
    """The shorter arc of a closed loop between two points on it, pa -> pb."""
    found = _arcs_between(loop, pa, pb)
    return found[0] if found else None


class FieldDecomposition(object):
    """Coarse quad layout from a cross field's separatrices.

    Attributes
    ----------
    background : BackgroundMesh
    field : CrossField
    separatrices : list[Separatrix]
    tracer : Tracer
    inputs : dict
        The arguments ``from_boundary`` was called with, so the same route
        can be rebuilt on another domain (a symmetric unit, say).
    mesh : CoarsePseudoQuadMesh or None
        The coarse layout, once ``decomposition_mesh`` has run -- or the
        triangulated backstop after a ``quad_mesh`` that fell back.
    dense : QuadMesh or None
        The last mesh ``quad_mesh`` returned.
    polylines : list or None
        The network the layout was built from, plus ``user_curves``.
    guides : list
        The guide curves the field was solved with.
    symmetry : Symmetry or None
        The in-solver symmetry group (``framefield.symmetry``).
    symmetry_report : SymmetryReport or None
        What ``find_symmetry`` found.
    trace_report, snap_report, densify_stats, edit_notes : dict
        What tracing, singularity snapping, densification and
        ``edit_coarse`` / ``edges_to_curves`` did.
    repair_notes : list[str]
        Everything that degraded the result, in words. See ``warnings``.
    last_error : str or None
        The traceback of an exception ``quad_mesh`` caught and fell back
        from.
    user_curves : list
        Curves the user drew on the layout (a line added in Rhino); kept in
        ``polylines`` so the edge they made densifies along them. Set with
        ``set_user_curves``.
    field_aware : bool
        ``False`` densifies patch interiors as plain Coons patches. For A/B
        measurements only.
    """

    def __init__(
        self,
        background: BackgroundMesh | None,
        field: CrossField | None,
        separatrices: list[Separatrix],
        tracer: Tracer | None,
    ) -> None:
        self.background = background
        self.field = field
        self.inputs = {}
        self.symmetry_report = None
        self.separatrices = separatrices
        self.tracer = tracer
        self.mesh = None
        self.dense = None
        self.polylines = None
        self.guides = []
        self.trace_report = {}
        self.repair_notes = []
        self.last_error = None
        self.symmetry = None
        self.snap_report = {}
        self.field_aware = True
        self.densify_stats = {}
        self.edit_notes = {}
        self.user_curves = []
        self._network = None
        # whether ``mesh`` is a hand edit: it is then not regenerated, and not
        # rerouted for covering less than the whole domain
        self._edited = False
        # the ``poles`` ``mesh`` was built with -- decomposition_mesh's cache key
        self._mesh_poles = None
        self._route = "field"

    def warnings(self) -> list[str]:
        """Everything that silently degrades the layout, in words; check ``arm_mismatch`` first."""
        out = []
        for fkey, got, expected in self.trace_report.get("arm_mismatch") or []:
            out.append("singularity {}: found {} launch directions, expected {}".format(fkey, got, expected))
        out.extend(self.repair_notes)
        for reason in ("cycle", "lost", "length"):
            n = (self.trace_report.get("reasons") or {}).get(reason)
            if n:
                out.append("{} separatrix/ices ended with reason {!r}".format(n, reason))
        return out

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_boundary(
        cls,
        outer_boundary: Any,
        inner_boundaries: Any = None,
        guides: Any = None,
        mode: str = "perpendicular",
        target_length: float | None = None,
        orthogonal: bool | None = None,
        guide_weight: float | None = 1.0,
        guide_band: float | None = None,
        relax: bool = False,
        tau: float | None = None,
        symmetry: Symmetry | str | None = "auto",
        solve: bool = True,
        field_tau: float | None = None,
    ) -> FieldDecomposition:
        """Solve the field and trace its separatrices.

        Parameters
        ----------
        outer_boundary : list[[x, y, z]]
        inner_boundaries : list[list[[x, y, z]]], optional
        guides : list[list[[x, y, z]]], optional
            Guide curves -- cables, force lines. A LIST of curves; each may be a
            list of points or a compas ``Polyline``.
        mode, target_length, guide_weight, guide_band, relax, tau, symmetry : optional
            As ``CrossField.from_boundary``. ``target_length`` is the
            background spacing, not the quad size -- that is set later by
            ``set_strips_density_target`` or ``quad_mesh``. With the
            default ``symmetry='auto'`` a symmetric input gives a symmetric
            layout; an asymmetric one is unaffected.
        orthogonal : bool, optional
            Only orthogonal (cross) fields are implemented; ``False`` raises.
        solve : bool, optional
            ``False`` stores the inputs and returns without solving, for
            ``find_symmetry`` and ``symmetry_unit``. Anything that needs
            the field raises until ``solve`` is called.
        field_tau : float, optional
            Old name of ``tau``.
        """
        if field_tau is not None:
            warnings.warn("'field_tau' is now 'tau'", DeprecationWarning, stacklevel=2)
            if tau is None:
                tau = field_tau
        inputs = {
            "outer_boundary": outer_boundary,
            "inner_boundaries": inner_boundaries,
            "guides": guides,
            "mode": mode,
            "target_length": target_length,
            "orthogonal": orthogonal,
            "guide_weight": guide_weight,
            "guide_band": guide_band,
            "relax": relax,
            "tau": tau,
            "symmetry": symmetry,
        }
        if not solve:
            out = cls(None, None, [], None)
            out.inputs = inputs
            return out

        if orthogonal is False:
            raise NotImplementedError("non-orthogonal frame fields are not implemented; only cross fields are solved")

        field = CrossField.from_boundary(
            outer_boundary,
            inner_boundaries=inner_boundaries,
            guides=guides,
            mode=mode,
            target_length=target_length,
            guide_weight=guide_weight,
            guide_band=guide_band,
            relax=relax,
            tau=tau,
            symmetry=symmetry,
        )

        tracer = Tracer(field, singularity_points=field.singularity_points)
        separatrices, trace_report = tracer.separatrices()
        out = cls(field.background, field, separatrices, tracer)
        out.guides = list(field.guides)
        out.trace_report = trace_report
        out.symmetry = field.symmetry
        out.snap_report = field.snap_report
        out.inputs = inputs
        return out

    @property
    def is_solved(self) -> bool:
        return self.field is not None

    def _require_solved(self) -> None:
        if self.field is None:
            raise ValueError("this FieldDecomposition was built with solve=False and has no field yet -- call solve() first, or use symmetry_unit()")

    def solve(self) -> FieldDecomposition:
        """**The solved decomposition** for the stored inputs -- a NEW object,
        or this one if it is already solved."""
        if self.field is not None:
            return self
        return type(self).from_boundary(**self.inputs)

    @classmethod
    def from_mesh(cls, trimesh: Mesh, **kwargs: Any) -> FieldDecomposition:
        """Construct from a triangulation's BOUNDARY, as
        ``SkeletonDecomposition.from_mesh`` does. The interior is rebuilt at a
        spacing a field can live on."""
        loops = sorted(trimesh.boundaries(), key=len, reverse=True)
        outer = [trimesh.vertex_coordinates(v) for v in loops[0]]
        inners = [[trimesh.vertex_coordinates(v) for v in loop] for loop in loops[1:]]
        return cls.from_boundary(outer, inners, **kwargs)

    # ------------------------------------------------------------------
    # symmetry
    # ------------------------------------------------------------------

    def find_symmetry(
        self,
        tol: float | None = None,
        include: tuple[str, ...] = ("walls", "holes", "guides", "poles"),
        max_order: int = 12,
    ) -> SymmetryReport:
        """**Detect the symmetry of the domain** -- walls, holes, guides.

        Works without a solve (``solve=False``). Kept on ``symmetry_report``.

        Returns
        -------
        compas_singular.symmetry.SymmetryReport
        """
        from compas_singular.symmetry import find_symmetry
        from compas_singular.symmetry.routes import domain_of

        self.symmetry_report = find_symmetry(domain=domain_of(self), tol=tol, include=include, max_order=max_order)
        return self.symmetry_report

    def symmetry_unit(
        self,
        keys: Any = None,
        centre: str = "route",
        seam: Any = None,
        report: SymmetryReport | None = None,
    ) -> SymmetricUnit:
        """Solve and trace one symmetric unit of the domain with this decomposition's settings.

        Parameters are those of ``SkeletonDecomposition.symmetry_unit``.

        Returns
        -------
        compas_singular.symmetry.SymmetricUnit
        """
        from compas_singular.symmetry import build_unit
        from compas_singular.symmetry.routes import mesher_for

        report = report or self.symmetry_report or self.find_symmetry()
        return build_unit(report, mesher_for(self), keys=keys, centre=centre, seam=seam)

    # ------------------------------------------------------------------
    # the SkeletonDecomposition-shaped API
    # ------------------------------------------------------------------

    def decomposition_polylines(self) -> list[list[list[float]]]:
        """All the polylines forming the decomposition -- separatrices and walls."""
        boundary, others, _ = self._build()
        self.polylines = boundary + others
        return self.polylines

    def decomposition_mesh(self, poles: Any = (), force: bool = False) -> CoarsePseudoQuadMesh:
        """The all-quad coarse mesh, cached so densities and strips set on it survive.

        Falls back to a one-polygon or triangulated layout; ``route`` says which.

        Parameters
        ----------
        poles : list[[x, y, z]], optional
            Preferred pole positions: where a triangular patch the repair turns
            into a pseudo-quad puts its collapsed corner.
        force : bool, optional
            Rebuild even when the cached layout would do.

        Returns
        -------
        CoarsePseudoQuadMesh
        """
        self._require_solved()
        poles_key = tuple(tuple(round(float(c), 6) for c in point) for point in (poles or ()))
        if not force and self.mesh is not None and not self._edited and self._mesh_poles == poles_key:
            return self.mesh

        boundary, others, _ = self._build()
        self.polylines = boundary + others
        self.repair_notes = []
        self._route = "field"

        if not others and self.background.inners:
            # no interior line and a hole: the wall arcs belong to two loops, and
            # no single face can be built from them
            mesh = self._triangulation_fallback(
                "no interior separatrices and {} boundary loops -- a domain with a hole has no single-patch layout".format(1 + len(self.background.inners))
            )
        elif not others:
            # nothing cuts the domain -- a square, a hexagon: one patch
            mesh = self._single_patch(boundary)
        elif self._cut_holes():
            # every patch corner lies on a wall, which ``from_polylines`` cannot
            # return (it keeps only faces with a vertex off the boundary)
            faces = faces_from_arrangement(boundary, others, self.background.loops)
            mesh = self._mesh_from_faces(faces)
            if mesh is None:
                mesh = self._triangulation_fallback("a hole was cut but the arrangement yielded no face")
        else:
            mesh = CoarsePseudoQuadMesh.from_polylines(boundary, others)
            # a face visiting a vertex twice means the network was not planar
            repeated = faces_with_repeated_vertices(mesh)
            if repeated:
                self.repair_notes.append(
                    "ARRANGEMENT WARNING: {} recovered face(s) repeat a vertex "
                    "({}). The polyline network is not a planar arrangement; the "
                    "layout that follows is unreliable.".format(len(repeated), repeated[:5])
                )

        mesh, note = solve_non_quad_faces(mesh, CoarsePseudoQuadMesh, self.background.loops, poles=poles)
        if note:
            self.repair_notes.append(note)

        ok, reason = densifiable(mesh)
        if not ok:
            # a layout that does not close cannot be subdivided into one that
            # does; a mesh that ignores the field beats no mesh
            fallback = self._fallback_patch(boundary, reason)
            if fallback is not None:
                mesh = fallback

        mesh.attributes["decomposition_type"] = "field"
        self.mesh = mesh
        self._mesh_poles = poles_key
        self._edited = False
        return self.mesh

    def coarse_mesh(self, poles: Any = (), force: bool = False) -> CoarsePseudoQuadMesh:
        """**The coarse quad layout** -- the name both front ends answer to.
        See ``decomposition_mesh``."""
        return self.decomposition_mesh(poles=poles, force=force)

    def get_field(self) -> CrossField:
        """**The cross field**, to hand to ``densification(field=...)`` of any
        coarse layout. The same object as ``field``."""
        self._require_solved()
        return self.field

    def edit_coarse(
        self,
        geometry: Any,
        poles: Any = None,
        snap_tol: float | None = None,
        strict: bool = True,
    ) -> CoarsePseudoQuadMesh:
        """Take a hand-edited coarse layout in place of the generated one: weld, snap, repair and check it.

        Parameters
        ----------
        geometry : mesh or list
            The edited layout: a mesh, or one closed polyline per patch with one
            point per corner. See ``editing.rebuild.coarse_from_skeleton``.
        poles : list[[x, y, z]], optional
            Preferred pole positions. Defaults to the current layout's.
        snap_tol : float, optional
            How far a corner on the layout's boundary may be moved onto a wall.
            Defaults to half the background spacing; ``0`` disables snapping.
        strict : bool, optional
            Raise if the edited layout will not densify (default), rather than
            keep it with a note.

        Returns
        -------
        CoarsePseudoQuadMesh
            A NEW mesh: vertices are renumbered and no strips are collected.
        """
        # ``user_curves`` go back into the network on every call: nothing
        # traced them, and without them an edge the user drew densifies as a chord
        boundary, others, _ = self._build()
        self.polylines = boundary + others + list(self.user_curves)

        if poles is None:
            poles = [self.mesh.vertex_coordinates(v) for v in self.mesh.poles()] if self.mesh is not None and hasattr(self.mesh, "poles") else []
        if snap_tol is None:
            snap_tol = self.background.target_length * 0.5

        mesh, notes = coarse_from_skeleton(geometry, loops=self.background.loops, poles=poles, snap_tol=snap_tol, cls=CoarsePseudoQuadMesh)
        if mesh is None:
            raise ValueError(
                "the edited layout has no usable patch ({} face(s) in, {} degenerate). A patch must be a closed polyline with one point per corner.".format(
                    notes["faces_in"], notes["degenerate"]
                )
            )

        ok, why = densifiable(mesh)
        if not ok and strict:
            raise ValueError(
                "the edited layout will not densify: {}. It is NOT being "
                "replaced by a fallback -- fix the layout and hand it back. "
                "({} patch(es) in, {} out, side counts {})".format(why, notes["faces_in"], notes["faces_out"], notes["sides"])
            )

        note = "EDITED coarse layout adopted: {} patch(es) in, {} out, {} corner(s) snapped to a wall".format(notes["faces_in"], notes["faces_out"], notes["snapped"])
        if notes["off_wall"]:
            note += (
                ", {} corner(s) on the layout edge sit on no wall (further "
                "than the snap tolerance from every loop) -- expected where "
                "a patch was deleted, a mistake otherwise".format(notes["off_wall"])
            )
        if notes["degenerate"]:
            note += ", {} degenerate patch(es) dropped".format(notes["degenerate"])
        if notes["repair"]:
            note += "; repair: {}".format(notes["repair"])
        if not ok:
            note += "; WILL NOT DENSIFY: {}".format(why)
        self.repair_notes.append(note)

        self.mesh = mesh
        self._edited = True
        self._route = "field"
        self.edit_notes = notes
        return self.mesh

    def set_user_curves(self, curves: list[list[Any]]) -> None:
        """Adopt the curves a user drew on the layout, replacing any set before, and add them to ``polylines``."""
        self.user_curves = [[list(point) for point in curve] for curve in curves]
        if self.polylines is not None:
            boundary, others, _ = self._build()
            self.polylines = boundary + others + list(self.user_curves)

    def quad_mesh(
        self,
        target_length: float | None = None,
        density: int | None = None,
        coarse: Any = None,
        densities: dict[Any, int] | None = None,
        strict: bool = False,
    ) -> QuadMesh:
        """An all-quad mesh of the domain, always: via the field layout, else one polygon, else a triangulation.

        ``route`` says which route ran and ``warnings`` why.

        Parameters
        ----------
        target_length : float, optional
            Target quad edge length.
        density : int, optional
            Fixed subdivision per coarse edge instead; takes precedence.
        coarse : mesh or list, optional
            A hand-edited layout, put through ``edit_coarse``. Once edited,
            the edit stays in use.
        densities : dict, optional
            ``{strip key: density}`` over the base pass. A key naming no strip is
            noted, not raised. Strip keys renumber whenever the layout is rebuilt.
        strict : bool, optional
            Re-raise an exception from densification instead of falling back.
            For tests and debugging.

        Returns
        -------
        compas_singular.datastructures.QuadMesh
        """
        self._require_solved()
        keep_preset = target_length is None and density is None
        target_length = target_length or self.background.target_length
        if coarse is not None:
            coarse = self.edit_coarse(coarse)
        elif self._edited:
            coarse = self.mesh
        else:
            coarse = self.decomposition_mesh()

        ok, why = densifiable(coarse)
        if ok:
            try:
                coarse.collect_strips()
                pinned = set()
                for skey, value in (densities or {}).items():
                    if skey in coarse.attributes["strips"]:
                        coarse.set_strip_density(skey, int(value))
                        pinned.add(skey)
                    else:
                        self.repair_notes.append(
                            "DENSITY: strip {!r} is not a strip of this layout; its density was dropped. Strip keys renumber on every rebuild -- key them geometrically.".format(
                                skey
                            )
                        )
                if keep_preset:
                    pinned |= set(coarse.get_strip_densities())
                todo = [skey for skey in coarse.strips() if skey not in pinned]
                if density is not None:
                    coarse.set_strips_density(density, skeys=todo)
                else:
                    coarse.set_strips_density_target(target_length, skeys=todo)
                dense = self.densify(coarse)
                good, why, metrics, note = self._acceptable(dense)
                if note:
                    self.repair_notes.append(note)
                if good:
                    self._note_quality(metrics)
                    self.dense = dense
                    return dense
                why = "densified mesh rejected: {}".format(why)
            except Exception as exc:  # noqa: BLE001 -- the contract is "always a mesh"
                if strict:
                    raise
                self.last_error = traceback.format_exc()
                why = "{}: {} during densification (traceback in last_error)".format(type(exc).__name__, exc)

        # Densifying the backstop is what fails, so it is built AT the element
        # size and returned directly
        mesh = self._triangulation_fallback(why, target_length)
        good, bad_why, metrics, note = self._acceptable(mesh)
        if note:
            self.repair_notes.append(note)
        if not good:
            self.repair_notes.append("WARNING: even the triangulation backstop is imperfect ({}). The mesh is returned anyway -- inspect it before using it.".format(bad_why))
        self._note_quality(metrics)
        self.mesh = mesh
        self._edited = False  # the backstop replaced whatever layout was here
        self._mesh_poles = None  # ... and is not the separatrix layout: rebuild next time
        self.dense = mesh
        return mesh

    def densify(self, coarse: Any = None, **kwargs: Any) -> QuadMesh:
        """**The layout densified**, patch edges along their separatrices and
        patch interiors along the field.

        Parameters
        ----------
        coarse : CoarsePseudoQuadMesh, optional
            Strips collected and densities set. Defaults to ``mesh``.
        **kwargs
            Passed to ``densify.field_densification``.

        Returns
        -------
        QuadMesh
        """
        if coarse is None:
            coarse = self.mesh
        if self.field_aware:
            dense, stats = self.field.densify(coarse, edges_to_curves=self.edges_to_curves(), **kwargs)
        else:
            dense, stats = field_densification(
                coarse, self.field, self.tracer, edges_to_curves=self.edges_to_curves(), field_aware=False, **dict({"spend": bool(self.guides)}, **kwargs)
            )
        self.densify_stats = stats
        if stats.get("guarded") and self.guides:
            # only news on a guided domain, where it means a guide did not reach
            # that patch's interior
            self.repair_notes.append("DENSIFY: {} of {} patch(es) rejected the field-integrated interior and kept the Coons one.".format(stats["guarded"], stats["patches"]))
        return dense

    def route(self) -> str:
        """Which route produced the last layout or mesh: ``'field'``, ``'polygon'`` or ``'triangulation'``."""
        return self._route

    def quality(self, mesh: Any = None, low_angle: float | None = None) -> dict[str, Any]:
        """**The numbers to gate on**: element quality of a mesh, as a dict.

        Parameters
        ----------
        mesh : Mesh, optional
            Defaults to the last mesh ``quad_mesh`` returned.
        low_angle : float, optional
            Threshold for ``share_below``, in degrees.

        Returns
        -------
        dict
            ``quality.mesh_quality``'s metrics, plus ``route``, ``coverage``
            (mesh area over domain area) and ``coarse_faces``.
        """
        if mesh is None:
            mesh = self.dense if self.dense is not None else self.mesh
        if mesh is None:
            raise ValueError("no mesh yet -- call quad_mesh() or decomposition_mesh() first")

        kwargs = {} if low_angle is None else {"low_angle": low_angle}
        out = mesh_quality(mesh, **kwargs)
        out["route"] = self.route()
        out["coverage"] = self._coverage(mesh)
        out["coarse_faces"] = self.mesh.number_of_faces() if self.mesh is not None else None
        return out

    def _coverage(self, mesh: Any) -> float:
        """The mesh's area as a fraction of the domain's. 1.0 is exact."""
        want = abs(signed_area(self.background.outer))
        want -= sum(abs(signed_area(loop)) for loop in self.background.inners)
        if want <= 0:
            return 1.0
        got = 0.0
        for fkey in mesh.faces():
            got += signed_area([mesh.vertex_coordinates(v) for v in mesh.face_vertices(fkey)])
        return abs(got) / want

    def _acceptable(self, mesh: Any) -> tuple[bool, str, dict[str, Any], str | None]:
        """Whether this mesh represents the domain: all-quad, manifold, 97-103% of its area.

        Element quality is measured and returned but never decides.

        Returns
        -------
        (bool, str, dict, str or None)
            The verdict, the first structural reason when false, the
            element-quality metrics (``{}`` when rejected before measuring), and
            a note for ``repair_notes`` or ``None``.
        """
        if not mesh.faces():
            return False, "no faces", {}, None
        face_pole = mesh.attributes.get("face_pole") or {}
        for fkey in mesh.faces():
            n = len(mesh.face_vertices(fkey))
            if n == 4 or (n == 3 and fkey in face_pole):
                continue
            return False, "not all quads (face {} has {} sides)".format(fkey, n), {}, None
        if not mesh.is_manifold():
            return False, "not manifold", {}, None

        metrics = mesh_quality(mesh)

        ratio = self._coverage(mesh)
        if not 0.97 <= ratio <= 1.03:
            if self._edited:
                return (
                    True,
                    "",
                    metrics,
                    (
                        "EDITED layout covers {:.0%} of the domain area. Kept -- the "
                        "area check does not overrule a hand edit. If that was not "
                        "intended, a patch is missing or overlapping.".format(ratio)
                    ),
                )
            return False, "covers {:.0%} of the domain area".format(ratio), metrics, None
        return True, "", metrics, None

    def _note_quality(self, metrics: dict[str, Any]) -> None:
        """Record a hard-floor breach loudly, without changing the route."""
        ok, why = hard_floor(metrics or {})
        if ok:
            return
        self.repair_notes.append(
            "QUALITY FAILURE: {}. The mesh is returned anyway -- the "
            '"always returns a mesh" contract stands -- but this element is '
            "degenerate, not merely poor, and nothing downstream should "
            "assume otherwise.".format(why)
        )

    def _loops(self) -> list[list[list[float]]]:
        """Every boundary loop, outer first."""
        return self.background.loops

    def _fallback_patch(self, boundary: list[list[list[float]]], reason: str) -> Any:
        """Mesh the domain as one polygon, ignoring the separatrices; the
        triangulation backstop if the domain has a hole or the polygon will not
        densify either."""
        if not self.background.inners:
            outer_arcs = [arc for arc in boundary if _on_loop(arc[0], self.background.outer)]
            mesh = self._single_patch(outer_arcs or boundary)
            mesh, _ = solve_non_quad_faces(mesh, CoarsePseudoQuadMesh, self.background.loops)
            ok, why = densifiable(mesh)
            if ok:
                self.repair_notes.append(
                    "FIELD LAYOUT DISCARDED -- not densifiable ({}). Meshed the domain as a single {}-gon instead; the result ignores the field.".format(
                        reason, len(outer_arcs or boundary)
                    )
                )
                self._route = "polygon"
                return mesh
            reason = "{}; single-polygon fallback also failed ({})".format(reason, why)

        return self._triangulation_fallback(reason)

    def _triangulation_fallback(self, reason: str, target_length: float | None = None) -> Any:
        """Last resort: quad-split a coarse triangulation of the domain, never coarser than the background."""
        spacing = (target_length * 2.0) if target_length else (self.background.diagonal / 6.0)
        spacing = min(spacing, self.background.target_length)
        coarse_bg = BackgroundMesh.from_boundary(self.background.outer, self.background.inners, target_length=spacing)

        index = {v: i for i, v in enumerate(coarse_bg.mesh.vertices())}
        vertices = [coarse_bg.mesh.vertex_coordinates(v) for v in coarse_bg.mesh.vertices()]
        faces = [[index[v] for v in coarse_bg.mesh.face_vertices(f)] for f in coarse_bg.mesh.faces()]
        vertices, faces = topological_quad_split(vertices, faces, self.background.loops)
        mesh = CoarsePseudoQuadMesh.from_vertices_and_faces(vertices, faces)
        mesh, _ = solve_non_quad_faces(mesh, CoarsePseudoQuadMesh, self.background.loops)

        ok, why = densifiable(mesh)
        self.repair_notes.append(
            "FIELD LAYOUT DISCARDED -- not densifiable ({}). Fell back to a "
            "quad-split triangulation of the domain: {} patches that cover the "
            "shape but follow the triangulation, not the field.{}".format(reason, mesh.number_of_faces(), "" if ok else " STILL not densifiable: {}".format(why))
        )
        self._route = "triangulation"
        return mesh

    def _single_patch(self, boundary: list[list[list[float]]]) -> CoarsePseudoQuadMesh:
        """One face spanning a domain with no interior lines, any corner count
        -- a hexagon is a good 6-gon that the repair turns into six quads."""
        corners = [list(arc[0]) for arc in boundary]
        if len(corners) < 3:
            raise ValueError("domain has neither interior separatrices nor enough boundary corners to bound a face ({} found)".format(len(corners)))
        if len(corners) != 4:
            self.repair_notes.append("no interior separatrices and {} boundary corners: built as a single {}-gon".format(len(corners), len(corners)))
        return CoarsePseudoQuadMesh.from_vertices_and_faces(corners, [list(range(len(corners)))])

    def _cut_holes(self) -> list[list[list[float]]]:
        """Holes with no corner, which ``Tracer.hole_launches`` cut open."""
        return [loop for loop in self.background.inners if not boundary_corners(loop)]

    def _mesh_from_faces(self, faces: list[list[list[float]]]) -> Any:
        """A coarse mesh from faces given as corner points, welded by position."""
        mesh, _dropped = mesh_from_faces(faces, CoarsePseudoQuadMesh)
        return mesh

    def edges_to_curves(self) -> dict[tuple[int, int], list[list[float]]]:
        """``{(u, v): polyline}`` for densification: exact, then boundary arc, then (edited only) warped, then chord.

        The tally is kept in ``edit_notes['edges']``.
        """
        self._require_solved()
        if self.mesh is None:
            self.decomposition_mesh()

        lookup = {}
        for polyline in self.polylines or []:
            key = (TOL.geometric_key(polyline[0]), TOL.geometric_key(polyline[-1]))
            lookup[key] = polyline
            lookup[(key[1], key[0])] = list(reversed(polyline))

        tally = {"exact": 0, "boundary": 0, "warped": 0, "chord": 0}
        claimed = set()
        # boundary corners only: an interior corner cannot end a boundary edge
        corners = [self.mesh.vertex_coordinates(w) for w in self.mesh.vertices() if self.mesh.is_vertex_on_boundary(w)]
        walls = [(loop, _loop_chord_tolerance(loop)) for loop in self.background.loops]
        out = {}
        for u, v in self.mesh.edges():
            pa = self.mesh.vertex_coordinates(u)
            pb = self.mesh.vertex_coordinates(v)
            curve = lookup.get((TOL.geometric_key(pa), TOL.geometric_key(pb)))
            how = "exact" if curve is not None else None

            if curve is None:
                for loop, wall_tol in walls:
                    if _on_loop(pa, loop, wall_tol) and _on_loop(pb, loop, wall_tol):
                        for arc in _arcs_between(loop, pa, pb):
                            if self._arc_is_one_edge(arc, pa, pb, corners):
                                curve = arc
                                how = "boundary"
                                break
                        break

            if curve is None and self._edited:
                curve = self._warped_curve(pa, pb, claimed)
                how = "warped" if curve is not None else None

            # every edge needs an entry once a mapping is given at all
            if curve is None or distance_point_point(curve[0], pa) > 1e-6 or distance_point_point(curve[-1], pb) > 1e-6:
                curve = [list(pa), list(pb)]
                how = "chord"
            tally[how] += 1
            out[u, v] = curve

        self.edit_notes["edges"] = tally
        return out

    def _arc_is_one_edge(
        self,
        arc: list[list[float]],
        pa: list[float],
        pb: list[float],
        corners: list[list[float]],
    ) -> bool:
        """Whether this wall arc is one layout edge, or runs through another boundary corner."""
        if not arc or len(arc) < 3:
            return bool(arc)

        tol = self.background.target_length * 0.25
        xs = [p[0] for p in arc]
        ys = [p[1] for p in arc]
        lo_x, hi_x = min(xs) - tol, max(xs) + tol
        lo_y, hi_y = min(ys) - tol, max(ys) + tol

        for point in corners:
            if distance_point_point(point, pa) < 1e-6 or distance_point_point(point, pb) < 1e-6:
                continue
            if not (lo_x <= point[0] <= hi_x and lo_y <= point[1] <= hi_y):
                continue
            d, s, total = project_on_polyline(point, arc)
            if d >= tol:
                continue
            margin = min(tol, 0.25 * total)  # never swallowing the arc itself
            if s <= margin or s >= total - margin:
                continue
            return False
        return True

    def _warped_curve(self, pa: list[float], pb: list[float], claimed: set[int]) -> list[list[float]] | None:
        """The traced polyline this edge came from, warped onto its moved endpoints, matched by an anchored end."""
        if not self.polylines:
            return None

        anchor = self.background.target_length * 0.1
        loose = self.background.target_length
        chord = distance_point_point(pa, pb)

        best = None
        for i, polyline in enumerate(self.polylines):
            for points in (polyline, list(reversed(polyline))):
                da = distance_point_point(points[0], pa)
                db = distance_point_point(points[-1], pb)
                if da <= anchor or db <= anchor:
                    tier, free = 0, (db if da <= anchor else da)
                elif da <= loose and db <= loose:
                    tier, free = 1, da + db
                else:
                    continue

                length = sum(distance_point_point(a, b) for a, b in pairwise(points))
                if chord > 1e-9 and not (0.25 * chord <= length <= 4.0 * chord):
                    continue

                score = (i in claimed, tier, free)
                if best is None or score < best[0]:
                    best = (score, i, points)

        if best is None:
            return None
        _score, index, points = best
        curve = warp_polyline(points, pa, pb)
        if curve is None:
            return None
        claimed.add(index)
        return curve

    # ------------------------------------------------------------------

    def _build(self) -> tuple[list[list[list[float]]], list[list[list[float]]], dict[str, Any]]:
        """The polyline network as a planar arrangement, cached: ``(wall arcs,
        separatrices, report)``."""
        self._require_solved()
        if self._network is None:
            boundary, others, info = build_network(self.field, self.separatrices, singularity_points=self.tracer.singularity_positions(), symmetry=self.symmetry)
            info = dict(info)
            boundary, others = planar_arrangement(boundary, others, self.background.target_length * 0.8, info, loops=self.background.loops)
            self._network = (boundary, others, info)
        return self._network

    def report(self) -> dict[str, Any]:
        """The network's and the field's reports, plus face counts of the layout."""
        boundary, others, info = self._build()
        out = dict(info)
        out.update(self.field.report())
        if self.mesh is not None:
            faces = [len(self.mesh.face_vertices(f)) for f in self.mesh.faces()]
            out["faces"] = len(faces)
            out["quads"] = sum(1 for n in faces if n == 4)
            out["non_quads"] = sorted(set(n for n in faces if n != 4))
        return out
