"""``SymmetricUnit``: a coarse layout of one fundamental region, with its group and seams as plain data.

``SymmetricUnit.check`` requires seam junctions at corners and matching rotation seams.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import json
from copy import deepcopy
from math import ceil
from math import radians
from typing import Any
from typing import Callable
from typing import Sequence

from compas.geometry import vector_average
from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.datastructures import PseudoQuadMesh
from compas_singular.datastructures.mesh_quad_coarse.mesh_quad_coarse import CoarseQuadMesh
from compas_singular.symmetry._geometry import point_in_polygon
from compas_singular.symmetry.cut import Seam
from compas_singular.symmetry.cut import cut_unit
from compas_singular.symmetry.cut import subgroups_avoiding_poles
from compas_singular.symmetry.group import SymmetryGroup
from compas_singular.symmetry.replicate import expand
from compas_singular.symmetry.replicate import rotation_partners
from compas_singular.symmetry.replicate import seam_membership
from compas_singular.symmetry.report import SymmetryReport

__all__ = ["SymmetricUnit", "SymmetricQuadUnit", "build_unit"]


def _symmetry(self) -> dict[str, Any]:
    return self.attributes.setdefault("symmetry", {})


def _group(self) -> SymmetryGroup:
    return SymmetryGroup.from_data(self.symmetry["group"])


def _seams(self) -> list[Seam]:
    centre = self.symmetry["group"]["centre"] + [0.0]
    return [Seam(s["name"], radians(s["angle"]), s["kind"], s["element"], centre) for s in self.symmetry.get("seams", [])]


def _eps(self) -> float:
    return self.symmetry.get("eps", 1e-7)


def _expand(self, cls: type, curves: dict[tuple[int, int], Any] | None = None) -> Any:
    return expand(self, self.group, self.seams, self.eps, cls, match_tol=self.symmetry.get("match_tol"), curves=curves)


# Shared by the coarse and the dense unit. Assigned into each class rather than
# inherited from a mixin: compas walks a data class's MRO to serialise it and
# requires every base to be a compas ``Data`` class, so a plain mixin breaks
# ``json_dump``.
_SHARED = dict(
    symmetry=property(_symmetry, doc="The unit's plain-data symmetry record."),
    group=property(_group, doc="The enforced ``SymmetryGroup``."),
    seams=property(_seams, doc="The seams, as ``.cut.Seam`` objects."),
    eps=property(_eps, doc="On-seam tolerance."),
    _expand=_expand,
)


class SymmetricUnit(CoarsePseudoQuadMesh):
    """A coarse quad layout of one symmetric unit.

    Build one with ``decomposition.symmetry_unit(...)`` or ``build_unit``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(SymmetricUnit, self).__init__(*args, **kwargs)
        self.attributes.setdefault("symmetry", {})
        #: The unit's own cross field on the field route, else ``None``. Used by
        #: ``quad_mesh`` unless another is passed. Not serialised.
        self.field = None
        #: The decomposition that meshed the unit. Not serialised.
        self.decomposition = None

    def __repr__(self) -> str:
        data = self.symmetry
        return "<SymmetricUnit {} of {}: {} patches, {} vertices, route {}>".format(
            data.get("group", {}).get("name", "?"), data.get("detected", "?"), self.number_of_faces(), self.number_of_vertices(), data.get("route")
        )

    __str__ = __repr__

    @classmethod
    def from_coarse(
        cls,
        coarse: CoarsePseudoQuadMesh,
        data: dict[str, Any] | None = None,
        field: Any = None,
        decomposition: Any = None,
    ) -> SymmetricUnit:
        """Wrap a coarse layout as a unit. ``data`` defaults to the layout's own
        ``attributes['symmetry']`` -- for re-wrapping a copy that lost its class."""
        unit = coarse.copy(cls=cls)
        if data is not None:
            unit.attributes["symmetry"] = deepcopy(data)
        if not unit.attributes.get("symmetry"):
            raise ValueError("no symmetry data to build a unit from")
        unit.field = field
        unit.decomposition = decomposition
        return unit

    # ------------------------------------------------------------------
    # invariants
    # ------------------------------------------------------------------

    def seam_partners(self) -> tuple[dict[int, int], list[tuple[float, int]], list[tuple[float, int]]]:
        """Rotation seams only: ``(partners, unmatched on A, unmatched on B)``."""
        seams = self.seams
        if not seams or seams[0].kind != "rotation":
            return {}, [], []
        membership = seam_membership(self, seams, self.eps)
        return rotation_partners(membership, self.symmetry.get("match_tol") or 1e3 * self.eps)

    def check(self) -> tuple[bool, list[str]]:
        """``(ok, notes)`` -- whether this unit can be densified and expanded.

        ``notes`` names every problem found; an empty list means none.
        """
        notes = []
        face_pole = self.attributes.get("face_pole") or {}
        bad = [f for f in self.faces() if not (len(self.face_vertices(f)) == 4 or (len(self.face_vertices(f)) == 3 and f in face_pole))]
        if bad:
            notes.append("{} face(s) are not quads or pseudo-quads: {}".format(len(bad), bad[:5]))
        try:
            if not self.is_manifold():
                notes.append("the unit layout is not manifold")
        except Exception as exc:  # compas raises on some degenerate meshes
            notes.append("manifold check failed: {}".format(exc))

        tol = max(10 * self.eps, 1e-9)
        positions = [self.vertex_coordinates(v) for v in self.vertices()]
        missing = []
        for p in self.symmetry.get("junctions", []):
            if not any(abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol for q in positions):
                missing.append([round(p[0], 4), round(p[1], 4)])
        if missing:
            notes.append("seam junction(s) without a corner -- the seam ends inside a coarse edge and cannot be glued: {}".format(missing[:5]))

        seams = self.seams
        gaps = self.seam_gaps()
        if gaps:
            notes.append("the layout does not follow its seams -- a boundary edge cuts across where the copies must meet: {}".format(gaps[:4]))
        if seams and seams[0].kind == "rotation":
            _, ua, ub = self.seam_partners()
            if ua or ub:
                notes.append("rotation seams unmatched: {} corner(s) on A and {} on B have no partner".format(len(ua), len(ub)))
            if self.has_densities():
                for group in self.glued_strips():
                    values = set(self.get_strip_density(k) for k in group)
                    if len(values) > 1:
                        notes.append("strips {} meet across a rotation seam but have densities {}".format(sorted(group), sorted(values)))
        return not notes, notes

    def seam_gaps(self) -> list[list[Any]]:
        """Stretches of seam the layout's boundary does NOT run along, as
        ``[seam name, t_lo, t_hi]``. Empty when every seam is fully covered."""
        runs = self.symmetry.get("seam_runs") or {}
        if not runs:
            return []
        eps = self.eps
        seams = dict((s.name, s) for s in self.seams)
        boundary = set()
        for loop in self.vertices_on_boundaries():
            for i in range(len(loop) - 1):
                boundary.add((loop[i], loop[i + 1]))
        gaps = []
        for name, intervals in runs.items():
            s = seams[name]
            dx, dy = s.direction
            cx, cy = s.centre[0], s.centre[1]
            covered = []
            for u, v in boundary:
                # By LINE, not by ray: in a half-plane unit one edge can run from
                # one seam straight through the centre onto the other.
                pu, pv = self.vertex_coordinates(u), self.vertex_coordinates(v)
                if max(abs(dx * (p[1] - cy) - dy * (p[0] - cx)) for p in (pu, pv)) > eps:
                    continue
                tu = dx * (pu[0] - cx) + dy * (pu[1] - cy)
                tv = dx * (pv[0] - cx) + dy * (pv[1] - cy)
                lo, hi = max(0.0, min(tu, tv)), max(tu, tv)
                if hi > lo:
                    covered.append((lo, hi))
            covered.sort()
            merged = []
            for lo, hi in covered:
                if merged and lo <= merged[-1][1] + eps:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            for lo, hi in intervals:
                t = lo
                for clo, chi in merged:
                    if chi < t - eps:
                        continue
                    if clo > t + eps:
                        break
                    t = max(t, chi)
                if t < hi - 10 * eps:
                    gaps.append([name, round(t, 6), round(hi, 6)])
        return gaps

    def snap_to_seams(self, fraction: float = 0.25) -> int:
        """Project boundary corners lying near a seam onto it, in place. Returns the number moved."""
        from compas_singular.symmetry.matching import _move

        runs = self.symmetry.get("seam_runs") or {}
        seams = dict((s.name, s) for s in self.seams)
        boundary = set(v for loop in self.vertices_on_boundaries() for v in loop)
        moved = 0
        for v in boundary:
            p = self.vertex_coordinates(v)
            if self.symmetry and any(s.locate(p, self.eps) is not None for s in seams.values()):
                continue
            lengths = [self.edge_length(v, w) for w in self.vertex_neighbors(v)]
            reach = fraction * (min(lengths) if lengths else 0.0)
            best = None
            for name, s in seams.items():
                dx, dy = s.direction
                px, py = p[0] - s.centre[0], p[1] - s.centre[1]
                off = abs(dx * py - dy * px)
                t = dx * px + dy * py
                if off > reach or not any(lo - self.eps <= t <= hi + self.eps for lo, hi in runs.get(name, [])):
                    continue
                if best is None or off < best[0]:
                    best = (off, s, t)
            if best is not None:
                _move(self, v, best[1].point_at(best[2]))
                moved += 1
        return moved

    def _require_ok(self, action: str) -> None:
        ok, notes = self.check()
        if not ok:
            raise ValueError("cannot {} this symmetric unit: {}".format(action, "; ".join(notes)))

    # ------------------------------------------------------------------
    # strips glued across rotation seams
    # ------------------------------------------------------------------

    def glued_strips(self) -> list[list[int]]:
        """Groups of strip keys that are one strip of the global mesh across rotation seams."""
        strips = self.attributes.get("strips") or {}
        parent = dict((k, k) for k in strips)

        def find(k: int) -> int:
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        partners, _, _ = self.seam_partners()
        if partners:
            edge_strip = {}
            for skey, edges in strips.items():
                for u, v in edges:
                    edge_strip[frozenset((u, v))] = skey
            for (u, v), skey in [((u, v), s) for s, edges in strips.items() for u, v in edges]:
                if u in partners and v in partners:
                    other = edge_strip.get(frozenset((partners[u], partners[v])))
                    if other is not None:
                        ra, rb = find(skey), find(other)
                        if ra != rb:
                            parent[rb] = ra
        groups = {}
        for k in strips:
            groups.setdefault(find(k), []).append(k)
        return list(groups.values())

    def _glued(self, skey: int) -> list[int]:
        for group in self.glued_strips():
            if skey in group:
                return group
        return [skey]

    def set_strip_density(self, skey: int, d: int) -> None:
        for k in self._glued(skey):
            CoarseQuadMesh.set_strip_density(self, k, d)

    def set_strip_density_target(self, skey: int, t: float) -> None:
        group = self._glued(skey)
        d = 1
        for k in group:
            lengths = [self.edge_length(u, v) for u, v in self.strip_edges(k) if u != v]
            if lengths:
                d = max(d, int(ceil(vector_average(lengths) / t)))
        for k in group:
            CoarseQuadMesh.set_strip_density(self, k, d)

    def set_strips_density_target(self, t: float, skeys: Sequence[int] | None = None) -> None:
        done = set()
        for skey in self.strips() if skeys is None else skeys:
            if skey in done:
                continue
            group = self._glued(skey)
            done.update(group)
            self.set_strip_density_target(skey, t)

    # ------------------------------------------------------------------
    # densify and expand
    # ------------------------------------------------------------------

    def fingerprint(self) -> str:
        """What a dense unit was built from: densities, patterns and the corners."""
        densities = sorted((str(k), v) for k, v in (self.attributes.get("strips_density") or {}).items())
        patterns = sorted((str(k), v) for k, v in (self.attributes.get("dense_pattern") or {}).items())
        corners = sorted((round(p[0], 9), round(p[1], 9)) for p in (self.vertex_coordinates(v) for v in self.vertices()))
        return json.dumps([densities, patterns, corners])

    def quad_mesh(self, *args: Any, **kwargs: Any) -> SymmetricQuadUnit:
        """Densify the unit into a ``SymmetricQuadUnit``, with the unit's own field by default."""
        self._require_ok("densify")
        kwargs.setdefault("field", self.field)
        if kwargs["field"] is not None and not args:
            # A field integrates the interiors itself and is only valid with the
            # plain grid; say so rather than make every caller spell it out.
            kwargs.setdefault("pattern_overwrite", "ortho")
        dense = super(SymmetricUnit, self).quad_mesh(*args, **kwargs)
        return self._wrap_dense(dense)

    def densification(self, *args: Any, **kwargs: Any) -> SymmetricQuadUnit:
        """As ``CoarsePseudoQuadMesh.densification``; returns a ``SymmetricQuadUnit``."""
        self._require_ok("densify")
        kwargs.setdefault("field", self.field)
        dense = super(SymmetricUnit, self).densification(*args, **kwargs)
        return self._wrap_dense(dense)

    def _wrap_dense(self, dense: PseudoQuadMesh) -> SymmetricQuadUnit:
        quad_unit = dense.copy(cls=SymmetricQuadUnit)
        data = deepcopy(self.symmetry)
        data["fingerprint"] = self.fingerprint()
        quad_unit.attributes["symmetry"] = data
        return quad_unit

    def expand_symmetrically(self) -> CoarsePseudoQuadMesh:
        """The global COARSE layout: every copy of the unit, welded along the seams.

        Returns a plain ``CoarsePseudoQuadMesh`` carrying the edge shapes and
        ``attributes['orbits']``.
        """
        self._require_ok("expand")
        out = self._expand(CoarsePseudoQuadMesh, curves=self.edges_to_curves())
        out.attributes["symmetry_source"] = {"group": self.symmetry["group"], "route": self.symmetry.get("route")}
        return out


class SymmetricQuadUnit(PseudoQuadMesh):
    """The dense mesh of a symmetric unit. Made by ``SymmetricUnit.quad_mesh``."""

    def __repr__(self) -> str:
        return "<SymmetricQuadUnit {}: {} faces, {} vertices>".format(self.symmetry.get("group", {}).get("name", "?"), self.number_of_faces(), self.number_of_vertices())

    __str__ = __repr__

    def expand_symmetrically(self, unit: SymmetricUnit | None = None) -> PseudoQuadMesh:
        """The global quad mesh.

        Parameters
        ----------
        unit : SymmetricUnit, optional
            The coarse unit this was densified from. When given, it is checked
            against what the dense mesh was built from, and a mismatch raises --
            a density changed after ``quad_mesh()`` would weld seams with
            different vertex counts on either side.

        Returns
        -------
        PseudoQuadMesh
            With ``attributes['orbits']``.
        """
        if unit is not None and unit.fingerprint() != self.symmetry.get("fingerprint"):
            raise ValueError("the unit changed since quad_mesh() was called (densities, patterns or corners) -- densify it again before expanding")
        return self._expand(PseudoQuadMesh)


for _name, _member in _SHARED.items():
    setattr(SymmetricUnit, _name, _member)
    setattr(SymmetricQuadUnit, _name, _member)


# ----------------------------------------------------------------------
# building a unit
# ----------------------------------------------------------------------


def _inside(point: Sequence[float], outer: Sequence[Sequence[float]], holes: Sequence[Sequence[Sequence[float]]]) -> bool:
    return point_in_polygon(point[0], point[1], outer) and not any(point_in_polygon(point[0], point[1], h) for h in holes)


def _join(parts: Sequence[CoarsePseudoQuadMesh]) -> CoarsePseudoQuadMesh:
    """Several unit components' coarse meshes as one layout (they share no vertices)."""
    vertices, faces, face_poles, curves = [], [], {}, {}
    for coarse in parts:
        offset = {}
        for v in coarse.vertices():
            offset[v] = len(vertices)
            vertices.append(coarse.vertex_coordinates(v))
        fp = coarse.attributes.get("face_pole") or {}
        for f in coarse.faces():
            faces.append([offset[v] for v in coarse.face_vertices(f)])
            if f in fp:
                face_poles[len(faces) - 1] = offset[fp[f]]
        for (u, v), pts in coarse.edges_to_curves().items():
            curves[(offset[u], offset[v])] = pts
    mesh = CoarsePseudoQuadMesh.from_vertices_and_faces_with_face_poles(vertices, faces, face_poles)
    mesh.set_edges_to_curves(curves)
    return mesh


def build_unit(
    report: SymmetryReport,
    mesher: Callable[..., Any],
    keys: str | Sequence[str] | None = None,
    centre: str = "route",
    seam: float | None = None,
    route: str | None = None,
) -> SymmetricUnit:
    """Cut the unit for ``keys``, mesh it with ``mesher``, return a ``SymmetricUnit``.

    Parameters
    ----------
    report : SymmetryReport
    mesher : callable
        ``mesher(outer, holes, guides, poles) -> (coarse, extras)``; see
        ``.routes``.
    keys : list[str], optional
        Symmetries to enforce, by key. Closed under composition. ``None`` is the
        whole detected group.
    centre, seam
        See ``.cut.cut_unit``.
    """
    if isinstance(keys, str):
        keys = [keys]
    group = report.group.subgroup(keys)
    if not group.mirrors and group.n > 1 and seam is None:
        # A rotation seam is free, and whether the route's corners on its two
        # sides can be matched depends on where it runs. Try the best few.
        from compas_singular.symmetry.cut import rotation_seam_candidates

        errors = []
        candidates = rotation_seam_candidates(report.domain, group, 1e-7 * (report.domain.diagonal or 1.0), count=4)
        for angle in candidates:
            try:
                unit = build_unit(report, mesher, keys=keys, centre=centre, seam=angle, route=route)
            except ValueError as exc:
                errors.append("seam at {:.1f} deg: {}".format(angle * 180.0 / 3.141592653589793, exc))
                continue
            if errors:
                unit.symmetry["notes"].append("{} seam angle(s) tried before this one: {}".format(len(errors), " | ".join(errors)))
            return unit
        raise ValueError("no seam angle gave a matchable unit for {}: {}".format(group.name, " | ".join(errors)))
    unit_domain = cut_unit(report.domain, group, centre=centre, seam=seam)

    if unit_domain.poles_on_seams:
        options = subgroups_avoiding_poles(report.domain, group)
        hint = "keys {} ({})".format(" ".join(options[0].keys()), options[0].name) if options else "no symmetry at all"
        raise ValueError(
            "{} pole(s) lie exactly on a seam of {} ({}), and a pole cannot be glued across a seam. Choose symmetries whose seams avoid them -- the largest is {}.".format(
                len(unit_domain.poles_on_seams), group.name, ", ".join("({:.3f}, {:.3f})".format(p[0], p[1]) for p in unit_domain.poles_on_seams), hint
            )
        )

    parts, extras = [], []
    for outer, holes in unit_domain.components:
        guides = [g for g in unit_domain.guides if _inside([0.5 * (g[0][0] + g[1][0]), 0.5 * (g[0][1] + g[1][1])], outer, holes)]
        poles = [p for p in unit_domain.poles if _inside(p, outer, holes)]
        coarse, extra = mesher(outer, holes, guides, poles)
        parts.append(coarse)
        extras.append(extra)
    coarse = parts[0] if len(parts) == 1 else _join(parts)

    diagonal = report.domain.diagonal or 1.0
    data = {
        "version": 1,
        "group": group.to_data(),
        "detected": report.group.name,
        "keys": group.keys(),
        "route": route or getattr(mesher, "route", None),
        "centre": centre,
        "seams": [s.to_data() for s in unit_domain.seams],
        "wedge": unit_domain.wedge * 180.0 / 3.141592653589793,
        "eps": max(unit_domain.eps, 1e-9 * diagonal) * 100.0,
        "match_tol": 1e-6 * diagonal,
        "junctions": unit_domain.junctions(),
        "seam_runs": unit_domain.seam_runs(),
        "components": len(unit_domain.components),
        "matching_cuts": 0,
        "notes": [],
    }
    field = extras[0].get("field") if len(parts) == 1 else None
    if len(parts) > 1 and any(e.get("field") for e in extras):
        data["notes"].append("the unit has {} separate pieces: their fields cannot be combined, so patch interiors are not integrated from a field".format(len(parts)))
    unit = SymmetricUnit.from_coarse(coarse, data, field=field, decomposition=extras[0].get("decomposition") if len(parts) == 1 else None)
    snapped = unit.snap_to_seams() if unit_domain.seams else 0
    if snapped:
        unit.symmetry["notes"].append("{} corner(s) the route placed just off a seam were moved onto it".format(snapped))
    if unit_domain.seams and unit_domain.seams[0].kind == "rotation":
        from compas_singular.symmetry.matching import match_rotation_seams

        match_rotation_seams(unit)
    return unit
