"""The coarse layout (``CoarsePseudoQuadMesh``) in Rhino: patches, corners, edges and strips."""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import struct
from typing import Any
from typing import Sequence

import rhinoscriptsyntax as rs  # type: ignore
import scriptcontext as sc  # type: ignore
from Rhino.Geometry import Point3d  # type: ignore

from compas.colors import Color
from compas_singular.rhino.helpers import bake_polylines
from compas_singular.rhino.mesh_ui import ensure_layer
from compas_singular.rhino.scene.meshobject import RhinoSingularMeshObject

__all__ = ["RhinoCoarseObject"]


# ------------------------------------------------------------------------------
# ribbon geometry -- moved here from mesh_ui with PickableMesh, 2026-09-18
# ------------------------------------------------------------------------------

#: How much of a pseudo-quad face is drawn, toward its own centroid. Big enough
#: to click on, small enough not to swallow the faces beside it.
POLE_FACE_SCALE = 0.6


def _shrunk_face(points: Sequence[Sequence[float]], scale: float) -> list[list[float]]:
    """``points`` pulled toward their own centroid. As many as came in."""
    n = float(len(points))
    cx = sum(p[0] for p in points) / n
    cy = sum(p[1] for p in points) / n
    cz = sum(p[2] for p in points) / n
    return [[cx + (p[0] - cx) * scale, cy + (p[1] - cy) * scale, cz + (p[2] - cz) * scale] for p in points]


def _f32(point: Sequence[float]) -> tuple[float, ...]:
    """``point`` rounded the way a Rhino mesh vertex (``Point3f``) stores it."""
    return struct.unpack("<3f", struct.pack("<3f", point[0], point[1], point[2]))


def _append_face(vertices: list[list[float]], points: Sequence[Sequence[float]]) -> list[list[int]]:
    """Append ``points`` as fresh corners of ``vertices`` and return their faces, collapsed at float32.

    One face with two coincident corners makes Rhino refuse the whole mesh.
    """
    kept = []
    for point in points:
        key = _f32(point)
        if any(key == other for other, _corner in kept):
            continue
        kept.append((key, list(point)))
    if len(kept) < 3:
        return []
    i = len(vertices)
    vertices += [corner for _key, corner in kept]
    if len(kept) == 3:
        return [[i, i + 1, i + 2, i + 2]]
    if len(kept) == 4:
        return [[i, i + 1, i + 2, i + 3]]
    # An n-gon has no business in a quad layout, but keeping only its first four
    # corners is how ``bake_mesh`` used to lose a fifth of the dual. Fan it.
    return [[i, i + k, i + k + 1, i + k + 1] for k in range(1, len(kept) - 1)]


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> list[float]:
    """The point ``t`` of the way from ``a`` to ``b``."""
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]


class _Ribbon(object):
    """One strip's ribbon as one welded mesh, corners shared and merged at float32."""

    def __init__(self, mesh: Any, half: float) -> None:
        self.mesh = mesh
        self.half = half
        self.vertices = []
        self.faces = []
        self._index = {}  # float32 point -> vertex index

    def _add(self, point: Sequence[float]) -> int:
        key = _f32(point)
        if key not in self._index:
            self._index[key] = len(self.vertices)
            self.vertices.append(list(point))
        return self._index[key]

    def rung_point(self, a: int, b: int, t: float) -> int:
        """The point ``t`` of the way from vertex ``a`` to ``b``, as an index."""
        if b < a:
            a, b, t = b, a, 1.0 - t
        return self._add(_lerp(self.mesh.vertex_coordinates(a), self.mesh.vertex_coordinates(b), t))

    def corner(self, vkey: int) -> int:
        """A mesh vertex itself, as an index -- the pole a ribbon tapers into."""
        return self._add(self.mesh.vertex_coordinates(vkey))

    def add_face(self, indices: Sequence[int]) -> None:
        """Add a face; dropped if fewer than three distinct corners are left."""
        kept = []
        for index in indices:
            if index not in kept:
                kept.append(index)
        if len(kept) == 3:
            self.faces.append(kept + [kept[2]])
        elif len(kept) == 4:
            self.faces.append(kept)


# ------------------------------------------------------------------------------
# the layout
# ------------------------------------------------------------------------------


class RhinoCoarseObject(RhinoSingularMeshObject):
    """The coarse layout. Patches are drawn one object each, so a patch can be picked.

    Parameters
    ----------
    show_patterns : bool, optional
        Colour each patch by its dense pattern and write the pattern on it.
    poles_layer, curves_layer, polylines_layer : str, optional
        The permanent display only: layers for the poles, the SHAPE of every
        edge (``edges_to_curves``) and the polylines those shapes come from
        (``shape_polylines``). Each is this object's to clear, like its own
        layer -- except ``curves_layer`` for a layout DRAWN by the user, whose
        curves there are the input ``CMD_read_coarse_mesh`` read the layout from.
    """

    #: One colour per dense pattern; see ``mesh_quad_coarse.patterns.PATTERNS``.
    PATTERN_COLORS = {
        "ortho": Color.from_rgb255(255, 100, 100),
        "diagonal": Color.from_rgb255(50, 255, 50),
        "fan": Color.from_rgb255(50, 50, 255),
    }

    def __init__(self, show_patterns: bool = False, poles_layer: str | None = None, curves_layer: str | None = None, polylines_layer: str | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("disjoint", True)
        super(RhinoCoarseObject, self).__init__(**kwargs)
        self.show_patterns = show_patterns
        self.poles_layer = poles_layer
        self.curves_layer = curves_layer
        self.polylines_layer = polylines_layer

    def _forget(self) -> None:
        super(RhinoCoarseObject, self)._forget()
        self._guid_strip = {}
        self._guid_strip_label = {}

    def _pickable_guids(self) -> list[Any]:
        return super(RhinoCoarseObject, self)._pickable_guids() + list(self._guid_strip)

    def _owned_curves_layer(self) -> str | None:
        """``curves_layer``, unless the curves there are the user's own input."""
        return None if self.mesh.attributes.get("route") == "drawn" else self.curves_layer

    def layers(self) -> list[str]:
        parts = [self.poles_layer, self._owned_curves_layer(), self.polylines_layer]
        return super(RhinoCoarseObject, self).layers() + [layer for layer in parts if layer]

    # --------------------------------------------------------------------------
    # patches, and the rest of the permanent display
    # --------------------------------------------------------------------------

    def draw(self) -> list[Any]:
        previous = rs.EnableRedraw(False)
        try:
            guids = super(RhinoCoarseObject, self).draw()
            if self.poles_layer:
                self.draw_poles()
            if self._owned_curves_layer():
                self._draw_polylines(list(self.mesh.edges_to_curves().values()), self.curves_layer, "edge curve")
            if self.polylines_layer:
                self._draw_polylines(self.mesh.shape_polylines(), self.polylines_layer, "shape polyline")
        finally:
            rs.EnableRedraw(previous)
        sc.doc.Views.Redraw()
        return guids

    def draw_faces(self) -> list[Any]:
        if self.show_patterns:
            for face in self.mesh.faces():
                self.facecolor[face] = self.PATTERN_COLORS.get(self.mesh.get_face_pattern(face), Color.grey())
        guids = super(RhinoCoarseObject, self).draw_faces()
        if self.show_patterns:
            self.draw_facelabels(text={face: self.mesh.get_face_pattern(face) for face in self._guid_face.values()}, color=Color.black())
        return guids

    def draw_poles(self) -> None:
        """The poles as points on ``poles_layer``."""
        ensure_layer(self.poles_layer)
        poles = self.mesh.poles() if hasattr(self.mesh, "poles") else []
        for vkey in poles:
            guid = rs.AddPoint(Point3d(*self.mesh.vertex_coordinates(vkey)))
            if guid:
                rs.ObjectLayer(guid, self.poles_layer)
                self._guids.append(guid)

    def _draw_polylines(self, polylines: Sequence[Sequence[Sequence[float]]], layer: str, what: str) -> None:
        """Through ``helpers.bake_polylines``, which cleans each one and says what it skipped."""
        ensure_layer(layer)
        guids, skipped = bake_polylines(polylines, layer, clear_existing=False)
        self._guids += guids
        if skipped:
            print("  {} {}(s) Rhino refused on {!r} -- those edges densify as chords".format(skipped, what, layer))

    # --------------------------------------------------------------------------
    # strips
    # --------------------------------------------------------------------------

    def draw_strips(
        self,
        strips: Sequence[int] | None = None,
        width: float = 0.5,
        highlight: Sequence[int] = (),
        collect: bool = True,
        strip_colors: dict[int, tuple[int, int, int]] | None = None,
        labels: dict[int, Any] | None = None,
    ) -> dict[Any, int]:
        """Draw strips as pickable ribbons down their middles, replacing any drawn before. ``{guid: skey}``.

        Parameters
        ----------
        strips : iterable, optional
            Which strips. Default: all of them.
        width : float, optional
            The ribbon width as a fraction of the rung it spans.
        highlight : iterable, optional
            Strips drawn in the highlight colour.
        collect : bool, optional
            Re-collect the strips on a copy first -- right for an editor whose
            mesh just changed. ``False`` draws the strips the mesh CARRIES, under
            their keys: a caller holding data keyed by strip (``strips_density``)
            needs the picked key to be its key, and a re-collection is free to
            number the same bands differently.
        strip_colors : dict, optional
            ``{skey: (r, g, b)}``. A highlighted strip still takes the highlight colour.
        labels : dict, optional
            ``{skey: text}``, a text dot on the middle rung of each strip. A rung
            belongs to one strip only, so unlike a face centre it is never shared
            with the crossing band's label.
        """
        self.clear_strips()
        ensure_layer(self.layer)
        mesh = self.mesh
        if collect or not mesh.attributes.get("strips"):
            work = mesh.copy()
            work.collect_strips()
        else:
            work = mesh  # read only below
        keys = list(work.attributes["strips"]) if strips is None else list(strips)
        highlight = set(highlight)
        strip_colors = strip_colors or {}
        labels = labels or {}
        half = max(0.05, min(0.49, float(width) / 2.0))
        missing = []

        previous = rs.EnableRedraw(False)
        try:
            for skey in keys:
                if skey not in work.attributes["strips"]:
                    continue
                vertices, faces = self._ribbon(work, skey, half)
                if not faces:
                    missing.append(skey)
                    continue
                # ``rs.AddMesh`` is safe here: every face built has exactly four
                # indices. It RAISES on refusal, and a strip that goes missing is
                # one the user can see no way to select, so the guard says so.
                try:
                    guid = rs.AddMesh(vertices, faces)
                except Exception as error:  # noqa: BLE001
                    print("strip {}: Rhino refused the band -- {}".format(skey, error))
                    guid = None
                if not guid:
                    missing.append(skey)
                    continue
                rs.ObjectLayer(guid, self.layer)
                if skey in highlight:
                    rs.ObjectColor(guid, self.colors["strip.highlight"])
                else:
                    rs.ObjectColor(guid, strip_colors.get(skey, self.colors["strip"]))
                self._guid_strip[guid] = skey
                if skey in labels:
                    self._label_strip(work, skey, labels[skey])
        finally:
            rs.EnableRedraw(previous)
        sc.doc.Views.Redraw()
        if missing:
            print("{} of {} strips could not be drawn, and cannot be picked: {}.".format(len(missing), len(keys), ", ".join(str(skey) for skey in missing)))
        return dict(self._guid_strip)

    def _ribbon(self, work: Any, skey: int, half: float) -> tuple[list[list[float]], list[list[int]]]:
        """``(vertices, faces)`` of one strip's ribbon, or of its faces shrunk."""
        rungs = set(frozenset(edge) for edge in work.strip_edges(skey))
        ribbon = _Ribbon(work, half)
        for fkey in work.strip_faces(skey):
            here = [(u, v) for u, v in work.face_halfedges(fkey) if u != v and frozenset((u, v)) in rungs]
            opposite = work.face_opposite_edge(*here[0]) if here else None
            if opposite is None:
                continue
            u, v = here[0]
            w, x = opposite
            if w == x:
                # The far side of a pseudo-quad collapsed to its pole: the ribbon
                # tapers into the pole instead of stopping.
                ribbon.add_face([ribbon.rung_point(u, v, 0.5 - half), ribbon.rung_point(u, v, 0.5 + half), ribbon.corner(w)])
            else:
                # (u, v, w, x) go round the face: u pairs with x and v with w.
                ribbon.add_face(
                    [ribbon.rung_point(u, v, 0.5 - half), ribbon.rung_point(u, v, 0.5 + half), ribbon.rung_point(x, w, 0.5 + half), ribbon.rung_point(x, w, 0.5 - half)]
                )
        vertices, faces = ribbon.vertices, ribbon.faces
        if not faces:
            for fkey in work.strip_faces(skey):
                faces += _append_face(vertices, _shrunk_face([work.vertex_coordinates(c) for c in work.face_vertices(fkey)], POLE_FACE_SCALE))
        return vertices, faces

    def _label_strip(self, work: Any, skey: int, text: Any) -> None:
        rungs = [(u, v) for u, v in work.strip_edges(skey) if u != v]
        if not rungs:
            return
        u, v = rungs[len(rungs) // 2]
        dot = rs.AddTextDot(str(text), Point3d(*_lerp(work.vertex_coordinates(u), work.vertex_coordinates(v), 0.5)))
        if dot:
            rs.ObjectLayer(dot, self.layer)
            self._guid_strip_label[dot] = skey

    def clear_strips(self) -> None:
        """Just the ribbons and their labels; the rest of the drawing stays."""
        guids = [guid for guid in list(self._guid_strip) + list(self._guid_strip_label) if rs.IsObject(guid)]
        if guids:
            rs.DeleteObjects(guids)
        self._guid_strip = {}
        self._guid_strip_label = {}

    def pick_strip(self, message: str = "Select a strip", preselect: bool = True) -> int | None:
        """A strip key, or ``None`` on Esc. A miss re-prompts."""
        while True:
            guid = rs.GetObject(message, rs.filter.mesh, preselect=preselect)
            if not guid:
                return None
            skey = self._guid_strip.get(guid)
            if skey is not None:
                return skey
            print("Not a strip of this layout -- pick one of the bands on {!r}.".format(self.layer))
