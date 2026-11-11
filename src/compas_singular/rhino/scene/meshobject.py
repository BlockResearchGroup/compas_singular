"""Drawing and picking shared by the coarse layout and dense mesh scene objects in Rhino.

Each object owns one layer; cleared objects are deleted, never purged, so Ctrl+Z still works.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any
from typing import Callable
from typing import Iterable
from typing import Sequence

import compas_rhino.layers
import Rhino  # type: ignore
import rhinoscriptsyntax as rs  # type: ignore
import scriptcontext as sc  # type: ignore
import System  # type: ignore
from compas_rhino.conversions import vertices_and_faces_to_rhino
from compas_rui.scene import RUIMeshObject
from Rhino.Geometry import Point3d  # type: ignore

from compas_singular.rhino.mesh_ui import DEFAULT_COLORS
from compas_singular.rhino.mesh_ui import FINISH
from compas_singular.rhino.mesh_ui import ensure_layer
from compas_singular.rhino.mesh_ui import relock
from compas_singular.rhino.mesh_ui import unlock

__all__ = ["RhinoSingularMeshObject"]


def _selection(flag: bool | Sequence[Any] | None, everything: Callable[[], Iterable[Any]]) -> list[Any]:
    """compas's ``show_*`` convention: ``True`` is all of them, a list is those."""
    return list(everything()) if flag is True else list(flag or [])


def _boundary_vertices(mesh: Any) -> set[int]:
    """Every vertex with a faceless halfedge, as a set, without walking a loop."""
    out = set()
    for u, nbrs in mesh.halfedge.items():
        for v, fkey in nbrs.items():
            if fkey is None:
                out.add(u)
                out.add(v)
    return out


class RhinoSingularMeshObject(RUIMeshObject):
    """A mesh drawn on its own layer as objects the user can click, by key.

    Parameters
    ----------
    special : iterable, optional
        Vertices drawn in the ``'vertex.special'`` colour: poles, singularities,
        whatever the user is trying to reach or to avoid.
    edge_shape : callable, optional
        ``(u, v) -> points or None``. An edge that carries a drawn SHAPE is drawn
        as that shape, not as its chord, or a cut that worked looks like one that
        snapped straight. It is still one object and picked the same way.
    colors : dict, optional
        Overrides for ``mesh_ui.DEFAULT_COLORS``.
    joined : bool, optional
        Draw the faces as ONE welded mesh, n-gons kept, instead of one object
        per face. The permanent display; nothing on it is picked.
    """

    def __init__(
        self,
        special: Sequence[int] = (),
        edge_shape: Callable[[int, int], Sequence[Sequence[float]] | None] | None = None,
        colors: dict[str, tuple[int, int, int]] | None = None,
        joined: bool = False,
        **kwargs: Any,
    ) -> None:
        super(RhinoSingularMeshObject, self).__init__(**kwargs)
        self.special = special
        self.edge_shape = edge_shape
        self.colors = dict(DEFAULT_COLORS, **(colors or {}))
        self.joined = joined
        self._forget()

    def _forget(self) -> None:
        """Empty every guid map. Touches nothing in the document."""
        self._guids = []
        self._guid_mesh = None
        self._guid_vertex = {}
        self._guid_edge = {}
        self._guid_face = {}
        self._guid_path = []
        self._path_colors = {}  # guid -> its colour before show_path
        # What is on screen, for ``sync``: vkey -> (guid, xyz, colour key), and
        # frozenset edge -> (guid, (xyz, xyz)), ends None for a SHAPED edge.
        self._drawn_vertices = {}
        self._drawn_edges = {}

    # --------------------------------------------------------------------------
    # clear and draw
    # --------------------------------------------------------------------------

    def layers(self) -> list[str]:
        """Every layer this object draws on. Only it draws there."""
        return [self.layer] if self.layer else []

    def clear(self) -> None:
        """Everything on this object's layers, and every guid map with it.

        Not their sublayers, which belong to other commands.
        """
        for layer in self.layers():
            compas_rhino.layers.clear_layer(layer, include_children=False, purge=False)
        self._forget()

    def redraw_faces(self) -> None:
        """The whole object: compas's ``clear_faces`` would leave face labels behind."""
        self.redraw()

    def draw(self) -> list[Any]:
        """Faces if shown, then corners and edges. Everything, every time."""
        ensure_layer(self.layer)
        self._guids = []
        # Restore, never force True: EnableRedraw is a flag, not a counter, so a
        # caller that turned redraw off would get it back on halfway through.
        previous = rs.EnableRedraw(False)
        try:
            if self.show_faces is True and self.joined:
                self.draw_joined_faces()
            elif self.show_faces:
                self.draw_faces()
            self.draw_vertices()
            self.draw_edges()
        finally:
            rs.EnableRedraw(previous)
        sc.doc.Views.Redraw()
        return self.guids

    def draw_joined_faces(self) -> Any | None:
        """The faces as one welded mesh with n-gons kept, as ``helpers.bake_mesh`` bakes it."""
        index = {vkey: i for i, vkey in enumerate(self.mesh.vertices())}
        vertices = [self.mesh.vertex_attributes(vkey, "xyz") for vkey in self.mesh.vertices()]
        faces = [[index[vkey] for vkey in self.mesh.face_vertices(fkey)] for fkey in self.mesh.faces()]
        geometry = vertices_and_faces_to_rhino(vertices, faces, disjoint=False)
        guid = sc.doc.Objects.AddMesh(geometry, self.compile_attributes())
        if guid == System.Guid.Empty:
            print("Rhino refused the mesh on {!r} -- it is not drawn.".format(self.layer))
            return None
        self._guid_mesh = guid
        self._guids.append(guid)
        return guid

    def draw_faces(self) -> list[Any]:
        """compas's faces, one object each -- plus a report of any Rhino refused."""
        faces = _selection(self.show_faces, self.mesh.faces)
        guids = super(RhinoSingularMeshObject, self).draw_faces()
        refused = [face for guid, face in zip(guids, faces) if guid == System.Guid.Empty]
        if refused:
            self._guid_face = {guid: face for guid, face in zip(guids, faces) if guid != System.Guid.Empty}
            print("{} of {} face(s) could not be drawn, and cannot be picked: {}".format(len(refused), len(faces), ", ".join(str(face) for face in refused)))
        return guids

    def draw_vertices(self) -> list[Any]:
        vertices = _selection(self.show_vertices, self.mesh.vertices)
        if not vertices:
            return []
        boundary = _boundary_vertices(self.mesh)
        special = set(self.special)
        guids = [self._add_vertex(vkey, self._vertex_color_key(vkey, boundary, special)) for vkey in vertices]
        return [guid for guid in guids if guid]

    def draw_edges(self) -> list[Any]:
        edges = _selection(self.show_edges, self.mesh.edges)
        guids, chorded, lost = [], [], []
        for u, v in edges:
            guid = self._add_shaped_edge(u, v, chorded) or self._add_edge(u, v)
            if guid:
                guids.append(guid)
            else:
                lost.append((u, v))
        if chorded:
            print("{} edge(s) drawn as a chord -- Rhino refused the curve.".format(len(chorded)))
        if lost:
            print("{} edge(s) could not be drawn, and cannot be picked: {}.".format(len(lost), ", ".join("{}-{}".format(*edge) for edge in lost)))
        return guids

    def sync(self) -> tuple[int, int]:
        """Bring the drawing up to date, replacing only objects whose key or geometry changed. ``(removed, added)``."""
        if not self._drawn_vertices and not self._drawn_edges:
            self.redraw()
            return 0, len(self._guid_vertex) + len(self._guid_edge)
        ensure_layer(self.layer)
        mesh = self.mesh
        boundary = _boundary_vertices(mesh)
        special = set(self.special)

        stale = []
        wanted_vertices = {vkey: (tuple(mesh.vertex_coordinates(vkey)), self._vertex_color_key(vkey, boundary, special)) for vkey in mesh.vertices()}
        for vkey, (guid, xyz, color_key) in list(self._drawn_vertices.items()):
            if wanted_vertices.get(vkey) != (xyz, color_key):
                stale.append(guid)
                self._guid_vertex.pop(guid, None)
                del self._drawn_vertices[vkey]

        wanted_edges = {frozenset((u, v)): (u, v) for u, v in mesh.edges()}
        for edge, (guid, ends) in list(self._drawn_edges.items()):
            keep = False
            if ends is not None and edge in wanted_edges:
                u, v = self._guid_edge.get(guid, (None, None))
                keep = u in mesh.vertex and v in mesh.vertex and ends == (tuple(mesh.vertex_coordinates(u)), tuple(mesh.vertex_coordinates(v)))
            if not keep:
                stale.append(guid)
                self._guid_edge.pop(guid, None)
                del self._drawn_edges[edge]

        added = 0
        previous = rs.EnableRedraw(False)
        try:
            live = [guid for guid in stale if rs.IsObject(guid)]
            if live:
                rs.DeleteObjects(live)
            for vkey, (_xyz, color_key) in wanted_vertices.items():
                if vkey not in self._drawn_vertices:
                    added += bool(self._add_vertex(vkey, color_key))
            for edge, (u, v) in wanted_edges.items():
                if edge not in self._drawn_edges:
                    added += bool(self._add_edge(u, v))
        finally:
            rs.EnableRedraw(previous)
        sc.doc.Views.Redraw()
        return len(stale), added

    def _vertex_color_key(self, vkey: int, boundary: Iterable[int], special: Iterable[int]) -> str:
        if vkey in special:
            return "vertex.special"
        if vkey in boundary:
            return "vertex.boundary"
        return "vertex"

    def _place(self, guid: Any, color_key: str) -> None:
        """Put a freshly added object on this layer, in its colour."""
        rs.ObjectLayer(guid, self.layer)
        rs.ObjectColor(guid, self.colors[color_key])
        self._guids.append(guid)

    def _add_vertex(self, vkey: int, color_key: str) -> Any | None:
        xyz = tuple(self.mesh.vertex_coordinates(vkey))
        guid = rs.AddPoint(Point3d(*xyz))
        if not guid:
            return None
        self._place(guid, color_key)
        self._guid_vertex[guid] = vkey
        self._drawn_vertices[vkey] = (guid, xyz, color_key)
        return guid

    def _add_edge(self, u: int, v: int) -> Any | None:
        ends = (tuple(self.mesh.vertex_coordinates(u)), tuple(self.mesh.vertex_coordinates(v)))
        try:
            guid = rs.AddLine(Point3d(*ends[0]), Point3d(*ends[1]))
        except Exception:  # noqa: BLE001
            guid = None  # a zero-length chord
        if not guid:
            return None
        self._place(guid, "edge")
        self._guid_edge[guid] = (u, v)
        self._drawn_edges[frozenset((u, v))] = (guid, ends)
        return guid

    def _add_shaped_edge(self, u: int, v: int, chorded: list[tuple[int, int]]) -> Any | None:
        """The edge drawn as its shape, or ``None`` to fall back to the chord."""
        shape = self.edge_shape(u, v) if self.edge_shape is not None else None
        if shape is None or len(shape) <= 2:
            return None
        try:
            guid = rs.AddPolyline([Point3d(*point) for point in shape])
        except Exception:  # noqa: BLE001
            chorded.append((u, v))
            return None
        if not guid:
            return None
        self._place(guid, "edge")
        self._guid_edge[guid] = (u, v)
        self._drawn_edges[frozenset((u, v))] = (guid, None)  # a SHAPE: sync replaces it
        return guid

    # --------------------------------------------------------------------------
    # pick
    # --------------------------------------------------------------------------

    def pick_vertex(self, message: str = "Select a vertex", preselect: bool = True) -> int | None:
        """A vertex key, or ``None`` on Esc. A miss re-prompts."""
        while True:
            guid = rs.GetObject(message, rs.filter.point, preselect=preselect)
            if not guid:
                return None
            vkey = self._guid_vertex.get(guid)
            if vkey is not None:
                return vkey
            print("Not a vertex of this mesh -- pick one of the points on {!r}.".format(self.layer))

    def pick_vertex_or_finish(self, message: str = "Select a vertex", preselect: bool = True) -> int | Any | None:
        """A vertex key, ``mesh_ui.FINISH`` on Enter, or ``None`` on Esc; the pick is unselected first."""
        while True:
            go = Rhino.Input.Custom.GetObject()
            go.SetCommandPrompt(message)
            go.GeometryFilter = Rhino.DocObjects.ObjectType.Point
            go.EnablePreSelect(preselect, True)
            go.AcceptNothing(True)
            result = go.Get()
            if result == Rhino.Input.GetResult.Cancel:
                return None
            if result == Rhino.Input.GetResult.Nothing:
                return FINISH
            if result != Rhino.Input.GetResult.Object:
                continue
            guid = go.Object(0).ObjectId
            sc.doc.Objects.UnselectAll()
            sc.doc.Views.Redraw()
            vkey = self._guid_vertex.get(guid)
            if vkey is not None:
                return vkey
            print("Not a vertex of this mesh -- pick one of the points on {!r}.".format(self.layer))

    def pick_edge(self, message: str = "Select an edge", preselect: bool = True) -> tuple[tuple[int, int], Any] | None:
        """``((u, v), guid)``, or ``None`` on Esc. A miss re-prompts.

        The guid comes back too: picking a point ALONG the edge
        (``rs.GetPointOnCurve``, or a drag constrained to it) needs the object.
        """
        while True:
            guid = rs.GetObject(message, rs.filter.curve, preselect=preselect)
            if not guid:
                return None
            edge = self._guid_edge.get(guid)
            if edge is not None:
                return edge, guid
            print("Not an edge of this mesh -- pick one of the lines on {!r}.".format(self.layer))

    def pick_face(self, message: str = "Pick a face") -> int | None:
        """One face key, or ``None`` on Esc. Only this object's faces can be picked."""
        guid = rs.GetObject(message, rs.filter.mesh, custom_filter=self._is_face)
        return self._guid_face.get(guid) if guid else None

    def pick_faces(self, message: str = "Pick faces, Enter when done") -> list[int] | None:
        """Face keys, or ``None`` on Esc. Click them one by one or drag a window."""
        guids = rs.GetObjects(message, rs.filter.mesh, custom_filter=self._is_face)
        if not guids:
            return None
        return [self._guid_face[guid] for guid in guids if guid in self._guid_face]

    def _is_face(self, rhino_object: Any, geometry: Any, component_index: Any) -> bool:
        return rhino_object.Id in self._guid_face

    def select_vertices(self, message: str = "Select a vertex") -> list[int] | None:
        vkey = self.pick_vertex(message)
        return None if vkey is None else [vkey]

    def select_edges(self, message: str = "Select an edge") -> list[tuple[int, int]] | None:
        picked = self.pick_edge(message)
        return None if picked is None else [picked[0]]

    def select_faces(self, message: str = "Pick faces, Enter when done") -> list[int] | None:
        return self.pick_faces(message)

    def closest_edge_point(self, point: Any, tol: float | None = None) -> Any | None:
        """``point`` snapped onto one of this mesh's edges if it lies on one, else ``None``."""
        if tol is None:
            tol = sc.doc.ModelAbsoluteTolerance * 2.0
        for guid in self._guid_edge:
            curve = rs.coercecurve(guid)
            if not curve:
                continue
            success, t = curve.ClosestPoint(point)
            if not success:
                continue
            candidate = curve.PointAt(t)
            if candidate.DistanceTo(point) <= tol:
                return candidate
        return None

    # --------------------------------------------------------------------------
    # show the run picked so far
    # --------------------------------------------------------------------------

    def show_path(self, vkeys: Sequence[int], color: tuple[int, int, int] | None = None) -> None:
        """Show the corners picked so far by recolouring their objects, replacing what was shown before."""
        self.clear_path()
        color = color or self.colors["path"]
        vkeys = list(vkeys)
        guid_of_vertex = {vkey: guid for guid, vkey in self._guid_vertex.items()}
        guid_of_edge = {frozenset(edge): guid for guid, edge in self._guid_edge.items()}

        targets = [guid_of_vertex[vkey] for vkey in vkeys if vkey in guid_of_vertex]
        chords = []
        for u, v in zip(vkeys[:-1], vkeys[1:]):
            guid = guid_of_edge.get(frozenset((u, v)))
            if guid is not None:
                targets.append(guid)
            else:
                chords.append((self.mesh.vertex_coordinates(u), self.mesh.vertex_coordinates(v)))

        for guid in targets:
            if guid in self._path_colors or not rs.IsObject(guid):
                continue
            self._path_colors[guid] = rs.ObjectColor(guid)
            rs.ObjectColor(guid, color)

        for a, b in chords:
            try:
                guid = rs.AddPolyline([Point3d(*a), Point3d(*b)])
            except Exception:  # noqa: BLE001
                guid = None
            if guid:
                rs.ObjectLayer(guid, self.layer)
                rs.ObjectColor(guid, color)
                self._guid_path.append(guid)
        sc.doc.Views.Redraw()

    def clear_path(self) -> None:
        """Put back the colours ``show_path`` changed. Safe to call twice."""
        for guid, original in self._path_colors.items():
            if rs.IsObject(guid):
                rs.ObjectColor(guid, original)
        self._path_colors = {}
        guids = [guid for guid in self._guid_path if rs.IsObject(guid)]
        if guids:
            rs.DeleteObjects(guids)
        self._guid_path = []

    # --------------------------------------------------------------------------
    # locks
    # --------------------------------------------------------------------------

    def unlock(self) -> dict[str, Any]:
        """Unlock this layer, its parents and the objects on it. Hand the result to ``relock``."""
        return unlock(self.layer, self._pickable_guids())

    def relock(self, state: dict[str, Any]) -> None:
        relock(state)

    def _pickable_guids(self) -> list[Any]:
        return list(self._guid_vertex) + list(self._guid_edge) + list(self._guid_face) + list(self._guid_path)
