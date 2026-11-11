from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

import compas_rhino as cr
from compas_rhino.conversions import mesh_to_compas
from compas_rhino.conversions import point_to_compas
from compas_rhino.conversions import polyline_to_rhino
from compas_rhino.conversions import vertices_and_faces_to_rhino

import compas
from compas.geometry import Polygon
from compas.geometry import Polyline
from compas.geometry import is_point_in_polygon_xy
from compas.geometry import is_polygon_in_polygon_xy
from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.rhino.project import ROOT

if TYPE_CHECKING:
    from compas.datastructures import Mesh

if compas.RHINO:
    import rhinoscriptsyntax as rs  # type: ignore  # noqa: I001
    import scriptcontext as sc  # type: ignore  # noqa: I001
    import System


def clear_layer(layer: str, clean_sublayers: bool = False) -> int:
    if not rs.IsLayer(layer):
        print(f"No layer named {layer}. Deleted nothing.")
        return 0

    layer = rs.LayerName(layer, fullpath=True)
    count = 0  # Number of deleted objects
    objs = rs.ObjectsByLayer(layer)
    if objs:
        rs.DeleteObjects(objs)
        count += len(objs)

    if clean_sublayers:
        all_layers = rs.LayerNames()
        if all_layers:
            prefix = layer + "::"
            for other in all_layers:
                if other != layer and other.startswith(prefix):
                    objs = rs.ObjectsByLayer(other)
                    if objs:
                        rs.DeleteObjects(objs)
                        count += len(objs)

    return count


def move_to_trash(obj_ids: Any) -> None:
    """Move objects to the hidden ``TrashBin`` layer instead of deleting them."""
    if not rs.IsLayer("TrashBin"):
        rs.AddLayer(name="TrashBin", parent=ROOT, color=(90, 90, 90), visible=False)
    rs.ObjectLayer(obj_ids, "TrashBin")


def clean_polyline_points(points: Sequence[Sequence[float]], tol: float | None = None) -> list[list[float]]:
    """``points`` with consecutive duplicates dropped at Rhino's document tolerance. ``[]`` if nothing is left."""
    if tol is None:
        try:
            tol = sc.doc.ModelAbsoluteTolerance
        except Exception:
            tol = 1e-6
    out = []
    for point in points:
        point = list(point)[:3]
        if out and _distance(out[-1], point) <= tol:
            continue
        out.append(point)
    return out if len(out) >= 2 else []


def bake_polylines(
    polylines: Sequence[Sequence[Sequence[float]]],
    layer: str,
    color: Any = None,
    clear_existing: bool = True,
) -> tuple[list[Any], int]:
    """Bake polylines, skipping any Rhino will not take. ``(guids, skipped)``.

    ``rs.AddPolyline`` raises rather than returning ``None``, so each add is guarded.
    """
    if not rs.IsLayer(layer):
        rs.AddLayer(layer, color)
    if clear_existing:
        clear_layer(layer)

    guids = []
    skipped = 0
    for polyline in polylines:
        points = clean_polyline_points(polyline)
        if not points:
            skipped += 1
            continue
        try:
            guid = rs.AddPolyline(polyline_to_rhino(Polyline(points)))
        except Exception:
            guid = None
        if not guid:
            skipped += 1
            continue
        rs.ObjectLayer(guid, layer=layer)
        guids.append(guid)
    return guids, skipped


def bake_mesh(mesh: Mesh, layer: str, color: Any = None, clear_existing: bool = True) -> Any:
    """Bake a compas mesh with n-gons kept and vertices welded. Returns the guid.

    Not ``rs.AddMesh``, which bakes an n-gon as the quad of its first four corners.
    """
    if not rs.IsLayer(layer):
        rs.AddLayer(layer, color, parent=ROOT)
        print("Layer did not exist. Added the layer.")
    if clear_existing:
        clear_layer(layer)

    # ``face_vertices`` gives vertex KEYS while ``vertices`` is positional, and
    # the two only agree while the keys happen to be 0..n-1. A repair, a weld or
    # a deleted patch leaves gaps, and then the baked faces reference the wrong
    # corners -- silently, as a valid mesh of the wrong shape. Map explicitly.
    index = {v: i for i, v in enumerate(mesh.vertices())}
    vertices = [mesh.vertex_attributes(v, "xyz") for v in mesh.vertices()]
    faces = [[index[v] for v in mesh.face_vertices(f)] for f in mesh.faces()]

    geometry = vertices_and_faces_to_rhino(vertices, faces, disjoint=False)
    guid = sc.doc.Objects.AddMesh(geometry)
    if guid == System.Guid.Empty:
        raise Exception("Unable to add the mesh to the document.")
    sc.doc.Views.Redraw()
    rs.ObjectLayer(guid, layer=layer)
    return guid


#: Fewest points a CLOSED curved input is divided into, whatever ``max_edge``
#: says. A curve is divided by CHORD LENGTH here, so the count a loop gets is
#: its own length over the background spacing -- and a small feature is short.
#: At the default spacing of 0.5 a round hole of radius 0.64 or less came out as
#: a regular OCTAGON: 45 degrees of turn per vertex, which is exactly the angle
#: a cross field cannot tell from a corner (see ``13_limits.py``). Rhino's own
#: Convert samples by DEVIATION instead and never does this, which is why the
#: same boundary worked once it had been converted by hand first.
#:
#: 28 caps the turn at 12.9 degrees, comfortably clear of that limit, and is
#: measured: it puts every curved domain in the test set on the field route,
#: while 32 and above start costing background triangle quality -- the loop's
#: chords fall well below ``target_length`` and ``background._densify_loop``
#: only ever subdivides, so it cannot coarsen them back. Measured minimum
#: background angle on a rounded plate with a radius 0.5 hole: 20.8 deg at the
#: old floor of 8, 11.0 at 28, 9.3 at 32, 6.3 at 48.
MIN_CLOSED_CURVE_POINTS = 28

#: A CLOSED loop's point count is rounded UP to a multiple of this.
#:
#: ``framefield.symmetry.Symmetry.detect`` matches by POINT SET at a 1e-6
#: tolerance, so a symmetric drawing is only detected as symmetric if its
#: SAMPLES are symmetric too. Dividing a circle into ``n`` equal parts gives a
#: set closed under a quarter turn only when ``n % 4 == 0``; ``n`` even gives
#: order 4 and ``n`` odd order 2. The count came straight from
#: ``length / max_edge``, so it was effectively arbitrary: measured on a square
#: plate with a centred hole at spacing 0.5, radii 0.5-2.0 detected the full D4
#: (count 28, from the floor above, which happens to be a multiple of 4) while
#: r=2.5 gave count 31 and collapsed to order 2, and r=3.0 gave count 38 and
#: order 4. Rounding up costs at most three points.
#:
#: This fixes the COUNT, not the PHASE. ``DivideByCount`` starts at the curve's
#: seam, so the same 28-point circle rotated 5 degrees still measures order 4:
#: the mirror axes no longer land on samples. Draw holes with the seam on a
#: symmetry axis (Rhino's own circles have theirs at angle 0, which is why this
#: is rarely hit), or pass an explicit group instead of ``'auto'``.
SYMMETRY_SAMPLE_MULTIPLE = 4


def curve_points(guid: Any, max_edge: float) -> list[list[float]]:
    """A Rhino curve as points: a polyline at its own vertices, any other curve divided by length.

    Closed loops get at least ``MIN_CLOSED_CURVE_POINTS``, rounded to ``SYMMETRY_SAMPLE_MULTIPLE``.
    """
    curve = rs.coercecurve(guid)
    ok, polyline = curve.TryGetPolyline()
    if ok:
        points = [[p.X, p.Y, 0.0] for p in polyline]
    else:
        count = int(round(curve.GetLength() / max(max_edge, 1e-6)))
        # A closed loop gets a floor of its own; an open guide does not need
        # one, having no winding for the field to read.
        count = max(MIN_CLOSED_CURVE_POINTS if curve.IsClosed else 8, count, 8)
        if curve.IsClosed:
            count = -(-count // SYMMETRY_SAMPLE_MULTIPLE) * SYMMETRY_SAMPLE_MULTIPLE
        points = []
        for t in curve.DivideByCount(count, True):
            p = curve.PointAt(t)
            points.append([p.X, p.Y, 0.0])
    while len(points) > 1 and _distance(points[0], points[-1]) < 1e-9:
        points = points[:-1]
    return points


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def curve_to_polyline(guid: Any, max_edge: float) -> Polyline:
    """``curve_points`` as a closed compas ``Polyline`` (first point repeated)."""
    points = curve_points(guid, max_edge)
    return Polyline(points + points[:1])


def read_boundary_loops(max_edge: float) -> tuple[list[list[float]], list[list[list[float]]]]:
    """``(outer, inners)`` as point lists sampled from the Rhino curves, for ``coarse_edges_to_curves``.

    Sample finer than the background: these points are the wall from here on.
    """
    outer_ids = rs.ObjectsByLayer("Outer")
    if not outer_ids:
        raise RuntimeError("No outer boundary selected.")

    outer = curve_points(outer_ids[0], max_edge)
    inners = [curve_points(guid, max_edge) for guid in rs.ObjectsByLayer("Inner") or []]
    return outer, inners


def read_polylines(layer: str) -> list[list[list[float]]]:
    """Every polyline on a layer as a point list. ``[]`` if the layer is empty.

    Used for the decomposition's branches on ``Skeleton::Polylines``. Unlike the
    coarse MESH, these are baked as curves and so keep full double precision.
    """
    out = []
    for guid in rs.ObjectsByLayer(layer) or []:
        curve = rs.coercecurve(guid)
        if curve is None:
            continue
        ok, polyline = curve.TryGetPolyline()
        if not ok:
            continue
        out.append([[p.X, p.Y, p.Z] for p in polyline])
    return out


def read_boundaries(spacing: float | None = None) -> tuple[Polyline, list[Polyline], list[Polyline], list[Any]]:
    """``(outer, inners, guides, poles)``: polylines closed except the guides."""
    max_edge = 0.125 if spacing is None else spacing
    outer_ids = rs.ObjectsByLayer("Outer")
    inner_ids = rs.ObjectsByLayer("Inner") or []
    guide_ids = rs.ObjectsByLayer("Guides") or []
    pole_ids = rs.ObjectsByLayer("PointFeatures") or []

    if not outer_ids:
        raise RuntimeError("No outer boundary selected.")

    outer = curve_to_polyline(outer_ids[0], max_edge)

    inners = []
    for obj_id in inner_ids:
        inner_compas = curve_to_polyline(obj_id, max_edge)
        if is_polygon_in_polygon_xy(Polygon(outer), Polygon(inner_compas)):
            inners.append(inner_compas)
        else:
            move_to_trash(obj_id)
            print("This inner boundary did not lie inside the outer boundary. It has been removed from the selection.")

    for boundary in inners + [outer]:
        if not boundary.is_closed:
            raise RuntimeError("Not all boundaries are closed.")

    guides = []
    for guide_id in guide_ids:
        # A guide is an OPEN curve -- do not close it the way a loop is closed.
        guide_compas = Polyline(curve_points(guide_id, max_edge))
        if is_polygon_in_polygon_xy(Polygon(outer), Polygon(guide_compas)):
            guides.append(guide_compas)
        else:
            move_to_trash(guide_id)
            print("This guide did not lie inside the outer boundary. It has been removed from the selection.")

    poles = []
    for pole_id in pole_ids:
        point = cr.objects.find_object(pole_id)
        point_compas = point_to_compas(point.Geometry)
        if is_point_in_polygon_xy(point_compas, Polygon(outer)):
            if all(not is_point_in_polygon_xy(point_compas, Polygon(inner)) for inner in inners):
                poles.append(point_compas)
            else:
                move_to_trash(pole_id)
                print("This pole lies inside an inner boundary. It has been removed from the selection.")
        else:
            move_to_trash(pole_id)
            print("This pole did not lie inside the outer boundary. It has been removed from the selection.")

    return outer, inners, guides, poles


def read_coarse() -> tuple[CoarsePseudoQuadMesh, list[Any]]:
    guids = rs.ObjectsByLayer("Mesh") if rs.IsLayer("Mesh") else None
    if not guids:
        raise RuntimeError("No coarse layout on '{}' ".format("Mesh"))
    print(cr.objects.find_object(guids[0]))
    mesh = mesh_to_compas(cr.objects.find_object(guids[0]).Geometry)
    vertices, faces = mesh.to_vertices_and_faces()
    faces = [[v for i, v in enumerate(f) if v != f[i - 1]] for f in faces]

    poles = [list(point_to_compas(cr.objects.find_object(guid).Geometry)) for guid in (rs.ObjectsByLayer("Poles") if rs.IsLayer("Poles") else None) or []]

    coarse_mesh = CoarsePseudoQuadMesh.from_vertices_and_faces_with_poles(vertices, faces, poles)
    return coarse_mesh, poles


def read_mesh(layer: str) -> Mesh:
    guids = rs.ObjectsByLayer(layer)
    if not guids:
        raise RuntimeError("No mesh on the layer '{}' ".format("Mesh"))

    mesh = mesh_to_compas(cr.objects.find_object(guids[0]).Geometry)
    return mesh


def mesh_from_rhino(rhinomesh: Any, cls: type | None = None) -> Any:
    """A Rhino mesh as a compas mesh, with its n-gons read back as single faces.

    Vertex keys are renumbered 0..n-1; hold nothing across the round trip.
    """
    if cls is None:
        from compas.datastructures import Mesh as cls

    grouped = set()
    polygons = []
    ngons = rhinomesh.Ngons
    for i in range(ngons.Count):
        ngon = ngons[i]
        grouped.update(int(fkey) for fkey in ngon.FaceIndexList())
        polygons.append([int(vkey) for vkey in ngon.BoundaryVertexIndexList()])

    faces = rhinomesh.Faces
    for i in range(faces.Count):
        if i in grouped:
            continue
        face = faces[i]
        polygons.append([face.A, face.B, face.C] if face.IsTriangle else [face.A, face.B, face.C, face.D])

    cleaned = []
    for polygon in polygons:
        polygon = [v for i, v in enumerate(polygon) if v != polygon[i - 1]]
        if len(polygon) >= 3:
            cleaned.append(polygon)

    used = sorted({v for polygon in cleaned for v in polygon})
    index = {v: i for i, v in enumerate(used)}
    vertices = []
    for v in used:
        point = rhinomesh.Vertices[v]
        vertices.append([float(point.X), float(point.Y), float(point.Z)])
    return cls.from_vertices_and_faces(vertices, [[index[v] for v in polygon] for polygon in cleaned])
