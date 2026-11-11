#! python3

# r: compas
# r: pydantic
# r: compas_singular

"""Step 3 (alternative) -- read a coarse layout drawn over the domain boundaries.

Input: curves on Outer, Inner and Skeleton::EdgeCurves, points on Skeleton::Poles. Output: the layout in the session, drawn on Skeleton.
"""

import re

import compas_rhino as cr
import rhinoscriptsyntax as rs
from compas_rhino.conversions import point_to_compas

from compas.tolerance import TOL
from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.datastructures import split_at_corners
from compas_singular.datastructures import split_at_junctions
from compas_singular.rhino.helpers import curve_points
from compas_singular.rhino.project import get_settings
from compas_singular.rhino.project import resolve_spacing
from compas_singular.rhino.session import RhinoSession

#: Sampling of a smooth input curve, as a multiple of the background spacing.
CURVE_SAMPLING_FACTOR = 0.25

#: Most points a smooth curve is sampled into.
MAX_CURVE_POINTS = 256

#: Distance within which a curve end counts as landed on another curve.
ON_CURVE = 1e-3


def point_in_loop(point, loop):
    """Ray casting in XY. ``loop`` is closed, its last point not repeated."""
    x, y = point[0], point[1]
    hit = False
    n = len(loop)
    for i in range(n):
        ax, ay = loop[i][0], loop[i][1]
        bx, by = loop[(i + 1) % n][0], loop[(i + 1) % n][1]
        if (ay > y) != (by > y):
            t = (y - ay) / (by - ay)
            if x < ax + t * (bx - ax):
                hit = not hit
    return hit


def interior_point(loop):
    """A point strictly inside a closed loop, or ``None``.

    The centroid, or else the first midpoint between two corners that is inside.
    """
    n = len(loop)
    centre = [sum(p[0] for p in loop) / n, sum(p[1] for p in loop) / n, 0.0]
    if point_in_loop(centre, loop):
        return centre
    for i in range(n):
        for j in range(i + 2, n):
            mid = [(loop[i][0] + loop[j][0]) / 2.0,
                   (loop[i][1] + loop[j][1]) / 2.0, 0.0]
            if point_in_loop(mid, loop):
                return mid
    return None


def midpoint(points):
    """The point halfway along a polyline, by arc length."""
    segments = list(zip(points, points[1:]))
    lengths = [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in segments]
    half = sum(lengths) / 2.0
    for (a, b), length in zip(segments, lengths):
        if half <= length and length > 0.0:
            f = half / length
            return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, 0.0]
        half -= length
    return list(points[-1])


def on_curve(guid, curve, point):
    """Is ``point`` within :data:`ON_CURVE` of the TRUE Rhino curve?"""
    t = rs.CurveClosestPoint(guid, point)
    if t is None:
        return False
    p = curve.PointAt(t)
    return ((p.X - point[0]) ** 2 + (p.Y - point[1]) ** 2) ** 0.5 <= ON_CURVE


def lies_on_wall(curve, walls):
    """Does this WHOLE curve lie along one of the walls?

    Tested on a polyline's vertices or a smooth curve's samples, plus the middle of a two-point line.
    """
    ok, polyline = curve.TryGetPolyline()
    if ok:
        points = [[p.X, p.Y, 0.0] for p in polyline]
    else:
        points = [[q.X, q.Y, 0.0] for q in (curve.PointAt(t) for t in curve.DivideByCount(16, True))]
    if len(points) == 2:
        points.append(midpoint(points))
    return any(all(on_curve(guid, wall, q) for q in points) for guid, wall in walls)


def sample_smooth(guid, curve, sampling, ends):
    """A SMOOTH curve as one point list, with every end landing on it inserted as a sample.

    A closed curve comes back closed (first point repeated).
    """
    count = int(round(curve.GetLength() / max(sampling, 1e-6)))
    count = min(max(8, count), MAX_CURVE_POINTS)
    entries = []
    for t in curve.DivideByCount(count, True):
        p = curve.PointAt(t)
        entries.append((t, [p.X, p.Y, 0.0]))
    for end in ends:
        t = rs.CurveClosestPoint(guid, end)
        if t is None:
            continue
        p = curve.PointAt(t)
        if ((p.X - end[0]) ** 2 + (p.Y - end[1]) ** 2) ** 0.5 <= ON_CURVE:
            entries.append((t, [end[0], end[1], 0.0]))
    entries.sort(key=lambda entry: entry[0])
    points = [point for _t, point in entries]
    if curve.IsClosed and points:
        points.append(list(points[0]))
    return points


def read_curves(guids):
    """``(pieces, owner, smooth)``: polylines split at their corners, smooth curves kept for a second pass."""
    pieces, owner, smooth = [], [], []
    for guid in guids:
        curve = rs.coercecurve(guid)
        if curve is None:
            continue
        ok, polyline = curve.TryGetPolyline()
        if ok:
            for piece in split_at_corners([[p.X, p.Y, 0.0] for p in polyline]):
                pieces.append(piece)
                owner.append(guid)
        else:
            smooth.append((guid, curve))
    return pieces, owner, smooth


def read_network(sampling):
    """Everything drawn, as one-curve-per-edge. ``(pieces, source, is_wall, marks)``.

    ``source[i]`` is the Rhino curve of piece ``i``; ``marks`` counts EdgeCurves pieces used as wall corners.
    """
    outer_guids = list(rs.ObjectsByLayer("Outer") or [])
    wall_guids = outer_guids + list(rs.ObjectsByLayer("Inner") or [])
    layer = rs.AddLayer(name="EdgeCurves", parent="Skeleton")
    division_guids = list(rs.ObjectsByLayer(layer) or [])
    if not division_guids and not outer_guids:
        print("nothing on 'Outer' or '{}' -- select the curves of the layout "
              "instead.".format(layer))
        division_guids = rs.GetObjects(message="Select the coarse layout's curves",
                                       filter=4, preselect=True) or []

    walls = [(guid, rs.coercecurve(guid)) for guid in wall_guids]
    walls = [(guid, curve) for guid, curve in walls if curve is not None]
    marks = []
    kept = []
    for guid in division_guids:
        curve = rs.coercecurve(guid)
        if curve is None:
            continue
        if walls and lies_on_wall(curve, walls):
            for p in (curve.PointAtStart, curve.PointAtEnd):
                marks.append([p.X, p.Y, 0.0])
        else:
            kept.append(guid)

    wall_pieces, wall_owner, wall_smooth = read_curves(wall_guids)
    divisions, owners, division_smooth = read_curves(kept)

    ends = [end for piece in wall_pieces + divisions for end in (piece[0], piece[-1])]
    for _guid, curve in wall_smooth + division_smooth:
        if not curve.IsClosed:
            for p in (curve.PointAtStart, curve.PointAtEnd):
                ends.append([p.X, p.Y, 0.0])
    ends.extend(marks)

    for smooth, pieces, owner in ((wall_smooth, wall_pieces, wall_owner),
                                  (division_smooth, divisions, owners)):
        for guid, curve in smooth:
            points = sample_smooth(guid, curve, sampling, ends)
            if len(points) >= 2:
                pieces.append(points)
                owner.append(guid)

    network = wall_pieces + divisions
    owner = wall_owner + owners
    flags = [True] * len(wall_pieces) + [False] * len(divisions)
    pieces, origin = split_at_junctions(network, tol=ON_CURVE, points=marks)
    return (pieces, [owner[i] for i in origin], [flags[i] for i in origin],
            len(marks) // 2)


def read_loops(layer, sampling):
    """The closed curves on a boundary layer, as point loops."""
    loops = []
    for guid in rs.ObjectsByLayer(layer) or []:
        curve = rs.coercecurve(guid)
        if curve is None or not curve.IsClosed:
            continue
        loop = curve_points(guid, sampling)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def read_poles():
    """The pole points already on the layer, if the layout had any."""
    layer = rs.AddLayer(name="Poles", parent="Skeleton")
    return [list(point_to_compas(cr.objects.find_object(guid).Geometry))
            for guid in rs.ObjectsByLayer(layer) or []]


def outside_domain(pieces, is_wall, outer_loops, inner_loops):
    """``(index, why)`` for the first DIVISION outside the domain, or ``None``.

    Only tested with exactly one ``Outer`` loop.
    """
    if len(outer_loops) != 1:
        return None
    for i, (piece, wall) in enumerate(zip(pieces, is_wall)):
        if wall:
            continue
        point = midpoint(piece)
        if not point_in_loop(point, outer_loops[0]):
            return i, "a division line runs OUTSIDE the Outer boundary -- trim it to the wall"
        if any(point_in_loop(point, loop) for loop in inner_loops):
            return i, "a division line runs INSIDE a hole -- a hole is not meshed"
    return None


def off_boundary(coarse, pieces, is_wall, has_outer):
    """``(index, why)`` for a wall inside the layout or a division on its boundary, or ``None``."""
    index = {TOL.geometric_key(coarse.vertex_coordinates(v)): v for v in coarse.vertices()}
    for i, (piece, wall) in enumerate(zip(pieces, is_wall)):
        u = index.get(TOL.geometric_key(piece[0]))
        v = index.get(TOL.geometric_key(piece[-1]))
        if u is None or v is None or v not in coarse.halfedge[u]:
            continue
        boundary = coarse.is_edge_on_boundary((u, v))
        if wall and not boundary:
            return i, "this Outer/Inner curve ends up INSIDE the layout -- a hole was not recognised"
        if not wall and has_outer and boundary:
            return i, "this division line ends up on the EDGE of the layout -- the patch beside it is missing"
    return None


def select(guids):
    rs.UnselectAllObjects()
    rs.SelectObjects(list(dict.fromkeys(guids)))


def show_refusal(error, source):
    """Print a refusal, and select the Rhino curves whose piece indices it names."""
    message = str(error)
    print("the drawing is not a coarse layout yet:")
    print("  {}".format(message))

    indices = set()
    for phrase in re.findall(r"polylines?(?:\(s\))?\s+([\d\s,\[\]and]+)", message):
        indices.update(int(n) for n in re.findall(r"\d+", phrase))
    guids = [source[i] for i in sorted(indices) if 0 <= i < len(source)]
    if guids:
        select(guids)
        print("  selected the curve(s) involved.")
    if "same corner" in message or "both run between" in message:
        print("  a closed curve is split into edges only where other curves land "
              "on it -- a hole or a round wall needs at least three.")
    if "separate pieces" in message:
        print("  if the selected curve is a hole, draw at least three division "
              "lines from it to the rest of the layout.")
    if "one curve only" in message:
        print("  if that end was meant to reach a curve, snap it onto the curve "
              "(End, Near or Int) -- a near miss is not a landing.")
    print("nothing was changed.")


def main():
    settings = get_settings()
    sampling = resolve_spacing(settings) * CURVE_SAMPLING_FACTOR

    # Read everything before writing: the poles layer is rewritten.
    pieces, source, is_wall, marks = read_network(sampling)
    if not pieces:
        print("no curves to read.")
        return
    outer_loops = read_loops("Outer", sampling)
    inner_loops = read_loops("Inner", sampling)
    holes = [point for point in (interior_point(loop) for loop in inner_loops)
             if point is not None]
    poles = read_poles()

    print("read {} curve(s) as {} edge(s): {} wall, {} division; {} hole(s), "
          "{} pole point(s)".format(len(set(source)), len(pieces), sum(is_wall),
                                    len(pieces) - sum(is_wall), len(holes), len(poles)))
    if marks:
        print("  {} piece(s) of EdgeCurves lie along a wall -- read as corners on "
              "it, not as edges".format(marks))

    stray = outside_domain(pieces, is_wall, outer_loops, inner_loops)
    if stray is not None:
        print("the drawing is not a coarse layout yet:")
        print("  {}.".format(stray[1]))
        select([source[stray[0]]])
        print("  selected the curve involved.")
        print("nothing was changed.")
        return

    try:
        coarse = CoarsePseudoQuadMesh.from_coarse_polylines(
            pieces, poles=poles, holes=holes)
    except ValueError as error:
        show_refusal(error, source)
        return

    wrong = off_boundary(coarse, pieces, is_wall, bool(outer_loops))
    if wrong is not None:
        print("the drawing is not a coarse layout yet:")
        print("  {}.".format(wrong[1]))
        select([source[wrong[0]]])
        print("  selected the curve involved.")
        print("nothing was changed.")
        return

    sides = {}
    for fkey in coarse.faces():
        n = len(coarse.face_vertices(fkey))
        sides[n] = sides.get(n, 0) + 1
    print("layout: {} patch(es), {} corner(s), sides {}, {} strip(s)".format(
        coarse.number_of_faces(), coarse.number_of_vertices(), sides,
        coarse.number_of_strips()))
    if sides.get(3):
        print("  {} triangular patch(es), kept as pseudo-quads with a collapsed "
              "corner".format(sides[3]))

    coarse.attributes["route"] = "drawn"
    coarse.set_shape_polylines(pieces)

    session = RhinoSession.current()
    session.coarse = coarse
    session.record("Read coarse mesh")
    print("layout stored in the session, and drawn")
    print("note: this replaces any previous layout -- densities (CMD_densities) and "
          "patterns (CMD_dense_pattern) set on it do not carry over.")
    print("next: CMD_densities, then CMD_quad_mesh.")


if __name__ == "__main__":
    main()
