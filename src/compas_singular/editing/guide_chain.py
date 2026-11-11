"""Choose the dense-mesh vertices that follow a guide curve, with no Rhino in it.

The chain is the longest run of one polyedge that stays near and parallel to the guide.
Design notes: ``design_notes/editing.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import cos
from math import degrees
from math import radians
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence

from compas.geometry import Point
from compas.geometry import Polyline
from compas.geometry import angle_vectors
from compas_singular.datastructures.mesh.smoothing import closest_point_on_constraint
from compas_singular.datastructures.mesh.smoothing import mesh_boundary_loops
from compas_singular.datastructures.mesh.smoothing import mesh_boundary_polylines

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh


__all__ = [
    "GuideCurve",
    "guide_chain",
    "chain_quality",
    "attach_chain",
    "collect_polyedges",
    "mean_edge_length",
]


#: How far off the guide a chain vertex may sit, in mean edge lengths -- i.e. in target
#: quad edge lengths, the number the user set when the mesh was densified.
#:
#: A vertex is YANKED exactly as far as it was allowed to sit off the guide, so this sets
#: both how much of the guide is covered and how hard the mesh is pulled about.
#:
#: There are TWO gates and they catch different failures, which is why both are kept.
#: Swept over 14 guides on three meshes (``examples/dev_examples_tests/guide_chain_tests``) -- worst face
#: angle after attaching, and mean coverage:
#:
#: ========= =================== =================== =========================
#: tolerance angle off (90 deg)  angle on (30 deg)   refused
#: ========= =================== =================== =========================
#: 0.6       31.6 deg / 70%      31.6 deg / 76%      diagonal, wall
#: 0.8       15.9 deg / 74%      31.6 deg / 79%      diagonal, wall
#: 1.0       0.0 deg / 78%       0.0 deg / 82%       diagonal
#: 1.5       0.0 deg / 80%       0.0 deg / 83%       diagonal
#: **2.0**   0.0 deg / 87%       **0.0 deg / 89%**   diagonal
#: ========= =================== =================== =========================
#:
#: **The angle gate is what makes a wide distance safe**, and it is why the default can be
#: this generous. Without it, widening lets a chain follow a guide past the point where it
#: has stopped being parallel and the worst face falls to 15.9 at 0.8 alone; with it, those
#: vertices are dropped instead. Going from 0.8 to 2.0 with the gate on changes only three
#: of the fourteen guides, and two of those are gains:
#:
#: * ``square/axis`` -- 78% coverage to 100%, worst face angle unchanged at 49 deg;
#: * ``square+hole/ring`` -- a circular guide over a square-cornered ring goes from a
#:   3-vertex stub at 13% to the whole 17-vertex ring at 100%, worst face 78 to 41 deg,
#:   because its corners sit 1.67 edge lengths out and nothing tighter can reach them;
#: * ``square/wall`` -- a guide drawn ON the outline is no longer refused, and the row one
#:   in gets pulled onto the wall (0.0 deg). This is the price, and no angle can catch it:
#:   the row one in from a wall is perfectly parallel to it. Drop to 0.8 to have the
#:   distance gate refuse that case for you.
#:
#: ``diagonal`` is refused at every setting and should be: it runs at 45 degrees to both
#: families, so no polyedge follows it.
DEFAULT_TOLERANCE_FACTOR = 2.0

#: How far a polyedge may run off the guide's tangent AT a vertex, in degrees, and still
#: have that vertex selected.
#:
#: Applied PER VERTEX, which is the whole point of it: distance asks whether a vertex is
#: near the guide, angle asks whether its line is still going the same way. A polyedge
#: that runs beside a guide and then veers off keeps passing the distance test for a
#: vertex or two after it has stopped following, and those are exactly the vertices whose
#: projection bends the mesh. Cutting them is what the gate is for.
#:
#: Swept at tolerance 1.5 so the distance gate cannot mask it: 45 and 90 select
#: everything; 30 refuses the 45-degree ``diagonal`` guide, which no polyedge follows; 20
#: additionally refuses a circular guide over a square-cornered ring. 30 is the value that
#: refuses what should be refused and nothing else. 90 turns the gate off.
DEFAULT_MAX_ANGLE = 30.0

_TOL = 1e-9


# ==============================================================================
# The guide
# ==============================================================================


class GuideCurve(object):
    """A guide curve as a polyline with an arc-length table and, optionally, the curve itself.

    Chains are chosen on the samples; ``closest_point`` lands on ``curve`` when given.

    Parameters
    ----------
    points : sequence[point] | compas.geometry.Polyline
        The guide, already discretised. Consecutive duplicates are dropped.
    curve : compas.geometry.Curve, optional
        The curve ``points`` were sampled from -- from ``compas_rhino``'s
        ``curve_to_compas``, from ``compas_occ``, a ``compas.geometry.Circle``, ...
        It must lie where the samples do. Default is None: vertices land on the samples.

    Attributes
    ----------
    points : list[list[float]]
        The guide points.
    length : float
        The total arc length of the samples.
    closed : bool
        Whether the first and last point coincide.
    curve : compas.geometry.Curve | None
        The curve a vertex lands on, if one was given.

    Raises
    ------
    ValueError
        If fewer than two distinct points are given.
    """

    def __init__(self, points: "Iterable[Sequence[float]]", curve: Any = None) -> None:
        self.curve = curve
        cleaned = []
        for point in points:
            xyz = [float(point[0]), float(point[1]), float(point[2])]
            if not cleaned or _distance(cleaned[-1], xyz) > _TOL:
                cleaned.append(xyz)
        if len(cleaned) < 2:
            raise ValueError("A guide needs at least two distinct points.")

        self.points = cleaned
        self.closed = _distance(cleaned[0], cleaned[-1]) <= _TOL

        cumulative = [0.0]
        for a, b in zip(cleaned, cleaned[1:]):
            cumulative.append(cumulative[-1] + _distance(a, b))
        self.cumulative = cumulative
        self.length = cumulative[-1]

    def _closest(self, xyz: list[float]) -> tuple[list[float] | None, float | None, float, tuple[float, float, float] | None]:
        """(closest point, distance, arc length, unit tangent) -- everything, once."""
        best = (None, None, 0.0, None)
        points, cumulative = self.points, self.cumulative
        for index, (a, b) in enumerate(zip(points, points[1:])):
            abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
            length2 = abx * abx + aby * aby + abz * abz
            if length2 <= 0.0:
                continue
            t = ((xyz[0] - a[0]) * abx + (xyz[1] - a[1]) * aby + (xyz[2] - a[2]) * abz) / length2
            t = max(0.0, min(1.0, t))
            closest = [a[0] + abx * t, a[1] + aby * t, a[2] + abz * t]
            distance = _distance(xyz, closest)
            if best[1] is None or distance < best[1]:
                norm = length2**0.5
                best = (closest, distance, cumulative[index] + t * norm, (abx / norm, aby / norm, abz / norm))
        return best

    def project(self, xyz: list[float]) -> tuple[float | None, float, tuple[float, float, float] | None]:
        """Project a point onto the guide.

        Parameters
        ----------
        xyz : [float, float, float]
            The point to project.

        Returns
        -------
        tuple[float | None, float, tuple[float, float, float] | None]
            The distance to the guide, the arc length of the closest point, and the unit
            tangent there. ``(None, 0.0, None)`` if the guide has no non-degenerate
            segment, which cannot happen for a guide built by this class.

        """
        _, distance, t, tangent = self._closest(xyz)
        return distance, t, tangent

    def closest_point(self, point: list[float]) -> list[float] | None:
        """The closest point on the guide (on ``curve`` if set), usable as a smoothing constraint."""
        if self.curve is not None:
            return closest_point_on_constraint(self.curve, point)
        return self._closest(point)[0]

    def delta(self, t: float, t0: float) -> float:
        """Signed progress from arc length ``t0`` to ``t``, wrapped the short way on a closed guide."""
        difference = t - t0
        if self.closed and self.length > 0.0:
            half = 0.5 * self.length
            while difference > half:
                difference -= self.length
            while difference < -half:
                difference += self.length
        return difference


def _distance(a: list[float], b: list[float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _unit(vector: list[float]) -> list[float] | None:
    norm = (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5
    if norm <= 0.0:
        return None
    return [component / norm for component in vector]


def _angle(a: list[float] | None, b: list[float] | None) -> float:
    """Degrees between two directions, or 0 if either has no direction."""
    if a is None or b is None:
        return 0.0
    return degrees(angle_vectors(a, b))


def _as_guide(guide: "GuideCurve | Iterable[Sequence[float]]") -> GuideCurve:
    if isinstance(guide, GuideCurve):
        return guide
    return GuideCurve(guide)


def mean_edge_length(mesh: "QuadMesh") -> float:
    """The mean length of the non-degenerate edges of a mesh."""
    lengths = [mesh.edge_length(edge) for edge in mesh.edges()]
    lengths = [length for length in lengths if length > 0.0]
    return sum(lengths) / len(lengths) if lengths else 1.0


def _boundary_vertices(mesh: "QuadMesh") -> set[int]:
    # NOT vertices_on_boundary(): in COMPAS 2 that returns the LONGEST boundary only, so
    # every hole would count as interior and a chain could run right round one.
    return set(vertex for loop in mesh_boundary_loops(mesh) for vertex in loop)


# ==============================================================================
# Choosing the chain
# ==============================================================================


def collect_polyedges(mesh: "QuadMesh") -> list[tuple[list[int], bool]]:
    """The polyedges of a quad mesh as vertex lists, closed ones without the repeated end.

    Collect once and reuse across guides; it is O(edges squared).
    """
    mesh.attributes["polyedges"] = {}
    polyedges = []
    for _, polyedge in mesh.collect_polyedges():
        if len(polyedge) > 2 and polyedge[0] == polyedge[-1]:
            polyedges.append((list(polyedge[:-1]), True))
        else:
            polyedges.append((list(polyedge), False))
    return polyedges


def _runs(polyedge: list[int], keep: list[bool], closed: bool) -> list[list[int]]:
    """The maximal runs of consecutive kept vertices, as lists of vertex keys.

    On a closed polyedge the run may wrap past the first vertex, which is the whole reason
    the repeated end vertex is dropped before we get here.
    """
    runs, current = [], []
    for vertex, keeping in zip(polyedge, keep):
        if keeping:
            current.append(vertex)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    if closed and len(runs) > 1 and keep[0] and keep[-1]:
        runs[0] = runs[-1] + runs[0]  # the last run continues into the first
        runs.pop()
    elif closed and len(runs) == 1 and all(keep):
        runs[0] = runs[0] + runs[0][:1]  # the whole loop: close it
    return runs


def _oriented(mesh: "QuadMesh", guide: GuideCurve, run: list[int]) -> list[int]:
    """The run, ordered so that it advances along the guide."""
    _, t_first, _ = guide.project(mesh.vertex_coordinates(run[0]))
    _, t_last, _ = guide.project(mesh.vertex_coordinates(run[-1]))
    if guide.delta(t_last, t_first) < 0.0:
        return list(reversed(run))
    return run


def _trimmed(mesh: "QuadMesh", guide: GuideCurve, run: list[int]) -> list[int]:
    """Drop vertices that project past the end of the guide, keeping the innermost one."""
    if guide.closed or len(run) < 3:
        return run
    parameters = [guide.project(mesh.vertex_coordinates(vertex))[1] for vertex in run]
    start, end = 0, len(run) - 1
    while end - start >= 2 and abs(parameters[start + 1] - parameters[start]) <= _TOL:
        start += 1
    while end - start >= 2 and abs(parameters[end] - parameters[end - 1]) <= _TOL:
        end -= 1
    return run[start : end + 1]


def _directions(mesh: "QuadMesh", polyedge: list[int], closed: bool) -> list[list[float] | None]:
    """The polyedge's own direction at each vertex, taken across the vertex."""
    points = [mesh.vertex_coordinates(vertex) for vertex in polyedge]
    count = len(points)
    if count < 2:
        return [None] * count
    directions = []
    for index in range(count):
        if closed:
            a, b = points[(index - 1) % count], points[(index + 1) % count]
        elif index == 0:
            a, b = points[0], points[1]
        elif index == count - 1:
            a, b = points[count - 2], points[count - 1]
        else:
            a, b = points[index - 1], points[index + 1]
        directions.append(_unit([b[i] - a[i] for i in range(3)]))
    return directions


def _parallel(direction: list[float] | None, tangent: tuple[float, float, float] | None, minimum_cosine: float) -> bool:
    """Whether a polyedge direction still follows the guide's tangent.

    ``abs`` because a polyedge has no direction of its own -- it may be collected either
    way round -- but NOT cross-symmetric: the crossing family must fail this.
    """
    if direction is None or tangent is None:
        return True  # nothing to judge it on
    return abs(sum(direction[i] * tangent[i] for i in range(3))) >= minimum_cosine


def _alignment(mesh: "QuadMesh", guide: GuideCurve, run: list[int]) -> float:
    """How closely the run's own edges follow the guide: 1 along it, 0 across it."""
    values = []
    for u, v in zip(run, run[1:]):
        a, b = mesh.vertex_coordinates(u), mesh.vertex_coordinates(v)
        direction = _unit([b[i] - a[i] for i in range(3)])
        midpoint = [(a[i] + b[i]) / 2.0 for i in range(3)]
        _, _, tangent = guide.project(midpoint)
        if direction is None or tangent is None:
            continue
        values.append(abs(sum(direction[i] * tangent[i] for i in range(3))))
    return sum(values) / len(values) if values else 0.0


def _span(mesh: "QuadMesh", chain: list[int], guide: GuideCurve, average: float) -> dict[str, float]:
    """Coverage and end gaps, summed step by step so a closed guide works too."""
    if len(chain) < 2 or guide.length <= 0.0:
        return {"coverage": 0.0, "gap_start": 0.0, "gap_end": 0.0}

    parameters = [guide.project(mesh.vertex_coordinates(vertex))[1] for vertex in chain]
    covered = sum(abs(guide.delta(b, a)) for a, b in zip(parameters, parameters[1:]))
    coverage = min(1.0, covered / guide.length)

    if guide.closed:
        return {"coverage": coverage, "gap_start": 0.0, "gap_end": 0.0}
    return {
        "coverage": coverage,
        "gap_start": min(parameters) / average,
        "gap_end": (guide.length - max(parameters)) / average,
    }


def guide_chain(
    mesh: "QuadMesh",
    guide: "GuideCurve | Iterable[Sequence[float]]",
    tolerance: float | None = None,
    max_angle: float = DEFAULT_MAX_ANGLE,
    boundary: str = "anchor",
    polyedges: list[tuple[list[int], bool]] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Choose the run of a polyedge that follows a guide curve.

    Parameters
    ----------
    mesh : compas_singular.datastructures.QuadMesh
        The dense quad mesh. A ``QuadMesh``, not a plain
        ``compas.datastructures.Mesh``: the polyedges are what this selects from.
    guide : GuideCurve | sequence[point]
        The guide curve, already discretised.
    tolerance : float, optional
        How far off the guide a chain vertex may sit. Default is
        ``DEFAULT_TOLERANCE_FACTOR`` times the mean edge length -- the target quad edge
        length the mesh was densified with.
    max_angle : float, optional
        How far the polyedge may run off the guide's tangent at a vertex, in degrees, and
        still have that vertex selected. Default is ``DEFAULT_MAX_ANGLE``; ``90`` turns it off.
    boundary : {'anchor', 'exclude', 'free'}, optional
        What the chain may do with boundary vertices.

        * ``'anchor'`` -- a polyedge that is entirely on the boundary is refused, so the
          only boundary vertices that can survive are the two ENDS of an interior
          polyedge. A cable drawn wall to wall reaches the wall.
        * ``'exclude'`` -- boundary vertices are dropped as well, so the chain stops one
          vertex short at each end.
        * ``'free'`` -- no restriction: the outline itself may be selected. This exists so
          a measurement can show what that costs; it is not a setting to use.

    polyedges : list, optional
        The result of ``collect_polyedges`` for this mesh, if it has already been
        collected. Default is None, in which case it is collected here.

    Returns
    -------
    tuple[list[int], dict]
        The chain, ordered along the guide, and a dictionary with:

        ``reason``
            Why the chain is empty, or ``''``.
        ``tolerance``, ``boundary``
            What it ran with.
        ``alignment``
            How closely the chosen chain runs along the guide, 0 to 1.
        ``candidates``
            How many polyedge runs were within tolerance of the guide.
        ``coverage``
            The share of the guide's length the chain spans, 0 to 1.
        ``gap_start``, ``gap_end``
            The length of guide left uncovered at each end, in mean edge lengths. Zero on
            a closed guide.

    """
    if boundary not in ("anchor", "exclude", "free"):
        raise ValueError("boundary must be 'anchor', 'exclude' or 'free', got {!r}".format(boundary))

    guide = _as_guide(guide)
    average = mean_edge_length(mesh)
    if tolerance is None:
        tolerance = DEFAULT_TOLERANCE_FACTOR * average
    if polyedges is None:
        polyedges = collect_polyedges(mesh)
    boundary_vertices = _boundary_vertices(mesh)

    info = {
        "reason": "",
        "tolerance": tolerance,
        "max_angle": max_angle,
        "boundary": boundary,
        "alignment": 0.0,
        "candidates": 0,
        "coverage": 0.0,
        "gap_start": 0.0,
        "gap_end": 0.0,
    }

    minimum_cosine = cos(radians(min(90.0, max(0.0, max_angle))))
    best, best_score = None, None
    candidates, near_count, crossing = 0, 0, 0
    for polyedge, closed in polyedges:
        on_boundary = [vertex in boundary_vertices for vertex in polyedge]
        if boundary != "free" and all(on_boundary):
            continue  # this polyedge IS the outline
        directions = _directions(mesh, polyedge, closed)
        keep = []
        for vertex, is_boundary, direction in zip(polyedge, on_boundary, directions):
            if boundary == "exclude" and is_boundary:
                keep.append(False)
                continue
            distance, _, tangent = guide.project(mesh.vertex_coordinates(vertex))
            near = distance is not None and distance <= tolerance
            if near:
                near_count += 1
                if not _parallel(direction, tangent, minimum_cosine):
                    near = False
                    crossing += 1  # near the guide, but no longer going its way
            keep.append(near)
        if not any(keep):
            continue

        for run in _runs(polyedge, keep, closed):
            if len(run) < 2:
                continue  # a single vertex is not a line
            candidates += 1
            run = _trimmed(mesh, guide, _oriented(mesh, guide, run))
            if len(run) < 2:
                continue
            covered = _span(mesh, run, guide, average)["coverage"]
            distances = [guide.project(mesh.vertex_coordinates(v))[0] for v in run]
            # longest first: what is wanted is the line that follows the guide furthest,
            # and only then the one that hugs it closest
            score = (covered, -sum(distances) / len(distances))
            if best_score is None or score > best_score:
                best, best_score = run, score

    info["candidates"] = candidates
    if best is None:
        # name the gate that actually refused it -- both can fire at once, and telling the
        # user to move the guide when the problem is its direction wastes their time
        if not near_count:
            info["reason"] = "no polyedge runs within {:.3g} of this guide".format(tolerance)
        elif crossing:
            info["reason"] = "no polyedge both runs within {:.3g} of this guide and stays within {:.0f} degrees of it".format(tolerance, max_angle)
        else:
            info["reason"] = "no polyedge follows this guide for more than one vertex"
        return [], info

    info["alignment"] = _alignment(mesh, guide, best)
    info.update(_span(mesh, best, guide, average))
    return best, info


# ==============================================================================
# Attaching it
# ==============================================================================


def _boundary_curves(mesh: "QuadMesh") -> dict[int, Polyline]:
    """{vertex: the closed polyline of the boundary loop it is on}.

    By loop membership, not by proximity: a vertex belongs to exactly one boundary, and
    on a mesh with a hole the nearest boundary to a vertex is not always its own.
    """
    curves = {}
    for loop in mesh_boundary_loops(mesh):
        points = [mesh.vertex_coordinates(vertex) for vertex in loop]
        polyline = Polyline(points + points[:1])
        for vertex in loop:
            curves[vertex] = polyline
    return curves


def attach_chain(
    mesh: "QuadMesh",
    chain: list[int],
    guide: "GuideCurve | Iterable[Sequence[float]]",
    hold: str = "fixed",
    boundary_curves: dict[int, Polyline] | None = None,
) -> tuple[dict[int, list[float]], dict[int, Any]]:
    """Where each chain vertex goes and what holds it there. The mesh is not modified.

    A boundary vertex is never moved onto the guide; at most it slides along its outline.

    Parameters
    ----------
    mesh : compas.datastructures.Mesh
        The mesh. NOT modified -- the moves are returned, not applied.
    chain : list[int]
        The vertices to attach.
    guide : GuideCurve | sequence[point]
        The guide to attach them to.
    hold : {'fixed', 'sliding'}, optional
        How an INTERIOR vertex is held once it is on the guide. ``'fixed'`` pins it where
        it lands; ``'sliding'`` re-projects it every iteration, so it may travel along the
        guide but never leave it. A boundary vertex slides on its outline either way.
    boundary_curves : dict, optional
        The result of ``_boundary_curves`` for this mesh, if already built.

    Returns
    -------
    tuple[dict, dict]
        ``{vertex: [x, y, z]}`` -- the vertices to move, and where to. Boundary vertices
        are absent: they do not move.
        ``{vertex: constraint}`` -- what holds each vertex of the chain, ready to be
        merged into the constraints of
        ``compas_singular.datastructures.constrained_smoothing``.
    """
    guide = _as_guide(guide)
    if boundary_curves is None:
        boundary_curves = _boundary_curves(mesh)

    moves, constraints = {}, {}
    for vertex in chain:
        xyz = mesh.vertex_coordinates(vertex)
        outline = boundary_curves.get(vertex)
        if outline is not None:
            constraints[vertex] = outline  # slides along the wall, never off it
            continue
        target = closest_point_on_constraint(guide, xyz)
        moves[vertex] = target
        constraints[vertex] = Point(*target) if hold == "fixed" else guide
    return moves, constraints


# ==============================================================================
# Is it a clean, long polyedge line?
# ==============================================================================


def chain_quality(mesh: "QuadMesh", chain: list[int], guide: "GuideCurve | Iterable[Sequence[float]]") -> dict[str, Any]:
    """Measure a chain against a clean, long polyedge line that follows the guide.

    Parameters
    ----------
    mesh : compas_singular.datastructures.QuadMesh
        The mesh the chain was selected on, unmodified.
    chain : list[int]
        The chain, as returned by ``guide_chain``.
    guide : GuideCurve | sequence[point]
        The guide it was selected for.

    Returns
    -------
    dict
        ``n``
            The number of vertices.
        ``boundary_interior``
            Boundary vertices that are NOT at an end of the chain. Must be 0: a chain
            running along the outline takes the outline with it when it is projected.
        ``boundary_ends``
            Boundary vertices at an end, 0 to 2. These are anchors, and are allowed.
        ``valid_path``
            Whether every consecutive pair is a real edge and no vertex repeats.
        ``purity``
            The share of interior triples that continue straight across the middle
            vertex. 1.0 means the chain IS a piece of a polyedge.
        ``runs``, ``longest_run``
            The number of maximal straight runs and the longest, in vertices.
        ``turn_max``, ``turn_mean``
            The geometric turn off straight at each interior vertex, in degrees. A
            polyedge is topologically straight but not geometrically straight, so these
            are read against the mesh, not against zero.
        ``alignment``
            How closely the chain's edges run along the guide, 0 to 1.
        ``off_max``, ``off_mean``
            Distance from the guide, in mean edge lengths.
        ``coverage``, ``gap_start``, ``gap_end``
            The share of the guide the chain spans, and what is left at each end.
        ``min_face_angle``, ``folded_faces``
            The worst face corner angle among the faces around the chain, and the number
            of faces that turn inside out, after the chain is projected onto the guide on
            a COPY of the mesh. A band selection folds faces; a course must not.
        ``end_drift``
            How far an anchored boundary end moves off the outline when it is projected,
            in mean edge lengths. This is the price of allowing anchors.
    """
    guide = _as_guide(guide)
    average = mean_edge_length(mesh)
    boundary_vertices = _boundary_vertices(mesh)

    quality = {
        "n": len(chain),
        "boundary_interior": 0,
        "boundary_ends": 0,
        "valid_path": False,
        "purity": 0.0,
        "runs": 0,
        "longest_run": 0,
        "turn_max": 0.0,
        "turn_mean": 0.0,
        "alignment": 0.0,
        "off_max": 0.0,
        "off_mean": 0.0,
        "coverage": 0.0,
        "gap_start": 0.0,
        "gap_end": 0.0,
        "min_face_angle": None,
        "folded_faces": 0,
        "end_drift": 0.0,
    }
    if not chain:
        return quality

    ends = set([chain[0], chain[-1]])
    on_boundary = [vertex for vertex in chain if vertex in boundary_vertices]
    quality["boundary_ends"] = len([vertex for vertex in on_boundary if vertex in ends])
    quality["boundary_interior"] = len(on_boundary) - quality["boundary_ends"]

    closed = len(chain) > 2 and chain[0] == chain[-1]
    unique = chain[:-1] if closed else chain
    quality["valid_path"] = len(set(unique)) == len(unique) and all(b in mesh.vertex_neighbors(a) for a, b in zip(chain, chain[1:]))

    distances = [guide.project(mesh.vertex_coordinates(vertex))[0] or 0.0 for vertex in chain]
    quality["off_max"] = max(distances) / average
    quality["off_mean"] = (sum(distances) / len(distances)) / average
    quality["alignment"] = _alignment(mesh, guide, chain)
    quality.update(_span(mesh, chain, guide, average))

    if len(chain) > 2:
        straight_steps, turns, run, runs = 0, [], 1, []
        for a, b, c in zip(chain, chain[1:], chain[2:]):
            try:
                opposite = mesh.vertex_opposite_vertex(a, b)
            except (IndexError, ValueError):
                # the boundary branch of vertex_opposite_vertex indexes a list that is
                # empty at a boundary corner
                opposite = None
            if opposite == c:
                straight_steps += 1
                run += 1
            else:
                runs.append(run + 1)
                run = 1
            xyz_a = mesh.vertex_coordinates(a)
            xyz_b = mesh.vertex_coordinates(b)
            xyz_c = mesh.vertex_coordinates(c)
            turns.append(_angle(_unit([xyz_b[i] - xyz_a[i] for i in range(3)]), _unit([xyz_c[i] - xyz_b[i] for i in range(3)])))
        runs.append(run + 1)
        quality["purity"] = straight_steps / float(len(chain) - 2)
        quality["runs"] = len(runs)
        quality["longest_run"] = max(runs)
        quality["turn_max"] = max(turns)
        quality["turn_mean"] = sum(turns) / len(turns)
    else:
        quality["purity"] = 1.0
        quality["runs"] = 1
        quality["longest_run"] = len(chain)

    quality.update(_projection_damage(mesh, chain, guide, boundary_vertices, ends, average))
    return quality


def _projection_damage(
    mesh: "QuadMesh",
    chain: list[int],
    guide: GuideCurve,
    boundary_vertices: set[int],
    ends: set[int],
    average: float,
) -> dict[str, Any]:
    """What attaching the chain does to the faces around it, measured on a copy."""
    copy = mesh.copy()
    outlines = mesh_boundary_polylines(mesh)

    moves, _ = attach_chain(mesh, chain, guide)
    for vertex, xyz in moves.items():
        copy.vertex_attributes(vertex, "xyz", xyz)

    drift = 0.0
    for vertex in chain:
        if vertex not in boundary_vertices or not outlines:
            continue
        xyz = copy.vertex_coordinates(vertex)
        drift = max(drift, min(_distance(xyz, closest_point_on_constraint(outline, xyz)) for outline in outlines))

    faces = set(face for vertex in chain for face in mesh.vertex_faces(vertex) if face is not None)

    worst, folded = None, 0
    for face in faces:
        before = mesh.face_normal(face, unitized=True)
        after = copy.face_normal(face, unitized=True)
        if sum(before[i] * after[i] for i in range(3)) < 0.0:
            folded += 1
        vertices = copy.face_vertices(face)
        count = len(vertices)
        for index in range(count):
            a = copy.vertex_coordinates(vertices[index - 1])
            b = copy.vertex_coordinates(vertices[index])
            c = copy.vertex_coordinates(vertices[(index + 1) % count])
            angle = _angle(_unit([a[i] - b[i] for i in range(3)]), _unit([c[i] - b[i] for i in range(3)]))
            if worst is None or angle < worst:
                worst = angle

    return {"min_face_angle": worst, "folded_faces": folded, "end_drift": drift / average}
