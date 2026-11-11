"""Turn traced separatrices into a network: snap ends together, drop duplicates, find wall split points."""

from __future__ import annotations

from math import pi
from typing import TYPE_CHECKING
from typing import Any

from compas.geometry import angle_vectors
from compas.geometry import distance_point_point
from compas.geometry import subtract_vectors
from compas.itertools import pairwise
from compas_singular.geometry.polyline import closest_on_polyline
from compas_singular.geometry.polyline import loop_arc_lengths
from compas_singular.geometry.polyline import near_loop
from compas_singular.geometry.polyline import point_at_length
from compas_singular.geometry.polyline import polyline_length

if TYPE_CHECKING:
    from compas_singular.framefield.field import CrossField
    from compas_singular.framefield.symmetry import Symmetry
    from compas_singular.framefield.trace import Separatrix


__all__ = ["build_network", "boundary_corners", "SHARP_TURN"]

# The mesh-repair half of this module lives in ``compas_singular.editing.repair``;
# re-exported so old imports keep working. Import from there in anything new.
from compas_singular.editing.repair import densifiable  # noqa: F401,E402
from compas_singular.editing.repair import solve_non_quad_faces  # noqa: F401,E402
from compas_singular.editing.repair import topological_quad_split  # noqa: F401,E402

#: A turn this sharp is a corner, whatever its neighbours do. An angle has no
#: length scale, so no small feature can fall under it; ``15_baseline.py
#: --corners`` checks the margin to every outline in the suite.
SHARP_TURN = pi / 4.0


def boundary_corners(
    loop: list[list[float]],
    limit: float = pi / 12.0,
    spacing: float | None = None,
    sharp: float = SHARP_TURN,
) -> list[int]:
    """Indices of the loop's corners, where the tangent jumps rather than a curve's sampling turning.

    Parameters
    ----------
    loop : list[[x, y, z]]
    limit : float, optional
        Turn above which a vertex is a candidate.
    spacing : float, optional
        Arc-length gap below which two candidates belong to one run. Defaults
        to 1/25 of the loop.
    sharp : float, optional
        Turn above which a vertex is a corner regardless. Defaults to
        ``SHARP_TURN``.

    Returns
    -------
    list[int]
    """
    n = len(loop)
    seg, cum = loop_arc_lengths(loop)
    total = cum[-1]
    if total <= 0.0 or n < 3:
        return []
    gap = spacing if spacing is not None else total / 25.0

    turn = {}  # candidates
    every_turn = {}  # every vertex, for the run-of-one test
    for i in range(n):
        if seg[(i - 1) % n] < 1e-12 or seg[i] < 1e-12:
            continue
        b = loop[i]
        angle = angle_vectors(subtract_vectors(b, loop[(i - 1) % n]), subtract_vectors(loop[(i + 1) % n], b))
        every_turn[i] = angle
        if angle > limit:
            turn[i] = angle

    # sharp corners are held out of the runs entirely, so they cannot set a
    # run's median; compared with a relative epsilon so a regular polygon whose
    # turn lands exactly on ``sharp`` is read the same at every vertex
    threshold = sharp * (1.0 - 1e-12)
    out = [i for i, a in turn.items() if a >= threshold]

    candidates = sorted(i for i, a in turn.items() if a < threshold)
    if not candidates:
        return sorted(out)
    if len(candidates) == 1:
        return sorted(out + candidates)

    def same_run(prev: int, i: int) -> bool:
        return cum[i] - cum[prev] <= gap or i == prev + 1

    runs = [[candidates[0]]]
    for prev, i in pairwise(candidates):
        if same_run(prev, i):
            runs[-1].append(i)
        else:
            runs.append([i])
    wraps = (total - cum[candidates[-1]]) + cum[candidates[0]] <= gap or (candidates[-1] == n - 1 and candidates[0] == 0)
    if len(runs) > 1 and wraps:
        runs[0] = runs.pop() + runs[0]

    for run in runs:
        if len(run) == 1:
            i = run[0]
            here = turn[i]
            before = every_turn.get((i - 1) % n, 0.0)
            after = every_turn.get((i + 1) % n, 0.0)
            if not (abs(here - before) <= limit and abs(here - after) <= limit):
                out.append(i)
            continue
        angles = sorted(turn[i] for i in run)
        typical = angles[len(angles) // 2]
        out.extend(i for i in run if turn[i] - typical > limit)
    return sorted(out)


def _cluster(
    points: list[list[float]],
    tol: float,
    loops: list[list[list[float]]] | None = None,
    wall_tol: float | None = None,
    groups: list[int] | None = None,
    symmetry: Symmetry | None = None,
) -> dict[int, list[float]]:
    """Greedy first-fit grouping of points within ``tol``. Returns ``index -> canonical member point``."""
    wall_tol = tol * 0.25 if wall_tol is None else wall_tol
    flags = [False] * len(points)
    for loop in loops or []:
        flags = [f or near for f, near in zip(flags, near_loop(points, loop, tol * 0.25))]

    canonical = {}
    reps = []
    members = {}
    for i, p in enumerate(points):
        for j, rep in reps:
            if groups is not None and groups[i] in members[j]:
                continue
            limit = wall_tol if (flags[i] and flags[j]) else tol
            if distance_point_point(p, rep) < limit:
                canonical[i] = j
                if groups is not None:
                    members[j].add(groups[i])
                break
        else:
            reps.append((i, p))
            canonical[i] = i
            members[i] = {groups[i]} if groups is not None else set()

    out = {i: points[j] for i, j in canonical.items()}
    if symmetry is not None and symmetry.enabled("network"):
        from compas_singular.framefield.symmetry import project_clusters

        out = project_clusters(points, out, symmetry)
    return out


def _decimate(points: list[list[float]], spacing: float) -> list[list[float]]:
    """Thin a traced polyline to about ``spacing``, keeping both ends."""
    if len(points) < 3:
        return [list(p) for p in points]
    out = [list(points[0])]
    for p in points[1:-1]:
        if distance_point_point(p, out[-1]) >= spacing:
            out.append(list(p))
    if distance_point_point(points[-1], out[-1]) < spacing * 0.5 and len(out) > 1:
        out.pop()
    out.append(list(points[-1]))
    return out


def _resample(points: list[list[float]], n: int) -> list[list[float]]:
    """``n`` points spread evenly by ARC LENGTH along a polyline, ends kept."""
    cum = [0.0]
    for a, b in pairwise(points):
        cum.append(cum[-1] + distance_point_point(a, b))
    total = cum[-1]
    if total == 0.0 or n < 2:
        return [list(points[0]), list(points[-1])]
    out = [point_at_length(points, cum, total * i / float(n - 1)) for i in range(n)]
    out[0], out[-1] = list(points[0]), list(points[-1])
    return out


def _launch_direction(points: list[list[float]], distance: float) -> list[float]:
    """The direction a polyline leaves its first point in, measured to the point
    ``distance`` along it -- a trace is noisiest where it starts."""
    acc = 0.0
    for a, b in pairwise(points):
        acc += distance_point_point(a, b)
        if acc >= distance:
            return subtract_vectors(b, points[0])
    return subtract_vectors(points[-1], points[0])


def _arc_midpoint_of(points: list[list[float]]) -> list[float]:
    """The point half way along a polyline BY LENGTH."""
    return _resample(points, 3)[1]


def _average_polylines(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Mean of two polylines sharing their endpoints, on a common arc-length
    parameter."""
    n = max(len(a), len(b), 3)
    ra, rb = _resample(a, n), _resample(b, n)
    out = [[(p[k] + q[k]) / 2.0 for k in range(3)] for p, q in zip(ra, rb)]
    out[0], out[-1] = list(a[0]), list(a[-1])
    return out


def _wall_splits(
    loop: list[list[float]],
    cuts: list[list[float]],
    corner_limit: float,
) -> list[list[float]]:
    """The points a closed wall is split at, in order round it: its own corners
    and ``cuts``, less duplicates. A wall with fewer than two gets three evenly
    spaced ones added -- a wall split fewer times bounds a degenerate patch.
    """
    marks = [((i, 0.0), list(loop[i])) for i in boundary_corners(loop, corner_limit)]
    for p in cuts:
        i, t, _q, _d = closest_on_polyline(p, loop, closed=True)
        marks.append(((i, t), list(p)))

    if len(marks) < 2:
        n = len(loop)
        for k in range(3):
            i = (k * n) // 3
            marks.append(((i, 0.0), list(loop[i])))

    marks.sort(key=lambda m: m[0])
    unique = [marks[0]]
    for m in marks[1:]:
        if distance_point_point(m[1], unique[-1][1]) > 1e-9:
            unique.append(m)
    if len(unique) > 1 and distance_point_point(unique[0][1], unique[-1][1]) < 1e-9:
        unique.pop()
    return [p for _, p in unique]


def build_network(
    field: CrossField,
    separatrices: list[Separatrix],
    tol: float | None = None,
    corner_limit: float = pi / 12.0,
    spacing: float | None = None,
    singularity_points: dict[int, list[float]] | None = None,
    symmetry: Symmetry | None = None,
) -> tuple[list[list[list[float]]], list[list[list[float]]], dict[str, Any]]:
    """Snap and de-duplicate traced separatrices, and find where walls split.

    Parameters
    ----------
    field : CrossField
    separatrices : list[Separatrix]
    tol : float, optional
        Snapping distance. Defaults to 0.8 x the background spacing.
    corner_limit : float, optional
        Turn above which a wall vertex is a corner candidate.
    spacing : float, optional
        Decimation spacing of traced polylines. Defaults to the background
        spacing.
    singularity_points : dict[int, [x, y, z]], optional
        Where each singularity is -- pass ``Tracer.singularity_positions()``, or
        the positions the traces launched from are lost. Face centroids when
        omitted.
    symmetry : framefield.symmetry.Symmetry, optional
        Projects the end-clustering onto the group. See ``_cluster``.

    Returns
    -------
    (list[polyline], list[polyline], dict)
        Wall splits, separatrices, and a report. Each wall contributes one
        two-point chord per pair of consecutive split points, in order round
        it; ``planar_arrangement`` rebuilds the true arcs from the loops.
    """
    background = field.background
    tol = tol if tol is not None else background.target_length * 0.8
    spacing = spacing if spacing is not None else background.target_length
    loops = background.loops

    if singularity_points:
        centroids = dict(singularity_points)
    else:
        centroids = {fkey: background.mesh.face_centroid(fkey) for fkey, _ in field.singularities()}

    corner_points = []
    for loop in loops:
        for i in boundary_corners(loop, corner_limit):
            corner_points.append(list(loop[i]))

    # -- endpoints: every arm of one singularity has to start at one exact point
    traces = []
    for s in separatrices:
        pts = [list(p) for p in s.points]
        if s.source in centroids:
            pts.insert(0, list(centroids[s.source]))
        elif s.source is None and corner_points:
            # a corner launch starts one inset inside its corner
            nearest = min(corner_points, key=lambda c: distance_point_point(c, pts[0]))
            pts.insert(0, list(nearest))
        if s.sink in centroids:
            pts.append(list(centroids[s.sink]))
        traces.append((s, _decimate(pts, spacing)))

    ends = []
    for _, pts in traces:
        ends.append(pts[0])
        ends.append(pts[-1])
    snapped = _cluster(ends, tol, loops=loops, groups=[k // 2 for k in range(len(ends))], symmetry=symmetry)

    others = []
    landings = []
    for k, (s, pts) in enumerate(traces):
        pts[0] = list(snapped[2 * k])
        pts[-1] = list(snapped[2 * k + 1])
        if distance_point_point(pts[0], pts[-1]) < 1e-9:
            continue  # collapsed to a point
        if s.reason == "boundary":
            # a landing next to a corner lands ON it, or it would split off a
            # sliver arc beside the corner
            if corner_points:
                nearest = min(corner_points, key=lambda c: distance_point_point(c, pts[-1]))
                if distance_point_point(nearest, pts[-1]) < tol:
                    pts[-1] = list(nearest)
            landings.append(pts[-1])
            if s.source == "hole":
                # a hole cut also STARTS on a wall, which it splits too
                landings.append(pts[0])
        others.append(pts)

    # -- duplicates: the same connection traced from both ends -----------------
    unique = []
    for pts in others:
        a, b = pts[0], pts[-1]
        match = None
        for k, kept in enumerate(unique):
            ka, kb = kept[0], kept[-1]
            forward = distance_point_point(a, ka) < tol and distance_point_point(b, kb) < tol
            backward = distance_point_point(a, kb) < tol and distance_point_point(b, ka) < tol
            if not (forward or backward):
                continue
            other = pts if forward else list(reversed(pts))
            # Same ends, and midpoints BY ARC LENGTH within a tolerance scaled to
            # the connection's length (two traces of one connection bow apart
            # in proportion to how far they run)...
            span = 0.5 * (polyline_length(other) + polyline_length(kept))
            if distance_point_point(_arc_midpoint_of(other), _arc_midpoint_of(kept)) >= max(tol, 0.35 * span):
                continue
            # ...and leaving in the same direction: two different ARMS of one
            # singularity can share both ends, but they leave it far apart
            if angle_vectors(_launch_direction(other, tol * 2.5), _launch_direction(kept, tol * 2.5)) > pi / 4.0:
                continue
            match = (k, other)
            break
        if match is None:
            unique.append(pts)
        else:
            k, other = match
            unique[k] = _average_polylines(unique[k], other)
    dropped = len(others) - len(unique)
    others = unique

    # -- walls: split at corners and landings ----------------------------------
    boundary = []
    for loop in loops:
        cuts = [p for p, near in zip(landings, near_loop(landings, loop, tol)) if near]
        splits = _wall_splits(loop, cuts, corner_limit)
        boundary.extend([splits[k], splits[(k + 1) % len(splits)]] for k in range(len(splits)))

    report = {
        "separatrices": len(others),
        "boundary_arcs": len(boundary),
        "landings": len(landings),
        "duplicates_dropped": dropped,
        "snapped": len(ends) - len({(p[0], p[1]) for p in snapped.values()}),
    }
    return boundary, others, report
