"""MCP tools that check inputs before a layout is built (``check_inputs``) and compare two history steps.

Thresholds are measured skeleton-route failure cases; see ``design_notes/mcp.md``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import hypot
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas_singular.mcp.handle import vertex_handle
from compas_singular.mcp.registry import tool

if TYPE_CHECKING:
    from compas_singular.mcp.session import MeshSession


__all__ = []


#: The background spacing's share of the bounding-box diagonal when none is
#: given -- ``SkeletonDecomposition.from_boundary``'s own ``alpha``.
ALPHA = 0.02


def _points(curve: Any) -> list[list[float]]:
    return [list(p) for p in getattr(curve, "points", curve)]


def _open(loop: Any) -> list[list[float]]:
    loop = _points(loop)
    if len(loop) > 1 and loop[0][:2] == loop[-1][:2]:
        loop = loop[:-1]
    return loop


def _segment_distance(p: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    span = dx * dx + dy * dy
    t = 0.0 if span == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / span))
    return hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


def _loop_distance(p: Sequence[float], loop: Sequence[Sequence[float]]) -> float:
    return min(_segment_distance(p, loop[i - 1], loop[i]) for i in range(len(loop)))


def _loop_gap(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> float:
    """Nearest approach of two loops, measured vertex to segment both ways."""
    return min(min(_loop_distance(p, b) for p in a), min(_loop_distance(p, a) for p in b))


def _inside(p: Sequence[float], loop: Sequence[Sequence[float]]) -> bool:
    inside = False
    for i in range(len(loop)):
        (x0, y0), (x1, y1) = loop[i - 1][:2], loop[i][:2]
        if (y0 > p[1]) != (y1 > p[1]) and p[0] < x0 + (p[1] - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _in_domain(p: Sequence[float], outer: Sequence[Sequence[float]], holes: Sequence[Sequence[Sequence[float]]]) -> bool:
    return _inside(p, outer) and not any(_inside(p, h) for h in holes)


@tool(
    "check_inputs",
    "Check the boundary, holes, guides and point features for the ways the "
    "skeleton decomposition fails WITHOUT an error -- run it before "
    "create_coarse_mesh, and again when a layout comes out wrong. Flags, with "
    "the measurement behind each: a point feature close to a wall (the "
    "strongest effect: 2 background spacings from a wall gave a 3.4 degree "
    "corner), a hole close to the outer wall, a hole too small for the "
    "background spacing, a guide that leaves the domain (adds poles nobody "
    "drew), and a point feature outside the domain or inside a hole (silently "
    "ignored). Distances are in background spacings T, the length the domain is "
    "triangulated at -- the same target_length create_coarse_mesh takes. The "
    "remedy is geometric (move the feature, or pass a smaller target_length), "
    "never a density. Changes nothing.",
    properties={
        "target_length": {"type": "number", "description": "The background spacing create_coarse_mesh will use. Default: 4% of the bounding-box diagonal, its own default."},
    },
    read_only=True,
    idempotent=True,
    title="Check the inputs",
)
def _t_check_inputs(session: MeshSession, target_length: float | None = None) -> dict[str, Any]:
    if not session.walls:
        return {"ok": False, "reason": "no boundary is loaded -- rhino_pull (or load_mesh with walls) first"}
    outer = _open(session.walls[0])
    holes = [_open(w) for w in session.walls[1:]]
    xs = [p[0] for loop in [outer] + holes for p in loop]
    ys = [p[1] for loop in [outer] + holes for p in loop]
    diagonal = hypot(max(xs) - min(xs), max(ys) - min(ys))
    T = float(target_length) if target_length else ALPHA * diagonal
    walls = [outer] + holes

    issues: list[dict[str, Any]] = []

    def issue(severity: str, kind: str, at: Sequence[float], message: str, **extra: Any) -> None:
        entry = {"severity": severity, "kind": kind, "at": vertex_handle(at), "message": message}
        entry.update(extra)
        issues.append(entry)

    for p in session.points:
        if not _in_domain(p, outer, holes):
            issue("error", "point_outside", p, "this point feature is outside the domain or inside a hole, and the decomposition IGNORES it -- no pole will be made")
            continue
        d = min(_loop_distance(p, w) for w in walls)
        ratio = d / T
        if ratio < 2.0:
            issue(
                "error",
                "point_near_wall",
                p,
                "this point feature is {:.2f}T from a wall. Measured: 2T gave "
                "a 3.4 degree corner, 1T 0.6 degrees, 0.25T an aspect ratio "
                "of 105. Move it inward, or pass a target_length under "
                "{:.3f}".format(ratio, d / 4.0),
                distance=round(d, 4),
                in_spacings=round(ratio, 2),
            )
        elif ratio < 4.0:
            issue(
                "warning",
                "point_near_wall",
                p,
                "this point feature is {:.2f}T from a wall; at 4T the worst corner measured was already 10.3 degrees".format(ratio),
                distance=round(d, 4),
                in_spacings=round(ratio, 2),
            )

    for index, hole in enumerate(holes):
        gap = _loop_gap(hole, outer)
        ratio = gap / T
        centre = [sum(p[0] for p in hole) / len(hole), sum(p[1] for p in hole) / len(hole), 0.0]
        if ratio < 0.5:
            issue(
                "error",
                "hole_near_wall",
                centre,
                "hole {} comes within {:.2f}T of the outer wall. Measured: 0.25T gave a degenerate element (0 degrees)".format(index, ratio),
                gap=round(gap, 4),
                in_spacings=round(ratio, 2),
            )
        elif ratio < 1.0:
            issue(
                "warning",
                "hole_near_wall",
                centre,
                "hole {} comes within {:.2f}T of the outer wall; aspect ratios of 12-22 were measured between 1T and 0.5T".format(index, ratio),
                gap=round(gap, 4),
                in_spacings=round(ratio, 2),
            )
        radius = max(hypot(p[0] - centre[0], p[1] - centre[1]) for p in hole)
        if radius < T:
            issue(
                "warning",
                "hole_small",
                centre,
                "hole {} has a radius of {:.2f}T -- too small for the background to resolve: aspect 7 was measured at 0.5T, 14 at 0.25T. Pass a smaller target_length".format(
                    index, radius / T
                ),
                radius=round(radius, 4),
                in_spacings=round(radius / T, 2),
            )

    for index, guide in enumerate(session.guides):
        points = _points(guide)
        outside = [p for p in points if not _in_domain(p, outer, holes)]
        if outside:
            issue(
                "error",
                "guide_leaves_domain",
                outside[0],
                "guide {} has {} of {} points outside the domain or inside a "
                "hole. Measured: a guide crossing the outer wall added two "
                "poles nobody drew and a 1.8 degree corner. Trim it to the "
                "domain".format(index, len(outside), len(points)),
            )

    errors = sum(1 for i in issues if i["severity"] == "error")
    return {
        "ok": True,
        "clean": not issues,
        "errors": errors,
        "warnings": len(issues) - errors,
        "target_length": round(T, 4),
        "walls": len(walls),
        "guides": len(session.guides),
        "point_features": len(session.points),
        "issues": issues,
        "reading": (
            "Nothing found that is known to fail quietly."
            if not issues
            else "{} error(s), {} warning(s). The layout will still build -- these are the inputs that make it build WRONG.".format(errors, len(issues) - errors)
        ),
    }


# ==============================================================================
# comparing two steps
# ==============================================================================

_COMPARED = ("min_angle", "max_angle", "aspect_max", "share_below", "faces", "irregular_interior", "poles")
_BETTER = {"min_angle": 1, "max_angle": -1, "aspect_max": -1, "share_below": -1}


def _metrics_at(session: MeshSession, step: Any) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """``(metrics, label, None)`` after ``step``, 'now', or ``(None, None, reason)``."""
    if step in (None, "now"):
        metrics = session.quality()
        if metrics is None:
            return None, None, "no dense mesh is loaded now"
        return metrics, "now", None
    try:
        step = int(step)
    except (TypeError, ValueError):
        return None, None, 'a step is a number from history, or "now"'
    if not 0 <= step < len(session.history):
        return None, None, "there is no step {} -- history has {}".format(step, len(session.history))
    entry = session.history[step]
    if not entry.get("after"):
        return None, None, ("step {} ({}) recorded no dense-mesh metrics -- coarse steps do not".format(step, entry["action"]))
    label = "{} {}{}".format(step, entry["action"], " (undone)" if entry.get("undone") else "")
    return entry["after"], label, None


@tool(
    "compare",
    "Compare the dense mesh's quality after two steps of this session -- "
    "or after one step and now -- metric by metric, with which side is "
    'better. Steps are the numbers in history; "now" is the mesh in hand. '
    "Numbers only: an old state is not kept, so it cannot be drawn. Use it to "
    "judge a sequence of passes, or whether an undone attempt was actually "
    "worse than what replaced it.",
    properties={
        "step_a": {"type": ["integer", "string"], "description": 'A history step, or "now". Default: the first step that recorded metrics.'},
        "step_b": {"type": ["integer", "string"], "description": 'A history step, or "now". Default "now".'},
    },
    read_only=True,
    idempotent=True,
    title="Compare two steps",
)
def _t_compare(session: MeshSession, step_a: Any = None, step_b: Any = "now") -> dict[str, Any]:
    if step_a is None:
        step_a = next((e["step"] for e in session.history if e.get("after")), None)
        if step_a is None:
            return {"ok": False, "reason": "no step has recorded dense-mesh metrics yet"}
    a, label_a, why = _metrics_at(session, step_a)
    if a is None:
        return {"ok": False, "reason": why}
    b, label_b, why = _metrics_at(session, step_b)
    if b is None:
        return {"ok": False, "reason": why}

    rows = {}
    better = {"a": 0, "b": 0}
    for key in _COMPARED:
        va, vb = a.get(key), b.get(key)
        row = {"a": va, "b": vb}
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            row["change"] = round(vb - va, 6)
            sign = _BETTER.get(key)
            if sign and vb != va:
                row["better"] = "b" if (vb - va) * sign > 0 else "a"
                better[row["better"]] += 1
        rows[key] = row
    if rows["faces"]["a"] != rows["faces"]["b"] or rows["irregular_interior"]["a"] != rows["irregular_interior"]["b"]:
        topology = "the topology differs between the two -- a densify or a line edit lies between them, so this compares two different meshes, not one mesh improved"
    else:
        topology = None
    out = {
        "ok": True,
        "a": label_a,
        "b": label_b,
        "metrics": rows,
        "better_on": better,
        "reading": ("{} is better on {} of the four quality measures, {} on {}.".format(label_b, better["b"], label_a, better["a"])),
    }
    if topology:
        out["note"] = topology
    return out
