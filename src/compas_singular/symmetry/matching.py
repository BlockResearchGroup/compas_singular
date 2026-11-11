"""Match the corners of two rotation seams so copies of the unit weld exactly.

Align corners by distance from the centre, split strips for unpaired ones, then slide and snap.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from copy import deepcopy
from math import hypot
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

from compas_singular.symmetry._geometry import signed_area
from compas_singular.symmetry.replicate import seam_membership

if TYPE_CHECKING:
    from compas_singular.symmetry.unit import SymmetricUnit


__all__ = ["match_rotation_seams"]


def _move(unit: SymmetricUnit, vkey: int, point: Sequence[float]) -> None:
    """Move a corner and bring its edge shapes along; seam edges are re-made straight."""
    unit.vertex_attributes(vkey, "xyz", [point[0], point[1], 0.0])
    curves = unit.edges_to_curves()
    if not curves:
        return
    seams = unit.seams
    eps = unit.eps
    changed = False
    for (u, v), pts in curves.items():
        if vkey not in (u, v) or not pts:
            continue
        pu, pv = unit.vertex_coordinates(u), unit.vertex_coordinates(v)
        if any(s.locate(pu, eps) is not None and s.locate(pv, eps) is not None for s in seams):
            curves[(u, v)] = [list(pu), list(pv)]
        elif u == vkey:
            pts[0] = [point[0], point[1], 0.0]
        else:
            pts[-1] = [point[0], point[1], 0.0]
        changed = True
    if changed:
        unit.set_edges_to_curves(curves)


def _is_junction(unit: SymmetricUnit, p: Sequence[float], tol: float) -> bool:
    return any(abs(p[0] - q[0]) <= tol and abs(p[1] - q[1]) <= tol for q in unit.symmetry.get("junctions", []))


def _move_junction(
    unit: SymmetricUnit,
    old: Sequence[float],
    new: Sequence[float],
    old_t: float,
    seam_name: str,
    tol: float | None = None,
) -> None:
    """Update ``junctions`` and ``seam_runs`` for a junction moved from ``old`` to ``new``."""
    tol = max(10 * unit.eps, 1e-9) if tol is None else tol
    data = unit.symmetry
    for i, q in enumerate(data.get("junctions", [])):
        if abs(q[0] - old[0]) <= tol and abs(q[1] - old[1]) <= tol:
            data["junctions"][i] = [new[0], new[1], 0.0]
    seam = dict((s.name, s) for s in unit.seams)[seam_name]
    new_t = seam.locate(new, max(tol, 1e-6))
    if new_t is None:
        return
    for run in data.get("seam_runs", {}).get(seam_name, []):
        for k in (0, 1):
            if abs(run[k] - old_t) <= max(tol, 1e-6):
                run[k] = new_t


def _align(
    on_a: list[tuple[float, int]],
    on_b: list[tuple[float, int]],
    junction: dict[int, bool],
    forbidden: set[tuple[str, int]],
    big: float,
) -> tuple[list[tuple[tuple[float, int], tuple[float, int]]], list[tuple[float, int]], list[tuple[float, int]]] | None:
    """Order-preserving alignment of two sorted ``(t, vkey)`` lists.

    Returns ``(pairs, unpaired_a, unpaired_b)``, or ``None`` if no alignment
    respects the junction and ``forbidden`` constraints.
    """
    m, n = len(on_a), len(on_b)
    inf = float("inf")
    cost = [[inf] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    cost[0][0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            here = cost[i][j]
            if here == inf:
                continue
            if i < m and j < n and junction[on_a[i][1]] == junction[on_b[j][1]]:
                c = here + abs(on_a[i][0] - on_b[j][0])
                if c < cost[i + 1][j + 1]:
                    cost[i + 1][j + 1], back[i + 1][j + 1] = c, ("pair", i, j)
            if i < m and not junction[on_a[i][1]] and ("A", on_a[i][1]) not in forbidden:
                c = here + big
                if c < cost[i + 1][j]:
                    cost[i + 1][j], back[i + 1][j] = c, ("skip_a", i, j)
            if j < n and not junction[on_b[j][1]] and ("B", on_b[j][1]) not in forbidden:
                c = here + big
                if c < cost[i][j + 1]:
                    cost[i][j + 1], back[i][j + 1] = c, ("skip_b", i, j)
    if cost[m][n] == inf:
        return None
    pairs, unpaired_a, unpaired_b = [], [], []
    i, j = m, n
    while i or j:
        step, pi, pj = back[i][j]
        if step == "pair":
            pairs.append((on_a[pi], on_b[pj]))
        elif step == "skip_a":
            unpaired_a.append(on_a[pi])
        else:
            unpaired_b.append(on_b[pj])
        i, j = pi, pj
    return pairs[::-1], unpaired_a[::-1], unpaired_b[::-1]


def _face_signs(unit: SymmetricUnit) -> dict[int, bool]:
    signs = {}
    for f in unit.faces():
        pts = [unit.vertex_coordinates(v) for v in unit.face_vertices(f)]
        signs[f] = signed_area(pts) > 0
    return signs


def match_rotation_seams(unit: SymmetricUnit, max_splits: int = 8) -> dict[str, int]:
    """Make the two rotation seams of ``unit`` agree. In place.

    Returns
    -------
    dict
        ``{'slid': n, 'snapped': n, 'splits': n}``.
    """
    seams = unit.seams
    if not seams or seams[0].kind != "rotation":
        return {"slid": 0, "snapped": 0, "splits": 0}
    group = unit.group
    rotation = group.element(seams[0].element)
    eps = unit.eps
    junction_tol = max(10 * eps, 1e-9)
    stats = {"slid": 0, "snapped": 0, "splits": 0}
    forbidden = set()
    xs = [unit.vertex_coordinates(v)[0] for v in unit.vertices()]
    ys = [unit.vertex_coordinates(v)[1] for v in unit.vertices()]
    diagonal = hypot(max(xs) - min(xs), max(ys) - min(ys)) if xs else 1.0

    while True:
        membership = seam_membership(unit, seams, eps)
        on_a = sorted((t, v) for v, here in membership.items() for name, t in here if name == "A")
        on_b = sorted((t, v) for v, here in membership.items() for name, t in here if name == "B")
        junction = dict((v, _is_junction(unit, unit.vertex_coordinates(v), junction_tol)) for _, v in on_a + on_b)
        aligned = _align(on_a, on_b, junction, forbidden, big=10.0 * (diagonal or 1.0))
        if aligned is None:
            raise ValueError(
                "the rotation seams cannot be matched: the corners on seam A {} and seam B {} have "
                "no order-preserving pairing, and the splits that would add the missing ones were "
                "refused".format([round(t, 3) for t, _ in on_a], [round(t, 3) for t, _ in on_b])
            )
        pairs, unpaired_a, unpaired_b = aligned
        if not unpaired_a and not unpaired_b:
            break
        if stats["splits"] >= max_splits:
            raise ValueError("rotation seams still unmatched after {} splits".format(max_splits))
        if unpaired_b:
            t, v = unpaired_b[0]
            side, target = "B", "A"
        else:
            t, v = unpaired_a[0]
            side, target = "A", "B"
        try:
            _split_across(unit, target, t)
            stats["splits"] += 1
        except ValueError:
            forbidden.add((side, v))

    before = _face_signs(unit)
    for (ta, va), (tb, vb) in pairs:
        if va == vb:
            continue
        pa = unit.vertex_coordinates(va)
        fixed_a = junction[va]
        fixed_b = junction[vb]
        t = ta if (fixed_a or fixed_b) else 0.5 * (ta + tb)
        if abs(t - ta) > 1e-12 or abs(t - tb) > 1e-12:
            stats["slid"] += 1
        new_a = pa if fixed_a else seams[0].point_at(t)
        if not fixed_a:
            _move(unit, va, new_a)
        new_b = group.apply(rotation, new_a)
        if fixed_b:
            # A junction on B that was not exactly the rotation of its partner
            # -- a sampled curve meets the two seams at slightly different
            # radii -- moves with it, and so do the records that name it.
            _move_junction(unit, unit.vertex_coordinates(vb), new_b, tb, seams[1].name)
        _move(unit, vb, new_b)
        stats["snapped"] += 1
    after = _face_signs(unit)
    flipped = [f for f in before if before[f] != after.get(f)]
    if flipped:
        raise ValueError(
            "matching the rotation seams would turn {} patch(es) over -- the two seams disagree too much for this layout; try another seam angle".format(len(flipped))
        )

    unit.symmetry["matching_cuts"] = unit.symmetry.get("matching_cuts", 0) + stats["splits"]
    unit.symmetry["matching"] = stats
    return stats


def _split_across(unit: SymmetricUnit, name: str, t: float) -> None:
    """Give seam ``name`` a corner at distance ``t`` by splitting the strip there."""
    from compas_singular.editing import CoarseEditor

    seam = dict((s.name, s) for s in unit.seams)[name]
    eps = unit.eps
    probe = unit.copy()
    probe.collect_strips()
    target = seam.point_at(t)
    found = None
    for skey in probe.strips():
        for u, v in probe.strip_edges(skey):
            if u == v:
                continue
            pu, pv = probe.vertex_coordinates(u), probe.vertex_coordinates(v)
            tu, tv = seam.locate(pu, eps), seam.locate(pv, eps)
            if tu is None or tv is None or not (min(tu, tv) + eps < t < max(tu, tv) - eps):
                continue
            length = hypot(pv[0] - pu[0], pv[1] - pu[1])
            found = (skey, hypot(target[0] - pu[0], target[1] - pu[1]) / length)
            break
        if found:
            break
    if found is None:
        raise ValueError("seam {} has no edge spanning t={:.4f} to split for a matching corner".format(name, t))

    data = deepcopy(unit.symmetry)
    field, decomposition = unit.field, unit.decomposition
    walls = _walls(unit)
    editor = CoarseEditor(unit, loops=walls)
    ok = editor.divide(skey=found[0], t=found[1])
    if isinstance(ok, tuple):
        ok = ok[0]
    if not ok:
        raise ValueError("could not split the strip crossing seam {} at t={:.4f}: {}".format(name, t, editor.last_reason))
    layout, notes = editor.commit()
    if layout is None:
        raise ValueError("could not commit the seam-matching split: {}".format(editor.last_reason))
    unit.attributes["symmetry"] = data
    mapping, _tally = editor.edge_curves(loops=walls)
    if mapping:
        unit.set_edges_to_curves(mapping)
    unit.field, unit.decomposition = field, decomposition


def _walls(unit: SymmetricUnit) -> list[list[Any]]:
    """The unit's walls as loops, read back off its own boundary."""
    curves = unit.edges_to_curves()
    loops = []
    for loop in unit.vertices_on_boundaries():
        pts = []
        for i in range(len(loop) - 1):
            u, v = loop[i], loop[i + 1]
            shape = curves.get((u, v)) or (list(reversed(curves[(v, u)])) if (v, u) in curves else None)
            if shape:
                pts.extend(shape[:-1])
            else:
                pts.append(unit.vertex_coordinates(u))
        loops.append(pts)
    return loops
