"""Keep a symmetric domain's symmetry through background, guides, field, singularities and network.

On by default; ``symmetry=None`` turns it off, and an asymmetric domain is unaffected.
"""

from __future__ import annotations

from math import ceil
from typing import Any

from compas_singular.framefield.background import _jitter
from compas_singular.framefield.background import inside_domain
from compas_singular.framefield.field import CrossField

__all__ = ["Symmetry", "STEPS", "interior_points", "symmetrise", "snap_singularities", "project_clusters", "invariance", "field_invariance", "singularity_orbits"]


#: ``(a, b, c, d, reflects, name)`` -- the linear part as a 2x2 matrix, whether
#: it reverses orientation, and a name. A 90 degree rotation leaves
#: ``exp(i*4*theta)`` unchanged and a reflection in one of the square's axes
#: conjugates it, so the field only needs the flag. **The names are stored in
#: saved fields; do not rename them.**
ELEMENTS = [
    (1, 0, 0, 1, False, "identity"),
    (0, -1, 1, 0, False, "rot90"),
    (-1, 0, 0, -1, False, "rot180"),
    (0, 1, -1, 0, False, "rot270"),
    (-1, 0, 0, 1, True, "mirror-vertical"),
    (1, 0, 0, -1, True, "mirror-horizontal"),
    (0, 1, 1, 0, True, "mirror-diagonal"),
    (0, -1, -1, 0, True, "mirror-antidiagonal"),
]


#: The places the pipeline is told about the symmetry, named so they can be
#: switched off one at a time (``24_symmetry.py`` does).
STEPS = ("background", "guides", "field", "singularities", "network")


class Symmetry(object):
    """A subgroup of the square's 8 symmetries, about a centre.

    Attributes
    ----------
    centre : [x, y, z]
    elements : list[tuple]
        A subset of ``ELEMENTS``, always containing the identity.
    steps : tuple[str]
        Which of ``STEPS`` to apply.
    """

    def __init__(
        self,
        centre: list[float],
        elements: list[tuple[int, int, int, int, bool, str]],
        steps: tuple[str, ...] = STEPS,
    ) -> None:
        self.centre = [float(centre[0]), float(centre[1]), 0.0]
        self.elements = list(elements)
        self.steps = tuple(steps)

    def enabled(self, step: str) -> bool:
        """Whether ``step`` should be applied. Never for a trivial group."""
        return not self.trivial and step in self.steps

    def __len__(self) -> int:
        return len(self.elements)

    def __repr__(self) -> str:
        return "<Symmetry order {} about ({:.3f}, {:.3f}): {}>".format(len(self.elements), self.centre[0], self.centre[1], ", ".join(self.names()))

    def names(self) -> list[str]:
        return [g[5] for g in self.elements]

    @property
    def trivial(self) -> bool:
        return len(self.elements) < 2

    def apply(self, g: tuple[int, int, int, int, bool, str], point: list[float]) -> list[float]:
        """``g`` applied to a point, about ``centre``."""
        a, b, c, d = g[0], g[1], g[2], g[3]
        x, y = point[0] - self.centre[0], point[1] - self.centre[1]
        return [a * x + b * y + self.centre[0], c * x + d * y + self.centre[1], 0.0]

    @staticmethod
    def linear(g: tuple[int, int, int, int, bool, str], vector: list[float]) -> list[float]:
        """``g``'s linear part applied to a free vector."""
        a, b, c, d = g[0], g[1], g[2], g[3]
        return [a * vector[0] + b * vector[1], c * vector[0] + d * vector[1], 0.0]

    @classmethod
    def detect(
        cls,
        loops: list[list[list[float]]],
        centre: list[float] | None = None,
        tol: float = 1e-6,
    ) -> Symmetry:
        """The subgroup mapping the union of ``loops`` (walls, holes and guides) onto itself, as a point set.

        Parameters
        ----------
        loops : list[list[[x, y, z]]]
            EVERY curve the field will see -- outer boundary, holes and guides;
            a symmetric plate with an off-centre cable is not symmetric.
        centre : [x, y, z], optional
            Defaults to the centre of the outer boundary's bounding box, which
            any subgroup of the square's symmetries fixes.
        tol : float, optional
            Point-matching tolerance. Loose, because CAD sampling of a mirrored
            curve is not bit-exact.

        Returns
        -------
        Symmetry
        """
        points = [p for loop in loops for p in loop]
        if not points:
            raise ValueError("no geometry to detect a symmetry from")
        if centre is None:
            xs = [p[0] for p in loops[0]]
            ys = [p[1] for p in loops[0]]
            centre = [(max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0, 0.0]

        self = cls(centre, [ELEMENTS[0]])
        have = {(round(p[0] / tol), round(p[1] / tol)) for p in points}
        found = []
        for g in ELEMENTS:
            ok = True
            for p in points:
                q = self.apply(g, p)
                if (round(q[0] / tol), round(q[1] / tol)) not in have:
                    ok = False
                    break
            if ok:
                found.append(g)
        return cls(centre, found)


# ----------------------------------------------------------------------
# 1 -- interior points invariant under the group
# ----------------------------------------------------------------------


def interior_points(
    symmetry: Symmetry,
    target_length: float,
    outer: list[list[float]],
    inners: Any = (),
    margin: float = 0.45,
    on_axis: bool = True,
) -> list[list[float]]:
    """Interior points as a set exactly invariant under ``symmetry``, one jitter per orbit."""
    centre = symmetry.centre
    half = target_length / 2.0
    parity = 0 if on_axis else 1  # doubled-lattice parity
    limit = margin * target_length

    xs = [p[0] for p in outer]
    ys = [p[1] for p in outer]
    reach = int(ceil(max(max(xs) - centre[0], centre[0] - min(xs), max(ys) - centre[1], centre[1] - min(ys)) / target_length)) + 1

    def point_of(u: int, v: int, jitter: list[float]) -> list[float]:
        return [centre[0] + u * half + jitter[0], centre[1] + v * half + jitter[1], 0.0]

    def act(g: tuple[int, int, int, int, bool, str], u: int, v: int) -> tuple[int, int]:
        """The group acting on DOUBLED lattice indices -- integers, so exact."""
        return (g[0] * u + g[1] * v, g[2] * u + g[3] * v)

    seen_orbits = set()
    candidates = []
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            u, v = 2 * i + parity, 2 * j + parity
            orbit = [act(g, u, v) for g in symmetry.elements]
            rep = min(orbit)
            if rep in seen_orbits:
                continue
            seen_orbits.add(rep)

            # one jitter per orbit, from the representative, projected onto
            # what the representative's stabiliser fixes
            raw = [_jitter(rep[0], rep[1], 0.2 * target_length), _jitter(rep[1], rep[0], 0.2 * target_length), 0.0]
            stabiliser = [g for g in symmetry.elements if act(g, *rep) == rep]
            jitter = [0.0, 0.0, 0.0]
            for g in stabiliser:
                moved = Symmetry.linear(g, raw)
                jitter = [jitter[k] + moved[k] / len(stabiliser) for k in range(3)]
            candidates.append((rep, jitter))

    # the whole orbit is kept or dropped on its representative's verdict
    keep = inside_domain([point_of(rep[0], rep[1], jitter) for rep, jitter in candidates], outer, inners, limit)
    out = []
    emitted = set()
    for (rep, jitter), inside in zip(candidates, keep):
        if not inside:
            continue
        for g in symmetry.elements:
            index = act(g, *rep)
            if index in emitted:
                continue
            emitted.add(index)
            out.append(point_of(index[0], index[1], Symmetry.linear(g, jitter)))
    return out


# ----------------------------------------------------------------------
# 3 -- project the solved field onto the symmetric subspace
# ----------------------------------------------------------------------


def symmetrise(field: CrossField, symmetry: Symmetry, tol: float = 1e-9) -> CrossField:
    """Group-average ``u`` as a new ``CrossField``; follow with ``snap_singularities``."""
    mesh = field.background.mesh
    grid = {}
    for vkey in mesh.vertices():
        p = mesh.vertex_coordinates(vkey)
        grid[(_snap(p[0], tol), _snap(p[1], tol))] = vkey

    values = {}
    orphans = 0
    for vkey in mesh.vertices():
        p = mesh.vertex_coordinates(vkey)
        acc, n = 0j, 0
        for g in symmetry.elements:
            q = symmetry.apply(g, p)
            partner = grid.get((_snap(q[0], tol), _snap(q[1], tol)))
            if partner is None:
                continue
            value = field.u[partner]
            acc += value.conjugate() if g[4] else value
            n += 1
        if n < len(symmetry.elements):
            orphans += 1
        values[vkey] = acc / n if n else field.u[vkey]

    out = CrossField(field.background, values, relaxed=field.relaxed, iterations=field.iterations, residual=field.residual)
    out.symmetry_orphans = orphans
    return out


def _snap(value: float, tol: float) -> int:
    return int(round(value / tol))


# ----------------------------------------------------------------------
# 4 -- take the singularity off the arbitrary face it was reported on
# ----------------------------------------------------------------------


def snap_singularities(
    field: CrossField,
    relative_tol: float = 1e-9,
) -> tuple[CrossField, dict[int, list[float]], dict[str, Any]]:
    """Move each singularity onto the field's own ``|u|`` minimum near it, merging coincident ones.

    Returns ``(field, positions, report)``.

    Returns
    -------
    (CrossField, dict, dict)
        A new field carrying the merged singularity list, the snapped position
        per singular face, and a report.
    """
    mesh = field.background.mesh
    singular = field.singularities()

    positions = {}
    for fkey, _ in singular:
        candidates = set(mesh.face_vertices(fkey))
        for vkey in list(candidates):
            candidates.update(mesh.vertex_neighbors(vkey))
        magnitudes = {v: abs(field.u[v]) for v in candidates}
        low = min(magnitudes.values())
        tied = [v for v, m in magnitudes.items() if m <= low + relative_tol * max(low, 1.0)]
        positions[fkey] = [sum(mesh.vertex_coordinates(v)[k] for v in tied) / len(tied) for k in range(3)]

    merged = {}
    for fkey, index in singular:
        key = tuple(_snap(c, 1e-9) for c in positions[fkey][:2])
        entry = merged.setdefault(key, {"faces": [], "index": 0, "point": positions[fkey]})
        entry["faces"].append(fkey)
        entry["index"] += index

    points, indices = {}, []
    dropped = 0
    for entry in merged.values():
        if entry["index"] == 0:
            dropped += len(entry["faces"])
            continue
        fkey = entry["faces"][0]
        points[fkey] = entry["point"]
        indices.append((fkey, entry["index"]))

    out = CrossField(field.background, dict(field.u), relaxed=field.relaxed, iterations=field.iterations, residual=field.residual)
    out._singularities = indices
    report = {
        "faces_in": len(singular),
        "singularities_out": len(indices),
        "merged": len(singular) - len(indices) - dropped,
        "dropped": dropped,
        "poincare_hopf": out.poincare_hopf(),
    }
    return out, points, report


# ----------------------------------------------------------------------
# 5 -- the end-snapping in separatrix_network.build_network
# ----------------------------------------------------------------------


def project_clusters(
    points: list[list[float]],
    base: dict[int, list[float]],
    symmetry: Symmetry,
    tol: float = 1e-6,
) -> dict[int, list[float]]:
    """Force an ``{index: point}`` clustering to commute with the group, keyed by position."""

    def key(p: list[float]) -> tuple[int, int]:
        return (_snap(p[0], tol), _snap(p[1], tol))

    by_position = {}
    for i, p in enumerate(points):
        by_position.setdefault(key(p), []).append(i)

    place = {}
    representative = {}
    for k, indices in by_position.items():
        place[k] = points[indices[0]]
        chosen = {key(base[i]) for i in indices}
        if len(chosen) == 1:
            representative[k] = base[indices[0]]

    out = dict(base)
    done = set()
    for k in sorted(representative):
        if k in done:
            continue
        orbit = []
        for g in symmetry.elements:
            q = symmetry.apply(g, place[k])
            if key(q) in representative:
                orbit.append((g, key(q)))
        if not orbit:
            continue
        # one member of the orbit decides for all of them
        source_g, source_k = min(orbit, key=lambda pair: pair[1])
        back = _inverse(source_g)
        for g, other in orbit:
            if other in done:
                continue
            moved = symmetry.apply(g, symmetry.apply(back, representative[source_k]))
            for i in by_position[other]:
                out[i] = moved
            done.add(other)
    return out


def _inverse(g: tuple[int, int, int, int, bool, str]) -> tuple[int, int, int, int, bool, str]:
    """The inverse of a signed permutation -- its transpose."""
    for other in ELEMENTS:
        if other[0] * g[0] + other[1] * g[2] == 1 and other[0] * g[1] + other[1] * g[3] == 0 and other[2] * g[0] + other[3] * g[2] == 0 and other[2] * g[1] + other[3] * g[3] == 1:
            return other
    raise ValueError("not an element of the group: {}".format(g))


# ----------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------


def invariance(
    points: list[list[float]],
    symmetry: Symmetry,
    tol: float = 1e-6,
) -> dict[str, tuple[float, float]]:
    """How invariant a point set is, per group element.

    Returns
    -------
    dict[str, (float, float)]
        Name -> ``(share of points whose image is also a point, worst distance
        from an image to its nearest point)``.
    """
    pts = [(p[0], p[1]) for p in points]
    out = {}
    if not pts:
        return out
    have = {(_snap(x, tol), _snap(y, tol)) for x, y in pts}
    for g in symmetry.elements:
        hit = 0
        worst = 0.0
        for p in pts:
            q = symmetry.apply(g, p)
            if (_snap(q[0], tol), _snap(q[1], tol)) in have:
                hit += 1
            worst = max(worst, min(((q[0] - r[0]) ** 2 + (q[1] - r[1]) ** 2) ** 0.5 for r in pts))
        out[g[5]] = (hit / float(len(pts)), worst)
    return out


def field_invariance(field: CrossField, symmetry: Symmetry, tol: float = 1e-9) -> float:
    """``max|u - rho(g)u|`` over vertices, worst over the group. Zero means the
    field itself is symmetric."""
    mesh = field.background.mesh
    grid = {}
    for vkey in mesh.vertices():
        p = mesh.vertex_coordinates(vkey)
        grid[(_snap(p[0], tol), _snap(p[1], tol))] = vkey

    worst = 0.0
    for vkey in mesh.vertices():
        p = mesh.vertex_coordinates(vkey)
        for g in symmetry.elements:
            q = symmetry.apply(g, p)
            partner = grid.get((_snap(q[0], tol), _snap(q[1], tol)))
            if partner is None:
                continue
            value = field.u[partner]
            worst = max(worst, abs(field.u[vkey] - (value.conjugate() if g[4] else value)))
    return worst


def singularity_orbits(
    field: CrossField,
    symmetry: Symmetry,
    points: dict[int, list[float]] | None = None,
    tol: float = 1e-6,
) -> list[list[tuple[int, int]]]:
    """Singularities grouped into orbits; an orbit smaller than the group is unmatched."""
    mesh = field.background.mesh
    sings = field.singularities()
    centroids = dict(points) if points else {f: mesh.face_centroid(f) for f, _ in sings}
    remaining = dict(sings)
    orbits = []
    while remaining:
        fkey = next(iter(remaining))
        index = remaining.pop(fkey)
        orbit = [(fkey, index)]
        for g in symmetry.elements:
            q = symmetry.apply(g, centroids[fkey])
            for other in list(remaining):
                c = centroids[other]
                if abs(c[0] - q[0]) < tol and abs(c[1] - q[1]) < tol:
                    orbit.append((other, remaining.pop(other)))
        orbits.append(orbit)
    return orbits
