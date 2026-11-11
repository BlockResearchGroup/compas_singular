# r: compas_singular
"""Why the symmetry of the drawing was or was not detected. READ-ONLY.

``Symmetry.detect`` answers one bit per element -- match or no match at 1e-6,
with a ``break`` on the first stray point -- so a plate that is symmetric to
the eye and a plate that is not both report ``order 1``. This prints the worst
mismatch DISTANCE for each of the eight elements instead, and names the point
that caused it, which separates the two cases: a deviation of 1e-5 is drawing
precision, a deviation of 3.0 is a genuinely asymmetric input.

Reads exactly what ``CMD_coarse_mesh`` reads -- same layers, same spacing, same
sampler -- so the numbers describe the real solve and not an idealisation.
"""


from compas_singular.rhino.helpers import read_boundaries
from compas_singular.framefield.symmetry import ELEMENTS, Symmetry
from compas_singular.rhino.project import get_settings, resolve_spacing

settings = get_settings()
SPACING = resolve_spacing(settings)

outer, inners, guides, point_features = read_boundaries(spacing=SPACING)

#: Label every point with the loop it came from, so the report can say WHICH
#: curve is the odd one out -- almost always the useful half of the answer.
tagged = [("outer", p) for p in outer]
for i, loop in enumerate(inners):
    tagged += [("hole {}".format(i + 1), p) for p in loop]
for i, guide in enumerate(guides):
    tagged += [("guide {}".format(i + 1), p) for p in guide]
points = [p for _, p in tagged]

#: ``detect``'s own default centre: the OUTER loop's bounding-box centre. Poles
#: are deliberately not in ``tagged`` -- ``from_boundary`` does not pass them to
#: ``detect`` either, so including them here would describe a different test.
xs = [p[0] for p in outer]
ys = [p[1] for p in outer]
centre = [(max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0, 0.0]
symmetry = Symmetry(centre, [ELEMENTS[0]])

#: A spatial hash, because the honest question -- distance to the NEAREST point
#: rather than presence in a bucket -- is O(n^2) brute force and these loops run
#: to thousands of points. Cell is one spacing so a 3x3 neighbourhood always
#: contains the true nearest whenever it is within one cell.
CELL = max(SPACING, 1e-6)
grid = {}
for tag, p in tagged:
    key = (int(p[0] // CELL), int(p[1] // CELL))
    grid.setdefault(key, []).append((tag, p))


def nearest(q):
    """``(distance, tag)`` of the point closest to ``q``, searching outwards."""
    cx, cy = int(q[0] // CELL), int(q[1] // CELL)
    for reach in (1, 2, 4):
        best, best_tag = float("inf"), None
        for i in range(cx - reach, cx + reach + 1):
            for j in range(cy - reach, cy + reach + 1):
                for tag, p in grid.get((i, j), ()):
                    d = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2) ** 0.5
                    if d < best:
                        best, best_tag = d, tag
        if best_tag is not None:
            return best, best_tag
    return float("inf"), None


print("")
print("symmetry report -- {} points, centre ({:.4f}, {:.4f})".format(
    len(points), centre[0], centre[1]))
print("  {} hole(s), {} guide(s), {} pole(s) -- poles are NOT part of the test".format(
    len(inners), len(guides), len(point_features)))
print("")
print("  {:<22} {:>12}  {:>7}  {}".format(
    "element", "worst dev", "matched", "worst point is on"))

TOL = 1e-6
for g in ELEMENTS:
    worst, worst_tag, worst_at, matched = 0.0, None, None, 0
    for tag, p in tagged:
        q = symmetry.apply(g, p)
        d, _ = nearest(q)
        if d <= TOL:
            matched += 1
        if d > worst:
            worst, worst_tag, worst_at = d, tag, p
    share = 100.0 * matched / len(tagged)
    if worst == 0.0:
        note = "exact"
    else:
        note = "{} at ({:.4f}, {:.4f})".format(worst_tag, worst_at[0], worst_at[1])
    print("  {:<22} {:>12.3e}  {:>6.1f}%  {}".format(g[5], worst, share, note))

print("")
detected = Symmetry.detect([outer] + list(inners) + list(guides))
print("  detect() says: {}".format(detected))
print("")
print("  Reading it: a worst deviation under ~1e-3 on an element that still")
print("  fails is DRAWING PRECISION -- the shape is symmetric to the eye and")
print("  not to detect()'s 1e-6. A deviation of the order of the plate itself")
print("  is a genuinely asymmetric input, and the tag says which curve.")
print("")
print("(read-only report -- nothing in the document was changed)")
