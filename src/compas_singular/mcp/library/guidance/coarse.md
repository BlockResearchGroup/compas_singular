# Building a coarse layout

Four tools, in this order, mirror the sequence
`CoarseQuadMesh.densification` documents at its own call site:

```
decomposition = SkeletonDecomposition.from_boundary(...)
coarse = decomposition.coarse_mesh()
coarse.set_strips_density_target(t=0.5)
dense = coarse.densification()
```

## 0. `check_inputs`

Run it first. The skeleton route never raises on a bad input; it builds a bad
layout. Measured on a 10 by 10 square at the default background spacing T:

| input | what came out |
|---|---|
| point feature 2T / 1T from a wall | min angle 3.4 / 0.6 degrees |
| hole 0.25T from the outer wall | a degenerate (0 degree) element |
| hole of radius 0.25T | aspect ratio 14 |
| guide crossing the outer wall | two poles nobody drew |
| point feature outside, or in a hole | silently ignored |

The remedy is geometric -- move the feature, or pass a smaller `target_length`
-- never a density.

## 1. `create_coarse_mesh`

Needs a boundary already loaded -- `rhino_pull`, or `load_mesh` with walls.
Builds the coarse layout from the domain's own outer and inner boundaries,
line features and point features, via a topological-skeleton decomposition.
**This is not a read of hand-drawn patch curves** -- the layout is derived,
not traced. Point features become the poles of the layout.

Replaces any coarse layout already held, and clears the coarse undo history
(`coarse_undo`).

## 2. `coarse_inspect`

Read-only. Every strip's `skey`, density and one representative edge as two
vertex handles; every patch with a handle inside it (`at`), its pattern and the
strips crossing it. **Neither a strip nor a patch has a handle of its own** --
read this before addressing one in `coarse_set_density`, `coarse_set_pattern`
or a strip edit.

**Call it with `image=true` before editing strips.** Each strip is drawn in
its own colour with its `skey` written on its representative edge. A strip is
a band running the whole width of the layout, and which edges belong to it is
not guessable from coordinates. The picture also shows what the numbers
cannot: a layout that cuts a corner of the wall, or a pole in the wrong place.

## 3. `coarse_set_density`

One tool, a selector: set every strip's density at once, one strip by
`skey`, several by `skeys`, a target edge length, or a target total face
count. See the tool description for the five shapes. Undo with `coarse_undo`.

## 3b. `coarse_set_pattern` -- optional

`ortho` (the default grid), `diagonal` (both patch diagonals cut through the
grid) or `fan` (four polar fans meeting at the centre), for all patches, the
patches along one strip, or patches picked by handle.

**A pattern costs density.** Diagonal and fan need both strips crossing a patch
at the same EVEN density, and a strip runs through other patches too, so whole
groups of strips are raised together. `density_raised` reports it. Set
patterns before your final densities, or expect a later odd density to be
raised again at `coarse_densify`.

**A pattern costs quality numbers, on purpose.** Measured on an L-shaped
domain at density 4:

| pattern  | min angle | max angle | verdict  |
|----------|-----------|-----------|----------|
| ortho    | 49.5      | 135.0     | usable   |
| diagonal | 16.7      | 130.5     | poor     |
| fan      | 8.5       | 170.2     | unusable |

Those angles are the pattern's own triangles and fan poles. Smoothing cannot
remove them; it only distorts the patches around them. Do not chase them.

## 4. `coarse_densify`

Generates the dense quad mesh and loads it as the working mesh -- what
`inspect`/`relax`/`smooth_*` then operate on. This clears the DENSE mesh's
undo history; the coarse layout and its own undo are untouched, so going back
to `coarse_set_density` and densifying again costs nothing.

Curved boundaries keep their shape automatically, using the walls
`create_coarse_mesh` recorded (or the edge shapes a `coarse_save` file
carries). Patterns are honoured; an all-ortho layout densifies bit-for-bit as
before patterns existed. No frame field is solved: a patch interior is blended
from its own four sides.

# Keeping a layout

`coarse_save` writes the layout with its walls, guides, point features and
edge shapes; `coarse_load` reads it back, densities and patterns included, and
densifies to the same mesh. Loading a file whose domain differs from the
session's replaces the domain and drops the dense mesh.

# Editing the coarse layout

Three edits, all strip-shaped: `coarse_add_strip`, `coarse_remove_strip` and
`coarse_divide`. The first two keep every strip's key and density, because
neither renumbers the layout.

**`coarse_divide` renumbers everything.** It splits a band of patches in two,
either a strip by `skey` at parameter `t`, or along a drawn line or arc
(`points`). A cut must cross each patch through OPPOSITE sides and cannot
cross a pole patch; its ends are snapped onto the nearest coarse edge, and one
that stops short is carried along its strip to the wall. The commit welds and
renumbers, so:

- densities are carried by position: a divided strip's density is shared
  between its two halves (4 at the middle is 2 and 2, which keeps the element
  size), every other strip keeps its own. `from_skey` in the result maps new
  keys to old;
- patterns are carried by which old patch a new one lies in;
- a drawn arc and the halves of any curved edge it crossed are kept as edge
  shapes, survive later edits and `coarse_save`, and the dense mesh follows
  them;
- **re-read `coarse_inspect` before using any skey again.**

A cut drawn right beside an existing line is legal and makes a sliver strip;
`warning`/`pinches` in the result say where. Look at the picture.

## Moving a corner

`coarse_move_corner` changes geometry only: keys, densities and patterns stay.
An interior corner goes exactly where it is told; a boundary corner is
projected back onto its own wall and can only slide along it. Refused: a drag
over a neighbouring patch, one that turns a patch inside out, and a boundary
corner that would land on a different wall. The curved edges at the corner are
warped onto its new position (`shapes_warped`). Without that, the decomposition
only moves a curve's END onto the new corner -- measured, a 0.18 nudge left a
24 degree kink there -- and past a quarter of an edge it gives up and hands back
a straight chord; warped, the edge keeps its traced shape at any distance.

Moving a corner that sits ON a corner of the domain is allowed, and warned
about: the patch then wraps round the kink, which densifies into a flattened
element.

**Plan a removal before making it.** There is no cheap test for whether a
strip can go without breaking the layout -- `coarse_plan_strip_deletion`
performs the deletion on a copy and reports what it would cost: the face
count after, how many OTHER strips would go with it (`collateral`), and
whether a hole in the domain would collapse (`boundaries_lost`). `ok: false`
there means `coarse_remove_strip` will refuse for the same reason.

Every coarse edit -- density, pattern, add, remove, divide, move corner --
snapshots the coarse layout first. `coarse_undo` restores it --
independent of `undo`, which only ever touches the dense mesh.

## Order

1. `create_coarse_mesh` -- once, from the boundary (or `coarse_load`).
2. `coarse_inspect image=true` -- look at the strips before touching anything.
3. `coarse_add_strip` / `coarse_remove_strip` / `coarse_divide`, if the layout
   itself needs to change; `coarse_move_corner` to reshape a patch. `coarse_plan_strip_deletion` first for a removal;
   `coarse_inspect` again after a divide.
4. `coarse_set_pattern`, if any patch should not be a plain grid.
5. `coarse_set_density` -- as many times as needed; each call is independent.
6. `coarse_densify` -- then continue with `inspect` and the smoothers.
7. `coarse_save`, if the layout is worth keeping.
8. `rhino_pull_coarse` brings it back after editing it in Rhino, with its
   strips, densities, patterns and edge shapes, exactly as the document's
   session holds it.
9. `rhino_push_coarse`, to carry on in Rhino: it writes the layout into the
   document's session, where `CMD04_densities` / `CMD06_quad_mesh` /
   `CMD03_edit_coarse_mesh` read one, and draws it on
   `TopologyProblem::Skeleton::{Mesh, Poles, Polylines, EdgeCurves}`. Look at
   it with `coarse_inspect image=true` first; it refuses otherwise.
