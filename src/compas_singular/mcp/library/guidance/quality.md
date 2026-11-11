# What a good quad mesh looks like

Quality here is element quality: the shape of the individual quads, measured by
`framefield.quality.mesh_quality`. It is not the same question as whether the
layout is right, and a mesh can score well on all of this while having its
singularities in the wrong place.

## The four numbers

`min_angle` and `max_angle`
: The smallest and largest corner angle anywhere in the mesh, in degrees. A
  perfect grid reads 90 and 90. These are **worst-case over the whole mesh**, so
  one bad face sets both. That is deliberate: a single sliver is a real defect.

`aspect_max`
: Longest edge over shortest edge, per face, worst over the mesh. A perfect grid
  reads 1.00. It is `inf` when a face has a zero-length edge, which is
  degenerate rather than merely poor.

`share_below`
: The fraction of all corners below the low-angle line (20 degrees by default).
  **This is the number that says whether a problem is local or general**, and so
  it is the one that decides between `smooth_region` and a global pass. A bad
  `min_angle` with `share_below` at zero is one face. A bad `min_angle` with
  `share_below` at 5% is the whole mesh.

## Bands

Set in `thresholds.json`, and editable. As shipped:

| | good | usable | poor |
|---|---|---|---|
| min angle | >= 45 | >= 25 | >= 15 |
| max angle | <= 135 | <= 155 | <= 168 |
| aspect | <= 2.0 | <= 4.0 | <= 8.0 |

A verdict is the **worst** band of any metric, not an average. A mesh that is
good on three and unusable on one is unusable.

## What is not a defect

**A high aspect ratio is not automatically wrong.** A layout densified at
different strip densities is graded on purpose -- elements stretch where one
strip is denser than the one crossing it. An aspect of 3 spread evenly across a
mesh that was asked to grade is the mesh doing its job. An aspect of 3 in one strip next to a strip at 1.1 is a defect.
Read `aspect_max` together with where it is, not on its own.

**An irregular vertex is not a defect at all.** `irregular_interior` counts
interior vertices whose valency is not 4, and a quad mesh over anything but a
rectangle must have some -- the count is fixed by the topology of the domain,
not by the quality of the meshing. Poles are the same. Neither number should be
"improved"; they are reported so a change in them is noticed, because a change
means the topology moved.

**Smoothing cannot fix a layout.** If the singularities are badly placed, the
elements around them will stay poor no matter how much relaxation is applied.
When two smoothing passes in a row change nothing, the honest report is that the
layout is the limit, not that the mesh cannot be improved.
