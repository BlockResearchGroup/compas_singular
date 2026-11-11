# Choosing a smoother

Three tools, and they are not interchangeable.

## `relax` -- try this first

Wraps `framefield.relax.relax_mesh`. It walks a schedule of settings
gentlest-first -- (20, 0.3), then (50, 0.5), then (100, 0.5) -- and **keeps a
result only if min angle, max angle and max aspect all improve together**. A
setting that trades one against another is rejected and the next is tried.

Because of that gate it is the safest thing to run, and it is the right first
move on almost any pulled mesh.

**`accepted: null` is a pass, not a failure.** It means no setting in the
schedule improved all three at once, so the mesh was left exactly as it was.
Nothing was broken and nothing was wasted. Do not retry it with different
arguments hoping for a different answer -- reach for a local pass instead, or
report that the layout is the limit.

**Leave `pin_singularities` off.** Pinning the irregular vertices is what stops
this pass working: they are precisely the vertices that need to move for the
elements around them to open up. It is exposed only because there are meshes
where a singularity is deliberately placed and must not drift.

`seams='free'` is the default and the only setting measured to help. `'slide'`
never hurts and never helps -- sliding along a seam cannot repair an angle that
the seam's own shape causes. `'fixed'` relaxes patch interiors only.

## `smooth_boundary_constrained` -- the workhorse

Interior vertices relax freely, boundary vertices **slide along the walls**, and
kink vertices stay pinned. This is the one to reach for when the mesh is
generally slack rather than locally broken.

It needs walls. Pulled from Rhino it has them. Without them it falls back to the
mesh's own boundary polylines, which means the boundary can only get smoother,
never truer -- a faceted boundary stays faceted.

`damping` is 0.5 by default. Raising `kmax` costs time linearly and has
diminishing returns past a few hundred. Unlike `relax`, **this pass has no gate**
-- it applies what it is told and can make a mesh worse. Snapshot first, measure
after, undo if it did not help.

## `smooth_region` -- for one defect

Area-weighted, and tapered over `blend` rings so the region blends into the rest
of the mesh instead of leaving a crease at its edge. Use it when `share_below` is
zero or near it and `min_angle` is bad: that combination means one face, and a
global pass would move the whole mesh to fix it.

`{"kind": "worst_faces", "count": 3}` is usually the right region. Do not set
`blend` to 0 -- a fully relaxed region against a fixed ring is a kink, which is
the kind of defect this is meant to remove.

Area-weighted rather than centroid on purpose: centroid smoothing equalises edge
lengths, which fights the grading of a mesh densified at different strip
densities.

## `smooth_guides` -- when the mesh has guide curves

Attaches the longest run of one polyedge that already follows each guide,
moves it onto the guide, and smooths with those vertices (plus the boundary)
held. The chain is CHOSEN, not built up vertex by vertex or by a radius round
the curve -- both of those were tried and both fold faces or zigzag. Every
guide is selected before any is attached, so the result does not depend on
guide order.

A boundary vertex is never moved onto a guide, only slid along its own wall --
attaching it would take the wall with it. **No gate**, same as
`smooth_boundary_constrained`: snapshot first, check `all_improved`.

Refuses cleanly if `session.guides` is empty -- nothing was pulled with a
guide curve on it, or none was loaded. Widen `tolerance_factor` or `max_angle`
if a guide gets nothing; both are explained in the tool description.

## `relax_fdm` -- force-density relaxation

Wraps `datastructures.mesh.smoothing.relaxation`. Fixed vertices held,
everything else finds a minimal-tension shape, boundary edges weighted
`q_factor` times heavier than interior ones.

**Two real gaps, not polish left for later.** The underlying function
overwrites its own `constraints` argument with an empty list (and uses zero
loads) before the solve runs, and its `algorithm` argument is accepted but
never read. Neither
is exposed as a tool parameter for exactly that reason -- there would be
nothing behind it. Do not reach for this tool expecting to steer it beyond the
`fixed` set; it is closer to a specialised alternative to `relax` than a
general constrained solve. Needs `compas_fd` installed; a missing package is a
clean refusal, not a traceback.

No gate, same as the other ungated passes here -- snapshot first, check
`all_improved`.

## `dense_add_line` / `dense_remove_line` -- when smoothing is not the answer

When the layout itself is the limit -- a band of elements too wide or too many
-- no smoother fixes it. These change the dense mesh's topology: a whole strip
is added beside, or removed through, the polyedge of an edge you name.
`dense_plan_line_removal` first. Three limits:

- **refused per strip, not per mesh** -- a strip stops at any face that is not a
  quad, so a line whose strip crosses or reaches a pole's fan (or a polygon) is
  refused; every other line on the same mesh is allowed;
- **lost on re-densifying** -- if the change can be made on the coarse layout,
  make it there;
- `dense_add_line` opens the new strip with plain centroid smoothing, and on the
  test L the verdict after it was "unusable" -- follow it with `relax` or
  `smooth_boundary_constrained`, and `compare` the result.

`undo` takes a line edit back like any other step.

## Order

1. `inspect` -- read `share_below` before choosing.
2. `relax` -- cheap, gated, usually enough.
3. If still poor and the problem is general: `smooth_boundary_constrained`.
4. If still poor and the problem is local: `smooth_region` on the worst faces.
5. If the mesh has guide curves it should follow: `smooth_guides`.
6. `relax_fdm` is a different kind of pass, not a rung on this ladder -- reach
   for it deliberately, not as a fallback when the others plateau.
7. `inspect` again, and `undo` anything that did not help.

Two passes that change nothing mean the layout is the limit. Say so.
