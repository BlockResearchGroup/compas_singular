# Smoothing and quality

A dense mesh straight from `densify()` is made of patches that were each filled on their own,
so the quads can kink where patches meet.
Smoothing moves vertices to even the quads out, without changing the topology.

## Measuring quality

Quality here is the shape of the individual quads, not whether the layout is right.
[`mesh_quality`][compas_singular.framefield.quality.mesh_quality] returns, among others:

`min_angle` / `max_angle`
:   The worst corner angle anywhere. 90 degrees is a perfect quad.

`aspect_max`
:   The longest over the shortest edge, of the worst face.

`share_below`
:   The share of corners below `low_angle`.
    It tells one bad face (near zero) from a generally slack mesh.

```python
from compas_singular.framefield import mesh_quality

print(mesh_quality(dense))
```

A high aspect ratio is not a defect in itself: a graded mesh has stretched quads on purpose.
Neither is an irregular vertex: a quad mesh over anything but a rectangle must have some.

## Smoothing functions

Try them in this order.

1. [`relax_mesh`][compas_singular.framefield.relax.relax_mesh] -- the safe first move.
   It tries a schedule of settings, gentlest first, with the outline sliding along itself,
   and keeps a result only if no quality measure gets worse.
   If nothing changes, the layout is the limit, not the smoothing.
2. [`boundary_constrained_smoothing`][compas_singular.datastructures.mesh.smoothing.boundary_constrained_smoothing] -- for a mesh that is slack overall.
   Interior vertices move freely, boundary vertices slide along the given curves, and corners stay put.
   There is no gate, so measure before and after.
3. [`region_smoothing`][compas_singular.datastructures.mesh.smoothing.region_smoothing] -- for one bad spot.
   Only the given vertices move, and the smoothing fades out over a few rings around them, so it blends in.
4. [`GuideCurve`][compas_singular.editing.guide_chain.GuideCurve] and [`attach_chain`][compas_singular.editing.guide_chain.attach_chain] --
   for a mesh that should hug guide curves.
   A chain of mesh vertices that already follows the guide is attached to it and held while the rest is smoothed.
5. [`relaxation`][compas_singular.datastructures.mesh.smoothing.relaxation] -- force-density relaxation with `compas_fd`,
   with stiffer boundary edges. Needs the `fd` extra.

```python
from compas.geometry import Polyline
from compas_singular.datastructures import boundary_constrained_smoothing
from compas_singular.datastructures import region_smoothing

boundary_constrained_smoothing(dense, curves=[Polyline(outer + outer[:1]), Polyline(hole + hole[:1])], kmax=50)
region_smoothing(dense, region, kmax=50)
```

[Smoothing the quad mesh](../examples/06_quad_mesh_smoothing.md) compares a fixed boundary, a sliding boundary and a region on the same mesh.

!!! note

    Do not smooth a mesh whose purpose is that its lines meet a boundary or a force line at right angles,
    such as a block-coursing mesh.
    Global smoothing has been measured to take those angles from about 90 down to 45-65 degrees,
    while barely improving the quads.

When the layout itself is the limit, change the topology instead:
edit the coarse layout ([Editing the coarse quad mesh](../examples/03_coarse_mesh_editing.md)),
or add and remove lines on the dense mesh ([Editing the dense quad mesh](../examples/07_quad_mesh_editing.md)).
A dense-mesh edit is lost when the coarse layout is densified again.
