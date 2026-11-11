# Overview

`compas_singular` turns a 2D domain into a quad mesh whose topology you control.
There is one pipeline, with two ways to build its first, coarse layout.
Everything downstream of that layout -- editing, densities, densification, smoothing, the dual -- is shared.

## Concepts

Coarse mesh
:   One quad face per topological patch.
    It is cheap to edit and decides the topology of the final mesh:
    where its singularities are, and which way its quads run.

Strip
:   A chain of patches connected through opposite edges.
    Every topological edit adds, removes or divides whole strips, which is what keeps every patch a quad.

Density
:   The number of quads across a strip.
    Because the patches on either side of an edge share the strip, densities set per strip always give a conforming mesh.

Pole
:   A vertex into which several strips converge, like the meridians of a sphere into its pole.
    The quads around it have two corners on the pole: they are pseudo-quads, triangles with the topology of a quad.

Dense mesh
:   The coarse mesh with every patch filled by a grid of quads.

## Two routes to a coarse layout

| | Skeleton route | Frame-field route |
|---|---|---|
| class | [`SkeletonDecomposition`][compas_singular.algorithms.skeleton_decomposition.SkeletonDecomposition] | [`FieldDecomposition`][compas_singular.framefield.field_decomposition.FieldDecomposition] |
| idea | The topological skeleton (medial axis) of a Delaunay triangulation of the domain cuts it into quad patches. | A cross field is solved over the domain. Its singularities become poles, and the separatrices traced out of them cut the domain into quad patches. |
| curve features | Yes. The domain is cut open along the curve, so it becomes a row of mesh edges. | No. Use a guide instead. |
| guides (cables, force lines) | No. A medial axis is equidistant from the walls by definition. | Yes. The field aligns with the guide near it, and the dense mesh follows the field. |
| point features | Yes | Yes |
| symmetry | Yes, by meshing one symmetric unit and expanding it ([Symmetry](../examples/11_symmetry.md)). | Yes, the same way. |
| weak spot | A free end of a curve feature becomes a pole with distorted quads around it. | A domain of uniform curvature loses quality at patch corners. A sampled arc that turns more than 45 degrees between two points reads as a corner to the field. |

Start with the frame-field route.
Use the skeleton route when the mesh must be cut along a curve feature, not only steered by it,
or when you want the medial-axis layout on purpose.

## The eight-step pipeline

Both routes share the same eight steps.
They exist as sections of a Python script and as Rhino commands.

| # | Step | What happens | Python / Rhino |
|---|---|---|---|
| 1 | Settings | Background spacing, target edge length, route options. | keyword arguments / `CMD99_settings` |
| 2 | Input | Outer boundary, holes, point features, curve features and guides. | point lists / `CMD01_boundary_selection` |
| 3 | Coarse layout | Build the patch layout. This is where the route is chosen. | `from_boundary(...).coarse_mesh()` / `CMD02_coarse_mesh` |
| 4 | Edit *(optional)* | Move a corner, divide a strip, add or remove a strip. | [`CoarseEditor`][compas_singular.editing.coarseeditor.CoarseEditor] / `CMD03_edit_coarse_mesh` |
| 5 | Densities | The number of quads across each strip, and the pattern of each patch. | `set_strips_density_target` / `CMD04_densities`, `CMD05_dense_pattern` |
| 6 | Quad mesh | Fill every patch with a grid of quads. | `coarse.densify()` / `CMD06_quad_mesh` |
| 7 | Smoothing *(optional)* | Move vertices to even out the quads, without changing the topology. | [Smoothing](smoothing.md) / `CMD07_smooth` |
| 8 | Dual *(optional)* | Turn the quad mesh into its dual, for example for a block layout. | `dual_mesh(mesh)` / `CMD08_dual` |

[The basic skeleton workflow](../examples/01_skeleton_workflow.md) and
[the basic frame-field workflow](../examples/02_framefield_workflow.md)
run this pipeline on the same domain, one route each.
