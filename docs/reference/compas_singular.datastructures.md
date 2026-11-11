# datastructures

Mesh data structures for quad meshes, coarse layouts and pseudo-quads, with the strip grammar, smoothing and colouring that act on them.

## Classes

::: compas_singular.datastructures.mesh.mesh.Mesh

::: compas_singular.datastructures.mesh_quad.mesh_quad.QuadMesh

::: compas_singular.datastructures.mesh_quad_coarse.mesh_quad_coarse.CoarseQuadMesh

::: compas_singular.datastructures.mesh_quad_pseudo.mesh_quad_pseudo.PseudoQuadMesh

::: compas_singular.datastructures.mesh_quad_pseudo_coarse.mesh_quad_pseudo_coarse.CoarsePseudoQuadMesh

::: compas_singular.datastructures.skeleton.skeleton.Skeleton

::: compas_singular.datastructures.mesh_quad_coarse.coarse_curves.BoundaryLoop

## Strip grammar

::: compas_singular.datastructures.mesh_quad.grammar.add_strip.add_strip

::: compas_singular.datastructures.mesh_quad.grammar.add_strip.add_strips

::: compas_singular.datastructures.mesh_quad.grammar.add_strip.split_strip

::: compas_singular.datastructures.mesh_quad.grammar.add_strip.split_strips

::: compas_singular.datastructures.mesh_quad.grammar.add_strip.open_added_strip

::: compas_singular.datastructures.mesh_quad.grammar.add_strip.strip_polyedge_update

::: compas_singular.datastructures.mesh_quad.grammar.add_strip.is_polyedge_valid_for_strip_addition

::: compas_singular.datastructures.mesh_quad.grammar.add_strip.polyedge_sides

::: compas_singular.datastructures.mesh_quad.grammar.delete_strip.delete_strip

::: compas_singular.datastructures.mesh_quad.grammar.delete_strip.delete_strips

::: compas_singular.datastructures.mesh_quad.grammar.delete_strip.collateral_strip_deletions

::: compas_singular.datastructures.mesh_quad.grammar.delete_strip.total_boundary_deletions

::: compas_singular.datastructures.mesh_quad.grammar.delete_strip.strips_to_split_to_prevent_boundary_collapse

## Pole grammar

::: compas_singular.datastructures.mesh_quad_pseudo.grammar_poles.split_quad_in_pseudo_quads

::: compas_singular.datastructures.mesh_quad_pseudo.grammar_poles.merge_pseudo_quads_in_quad

## Shape grammar

::: compas_singular.datastructures.mesh_quad.grammar_shape.add_opening

::: compas_singular.datastructures.mesh_quad.grammar_shape.add_handle

## Smoothing

::: compas_singular.datastructures.mesh.smoothing.constrained_smoothing

::: compas_singular.datastructures.mesh.smoothing.boundary_constrained_smoothing

::: compas_singular.datastructures.mesh.smoothing.boundary_smoothing

::: compas_singular.datastructures.mesh.smoothing.region_smoothing

::: compas_singular.datastructures.mesh.smoothing.relaxation

::: compas_singular.datastructures.mesh.smoothing.automated_boundary_constraints

::: compas_singular.datastructures.mesh.smoothing.closest_point_on_constraint

::: compas_singular.datastructures.mesh.smoothing.mesh_boundary_loops

::: compas_singular.datastructures.mesh.smoothing.mesh_boundary_polylines

::: compas_singular.datastructures.mesh.smoothing.mesh_boundary_corners

::: compas_singular.datastructures.mesh.projection.automated_smoothing_surface_constraints

::: compas_singular.datastructures.mesh.projection.automated_smoothing_constraints

::: compas_singular.datastructures.mesh.projection.surface_constrained_smoothing

## Coarse edge curves

::: compas_singular.datastructures.mesh_quad_coarse.coarse_curves.coarse_edges_to_curves

::: compas_singular.datastructures.mesh_quad_coarse.coarse_curves.snap_corners_to_walls

::: compas_singular.datastructures.mesh_quad_coarse.coarse_curves.mean_edge_length

## Coarse mesh from polylines

::: compas_singular.datastructures.mesh_quad_coarse.coarse_network.split_at_corners

::: compas_singular.datastructures.mesh_quad_coarse.coarse_network.split_at_junctions

::: compas_singular.datastructures.mesh_quad_coarse.coarse_network.weld_network

::: compas_singular.datastructures.mesh_quad_coarse.coarse_network.check_network

::: compas_singular.datastructures.mesh_quad_coarse.coarse_network.faces_from_network

::: compas_singular.datastructures.mesh_quad_coarse.coarse_network.check_faces

## Colouring

::: compas_singular.datastructures.mesh.coloring.mesh_vertex_2_coloring

::: compas_singular.datastructures.mesh.coloring.mesh_vertex_n_coloring

::: compas_singular.datastructures.mesh.coloring.mesh_face_2_coloring

::: compas_singular.datastructures.mesh.coloring.mesh_face_n_coloring

::: compas_singular.datastructures.mesh_quad.coloring.quad_mesh_strip_2_coloring

::: compas_singular.datastructures.mesh_quad.coloring.quad_mesh_strip_n_coloring

::: compas_singular.datastructures.mesh_quad.coloring.quad_mesh_polyedge_2_coloring

::: compas_singular.datastructures.mesh_quad.coloring.quad_mesh_polyedge_n_coloring

::: compas_singular.datastructures.mesh_quad_coarse.coloring.dense_quad_mesh_polyedge_2_coloring

## Morphing

::: compas_singular.datastructures.mesh_quad.morphing.fold

::: compas_singular.datastructures.mesh_quad.morphing.fold_vertex_group

## Operations

::: compas_singular.datastructures.mesh.operations.mesh_move_by

::: compas_singular.datastructures.mesh.operations.mesh_move_vertices_by

::: compas_singular.datastructures.mesh.operations.mesh_move_vertex_to

::: compas_singular.datastructures.mesh.operations.is_face_degenerate

::: compas_singular.datastructures.mesh.operations.trimesh_face_circle

::: compas_singular.datastructures.mesh.operations.mesh_weld

::: compas_singular.datastructures.mesh.operations.meshes_join

::: compas_singular.datastructures.mesh.operations.meshes_join_and_weld
