# MCP server

The meshing tools are also available as a Model Context Protocol server,
so an MCP client can drive them from natural language.
The client supplies the model; this package supplies the mesh operations.

```bash
compas-singular-mcp
```

or `python -m compas_singular.mcp`.

## Tools

The server offers typed tools for each stage of the pipeline, among others:

- building a coarse layout from a boundary: `check_inputs`, `create_coarse_mesh`;
- editing it: `coarse_add_strip`, `coarse_remove_strip`, `coarse_divide`, `coarse_move_corner`,
  `coarse_set_density`, `coarse_set_pattern`, `coarse_densify`;
- improving a dense mesh: `relax`, `smooth_boundary_constrained`, `smooth_region`, `smooth_guides`,
  `relax_fdm`, `dense_add_line`, `dense_remove_line`;
- measuring and comparing: `inspect`, `coarse_inspect`, `compare`, `history`, `undo`;
- exchanging with an open Rhino document: `rhino_pull`, `rhino_push` and their `_coarse` counterparts.

Run `check_inputs` first: the skeleton route never raises on a bad input, it builds a bad layout.
`inspect` and `coarse_inspect` can return an image, for what the numbers cannot say.

The server builds layouts on the skeleton route only.
A frame-field layout reaches it from Rhino, through `CMD02_coarse_mesh` and `rhino_pull_coarse`.
The smoothing tools follow the order in [Smoothing and quality](smoothing.md).
