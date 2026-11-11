# Rhino 8

In Rhino, the pipeline is a sequence of commands, numbered in workflow order.
Each reads and writes named layers under `TopologyProblem::`,
and the session state is kept with the document, so it survives between commands and a save and reopen.

The commands are the scripts in `rhino_plugin/commands/`.
They run on Rhino 8's CPython 3.9; see [Installation](../installation.md#rhino-8) to install the package there.

## Commands

| Command | Step | What it does |
|---|---|---|
| `CMD00_start` | 1 | Reset the project: empty the `TopologyProblem` layers and recreate them. The settings stay. |
| `CMD99_settings` | 1 | Show and edit the settings of this document. |
| `CMD01_boundary_selection` | 2 | Pick the outer boundary, holes, point features and guides. |
| `CMD02_coarse_mesh` | 3 | Choose the frame-field or the skeleton route, and build the coarse layout. |
| `CMD02_read_coarse_mesh` | 3 | Read a coarse layout drawn by hand over the domain boundaries, instead of generating one. |
| `CMD03_edit_coarse_mesh` | 4 | Edit the coarse layout by hand: move a corner or a pole, cut a new line along a drawn polyline or arc, unzip a run of corners into a strip, remove a strip, undo. Nothing is written until you commit. |
| `CMD04_densities` | 5 | Pick a strip and set its density, or set a target length for every strip. |
| `CMD05_dense_pattern` | 5 | Set the pattern of each coarse patch: ortho, diagonal or fan. |
| `CMD06_quad_mesh` | 6 | Densify the coarse layout into the quad mesh. |
| `CMD07_smooth` | 7 | Smooth the whole mesh or a region, optionally with guide curves, from one option line. |
| `CMD08_dual` | 8 | Take the dual of a picked quad mesh. |
| `CMD09_edit_quad_mesh` | | Edit the final mesh by hand: move or remove vertices, edges and faces, draw new edges, add or remove a line of quads. The result no longer has to be all quads. Densifying the layout again discards the edit. |
| `CMD99_session_save` / `CMD99_session_open` | | Save the whole project to one JSON file, or open one into this document. |
| `CMD99_mesh_export` | | Export one picked mesh to a JSON file a script can load again. |
| `CMD99_clear` | | Clear chosen layers to start over. |

Densities are matched to strips by geometry, never by strip number,
because a layout read back from Rhino can come back with its strips renumbered.

## Using the commands

### Reading a drawn layout (`CMD02_read_coarse_mesh`)

Draw the outer boundary on `Outer`, holes on `Inner` and the coarse division lines on `Skeleton::EdgeCurves`.
All three layers are read as edges of one network; the layer only says which edges are walls, and so which closed region is a hole.

- A polyline is split at its own corners; a smooth curve (arc, circle, interpolated curve) is one edge.
- Every curve is split where another curve meets it. Snap ends onto the curve they land on (End, Near, Int): an end that stops short is refused as dangling.
- A piece of `EdgeCurves` lying along a wall only marks corners on that wall. A disc as one patch is its circle on `Outer` and four arcs of it on `EdgeCurves`.
- A dangling curve, a division outside the domain or inside a hole, a piece touching nothing else, a patch that is not a quad or a triangle, or two curves between the same corners stop the command with the curve selected. Nothing is repaired; a triangle is kept as a pseudo-quad.
- With no `Outer` curve, the divisions carry the outer boundary themselves.

### Editing the coarse layout (`CMD03_edit_coarse_mesh`)

- **move_vertices**: drag a corner. A corner on the layout boundary stays on the domain wall.
- **move_pole**: pick a highlighted pole, then one of the highlighted corners it may move to. Only a corner all its patches share is allowed.
- **add_polyedge**: draw a polyline or an arc from one layout edge to another. Every edge crossed is split and every patch passed through is split in two; the drawn shape is kept. A run that stops inside is carried on along its strip to the wall.
- **add_strip**: pick a run of existing corners, Enter to add; each corner is unzipped into two and a new strip opens between them. Esc cancels the run.
- **divide_strip**: pick a strip and split it down the middle.
- **remove_strip**: pick a strip; it is deleted at once and the two sides weld together.
- **undo** goes back one edit, **reset** back to the start of the command, **commit** writes the layout back to the session.

Nothing is written until commit. A move keeps the strip densities and patterns; a topology change drops them. A layout that will not densify is refused at commit.
Every operation keeps the layout all-quad, so a line always runs the full width of the layout: from boundary to boundary, or closed on itself.

### Densities and patterns (`CMD04_densities`, `CMD05_dense_pattern`)

`CMD04_densities`: **Pick** a strip and type its number of elements, or set **Target_length** or **Target_density** for every strip at once. Strips are drawn as ribbons shaded by density; **Clear** drops the picks and returns every strip to the target.
The densities are stored on the layout and are lost when the layout is rebuilt; they are then re-set from the target.

`CMD05_dense_pattern`: choose a pattern, then pick patches one at a time (**Pick**), several at once (**Pick_multiple**) or all (**All**). **Finish** keeps them for `CMD06_quad_mesh`.

### Densifying (`CMD06_quad_mesh`)

Each coarse edge is densified along its real shape: walls from the curves on `InputBoundaries`, interior edges from the layout's own polylines.
Read the printed `coarse edges` tally: `chord` counts edges densified as straight lines, and on a curved domain it should be interior edges only.
A field solved for a different outline or different solver settings is ignored, with a message to rerun step 3.
Rerunning this step discards any edit made in `CMD09_edit_quad_mesh`.

### Smoothing (`CMD07_smooth`)

Pick the mesh, answer **Whole** or **Region**, set the options and press Enter:

- **Algorithm**: Area, Centroid, CenterOfMass or ForceDensity. A region is always smoothed by area.
- **Boundary**: Sliding keeps every boundary vertex on its outline with the corners pinned; Fixed pins the boundary; Free holds nothing, so only fixed vertices and guides anchor the mesh.
- **FixedVertices**: vertices that do not move, picked or taken from the points on `PointFeatures`.
- **Guides**: pick a guide curve, or **AllGuides** for every curve on `Guides`, and a chain of mesh vertices along it is proposed. Edit it with Add, Remove and Clear and set its Hold (Fixed or Sliding).

ForceDensity cannot hold a vertex on a curve: a sliding boundary keeps only its corners, and a guide vertex is pinned where it lands.
The result is baked to `QuadMesh::Smoothed::<algorithm>`, or `::Relaxed` for a region.

### Editing the final mesh (`CMD09_edit_quad_mesh`)

The mesh no longer has to be all quads here.

- **move_vertex**: drag a vertex; one on the boundary stays on the wall.
- **remove_vertex**, **remove_edge**, **remove_face**: remove the element (a removed edge merges its two faces).
- **draw_edges**: draw a polyline across the mesh; every edge crossed and face passed through is split. Loose ends inside a face are trimmed.
- **add_line** / **remove_line**: add a strip beside the line through a picked edge, or remove the strip through it. Only where the mesh is still quads.
- **relax** smooths the interior with every boundary held; **undo** and **reset** go back.

Every edit happens at once, and `undo` is where a wrong pick goes. A successful **save** bakes to `QuadMesh::Edited` and ends the command; this session's starting mesh is kept on `QuadMesh::Unedited`.

### Projects and exports (`CMD99_session_save`, `CMD99_session_open`, `CMD99_mesh_export`)

`CMD99_session_save` writes the settings, domain, layout, field and dense mesh to one JSON file, which a plain Python script opens with `SingularSession.load(path)`.
`CMD99_session_open` reads it back; the domain is drawn only into empty input layers. One Ctrl+Z undoes the whole open.
`CMD99_mesh_export` writes one picked mesh to JSON; read it back with `Mesh.load_from_json(path)`. A mesh the session drew keeps its attributes (strips, densities, edge curves).

### Settings (`CMD99_settings`)

The settings are stored in the `.3dm`. `CMD01_boundary_selection`, `CMD04_densities` and `CMD06_quad_mesh` also change single settings through the same store.
