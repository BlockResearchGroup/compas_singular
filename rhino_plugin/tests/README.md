# Rhino plugin tests

The Rhino half of the plugin -- the `rhino_plugin/` commands and
`compas_singular.rhino` -- exercised **without opening Rhino**: `rhinoscriptsyntax`,
`scriptcontext` and the document helpers are stubbed, everything the library does
is real. The Rhino-free library itself is tested by pytest in `tests/`
(`invoke test`); these are scripts, not pytest tests, and `invoke test` does not
run them.

```
python run_all.py
```

Each file runs in its own process: they install fake `Rhino` and
`rhinoscriptsyntax` modules, and compas decides whether it is inside Rhino by
looking for exactly those.

**Use `run_all.py`, not `python test_x.py`, while Rhino is open.** `rhino_plugin/`
is the library base of `Topology.rhproj`, so Rhino watches this folder, and a
`.pyc` written into `__pycache__` here is an atomic write -- the file event that
has crashed Rhino before. `run_all.py` turns bytecode writing off; a bare run of
one file does not unless you set `PYTHONDONTWRITEBYTECODE=1`.

Run under **both** interpreters. `singular312` is the development env; Rhino 8's
own CPython is the one the commands run on, so a name that resolves there resolves
in Rhino. Its `PYTHONPATH` must include a site-env -- the names change, so list
`C:/Users/Casper/.rhinocode/py39-rh8/site-envs/` first:

```
PYTHONPATH="C:/Users/Casper/.rhinocode/py39-rh8/site-envs/<env>;C:/Users/Casper/libraries/carbcomn/compas_singular/compas_singular/src" \
    C:/Users/Casper/.rhinocode/py39-rh8/python.exe run_all.py
```

| File | What it covers |
|---|---|
| `test_bake_polylines.py` | `helpers.bake_polylines` never raises on geometry Rhino refuses. |
| `test_cmd_test_symmetry.py` | `CMD_test_symmetry` end to end with scripted prompts. |
| `test_coarse_edit_rhino_side.py` | The coarse-edit command's Rhino half and `mesh_ui`. |
| `test_dense_edit_rhino_side.py` | `helpers.mesh_from_rhino` n-gons, and every name `CMD09_edit_quad_mesh` imports. |
| `test_mcp_link_rhino_side.py` | `CMD_mcp_link` -- above all the four gates that keep Rhino usable. |

`_harness.py` and `_cases.py` are shared helpers. Being inside the library base, this folder is also part of what the Rhino project sees. The scene objects
(`compas_singular.rhino.scene`) are checked in
`examples/dev_examples_tests/scene_tests/test_scene_objects.py`, which needs
`compas_rui` and so only runs under Rhino's interpreter.

## What these cannot check

Two things need a real Rhino:

- **Whether `RhinoApp.Idle` fires often enough** for the MCP link. It can go quiet
  when Rhino has nothing to redraw. Attach the link, post 20 requests, and count
  how many drain untouched. If it drops them, swap the pump for a blocking
  `serve()` loop -- `handle()` does not change.
- **Whether the link's gates hold in practice.** Draw a polyline, run
  `CMD04_densities` end to end, and Ctrl+Z your own edit while requests are in
  flight. A `push` arriving mid-command must defer, your selection must survive,
  and the bake must appear in the undo stack as one step named "MCP push".
