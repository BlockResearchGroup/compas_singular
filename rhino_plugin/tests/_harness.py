"""Shared scaffolding for the MCP offline tests: paths, a checker, a mesh.

Run every test under Rhino 8's own CPython -- the same interpreter the server
runs on and the same one the link runs on inside Rhino, so an import or a name
that resolves here resolves there.
"""
import os
import sys
import tempfile

REPO = r"C:\Users\Casper\libraries\carbcomn\compas_singular\compas_singular"
COMMANDS = r"C:\Users\Casper\libraries\carbcomn\compas_singular\compas_singular\rhino_plugin\dev"
for _path in (os.path.join(REPO, "src"), COMMANDS):
    if _path not in sys.path:
        sys.path.insert(0, _path)

FAILURES = []


def check(name, condition, detail=""):
    print("  {} {}{}".format("ok  " if condition else "FAIL", name,
                             (" -- " + str(detail)) if detail else ""))
    if not condition:
        FAILURES.append(name)
    return bool(condition)


def report(title):
    print("\n" + "=" * 62)
    if FAILURES:
        print("{}: {} FAILURE(S): {}".format(title, len(FAILURES),
                                             ", ".join(FAILURES)))
        return 1
    print("{}: all checks passed".format(title))
    return 0


def spool_dir():
    """An isolated spool, so a test never touches the real one."""
    folder = tempfile.mkdtemp(prefix="mcp-spool-")
    os.environ["COMPAS_SINGULAR_MCP_SPOOL"] = folder
    return folder


def library_dir():
    """An isolated library, so saving an example never writes into the repo."""
    folder = tempfile.mkdtemp(prefix="mcp-library-")
    os.environ["COMPAS_SINGULAR_MCP_LIBRARY"] = folder
    return folder


def data_path(name):
    """An example data file, by absolute path.

    Absolute, not relative: run_all.py runs each test from this directory and a
    developer runs one from the repo root, and a relative path cannot be right
    for both.
    """
    return os.path.join(REPO, "tests", "data", name)


def dense_mesh(density=3):
    """A dense quad mesh with something wrong with it, and its boundary loops."""
    from compas_singular.datastructures import CoarseQuadMesh
    from compas_singular.datastructures.mesh.smoothing import mesh_boundary_polylines
    coarse = CoarseQuadMesh.from_json(
        data_path("coarse_quad_mesh_british_museum.json"))
    coarse.collect_strips()
    coarse.set_strips_density(density)
    coarse.densification()
    mesh = coarse.get_quad_mesh()
    walls = [[list(p) for p in pl.points] for pl in mesh_boundary_polylines(mesh)]
    return mesh, walls
