"""The RHINO half of dense editing, without opening Rhino.

Three things the command depends on and cannot be seen working until it runs:

1. ``helpers.mesh_from_rhino`` rebuilds n-gons from a Rhino mesh's n-gon table
   instead of handing back the triangle fan ``bake_mesh`` stores them as;
2. every name ``CMD09_edit_quad_mesh`` imports exists.

``sync`` -- the drawing kept exact after every kind of dense edit -- moved to the
scene objects with ``PickableMesh`` (2026-09-18) and is checked in
``examples/dev_examples_tests/scene_tests/test_scene_objects.py``.

Run under BOTH interpreters -- Rhino 8's CPython is the one that matters:
    C:/Users/Casper/anaconda3/envs/singular312/python.exe test_dense_edit_rhino_side.py
    PYTHONPATH=<py39 site-envs/default-gdaObJ1D> C:/Users/Casper/.rhinocode/py39-rh8/python.exe test_dense_edit_rhino_side.py
"""
import ast
import importlib
import os
import sys
import types

REPO = r"C:\Users\Casper\libraries\carbcomn\compas_singular\compas_singular"
COMMANDS = os.path.join(REPO, "rhino_plugin", "commands")
for path in (os.path.join(REPO, "src"),):
    if path not in sys.path:
        sys.path.insert(0, path)

FAILURES = []


def check(name, condition, detail=""):
    print("  {} {}{}".format("ok  " if condition else "FAIL", name,
                             (" -- " + str(detail)) if detail else ""))
    if not condition:
        FAILURES.append(name)


# ``helpers`` imports ``compas_rhino.conversions`` at module level, which imports
# the .NET ``Rhino`` assembly. Stand it in, as test_bake_polylines does.
_conv = types.ModuleType("compas_rhino.conversions")
_conv.__getattr__ = lambda name: (lambda *a, **k: a[0] if a else None)
_cr = types.ModuleType("compas_rhino")
_cr.conversions = _conv
_cr.objects = types.ModuleType("compas_rhino.objects")
sys.modules.setdefault("compas_rhino", _cr)
sys.modules.setdefault("compas_rhino.conversions", _conv)
sys.modules.setdefault("compas_rhino.objects", _cr.objects)

import compas                                                      # noqa: E402
compas.RHINO = False
from compas_singular.datastructures import QuadMesh                # noqa: E402
from compas_singular.editing import DenseMeshEditor                # noqa: E402
from compas_singular.rhino import mesh_ui                          # noqa: E402
from compas_singular.rhino import helpers                          # noqa: E402


# ======================================================================
print("\n=== 1. mesh_from_rhino keeps n-gons ===")


class _P(object):
    def __init__(self, x, y, z):
        self.X, self.Y, self.Z = x, y, z


class _Face(object):
    def __init__(self, a, b, c, d=None):
        self.A, self.B, self.C = a, b, c
        self.D = c if d is None else d
        self.IsTriangle = self.C == self.D


class _Table(object):
    def __init__(self, items):
        self.items = list(items)
        self.Count = len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class _Ngon(object):
    def __init__(self, boundary, faces):
        self.boundary, self.faces = boundary, faces

    def BoundaryVertexIndexList(self):
        return list(self.boundary)

    def FaceIndexList(self):
        return list(self.faces)


class _RhinoMesh(object):
    """Built the way ``vertices_and_faces_to_rhino(disjoint=False)`` builds one,
    as measured in Rhino 8: a quad, a pentagon (five triangles round an ADDED
    centroid vertex, grouped by an n-gon record), a triangle and a quad."""

    def __init__(self):
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1.5, 0), (1, 1, 0),
                  (0, 1, 0), (3, 0, 0), (3, 1, 0), (2.5, 2, 0)]
        centroid = (sum(p[0] for p in points[1:6]) / 5.0, sum(p[1] for p in points[1:6]) / 5.0, 0.0)
        self.Vertices = _Table([_P(*p) for p in points + [centroid]])
        faces = [_Face(0, 1, 5, 6)]
        fan = []
        for i, j in [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]:
            fan.append(len(faces))
            faces.append(_Face(i, j, 10))
        faces.append(_Face(2, 7, 3))
        faces.append(_Face(3, 7, 8, 9))
        self.Faces = _Table(faces)
        self.Ngons = _Table([_Ngon([1, 2, 3, 4, 5], fan)])


mesh = helpers.mesh_from_rhino(_RhinoMesh(), cls=QuadMesh)
degrees = sorted(len(mesh.face_vertices(f)) for f in mesh.faces())
check("four faces: a triangle, two quads and the pentagon", degrees == [3, 4, 4, 5], degrees)
check("the fan's centroid vertex is dropped", mesh.number_of_vertices() == 10,
      mesh.number_of_vertices())
check("it comes back as the class asked for", type(mesh) is QuadMesh)
pentagon = next(f for f in mesh.faces() if len(mesh.face_vertices(f)) == 5)
check("the pentagon has its own corners, in order",
      [tuple(mesh.vertex_coordinates(v)[:2]) for v in mesh.face_vertices(pentagon)]
      == [(1, 0), (2, 0), (2, 1), (1, 1.5), (1, 1)])
check("and the mesh is manifold", mesh.is_manifold())

plain = helpers.mesh_from_rhino(_RhinoMesh())
check("with no class it is a plain compas Mesh", type(plain).__name__ == "Mesh")


# ======================================================================
print("\n=== 2. every name CMD09_edit_quad_mesh imports exists ===")

source = open(os.path.join(COMMANDS, "CMD09_edit_quad_mesh.py"), encoding="utf-8").read()
tree = ast.parse(source)
missing = []
for node in tree.body:
    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("compas_singular"):
        module = importlib.import_module(node.module)
        for alias in node.names:
            if not hasattr(module, alias.name):
                missing.append("{}.{}".format(node.module, alias.name))
check("all compas_singular imports resolve", not missing, missing)
check("no import of the deleted _compat module", "_compat" not in source)
editor_calls = set(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                   and isinstance(n.value, ast.Name) and n.value.id == "editor")
absent = sorted(name for name in editor_calls if not hasattr(DenseMeshEditor, name)
                and name not in ("mesh", "walls", "edited", "relax_iterations"))
check("every editor.* the command calls exists", not absent, absent)
ui_calls = set(n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
               and isinstance(n.value, ast.Name) and n.value.id == "mesh_ui")
absent = sorted(name for name in ui_calls if not hasattr(mesh_ui, name))
check("every mesh_ui.* the command calls exists", not absent, absent)


print("\n{} check(s) failed{}".format(
    len(FAILURES), (": " + ", ".join(FAILURES)) if FAILURES else ""))
sys.exit(1 if FAILURES else 0)
