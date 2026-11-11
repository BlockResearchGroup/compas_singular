"""Exercise the RHINO half without opening Rhino, by stubbing rhinoscriptsyntax.

Run under Rhino 8's own CPython -- same interpreter as in-app, so an import or
a name that resolves here resolves there. This catches the class of bug that
killed CoarseLayoutObject before: a name bound inside a try/except ImportError
block that only fails at use.
"""
import os
import sys
import types

REPO = r"C:\Users\Casper\libraries\carbcomn\compas_singular\compas_singular"
COMMANDS = r"C:\Users\Casper\libraries\carbcomn\compas_singular\compas_singular\rhino_plugin\commands"
for path in (os.path.join(REPO, "src"),
             COMMANDS):
    if path not in sys.path:
        sys.path.insert(0, path)

FAILURES = []


def check(name, condition, detail=""):
    print("  {} {}{}".format("ok  " if condition else "FAIL", name,
                             (" -- " + str(detail)) if detail else ""))
    if not condition:
        FAILURES.append(name)


# ----------------------------------------------------------------------
print("\n=== 1. the editing core imports with NO Rhino present ===")
try:
    from compas_singular.editing import CoarseEditor
    ok = True
    err = ""
except Exception as e:                                   # noqa: BLE001
    ok, err = False, "{}: {}".format(type(e).__name__, e)
check("compas_singular.editing imports", ok, err)
check("CoarseEditor is a class", ok and isinstance(CoarseEditor, type))

print("\n=== 2. mesh_ui imports without Rhino and says so on use ===")
from compas_singular.rhino import mesh_ui
check("mesh_ui imports outside Rhino", True)
check("mesh_ui knows Rhino is absent", mesh_ui.RHINO is False, mesh_ui.RHINO)
try:
    mesh_ui.ensure_layer("X")
    raised = ""
except RuntimeError as e:
    raised = str(e)
check("it raises a message that names the alternative",
      "compas_singular.editing" in raised, raised[:70])

# ----------------------------------------------------------------------
print("\n=== 4. now WITH a stubbed Rhino, every mesh_ui name resolves ===")


class _Filter(object):
    point = 1
    curve = 4
    mesh = 32


class _StubRs(object):
    """Enough rhinoscriptsyntax to walk the code paths, and no more."""

    filter = _Filter()

    def __init__(self):
        self.layers = set()
        self.objects = {}
        self.meshes = {}
        self.picks = []
        self.messages = []
        self._next = 0
        #: Queue of GetString responses, so a menu answer -- including Escape,
        #: queued as ``None`` -- can be scripted the way GetObject already is.
        self.strings = []

    # layers
    def IsLayer(self, name):
        return name in self.layers

    def AddLayer(self, name=None, parent=None, color=None):
        full = "{}::{}".format(parent, name) if parent else name
        self.layers.add(full)
        return full

    def LayerLocked(self, name, value=None):
        return False

    # objects
    def _add(self, kind):
        self._next += 1
        guid = "{}-{}".format(kind, self._next)
        self.objects[guid] = kind
        return guid

    def AddPoint(self, point):
        return self._add("point")

    def AddLine(self, a, b):
        return self._add("line")

    def AddPolyline(self, points):
        return self._add("polyline")

    def AddMesh(self, vertices, faces):
        guid = self._add("mesh")
        self.meshes[guid] = (list(vertices), list(faces))
        return guid

    def IsObject(self, guid):
        return guid in self.objects

    def IsObjectLocked(self, guid):
        return False

    def UnlockObject(self, guid):
        pass

    def LockObject(self, guid):
        pass

    def ObjectLayer(self, guid, layer=None):
        return layer

    def ObjectColor(self, guid, color=None):
        return color

    def DeleteObjects(self, guids):
        for guid in guids:
            self.objects.pop(guid, None)

    def EnableRedraw(self, value):
        pass

    def MessageBox(self, message, style, title):
        self.messages.append((title, message))
        return 1

    def GetString(self, message, default=None, options=None):
        if self.strings:
            return self.strings.pop(0)
        return default

    #: Queue of guids for GetObject to hand back, so a pick can be simulated.
    def GetObject(self, message=None, filter=None, preselect=False):
        return self.picks.pop(0) if getattr(self, 'picks', None) else None


class _StubSc(object):
    class _Views(object):
        def Redraw(self):
            pass

    class _Doc(object):
        ModelAbsoluteTolerance = 0.001
        Views = None

    def __init__(self):
        self.doc = self._Doc()
        self.doc.Views = self._Views()


class _Point3d(object):
    """Stands in for Rhino.Geometry.Point3d, which is a .NET type.

    It has to be stubbed too: mesh_ui binds it inside the same
    ``try: import ... except ImportError`` block as ``rs``, so replacing only
    ``rs`` leaves it undefined -- which is the exact failure mode this file
    exists to catch, and it caught it.
    """

    def __init__(self, x, y, z=0.0):
        self.X, self.Y, self.Z = x, y, z

    def DistanceTo(self, other):
        return ((self.X - other.X) ** 2 + (self.Y - other.Y) ** 2) ** 0.5


stub_rs = _StubRs()
mesh_ui.rs = stub_rs
mesh_ui.sc = _StubSc()
mesh_ui.Point3d = _Point3d
mesh_ui.RHINO = True

from math import cos, pi, sin                                            # noqa: E402
from compas_singular.framefield.field_decomposition import FieldDecomposition  # noqa: E402

square = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 10.0, 0.0], [0.0, 10.0, 0.0],
          [0.0, 0.0, 0.0]]
circle = [[5.0 + 1.5 * cos(2 * pi * i / 24), 5.0 + 1.5 * sin(2 * pi * i / 24), 0.0]
          for i in range(24)]
circle = circle + circle[:1]

# Drawing and picking the layout moved to the scene objects (2026-09-18):
# examples/dev_examples_tests/scene_tests/test_scene_objects.py checks them, on this same plate.

mesh_ui.refuse("Add line", "nope")
check("refuse goes to a dialog, not a print", stub_rs.messages == [("Add line", "nope")],
      stub_rs.messages)

answer = mesh_ui.ask("next", ["Yes", "No"], "Yes")
check("ask lowercases the answer", answer == "yes", answer)

print("\n=== 7. the main loop's exit branch never commits on Escape or 'exit' ===")
# Mirrors CMD_edit_coarse_mesh.py's own main loop (post-rename, lines 673-697),
# with a stand-in commit() so this stays headless. This is the shape of test
# that would have caught CMD_edit_quad_mesh.py's Escape-saves bug (RHINO_
# COMMAND_EXIT_PLAN.md, S3.1) had it existed for that file's loop too.

d2 = FieldDecomposition.from_boundary(square, [circle], target_length=1.0)
d2.decomposition_mesh()
editor2 = CoarseEditor(d2.mesh, field=d2.get_field(), loops=d2._loops(),
                       polylines=d2.polylines or ())


def run_main_loop(responses):
    stub_rs.strings = list(responses)
    committed = [None]

    def commit():
        committed[0] = editor2.mesh
        return True

    while True:
        operation = mesh_ui.ask(
            "next", ["move_vertices", "add_polyedge", "add_strip", "divide_strip",
                     "remove_strip", "reset", "commit", "exit"], "exit")
        if operation == "commit":
            if commit():
                break
        elif operation == "reset":
            editor2.reset()
        else:
            break
    return committed[0]


# (a) Escape immediately -> mesh_ui.ask returns '', matching nothing -> break.
result = run_main_loop([None])
check("(a) Escape immediately commits nothing", result is None, result)

# (b) one move_vertex, then typed 'exit' -> still nothing committed, but the
# editor is marked dirty -- this is what the optional S2.3 guard would key off.
vkey = next(iter(editor2.mesh.vertices()))
x, y, z = editor2.mesh.vertex_coordinates(vkey)
moved, _notes = editor2.move_vertex(vkey, [x + 0.05, y + 0.05, z])
check("(b) move_vertex succeeds", moved, _notes)
check("(b) move_vertex marks the editor dirty", editor2.edited is True, editor2.edited)
result = run_main_loop(["exit"])
check("(b) typed 'exit' after an edit still commits nothing", result is None, result)

# (c) the same pending edit, now committed -> the edited mesh comes back.
result = run_main_loop(["commit"])
check("(c) 'commit' commits the edited mesh", result is editor2.mesh, result is editor2.mesh)


print("\n" + "=" * 60)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)

print("ALL CHECKS PASSED")
