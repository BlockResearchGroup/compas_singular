"""``bake_polylines`` must never raise -- the bug that broke a committed edit.

``rs.AddPolyline`` raises ``Unable to add polyline to document`` on geometry
Rhino refuses. These bakes run AFTER the coarse mesh has been written, so an
exception there leaves the document half updated: the new layout on
``Skeleton::Mesh``, ``Skeleton::Polylines`` emptied by the clear and never
refilled, and CMD_quad_mesh then quietly densifying every edge as a chord.
"""
import os
import sys

REPO = r"C:\Users\Casper\libraries\carbcomn\compas_singular\compas_singular"
for path in (os.path.join(REPO, "src"),):
    if path not in sys.path:
        sys.path.insert(0, path)

FAILURES = []


def check(name, condition, detail=""):
    print("  {} {}{}".format("ok  " if condition else "FAIL", name,
                             (" -- " + str(detail)) if detail else ""))
    if not condition:
        FAILURES.append(name)


# ``helpers`` imports ``compas_rhino.conversions`` at module level, which does
# ``import Rhino`` -- a .NET assembly that exists only inside Rhino. So the
# module cannot be imported anywhere else without standing that in, whichever
# interpreter is used. Only the four names helpers actually pulls are needed.
import types                                          # noqa: E402

_conv = types.ModuleType("compas_rhino.conversions")
_conv.polyline_to_rhino = lambda pl: [list(p) for p in pl]
# Anything else helpers pulls from conversions resolves to a passthrough. Naming
# them one by one meant this file broke every time helpers grew an import -- it
# already did once, when bake_mesh moved to vertices_and_faces_to_rhino. Only
# polyline_to_rhino above is actually exercised here.
_conv.__getattr__ = lambda name: (lambda *a, **k: a[0] if a else None)
_cr = types.ModuleType("compas_rhino")
_cr.conversions = _conv
_cr.objects = types.ModuleType("compas_rhino.objects")
sys.modules.setdefault("compas_rhino", _cr)
sys.modules.setdefault("compas_rhino.conversions", _conv)
sys.modules.setdefault("compas_rhino.objects", _cr.objects)

import compas                                        # noqa: E402
compas.RHINO = False                                 # keep the module importable
from compas_singular.rhino import helpers            # noqa: E402


class _StubRs(object):
    """Refuses what Rhino refuses, and RAISES the way rhinoscript does."""

    def __init__(self, tol=0.001):
        self.tol = tol
        self.layers = set()
        self.objects = {}
        self.added = []

    def IsLayer(self, name):
        return name in self.layers

    def AddLayer(self, name, color=None, parent=None):
        full = "{}::{}".format(parent, name) if parent else name
        self.layers.add(full)
        return full

    def LayerName(self, name, fullpath=True):
        return name

    def ObjectsByLayer(self, name):
        return [g for g, layer in self.objects.items() if layer == name]

    def DeleteObjects(self, guids):
        for guid in guids:
            self.objects.pop(guid, None)

    def ObjectLayer(self, guid, layer=None):
        self.objects[guid] = layer
        return layer

    def AddPolyline(self, points):
        pts = list(points)
        self.added.append(len(pts))
        if len(pts) < 2:
            raise Exception("Unable to add polyline to document")
        for a, b in zip(pts, pts[1:]):
            d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
            if d <= self.tol:
                raise Exception("Unable to add polyline to document")
        guid = "pl-{}".format(len(self.objects) + 1)
        self.objects[guid] = None
        return guid


class _StubSc(object):
    class _Doc(object):
        ModelAbsoluteTolerance = 0.001
    doc = _Doc()


stub = _StubRs()
helpers.rs = stub
helpers.sc = _StubSc()
stub.layers.add("Polylines")

print("\n=== clean_polyline_points ===")
check("drops a duplicate at document tolerance",
      helpers.clean_polyline_points([[0, 0, 0], [0.0001, 0, 0], [1, 0, 0]]) ==
      [[0, 0, 0], [1, 0, 0]])
check("keeps points further apart than the tolerance",
      len(helpers.clean_polyline_points([[0, 0, 0], [0.5, 0, 0], [1, 0, 0]])) == 3)
check("a one-point rib comes back empty",
      helpers.clean_polyline_points([[0, 0, 0]]) == [])
check("an all-coincident rib comes back empty",
      helpers.clean_polyline_points([[0, 0, 0], [0.0001, 0, 0]]) == [])

print("\n=== bake_polylines survives what Rhino refuses ===")
good = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
one_point = [[5, 5, 0]]
coincident = [[3, 3, 0], [3.0002, 3, 0]]
near_dupe_mid = [[0, 1, 0], [0.5, 1, 0], [0.5001, 1, 0], [1, 1, 0]]

try:
    guids, skipped = helpers.bake_polylines(
        [good, one_point, coincident, near_dupe_mid, good], "Polylines")
    raised = None
except Exception as e:                                # noqa: BLE001
    guids, skipped, raised = [], 0, e

check("it did not raise", raised is None, raised)
check("the three bakeable ribs went in", len(guids) == 3, len(guids))
check("the two impossible ribs were skipped", skipped == 2, skipped)
check("the near-duplicate was CLEANED, not skipped",
      4 not in stub.added and 3 in stub.added, stub.added)

print("\n" + "=" * 60)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL CHECKS PASSED")
