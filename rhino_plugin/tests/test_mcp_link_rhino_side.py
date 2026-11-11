"""CMD_mcp_link, with Rhino stubbed out. Verification step D.

Run under Rhino 8's own CPython -- the same interpreter the link runs on inside
Rhino, so a name that resolves here resolves there. This catches the class of
bug that only fails at use: a helper imported but never called, or an API that
moved.

The checks that matter most are the GATES, because they are what keeps the link
out of the way of somebody using Rhino: it must not act inside a command, it
must wrap a write in one undo record, and it must give the selection back.
"""
import os
import sys
import types

from _harness import check, dense_mesh, report, spool_dir

spool_dir()

# Import compas BEFORE the Rhino stubs go in. ``compas.plugins`` imports .NET's
# ``System`` when it believes it is inside Rhino, and it decides that by looking
# for a ``Rhino`` module -- so stubbing first makes a real compas import fail on
# a module CPython does not have. Importing here caches it while compas.RHINO is
# still False.
import compas.geometry            # noqa: F401
import compas.plugins             # noqa: F401

# ----------------------------------------------------------------------
# a Rhino that records what was asked of it
# ----------------------------------------------------------------------
CALLS = []


class StubDoc(object):
    def __init__(self):
        self.undo_records = []
        self.open_record = None

    def BeginUndoRecord(self, name):
        self.undo_records.append(name)
        self.open_record = name
        return len(self.undo_records)

    def EndUndoRecord(self, serial):
        self.open_record = None
        return True


class StubRs(types.ModuleType):
    """Just enough rhinoscriptsyntax to drive the link."""

    def __init__(self):
        types.ModuleType.__init__(self, "rhinoscriptsyntax")
        self.layers = {"Outer", "Mesh", "TopologyProblem"}
        self.objects = {}
        self.selected = ["guid-a", "guid-b"]
        self.unselected = False

    def DocumentName(self):
        return "plate.3dm"

    def IsLayer(self, name):
        return name in self.layers

    def AddLayer(self, name=None, parent=None, color=None):
        full = "{}::{}".format(parent, name) if parent else name
        self.layers.add(full)
        CALLS.append(("AddLayer", full))
        return full

    def ObjectsByLayer(self, layer):
        return list(self.objects.get(layer, []))

    def ObjectLayer(self, guid, layer):
        CALLS.append(("ObjectLayer", guid, layer))
        self.objects.setdefault(layer, []).append(guid)
        return layer

    def SelectedObjects(self):
        return list(self.selected)

    def UnselectAllObjects(self):
        self.unselected = True
        CALLS.append(("UnselectAllObjects",))

    def SelectObjects(self, guids):
        CALLS.append(("SelectObjects", tuple(guids)))
        return len(guids)

    def IsObject(self, guid):
        return True

    def AddPoints(self, points):
        guids = ["point-{}".format(i) for i, _ in enumerate(points)]
        CALLS.append(("AddPoints", len(points)))
        return guids

    # -- what the selection pull and the markers ask of Rhino ----------------
    #: guid -> ("mesh" | "curve-closed" | "curve-open" | "point", payload)
    kinds = {}

    def IsMesh(self, guid):
        return self.kinds.get(guid, ("",))[0] == "mesh"

    def IsCurve(self, guid):
        return self.kinds.get(guid, ("",))[0].startswith("curve")

    def IsCurveClosed(self, guid):
        return self.kinds.get(guid, ("",))[0] == "curve-closed"

    def IsPoint(self, guid):
        return self.kinds.get(guid, ("",))[0] == "point"

    def PointCoordinates(self, guid):
        return self.kinds[guid][1]

    def coercemesh(self, guid):
        return self.kinds[guid][1]

    def AddTextDot(self, text, point):
        CALLS.append(("AddTextDot", text, tuple(point)))
        return "dot-{}".format(len(CALLS))

    def CurrentLayer(self, *a, **k):
        raise AssertionError("the link must never change the current layer")

    def ZoomExtents(self, *a, **k):
        raise AssertionError("the link must never change the view")


class StubCommand(object):
    active = 0

    @staticmethod
    def InCommand():
        return StubCommand.active


rhino = types.ModuleType("Rhino")
rhino.Commands = types.SimpleNamespace(Command=StubCommand)


class _Idle(object):
    def __init__(self):
        self.handlers = []

    def __iadd__(self, handler):
        self.handlers.append(handler)
        return self

    def __isub__(self, handler):
        self.handlers.remove(handler)
        return self


rhino.RhinoApp = types.SimpleNamespace(Idle=_Idle())

stub_rs = StubRs()
stub_sc = types.ModuleType("scriptcontext")
stub_sc.doc = StubDoc()
stub_sc.sticky = {}

sys.modules["Rhino"] = rhino
sys.modules["rhinoscriptsyntax"] = stub_rs
sys.modules["scriptcontext"] = stub_sc

# ``compas_rhino.conversions`` genuinely needs RhinoCommon types, and
# ``helpers.py`` imports it at module scope. The link binds the whole of helpers
# at attach time on purpose -- no late imports inside the idle handler -- so the
# test has to satisfy that import. The four names below are all it takes; the
# bake and read functions themselves are replaced where they are exercised.
conversions = types.ModuleType("compas_rhino.conversions")
for _name in ("point_to_rhino", "polyline_to_rhino", "vertices_and_faces_to_rhino",
              "mesh_to_compas", "point_to_compas"):
    setattr(conversions, _name, lambda *a, **k: None)
sys.modules["compas_rhino.conversions"] = conversions

# The link's bootstrap would purge compas_singular out of sys.modules, taking
# the stubs above with it.
sys.compas_singular_keep_modules = True

import CMD_mcp_link as link



class FakeSession(object):
    """Stands in for ``RhinoSession.current()``, which needs a real document.
    Remembers each record and whether an undo record was open at the time."""

    def __init__(self):
        self.coarse = None
        self.recorded = []

    def record(self, name):
        self.recorded.append((name, stub_sc.doc.open_record))


SESSION = FakeSession()

print("\n=== 1. the link imports inside a (stubbed) Rhino ===")
check("CMD_mcp_link imports", link is not None)
check("it binds its dependencies", bool(link._bind()))
check("including the session", "session" in link._BOUND)
link._BOUND["session"] = lambda: SESSION
check("it understands exactly six verbs",
      sorted(link.VERBS) == ["ping", "pull", "pull_coarse", "push", "push_coarse",
                             "push_markers"], sorted(link.VERBS))

print("\n=== 2. it never reads through the DESTRUCTIVE boundary reader ===")
source = open(link.__file__).read()
# The name appears in the comment saying why it is NOT used, so look for a
# CALL and an IMPORT rather than the bare name.
# The destructive reader's name appears in the comment saying why it is NOT
# used, so look for the two ways it could actually be reached: a direct call,
# and the bound-dict key the handler resolves its helpers through.
check("read_boundaries is never called", "read_boundaries(" not in source)
check("and never imported", "import read_boundaries" not in source)
check("and is not in the bound helpers", '"read_boundaries"' not in source)
check("read_boundary_loops is what is bound instead",
      '"read_boundary_loops"' in source)
check("and the reason is written down where the next reader will see it",
      "DELETES" in source)

print("\n=== 3. ping answers, and the spool round-trips through handle() ===")
spool = link._BOUND["spool"]
rid = spool.post("ping")
link.handle(*spool.take())
out = spool.wait(rid, timeout=2)
check("ping is answered", out.get("ok") is True, out)
check("naming the document", out["result"]["document"] == "plate.3dm")

print("\n=== 4. an unknown verb is refused, not crashed on ===")
rid = spool.post("frobnicate")
link.handle(*spool.take())
out = spool.wait(rid, timeout=2)
check("refused", out.get("ok") is False)
check("and says what it does understand",
      all(verb in out["reason"] for verb in link.VERBS), out["reason"])

print("\n=== 5. a verb that raises becomes a refusal carrying the cause ===")
rid = spool.post("push", {"layer": "QuadMesh"})          # no mesh in the payload
link.handle(*spool.take())
out = spool.wait(rid, timeout=2)
check("refused rather than raised", out.get("ok") is False)
check("naming the verb and the error", "push failed in Rhino" in out["reason"],
      out["reason"][:60])

print("\n=== 6. THE GATE: nothing happens while the user is in a command ===")
rid = spool.post("ping")
StubCommand.active = 1
link.on_idle(None, None)
check("the request is still queued", spool.pending() == [rid], spool.pending())
# Probe the response file directly: a timed-out wait would WITHDRAW the request,
# which is the point of it, and leave nothing here to drain.
check("and no answer was written",
      not os.path.exists(os.path.join(spool.spool_directory(),
                                      rid + spool.RESPONSE_SUFFIX)))
StubCommand.active = 0
link.on_idle(None, None)
check("it drains once the command ends", spool.wait(rid, timeout=2).get("ok") is True)

print("\n=== 7. an idle tick with nothing to do only beats ===")
before = len(CALLS)
link.on_idle(None, None)
check("no document calls on an empty queue", len(CALLS) == before, CALLS[before:])
check("but the heartbeat is written", spool.link_state() is not None)
check("naming the document", spool.link_state()["document"] == "plate.3dm")

print("\n=== 8. a push is ONE undo record, and gives the selection back ===")
mesh, _walls = dense_mesh(density=2)
wire = link._BOUND["wire"]
stub_rs.objects["TopologyProblem::QuadMesh::Dense"] = ["old-mesh-guid"]
baked = {}
link._BOUND["bake_mesh"] = lambda m, layer, **kw: baked.setdefault("layer", layer)
link._BOUND["clear_layer"] = lambda layer, **kw: 0
link._BOUND["ensure_layer"] = lambda layer, color=None: layer   # mesh_ui's needs Rhino
CALLS[:] = []
records_before = len(stub_sc.doc.undo_records)

rid = spool.post("push", {"layer": "QuadMesh", "mesh": wire.mesh_to_wire(mesh)})
link.handle(*spool.take())
out = spool.wait(rid, timeout=2)
check("the push succeeds", out.get("ok") is True, out.get("reason"))
check("exactly one undo record was opened",
      len(stub_sc.doc.undo_records) == records_before + 1,
      stub_sc.doc.undo_records)
check("named so a person recognises it in Rhino",
      stub_sc.doc.undo_records[-1] == "MCP push")
check("and it was closed", stub_sc.doc.open_record is None)
check("the previous mesh was moved aside, not deleted",
      ("ObjectLayer", "old-mesh-guid", link.BEFORE_LAYER) in CALLS, CALLS)
check("the tool is told where it went",
      out["result"]["before_layer"] == link.BEFORE_LAYER)
check("the selection was restored",
      ("SelectObjects", ("guid-a", "guid-b")) in CALLS, CALLS)
check("the face count came through",
      out["result"]["faces"] == mesh.number_of_faces())

print("\n=== 8b. push_coarse makes the layout the session's ===")
import compas
from compas_singular.algorithms.skeleton_decomposition import SkeletonDecomposition
L = [[0, 0, 0], [10, 0, 0], [10, 4, 0], [4, 4, 0], [4, 10, 0], [0, 10, 0]]
layout = SkeletonDecomposition.from_boundary(L, point_features=[[2, 2, 0]]).coarse_mesh()
layout.set_strips_density(3)
layout.set_edges_to_curves({edge: [layout.vertex_coordinates(edge[0]), layout.vertex_coordinates(edge[1])]
                            for edge in list(layout.edges())[:2]})
layout.set_shape_polylines([[[0, 0, 0], [1, 1, 0], [2, 0, 0]]])
for name in ("Mesh", "Poles", "Polylines", "EdgeCurves"):
    stub_rs.objects["TopologyProblem::Skeleton::" + name] = ["old-" + name]
SESSION.coarse = "the old layout"
CALLS[:] = []
records_before = len(stub_sc.doc.undo_records)
rid = spool.post("push_coarse", {"coarse": compas.json_dumps(layout)})
link.handle(*spool.take())
out = spool.wait(rid, timeout=2)
check("the push succeeds", out.get("ok") is True, out.get("reason"))
result = out.get("result") or {}
check("ONE undo record, named for what it did",
      len(stub_sc.doc.undo_records) == records_before + 1
      and stub_sc.doc.undo_records[-1] == "MCP push coarse", stub_sc.doc.undo_records)
check("and it was closed", stub_sc.doc.open_record is None)
# Recording DRAWS the layout (RhinoSession.draw, checked in dev_examples_tests/scene_tests); the
# link itself no longer bakes anything.
check("with its pseudo-quads intact",
      len(SESSION.coarse.attributes.get("face_pole") or {}) == len(layout.attributes["face_pole"]))
check("with its shape polylines",
      SESSION.coarse.shape_polylines() == [[[0, 0, 0], [1, 1, 0], [2, 0, 0]]])
check("and the answer counts its poles, shapes and edge curves",
      (result.get("poles"), result.get("polylines"), result.get("edge_curves"))
      == (len(layout.poles()), 1, 2), result)
check("all four old layers were moved aside, not deleted",
      all(("ObjectLayer", "old-" + name, link.COARSE_BEFORE_LAYER) in CALLS
          for name in ("Mesh", "Poles", "Polylines", "EdgeCurves")))
check("the backup layer's leaf is not a name read_coarse looks up",
      link.COARSE_BEFORE_LAYER.split("::")[-1] not in ("Mesh", "Poles"))
check("the layout goes into the session, strips and densities included",
      SESSION.coarse is not layout and SESSION.coarse is not None
      and getattr(SESSION.coarse, "number_of_faces", lambda: -1)() == layout.number_of_faces()
      and SESSION.coarse.get_strip_densities() == layout.get_strip_densities())
check("recorded INSIDE the push's undo record, so one Ctrl+Z takes both back",
      SESSION.recorded[-1:] == [("MCP push coarse", "MCP push coarse")], SESSION.recorded)
check("the selection was restored", ("SelectObjects", ("guid-a", "guid-b")) in CALLS)
rid = spool.post("push_coarse", {})
link.handle(*spool.take())
check("a push with no layout is refused, not crashed on",
      spool.wait(rid, timeout=2).get("ok") is False)

print("\n=== 8c. pull_coarse reads the session's layout and the domain ===")
link._BOUND["read_boundary_loops"] = lambda spacing: (L, [])
stub_rs.objects["Guides"] = ["arc-guide"]
stub_rs.objects["PointFeatures"] = ["pf-1"]
stub_rs.kinds["pf-1"] = ("point", [2.0, 2.0, 0.0])
# A guide Rhino cannot express as a polyline -- an arc. read_polylines skipped
# these; curve_points samples them.
link._BOUND["curve_points"] = lambda guid, spacing: (
    [[5, 5, 0], [6, 5.5, 0], [7, 5, 0]] if guid == "arc-guide" else [])
SESSION.coarse = layout
rid = spool.post("pull_coarse", {"spacing": 0.2})
link.handle(*spool.take())
out = spool.wait(rid, timeout=2)
check("pull_coarse answers", out.get("ok") is True, out.get("reason"))
result = out.get("result") or {}
check("the session's layout as TEXT, exactly as the session holds it",
      result.get("coarse") == compas.json_dumps(layout))
check("the domain, including its point features", result.get("outer") == L
      and result.get("points") == [[2.0, 2.0, 0.0]])
check("and a guide that is an ARC, which the old read dropped",
      result.get("guides") == [[[5, 5, 0], [6, 5.5, 0], [7, 5, 0]]], result.get("guides"))
SESSION.coarse = None
rid = spool.post("pull_coarse", {})
link.handle(*spool.take())
check("no layout in the session is None, not an error",
      spool.wait(rid, timeout=2).get("result", {}).get("coarse") is None)

print("\n=== 8d. pull with selection=true classifies what is selected ===")
square = [[0, 0, 0], [6, 0, 0], [6, 6, 0], [0, 6, 0]]
hole = [[2, 2, 0], [3, 2, 0], [3, 3, 0], [2, 3, 0]]
stub_rs.kinds.update({
    "sel-hole": ("curve-closed", hole), "sel-outer": ("curve-closed", square),
    "sel-guide": ("curve-open", [[1, 1, 0], [5, 1, 0]]),
    "sel-point": ("point", [4.0, 4.0, 0.0]),
    "sel-mesh": ("mesh", "the-mesh")})
link._BOUND["curve_points"] = lambda guid, spacing: stub_rs.kinds[guid][1]
# The link reads a selected mesh with helpers.mesh_from_rhino (it keeps n-gons),
# not compas_rhino's mesh_to_compas.
link._BOUND["mesh_from_rhino"] = lambda geometry, cls=None: (mesh if geometry == "the-mesh" else None)
stub_rs.selected = ["sel-hole", "sel-guide", "sel-outer", "sel-point", "sel-mesh"]
rid = spool.post("pull", {"selection": True})
link.handle(*spool.take())
out = spool.wait(rid, timeout=2)
result = out.get("result") or {}
check("the selection pull answers", out.get("ok") is True, out.get("reason"))
check("the LARGEST closed curve is the outer wall, whatever order it was picked in",
      result.get("outer") == square and result.get("inners") == [hole], result.get("outer"))
check("an open curve is a guide, a point a point feature",
      result.get("guides") == [[[1, 1, 0], [5, 1, 0]]] and result.get("points") == [[4.0, 4.0, 0.0]])
check("and the mesh comes with it", result.get("mesh") is not None
      and len(result["mesh"]["faces"]) == mesh.number_of_faces())
check("it reads the selection without changing it",
      stub_rs.selected == ["sel-hole", "sel-guide", "sel-outer", "sel-point", "sel-mesh"])
stub_rs.selected = ["guid-a", "guid-b"]

print("\n=== 8e. push_markers replaces its own dots, and only those ===")
CALLS[:] = []
records_before = len(stub_sc.doc.undo_records)
rid = spool.post("push_markers", {
    "markers": [{"kind": "worst", "point": [1, 2, 0], "text": "12.3 deg"},
                {"kind": "pole", "point": [3, 4, 0], "text": "pole"}],
    "kinds": ["worst", "singularity", "pole"]})
link.handle(*spool.take())
out = spool.wait(rid, timeout=2)
check("markers are pushed", out.get("ok") is True, out.get("reason"))
check("as one undo record", stub_sc.doc.undo_records[-1] == "MCP markers"
      and len(stub_sc.doc.undo_records) == records_before + 1)
check("each dot carries its text",
      ("AddTextDot", "12.3 deg", (1, 2, 0)) in CALLS and ("AddTextDot", "pole", (3, 4, 0)) in CALLS)
check("on its own kind's layer under MCP::Markers",
      any(c[0] == "ObjectLayer" and c[2] == link.MARKERS_LAYER + "::worst" for c in CALLS))
check("a kind with nothing to mark is still cleared",
      out["result"]["counts"].get("singularity") == 0)
check("and nothing is written outside the markers branch",
      all(c[2].startswith(link.MARKERS_LAYER) for c in CALLS if c[0] == "ObjectLayer"))

print("\n=== 9. it survives another command purging compas_singular ===")
import compas_singular
orphan = types.ModuleType("compas_singular")
sys.modules["compas_singular"] = orphan          # what a command's bootstrap does
check("the link notices the package changed",
      link._BOUND["package"] is not sys.modules["compas_singular"])
sys.modules["compas_singular"] = compas_singular
rebound = link._fresh()
check("and rebinds rather than refusing",
      rebound["package"] is sys.modules["compas_singular"])
rid = spool.post("ping")
link.handle(*spool.take())
check("serving still works afterwards", spool.wait(rid, timeout=2).get("ok") is True)

print("\n=== 10. attach and detach toggle cleanly ===")
check("not attached to begin with", link.attached() is False)
link.attach()
check("attached", link.attached() is True)
check("the idle handler is registered", link.on_idle in rhino.RhinoApp.Idle.handlers)
check("the heartbeat is live", spool.link_state() is not None)
link.detach()
check("detached", link.attached() is False)
check("the handler is gone", link.on_idle not in rhino.RhinoApp.Idle.handlers)
check("and the server is told nothing is listening", spool.link_state() is None)

sys.exit(report("test_mcp_link_rhino_side"))
