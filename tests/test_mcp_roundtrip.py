"""Pull, improve, push -- the whole MCP loop, with a thread playing ``CMD_mcp_link``.

It covers ``tools_rhino``'s SUCCESS path, which is otherwise only ever seen
refusing. The mesh work in between is real; only the document is fake.
"""

import math
import threading
import time

import compas
import pytest
from conftest import make_dense_mesh

from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.mcp.bridge import spool
from compas_singular.mcp.bridge import wire
from compas_singular.mcp.handle import vertex_handle
from compas_singular.mcp.tools_rhino import _float32_collapsed

L_WALL = [[0, 0, 0], [10, 0, 0], [10, 4, 0], [4, 4, 0], [4, 10, 0], [0, 10, 0]]
# The scripted cut below was drawn on the layout the old default (alpha = 0.04)
# gives, so the spacing is pinned to it.
L_SPACING = 0.04 * math.hypot(10, 10)
#: What the fake document holds: its domain and its current selection.
DOMAIN = {"outer": L_WALL, "points": [[2, 2, 0]]}
SELECTION = {"outer": [[0, 0, 0], [6, 0, 0], [6, 6, 0], [0, 6, 0]], "points": [[3, 3, 0]]}
ARC = [[7 + 0.5 * math.sin(math.pi * i / 20), 4.0 * i / 20, 0] for i in range(21)]


class FakeLink(object):
    """What CMD_mcp_link does, minus Rhino."""

    def __init__(self, mesh, walls):
        self.mesh = mesh
        self.walls = walls
        self.pushed = {}
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            job = spool.take()
            if job is None:
                spool.heartbeat("fake.3dm")
                time.sleep(0.01)
                continue
            self.serve(*job)

    def serve(self, request_id, verb, args):
        if verb == "ping":
            spool.answer(request_id, True, result={"document": "fake.3dm"})
        elif verb == "pull" and not args.get("selection"):
            spool.answer(
                request_id,
                True,
                result={"document": "fake.3dm", "layer": args.get("layer"), "outer": self.walls[0], "inners": self.walls[1:], "guides": [], "mesh": wire.mesh_to_wire(self.mesh)},
            )
        elif verb == "pull":
            spool.answer(
                request_id,
                True,
                result={"document": "fake.3dm", "layer": "(selection)", "outer": SELECTION["outer"], "inners": [], "guides": [], "points": SELECTION["points"], "mesh": None},
            )
        elif verb == "pull_coarse":
            # the document holds whatever was last pushed
            sent = self.pushed.get("coarse") or {}
            spool.answer(
                request_id, True, result={"document": "fake.3dm", "outer": DOMAIN["outer"], "inners": [], "guides": [], "points": DOMAIN["points"], "coarse": sent.get("coarse")}
            )
        elif verb == "push_markers":
            self.pushed["markers"] = args
            spool.answer(request_id, True, result={"document": "fake.3dm", "layer": "TopologyProblem::MCP::Markers", "undo_record": "MCP markers"})
        elif verb == "push_coarse":
            self.pushed["coarse"] = args
            pushed = compas.json_loads(args["coarse"])
            spool.answer(
                request_id,
                True,
                result={
                    "document": "fake.3dm",
                    "layers": {"mesh": "TopologyProblem::Skeleton::Mesh"},
                    "poles": len(pushed.poles()),
                    "polylines": len(pushed.shape_polylines()),
                    "edge_curves": len(pushed.edges_to_curves()),
                    "before_layer": "TopologyProblem::Skeleton::MCP::Before",
                    "undo_record": "MCP push coarse",
                },
            )
        elif verb == "push":
            self.pushed["mesh"] = args["mesh"]
            spool.answer(
                request_id, True, result={"document": "fake.3dm", "layer": args.get("layer"), "before_layer": "TopologyProblem::QuadMesh::MCP::Before", "undo_record": "MCP push"}
            )
        else:
            spool.answer(request_id, False, error="unknown verb")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        spool.clear_link()


@pytest.fixture
def link(mcp_spool):
    mesh, walls = make_dense_mesh(density=4)
    fake = FakeLink(mesh, walls)
    fake.thread.start()
    time.sleep(0.1)
    yield fake
    fake.close()


def _pushed_coarse(mcp, link):
    """Build a layout with a drawn arc and a pattern, look at it, and push it."""
    mcp.session.adopt(None, walls=[L_WALL], guides=[], points=[[2, 2, 0]])
    mcp("create_coarse_mesh", target_length=L_SPACING)
    mcp("coarse_set_density", density={"kind": "all", "value": 3})
    cut = mcp("coarse_divide", points=ARC)
    assert cut.get("ok") is True, cut.get("reason")
    mcp("coarse_set_pattern", pattern="diagonal", faces={"kind": "strip", "skey": 0})
    mcp("coarse_inspect", image=True)
    out = mcp("rhino_push_coarse")
    assert out.get("ok") is True, out.get("reason")
    return out, compas.json_loads(link.pushed["coarse"]["coarse"])


def test_pull_brings_back_the_mesh_and_the_walls(mcp, link):
    out = mcp("rhino_pull", layer="QuadMesh")
    assert out.get("ok") is True, out.get("reason")
    assert out["faces"] == link.mesh.number_of_faces()
    assert out["walls"] == len(link.walls)
    assert "warning" not in out
    assert len(out["reading"]) > 40
    assert len(mcp.session.walls) == len(link.walls)
    first = list(mcp.session.mesh.vertices())[0]
    assert mcp.session.mesh.vertex_coordinates(first) == link.mesh.vertex_coordinates(list(link.mesh.vertices())[0])


def test_improve_and_push_the_mesh_back(mcp, link):
    mcp("rhino_pull", layer="QuadMesh")
    start = mcp.session.triple()
    out = mcp("relax")
    assert out.get("ok") is True, out.get("reason")
    end = mcp.session.triple()
    if out["accepted"] is not None:
        assert out["all_improved"] is True
        assert end[0] > start[0]
    else:
        assert start == end
    assert mcp("rhino_push", layer="QuadMesh").get("ok") is False  # nothing has looked at it
    looked = mcp("inspect", image=True)
    assert looked["inputs"]["walls"] == len(link.walls)
    out = mcp("rhino_push", layer="QuadMesh")
    assert out.get("ok") is True, out.get("reason")
    assert "Before" in (out.get("kept_previous_on") or "")
    assert out.get("undo_record") == "MCP push"
    assert len(link.pushed["mesh"]["faces"]) == link.mesh.number_of_faces()
    # the IMPROVED positions, not the pulled ones
    assert link.pushed["mesh"]["vertices"] == wire.mesh_to_wire(mcp.session.mesh)["vertices"]
    if end != start:
        assert link.pushed["mesh"]["vertices"] != wire.mesh_to_wire(link.mesh)["vertices"]
    text = mcp.handler.read_resource("session://current")
    assert "QuadMesh" in text
    assert "`relax`" in text and "`rhino_push`" in text


def test_push_coarse_needs_a_layout_and_a_look(mcp, link):
    mcp.session.adopt(None, walls=[L_WALL], guides=[], points=[[2, 2, 0]])
    assert mcp("rhino_push_coarse").get("ok") is False
    mcp("create_coarse_mesh")
    out = mcp("rhino_push_coarse")
    assert out.get("ok") is False
    assert (out.get("fix") or {}).get("tool") == "coarse_inspect"


def test_push_coarse_sends_the_layout_as_the_cmd_commands_will_read_it(mcp, link):
    out, sent = _pushed_coarse(mcp, link)
    coarse = mcp.session.coarse
    assert sent.number_of_faces() == coarse.number_of_faces()
    assert len(sent.poles()) == len(coarse.poles())
    assert len(sent.shape_polylines()) == coarse.number_of_edges()
    # the drawn arc is among the shape polylines CMD_quad_mesh will match
    assert any(any(math.dist(p[:2], [7.5, 2.0]) < 0.05 for p in curve) for curve in sent.shape_polylines())
    assert sent.get_strip_densities() == coarse.get_strip_densities()
    assert sent.attributes["dense_pattern"] == coarse.attributes["dense_pattern"]
    assert sent.attributes.get("quad_mesh") is None
    assert out.get("undo_record") == "MCP push coarse"
    corner = coarse.vertex_coordinates(next(k for k in coarse.vertices() if not coarse.is_vertex_on_boundary(k)))
    moved = mcp("coarse_move_corner", corner=vertex_handle(corner), to=[corner[0] + 0.1, corner[1] + 0.1])
    assert moved.get("ok") is True, moved.get("reason")
    assert mcp("rhino_push_coarse").get("ok") is False  # an edit needs another look


def test_a_face_with_two_corners_equal_at_float32_is_caught_before_rhino_refuses_it():
    # 1e-9 at a coordinate of 1 is below float32's step there (1.2e-7)
    pinched = CoarsePseudoQuadMesh.from_vertices_and_faces([[0, 0, 0], [1, 1, 0], [1 + 1e-9, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]])
    assert len(_float32_collapsed(pinched)) == 1
    spread = CoarsePseudoQuadMesh.from_vertices_and_faces([[0, 0, 0], [1, 1, 0], [1 + 1e-3, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]])
    assert _float32_collapsed(spread) == []


def test_pull_coarse_brings_the_pushed_layout_back(mcp, link):
    _, pushed = _pushed_coarse(mcp, link)
    out = mcp("rhino_pull_coarse")
    assert out.get("ok") is True, out.get("reason")
    assert out["faces"] == pushed.number_of_faces()
    assert mcp.session.coarse.get_strip_densities() == pushed.get_strip_densities()
    assert mcp.session.coarse.attributes["dense_pattern"] == pushed.attributes["dense_pattern"]
    assert out["dense_mesh_dropped"] is False
    assert mcp("history")["steps"][-1]["layer"] == "coarse"
    dense = mcp("coarse_densify")
    assert dense.get("ok") is True, dense.get("reason")
    mesh = mcp.session.mesh
    apex = [p for p in (mesh.vertex_coordinates(k) for k in mesh.vertices()) if math.dist(p[:2], [7.5, 2.0]) < 0.01]
    assert apex  # the drawn ARC that went out, not its chord


def test_pull_coarse_refuses_when_the_document_has_no_layout(mcp, link):
    out = mcp("rhino_pull_coarse")
    assert out.get("ok") is False and "no coarse layout" in out.get("reason", "")


def test_push_markers_labels_the_worst_faces_and_clears_every_kind(mcp, link):
    _pushed_coarse(mcp, link)
    mcp("coarse_densify")
    mcp("inspect", image=True)
    out = mcp("rhino_push_markers", worst=3)
    assert out.get("ok") is True, out.get("reason")
    sent = link.pushed["markers"]
    worst = [m for m in sent["markers"] if m["kind"] == "worst"]
    assert len(worst) == 3 and all(m["text"].endswith("deg") for m in worst)
    assert sorted(sent["kinds"]) == ["pole", "singularity", "worst"]
    assert sum(out["counts"].values()) == len(sent["markers"])
    out = mcp("rhino_push_markers", clear=True)
    assert out.get("ok") is True
    assert link.pushed["markers"]["markers"] == [] and len(link.pushed["markers"]["kinds"]) == 3


def test_a_pull_from_the_selection_takes_its_walls_and_points(mcp, link):
    out = mcp("rhino_pull", selection=True)
    assert out.get("ok") is True, out.get("reason")
    assert out.get("layer") == "(selection)" and out["walls"] == 1 and out["point_features"] == 1
    assert mcp.session.points == SELECTION["points"]


def test_a_link_that_goes_away_is_reported_not_hung_on(mcp, link):
    link.close()
    out = mcp("rhino_pull", timeout=0.3)
    assert out.get("ok") is False
    assert out.get("timed_out") is True
    assert "CMD_mcp_link" in out.get("reason", "")
