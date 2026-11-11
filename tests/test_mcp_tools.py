"""Every MCP mesh tool, driven through the server, with no Rhino and no model."""

import inspect
import os

import pytest
from conftest import make_dense_mesh

from compas_singular.editing.guide_chain import collect_polyedges
from compas_singular.mcp import registry
from compas_singular.mcp import tools_mesh


@pytest.fixture
def mesh_file(tmp_path):
    mesh, walls = make_dense_mesh(density=4)
    path = str(tmp_path / "dense.json")
    mesh.to_json(path)
    return path, mesh, walls


@pytest.fixture
def loaded(mcp, mesh_file):
    path, mesh, walls = mesh_file
    out = mcp("load_mesh", path=path, walls=walls)
    assert out.get("ok") is True, out.get("reason")
    return mcp


@pytest.mark.parametrize(
    "tool, arguments",
    [
        ("inspect", {}),
        ("relax", {}),
        ("smooth_boundary_constrained", {}),
        ("save_mesh", {"path": "x.json"}),
        ("undo", {}),
        ("smooth_region", {"region": {"kind": "all"}}),
    ],
)
def test_every_tool_refuses_cleanly_with_nothing_loaded(mcp, tool, arguments):
    assert mcp(tool, **arguments).get("ok") is False


def test_load_mesh_works_with_rhino_closed(mcp, mesh_file):
    path, mesh, walls = mesh_file
    out = mcp("load_mesh", path=path)
    assert out.get("ok") is True
    assert "warning" in out  # the boundary cannot be corrected without walls
    out = mcp("load_mesh", path=path, walls=walls)
    assert out.get("ok") is True and out["walls"] == len(walls)
    assert "warning" not in out
    assert out["faces"] == mesh.number_of_faces()
    assert mcp("load_mesh", path="nope.json").get("ok") is False


def test_inspect_reports_numbers_and_a_reading(loaded):
    out = loaded("inspect")
    assert out.get("ok") is True
    for key in ("min_angle", "max_angle", "aspect_max", "share_below", "worst_face"):
        assert key in out["quality"]
    assert len(out["reading"]) > 40
    assert out["verdict"] in ("good", "usable", "poor", "unusable")
    assert out["worst_face_at"] is None or out["worst_face_at"].startswith("v:")
    assert "worst_face" not in out["reading"]  # no raw face key escapes
    assert all(h.startswith("v:") for h in out["singularities"]["at"])


def test_relax_is_gated_and_never_makes_the_mesh_worse(loaded):
    before = loaded.session.triple()
    out = loaded("relax")
    after = loaded.session.triple()
    assert out.get("ok") is True, out.get("reason")
    if out["accepted"] is None:
        assert before == after
        assert out["all_improved"] is None
        assert "pass" in out["note"].lower()
    else:
        assert out["all_improved"] is True
        assert after[0] >= before[0] and after[1] <= before[1] and after[2] <= before[2]
    assert loaded("relax", seams="sideways").get("ok") is False


def test_an_ungated_pass_snapshots_itself_and_undo_restores_exactly(loaded):
    positions = dict(loaded.session.positions())
    out = loaded("smooth_boundary_constrained", kmax=30)
    assert out.get("ok") is True, out.get("reason")
    assert "all_improved" in out
    assert out["undo_available"] is True
    assert any(v != positions[k] for k, v in loaded.session.positions().items())
    assert loaded("undo").get("ok") is True
    assert loaded.session.positions() == positions
    assert loaded("smooth_boundary_constrained", algorithm="wishful").get("ok") is False


def test_smooth_region_selectors_and_the_blend_guard(loaded):
    out = loaded("smooth_region", region={"kind": "worst_faces", "count": 3}, kmax=20)
    assert out.get("ok") is True, out.get("reason")
    assert "worst face" in out.get("selected", "")
    loaded("undo")
    assert "crease" in loaded("smooth_region", region={"kind": "all"}, blend=0).get("reason", "")
    assert loaded("smooth_region", region={"kind": "banana"}).get("ok") is False
    assert loaded("smooth_region", region={"kind": "near_point"}).get("ok") is False


def test_remarks_and_history(loaded):
    loaded("smooth_boundary_constrained", kmax=10)
    assert loaded("remark", text="   ").get("ok") is False
    assert loaded("remark", text="x", about="v:999.000,999.000,0.000").get("ok") is False
    out = loaded("remark", text="relax was enough; the rest is the layout.")
    assert out.get("ok") is True and out["total"] == 1
    history = loaded("history")
    assert len(history["steps"]) >= 2
    assert len(history["remarks"]) == 1
    assert all("before" in s and "after" in s for s in history["steps"])


def test_save_and_recall_a_worked_example(loaded):
    assert loaded("save_example", name="x", verdict="excellent").get("ok") is False
    out = loaded("save_example", name="British Museum dense", verdict="good", lesson="relax alone was enough.", instruction="improve this mesh")
    assert out.get("ok") is True, out.get("reason")
    assert os.path.isfile(out.get("path", ""))
    found = loaded("recall_examples")
    assert len(found["examples"]) == 1
    assert "score" in found["examples"][0]
    assert "British" in (loaded.handler.read_resource(found["examples"][0]["uri"]) or "")
    second = loaded("save_example", name="British Museum dense", verdict="bad")
    assert second.get("path") != out.get("path")
    assert len(loaded("recall_examples", limit=5)["examples"]) == 2  # bad ones too, on purpose


def test_save_mesh_needs_a_look_first_and_round_trips(loaded, tmp_path):
    target = str(tmp_path / "out.json")
    assert loaded("save_mesh", path=target).get("ok") is False
    loaded("inspect", image=True)
    out = loaded("save_mesh", path=target)
    assert out.get("ok") is True, out.get("reason")
    assert loaded("load_mesh", path=target)["faces"] == out["faces"]
    loaded("inspect", image=True)
    assert loaded("save_mesh", path=str(tmp_path / "nope" / "x.json")).get("ok") is False


def test_the_rhino_tools_refuse_gracefully_with_no_rhino(mcp):
    out = mcp("rhino_status")
    assert out.get("ok") is True
    assert out["attached"] is False
    assert "CMD_mcp_link" in out["reading"]
    out = mcp("rhino_pull", timeout=0.2)
    assert out.get("ok") is False and "not listening" in out.get("reason", "")
    assert mcp("rhino_push", timeout=0.2).get("ok") is False


def test_bad_calls_are_refusals_not_crashes(loaded):
    out = loaded("no_such_tool")
    assert out.get("ok") is False and "available" in out
    out = loaded("inspect", nonsense=1)
    assert out.get("ok") is False
    # the message names the TOOL, not the implementation
    assert "inspect" in out.get("reason", "") and "_t_" not in out.get("reason", "")


@pytest.mark.parametrize("name", sorted(registry.TOOLS))
def test_every_schema_names_exactly_the_arguments_its_function_takes(name):
    # additionalProperties is false, so an argument the schema omits can never be passed.
    spec = registry.TOOLS[name]
    taken = sorted(list(inspect.signature(spec.function).parameters)[1:])
    assert taken == sorted(spec.schema["properties"])


def _mesh_and_guide():
    mesh, walls = make_dense_mesh(density=4)
    open_polyedges = [pe for pe, closed in collect_polyedges(mesh) if not closed and len(pe) > 6]
    # a guide just off a real polyedge, so a chain exists to be chosen
    guide = [[x + 0.05, y + 0.05, z] for x, y, z in (mesh.vertex_coordinates(k) for k in max(open_polyedges, key=len)[1:-1])]
    return mesh, walls, guide


def test_smooth_guides_attaches_a_guide_and_snapshots_first(mcp):
    mesh, walls, guide = _mesh_and_guide()
    mcp.session.adopt(mesh, walls=walls, guides=[], points=[])
    assert "no guide" in mcp("smooth_guides").get("reason", "")
    mcp.session.adopt(mesh, walls=walls, guides=[guide], points=[])
    positions = dict(mcp.session.positions())
    out = mcp("smooth_guides")
    assert out.get("ok") is True, out.get("reason")
    assert out.get("guides_attached") == 1
    assert out.get("vertices_moved_onto_guides", 0) > 0
    assert "all_improved" in out
    mcp("undo")
    assert mcp.session.positions() == positions
    assert mcp("smooth_guides", hold="glued").get("ok") is False


def test_relax_fdm_runs_snapshots_and_validates_its_fixed_set(loaded):
    positions = dict(loaded.session.positions())
    out = loaded("relax_fdm")
    if "compas_fd" in out.get("reason", ""):
        assert out.get("ok") is False
    else:
        assert out.get("ok") is True, out.get("reason")
        assert "all_improved" in out
        loaded("undo")
        assert loaded.session.positions() == positions
    assert loaded("relax_fdm", fixed="manual").get("ok") is False
    assert loaded("relax_fdm", fixed="nails").get("ok") is False


@pytest.mark.parametrize(
    "tool, attribute",
    [
        ("smooth_boundary_constrained", "boundary_constrained_smoothing"),
        ("smooth_guides", "constrained_smoothing"),
    ],
)
def test_a_pass_that_raises_halfway_puts_the_mesh_back(mcp, monkeypatch, tool, attribute):
    def move_then_raise(mesh, *args, **kwargs):
        for key in list(mesh.vertices())[:5]:
            x, y, z = mesh.vertex_coordinates(key)
            mesh.vertex_attributes(key, "xyz", [x + 0.3, y, z])
        raise RuntimeError("boom")

    mesh, walls, guide = _mesh_and_guide()
    mcp.session.adopt(mesh, walls=walls, guides=[guide], points=[])
    monkeypatch.setattr(tools_mesh, attribute, move_then_raise)
    positions = dict(mcp.session.positions())
    depth = mcp.session.state()["undo_depth"]
    out = mcp(tool)
    assert out.get("ok") is False and "boom" in out.get("reason", "")
    assert mcp.session.positions() == positions and out.get("mesh_restored") is True
    assert mcp.session.state()["undo_depth"] == depth
