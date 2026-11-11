"""MCP dense line edits, input checks and step comparison -- no Rhino, no model."""

import math

import pytest

from compas_singular.mcp.handle import vertex_handle

L_SHAPE = [[0, 0, 0], [10, 0, 0], [10, 4, 0], [4, 4, 0], [4, 10, 0], [0, 10, 0]]
SQUARE = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]
# ``_densified_l``'s pole and line checks were written on the layout the old
# default (alpha = 0.04) gives, so its spacing is pinned to it.
L_SPACING = 0.04 * math.hypot(10, 10)


def _circle(cx, cy, r, n=48):
    return [[cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n), 0] for i in range(n)]


def _interior_edge(mesh):
    u = next(k for k in mesh.vertices() if not mesh.is_vertex_on_boundary(k))
    v = mesh.vertex_neighbors(u)[0]
    return [vertex_handle(mesh.vertex_coordinates(u)), vertex_handle(mesh.vertex_coordinates(v))]


def _densified_l(mcp, points=(), density=3):
    mcp.session.adopt(None, walls=[L_SHAPE], guides=[], points=[list(p) for p in points])
    mcp("create_coarse_mesh", target_length=L_SPACING)
    mcp("coarse_set_density", density={"kind": "all", "value": density})
    mcp("coarse_densify")
    return mcp.session.mesh


@pytest.mark.parametrize("tool", ["dense_plan_line_removal", "dense_remove_line", "dense_add_line"])
def test_the_dense_line_tools_refuse_cleanly_with_nothing_to_edit(mcp, tool):
    assert mcp(tool, edge=["v:0.000,0.000,0.000", "v:1.000,0.000,0.000"]).get("ok") is False


def test_remove_a_line_planned_done_and_undone_exactly(mcp):
    mesh = _densified_l(mcp)
    assert all(len(mesh.face_vertices(f)) == 4 for f in mesh.faces())
    edge = _interior_edge(mesh)
    faces = mesh.number_of_faces()
    plan = mcp("dense_plan_line_removal", edge=edge)
    assert plan.get("ok") is True, plan.get("reason")
    assert plan["faces_after"] < faces
    assert mcp.session.mesh.number_of_faces() == faces  # planning changes nothing
    out = mcp("dense_remove_line", edge=edge)
    assert out.get("ok") is True, out.get("reason")
    assert out["faces"] == plan["faces_after"]
    assert mcp.session.mesh.number_of_faces() == out["faces"]
    assert "coarse_densify" in out.get("note", "")  # a re-densify would discard it
    assert mcp("history")["steps"][-1]["layer"] == "dense"
    assert mcp("undo").get("ok") is True
    assert mcp.session.mesh.number_of_faces() == faces


def test_add_a_line_and_undo_across_a_smoothing_pass(mcp):
    mesh = _densified_l(mcp)
    edge = _interior_edge(mesh)
    faces = mesh.number_of_faces()
    positions = dict(mcp.session.positions())
    out = mcp("dense_add_line", edge=edge)
    assert out.get("ok") is True, out.get("reason")
    assert out["faces"] > faces
    assert "relaxed" not in out
    mcp("smooth_boundary_constrained", kmax=10)  # a POSITION snapshot on top
    assert mcp("undo").get("ok") is True
    assert mcp("undo").get("ok") is True
    assert mcp.session.mesh.number_of_faces() == faces
    assert dict(mcp.session.positions()) == positions


def test_a_line_between_handles_that_are_not_neighbours_or_off_the_mesh_is_refused(mcp):
    mesh = _densified_l(mcp)
    edge = _interior_edge(mesh)
    first = next(v for v in mesh.vertices() if vertex_handle(mesh.vertex_coordinates(v)) == edge[0])
    far = next(k for k in mesh.vertices() if vertex_handle(mesh.vertex_coordinates(k)) not in edge and k not in mesh.vertex_neighbors(first))
    out = mcp("dense_add_line", edge=[edge[0], vertex_handle(mesh.vertex_coordinates(far))])
    assert "not joined by an edge" in out.get("reason", "")
    assert mcp("dense_remove_line", edge=["v:99.000,99.000,0.000", edge[1]]).get("ok") is False


def test_a_line_touching_a_pole_is_refused_but_one_clear_of_it_is_not(mcp):
    mesh = _densified_l(mcp, points=[[2, 2, 0]])
    pole_faces = [f for f in mesh.faces() if len(mesh.face_vertices(f)) != 4]
    assert pole_faces
    touching = [v for f in pole_faces for v in mesh.face_vertices(f)]
    near = next(v for v in touching if not mesh.is_vertex_on_boundary(v))
    out = mcp("dense_add_line", edge=[vertex_handle(mesh.vertex_coordinates(near)), vertex_handle(mesh.vertex_coordinates(mesh.vertex_neighbors(near)[0]))])
    assert out.get("ok") is False and "not quads" in out.get("reason", "")
    assert mcp.session.state()["undo_depth"] == 0  # a refusal leaves no snapshot
    clear = None
    for pe in dict(mesh.copy().collect_polyedges()).values():
        if (
            len(pe) > 2
            and mesh.is_vertex_on_boundary(pe[0])
            and mesh.is_vertex_on_boundary(pe[-1])
            and pe[0] != pe[-1]
            and all(len(mesh.face_vertices(f)) == 4 for v in pe for f in mesh.vertex_faces(v))
        ):
            clear = pe
            break
    assert clear is not None
    out = mcp("dense_add_line", edge=[vertex_handle(mesh.vertex_coordinates(clear[0])), vertex_handle(mesh.vertex_coordinates(clear[1]))])
    assert out.get("ok") is True, out.get("reason")


def test_check_inputs_passes_a_clean_domain(mcp):
    mcp.session.adopt(None, walls=[SQUARE, _circle(5, 5, 1.5)], guides=[], points=[])
    out = mcp("check_inputs")
    assert out.get("clean") is True, out.get("issues")


def test_check_inputs_names_every_quiet_failure(mcp):
    t = 0.02 * math.hypot(10, 10)
    mcp.session.adopt(
        None, walls=[SQUARE, _circle(1 + 0.25 * t, 3, 1), _circle(7, 7, 0.2)], guides=[[[5, 8, 0], [12, 8, 0]]], points=[[1.5 * t, 8, 0], [5, 3.0 * t, 0], [12, 5, 0], [7, 7, 0]]
    )
    out = mcp("check_inputs")
    kinds = sorted((i["kind"], i["severity"]) for i in out["issues"])
    assert abs(out["target_length"] - t) < 1e-3
    assert ("point_near_wall", "error") in kinds  # 1.5T from a wall
    assert ("point_near_wall", "warning") in kinds  # 3T from a wall
    assert sum(1 for k in kinds if k == ("point_outside", "error")) == 2  # outside, and in a hole
    assert ("hole_near_wall", "error") in kinds
    assert ("hole_small", "warning") in kinds
    assert ("guide_leaves_domain", "error") in kinds


def test_check_inputs_refuses_without_a_boundary(mcp):
    mcp.session.adopt(None, walls=[], guides=[], points=[])
    assert mcp("check_inputs").get("ok") is False


def test_compare_reads_two_steps_back(mcp):
    assert mcp("compare").get("ok") is False
    _densified_l(mcp, density=4)
    mcp("smooth_boundary_constrained", kmax=50)
    out = mcp("compare")
    assert out.get("ok") is True, out.get("reason")
    assert out["metrics"]["min_angle"]["b"] == mcp.session.quality()["min_angle"]
    assert sum(out["better_on"].values()) == sum(1 for k in ("min_angle", "max_angle", "aspect_max", "share_below") if out["metrics"][k].get("better"))
    assert "coarse steps" in mcp("compare", step_a=0).get("reason", "")
    assert "no step" in mcp("compare", step_a=99).get("reason", "")
