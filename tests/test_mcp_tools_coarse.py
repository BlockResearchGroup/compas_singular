"""The MCP coarse-layout tools, driven through the server, with no Rhino and no model.

The domain is an L, which decomposes into 9 patches and 7 strips, every one of
them removable -- enough to edit without a refusal hiding a wiring fault.
"""

import json
import math

import pytest

from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.mcp import tools_coarse
from compas_singular.mcp.handle import vertex_handle

L_SHAPE = [[0, 0, 0], [10, 0, 0], [10, 4, 0], [4, 4, 0], [4, 10, 0], [0, 10, 0]]
SQUARE = [[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0]]
# ``l_layout``'s scripted cuts were drawn on the layout the old default
# (alpha = 0.04) gives, so its spacing is pinned to it.
L_SPACING = 0.04 * math.hypot(10, 10)
HOLE = [[5 + 2 * math.cos(2 * math.pi * i / 48), 5 + 2 * math.sin(2 * math.pi * i / 48), 0] for i in range(48)]
POLE = [2, 2, 0]


def _history(mcp):
    return [(s["action"], s["undone"]) for s in mcp("history")["steps"]]


def _raw(mcp, _tool, **arguments):
    return mcp.dispatcher.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": _tool, "arguments": arguments}})["result"]


def _rounded_positions(mesh):
    return sorted(tuple(round(c, 9) for c in mesh.vertex_coordinates(k)) for k in mesh.vertices())


def _wall_to_wall_polyedge(coarse):
    """Handles of a straight run of coarse corners from one wall to another."""
    boundary = set(coarse.vertices_on_boundary())
    for u in sorted(boundary):
        for v in coarse.vertex_neighbors(u):
            if v in boundary:
                continue
            loop = coarse.edge_loop((u, v))
            keys = [loop[0][0]] + [edge[1] for edge in loop]
            if len(keys) >= 3 and keys[0] in boundary and keys[-1] in boundary:
                return [vertex_handle(coarse.vertex_coordinates(k)) for k in keys]
    return None


def _kink_at_start(curve):
    """Degrees the curve turns between its first and second segment."""
    a, b, c = curve[0], curve[1], curve[2]
    v1, v2 = (b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1])
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _arc():
    arc = [[7 + 0.5 * math.sin(math.pi * i / 20), 4.0 * i / 20, 0] for i in range(21)]
    arc[0], arc[-1] = [7.03, -0.05, 0], [6.98, 4.04, 0]  # a little off the walls
    return arc


@pytest.fixture
def l_layout(mcp):
    mcp.session.adopt(None, walls=[L_SHAPE], guides=[], points=[])  # a boundary-only pull
    out = mcp("create_coarse_mesh", target_length=L_SPACING)
    assert out.get("ok") is True, out.get("reason")
    return mcp


@pytest.fixture
def pole_layout(mcp):
    """The L with a point feature at density 3, and the mesh the old densifier makes of it."""
    mcp.session.adopt(None, walls=[L_SHAPE], guides=[], points=[POLE])
    mcp("create_coarse_mesh")
    mcp("coarse_set_density", density={"kind": "all", "value": 3})
    curves, _ = mcp.session.decomposition.edges_to_curves(coarse=mcp.session.coarse)
    old = mcp.session.coarse.copy().densification(overwrite_edges_to_curves=curves)
    new = mcp.session.coarse.copy().quad_mesh(overwrite_edges_to_curves=curves)
    return mcp, old, new


@pytest.fixture
def fan_layout(pole_layout):
    """The pole L in the fan pattern, with an odd density set after the pattern, densified."""
    mcp, _, new = pole_layout
    mcp("coarse_set_pattern", pattern="fan")
    mcp("coarse_set_density", density={"kind": "strip", "skey": 1, "value": 5})
    out = mcp("coarse_densify")
    return mcp, out, new


@pytest.fixture
def holed_layout(mcp):
    mcp.session.adopt(None, walls=[SQUARE, HOLE], guides=[], points=[])
    mcp("create_coarse_mesh")
    mcp("coarse_set_density", density={"kind": "all", "value": 4})
    mcp("coarse_set_pattern", pattern="diagonal", faces={"kind": "strip", "skey": 0})
    mcp("coarse_densify")
    return mcp


def _interior_curved_corner(mcp):
    coarse = mcp.session.coarse
    # NOT vertices_on_boundary(): in compas 2 that is the longest boundary only,
    # and the corners on the hole would pass for interior ones.
    shapes = tools_coarse._edges_to_curves(mcp.session)
    return next(
        (k, n) for (u, v), curve in shapes.items() if len(curve) > 2 and not coarse.is_edge_on_boundary((u, v)) for k, n in ((u, v), (v, u)) if not coarse.is_vertex_on_boundary(k)
    )


# ==============================================================================
# the layout
# ==============================================================================


def test_create_coarse_mesh_refuses_with_no_boundary(mcp):
    assert "no boundary" in mcp("create_coarse_mesh").get("reason", "")


@pytest.mark.parametrize(
    "tool, arguments",
    [
        ("coarse_inspect", {}),
        ("coarse_densify", {}),
        ("coarse_set_density", {"density": {"kind": "all", "value": 2}}),
        ("coarse_plan_strip_deletion", {"skey": 0}),
        ("coarse_remove_strip", {"skey": 0}),
        ("coarse_add_strip", {"polyedge": ["a", "b", "c"]}),
        ("coarse_undo", {}),
    ],
)
def test_every_coarse_tool_refuses_cleanly_before_its_input_exists(mcp, tool, arguments):
    assert mcp(tool, **arguments).get("ok") is False


def test_a_boundary_alone_builds_a_layout(mcp):
    mcp.session.adopt(None, walls=[L_SHAPE], guides=[], points=[])
    out = mcp("create_coarse_mesh")
    assert out.get("ok") is True, out.get("reason")
    assert (out["faces"], out["strips"]) == (9, 7)
    assert mcp("history")["steps"][-1]["layer"] == "coarse"


def test_coarse_inspect_names_every_strip_by_an_edge_of_handles(l_layout):
    ins = l_layout("coarse_inspect")
    assert len(ins["strips"]) == 7
    assert all(s["density"] == 1 for s in ins["strips"])
    assert all(len(s["edge"]) == 2 and all(h.startswith("v:") for h in s["edge"]) for s in ins["strips"])


def test_planning_a_deletion_changes_nothing(l_layout):
    ins = l_layout("coarse_inspect")
    faces = l_layout.session.coarse.number_of_faces()
    plan = l_layout("coarse_plan_strip_deletion", edge=ins["strips"][0]["edge"])
    assert plan.get("ok") is True, plan.get("reason")
    assert plan.get("faces_after", faces) < faces
    assert l_layout.session.coarse.number_of_faces() == faces
    assert l_layout("coarse_plan_strip_deletion").get("ok") is False  # neither edge nor skey
    assert l_layout("coarse_plan_strip_deletion", edge=["v:99.000,99.000,0.000", "v:98.000,99.000,0.000"]).get("ok") is False


def test_strip_edits_and_coarse_undo(l_layout):
    polyedge = _wall_to_wall_polyedge(l_layout.session.coarse)
    assert polyedge is not None
    out = l_layout("coarse_add_strip", polyedge=polyedge)
    assert out.get("ok") is True, out.get("reason")
    assert out.get("strips") == 8
    assert l_layout("coarse_add_strip", polyedge=polyedge[:2]).get("ok") is False
    out = l_layout("coarse_undo")
    assert out.get("ok") is True and out["strips"] == 7
    assert _history(l_layout)[-1] == ("coarse_add_strip", True)
    out = l_layout("coarse_remove_strip", skey=0)
    assert out.get("ok") is True, out.get("reason")
    assert out.get("strips") == 6


def test_all_the_density_shapes(l_layout):
    for density in (
        {"kind": "all", "value": 2},
        {"kind": "strip", "skey": 1, "value": 3},
        {"kind": "strips", "skeys": [1, 2], "value": 4},
        {"kind": "target_length", "value": 1.0},
        {"kind": "target_length", "value": 1.0, "skeys": [1]},
        {"kind": "face_count", "value": 200},
    ):
        out = l_layout("coarse_set_density", density=density)
        assert out.get("ok") is True, (density, out.get("reason"))
    assert l_layout("coarse_set_density", density={"kind": "strip", "value": 2}).get("ok") is False
    assert l_layout("coarse_set_density", density={"kind": "vibes", "value": 2}).get("ok") is False
    assert l_layout.session.coarse.get_strip_density(1) == 5


def test_densify_hands_the_dense_tools_a_mesh(l_layout):
    l_layout("coarse_set_density", density={"kind": "all", "value": 2})
    out = l_layout("coarse_densify")
    assert out.get("ok") is True, out.get("reason")
    assert l_layout.session.loaded and out["faces"] > 9
    assert l_layout.session.coarse is not None
    assert l_layout("history")["steps"][-1]["layer"] == "dense"


def test_the_two_undo_stacks_share_a_history_without_eating_it(l_layout):
    l_layout("coarse_set_density", density={"kind": "all", "value": 2})
    l_layout("coarse_densify")
    l_layout("smooth_boundary_constrained", kmax=10)  # dense snapshot taken here
    l_layout("coarse_set_density", density={"kind": "all", "value": 3})
    out = l_layout("undo")
    assert out.get("ok") is True, out.get("reason")
    steps = _history(l_layout)
    assert ("smooth_boundary_constrained", True) in steps
    assert steps[-1] == ("coarse_set_density", False)  # the LATER coarse step is left alone
    assert l_layout("history")["state"]["steps"] == sum(1 for _, u in steps if not u)
    assert "(UNDONE)" in l_layout.handler.read_resource("session://current")


def test_a_different_domain_drops_the_layout_and_the_same_one_keeps_it(l_layout, tmp_path):
    l_layout("coarse_set_density", density={"kind": "all", "value": 2})
    l_layout("coarse_densify")
    session = l_layout.session
    session.adopt(session.mesh, walls=[L_SHAPE], guides=[], points=[])
    assert session.coarse is not None
    path = str(tmp_path / "dense.json")
    session.mesh.to_json(path)
    out = l_layout("load_mesh", path=path, walls=[SQUARE])
    assert session.coarse is None
    assert "coarse_cleared" in out
    assert l_layout("coarse_densify").get("ok") is False  # refuses instead of mixing domains


# ==============================================================================
# densifying, drawing and patterns
# ==============================================================================


def test_the_ortho_path_is_bit_identical_to_the_old_densifier(pole_layout):
    mcp, old, new = pole_layout
    assert sorted(old.vertex_coordinates(k) for k in old.vertices()) == sorted(new.vertex_coordinates(k) for k in new.vertices())
    assert (old.number_of_faces(), len(old.attributes["face_pole"])) == (new.number_of_faces(), len(new.attributes["face_pole"]))
    out = mcp("coarse_densify")
    assert out.get("faces") == new.number_of_faces()
    assert out["density_raised"] == {} and "pattern_caveat" not in out


def test_coarse_inspect_draws_the_layout_with_numbered_strips(pole_layout):
    mcp, _, _ = pole_layout
    mcp("coarse_densify")
    ins = mcp("coarse_inspect")
    assert len(ins["patches"]) == mcp.session.coarse.number_of_faces()
    # every patch handle resolves back to it
    assert all(mcp("coarse_set_pattern", pattern="ortho", faces={"kind": "at", "handles": [p["at"]]}).get("set") == 1 for p in ins["patches"][:4])
    assert all(s["edge"][0] != s["edge"][1] for s in ins["strips"])  # none named by the pole
    seen = mcp.session.visually_current
    result = _raw(mcp, "coarse_inspect", image=True)
    images = [c for c in result["content"] if c["type"] == "image"]
    payload = json.loads(result["content"][0]["text"])
    assert len(images) == 1
    assert images[0]["data"].startswith("iVBORw0KGgo")  # a real PNG
    assert len(payload.get("strip_colors", {})) == len(ins["strips"])
    assert mcp.session.visually_current == seen  # not looking at the DENSE mesh


def test_pattern_selectors_are_validated(pole_layout):
    mcp, _, _ = pole_layout
    assert mcp("coarse_set_pattern", pattern="herringbone").get("ok") is False
    assert mcp("coarse_set_pattern", pattern="fan", faces={"kind": "strip", "skey": 999}).get("ok") is False
    assert mcp("coarse_set_pattern", pattern="fan", faces={"kind": "at", "handles": ["v:50.000,50.000,0.000"]}).get("ok") is False


def test_a_diagonal_pattern_raises_density_to_even_and_undoes_with_it(pole_layout):
    mcp, _, _ = pole_layout
    out = mcp("coarse_set_pattern", pattern="diagonal", faces={"kind": "strip", "skey": 0})
    assert out.get("ok") is True and out["set"] >= 1, out.get("reason")
    assert out["density_raised"] and all(new == 4 for _, new in out["density_raised"].values())
    assert all(mcp.session.coarse.get_strip_density(int(s)) == 4 for s in out["density_raised"])
    mcp("coarse_undo")
    assert mcp("coarse_inspect")["patterns"]["diagonal"] == 0
    assert mcp.session.coarse.get_strip_density(0) == 3


def test_densify_re_raises_an_odd_density_set_after_the_pattern(fan_layout):
    _, out, new = fan_layout
    assert "1" in out.get("density_raised", {})
    assert "pattern_caveat" in out
    assert out["faces"] > new.number_of_faces()


# ==============================================================================
# saving and loading
# ==============================================================================


def test_coarse_save_and_load_round_trip_the_layout_and_its_domain(fan_layout, tmp_path):
    mcp, out, new = fan_layout
    session = mcp.session
    path = str(tmp_path / "layout.json")
    patterned_faces = out["faces"]
    patterned_positions = _rounded_positions(session.mesh)
    saved = mcp("coarse_save", path=path)
    assert saved.get("ok") is True, saved.get("reason")
    assert saved.get("curved_edges", 0) > 0
    assert mcp("coarse_save", path=str(tmp_path / "nope" / "x.json")).get("ok") is False
    session.adopt(None, walls=[SQUARE], guides=[], points=[])  # somewhere else entirely
    loaded = mcp("coarse_load", path=path)
    assert loaded.get("ok") is True, loaded.get("reason")
    assert session._domain()[0] == [L_SHAPE] and session.points == [POLE]
    assert loaded["patterns"]["fan"] == session.coarse.number_of_faces()
    assert mcp("history")["steps"][-1]["layer"] == "coarse"
    # with no decomposition behind it, the same faces AND positions: the edge curves came back
    assert mcp("coarse_densify").get("faces") == patterned_faces
    assert _rounded_positions(session.mesh) == patterned_positions
    loaded = mcp("coarse_load", path=path)
    assert loaded["dense_mesh_dropped"] is False and session.loaded
    session.adopt(session.mesh, walls=[SQUARE], guides=[], points=[])
    assert mcp("coarse_load", path=path)["dense_mesh_dropped"] is True and not session.loaded
    mesh_path = str(tmp_path / "mesh.json")
    new.to_json(mesh_path)
    assert "load_mesh" in mcp("coarse_load", path=mesh_path).get("reason", "")
    assert mcp("coarse_load", path=str(tmp_path / "nope.json")).get("ok") is False


def test_a_coarse_mesh_survives_to_json_with_integer_keys(fan_layout, tmp_path):
    mcp, _, _ = fan_layout
    layout = mcp.session.coarse
    path = str(tmp_path / "plain.json")
    layout.to_json(path)
    back = CoarsePseudoQuadMesh.from_json(path)
    for name in ("strips", "strips_density", "dense_pattern", "face_pole"):
        assert back.attributes[name] == layout.attributes[name], name


# ==============================================================================
# dividing
# ==============================================================================


def test_coarse_divide_by_strip_shares_densities_and_keeps_element_size(l_layout):
    l_layout("coarse_set_density", density={"kind": "all", "value": 4})
    ortho_faces = l_layout("coarse_densify")["faces"]
    assert l_layout("coarse_divide").get("ok") is False
    assert l_layout("coarse_divide", skey=1, points=[[0, 0, 0], [1, 1, 0]]).get("ok") is False
    assert l_layout("coarse_divide", points=[[0, 0, 0], [1, 1, 0]], t=0.3).get("ok") is False
    assert l_layout("coarse_divide", skey=4, t=1.0).get("ok") is False
    strips_before = len(list(l_layout.session.coarse.strips()))
    out = l_layout("coarse_divide", skey=4)
    assert out.get("ok") is True, out.get("reason")
    assert len(out["strips"]) == strips_before + 1
    halves = [s for s in out["strips"] if s["from_skey"] == 4]
    assert len(halves) == 2 and all(s["density"] == 2 for s in halves)
    assert all(s["density"] == 4 for s in out["strips"] if s["from_skey"] != 4)
    assert l_layout("history")["steps"][-1]["layer"] == "coarse"
    assert l_layout("coarse_densify")["faces"] == ortho_faces
    undone = l_layout("coarse_undo")
    assert undone.get("ok") is True and undone["strips"] == strips_before
    assert _history(l_layout)[-2] == ("coarse_divide", True)


def test_coarse_divide_refuses_a_diagonal_and_an_end_off_the_layout(l_layout):
    assert "OPPOSITE" in l_layout("coarse_divide", points=[[0, 0, 0], [4, 4, 0]]).get("reason", "")
    assert "snap" in l_layout("coarse_divide", points=[[20, 20, 0], [7, 4, 0]]).get("reason", "")


def test_coarse_divide_along_a_drawn_arc(l_layout):
    l_layout("coarse_set_density", density={"kind": "all", "value": 4})
    l_layout("coarse_set_pattern", pattern="diagonal", faces={"kind": "strip", "skey": 0})
    out = l_layout("coarse_divide", points=_arc())
    assert out.get("ok") is True, out.get("reason")
    assert out.get("snapped_by") and all(0 < d < 0.1 for d in out["snapped_by"])
    assert out.get("shaped_edges", 0) > 0
    assert "warning" not in out  # clear of other lines, so no sliver
    assert out["patterns"]["diagonal"] == 4  # the patches it split kept their pattern
    out = l_layout("coarse_densify")
    assert out.get("ok") is True, out.get("reason")
    mesh = l_layout.session.mesh
    apex = [p for p in (mesh.vertex_coordinates(k) for k in mesh.vertices()) if abs(p[1] - 2.0) < 0.01 and abs(p[0] - 7.5) < 0.01]
    assert apex  # the mesh follows the ARC, not its chord
    text = _raw(l_layout, "coarse_inspect", image=True)["content"][0]["text"]
    assert "image_error" not in json.loads(text)


def test_a_later_strip_edit_does_not_erase_the_arcs_shapes(l_layout):
    # A fresh editor used to start with an empty curve map and write it back empty.
    l_layout("coarse_set_density", density={"kind": "all", "value": 4})
    l_layout("coarse_divide", points=_arc())
    coarse = l_layout.session.coarse
    shapes_before = len(coarse.attributes.get("user_curves") or [])
    assert shapes_before > 0
    removable = next(s["skey"] for s in l_layout("coarse_inspect")["strips"] if l_layout("coarse_plan_strip_deletion", skey=s["skey"]).get("ok"))
    assert l_layout("coarse_remove_strip", skey=removable).get("ok") is True
    assert len(l_layout.session.coarse.attributes.get("user_curves") or []) == shapes_before


def test_a_refused_cut_changes_nothing_and_a_saved_arc_densifies_identically(l_layout, tmp_path):
    session = l_layout.session
    l_layout("coarse_set_density", density={"kind": "all", "value": 4})
    l_layout("coarse_divide", points=_arc())
    l_layout("coarse_densify")
    arc_positions = _rounded_positions(session.mesh)
    faces = session.coarse.number_of_faces()
    steps = len(l_layout("history")["steps"])
    depth = session.state()["coarse_undo_depth"]
    assert l_layout("coarse_divide", points=[[0, 0, 0], [4, 4, 0]]).get("ok") is False
    assert session.coarse.number_of_faces() == faces
    assert len(l_layout("history")["steps"]) == steps
    assert session.state()["coarse_undo_depth"] == depth
    path = str(tmp_path / "arc.json")
    l_layout("coarse_save", path=path)
    session.decomposition = None
    l_layout("coarse_load", path=path)
    l_layout("coarse_densify")
    assert _rounded_positions(session.mesh) == arc_positions


def test_a_cut_drawn_a_hair_beside_an_existing_line_is_warned_about(l_layout):
    hugging = [[7 + 0.8 * math.sin(math.pi * i / 20), 4.0 * i / 20, 0] for i in range(21)]
    out = l_layout("coarse_divide", points=hugging)
    assert out.get("ok") is True, out.get("reason")  # it breaks no all-quad rule
    assert "sliver" in out.get("warning", "") and out.get("pinches")


# ==============================================================================
# moving corners
# ==============================================================================


def test_coarse_move_corner_warps_the_curved_edges_of_an_interior_corner(holed_layout):
    mcp = holed_layout
    session = mcp.session
    welded_vertices = session.mesh.number_of_vertices()
    coarse = session.coarse
    inner, neighbour = _interior_curved_corner(mcp)
    start = list(coarse.vertex_coordinates(inner))
    densities = dict(coarse.get_strip_densities())
    patterns = dict(coarse.attributes["dense_pattern"])
    assert mcp("coarse_move_corner", corner="v:55.000,55.000,0.000", to=[1, 1]).get("ok") is False
    out = mcp("coarse_move_corner", corner=vertex_handle(start), to=[start[0] + 0.25, start[1] + 0.15])
    assert out.get("ok") is True, out.get("reason")
    assert out["projected"] is False
    assert out["to"] == vertex_handle([start[0] + 0.25, start[1] + 0.15, 0])
    assert out["shapes_warped"] > 0 and out["shapes_straightened"] == 0
    moved = tools_coarse._edges_to_curves(session)
    shape = moved.get((inner, neighbour)) or list(reversed(moved[(neighbour, inner)]))
    # The decomposition alone only re-pins the END: measured 24.3 degrees of kink
    # at the corner after a 0.18 nudge, and a straight chord past a quarter edge.
    assert len(shape) > 2 and _kink_at_start(shape) < 10.0
    assert coarse.number_of_faces() == 12
    assert dict(coarse.get_strip_densities()) == densities
    assert dict(coarse.attributes["dense_pattern"]) == patterns
    assert mcp("history")["steps"][-1]["layer"] == "coarse"
    out = mcp("coarse_densify")
    assert out.get("ok") is True, out.get("reason")
    here = coarse.vertex_coordinates(inner)
    assert session.mesh.number_of_vertices() == welded_vertices  # it still WELDS there
    assert sum(1 for v in session.mesh.vertices() if math.dist(session.mesh.vertex_coordinates(v)[:2], here[:2]) < 1e-3) == 1
    mcp("coarse_undo")
    # undo swaps in the restored COPY; the old object is stale
    assert session.coarse.vertex_coordinates(inner) == start


def test_coarse_move_corner_refuses_a_fold_and_a_jump_to_another_wall(holed_layout):
    mcp = holed_layout
    session = mcp.session
    coarse = session.coarse
    inner, _ = _interior_curved_corner(mcp)
    start = list(coarse.vertex_coordinates(inner))
    faces, steps = coarse.number_of_faces(), len(mcp("history")["steps"])
    depth = session.state()["coarse_undo_depth"]
    out = mcp("coarse_move_corner", corner=vertex_handle(start), to=[5, 5])
    assert out.get("ok") is False
    assert "cross" in out.get("reason", "")
    assert coarse.vertex_coordinates(inner) == start and coarse.number_of_faces() == faces
    assert len(mcp("history")["steps"]) == steps
    assert session.state()["coarse_undo_depth"] == depth
    on_hole = next(k for k in coarse.vertices() if coarse.is_vertex_on_boundary(k) and abs(math.dist(coarse.vertex_coordinates(k)[:2], [5, 5]) - 2) < 0.05)
    q = coarse.vertex_coordinates(on_hole)
    out = mcp("coarse_move_corner", corner=vertex_handle(q), to=[0.2, q[1]])
    assert "its own wall" in out.get("reason", "")


def test_a_boundary_corner_slides_along_its_wall(holed_layout):
    mcp = holed_layout
    coarse = mcp.session.coarse
    on_bottom = next(
        k for k in coarse.vertices() if coarse.is_vertex_on_boundary(k) and 0.5 < coarse.vertex_coordinates(k)[0] < 9.5 and abs(coarse.vertex_coordinates(k)[1]) < 1e-6
    )
    q = coarse.vertex_coordinates(on_bottom)
    out = mcp("coarse_move_corner", corner=vertex_handle(q), to=[q[0] + 0.4, 0.3])
    assert out.get("ok") is True, out.get("reason")
    assert out["projected"] and abs(coarse.vertex_coordinates(on_bottom)[1]) < 1e-9 and "note" in out
    assert "warning" not in out  # no domain-corner warning on a straight wall
    mcp("coarse_undo")
    out = mcp("coarse_move_corner", corner="v:10.000,10.000,0.000", to=[10, 9])
    assert out.get("ok") is True and "OF THE DOMAIN" in out.get("warning", "")


# ==============================================================================
# density refusals
# ==============================================================================


def test_coarse_set_density_refuses_an_unknown_strip_and_a_density_under_one(holed_layout):
    mcp = holed_layout
    coarse = mcp.session.coarse
    before = dict(coarse.get_strip_densities())
    depth = mcp.session.state()["coarse_undo_depth"]
    out = mcp("coarse_set_density", density={"kind": "strip", "skey": 9999, "value": 3})
    assert out.get("ok") is False and "9999" in out.get("reason", "")
    assert sorted(out.get("skeys") or []) == sorted(coarse.strips())
    assert mcp("coarse_set_density", density={"kind": "strips", "skeys": [0, 9999], "value": 3}).get("ok") is False
    for bad in ({"kind": "all", "value": 0}, {"kind": "all", "value": -2}, {"kind": "target_length", "value": 0}, {"kind": "face_count", "value": 1}):
        assert mcp("coarse_set_density", density=bad).get("ok") is False, bad
    assert dict(coarse.get_strip_densities()) == before  # no phantom density stored
    assert mcp.session.state()["coarse_undo_depth"] == depth
