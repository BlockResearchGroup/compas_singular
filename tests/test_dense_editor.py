"""DenseMeshEditor: strip deletion on one gated path, and every edit the command offers."""

import math

import pytest
from compas.datastructures import Mesh

from compas_singular.datastructures import PseudoQuadMesh
from compas_singular.datastructures import QuadMesh
from compas_singular.editing import DenseMeshEditor


def _grid(n):
    vertices = [[float(x), float(y), 0.0] for y in range(n + 1) for x in range(n + 1)]
    faces = [[y * (n + 1) + x, y * (n + 1) + x + 1, (y + 1) * (n + 1) + x + 1, (y + 1) * (n + 1) + x] for y in range(n) for x in range(n)]
    return QuadMesh.from_vertices_and_faces(vertices, faces)


@pytest.fixture
def editor():
    return DenseMeshEditor(_grid(3))


def test_a_refused_plan_has_the_same_keys_as_an_accepted_one(editor):
    ok_plan = editor.plan_strip_deletion((0, 1))
    bad_plan = editor.plan_strip_deletion((0, 99))
    assert ok_plan["ok"] and not bad_plan["ok"]
    assert set(ok_plan) == set(bad_plan)


def test_a_non_edge_is_reported_as_a_non_edge(editor):
    plan = editor.plan_strip_deletion((0, 99))
    assert "not an edge" in plan["reason"]


def test_remove_strip_runs_the_same_checks_as_remove_line(editor):
    # a pentagon on the strip through (0, 1) -- remove_line refuses it, so must remove_strip
    editor.mesh.delete_face(0)
    editor.mesh.delete_face(1)
    editor.mesh.add_face([0, 1, 2, 6, 5, 4])
    ok_line, notes_line = editor.remove_line((0, 4))
    ok_strip, notes_strip = editor.remove_strip((0, 4))
    assert not ok_line and not ok_strip
    assert notes_strip["error"] == notes_line["error"]


def test_a_strip_can_be_planned_and_removed_by_key(editor):
    work = editor.mesh.copy()
    work.collect_strips()
    skey = work.edge_strip((0, 1))
    plan = editor.plan_strip_deletion(skey=skey)
    assert plan["ok"], plan["reason"]
    ok, notes = editor.remove_strip(skey=skey)
    assert ok, notes
    assert notes["faces_after"] == plan["faces_after"]


def test_pre_split_is_always_a_dict(editor):
    ok, notes = editor.remove_line((0, 1))
    assert ok, notes
    assert notes["pre_split"] == {}


# ==============================================================================
# every operation CMD09_edit_quad_mesh offers, headless
# ==============================================================================


def _degrees(mesh):
    out = {}
    for fkey in mesh.faces():
        n = len(mesh.face_vertices(fkey))
        out[n] = out.get(n, 0) + 1
    return out


def _area(mesh):
    return sum(mesh.face_area(fkey) for fkey in mesh.faces())


def _simple(mesh):
    """No face repeats a corner, and every halfedge points at its own face."""
    for fkey in mesh.faces():
        corners = mesh.face_vertices(fkey)
        if len(set(corners)) != len(corners):
            return False
        for u, v in mesh.face_halfedges(fkey):
            if mesh.halfedge[u][v] != fkey:
                return False
    return True


def _rect(nx=4, ny=3, size=1.0, cls=QuadMesh):
    """An nx by ny quad grid. Vertex (i, j) at index j * (nx + 1) + i."""
    vertices = [[i * size, j * size, 0.0] for j in range(ny + 1) for i in range(nx + 1)]

    def index(i, j):
        return j * (nx + 1) + i

    faces = [[index(i, j), index(i + 1, j), index(i + 1, j + 1), index(i, j + 1)] for j in range(ny) for i in range(nx)]
    return cls.from_vertices_and_faces(vertices, faces)


def _vkey_at(mesh, x, y):
    return min(mesh.vertices(), key=lambda v: (mesh.vertex_coordinates(v)[0] - x) ** 2 + (mesh.vertex_coordinates(v)[1] - y) ** 2)


def _edge_at(mesh, a, b):
    return (_vkey_at(mesh, *a), _vkey_at(mesh, *b))


def _annulus(segments=12, rings=3, r0=1.0, dr=1.0):
    """A polar quad annulus: its circumferential polyedges are CLOSED."""
    vertices = []
    for k in range(rings + 1):
        for s in range(segments):
            angle = 2 * math.pi * s / segments
            r = r0 + k * dr
            vertices.append([r * math.cos(angle), r * math.sin(angle), 0.0])

    def index(k, s):
        return k * segments + s % segments

    faces = [[index(k, s), index(k, s + 1), index(k + 1, s + 1), index(k + 1, s)] for k in range(rings) for s in range(segments)]
    return QuadMesh.from_vertices_and_faces(vertices, faces)


def _square_walls(nx=4, ny=3, size=1.0):
    w, h = nx * size, ny * size
    return [[[0.0, 0.0, 0.0], [w, 0.0, 0.0], [w, h, 0.0], [0.0, h, 0.0], [0.0, 0.0, 0.0]]]


def _interior_edge(mesh):
    return next((u, v) for u, v in mesh.edges() if not mesh.is_edge_on_boundary(u, v))


def _single_quad():
    return DenseMeshEditor(QuadMesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]]))


def test_construction():
    mesh = _rect()
    editor = DenseMeshEditor(mesh, walls=_square_walls())
    assert editor.mesh is mesh  # in place
    assert DenseMeshEditor(_rect()).walls == []
    assert len(editor.walls) == 1
    assert editor.last_reason == ""
    assert editor.non_quad_faces() == []


def test_a_plain_mesh_is_refused():
    with pytest.raises(TypeError, match="QuadMesh"):
        DenseMeshEditor(Mesh.from_vertices_and_faces([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [[0, 1, 2, 3]]))


def test_move_vertex():
    editor = DenseMeshEditor(_rect(), walls=_square_walls())
    vkey = _vkey_at(editor.mesh, 1, 1)
    ok, notes = editor.move_vertex(vkey, [1.2, 1.1, 0.0])
    assert ok and notes["projected"] is False
    assert editor.mesh.vertex_coordinates(vkey) == [1.2, 1.1, 0.0]
    bvkey = _vkey_at(editor.mesh, 2, 0)
    ok, notes = editor.move_vertex(bvkey, [2.0, -0.5, 0.0])
    assert ok and abs(editor.mesh.vertex_coordinates(bvkey)[1]) < 1e-6  # held on the wall
    editor.move_vertex(bvkey, [2.0, -0.5, 0.0], project=False)
    assert abs(editor.mesh.vertex_coordinates(bvkey)[1] + 0.5) < 1e-9
    ok, notes = editor.move_vertex(9999, [0, 0, 0])
    assert ok is False and "not in the mesh" in editor.last_reason


def test_remove_an_interior_vertex_leaves_a_hole():
    editor = DenseMeshEditor(_rect(nx=4, ny=4))
    ok, notes = editor.remove_vertex(_vkey_at(editor.mesh, 2, 2))
    assert ok, notes
    assert notes.get("faces_removed") == 4 and notes["vertices_culled"] == 0
    assert editor.mesh.number_of_faces() == 12
    assert len(editor.mesh.vertices_on_boundaries()) == 2
    assert all(editor.mesh.vertex_neighbors(v) for v in editor.mesh.vertices())


def test_remove_a_corner_vertex_with_its_one_face():
    editor = DenseMeshEditor(_rect())
    ok, notes = editor.remove_vertex(_vkey_at(editor.mesh, 0, 0))
    assert ok and notes["faces_removed"] == 1 and notes["vertices_culled"] == 0
    assert editor.mesh.number_of_vertices() == 20 - 1
    assert editor.remove_vertex(9999)[0] is False


def test_removing_the_last_face_is_refused():
    ok, notes = _single_quad().remove_vertex(0)
    assert ok is False and "every face" in notes["error"]
    assert _single_quad().remove_face(0)[0] is False


def test_remove_face_culls_its_lone_corner():
    editor = DenseMeshEditor(_rect())
    ok, notes = editor.remove_face(next(iter(editor.mesh.faces())))
    assert ok and editor.mesh.number_of_faces() == 11
    assert notes["vertices_culled"] == 1
    assert editor.remove_face(9999)[0] is False


def test_remove_an_interior_edge_merges_two_quads_into_a_hexagon():
    editor = DenseMeshEditor(_rect())
    a0 = _area(editor.mesh)
    edge = _edge_at(editor.mesh, (1, 1), (1, 2))
    ok, notes = editor.remove_edge(edge)
    assert ok, notes
    assert _degrees(editor.mesh) == {4: 10, 6: 1} and notes["degree"] == 6
    assert abs(_area(editor.mesh) - a0) < 1e-9 and _simple(editor.mesh)
    assert edge[1] not in editor.mesh.halfedge[edge[0]]
    centre = _vkey_at(editor.mesh, 2, 1)
    for nbr in list(editor.mesh.vertex_neighbors(centre)):
        if centre in editor.mesh.vertex and nbr in editor.mesh.halfedge.get(centre, {}):
            editor.remove_edge((centre, nbr))
    assert centre not in editor.mesh.vertex and _simple(editor.mesh)  # every edge round it gone


def test_remove_a_boundary_edge_removes_its_one_face():
    editor = DenseMeshEditor(_rect())
    ok, notes = editor.remove_edge(_edge_at(editor.mesh, (1, 0), (2, 0)))
    assert ok and notes["merged"] is None and notes["on_boundary"]
    assert editor.mesh.number_of_faces() == 11
    assert editor.remove_edge((0, 9999))[0] is False
    assert editor.remove_edge(_edge_at(editor.mesh, (2, 1), (2, 2)))[0]


def test_draw_edges_straight_across_a_row():
    editor = DenseMeshEditor(_rect())
    a0 = _area(editor.mesh)
    ok, notes = editor.draw_edges([[-1.0, 1.5, 0.0], [5.0, 1.5, 0.0]])
    assert ok, notes
    assert notes.get("split_faces") == 4 and notes.get("split_edges") == 5
    assert _degrees(editor.mesh) == {4: 16}
    assert abs(_area(editor.mesh) - a0) < 1e-9 and _simple(editor.mesh)
    assert notes.get("trimmed") == 2  # the parts outside the mesh were left out


def test_draw_edges_diagonally_through_vertices_makes_triangles():
    editor = DenseMeshEditor(_rect())
    ok, _ = editor.draw_edges([[0.0, 0.0, 0.0], [3.0, 3.0, 0.0]])
    assert ok and _degrees(editor.mesh) == {4: 9, 3: 6} and _simple(editor.mesh)


def test_draw_edges_a_u_crossing_the_same_edges_twice():
    editor = DenseMeshEditor(_rect())
    ok, notes = editor.draw_edges([[0.5, -1.0, 0.0], [0.5, 1.5, 0.0], [1.5, 1.5, 0.0], [1.5, -1.0, 0.0]])
    assert ok and _simple(editor.mesh) and editor.mesh.is_manifold()
    assert notes.get("new_vertices", 0) >= notes.get("split_edges", 0) + 2  # bends inside a face kept


def test_draw_edges_trims_ends_inside_faces_back_to_the_last_crossing():
    editor = DenseMeshEditor(_rect())
    ok, notes = editor.draw_edges([[0.3, 0.5, 0.0], [2.7, 0.5, 0.0]])
    assert ok and notes["trimmed"] == 2 and notes["split_faces"] == 1 and notes["split_edges"] == 2


def test_draw_edges_refusals():
    editor = DenseMeshEditor(_rect())
    before = editor.mesh.number_of_faces()
    ok, notes = editor.draw_edges([[0.0, 1.0, 0.0], [4.0, 1.0, 0.0]])
    assert ok is False and "already there" in notes["error"]
    assert editor.mesh.number_of_faces() == before
    assert editor.draw_edges([[10.0, 10.0, 0.0], [12.0, 12.0, 0.0]])[0] is False  # misses the mesh
    assert editor.draw_edges([[0.5, 0.5, 0.0]])[0] is False  # one point


def test_draw_edges_a_second_line_crossing_the_first():
    editor = DenseMeshEditor(_rect())
    editor.draw_edges([[-1.0, 1.5, 0.0], [5.0, 1.5, 0.0]])
    ok, _ = editor.draw_edges([[1.5, -1.0, 0.0], [1.5, 4.0, 0.0]])
    assert ok and _simple(editor.mesh) and editor.mesh.is_manifold()


def test_draw_edges_on_a_mesh_with_height_puts_new_vertices_on_it():
    lifted = _rect()
    for v in lifted.vertices():
        lifted.vertex_attribute(v, "z", lifted.vertex_coordinates(v)[0])
    editor = DenseMeshEditor(lifted)
    ok, _ = editor.draw_edges([[0.25, -1.0, 0.0], [0.75, 4.0, 0.0]])
    assert ok
    assert all(abs(editor.mesh.vertex_coordinates(v)[2] - editor.mesh.vertex_coordinates(v)[0]) < 1e-9 for v in editor.mesh.vertices())


@pytest.fixture
def with_hexagon():
    editor = DenseMeshEditor(_rect(nx=4, ny=4))
    editor.remove_edge(_edge_at(editor.mesh, (1, 1), (1, 2)))  # a hexagon at column 0-1, row 1
    return editor


def test_a_strip_walk_stops_at_a_face_that_is_not_a_quad(with_hexagon):
    m = with_hexagon.mesh
    assert len(m.collect_strip(*_edge_at(m, (0, 1), (1, 1)))) == 2


def test_only_the_strips_reaching_a_hexagon_are_refused(with_hexagon):
    editor = with_hexagon
    blocked = _edge_at(editor.mesh, (0, 1), (1, 1))
    plan = editor.plan_strip_deletion(blocked)
    assert plan["ok"] is False and "sides" in plan["reason"]
    ok, notes = editor.remove_line(blocked)
    assert ok is False and notes["error"] == plan["reason"]
    clear = _edge_at(editor.mesh, (3, 0), (4, 0))
    plan = editor.plan_strip_deletion(clear)
    assert plan["ok"], plan.get("reason")
    ok, notes = editor.remove_line(clear)
    assert ok and _degrees(editor.mesh).get(6) == 1 and _simple(editor.mesh)


def test_adding_a_line_that_touches_a_hexagon_is_refused(with_hexagon):
    editor = with_hexagon
    ok, notes = editor.add_line(_edge_at(editor.mesh, (2, 0), (2, 1)))  # x = 2 runs past two hexagon corners
    assert ok is False and "not quads" in notes["error"]
    ok, notes = editor.add_line(_edge_at(editor.mesh, (3, 0), (3, 1)))
    assert ok and _degrees(editor.mesh).get(6) == 1


def test_a_strip_through_a_pole_triangle_is_refused_but_one_beside_it_is_not():
    poled = PseudoQuadMesh.from_vertices_and_faces(
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 2, 0], [1, 2, 0]], [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 7]]
    )
    poled.attributes["face_pole"] = {3: 5}
    editor = DenseMeshEditor(poled)
    assert editor.plan_strip_deletion((4, 7))["ok"] is False  # a side of the pole triangle
    assert editor.plan_strip_deletion((0, 3))["ok"]  # runs only beside it
    ok, _ = editor.remove_face(3)
    assert ok and editor.mesh.attributes["face_pole"] == {}


def test_add_line_opens_the_new_strip_without_asking():
    editor = DenseMeshEditor(_rect(), walls=_square_walls())
    faces_before = editor.mesh.number_of_faces()
    ok, notes = editor.add_line(_interior_edge(editor.mesh))
    assert ok, notes
    assert editor.mesh.number_of_faces() > faces_before and editor.non_quad_faces() == []
    assert notes.get("strip") is not None and notes == editor.last_addition
    pair = notes["new_vertices"]
    spread = max(abs(editor.mesh.vertex_coordinates(pair[0])[k] - editor.mesh.vertex_coordinates(pair[1])[k]) for k in range(2))
    assert spread > 1e-6


def test_a_wall_to_wall_line_keeps_the_boundary_on_its_walls_and_moves_nothing_else():
    # Its end pairs sit ON the walls, and used to be pulled ~0.3 off them when the
    # strip was opened by smoothing with those pairs released.
    editor = DenseMeshEditor(_rect(), walls=_square_walls())
    positions = {v: editor.mesh.vertex_coordinates(v) for v in editor.mesh.vertices()}
    ok, notes = editor.add_line(_edge_at(editor.mesh, (2, 1), (2, 2)))
    assert ok
    assert max(math.dist(editor.mesh.vertex_coordinates(v), editor.project_to_wall(editor.mesh.vertex_coordinates(v))) for v in editor.boundary_vertices()) < 1e-9
    moved = [v for v, xyz in positions.items() if v in editor.mesh.vertex and math.dist(xyz, editor.mesh.vertex_coordinates(v)) > 1e-9 and v not in notes["new_vertices"]]
    assert not moved
    before = editor.mesh
    ok, _ = editor.add_line((9998, 9999))
    assert ok is False and editor.mesh is before


def test_a_strip_along_a_closed_interior_ring():
    # the grammar used to raise KeyError here
    editor = DenseMeshEditor(_annulus())
    edge = _edge_at(editor.mesh, (2.0, 0.0), (2.0 * math.cos(math.pi / 6), 2.0 * math.sin(math.pi / 6)))
    ok, notes = editor.add_line(edge)
    assert ok and notes["closed"]
    assert editor.non_quad_faces() == [] and editor.mesh.is_manifold() and _simple(editor.mesh)
    stored = sorted(sorted(tuple(sorted(e)) for e in editor.mesh.strip_edges(s)) for s in editor.mesh.strips())
    fresh = editor.mesh.copy()
    fresh.collect_strips()
    assert stored == sorted(sorted(tuple(sorted(e)) for e in fresh.strip_edges(s)) for s in fresh.strips())


def test_a_strip_along_the_holes_own_boundary_ring():
    editor = DenseMeshEditor(_annulus())
    ok, notes = editor.add_line(_edge_at(editor.mesh, (1.0, 0.0), (math.cos(math.pi / 6), math.sin(math.pi / 6))))
    assert ok and editor.mesh.is_manifold() and _simple(editor.mesh), notes


def test_plan_then_remove_line():
    editor = DenseMeshEditor(_rect(), walls=_square_walls())
    edge = _interior_edge(editor.mesh)
    plan = editor.plan_strip_deletion(edge)
    assert editor.mesh.number_of_faces() == 12  # the plan mutates nothing
    assert plan["strip"] is not None and plan["faces"] > 0
    assert isinstance(plan["boundaries_lost"], int)
    ok, notes = editor.remove_line(edge)
    assert ok, notes
    assert plan["faces_after"] == editor.mesh.number_of_faces()
    assert notes == editor.last_deletion and notes["strip"] is not None
    ok, notes = editor.remove_line((9998, 9999))
    assert ok is False and "is not an edge" in notes["error"]


def test_plan_and_perform_agree_on_every_edge():
    base = _rect()
    performed = 0
    for edge in list(base.edges()):
        editor = DenseMeshEditor(base.copy())
        plan = editor.plan_strip_deletion(edge)
        ok, _ = editor.remove_line(edge)
        performed += bool(ok)
        assert bool(ok) == bool(plan["ok"]), edge
        assert not (ok and editor.non_quad_faces()), edge
    assert performed > 0


def test_faces_touching_only_at_a_corner_block_strip_edits_but_not_drawing():
    editor = DenseMeshEditor(_rect(nx=4, ny=4))
    editor.remove_face(editor.face_at([1.5, 1.5, 0.0]))
    ok, notes = editor.remove_face(editor.face_at([2.5, 2.5, 0.0]))
    assert ok
    assert notes["non_manifold"] == [_vkey_at(editor.mesh, 2, 2)]
    ok, notes = editor.remove_line(_edge_at(editor.mesh, (3, 0), (4, 0)))
    assert ok is False and "corner" in notes["error"]  # refuses instead of raising
    ok, notes = editor.add_line(_edge_at(editor.mesh, (4, 0), (4, 1)))
    assert ok is False and "corner" in notes["error"]
    assert editor.draw_edges([[-1.0, 0.5, 0.0], [5.0, 0.5, 0.0]])[0]


def test_remove_vertex_next_to_a_polygon_leaves_no_dangling_edge():
    # compas ``delete_vertex`` only clears edges touching the vertex's neighbours, so
    # the far side of a deleted HEXAGON used to stay behind with no face on either
    # side -- and ``vertices_on_boundaries`` walked such an edge forever.
    editor = DenseMeshEditor(_rect(nx=4, ny=4))
    editor.remove_edge(_edge_at(editor.mesh, (2, 3), (2, 4)))  # a hexagon in the top row
    ok, notes = editor.remove_vertex(_vkey_at(editor.mesh, 2, 3))
    assert ok, notes
    halfedge = editor.mesh.halfedge
    assert not [(u, v) for u in halfedge for v, f in halfedge[u].items() if f is None and halfedge[v].get(u) is None]
    assert all(any(f is not None for f in halfedge[v].values()) for v in editor.mesh.vertices())
    assert len(editor.boundary_vertices()) > 0 and editor.relax(iterations=2) is editor.mesh


def test_undo_steps_back_one_edit_at_a_time():
    editor = DenseMeshEditor(_rect())
    start = editor.mesh.number_of_faces()
    editor.push_undo()
    editor.remove_edge(_interior_edge(editor.mesh))
    editor.push_undo()
    editor.draw_edges([[-1.0, 0.5, 0.0], [5.0, 0.5, 0.0]])
    after_draw = editor.mesh.number_of_faces()
    ok, _ = editor.undo()
    assert ok and editor.mesh.number_of_faces() == start - 1 != after_draw
    editor.undo()
    assert editor.mesh.number_of_faces() == start
    assert editor.undo()[0] is False


def test_reset_restores_the_topology_and_clears_the_notes():
    editor = DenseMeshEditor(_rect())
    start = editor.mesh.number_of_faces()
    editor.add_line(_interior_edge(editor.mesh))
    editor.reset()
    assert editor.mesh.number_of_faces() == start
    assert editor.last_addition == {} and editor.last_reason == ""
