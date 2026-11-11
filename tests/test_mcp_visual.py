"""The MCP renderer, the image content block, and the look-before-you-deliver gate."""

import base64
import json
import os
import struct
import zlib

import pytest
from conftest import data_path

from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.datastructures.mesh.smoothing import mesh_boundary_polylines
from compas_singular.mcp import render
from compas_singular.mcp.session import MeshSession

PNG = b"\x89PNG\r\n\x1a\n"


def _poled_mesh():
    coarse = CoarsePseudoQuadMesh.from_json(data_path("coarse_quad_mesh_british_museum_poles.json"))
    coarse.collect_strips()
    coarse.set_strips_density(3)
    coarse.densification()
    return coarse.get_quad_mesh()


@pytest.fixture(scope="module")
def domain():
    mesh = _poled_mesh()
    walls = [[list(p) for p in pl.points] for pl in mesh_boundary_polylines(mesh)]
    poles = [list(mesh.vertex_coordinates(v)) for v in mesh.poles()]
    return mesh, walls, poles


@pytest.fixture
def session(domain):
    mesh, walls, poles = domain
    return MeshSession().adopt(mesh, walls=walls, points=poles)


@pytest.fixture
def served(mcp, domain):
    mesh, walls, poles = domain
    mcp.session.adopt(mesh, walls=walls, points=poles, source={"kind": "file", "name": "x"})
    return mcp


def _png_size(data):
    """Width and height straight out of the IHDR chunk."""
    assert data[:8] == PNG
    return struct.unpack(">II", data[16:24])


def _orange_pixels(session):
    """How much of the wall colour is visible. The whole point of the drawing."""
    data = render.render_png(session, 300, 300)
    raw = zlib.decompress(data[data.index(b"IDAT") + 4 : -12])
    width, height = _png_size(data)
    stride = width * 3 + 1
    wall = render.COLORS["wall"]
    seen = 0
    for y in range(height):
        row = raw[y * stride + 1 : (y + 1) * stride]
        for x in range(width):
            r, g, b = row[x * 3], row[x * 3 + 1], row[x * 3 + 2]
            if abs(r - wall[0]) < 40 and abs(g - wall[1]) < 40 and abs(b - wall[2]) < 40:
                seen += 1
    return seen


def _raw(mcp, _tool, **arguments):
    return mcp.dispatcher.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": _tool, "arguments": arguments}})["result"]


def test_it_produces_a_real_png(session):
    data = render.render_png(session, 400, 400)
    assert data and data[:8] == PNG
    assert _png_size(data) == (400, 400)
    assert zlib.decompress(data[data.index(b"IDAT") + 4 : -12])
    assert data[-8:-4] == b"IEND"
    assert base64.b64decode(render.render_png_base64(session, 200, 200)) is not None


def test_sizes_are_clamped_not_obeyed_blindly(session):
    assert _png_size(render.render_png(session, 9000, 9000)) == (render.MAX_SIZE, render.MAX_SIZE)
    assert _png_size(render.render_png(session, 1, 1)) == (160, 160)


def test_an_empty_session_draws_nothing_rather_than_a_white_square():
    assert render.render_png(MeshSession()) is None
    assert render.render_png_base64(MeshSession()) is None


def test_orange_appears_only_when_the_mesh_leaves_its_wall(session, domain):
    _, walls, poles = domain
    on_wall = _orange_pixels(session)
    assert on_wall < 200
    drifted = _poled_mesh()
    n = drifted.number_of_vertices()
    cx = sum(drifted.vertex_coordinates(v)[0] for v in drifted.vertices()) / n
    cy = sum(drifted.vertex_coordinates(v)[1] for v in drifted.vertices()) / n
    for v in drifted.vertices_on_boundary():
        x, y, z = drifted.vertex_coordinates(v)
        drifted.vertex_attributes(v, "xyz", [x + (cx - x) * 0.10, y + (cy - y) * 0.10, z])
    off_wall = _orange_pixels(MeshSession().adopt(drifted, walls=walls, points=poles))
    assert off_wall > on_wall * 5


def test_inspect_without_image_is_only_text(served):
    plain = _raw(served, "inspect")
    assert [c["type"] for c in plain["content"]] == ["text"]
    assert "_image" not in plain["content"][0]["text"]


def test_inspect_with_image_appends_an_image_block(served):
    shown = _raw(served, "inspect", image=True)
    assert [c["type"] for c in shown["content"]] == ["text", "image"]
    image = shown["content"][1]
    assert image["mimeType"] == "image/png"
    assert base64.b64decode(image["data"])[:8] == PNG
    body = json.loads(shown["content"][0]["text"])
    assert "_image" not in body
    assert "orange" in body["image_legend"]
    assert all(word in body["look_at_this"] for word in ("BOUNDARIES", "POINT FEATURES", "GUIDES", "POLYEDGE"))
    assert body["inputs"] == {"walls": 2, "guides": 0, "point_features": 4}
    sized = _raw(served, "inspect", image=True, size=300)["content"][1]["data"]
    assert _png_size(base64.b64decode(sized)) == (300, 300)


def test_nothing_is_delivered_unlooked_at(served, tmp_path):
    out = str(tmp_path / "out.json")
    assert served.session.visually_current is False
    refused = served("save_mesh", path=out)
    assert refused.get("ok") is False
    assert refused.get("fix", {}).get("arguments") == {"image": True}
    assert "image=true" in refused["reason"]
    assert not os.path.exists(out)
    assert served("rhino_push", timeout=0.2).get("reason", "").startswith("this mesh has changed")
    served("inspect")
    assert served.session.visually_current is False  # WITHOUT the image does not count
    served("inspect", image=True)
    assert served.session.visually_current is True
    assert served("save_mesh", path=out).get("ok") is True
    assert os.path.isfile(out)


def test_any_change_to_the_mesh_invalidates_the_last_look(served, tmp_path):
    out = str(tmp_path / "out.json")
    served("inspect", image=True)
    served("relax")
    assert served.session.visually_current is False
    assert served("save_mesh", path=out).get("ok") is False
    served("inspect", image=True)
    served("undo")
    assert served.session.visually_current is False  # an undo moves every vertex
    served("inspect", image=True)
    assert served("save_mesh", path=out).get("ok") is True
    assert served("inspect")["state"]["visually_current"] is True
