"""The MCP mesh encoding that crosses to Rhino: plain JSON, no compas types."""

import json

import pytest
from compas.geometry import Point
from compas.geometry import Polyline
from conftest import data_path

from compas_singular.datastructures import CoarsePseudoQuadMesh
from compas_singular.datastructures import CoarseQuadMesh
from compas_singular.mcp.bridge import wire


def test_a_quad_mesh_round_trips_exactly():
    mesh = CoarseQuadMesh.from_json(data_path("coarse_quad_mesh_british_museum.json"))
    payload = wire.mesh_to_wire(mesh)
    assert all(len(v) == 3 for v in payload["vertices"])
    # faces index INTO the vertex list, not vertex keys
    assert all(all(isinstance(i, int) and 0 <= i < len(payload["vertices"]) for i in f) for f in payload["faces"])
    back = wire.mesh_from_wire(payload)
    assert back.number_of_vertices() == mesh.number_of_vertices()
    assert back.number_of_faces() == mesh.number_of_faces()
    assert wire.mesh_to_wire(back)["vertices"] == payload["vertices"]
    assert type(back).__name__ == "QuadMesh"


def test_poles_travel_as_points_because_keys_renumber():
    poled = CoarsePseudoQuadMesh.from_json(data_path("coarse_quad_mesh_british_museum_poles.json"))
    payload = wire.mesh_to_wire(poled)
    assert len(payload["poles"]) == len(list(poled.poles()))
    assert all(len(p) == 3 and isinstance(p[0], float) for p in payload["poles"])
    back = wire.mesh_from_wire(payload)
    assert type(back).__name__ == "CoarsePseudoQuadMesh"
    assert len(list(back.poles())) == len(list(poled.poles()))


def test_the_payload_is_plain_json():
    poled = CoarsePseudoQuadMesh.from_json(data_path("coarse_quad_mesh_british_museum_poles.json"))
    payload = wire.mesh_to_wire(poled)
    text = json.dumps(payload)
    assert json.loads(text) == payload
    assert "dtype" not in text and "compas" not in text


@pytest.mark.parametrize(
    "bad, want",
    [
        ({"vertices": [[0, 0, 0]], "faces": [[0, 1, 2, 3]]}, "refers to vertex"),
        ({"vertices": [], "faces": []}, "no vertices"),
        ({"faces": []}, "needs"),
        ({"vertices": [[0, 0, 0]], "faces": [[0, 1]]}, "at least 3"),
        ("nope", "not a dict"),
    ],
)
def test_a_malformed_payload_fails_where_the_fault_is(bad, want):
    with pytest.raises(wire.WireError, match=want):
        wire.mesh_from_wire(bad)


def test_curves_coerce_from_whatever_the_library_hands_around():
    expected = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]]
    assert wire.points_of(Polyline([[0, 0, 0], [1, 1, 0]])) == expected
    assert wire.points_of([Point(0, 0, 0), Point(1, 1, 0)]) == expected
    assert wire.points_of([[0, 0, 0], [1, 1, 0]]) == expected
    assert len(wire.curves_to_wire([[[0, 0, 0], [1, 0, 0]]])) == 1
    assert wire.curves_to_wire(None) == []
