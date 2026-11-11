"""Fixtures shared by the MCP tests: an isolated spool and library, and a dense mesh."""

import json
import os

import pytest

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def data_path(name):
    return os.path.join(DATA, name)


@pytest.fixture
def mcp_spool(tmp_path, monkeypatch):
    """An isolated spool, so a test never touches the real one."""
    folder = tmp_path / "spool"
    folder.mkdir()
    monkeypatch.setenv("COMPAS_SINGULAR_MCP_SPOOL", str(folder))
    return str(folder)


@pytest.fixture
def mcp_library(tmp_path, monkeypatch):
    """An isolated library, so saving an example never writes into the repo."""
    folder = tmp_path / "library"
    folder.mkdir()
    monkeypatch.setenv("COMPAS_SINGULAR_MCP_LIBRARY", str(folder))
    return str(folder)


class McpServer(object):
    """A freshly built MCP server, called the way a client calls it."""

    def __init__(self):
        from compas_singular.mcp.server import build

        self.dispatcher = build()
        self.handler = self.dispatcher.handler
        self.session = self.handler.session

    def __call__(self, _tool, **arguments):
        # ``_tool`` is underscored because some tools take a ``name`` argument.
        reply = self.dispatcher.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": _tool, "arguments": arguments}})
        return json.loads(reply["result"]["content"][0]["text"])


@pytest.fixture
def mcp(mcp_spool, mcp_library):
    return McpServer()


def make_dense_mesh(density=3):
    """A dense quad mesh with something wrong with it, and its boundary loops."""
    from compas_singular.datastructures import CoarseQuadMesh
    from compas_singular.datastructures.mesh.smoothing import mesh_boundary_polylines

    coarse = CoarseQuadMesh.from_json(data_path("coarse_quad_mesh_british_museum.json"))
    coarse.collect_strips()
    coarse.set_strips_density(density)
    coarse.densification()
    mesh = coarse.get_quad_mesh()
    walls = [[list(p) for p in pl.points] for pl in mesh_boundary_polylines(mesh)]
    return mesh, walls


@pytest.fixture
def dense_mesh():
    return make_dense_mesh()
