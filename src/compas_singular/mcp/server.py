"""Bind the tools, the one session and the library to the protocol's methods.

Importing it registers the tools; ``build`` refuses to start on an empty registry.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any

from compas_singular import __version__
from compas_singular.mcp import library
from compas_singular.mcp import registry
from compas_singular.mcp import tools_checks  # noqa: F401  -- registers its tools
from compas_singular.mcp import tools_coarse  # noqa: F401  -- registers its tools
from compas_singular.mcp import tools_dense  # noqa: F401  -- registers its tools
from compas_singular.mcp import tools_io  # noqa: F401  -- registers its tools
from compas_singular.mcp import tools_library  # noqa: F401  -- registers its tools
from compas_singular.mcp import tools_mesh  # noqa: F401  -- registers its tools
from compas_singular.mcp import tools_rhino  # noqa: F401  -- registers its tools
from compas_singular.mcp.describe import describe
from compas_singular.mcp.protocol import Dispatcher
from compas_singular.mcp.session import MeshSession

__all__ = ["Handler", "build", "INSTRUCTIONS"]


INSTRUCTIONS = """Build, and improve, a quad mesh from the compas_singular library.

Two ways in. Pull a mesh from the open Rhino document with rhino_pull (from
the domain layers, or selection=true for whatever is selected), measure it with
inspect, improve it with relax / smooth_region / smooth_boundary_constrained /
smooth_guides / relax_fdm, change its topology with dense_add_line /
dense_remove_line, and push it back with rhino_push -- rhino_push_markers puts
dots on its worst faces and singularities for the person to see. Or build one from a domain: with a boundary
loaded (run check_inputs first -- the skeleton route builds wrong quietly),
create_coarse_mesh / coarse_set_density / coarse_densify take it from a
boundary, plus optional line and point features, to a dense quad mesh, and
coarse_add_strip / coarse_remove_strip / coarse_divide / coarse_move_corner /
coarse_set_pattern edit the coarse layout before that final step. coarse_inspect with image=true draws the layout
with every strip coloured and numbered -- look before editing a strip.
coarse_save / coarse_load keep a layout across sessions; rhino_push_coarse
hands it to the CMD_ commands in Rhino (CMD_densities, CMD_quad_mesh) and
rhino_pull_coarse brings an edited one back. Read guidance://coarse for that
sequence. compare reads two steps of history back side by side.

Read guidance://quality and guidance://smoothing before choosing a smoother --
they carry the measurements that decide which one is right, and the order to try
them in. inspect's share_below is what separates one bad face from a generally
slack mesh.

This version does not solve a frame field: a coarse layout comes from a
topological-skeleton decomposition, and a densified patch interior is blended
from its own four sides. Both meshes' topology can be edited: the coarse
layout's by adding, removing or dividing a strip (the last also along a drawn
curve), the dense mesh's by adding or removing a line -- which re-densifying
discards, and which refuses only a line whose strip crosses or reaches a face
that is not a quad (a pole's fan, say); every other line on the same mesh is
allowed. Reading a hand-drawn skeleton back in is not exposed yet."""


class Handler(object):
    """What ``compas_singular.mcp.protocol.Dispatcher`` asks for content."""

    def __init__(self, session: MeshSession | None = None) -> None:
        self.session = session if session is not None else MeshSession()

    # -- tools ----------------------------------------------------------

    def list_tools(self) -> list[dict[str, Any]]:
        return registry.schemas()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return registry.call(self.session, name, arguments)

    # -- resources ------------------------------------------------------

    def list_resources(self) -> list[dict[str, Any]]:
        resources = list(library.list_resources())
        # The live journal is not a file, so the library cannot list it.
        resources.insert(
            0,
            {
                "uri": "session://current",
                "name": "this session",
                "description": "What has been done so far: state, history, remarks and the current reading.",
                "mimeType": "text/markdown",
            },
        )
        return resources

    def read_resource(self, uri: str) -> str | None:
        if uri == "session://current":
            return self.render_session()
        return library.read_resource(uri)

    # -- prompts --------------------------------------------------------

    def list_prompts(self) -> list[dict[str, Any]]:
        return library.list_prompts()

    def get_prompt(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any] | None:
        return library.get_prompt(name, arguments)

    # -- the live journal -----------------------------------------------

    def render_session(self) -> str:
        """The session as Markdown, so it survives a context compaction.

        Deliberately the same shape as a stored example, so what a model reads
        about its own run and what it reads about a past one look alike.
        """
        session = self.session
        state = session.state()
        lines = ["# This session", ""]
        if not session.loaded:
            if session.coarse is not None:
                lines.append(
                    "No dense mesh is loaded, but a coarse layout is: {} face(s), {} strip(s). Call coarse_densify for a dense mesh, or coarse_inspect to read the layout.".format(
                        state["coarse_faces"], state["coarse_strips"]
                    )
                )
            elif session.walls:
                lines.append(
                    "No mesh is loaded, but a boundary is: {} wall(s). Call "
                    "create_coarse_mesh to build a coarse layout from it, or "
                    "rhino_pull again on a layer that has a mesh.".format(state["walls"])
                )
            else:
                lines.append("Nothing is loaded. Call rhino_pull to take a mesh out of the open Rhino document.")
            return "\n".join(lines)

        source = session.source or {}
        lines.append(
            "Mesh: {} faces, {} vertices, from {}.".format(
                state["faces"], state["vertices"], source.get("layer") or source.get("name") or source.get("kind") or "an unknown source"
            )
        )
        metrics = session.quality()
        told = describe(session.mesh, metrics, library.thresholds())
        lines += ["", "**Now:** {} -- {}".format(told["verdict"], told["reading"])]

        if session.history:
            lines += ["", "## Steps", ""]
            marks = {True: "improved all three", False: "did NOT improve all three", None: "no change"}
            for entry in session.history:
                lines.append("{}. `{}` -- {}{}".format(entry["step"] + 1, entry["action"], marks.get(entry.get("all_improved"), ""), " (UNDONE)" if entry.get("undone") else ""))
                for remark in session.remarks_for(entry["step"]):
                    lines.append("   - {}".format(remark))
        loose = [r for r in session.remarks if r["step"] >= len(session.history)]
        if loose:
            lines += ["", "## Remarks", ""]
            lines += ["- {}".format(r["text"]) for r in loose]
        return "\n".join(lines)


def build(session: MeshSession | None = None) -> Dispatcher:
    """A dispatcher wired to a handler. The one thing ``__main__`` needs."""
    handler = Handler(session)
    if not handler.list_tools():
        raise RuntimeError("no tools are registered -- the tools_* imports in server.py are what register them, so one has been dropped")
    return Dispatcher(handler, name="compas_singular.mcp", version=__version__, instructions=INSTRUCTIONS)
