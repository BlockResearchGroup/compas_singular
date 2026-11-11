"""The meshing tools as an MCP server: the client supplies the model, this package the mesh operations.

Runs standalone; ``compas_singular.mcp.bridge`` links to a live Rhino. Design notes: ``design_notes/mcp.md``.

Modules
=======

``protocol``
    JSON-RPC 2.0 over stdio. Standard library only.
``server``
    Binds the registry, the session and the library to the protocol's methods.
``registry``
    The ``@tool`` decorator and the JSON schemas it emits.
``session``
    Mesh, walls, guides, history, remarks, and a position-map undo.
``handle``
    Geometric addressing. No raw vertex key ever crosses the protocol.
``describe``
    Turns a metric dict into a reading a person or a model can act on.
``library``
    Guidance, prompts and the corpus of worked examples.
``bridge``
    The Rhino wire: a spool of JSON files, and the mesh encoding on it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__all__ = []
