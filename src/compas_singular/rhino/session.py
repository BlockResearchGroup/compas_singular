"""The session of one Rhino document, stored inside the document and undone with Rhino's Ctrl+Z.

Commands edit a copy, assign it back, then call ``RhinoSession.record``.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import uuid
from typing import TYPE_CHECKING
from typing import Any

import compas
from compas_singular.rhino import document
from compas_singular.session import SingularSession

if TYPE_CHECKING:
    from compas.scene import Scene

try:
    import scriptcontext as sc  # type: ignore  # noqa: I001
except ImportError:
    sc = None

try:
    import rhinoscriptsyntax as rs  # type: ignore  # noqa: I001
except ImportError:
    rs = None


__all__ = ["RhinoSession"]


def display_options() -> dict[str, dict[str, Any]]:
    """How the permanent display draws each item, by item name."""
    from compas_singular.rhino.project import layer_path

    return {
        "coarse": dict(
            layer=layer_path("Mesh"), show_faces=True, joined=True, poles_layer=layer_path("Poles"), curves_layer=layer_path("EdgeCurves"), polylines_layer=layer_path("Polylines")
        ),
        "dense": dict(layer=layer_path("Dense"), show_faces=True, joined=True, show_vertices=False, show_edges=False),
    }


class RhinoSession(SingularSession):
    """A ``compas_singular.session.SingularSession`` stored in a Rhino document.

    Attributes
    ----------
    doc : Rhino.RhinoDoc
        The document this session belongs to.
    revision : str or None
        The revision of the snapshot this session was loaded from or last wrote.
        ``None`` for a document that holds no snapshot yet.
    """

    #: ``doc.RuntimeSerialNumber`` -> the session of that document.
    _sessions = {}

    #: The items that have a permanent display.
    DISPLAYED = ("coarse", "dense")

    #: The user-text key every drawn object carries, its value the item's name.
    #: Stored in the object, so it survives saving the ``.3dm`` and Ctrl+Z.
    ITEM_KEY = "compas_singular.item"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(RhinoSession, self).__init__(*args, **kwargs)
        self.doc = None
        self.revision = None
        self._scene = None
        self._drawn = {}  # item name -> the object the display shows

    @property
    def scene(self) -> Scene:
        """The compas ``Scene`` that draws this session's items. Never saved: it
        is rebuilt from the items (see ``compas_singular.session``)."""
        if self._scene is None:
            from compas.scene import Scene
            from compas_singular.rhino.scene import ensure_registered

            ensure_registered()
            self._scene = Scene(context="Rhino")
        return self._scene

    @classmethod
    def current(cls, doc: Any = None) -> RhinoSession:
        """The session of ``doc`` (default: the active document), reloaded when the document's revision differs."""
        doc = doc or sc.doc
        revision = document.read_revision(doc)
        session = cls._sessions.get(doc.RuntimeSerialNumber)
        if session is None or session.revision != revision:
            session = cls._load(doc, revision)
            cls._sessions[doc.RuntimeSerialNumber] = session
        return session

    @classmethod
    def _load(cls, doc: Any, revision: str | None) -> RhinoSession:
        session = cls()
        session.doc = doc
        session.revision = revision
        text = document.read_snapshot(doc) if revision else None
        if text:
            session.take(compas.json_loads(text))
        # The document already shows what it holds: Rhino restored the display
        # together with the snapshot, or it was drawn when the snapshot was.
        session._drawn = {name: getattr(session, name) for name in cls.DISPLAYED}
        return session

    def record(self, name: str) -> None:
        """Bring the display up to date and write this session into its document, inside the running command."""
        previous = rs.EnableRedraw(False)
        try:
            self.draw()
        finally:
            rs.EnableRedraw(previous)
        self.doc.Views.Redraw()
        self.revision = str(uuid.uuid4())
        document.write_snapshot(self.doc, self.revision, compas.json_dumps(self))

    # --------------------------------------------------------------------------
    # the permanent display
    # --------------------------------------------------------------------------

    def draw(self) -> None:
        """Redraw every displayed item whose object is not the one drawn last time."""
        options = display_options()
        for name in self.DISPLAYED:
            item = getattr(self, name)
            if name in self._drawn and self._drawn[name] is item:
                continue
            self._draw_item(name, item, options[name])
            self._drawn[name] = item

    def item_of(self, guid: Any) -> tuple[str, Any] | None:
        """The session item a document object shows, as ``(name, item)``, or ``None``."""
        name = rs.GetUserText(guid, self.ITEM_KEY)
        item = getattr(self, name) if name in self.DISPLAYED else None
        return (name, item) if item is not None else None

    def _draw_item(self, name: str, item: Any, options: dict[str, Any]) -> None:
        """Clear the item's layers and draw it there, every object tagged with
        its ``name``; with ``None``, only clear."""
        from compas_singular.rhino.project import ensure_layers

        ensure_layers()
        if item is None:
            import compas_rhino.layers

            for key in ("layer", "poles_layer", "curves_layer", "polylines_layer"):
                if options.get(key):
                    compas_rhino.layers.clear_layer(options[key], include_children=False, purge=False)
            return
        display = self.scene.add(item, **options)
        display.clear()
        for guid in display.draw():
            rs.SetUserText(guid, self.ITEM_KEY, name)
        self.scene.remove(display)
