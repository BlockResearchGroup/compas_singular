"""Where a session is kept in a Rhino document: a compressed snapshot on a hidden anchor object's attributes.

Changed via ``ModifyAttributes`` so Rhino's undo restores it; redo not yet verified in Rhino.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import base64
import zlib
from typing import Any

try:
    import Rhino  # type: ignore
except ImportError:
    Rhino = None


__all__ = ["ANCHOR_NAME", "ANCHOR_LAYER", "read_revision", "read_snapshot", "write_snapshot", "pack", "unpack"]


ANCHOR_NAME = "compas_singular.session"
ANCHOR_LAYER = "TopologyProblem::Session"

_REVISION = "revision"
_SNAPSHOT = "snapshot"


def read_revision(doc: Any) -> str | None:
    """The revision of the snapshot in ``doc``, or ``None`` if it holds none."""
    return _read(doc, _REVISION)


def read_snapshot(doc: Any) -> str | None:
    """The session JSON stored in ``doc``, or ``None`` if it holds none."""
    packed = _read(doc, _SNAPSHOT)
    return None if packed is None else unpack(packed)


def write_snapshot(doc: Any, revision: Any, text: str) -> None:
    """Store ``text`` in ``doc`` as ``revision``, inside the running command's undo record."""
    anchor = _anchor(doc) or _add_anchor(doc)
    attributes = anchor.Attributes.Duplicate()
    attributes.UserDictionary.Set(_REVISION, revision)
    attributes.UserDictionary.Set(_SNAPSHOT, pack(text))
    doc.Objects.ModifyAttributes(anchor, attributes, True)


def pack(text: str) -> str:
    """``text`` compressed into a plain ASCII string."""
    return base64.b64encode(zlib.compress(text.encode("utf-8"), 6)).decode("ascii")


def unpack(packed: str) -> str:
    """The inverse of ``pack``."""
    return zlib.decompress(base64.b64decode(packed)).decode("utf-8")


# ------------------------------------------------------------------------------
# the anchor
# ------------------------------------------------------------------------------


def _read(doc: Any, key: str) -> str | None:
    anchor = _anchor(doc)
    if anchor is None:
        return None
    dictionary = anchor.Attributes.UserDictionary
    if not dictionary.ContainsKey(key):
        return None
    return dictionary.GetString(key)


def _anchor(doc: Any) -> Any | None:
    """The anchor object in ``doc``, or ``None``. Found by name, hidden and locked included."""
    settings = Rhino.DocObjects.ObjectEnumeratorSettings()
    settings.NameFilter = ANCHOR_NAME
    settings.HiddenObjects = True
    settings.LockedObjects = True
    settings.DeletedObjects = False
    for anchor in doc.Objects.GetObjectList(settings):
        return anchor
    return None


def _add_anchor(doc: Any) -> Any:
    """A hidden text dot on a locked layer: nothing a user can select or delete by accident."""
    attributes = Rhino.DocObjects.ObjectAttributes()
    attributes.Name = ANCHOR_NAME
    attributes.LayerIndex = _session_layer(doc)
    attributes.Visible = False
    dot = Rhino.Geometry.TextDot("compas_singular session", Rhino.Geometry.Point3d.Origin)
    guid = doc.Objects.AddTextDot(dot, attributes)
    return doc.Objects.FindId(guid)


def _session_layer(doc: Any) -> int:
    """The index of ``ANCHOR_LAYER``, created (parents first) and locked if needed."""
    index = doc.Layers.FindByFullPath(ANCHOR_LAYER, -1)
    if index < 0:
        parent_id = None
        path = []
        for name in ANCHOR_LAYER.split("::"):
            path.append(name)
            index = doc.Layers.FindByFullPath("::".join(path), -1)
            if index < 0:
                layer = Rhino.DocObjects.Layer()
                layer.Name = name
                if parent_id is not None:
                    layer.ParentLayerId = parent_id
                index = doc.Layers.Add(layer)
            parent_id = doc.Layers[index].Id
    layer = doc.Layers[index]
    if not layer.IsLocked:
        layer.IsLocked = True
        layer.CommitChanges()
    return index
