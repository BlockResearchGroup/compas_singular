#! python3
# r: compas_singular

"""Development only: does Rhino's undo restore what the session stores in a document?

The session is kept as one snapshot on a hidden anchor object
(``compas_singular.rhino.document``), and Ctrl+Z / Ctrl+Y only restore it if
Rhino records that change. This checks exactly that code, plus document user
text (``doc.Strings``) for comparison. It loads ``document.py`` by its path, so
it imports nothing else of compas_singular.

**Run it in a new, empty document.** It uses its own anchor name and layer
(``UndoProbe``), so it never touches a real session. Run the command once per
step:

1. SetA, SetB, Ctrl+Z, Print  -> expect ``A`` / ``A``
2. SetA, SetB, Ctrl+Z, Ctrl+Y, Print  -> expect ``B`` / ``B``
3. SetBig, Save, close, reopen, Print  -> expect ``C`` and 2 000 001 characters
4. SetBig, SetA, Ctrl+Z, Print  -> expect ``C`` (a 2 MB value is undone as well)

Each line prints what the anchor holds and what the document strings hold.
"""
import base64
import importlib.util
import os

import rhinoscriptsyntax as rs
import scriptcontext as sc

DOCUMENT = os.path.join(
    r"C:\Users\Casper\libraries\carbcomn\compas_singular\compas_singular\src",
    "compas_singular", "rhino", "document.py")
KEY = "undo_probe"

spec = importlib.util.spec_from_file_location("undo_probe_document", DOCUMENT)
document = importlib.util.module_from_spec(spec)
spec.loader.exec_module(document)
document.ANCHOR_NAME = "compas_singular.undo_probe"
document.ANCHOR_LAYER = "UndoProbe"


def value_for(mode):
    if mode == "SetBig":
        # Random, so compression cannot shrink it: this really stores ~2 MB.
        return "C" + base64.b64encode(os.urandom(1500000)).decode("ascii")[:2000000]
    return {"SetA": "A", "SetB": "B"}[mode]


def main():
    mode = rs.GetString("Undo probe", "Print", ["SetA", "SetB", "SetBig", "Print"])
    if not mode:
        return
    doc = sc.doc
    if mode == "Print":
        anchor = document.read_snapshot(doc) or ""
        strings = doc.Strings.GetValue(KEY) or ""
        print("anchor: {!r} ({} characters, revision {!r}) | doc strings: {!r} ({} characters)"
              .format(anchor[:1], len(anchor), document.read_revision(doc),
                      strings[:1], len(strings)))
        return
    value = value_for(mode)
    document.write_snapshot(doc, mode, value)
    doc.Strings.SetString(KEY, value)
    print("{}: stored {} characters in both places".format(mode, len(value)))


if __name__ == "__main__":
    main()
