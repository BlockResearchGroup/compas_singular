#! python3
# r: compas_singular

"""Development only: make the next command import compas_singular afresh.

Rhino keeps ``sys.modules`` between script runs, so after editing ``src`` the
commands would keep running the old code. Every command used to drop the modules
itself (the "bootstrap purge"); now that compas_singular is installed into
Rhino's Python (an editable install, so ``src`` IS the installed package), that
is this command's job, run by hand after an edit.

Safe to run at any time. The only state it drops is the in-memory copy of each
document's session, and the next command reads that back from the document.
"""
import sys


def main():
    names = [name for name in sys.modules
             if name == "compas_singular" or name.startswith("compas_singular.")]
    for name in names:
        del sys.modules[name]
    print("CMD_dev_reload: dropped {} compas_singular module(s); the next command "
          "imports them fresh from src.".format(len(names)))


if __name__ == "__main__":
    main()
