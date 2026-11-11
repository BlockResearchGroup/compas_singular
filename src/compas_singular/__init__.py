"""Topology finding of singularities in quad meshes."""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os


__author__ = ["Robin Oval", "Casper Bladt"]
__copyright__ = "Copyright 2019 - Block Research Group, ETH Zurich"
__license__ = "MIT License"
__email__ = "rpho2@cam.ac.uk"

#: Set from the git tags by setuptools-scm, which writes ``_version.py`` on install.
try:
    from compas_singular._version import __version__
except ImportError:  # a source tree that was never installed
    __version__ = "0.0.0+unknown"


HERE = os.path.dirname(__file__)
HOME = os.path.abspath(os.path.join(HERE, "../.."))
DATA = os.path.abspath(os.path.join(HERE, "../../data"))
TEMP = os.path.abspath(os.path.join(HERE, "../../temp"))

__all__ = []

#: compas finds scene objects through this: ``compas_singular.rhino.scene``
#: registers how a coarse layout is drawn in Rhino (only when Rhino is present).
__all_plugins__ = ["compas_singular.rhino.scene"]
