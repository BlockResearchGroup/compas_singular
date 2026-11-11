import importlib
import inspect
import types

import pytest

import compas_singular
import compas_singular.datastructures as ds


def test_trivial():
    print(compas_singular.__version__)
    assert True


# ----------------------------------------------------------------------------
# package namespace
#
# ``from .foo import *`` binds ``foo`` as an attribute of the package. When the
# package __all__ is computed with ``dir()`` that module object gets re-exported
# and shadows any function or subpackage of the same name, which used to make
# ``compas_singular.datastructures.add_strip`` a module rather than a function.
# ----------------------------------------------------------------------------

PACKAGES = [
    "compas_singular.algorithms",
    "compas_singular.datastructures",
    "compas_singular.datastructures.lizard",
    "compas_singular.datastructures.mesh",
    "compas_singular.datastructures.mesh_quad",
    "compas_singular.datastructures.mesh_quad.grammar",
    "compas_singular.datastructures.mesh_quad_coarse",
    "compas_singular.datastructures.mesh_quad_pseudo",
    "compas_singular.datastructures.mesh_quad_pseudo_coarse",
    "compas_singular.datastructures.skeleton",
    "compas_singular.geometry",
    "compas_singular.topology",
    "compas_singular.utilities",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_subpackage_is_importable(name):
    importlib.import_module(name)


@pytest.mark.parametrize("name", PACKAGES)
def test_package_does_not_re_export_submodules(name):
    package = importlib.import_module(name)
    leaked = [n for n in getattr(package, "__all__", []) if isinstance(getattr(package, n, None), types.ModuleType)]
    assert leaked == []


@pytest.mark.parametrize("name", ["add_strip", "add_strips", "delete_strip", "delete_strips"])
def test_grammar_functions_are_callable(name):
    assert callable(getattr(ds, name))


@pytest.mark.parametrize("name", ["delete_strip", "delete_strips"])
def test_deletion_takes_no_preserve_boundaries_flag(name):
    """Pre-splitting to save a boundary is the CALLER's step, not the grammar's.

    It used to be a ``preserve_boundaries`` kwarg here, which meant the policy
    was decided both here and in ``MeshEditor._split_strips``. Callers now pair
    ``strips_to_split_to_prevent_boundary_collapse`` with ``split_strips``
    themselves -- see ``unused.twocoloring.delete_strips_preserving_boundaries``.
    """
    assert "preserve_boundaries" not in inspect.signature(getattr(ds, name)).parameters


def test_the_grammar_exports_what_a_caller_needs_to_preserve_boundaries():
    for name in ("strips_to_split_to_prevent_boundary_collapse", "split_strips"):
        assert callable(getattr(ds, name)), name


def test_datastructures_mesh_quad_is_the_package():
    assert hasattr(ds.mesh_quad, "__path__")
