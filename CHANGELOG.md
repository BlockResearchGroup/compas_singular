# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

First release on PyPI. It picks up from commit `66024561` (April 2022), the last commit before this rewrite.

### Added

* Frame-field route: `compas_singular.framefield.FieldDecomposition` solves a cross field,
  traces separatrices into a coarse layout and densifies along the field. It falls back to
  simpler layouts when a step fails and reports that through `warnings()`, `route()` and
  `quality()`.
* Curve (line) and point features on the skeleton route; curved boundaries and
  `edges_to_curves` carried through densification.
* Face patterns (`ortho`, `diagonal`, `fan`) with automatic strip-density reconciliation.
* Symmetry: `find_symmetry`, `symmetry_unit` and `expand_symmetrically` on either route.
* Layout and mesh editing without CAD: `compas_singular.editing` (`CoarseEditor`,
  `DenseMeshEditor`) and one strip grammar for adding and removing strips.
* Coarse layouts from drawn polylines (`from_coarse_polylines`).
* Constrained smoothing with sliding boundaries, and optional compas_fd smoothing (`[fd]` extra).
* `SingularSession` and the typed settings model (`compas_singular.session`, `compas_singular.settings`).
* Rhino 8 support: `compas_singular.rhino` (session, scene objects, project layers) and the
  command scripts in `rhino_plugin/commands/`.
* An MCP server (`python -m compas_singular.mcp`) exposing the workflow as tools.
* Extras `[viewer]`, `[fd]` and `[rhino]`.

### Changed

* Ported to COMPAS 2 (tested with compas 2.15.1). Requires Python 3.9 or later.
* `numpy` and `scipy` are declared dependencies; `networkx` is no longer one.
* The version comes from the git tags (setuptools-scm) and starts at 0.2.0, above upstream's last
  release (0.1.5). Every commit after a tag has its own version, so `pip install -U` from git updates.
* Packaging follows compas_model: dependencies, pytest, bump-my-version and ruff are configured in
  `pyproject.toml`; `tasks.py` uses `compas_invocations2`; `environment.yml` creates the `singular-dev`
  conda environment. `setup.py`, `setup.cfg`, `pytest.ini`, `.bumpversion.cfg` and the `requirements*.txt`
  files are removed.
* The documentation is built with MkDocs (Material + mkdocstrings) instead of Sphinx, and deployed by
  `mkdocs gh-deploy` on every push to `main`; PRs to `main` run `mkdocs build --strict`. The site moved from
  `.../compas_singular/latest/` to `.../compas_singular/`, and `latest/` redirects there.
* Docstrings are plain numpy style: the Sphinx roles (`:class:` etc.) and the `autosummary` listings in the
  `__init__.py` files are gone, and 19 parameter names that did not match their signatures are fixed.

### Removed

* COMPAS 1 Rhino modules (`rhino.artists`, `rhino.grasshopper`, `rhino.helpers.tna`).
* The singularity-block and guide-line modules.
* The empty `rhino.objects` package.
* `algorithms.unused` and `rhino.geometry` are no longer shipped in the package; `rhino.geometry`
  returns once it is ported to COMPAS 2.
