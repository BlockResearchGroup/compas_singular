# Installation

`compas_singular` requires Python 3.9 or later.

## Stable

Install the latest release from PyPI with pip.

```bash
pip install compas_singular
```

Optional extras add `viewer` for the 3D viewer used by the examples,
`fd` for constrained smoothing with `compas_fd`,
and `rhino` for the Rhino 8 scene objects.

```bash
pip install "compas_singular[viewer]"
```

## Latest

Install the latest development version from GitHub.

```bash
pip install "compas_singular @ git+https://github.com/BlockResearchGroup/compas_singular.git@dev"
```

## Development

To work on the code, clone the repository and install it editable from local source,
with the `viewer` extra to run the examples.

```bash
git clone https://github.com/BlockResearchGroup/compas_singular.git
cd compas_singular
pip install -e ".[viewer]"
```

A conda development environment (COMPAS, the viewer, and an editable install with
the `dev`, `docs`, `fd` and `viewer` extras) is described in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate singular-dev
```

To build this documentation, install the `docs` extra and run the `docs` task,
or `mkdocs serve` for a live preview.

```bash
pip install -e ".[dev,docs]"
invoke docs
```

The HTML is written to `dist/docs`.

## Verify

```bash
python -c "import compas_singular; print(compas_singular.__file__)"
```

If this prints `None` instead of a path ending in `src/compas_singular/__init__.py`,
Python found the bare repository folder and treated it as an empty namespace package.
Run one of the install commands above.

## Rhino 8

The Rhino commands are in `rhino_plugin/commands/`, numbered in workflow order
(`CMD00_start` to `CMD09_edit_quad_mesh`).
They run on Rhino 8's CPython 3.9 and need `compas_singular` installed in Rhino's Python.
Close Rhino, then install the package with Rhino's own interpreter.

```bash
%USERPROFILE%\.rhinocode\py39-rh8\python.exe -m pip install "compas_singular[rhino,fd] @ git+https://github.com/BlockResearchGroup/compas_singular.git@dev"
```

The modules under `compas_singular.rhino` need `compas_rhino` and are only importable inside Rhino.
See [Rhino 8](tutorial/rhino.md) for the commands and the workflow they implement.
