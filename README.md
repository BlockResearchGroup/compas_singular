# compas_singular

Welcome to **compas_singular**, a Python library tackles topology finding of patterns, particularly singularities in structured quad meshes.
Based on the PhD research of Robin Oval ([*Topology Finding of Patterns for Structural Design*](https://pastel.hal.science/tel-02917467), Université Paris-Est, 2019), this library implements data structures for quad mesh modelling, several algorithms for topological exploration and interface with Rhino3D.

**compas_singular** is based on the **COMPAS** framework, which is an open-source, Python-based framework for computational research and collaboration in architecture, engineering, digital fabrication and construction.

## Getting Started

### Installation

Python 3.9 or later.

```bash
pip install compas_singular
```

For the latest development version, install from GitHub:

```bash
pip install "compas_singular @ git+https://github.com/BlockResearchGroup/compas_singular.git@dev"
```

Optional extras: `[viewer]` for the 3D viewer used by the examples, `[fd]` for
constrained smoothing (compas_fd), and `[rhino]` for the Rhino 8 scene objects
(compas_rui).

To work on the code, install it editable from the root of this repository (the
folder containing `pyproject.toml`):

```bash
# runtime only
pip install -e .

# runtime + the 3D viewer used by the examples
pip install -e .[viewer]
```

Prefer conda? `environment.yml` describes a development environment: COMPAS,
the viewer, and an editable install with the `dev`, `fd` and `viewer` extras:

```bash
conda env create -f environment.yml
conda activate singular-dev
```

Verify with:

```bash
python -c "import compas_singular; print(compas_singular.__file__)"
```

If that prints `None` instead of a path ending in
`src/compas_singular/__init__.py`, you are not running an installed copy —
Python found the bare repository folder and treated it as an empty namespace
package. Run one of the install commands above.

All dependencies are declared in `pyproject.toml`: the core ones under
`dependencies`, and the `viewer`, `fd`, `rhino`, `dev` and `docs` extras under
`[project.optional-dependencies]`.
The modules under `src/compas_singular/rhino/` additionally need
`compas_rhino` and are only importable inside Rhino.

### Quick start

A rectangle with a round hole, meshed by both routes:

```python
import math

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.framefield import FieldDecomposition

outer = [[-6, -4, 0], [6, -4, 0], [6, 4, 0], [-6, 4, 0]]
hole = [[1.6 * math.cos(2 * math.pi * i / 64), 1.6 * math.sin(2 * math.pi * i / 64), 0] for i in range(64)]

# Skeleton route: boundary -> coarse quad layout -> densities -> dense quad mesh.
coarse = SkeletonDecomposition.from_boundary(outer, inner_boundaries=[hole], target_length=0.5).coarse_mesh()
coarse.collect_strips()
coarse.set_strips_density_target(0.5)
mesh = coarse.densify()
print(mesh.number_of_faces(), 'quads')

# Frame-field route: the same domain, the layout traced from a cross field.
decomposition = FieldDecomposition.from_boundary(outer, inner_boundaries=[hole], target_length=0.5)
mesh = decomposition.quad_mesh(target_length=0.5)
print(mesh.number_of_faces(), 'quads on route', decomposition.route())
```

[`examples/`](https://github.com/BlockResearchGroup/compas_singular/tree/dev/examples) has thirteen worked examples, from this workflow to layout
editing, patterns, guides, symmetry and dual blocks.

### Rhino 8

The Rhino commands are in `rhino_plugin/commands/`, numbered in workflow order
(`CMD00_start` to `CMD09_edit_quad_mesh`). They run on Rhino 8's CPython 3.9
and need `compas_singular` installed in Rhino's Python. Until the package is
published, install it with Rhino's interpreter (Rhino closed):

```bash
%USERPROFILE%\.rhinocode\py39-rh8\python.exe -m pip install "compas_singular[rhino,fd] @ git+https://github.com/BlockResearchGroup/compas_singular.git@dev"
```

The [Rhino page](https://blockresearchgroup.github.io/compas_singular/tutorial/rhino/) of the documentation describes the workflow and lists the commands.

## First Steps

The [documentation](https://blockresearchgroup.github.io/compas_singular/) has an [overview](https://blockresearchgroup.github.io/compas_singular/tutorial/overview/) of the workflow, the [examples](https://blockresearchgroup.github.io/compas_singular/examples/) with pictures, and the [API reference](https://blockresearchgroup.github.io/compas_singular/reference/compas_singular.algorithms/).

## Questions and feedback

Open an issue on the [issue tracker](https://github.com/BlockResearchGroup/compas_singular/issues).

## Issue tracker

Bugs and feature requests: <https://github.com/BlockResearchGroup/compas_singular/issues>.
Please include a script that reproduces the problem.

## Contributing

See [CONTRIBUTING.md](https://github.com/BlockResearchGroup/compas_singular/blob/dev/CONTRIBUTING.md).

## Changelog

See [CHANGELOG.md](https://github.com/BlockResearchGroup/compas_singular/blob/dev/CHANGELOG.md), which also lists what changed since the
2022 version (commit `66024561`).

## License

**compas_singular** is [released under the MIT license](LICENSE).
