"""compas_viewer scenes for the frame-field front end, one group per stage on an XY grid."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from compas.colors import Color
from compas.datastructures import Graph
from compas.geometry import Point
from compas.geometry import Polyline
from compas_singular.datastructures import QuadMesh

if TYPE_CHECKING:
    from compas_singular.datastructures import Mesh
    from compas_singular.framefield.field_decomposition import FieldDecomposition


__all__ = [
    "Grid",
    "translated",
    "add_background",
    "add_field",
    "add_boundary",
    "add_separatrices",
    "add_singularities",
    "add_layout",
    "add_dense",
    "add_guides",
]


#: singularities are read by colour: red where a quad mesh wants valence 3,
#: blue where it wants valence 5
POSITIVE = Color.red()
NEGATIVE = Color.blue()
SEPARATRIX = Color.orange()
WALL = Color.black()
FIELD = Color.grey()
GUIDE = Color.purple()
PATCH_EDGE = Color.from_rgb255(70, 110, 170)


def translated(mesh: Mesh, dx: float, dy: float, cls: type | None = None) -> Mesh:
    """A display-only copy of ``mesh`` moved in XY, without strip or pole attributes."""
    index = {vkey: i for i, vkey in enumerate(mesh.vertices())}
    vertices = []
    for vkey in mesh.vertices():
        x, y, z = mesh.vertex_coordinates(vkey)
        vertices.append([x + dx, y + dy, z])
    faces = [[index[v] for v in mesh.face_vertices(f)] for f in mesh.faces()]
    return (cls or QuadMesh).from_vertices_and_faces(vertices, faces)


def _shift(points: list[list[float]], dx: float, dy: float, dz: float = 0.0) -> list[list[float]]:
    return [[p[0] + dx, p[1] + dy, p[2] + dz] for p in points]


class Grid(object):
    """Panel placement, and a camera that frames the whole set."""

    def __init__(self, pitch: float = 16.0, cols: int = 3) -> None:
        self.pitch = pitch
        self.cols = cols
        self.used = 0

    def cell(self, i: int | None = None) -> tuple[float, float]:
        """Offset of panel ``i``. Rows run downwards, as they read."""
        if i is None:
            i = self.used
        self.used = max(self.used, i + 1)
        return (i % self.cols) * self.pitch, -(i // self.cols) * self.pitch

    def frame(self, viewer: Any, margin: float = 1.1) -> Grid:
        """Point the camera at the middle of the grid and pull it back.

        The default camera frames a single object, so a multi-panel scene opens
        zoomed into one corner of itself without this.
        """
        rows = max(1, (self.used + self.cols - 1) // self.cols)
        cols = min(self.cols, max(1, self.used))
        cx = (cols - 1) * self.pitch * 0.5 + self.pitch * 0.35
        cy = -(rows - 1) * self.pitch * 0.5 + self.pitch * 0.35
        # The camera distance has to cover the grid's HEIGHT through the vertical
        # field of view, not just equal the grid's diagonal -- at the default
        # ~45 degrees a distance equal to the span shows only about 0.8 of it,
        # which silently clips the bottom row of a 3x3 layout.
        extent = max(cols, rows) * self.pitch
        distance = extent * margin / (2.0 * 0.414)  # tan(22.5 deg)
        viewer.renderer.camera.target = [cx, cy, 0.0]
        # a small y offset keeps the view very slightly oblique; looking exactly
        # straight down leaves the up-vector undefined
        viewer.renderer.camera.position = [cx, cy - distance * 0.02, distance]
        return self


# ----------------------------------------------------------------------------
# layers
# ----------------------------------------------------------------------------


def add_background(group: Any, decomposition: FieldDecomposition, dx: float = 0.0, dy: float = 0.0) -> None:
    """The triangulation the field lives on."""
    group.add(translated(decomposition.background.mesh, dx, dy), show_points=False, show_lines=True, show_faces=True, opacity=0.25, name="background triangulation")


def add_field(
    group: Any,
    decomposition: FieldDecomposition,
    dx: float = 0.0,
    dy: float = 0.0,
    budget: int = 300,
) -> int:
    """Cross ticks, two arms per sample, capped at ``budget`` samples, as one ``Graph``."""
    mesh = decomposition.background.mesh
    step = max(1, mesh.number_of_vertices() // budget)
    # keeping every step-th vertex thins the sample spacing by ~sqrt(step)
    half = decomposition.background.target_length * 0.42 * (step**0.5)

    graph = Graph()
    n = 0
    for i, vkey in enumerate(mesh.vertices()):
        if i % step:
            continue
        x, y, _ = mesh.vertex_coordinates(vkey)
        for arm in decomposition.field.directions(vkey)[:2]:
            graph.add_node(n, x=x - arm[0] * half + dx, y=y - arm[1] * half + dy, z=0.0)
            graph.add_node(n + 1, x=x + arm[0] * half + dx, y=y + arm[1] * half + dy, z=0.0)
            graph.add_edge(n, n + 1)
            n += 2
    if n:
        group.add(graph, linecolor=FIELD, linewidth=1, show_points=False, name="cross field ({} ticks)".format(n // 2))
    return n // 2


def add_boundary(group: Any, decomposition: FieldDecomposition, dx: float = 0.0, dy: float = 0.0) -> None:
    """Outer wall and any holes."""
    bg = decomposition.background
    for i, loop in enumerate([bg.outer] + list(bg.inners)):
        group.add(Polyline(_shift(list(loop) + [loop[0]], dx, dy, 0.02)), linecolor=WALL, linewidth=3, name="outer wall" if i == 0 else "hole {}".format(i))


def add_separatrices(group: Any, decomposition: FieldDecomposition, dx: float = 0.0, dy: float = 0.0) -> int:
    """The traced field lines that cut the domain into patches."""
    _, others, _ = decomposition._build()
    for i, polyline in enumerate(others):
        group.add(Polyline(_shift(polyline, dx, dy, 0.04)), linecolor=SEPARATRIX, linewidth=4, name="separatrix {}".format(i))
    return len(others)


def add_singularities(
    group: Any,
    decomposition: FieldDecomposition,
    dx: float = 0.0,
    dy: float = 0.0,
    size: int = 20,
) -> None:
    """Field singularities, coloured by index sign, where the traces start."""
    positions = decomposition.tracer.singularity_positions()
    for fkey, k in decomposition.field.singularities():
        x, y, _ = positions[fkey]
        group.add(Point(x + dx, y + dy, 0.06), pointcolor=POSITIVE if k > 0 else NEGATIVE, pointsize=size, name="singularity {:+d} (wants valence {})".format(k, 4 - k))


def add_layout(
    group: Any,
    decomposition: FieldDecomposition,
    dx: float = 0.0,
    dy: float = 0.0,
    faces: bool = True,
) -> None:
    """The coarse quad layout: patches, their corners, and the separatrices."""
    mesh = decomposition.mesh
    if mesh is not None:
        group.add(translated(mesh, dx, dy), show_points=False, show_lines=True, show_faces=faces, opacity=0.55, linewidth=2, name="coarse patches")
        for vkey in mesh.vertices():
            x, y, _ = mesh.vertex_coordinates(vkey)
            group.add(Point(x + dx, y + dy, 0.05), pointcolor=PATCH_EDGE, pointsize=10, name="patch corner")
    add_separatrices(group, decomposition, dx, dy)
    add_boundary(group, decomposition, dx, dy)


def add_dense(group: Any, dense: Mesh, dx: float = 0.0, dy: float = 0.0, name: str = "dense quad mesh") -> None:
    """The densified quad mesh."""
    group.add(translated(dense, dx, dy), show_points=False, show_lines=True, show_faces=True, name=name)


def add_guides(group: Any, guides: list[Any], dx: float = 0.0, dy: float = 0.0, name: str = "cable") -> None:
    """Cables or force lines, drawn above everything else."""
    for i, curve in enumerate(guides or []):
        pts = getattr(curve, "points", curve)
        group.add(Polyline(_shift([list(p) for p in pts], dx, dy, 0.08)), linecolor=GUIDE, linewidth=6, name="{} {}".format(name, i) if len(guides) > 1 else name)
