"""Render the mesh and its inputs to PNG with the standard library only, so the model can look at it.

Walls and guides are drawn under the mesh, so where the mesh left its input the colour shows.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import base64
import struct
import zlib
from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

if TYPE_CHECKING:
    from compas.datastructures import Mesh
    from compas.geometry import Polyline
    from compas_singular.datastructures import CoarsePseudoQuadMesh
    from compas_singular.mcp.session import MeshSession


__all__ = ["LEGEND", "COARSE_LEGEND", "render_png", "render_png_base64", "scene", "coarse_scene", "strip_representative_edge", "Raster"]


#: Supersampling factor. 2 is enough to read a kink; 3 costs four times the
#: memory for a difference nobody has been able to point at.
SCALE = 2

#: Default canvas. Image cost to a model runs with width times height, so this
#: is deliberately modest -- large enough to see a wandering polyedge on a mesh
#: of a few hundred faces, small enough not to dominate a turn.
DEFAULT_SIZE = 800

#: Refuse to render larger than this. A caller that asks for 4000 is not going
#: to read four times as much; it is going to spend four times as many tokens.
MAX_SIZE = 1400

WHITE = (255, 255, 255)

#: What each colour means. Returned WITH the image, because rendering a legend
#: would mean rendering glyphs, which would mean shipping a font.
LEGEND = (
    "orange = the input boundary curves (outer and holes); "
    "green = the input guide curves; "
    "magenta ring = an input point feature; "
    "magenta disc = a pole in the mesh; "
    "red disc = an irregular interior vertex (a singularity); "
    "black = the mesh outline; "
    "grey = the mesh interior edges."
)

#: Stroke widths, in supersampled pixels. The ORDER of these two matters more
#: than either value: see ``scene``.
WALL_WIDTH = 5
OUTLINE_WIDTH = 9
GUIDE_WIDTH = 5

COLORS = {
    "wall": (245, 150, 45),
    "guide": (40, 165, 95),
    "point": (205, 55, 190),
    "pole": (205, 55, 190),
    "singularity": (225, 45, 45),
    "edge": (125, 125, 135),
    "outline": (25, 25, 35),
    "pattern": (150, 185, 230),
    "label": (0, 0, 0),
}

#: One colour per strip on a coarse picture, named so the payload can say which
#: is which. Saturated and mutually distinct; orange, green and magenta are left
#: out because walls, guides and poles already mean those.
STRIP_PALETTE = (
    ("blue", (40, 90, 220)),
    ("red", (215, 40, 40)),
    ("teal", (0, 150, 150)),
    ("purple", (120, 60, 180)),
    ("brown", (140, 85, 35)),
    ("navy", (20, 30, 110)),
    ("olive", (120, 125, 20)),
    ("grey", (110, 110, 110)),
    ("crimson", (150, 10, 60)),
    ("sky", (60, 160, 230)),
)

COARSE_LEGEND = (
    "thick coloured lines = coarse edges, ONE COLOUR PER STRIP (strip_colors "
    "says which); a black number in a white box = that strip's skey, written "
    "on the edge coarse_inspect lists for it; light-blue lines inside a patch = "
    "its dense pattern (lines from the centre to the CORNERS = diagonal, from "
    "the centre to the side MIDPOINTS = fan, nothing = ortho); orange = the "
    "input boundary curves, drawn "
    "underneath -- on a CURVED wall orange beside a straight coarse edge is "
    "expected, since densifying follows the wall; green = guides; magenta ring "
    "= a point feature; magenta disc = a pole of the layout."
)

#: 3x5 bitmap digits, so a strip can be labelled by its key without a font.
#: Only digits and a minus sign: strip keys are integers, and that is all a
#: label here ever needs to say.
_GLYPHS = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "001", "001", "001"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "-": ("000", "000", "111", "000", "000"),
}

#: Supersampled pixels per glyph cell. 6 makes a digit 9 by 15 output pixels.
GLYPH_CELL = 6


class Raster(object):
    """An RGB byte buffer that can draw thick lines and discs, and emit a PNG."""

    def __init__(self, width: int, height: int, background: tuple[int, int, int] = WHITE) -> None:
        self.width = int(width)
        self.height = int(height)
        self.pixels = bytearray(bytes(background) * (self.width * self.height))

    def _put(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            i = (y * self.width + x) * 3
            self.pixels[i] = color[0]
            self.pixels[i + 1] = color[1]
            self.pixels[i + 2] = color[2]

    def disc(self, centre: Sequence[float], radius: float, color: tuple[int, int, int]) -> None:
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        radius = int(round(radius))
        squared = radius * radius
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= squared:
                    self._put(cx + dx, cy + dy, color)

    def ring(self, centre: Sequence[float], radius: float, color: tuple[int, int, int], thickness: int = 2) -> None:
        cx, cy = int(round(centre[0])), int(round(centre[1]))
        radius = int(round(radius))
        outer = radius * radius
        inner = max(0, radius - thickness) ** 2
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                d = dx * dx + dy * dy
                if inner <= d <= outer:
                    self._put(cx + dx, cy + dy, color)

    def rect(self, x0: float, y0: float, x1: float, y1: float, color: tuple[int, int, int]) -> None:
        for y in range(int(round(y0)), int(round(y1))):
            for x in range(int(round(x0)), int(round(x1))):
                self._put(x, y, color)

    def text(self, centre: Sequence[float], text: Any, color: tuple[int, int, int], cell: int = GLYPH_CELL) -> None:
        """Digits centred on a point, on a white box so they read over lines."""
        glyphs = [_GLYPHS[ch] for ch in str(text) if ch in _GLYPHS]
        if not glyphs:
            return
        width = (len(glyphs) * 4 - 1) * cell
        height = 5 * cell
        x0 = centre[0] - width / 2.0
        y0 = centre[1] - height / 2.0
        pad = cell
        self.rect(x0 - pad, y0 - pad, x0 + width + pad, y0 + height + pad, WHITE)
        for index, glyph in enumerate(glyphs):
            left = x0 + index * 4 * cell
            for row, bits in enumerate(glyph):
                for col, bit in enumerate(bits):
                    if bit == "1":
                        self.rect(left + col * cell, y0 + row * cell, left + (col + 1) * cell, y0 + (row + 1) * cell, color)

    def line(self, a: Sequence[float], b: Sequence[float], color: tuple[int, int, int], width: int = 1) -> None:
        """A thick segment, stamped as discs along a DDA walk."""
        x0, y0 = a[0], a[1]
        x1, y1 = b[0], b[1]
        dx, dy = x1 - x0, y1 - y0
        steps = int(max(abs(dx), abs(dy))) + 1
        radius = max(0, (width - 1) // 2)
        for i in range(steps + 1):
            t = i / float(steps)
            x, y = x0 + dx * t, y0 + dy * t
            if radius:
                self.disc((x, y), radius, color)
            else:
                self._put(int(round(x)), int(round(y)), color)

    def downsample(self, factor: int) -> Raster:
        """Box-filter to 1/factor. This is where the anti-aliasing comes from."""
        if factor <= 1:
            return self
        width, height = self.width // factor, self.height // factor
        out = Raster(width, height)
        area = factor * factor
        for y in range(height):
            for x in range(width):
                r = g = b = 0
                for sy in range(factor):
                    row = (y * factor + sy) * self.width
                    for sx in range(factor):
                        i = (row + x * factor + sx) * 3
                        r += self.pixels[i]
                        g += self.pixels[i + 1]
                        b += self.pixels[i + 2]
                out._put(x, y, (r // area, g // area, b // area))
        return out

    def to_png(self) -> bytes:
        """The buffer as PNG bytes: IHDR, one IDAT, IEND."""
        stride = self.width * 3
        raw = bytearray()
        for y in range(self.height):
            raw.append(0)  # filter type 0, none
            raw += self.pixels[y * stride : (y + 1) * stride]

        def chunk(tag: bytes, payload: bytes) -> bytes:
            body = tag + payload
            return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")


# ==============================================================================
# what to draw
# ==============================================================================


def _mesh_edges(mesh: Mesh) -> tuple[list[Any], list[Any]]:
    """``(interior, outline)`` edge lists, so the mesh's own border reads darker."""
    interior, outline = [], []
    for u, v in mesh.edges():
        segment = (mesh.vertex_coordinates(u), mesh.vertex_coordinates(v))
        try:
            on_boundary = mesh.is_edge_on_boundary(u, v)
        except Exception:
            on_boundary = False
        (outline if on_boundary else interior).append(segment)
    return interior, outline


def _singular_points(mesh: Mesh) -> tuple[list[Any], list[Any]]:
    """Irregular interior vertices, and poles, separately.

    Poles are separate because a pole is what an input point feature is supposed
    to have BECOME -- drawing the two alike would hide the thing worth checking.
    """
    poles = []
    getter = getattr(mesh, "poles", None)
    if getter is not None:
        try:
            poles = [mesh.vertex_coordinates(v) for v in getter()]
        except Exception:
            poles = []
    pole_keys = set()
    if getter is not None:
        try:
            pole_keys = set(getter())
        except Exception:
            pole_keys = set()
    irregular = []
    for vertex in mesh.vertices():
        if vertex in pole_keys:
            continue
        try:
            if mesh.is_vertex_on_boundary(vertex):
                continue
            if len(mesh.vertex_neighbors(vertex)) != 4:
                irregular.append(mesh.vertex_coordinates(vertex))
        except Exception:
            continue
    return irregular, poles


def _points_of(curve: Polyline | Sequence[Any]) -> list[list[float]]:
    points = getattr(curve, "points", None)
    if points is None:
        points = curve
    return [[float(p[0]), float(p[1])] for p in points]


def scene(session: MeshSession) -> list[dict[str, Any]]:
    """The drawable layers, back to front, with walls and guides under the mesh."""
    mesh = session.mesh
    layers = []

    # WALL_WIDTH < OUTLINE_WIDTH is the whole trick, and getting it backwards
    # makes the picture lie: a wall drawn thicker than the mesh outline shows
    # orange along every edge even where the mesh sits exactly on it, and a
    # reader is then told to report a deviation on every mesh. Drawn narrower
    # and underneath, the orange is completely hidden wherever the mesh is on
    # its boundary, and appears ONLY where the mesh has actually drifted off.
    for wall in session.walls:
        layers.append({"kind": "path", "points": _points_of(wall), "color": COLORS["wall"], "width": WALL_WIDTH})
    for guide in session.guides:
        layers.append({"kind": "path", "points": _points_of(guide), "color": COLORS["guide"], "width": GUIDE_WIDTH})

    if mesh is not None:
        interior, outline = _mesh_edges(mesh)
        layers.append({"kind": "lines", "segments": interior, "color": COLORS["edge"], "width": 1})
        layers.append({"kind": "lines", "segments": outline, "color": COLORS["outline"], "width": OUTLINE_WIDTH})

        irregular, poles = _singular_points(mesh)
        layers.append({"kind": "discs", "points": poles, "color": COLORS["pole"], "radius": 11})
        layers.append({"kind": "discs", "points": irregular, "color": COLORS["singularity"], "radius": 10})

    # Input point features last, as rings, so a respected one reads as a
    # magenta disc sitting inside a magenta ring and an ignored one as an
    # empty ring.
    layers.append({"kind": "rings", "points": [list(p) for p in session.points], "color": COLORS["point"], "radius": 19})
    return layers


def _rounded(point: Sequence[float]) -> tuple[float, float]:
    return (round(point[0], 3), round(point[1], 3))


def strip_representative_edge(coarse: CoarsePseudoQuadMesh, skey: Any) -> tuple[Any, Any]:
    """The edge a strip is named by: its first non-collapsed edge."""
    edges = coarse.strip_edges(skey)
    for u, v in edges:
        if u != v:
            return u, v
    return edges[0]


def coarse_scene(session: MeshSession) -> tuple[list[dict[str, Any]], dict[Any, str]]:
    """The coarse layout's drawable layers, one colour per strip. ``(layers, strip_colors)``.

    Returns
    -------
    tuple
        ``(layers, strip_colors)`` -- ``strip_colors`` is ``{skey: name}``.
    """
    coarse = session.coarse
    layers = []
    for wall in session.walls:
        layers.append({"kind": "path", "points": _points_of(wall), "color": COLORS["wall"], "width": WALL_WIDTH})
    for guide in session.guides:
        layers.append({"kind": "path", "points": _points_of(guide), "color": COLORS["guide"], "width": GUIDE_WIDTH})

    patterns = coarse.attributes.get("dense_pattern") or {}
    marks = []
    for fkey in coarse.faces():
        pattern = patterns.get(fkey, "ortho")
        corners = [coarse.vertex_coordinates(v) for v in coarse.face_vertices(fkey)]
        centre = [sum(c[i] for c in corners) / len(corners) for i in range(3)]
        # Centre-to-corners for diagonal, centre-to-midsides for fan: the two
        # read apart at a glance, and both work on a three-cornered pole patch,
        # where "both diagonals" has no meaning to draw.
        if pattern == "diagonal":
            marks += [(centre, corner) for corner in corners]
        elif pattern == "fan":
            for a, b in zip(corners, corners[1:] + corners[:1]):
                if a != b:
                    marks.append((centre, [(a[i] + b[i]) / 2.0 for i in range(3)]))
    layers.append({"kind": "lines", "segments": marks, "color": COLORS["pattern"], "width": 3})

    # An edge a cut gave a shape is drawn along it: a curved cut drawn as its
    # chord looks exactly like one that snapped straight. Only the shapes stored
    # ON the layout, so drawing never has to ask the decomposition, which snaps.
    shaped = {}
    for curve in coarse.attributes.get("user_curves") or []:
        if len(curve) > 2:
            ends = (_rounded(curve[0]), _rounded(curve[-1]))
            shaped[ends] = curve
            shaped[ends[::-1]] = list(reversed(curve))

    strip_colors, labels = {}, []
    for index, skey in enumerate(sorted(coarse.strips())):
        name, color = STRIP_PALETTE[index % len(STRIP_PALETTE)]
        strip_colors[skey] = name
        edges = [(u, v) for u, v in coarse.strip_edges(skey) if u != v]
        segments = []
        for u, v in edges:
            pu, pv = coarse.vertex_coordinates(u), coarse.vertex_coordinates(v)
            curve = shaped.get((_rounded(pu), _rounded(pv)), [pu, pv])
            segments += list(zip(curve, curve[1:]))
        layers.append({"kind": "lines", "color": color, "width": 7, "segments": segments})
        if edges:
            u, v = strip_representative_edge(coarse, skey)
            a, b = coarse.vertex_coordinates(u), coarse.vertex_coordinates(v)
            labels.append({"at": [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0], "text": str(skey)})

    poles = [coarse.vertex_coordinates(v) for v in coarse.poles()] if hasattr(coarse, "poles") else []
    layers.append({"kind": "discs", "points": poles, "color": COLORS["pole"], "radius": 11})
    layers.append({"kind": "rings", "points": [list(p) for p in session.points], "color": COLORS["point"], "radius": 19})
    # Last, so no line is drawn over a number.
    layers.append({"kind": "labels", "items": labels, "color": COLORS["label"]})
    return layers, strip_colors


def _bounds(layers: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    xs, ys = [], []
    for layer in layers:
        if layer["kind"] == "lines":
            for a, b in layer["segments"]:
                xs += [a[0], b[0]]
                ys += [a[1], b[1]]
        else:
            points = layer.get("points") or [item["at"] for item in layer.get("items", [])]
            for point in points:
                xs.append(point[0])
                ys.append(point[1])
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


# ==============================================================================
# rendering
# ==============================================================================


def render_png(session: MeshSession, width: int = DEFAULT_SIZE, height: int = DEFAULT_SIZE, margin: float = 0.06, layers: list[dict[str, Any]] | None = None) -> bytes | None:
    """The session's mesh and its inputs as PNG bytes, or ``None`` when there is nothing to draw."""
    width = max(160, min(int(width), MAX_SIZE))
    height = max(160, min(int(height), MAX_SIZE))
    if layers is None:
        layers = scene(session)
    box = _bounds(layers)
    if box is None:
        return None

    big = Raster(width * SCALE, height * SCALE)
    left, bottom, right, top = box
    span_x = max(right - left, 1e-9)
    span_y = max(top - bottom, 1e-9)
    pad = margin * max(span_x, span_y)
    left, bottom, right, top = left - pad, bottom - pad, right + pad, top + pad
    scale = min((width * SCALE) / (right - left), (height * SCALE) / (top - bottom))
    offset_x = (width * SCALE - (right - left) * scale) / 2.0
    offset_y = (height * SCALE - (top - bottom) * scale) / 2.0

    def project(point: Sequence[float]) -> tuple[float, float]:
        # y is flipped: model space runs up, a raster runs down.
        return (offset_x + (point[0] - left) * scale, (height * SCALE) - (offset_y + (point[1] - bottom) * scale))

    for layer in layers:
        color = layer["color"]
        if layer["kind"] == "lines":
            for a, b in layer["segments"]:
                big.line(project(a), project(b), color, layer["width"])
        elif layer["kind"] == "path":
            points = layer["points"]
            for i in range(len(points) - 1):
                big.line(project(points[i]), project(points[i + 1]), color, layer["width"])
        elif layer["kind"] == "discs":
            for point in layer["points"]:
                big.disc(project(point), layer["radius"], color)
        elif layer["kind"] == "rings":
            for point in layer["points"]:
                big.ring(project(point), layer["radius"], color, thickness=6)
        elif layer["kind"] == "labels":
            for item in layer["items"]:
                big.text(project(item["at"]), item["text"], color)

    return big.downsample(SCALE).to_png()


def render_png_base64(session: MeshSession, width: int = DEFAULT_SIZE, height: int = DEFAULT_SIZE, layers: list[dict[str, Any]] | None = None) -> str | None:
    """The PNG, base64 encoded for an MCP image content block. ``None`` if empty."""
    data = render_png(session, width=width, height=height, layers=layers)
    if data is None:
        return None
    return base64.b64encode(data).decode("ascii")
