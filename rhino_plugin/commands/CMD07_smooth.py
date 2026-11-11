#! python3
# r: compas
# r: pydantic
# r: compas_fd
# r: compas_singular
"""Step 7 -- smooth a dense quad mesh, whole or a region, optionally held to guide curves.

Input: a mesh picked from TopologyProblem::QuadMesh or a sublayer. Output: the smoothed mesh on QuadMesh::Smoothed::<algorithm or Relaxed>.
"""

import compas_rhino as cr
import Rhino
import rhinoscriptsyntax as rs
import scriptcontext as sc
import System
from compas_rhino.conversions import curve_to_compas

from compas.geometry import Point
from compas_singular.datastructures import QuadMesh
from compas_singular.datastructures.mesh.smoothing import automated_boundary_constraints
from compas_singular.datastructures.mesh.smoothing import constrained_smoothing
from compas_singular.datastructures.mesh.smoothing import mesh_boundary_corners
from compas_singular.datastructures.mesh.smoothing import region_smoothing
from compas_singular.datastructures.mesh.smoothing import relaxation
from compas_singular.editing import GuideCurve
from compas_singular.editing import attach_chain
from compas_singular.editing import chain_quality
from compas_singular.editing import collect_polyedges
from compas_singular.editing import guide_chain
from compas_singular.editing import mean_edge_length
from compas_singular.editing.guide_chain import DEFAULT_MAX_ANGLE
from compas_singular.editing.guide_chain import DEFAULT_TOLERANCE_FACTOR
from compas_singular.rhino.helpers import bake_mesh
from compas_singular.rhino.helpers import clear_layer
from compas_singular.rhino.helpers import curve_points
from compas_singular.rhino.helpers import mesh_from_rhino
from compas_singular.rhino.project import LAYER_DATA

ALGORITHMS = ["Area", "Centroid", "CenterOfMass", "ForceDensity"]
BOUNDARY_MODES = ["Sliding", "Fixed", "Free"]

#: Layer a smoothed region is baked to, under QuadMesh::Smoothed.
REGION_OUTPUT_LAYER = "Relaxed"

PREVIEW_LAYER = "SmoothPreview"
CORE_COLOUR = (255, 0, 0)
BLEND_COLOUR = (255, 160, 0)
GUIDE_COLOUR = (200, 0, 255)
FIXED_COLOUR = (0, 110, 255)

#: Full layer paths of the guides and point features.
GUIDE_LAYER = LAYER_DATA.get("Guides", ("Guides", None))[0]
POINT_LAYER = LAYER_DATA.get("PointFeatures", ("PointFeatures", None))[0]

#: Rings of vertices outside a region over which the damping falls to zero.
DEFAULT_BLEND = 3

#: Force density of a boundary edge relative to an interior one, under ForceDensity.
DEFAULT_Q_FACTOR = 10.0

#: Largest distance, in mean edge lengths, from a point feature to the vertex it fixes.
POINT_FEATURE_REACH = 0.5

#: What option_line returns when the user presses Enter.
ENTER = "<enter>"


def boundary_vertices(mesh):
    """Every vertex on every boundary, holes included."""
    return set(vertex for loop in mesh.vertices_on_boundaries() for vertex in loop)


def grow(mesh, core, rings=1):
    """The core plus ``rings`` of neighbours around it."""
    out = set(core)
    frontier = set(core)
    for _ in range(rings):
        frontier = set(n for vertex in frontier for n in mesh.vertex_neighbors(vertex)) - out
        out |= frontier
    return out


def taper(mesh, core, blend):
    """Per-vertex damping weight: 1 in the core, falling to 0 outside the blend."""
    weights = dict((vertex, 1.0) for vertex in core)
    frontier = set(core)
    for ring in range(1, blend + 1):
        frontier = set(n for vertex in frontier for n in mesh.vertex_neighbors(vertex)) - set(weights)
        if not frontier:
            break
        weight = 1.0 - ring / float(blend + 1)
        for vertex in frontier:
            weights[vertex] = weight
    return weights


def propose_chain(mesh, guide, tolerance, max_angle, polyedges):
    """The chain for one guide, why it is empty if it is, and how it measures."""
    selected, info = guide_chain(mesh, guide, tolerance=tolerance, max_angle=max_angle,
                                 polyedges=polyedges)
    quality = chain_quality(mesh, selected, guide) if selected else None
    return selected, info, quality


def attach_guides(mesh, guides, fixed=(), allowed=None):
    """Move every guide's chain onto its guide, and return what holds each vertex there.

    Note: all moves are computed before any is applied; a boundary vertex only slides on its outline.

    Parameters
    ----------
    guides : list[dict]
        ``{'guide': GuideCurve, 'chain': [vertex, ...], 'hold': 'fixed' | 'sliding'}``.
    fixed : iterable[int]
        Vertices left out of every chain.
    allowed : set[int], optional
        If given, every vertex not in it is left out as well.
    """
    fixed = set(fixed)
    moves, constraints, overlap = {}, {}, set()
    skipped_fixed, skipped_outside, attached_guides = 0, 0, 0

    for entry in guides:
        chain = []
        for vertex in entry["chain"]:
            if vertex in fixed:
                skipped_fixed += 1
            elif allowed is not None and vertex not in allowed:
                skipped_outside += 1
            else:
                chain.append(vertex)
        if not chain:
            continue
        attached_guides += 1
        entry_moves, entry_constraints = attach_chain(mesh, chain, entry["guide"],
                                                      hold=entry["hold"])
        overlap.update(vertex for vertex in entry_constraints if vertex in constraints)
        moves.update(entry_moves)
        constraints.update(entry_constraints)

    for vertex, xyz in moves.items():
        mesh.vertex_attributes(vertex, "xyz", xyz)

    return {
        "constraints": constraints,
        "moved": set(moves),
        "sliding": set(vertex for vertex in moves if not isinstance(constraints[vertex], Point)),
        "anchored": set(constraints) - set(moves),
        "guides": attached_guides,
        "overlap": len(overlap),
        "skipped_fixed": skipped_fixed,
        "skipped_outside": skipped_outside,
    }


def _report(attached, pinned, notes, **extra):
    report = {
        "guides": attached["guides"],
        "attached": len(attached["constraints"]),
        "moved": len(attached["moved"]),
        "anchored": len(attached["anchored"]),
        "overlap": attached["overlap"],
        "skipped_fixed": attached["skipped_fixed"],
        "skipped_outside": attached["skipped_outside"],
        "pinned": len(pinned),
        "notes": notes,
    }
    report.update(extra)
    return report


def boundary_holds(mesh, boundary):
    """``(constraints, pinned)`` for a Boundary setting: sliding, fixed or free."""
    if boundary == "sliding":
        return automated_boundary_constraints(mesh), set()
    if boundary == "fixed":
        return {}, boundary_vertices(mesh)
    if boundary == "free":
        return {}, set()
    raise ValueError("boundary must be sliding, fixed or free, got {!r}".format(boundary))


def _release_anchors(constraints, attached, boundary):
    """On a Free boundary, drop the outline constraints of the chains' boundary ends."""
    if boundary == "free":
        for vertex in attached["anchored"]:
            constraints.pop(vertex, None)


def smooth_whole(mesh, algorithm="area", boundary="sliding", fixed=(), guides=(),
                 kmax=100, damping=0.5, q_factor=DEFAULT_Q_FACTOR):
    """Smooth the whole mesh in place with the options of the command. Returns a report.

    Raises
    ------
    ValueError
        If ForceDensity would run with nothing fixed.
    """
    algorithm = algorithm.lower()
    boundary = boundary.lower()
    fixed = set(fixed)
    walls = boundary_vertices(mesh)
    notes = []

    if algorithm == "forcedensity":
        if boundary == "fixed":
            pinned = set(walls)
        elif boundary == "free":
            pinned = set()
        else:
            pinned = set(mesh_boundary_corners(mesh))
            notes.append(
                "ForceDensity cannot slide a boundary: only the {} corner(s) are held, and "
                "the rest of the outline is held by stiff edges alone, so it can drift off "
                "the wall. Use Boundary=Fixed, or another algorithm, to keep it on the "
                "wall.".format(len(pinned)))
        attached = attach_guides(mesh, guides, fixed)
        pinned |= attached["moved"]
        if attached["sliding"]:
            notes.append(
                "ForceDensity cannot hold a vertex on a guide: {} vertices with a Sliding "
                "hold were pinned where they landed instead.".format(len(attached["sliding"])))
        pinned |= fixed
        if not pinned:
            raise ValueError(
                "ForceDensity needs at least one fixed vertex and there is none: no fixed "
                "boundary, no corners, no fixed vertices and no guides. Fix some vertices, "
                "or set Boundary=Fixed or Sliding.")
        relaxation(mesh, fixed=sorted(pinned), q_factor=q_factor)
        return _report(attached, pinned, notes)

    constraints, pinned = boundary_holds(mesh, boundary)

    attached = attach_guides(mesh, guides, fixed)
    constraints.update(attached["constraints"])
    _release_anchors(constraints, attached, boundary)
    if boundary == "free" and not fixed and not constraints:
        notes.append(
            "the boundary is free and nothing is fixed or on a guide, so nothing holds the "
            "mesh: it shrinks a little with every iteration.")
    pinned -= set(attached["constraints"])
    pinned |= fixed
    for vertex in fixed:
        constraints.pop(vertex, None)

    if algorithm == "centerofmass":
        lonely = set(vertex for vertex in mesh.vertices()
                     if vertex not in pinned and len(mesh.vertex_neighbors(vertex)) < 3)
        mobile = [vertex for vertex in lonely if not isinstance(constraints.get(vertex), Point)]
        if mobile:
            notes.append(
                "CenterOfMass cannot move a vertex with only two neighbours: {} of them, which "
                "would have moved, were pinned instead.".format(len(mobile)))
        pinned |= lonely
        for vertex in lonely:
            constraints.pop(vertex, None)

    constrained_smoothing(mesh, kmax=kmax, damping=damping, constraints=constraints,
                          algorithm=algorithm, fixed=sorted(pinned))
    return _report(attached, pinned, notes)


def smooth_region(mesh, core, blend=DEFAULT_BLEND, boundary="sliding", fixed=(), guides=(),
                  kmax=50, damping=0.5):
    """Smooth a region of the mesh in place with the options of the command. Returns a report.

    Note: guide chains are clipped to the core.
    """
    boundary = boundary.lower()
    core = set(core)
    fixed = set(fixed)
    weights = taper(mesh, core, blend)
    constraints, pinned = boundary_holds(mesh, boundary)

    attached = attach_guides(mesh, guides, fixed, allowed=core)
    constraints.update(attached["constraints"])
    _release_anchors(constraints, attached, boundary)
    pinned -= set(attached["constraints"])
    pinned |= fixed

    for vertex in pinned:
        if vertex in weights:
            weights[vertex] = 0.0
        constraints.pop(vertex, None)

    region_smoothing(mesh, weights, kmax=kmax, damping=damping, constraints=constraints)
    moving = len([vertex for vertex, weight in weights.items() if weight > 0.0])
    return _report(attached, set(vertex for vertex in pinned if vertex in weights), [],
                   core=len(core), moving=moving)


def quad_mesh_filter(rhobj, geometry, component_index):
    layer = rs.ObjectLayer(rhobj)
    quad_layer = rs.LayerName("QuadMesh", fullpath=True)
    return layer == quad_layer or rs.IsLayerChildOf(quad_layer, layer)


def _xyz(point):
    if hasattr(point, "X"):
        return [point.X, point.Y, point.Z]
    return [point[0], point[1], point[2]]


def _index(added):
    """The index an ``AddOption*`` call returns; 0 if Rhino refused the option.

    Note: the by-ref overloads (Toggle, Integer, Double) return ``(index, option)`` in Rhino 8 CPython.
    """
    return added[0] if isinstance(added, tuple) else added


def option_line(prompt, options):
    """Show one command-line option line and wait.

    ``options`` is a list of ``(key, option)``: ``option`` is a plain option name, or a
    function that adds the option to the GetOption and returns what that call returned.

    Returns ``(key, option)`` for a click, ``(ENTER, None)`` for Enter and ``(None, None)``
    for Esc. A typed Integer or Double option has already stored its new value when this
    returns.
    """
    go = Rhino.Input.Custom.GetOption()
    go.SetCommandPrompt(prompt)
    go.AcceptNothing(True)
    keys = {}
    for key, option in options:
        index = go.AddOption(option) if isinstance(option, str) else _index(option(go))
        if index > 0:
            keys[index] = key
    result = go.Get()
    if result == Rhino.Input.GetResult.Nothing:
        return ENTER, None
    if result != Rhino.Input.GetResult.Option:
        return None, None
    return keys.get(go.OptionIndex()), go.Option()


def guide_curves():
    """Every curve on the Guides layer and its sublayers."""
    if not rs.IsLayer(GUIDE_LAYER):
        return []
    found, seen = [], set()
    for layer in [GUIDE_LAYER] + (rs.LayerChildren(GUIDE_LAYER) or []):
        for guid in rs.ObjectsByLayer(layer) or []:
            if rs.IsCurve(guid) and str(guid) not in seen:
                seen.add(str(guid))
                found.append(guid)
    return found


def build_guide(curve_id):
    """A Rhino curve as a GuideCurve: samples to choose the chain on, the curve to land on."""
    points = curve_points(curve_id, 0.25)
    # curve_points drops the closing point of a closed curve.
    if rs.IsCurveClosed(curve_id) and len(points) > 2:
        points = points + points[:1]
    return GuideCurve(points, curve=guide_geometry(curve_id))


def guide_geometry(curve_id):
    """The guide as a compas curve flattened onto World XY, for its vertices to land ON.

    None for a polyline, or if Rhino cannot flatten the curve.
    """
    curve = rs.coercecurve(curve_id)
    if curve is None or curve.TryGetPolyline()[0]:
        return None
    flat = Rhino.Geometry.Curve.ProjectToPlane(curve, Rhino.Geometry.Plane.WorldXY)
    if flat is None:
        return None
    return curve_to_compas(flat)


class Preview(object):
    """Coloured points on their own layer, showing what the smoothing will hold.

    Note: drawn with RhinoCommon and redrawn once, only when the picture changed.
    """

    def __init__(self, doc, layer=PREVIEW_LAYER):
        self.doc = doc
        self.layer = layer
        self.guids = []
        self.shown = None

    def _layer_index(self):
        index = self.doc.Layers.FindByFullPath(self.layer, -1)
        if index < 0:
            index = self.doc.Layers.Add(self.layer, System.Drawing.Color.Red)
        elif self.shown is None:
            for rhino_object in self.doc.Objects.FindByLayer(self.doc.Layers[index]) or []:
                self.doc.Objects.Delete(rhino_object.Id, True)
        return index

    def draw(self, colours, coordinates):
        """``colours``: {vertex: (r, g, b)}; ``coordinates``: vertex -> [x, y, z]."""
        signature = frozenset(colours.items())
        if signature == self.shown:
            return False
        index = self._layer_index()
        self.clear(redraw=False)
        groups = {}
        for vertex, colour in colours.items():
            groups.setdefault(colour, []).append(vertex)
        for colour, vertices in groups.items():
            attributes = Rhino.DocObjects.ObjectAttributes()
            attributes.LayerIndex = index
            attributes.ColorSource = Rhino.DocObjects.ObjectColorSource.ColorFromObject
            attributes.ObjectColor = System.Drawing.Color.FromArgb(colour[0], colour[1], colour[2])
            for vertex in vertices:
                x, y, z = coordinates(vertex)
                self.guids.append(self.doc.Objects.AddPoint(Rhino.Geometry.Point3d(x, y, z), attributes))
        self.shown = signature
        self.doc.Views.Redraw()
        return True

    def clear(self, redraw=True):
        for guid in self.guids:
            self.doc.Objects.Delete(guid, True)
        self.guids = []
        if redraw:
            self.shown = None
            self.doc.Views.Redraw()


class SmoothCommand(object):
    """The state of the option lines, the picks that change it, and the preview."""

    def __init__(self, mesh_id, region):
        self.mesh_id = mesh_id
        self.region = region
        geometry = cr.objects.find_object(mesh_id).Geometry
        self.mesh = QuadMesh.from_vertices_and_faces(
            *mesh_from_rhino(geometry).to_vertices_and_faces())
        self.rhino_vertices = rs.MeshVertices(mesh_id)
        self.coordinates = [(vertex, self.mesh.vertex_coordinates(vertex))
                            for vertex in self.mesh.vertices()]
        self.average = mean_edge_length(self.mesh)
        self._polyedges = None

        self.preview = Preview(sc.doc)

        self.algorithm = 0
        self.boundary = "Sliding"
        self.fixed = []
        self.guides = []
        self.core = []
        self.blend = Rhino.Input.Custom.OptionInteger(DEFAULT_BLEND, 0, 1000)
        self.iterations = Rhino.Input.Custom.OptionInteger(50 if region else 100, 1, 100000)
        self.damping = Rhino.Input.Custom.OptionDouble(0.5, 0.0, 1.0)
        self.q_factor = Rhino.Input.Custom.OptionDouble(DEFAULT_Q_FACTOR, 0.001, 1.0e6)
        self.tolerance = Rhino.Input.Custom.OptionDouble(float(DEFAULT_TOLERANCE_FACTOR), 0.05, 5.0)
        self.max_angle = Rhino.Input.Custom.OptionDouble(float(DEFAULT_MAX_ANGLE), 1.0, 90.0)

    def nearest_vertex(self, xyz):
        """The compas vertex closest to a point, and how far it is. Matched by position, not index."""
        best, best_d = None, None
        for vertex, point in self.coordinates:
            d = (point[0] - xyz[0]) ** 2 + (point[1] - xyz[1]) ** 2 + (point[2] - xyz[2]) ** 2
            if best_d is None or d < best_d:
                best, best_d = vertex, d
        return best, (best_d or 0.0) ** 0.5

    def pick_vertices(self, message):
        picked = rs.GetMeshVertices(self.mesh_id, message)
        if not picked:
            return []
        return [self.nearest_vertex(_xyz(self.rhino_vertices[i]))[0] for i in picked]

    def vertices_in_curves(self):
        curve_ids = rs.GetObjects(message="Pick one or more CLOSED curves around the zone",
                                  filter=rs.filter.curve, preselect=False, select=False,
                                  minimum_count=1) or []
        closed = [curve_id for curve_id in curve_ids if rs.IsCurveClosed(curve_id)]
        if len(closed) < len(curve_ids):
            print("{} of the picked curves not closed -- skipped.".format(
                len(curve_ids) - len(closed)))
        inside = []
        for vertex, xyz in self.coordinates:
            # PointInPlanarClosedCurve: 1 = inside, 2 = on the curve.
            if any(rs.PointInPlanarClosedCurve(xyz, curve_id) in (1, 2) for curve_id in closed):
                inside.append(vertex)
        return inside

    def point_feature_vertices(self):
        """The vertex at every point on the PointFeatures layer."""
        if not rs.IsLayer(POINT_LAYER):
            print("No {} layer -- no point features to fix.".format(POINT_LAYER))
            return []
        found, far = [], 0
        for guid in rs.ObjectsByLayer(POINT_LAYER) or []:
            if not rs.IsPoint(guid):
                continue
            vertex, distance = self.nearest_vertex(_xyz(rs.PointCoordinates(guid)))
            if distance > POINT_FEATURE_REACH * self.average:
                far += 1
                continue
            found.append(vertex)
        print("{} point feature(s) matched to a vertex.".format(len(found)))
        if far:
            print("  {} skipped: no vertex within {} edge length(s) of them, so they are not "
                  "vertices of this mesh.".format(far, POINT_FEATURE_REACH))
        return found

    def polyedges(self):
        if self._polyedges is None:
            nonquad = [f for f in self.mesh.faces() if len(self.mesh.face_vertices(f)) != 4]
            if nonquad:
                print("note: {} non-quad face(s) -- polyedges stop at those.".format(len(nonquad)))
            self._polyedges = collect_polyedges(self.mesh)
        return self._polyedges

    @staticmethod
    def _add(target, vertices):
        for vertex in vertices:
            if vertex not in target:
                target.append(vertex)

    def show(self, pending=()):
        """Draw what will happen: fixed blue, guide chains purple, region core red, blend orange.

        ``pending`` are guide entries being edited, drawn in place of their saved versions.
        """
        colours = {}
        if self.region:
            for vertex, weight in taper(self.mesh, self.core, self.blend.CurrentValue).items():
                colours[vertex] = CORE_COLOUR if weight >= 1.0 else BLEND_COLOUR
        allowed = set(self.core) if self.region else None
        for entry in self.guide_entries(pending):
            for vertex in entry["chain"]:
                if allowed is None or vertex in allowed:
                    colours[vertex] = GUIDE_COLOUR
        for vertex in self.fixed:
            colours[vertex] = FIXED_COLOUR
        self.preview.draw(colours, self.mesh.vertex_coordinates)

    def guide_entries(self, pending=()):
        keys = set(entry["key"] for entry in pending)
        return [entry for entry in self.guides if entry["key"] not in keys] + list(pending)

    def menu(self):
        """The main option line. True to smooth, False when cancelled."""
        while True:
            self.show()
            force_density = not self.region and ALGORITHMS[self.algorithm] == "ForceDensity"
            options = []
            if self.region:
                options.append(("region", lambda go: go.AddOption("Region", str(len(self.core)))))
                options.append(("blend", lambda go: go.AddOptionInteger("Blend", self.blend)))
            else:
                options.append(("algorithm", lambda go: go.AddOptionList(
                    "Algorithm", ALGORITHMS, self.algorithm)))
            options.append(("boundary", lambda go: go.AddOptionList(
                "Boundary", BOUNDARY_MODES, BOUNDARY_MODES.index(self.boundary))))
            options.append(("fixed", lambda go: go.AddOption("FixedVertices", str(len(self.fixed)))))
            options.append(("guides", lambda go: go.AddOption("Guides", str(len(self.guides)))))
            if force_density:
                options.append(("stiffness", lambda go: go.AddOptionDouble(
                    "BoundaryStiffness", self.q_factor)))
            else:
                options.append(("iterations", lambda go: go.AddOptionInteger(
                    "Iterations", self.iterations)))
                options.append(("damping", lambda go: go.AddOptionDouble("Damping", self.damping)))

            what = "a region" if self.region else "the whole mesh"
            key, option = option_line("Smooth {}. Press Enter to run".format(what), options)
            if key is None:
                return False
            if key == ENTER:
                if self.region and not self.core:
                    print("The region is empty -- add vertices to it first.")
                    continue
                return True
            if key == "algorithm":
                self.algorithm = option.CurrentListOptionIndex
            elif key == "boundary":
                self.boundary = BOUNDARY_MODES[option.CurrentListOptionIndex]
            elif key == "region":
                self.edit_region()
            elif key == "fixed":
                self.edit_fixed()
            elif key == "guides":
                self.edit_guides()

    def pick_region(self):
        """The first pick of the region. True if it selected anything."""
        key, _ = option_line("Select the zone with closed curves or vertices. Press Enter for Curve",
                             [("curve", "Curve"), ("vertices", "Vertices")])
        if key is None:
            return False
        if key == "vertices":
            self._add(self.core, self.pick_vertices("Pick vertices in the zone"))
        else:
            self._add(self.core, self.vertices_in_curves())
        print("region: {} core vertices.".format(len(self.core)))
        return bool(self.core)

    def edit_region(self):
        while True:
            self.show()
            prompt = "Region: {} core vertices, {} with the blend. Press Enter when done".format(
                len(self.core), len(taper(self.mesh, self.core, self.blend.CurrentValue)))
            key, _ = option_line(prompt, [("curve", "Curve"), ("add", "Add"), ("remove", "Remove"),
                                          ("grow", "Grow"), ("clear", "Clear")])
            if key in (None, ENTER):
                return
            if key == "curve":
                self._add(self.core, self.vertices_in_curves())
            elif key == "add":
                self._add(self.core, self.pick_vertices("Pick vertices to ADD to the region"))
            elif key == "remove":
                drop = set(self.pick_vertices("Pick vertices to REMOVE from the region"))
                self.core = [vertex for vertex in self.core if vertex not in drop]
            elif key == "grow":
                self._add(self.core, grow(self.mesh, self.core, 1))
            elif key == "clear":
                self.core = []

    def edit_fixed(self):
        while True:
            self.show()
            prompt = "Fixed vertices: {}. Press Enter when done".format(len(self.fixed))
            key, _ = option_line(prompt, [("add", "Add"), ("points", "PointFeatures"),
                                          ("remove", "Remove"), ("clear", "Clear")])
            if key in (None, ENTER):
                return
            if key == "add":
                self._add(self.fixed, self.pick_vertices("Pick vertices to FIX"))
            elif key == "points":
                self._add(self.fixed, self.point_feature_vertices())
            elif key == "remove":
                drop = set(self.pick_vertices("Pick fixed vertices to FREE"))
                self.fixed = [vertex for vertex in self.fixed if vertex not in drop]
            elif key == "clear":
                self.fixed = []

    def edit_guides(self):
        while True:
            self.show()
            count = len(set(vertex for entry in self.guides for vertex in entry["chain"]))
            prompt = ("Guides: {} with {} vertices (Distance is in edge lengths). Press Enter "
                      "when done").format(len(self.guides), count)
            # Distance: in mean edge lengths, off the guide. Angle: in degrees, 90 = off.
            key, _ = option_line(prompt, [
                ("all", "AllGuides"), ("pick", "Pick"), ("remove", "Remove"), ("clear", "Clear"),
                ("distance", lambda go: go.AddOptionDouble("Distance", self.tolerance)),
                ("angle", lambda go: go.AddOptionDouble("Angle", self.max_angle)),
            ])
            if key in (None, ENTER):
                return
            if key == "pick":
                self.pick_guide()
            elif key == "all":
                self.all_guides()
            elif key == "remove":
                self.remove_guide()
            elif key == "clear":
                self.guides = []

    def propose(self, guide):
        """Propose a chain for one guide and say what it is. Returns (chain, info)."""
        selected, info, quality = propose_chain(
            self.mesh, guide, self.tolerance.CurrentValue * self.average,
            self.max_angle.CurrentValue, self.polyedges())
        if not selected:
            return selected, info
        print("  {} vertices: {:.0f}% of the guide, alignment {:.2f}, worst face angle if "
              "attached {:.0f}.".format(len(selected), 100 * quality["coverage"],
                                        quality["alignment"], quality["min_face_angle"] or 0.0))
        if quality["alignment"] < 0.8:
            print("  note: this guide does not follow a course of the mesh.")
        if quality["folded_faces"]:
            print("  WARNING: attaching this one turns {} face(s) inside out. It is reaching "
                  "{:.1f} edge lengths to do it -- lower Distance to stop it.".format(
                      quality["folded_faces"], quality["off_max"]))
        if self.region:
            core = set(self.core)
            outside = len([vertex for vertex in selected if vertex not in core])
            if outside:
                print("  {} of them are outside the region and will be left alone.".format(outside))
        return selected, info

    def pick_guide(self):
        curve_id = rs.GetObject(message="Pick a guide curve", filter=rs.filter.curve,
                                preselect=False, select=False)
        if not curve_id:
            return
        key = str(curve_id)
        existing = [entry for entry in self.guides if entry["key"] == key]
        if existing:
            print("editing the chain this guide already has.")
            entry = existing[0]
        else:
            print("guide:")
            guide = build_guide(curve_id)
            selected, info = self.propose(guide)
            if not selected:
                print("  {} -- use Add to pick the chain by hand.".format(info["reason"]))
            entry = {"key": key, "guide": guide, "chain": selected, "hold": "fixed"}

        edited = self.edit_chain(entry)
        if edited is None:
            print("  chain left as it was." if existing else "  discarded.")
            return
        self.guides = [other for other in self.guides if other["key"] != key]
        if not edited["chain"]:
            print("  no vertices -- this guide holds nothing.")
            return
        self.guides.append(edited)

    def edit_chain(self, entry):
        """Add / Remove / Clear on one chain. The edited entry, or None when discarded."""
        entry = dict(entry, chain=list(entry["chain"]))
        while True:
            self.show(pending=[entry])
            prompt = "Chain: {} vertices. Press Enter to keep it, Esc to discard".format(
                len(entry["chain"]))
            # Hold: Fixed pins a vertex where it lands; Sliding lets it travel along the guide.
            key, _ = option_line(prompt, [
                ("add", "Add"), ("remove", "Remove"), ("clear", "Clear"),
                ("hold", lambda go: go.AddOption("Hold", entry["hold"].capitalize())),
            ])
            if key is None:
                return None
            if key == ENTER:
                return entry
            if key == "add":
                self._add(entry["chain"], self.pick_vertices("Pick vertices to ADD to the chain"))
            elif key == "remove":
                drop = set(self.pick_vertices("Pick vertices to REMOVE from the chain"))
                entry["chain"] = [vertex for vertex in entry["chain"] if vertex not in drop]
            elif key == "clear":
                entry["chain"] = []
            elif key == "hold":
                entry["hold"] = "sliding" if entry["hold"] == "fixed" else "fixed"

    def all_guides(self):
        curve_ids = guide_curves()
        if not curve_ids:
            print("No curves on the {} layer -- pick the guides one at a time instead.".format(
                GUIDE_LAYER))
            return
        present = set(entry["key"] for entry in self.guides)
        print("proposing a chain for every guide on the {} layer: {} curve(s), distance {:.2f} "
              "edge lengths, angle {:.0f} degrees.".format(
                  GUIDE_LAYER, len(curve_ids), self.tolerance.CurrentValue,
                  self.max_angle.CurrentValue))

        new, refused = [], []
        for number, curve_id in enumerate(curve_ids, 1):
            if str(curve_id) in present:
                print("guide {}: already has a chain -- kept as it is.".format(number))
                continue
            print("guide {}:".format(number))
            guide = build_guide(curve_id)
            selected, info = self.propose(guide)
            if not selected:
                print("  skipped -- {}.".format(info["reason"]))
                refused.append(number)
                continue
            new.append({"key": str(curve_id), "guide": guide, "chain": selected, "hold": "fixed"})

        if refused:
            print("{} guide(s) got nothing: {}. Widen Distance or Angle, or Pick them to choose "
                  "the chain by hand.".format(len(refused), ", ".join(str(i) for i in refused)))
        if not new:
            return

        hold = "fixed"
        count = len(set(vertex for entry in new for vertex in entry["chain"]))
        while True:
            self.show(pending=new)
            prompt = "{} new chain(s), {} vertices. Press Enter to add them, Esc to discard".format(
                len(new), count)
            key, _ = option_line(prompt, [("hold", lambda go: go.AddOption("Hold", hold.capitalize()))])
            if key is None:
                print("discarded.")
                return
            if key == ENTER:
                break
            hold = "sliding" if hold == "fixed" else "fixed"
        for entry in new:
            entry["hold"] = hold
        self.guides.extend(new)

    def remove_guide(self):
        curve_id = rs.GetObject(message="Pick the guide curve to remove", filter=rs.filter.curve,
                                preselect=False, select=False)
        if not curve_id:
            return
        before = len(self.guides)
        self.guides = [entry for entry in self.guides if entry["key"] != str(curve_id)]
        if len(self.guides) == before:
            print("That curve has no chain.")

    def run(self):
        """Smooth, and return (report, the layer to bake to, a line saying what was done)."""
        kmax = int(self.iterations.CurrentValue)
        damping = float(self.damping.CurrentValue)
        if self.region:
            blend = int(self.blend.CurrentValue)
            report = smooth_region(self.mesh, self.core, blend=blend, boundary=self.boundary,
                                   fixed=self.fixed, guides=self.guides, kmax=kmax, damping=damping)
            summary = ("smoothed a region: {} core vertices, {} moving with a blend of {} rings, "
                       "boundary {}, {} iterations at damping {}.").format(
                report["core"], report["moving"], blend, self.boundary.lower(), kmax, damping)
            return report, REGION_OUTPUT_LAYER, summary

        algorithm = ALGORITHMS[self.algorithm]
        report = smooth_whole(self.mesh, algorithm, boundary=self.boundary, fixed=self.fixed,
                              guides=self.guides, kmax=kmax, damping=damping,
                              q_factor=float(self.q_factor.CurrentValue))
        if algorithm == "ForceDensity":
            how = "boundary stiffness {}".format(self.q_factor.CurrentValue)
        else:
            how = "{} iterations at damping {}".format(kmax, damping)
        summary = "smoothed the whole mesh: {}, boundary {}, {}.".format(
            algorithm, self.boundary.lower(), how)
        return report, algorithm, summary


def print_report(report, fixed_count, boundary):
    anchored = "held nowhere, like the rest of the free boundary" if boundary.lower() == "free" \
        else "they slide along it"
    if fixed_count:
        print("  {} fixed vertices.".format(fixed_count))
    if report["attached"] or report["skipped_fixed"] or report["skipped_outside"]:
        print("  {} guide(s): {} vertices attached, {} moved onto their guide, {} anchored "
              "on the boundary ({}).".format(
                  report["guides"], report["attached"], report["moved"], report["anchored"],
                  anchored))
    if report["overlap"]:
        print("  note: {} vertices were in two chains and are held by the guide picked "
              "last.".format(report["overlap"]))
    if report["skipped_fixed"]:
        print("  note: {} chain vertices are fixed vertices, and stayed where they "
              "are.".format(report["skipped_fixed"]))
    if report["skipped_outside"]:
        print("  note: {} chain vertices are outside the region and were left "
              "alone.".format(report["skipped_outside"]))
    for note in report["notes"]:
        print("  note: " + note)


def main():
    if not rs.IsLayer("QuadMesh"):
        print("No dense quad mesh yet -- the QuadMesh layer does not exist.")
        return

    mesh_id = rs.GetObject(
        message="Pick a dense quad mesh to smooth from the layer QuadMesh and its sublayers",
        filter=rs.filter.mesh, preselect=False, select=False, custom_filter=quad_mesh_filter)
    if not mesh_id:
        print("No mesh picked.")
        return

    key, _ = option_line("Smooth what? Press Enter for Whole",
                         [("whole", "Whole"), ("region", "Region")])
    if key is None:
        print("Cancelled -- nothing smoothed.")
        return

    command = SmoothCommand(mesh_id, region=(key == "region"))
    try:
        if command.region and not command.pick_region():
            print("No region selected -- nothing smoothed.")
            return
        if not command.menu():
            print("Cancelled -- nothing smoothed.")
            return
    finally:
        if rs.IsLayer(PREVIEW_LAYER):
            rs.PurgeLayer(PREVIEW_LAYER)

    try:
        report, layer, summary = command.run()
    except Exception as exc:
        print("Smoothing failed, nothing baked: {}: {}".format(type(exc).__name__, exc))
        return

    rs.AddLayer("Smoothed", parent="QuadMesh")
    rs.AddLayer(layer, parent="Smoothed")
    clear_layer(layer, clean_sublayers=True)
    bake_mesh(command.mesh, layer)
    print(summary)
    print_report(report, len(command.fixed), command.boundary)
    print("  baked to Smoothed::{}.".format(layer))


if __name__ == "__main__":
    main()
