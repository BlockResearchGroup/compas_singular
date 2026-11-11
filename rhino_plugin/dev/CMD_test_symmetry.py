#! python3
# r: compas
# r: pydantic
# r: compas_singular

"""**TEST -- the whole symmetry workflow on this document, in one command.**

    reads   Outer, Inner, Guides, PointFeatures      the domain (step 1)
            document settings                        triangulation_spacing, target_length
    writes  TopologyProblem::SymmetryTest::Symmetry       centre, mirror axes, rotation arc
            TopologyProblem::SymmetryTest::UnitOutline    the unit, before meshing
            TopologyProblem::SymmetryTest::CoarseUnit     the coarse layout of the unit
            TopologyProblem::SymmetryTest::Seams          the unit's seam edges
            TopologyProblem::SymmetryTest::CoarseLayout   the whole coarse layout
            TopologyProblem::SymmetryTest::QuadUnit       the dense unit
            TopologyProblem::SymmetryTest::QuadMesh       the whole quad mesh
    never   touches any other layer, the settings, or the session

A standalone check of ``compas_singular.symmetry`` -- it is NOT a step of the
workflow and nothing else reads what it bakes. It runs exactly what a script
would::

    report = decomposition.find_symmetry()
    coarse_unit = decomposition.symmetry_unit(keys=...)
    coarse_unit.collect_strips()
    coarse_unit.set_strips_density_target(target_length)
    coarse_layout = coarse_unit.expand_symmetrically()
    quad_unit = coarse_unit.quad_mesh()
    quad_mesh = quad_unit.expand_symmetrically(coarse_unit)

and bakes every stage on its own layer, so each can be switched on and off.

Prompts: the route, then which symmetries to enforce (by key, or All), then
Continue after the unit outline has been drawn. Every failure is printed with
the stage it happened in and the reason, and the command stops there.
"""

import time

import rhinoscriptsyntax as rs
import scriptcontext as sc

from compas_singular.algorithms import SkeletonDecomposition
from compas_singular.framefield.field_decomposition import FieldDecomposition
from compas_singular.framefield.quality import mesh_quality
from compas_singular.rhino.helpers import bake_mesh, bake_polylines, clear_layer, read_boundaries
from compas_singular.symmetry.measure import mesh_invariance

from compas_singular.rhino.project import get_settings, resolve_spacing
from compas_singular.rhino.project import ROOT as PROJECT_ROOT


ROOT = PROJECT_ROOT + "::SymmetryTest"

LAYERS = {
    "Symmetry": (40, 120, 220),
    "UnitOutline": (255, 170, 60),
    "CoarseUnit": (90, 90, 90),
    "Seams": (220, 30, 30),
    "CoarseLayout": (90, 90, 90),
    "QuadUnit": (60, 60, 60),
    "QuadMesh": (60, 60, 60),
}


def layer(name):
    """The full path of a SymmetryTest sublayer, created and emptied."""
    if not rs.IsLayer(PROJECT_ROOT):
        rs.AddLayer(PROJECT_ROOT)
    if not rs.IsLayer(ROOT):
        rs.AddLayer("SymmetryTest", parent=PROJECT_ROOT)
    path = ROOT + "::" + name
    if not rs.IsLayer(path):
        rs.AddLayer(name, color=LAYERS.get(name), parent=ROOT)
    clear_layer(path)
    return path


def points_of(curve):
    return [list(p) for p in getattr(curve, "points", curve)]


def ask_route():
    route = rs.GetString("Symmetry test -- route", "Skeleton", ["Skeleton", "FrameField", "Exit"])
    return (route or "Exit").lower()


def ask_keys(report):
    """Keys to enforce: pick one at a time, then Done. All = the detected group."""
    available = report.keys
    if not available:
        return None
    chosen = []
    while True:
        options = ["All", "Done"] + [k for k in available if k not in chosen]
        prompt = "Enforce which symmetries? chosen: {} (All = {})".format(
            " ".join(chosen) or "none", report.group.name)
        key = rs.GetString(prompt, "All" if not chosen else "Done", options)
        if key is None or key == "All":
            return None
        if key == "Done":
            return chosen or None
        if key in available and key not in chosen:
            chosen.append(key)


def bake_geometry(report, keys, seam=None):
    path = layer("Symmetry")
    geo = report.geometry()
    c = geo["centre"]
    guid = rs.AddPoint([c[0], c[1], 0.0])
    rs.ObjectLayer(guid, path)
    for element, line in zip(report.group.mirror_elements(), geo["mirrors"]):
        guid = rs.AddLine(list(line.start), list(line.end))
        rs.ObjectLayer(guid, path)
        guid = rs.AddTextDot(element.key, list(line.end))
        rs.ObjectLayer(guid, path)
    bake_polylines([points_of(p) for p in geo["rotations"]], path, clear_existing=False)
    rotations = [e.key for e in report.group.rotations() if e.kind != "identity"]
    if rotations and len(geo["rotations"]) > 1:
        tick = points_of(geo["rotations"][1])
        guid = rs.AddTextDot(" ".join(rotations), tick[-1])
        rs.ObjectLayer(guid, path)

    outline = report.unit_outline(keys=keys, seam=seam)
    bake_polylines([points_of(p) for p in outline["loops"]], layer("UnitOutline"))


def bake_seams(mesh):
    path = layer("Seams")
    segments = []
    for u, v in mesh.edges():
        pu, pv = mesh.vertex_coordinates(u), mesh.vertex_coordinates(v)
        if any(s.locate(pu, mesh.eps) is not None and s.locate(pv, mesh.eps) is not None for s in mesh.seams):
            segments.append([pu, pv])
    bake_polylines(segments, path, clear_existing=False)


def main():
    settings = get_settings()
    spacing = settings["triangulation_spacing"]
    target = settings["target_length"]

    route = ask_route()
    if route == "exit":
        return
    stage = "reading the domain"
    try:
        outer, inners, guides, poles = read_boundaries(spacing=resolve_spacing(settings))
        outer = points_of(outer)
        inners = [points_of(loop) for loop in inners]
        guides = [points_of(curve) for curve in guides]
        poles = [[p[0], p[1], 0.0] for p in poles]

        stage = "building the decomposition"
        if route == "skeleton":
            decomposition = SkeletonDecomposition.from_boundary(
                outer, inner_boundaries=inners, polyline_features=guides, point_features=poles,
                target_length=spacing)
        else:
            if poles:
                print("note: the field route takes no point features -- {} pole(s) ignored".format(len(poles)))
            decomposition = FieldDecomposition.from_boundary(
                outer, inner_boundaries=inners, guides=guides or None, target_length=spacing,
                relax=bool(guides), solve=False)

        stage = "find_symmetry"
        report = decomposition.find_symmetry()
        print(report.summary())
        for label, stage_report in report.by_feature():
            print("  {:8s} {}".format(label, stage_report.group.name))
        bake_geometry(report, None)
        sc.doc.Views.Redraw()
        if report.group.is_trivial:
            print("no symmetry detected -- the unit is the whole domain")

        keys = ask_keys(report)
        bake_geometry(report, keys)
        sc.doc.Views.Redraw()
        go = rs.GetString("Unit drawn on {}::UnitOutline -- mesh it?".format(ROOT), "Continue", ["Continue", "Exit"])
        if go != "Continue":
            return

        timings = {}
        start = time.time()
        stage = "symmetry_unit"
        coarse_unit = decomposition.symmetry_unit(keys=keys)
        timings["unit"] = time.time() - start
        seams = coarse_unit.seams
        if seams and seams[0].kind == "rotation":
            bake_geometry(report, keys, seam=seams[0].angle)
        ok, notes = coarse_unit.check()
        print(coarse_unit, "check:", "ok" if ok else "; ".join(notes))
        for note in coarse_unit.symmetry.get("notes", []):
            print("  note:", note)
        bake_mesh(coarse_unit, layer("CoarseUnit"))
        bake_seams(coarse_unit)

        stage = "densities"
        coarse_unit.collect_strips()
        coarse_unit.set_strips_density_target(target)

        stage = "expand_symmetrically (coarse)"
        start = time.time()
        coarse_layout = coarse_unit.expand_symmetrically()
        timings["expand coarse"] = time.time() - start
        bake_mesh(coarse_layout, layer("CoarseLayout"))

        stage = "quad_mesh"
        start = time.time()
        quad_unit = coarse_unit.quad_mesh()
        timings["densify unit"] = time.time() - start
        bake_mesh(quad_unit, layer("QuadUnit"))

        stage = "expand_symmetrically (dense)"
        start = time.time()
        quad_mesh = quad_unit.expand_symmetrically(coarse_unit)
        timings["expand dense"] = time.time() - start
        bake_mesh(quad_mesh, layer("QuadMesh"))
    except Exception as exc:
        print("SYMMETRY TEST FAILED at {}: {}: {}".format(stage, type(exc).__name__, exc))
        return

    group = coarse_unit.group
    inv_coarse = mesh_invariance(coarse_layout, group)
    inv_dense = mesh_invariance(quad_mesh, group)
    quality = mesh_quality(quad_mesh)
    print("")
    print("symmetry test -- {} route".format(route))
    print("  enforced   {} ({}) of detected {}".format(group.name, " ".join(group.keys()) or "-", report.group.name))
    print("  unit       {} patches, seam-matching splits {}".format(
        coarse_unit.number_of_faces(), coarse_unit.symmetry.get("matching_cuts", 0)))
    print("  coarse     {} patches, symmetric {:.0%} (worst {:.2e}), manifold {}".format(
        coarse_layout.number_of_faces(), inv_coarse["share"], inv_coarse["worst"], coarse_layout.is_manifold()))
    print("  dense      {} faces, symmetric {:.0%} (worst {:.2e}), manifold {}".format(
        quad_mesh.number_of_faces(), inv_dense["share"], inv_dense["worst"], quad_mesh.is_manifold()))
    print("  quality    min angle {:.2f}, max aspect {:.2f}".format(quality["min_angle"], quality["aspect_max"]))
    print("  time       " + ", ".join("{} {:.2f}s".format(k, v) for k, v in timings.items()))
    print("  (standalone check, baked under '{}' only -- it feeds nothing into "
          "the main pipeline)".format(ROOT))
    sc.doc.Views.Redraw()


if __name__ == "__main__":
    main()
