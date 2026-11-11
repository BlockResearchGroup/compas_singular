"""Run a meshing route on a unit domain with the caller's settings; the only route-aware module.

The background spacing is the whole domain's resolved spacing, never re-derived from the unit.
"""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING
from typing import Any
from typing import Sequence

if TYPE_CHECKING:
    from compas_singular.datastructures import CoarsePseudoQuadMesh
    from compas_singular.symmetry.domain import Domain


__all__ = ["mesher_for", "route_name", "domain_of"]


def route_name(decomposition: Any) -> str:
    name = type(decomposition).__name__
    if name == "SkeletonDecomposition":
        return "skeleton"
    if name == "FieldDecomposition":
        return "field"
    raise TypeError("no symmetry adapter for {}".format(name))


def domain_of(decomposition: Any) -> Domain:
    """The ``.domain.Domain`` a decomposition was built from."""
    from compas_singular.symmetry.domain import Domain

    inputs = getattr(decomposition, "inputs", None) or {}
    if not inputs.get("outer_boundary"):
        raise ValueError("this decomposition has no stored inputs -- build it with from_boundary to use symmetry")
    route = route_name(decomposition)
    if route == "skeleton":
        guides = inputs.get("polyline_features") or []
        poles = inputs.get("point_features") or []
    else:
        from compas_singular.framefield.constraints import as_curve_list

        guides = as_curve_list(inputs.get("guides")) or []
        poles = inputs.get("poles") or []
    return Domain(inputs["outer_boundary"], inputs.get("inner_boundaries") or [], guides, poles)


def mesher_for(decomposition: Any) -> SkeletonMesher | FieldMesher:
    route = route_name(decomposition)
    if route == "skeleton":
        return SkeletonMesher(decomposition)
    return FieldMesher(decomposition)


class SkeletonMesher(object):
    route = "skeleton"

    def __init__(self, decomposition: Any) -> None:
        self.cls = type(decomposition)
        self.inputs = dict(decomposition.inputs)

    def __call__(
        self,
        outer: Sequence[Sequence[float]],
        holes: Sequence[Sequence[Sequence[float]]],
        guides: Sequence[Any],
        poles: Sequence[Sequence[float]],
    ) -> tuple[CoarsePseudoQuadMesh, dict[str, Any]]:
        d = self.cls.from_boundary(
            outer,
            inner_boundaries=holes,
            polyline_features=guides,
            point_features=poles,
            target_length=self.inputs.get("target_length"),
            alpha=self.inputs.get("alpha", 0.02),
            d_min=self.inputs.get("d_min", 5),
        )
        coarse = d.coarse_mesh()
        curves, tally = d.edges_to_curves(coarse)
        coarse.set_edges_to_curves(curves)
        return coarse, {"decomposition": d, "field": None, "tally": tally, "polylines": [list(p) for p in (d.polylines or [])]}


class FieldMesher(object):
    route = "field"

    def __init__(self, decomposition: Any) -> None:
        self.cls = type(decomposition)
        self.inputs = dict(decomposition.inputs)
        target = self.inputs.get("target_length")
        if target is None:
            from compas_singular.geometry import bounding_box_diagonal
            from compas_singular.symmetry._geometry import open_loop

            outer = open_loop(self.inputs["outer_boundary"])
            inners = [open_loop(h) for h in (self.inputs.get("inner_boundaries") or [])]
            target = 0.02 * bounding_box_diagonal(outer, *inners)
        self.target = target

    def __call__(
        self,
        outer: Sequence[Sequence[float]],
        holes: Sequence[Sequence[Sequence[float]]],
        guides: Sequence[Any],
        poles: Sequence[Sequence[float]],
    ) -> tuple[CoarsePseudoQuadMesh, dict[str, Any]]:
        i = self.inputs
        d = self.cls.from_boundary(
            outer,
            inner_boundaries=holes,
            guides=guides or None,
            mode=i.get("mode", "perpendicular"),
            target_length=self.target,
            orthogonal=i.get("orthogonal"),
            guide_weight=i.get("guide_weight", 1.0),
            guide_band=i.get("guide_band"),
            relax=i.get("relax", False),
            tau=i.get("tau", i.get("field_tau")),
            symmetry=None,
        )
        coarse = d.coarse_mesh(poles=poles or ())
        curves = d.edges_to_curves()
        coarse.set_edges_to_curves(curves)
        return coarse, {"decomposition": d, "field": d.get_field(), "polylines": [list(p) for p in (d.polylines or [])]}
