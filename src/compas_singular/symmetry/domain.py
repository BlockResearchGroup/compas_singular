"""The input every meshing route takes: walls, holes, guides and poles."""

from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any
from typing import Sequence

from compas.data import Data
from compas_singular.symmetry._geometry import area_centroid
from compas_singular.symmetry._geometry import as_points
from compas_singular.symmetry._geometry import bbox_diagonal
from compas_singular.symmetry._geometry import open_loop
from compas_singular.symmetry._geometry import point_in_polygon

__all__ = ["Domain"]


class Domain(Data):
    """``outer``, ``inners``, ``guides`` and ``poles`` as plain point lists (loops open), as compas ``Data``."""

    def __init__(
        self,
        outer: Sequence[Any] | None = None,
        inners: Sequence[Sequence[Any]] | None = None,
        guides: Sequence[Sequence[Any]] | None = None,
        poles: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super(Domain, self).__init__()
        self.outer = open_loop(as_points(outer)) if outer is not None else []
        self.inners = [open_loop(as_points(loop)) for loop in (inners or [])]
        self.guides = [as_points(curve) for curve in (guides or [])]
        self.guides = [curve for curve in self.guides if len(curve) >= 2]
        self.poles = [[float(p[0]), float(p[1]), 0.0] for p in (poles or [])]

    @property
    def diagonal(self) -> float:
        return bbox_diagonal([self.outer] + self.inners + self.guides + [self.poles])

    def area_centroid(self) -> list[float] | None:
        """The centroid of the REGION: outer minus holes. ``None`` with no outer."""
        if len(self.outer) < 3:
            return None
        area, centre = area_centroid(self.outer)
        cx, cy = centre[0] * area, centre[1] * area
        total = area
        for loop in self.inners:
            if len(loop) < 3:
                continue
            a, c = area_centroid(loop)
            cx -= c[0] * a
            cy -= c[1] * a
            total -= a
        if total <= 1e-300:
            return [centre[0], centre[1], 0.0]
        return [cx / total, cy / total, 0.0]

    def contains(self, point: Sequence[float]) -> bool:
        if len(self.outer) < 3 or not point_in_polygon(point[0], point[1], self.outer):
            return False
        return not any(point_in_polygon(point[0], point[1], loop) for loop in self.inners if len(loop) >= 3)

    @property
    def __data__(self) -> dict[str, Any]:
        return self.to_data()

    @classmethod
    def __from_data__(cls, data: dict[str, Any]) -> Domain:
        return cls.from_data(data)

    def to_data(self) -> dict[str, Any]:
        return {"outer": self.outer, "inners": self.inners, "guides": self.guides, "poles": self.poles}

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Domain:
        return cls(data.get("outer"), data.get("inners"), data.get("guides"), data.get("poles"))
