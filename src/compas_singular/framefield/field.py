"""The cross field and its singularities, stored as ``u = exp(4i theta)`` per background vertex.

``relax=False`` is a linear Dirichlet solve; ``relax=True`` adds diffusion + normalisation (disc poles at ~0.85).
"""

from __future__ import annotations

from cmath import phase
from dataclasses import dataclass
from math import cos
from math import sin
from numbers import Number
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import factorized
from scipy.sparse.linalg import spsolve

from compas.data import Data
from compas_singular.framefield.constraints import PERIOD
from compas_singular.framefield.constraints import Constraint
from compas_singular.framefield.constraints import from_boundary
from compas_singular.framefield.constraints import representation
from compas_singular.geometry.polyline import signed_area

if TYPE_CHECKING:
    from compas_singular.framefield.background import BackgroundMesh
    from compas_singular.framefield.locator import PointLocator
    from compas_singular.framefield.symmetry import Symmetry


__all__ = ["CrossField", "FieldInputs", "field_provenance", "wrap_to_period"]


#: Bumped when the layout of ``CrossField.save_to_json`` changes in a way a
#: reader has to notice. Newer files are refused rather than guessed at.
JSON_VERSION = 1

#: Written into every file and checked on load.
JSON_TYPE = "compas_singular.framefield.CrossField"

#: Coordinate rounding for ``CrossField.inputs``: well below anything
#: geometric, coarse enough to absorb the last-bit noise of a CAD round trip.
#: **Changing it makes every stored field report its outline changed.**
COORDINATE_DIGITS = 9


def _canonical_curve(curve: Any) -> list[list[float]]:
    """One curve as a plain, rounded list of ``[x, y, z]``, without a repeated
    closing point -- a loop given closed and given open is the same input."""
    points = [[round(float(point[i]), COORDINATE_DIGITS) for i in range(3)] for point in curve]
    if len(points) > 1 and points[0] == points[-1]:
        points = points[:-1]
    return points


def _plain(value: Any) -> Any:
    """A solve parameter as something JSON can hold and a human can read."""
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, Number):
        return float(value)
    # a Symmetry: centre, elements and enabled steps are its identity
    if hasattr(value, "centre") and hasattr(value, "steps"):
        return {
            "symmetry": [round(float(c), COORDINATE_DIGITS) for c in value.centre],
            "names": sorted(value.names()),
            "steps": sorted(value.steps),
        }
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return dict((str(k), _plain(v)) for k, v in value.items())
    return repr(value)


def _describe_difference(before: dict[str, Any] | None, after: dict[str, Any] | None) -> str:
    """The first differing entry of two dicts, as ``'name a -> b'``."""
    before = before or {}
    after = after or {}
    for name in sorted(set(before) | set(after)):
        old = before.get(name, "<unset>")
        new = after.get(name, "<unset>")
        if old != new:
            return "{} {} -> {}".format(name, old, new)
    return "no visible difference"


@dataclass
class FieldInputs(object):
    """Everything a field is solved from, with the defaults; see ``CrossField.from_boundary``."""

    outer_boundary: Any
    inner_boundaries: Any = None
    guides: Any = None
    mode: str = "perpendicular"
    target_length: float | None = None
    guide_weight: float | None = 1.0
    guide_band: float | None = None
    relax: bool = False
    tau: float | None = None
    symmetry: Any = "auto"

    def resolved(self) -> FieldInputs:
        """A copy with ``guides`` as a list and ``'auto'`` symmetry detected."""
        from compas_singular.framefield.constraints import as_curve_list
        from compas_singular.framefield.symmetry import Symmetry

        guides = as_curve_list(self.guides)
        symmetry = self.symmetry
        if symmetry == "auto":
            symmetry = Symmetry.detect([self.outer_boundary] + list(self.inner_boundaries or []) + list(guides or []))
        return FieldInputs(self.outer_boundary, self.inner_boundaries, guides, self.mode, self.target_length, self.guide_weight, self.guide_band, self.relax, self.tau, symmetry)

    def provenance(self) -> dict[str, Any]:
        """The inputs canonicalised for comparison and storage, as plain JSON.

        Stored in every saved field; its shape must not change.
        """
        inputs = self.resolved()
        params = (
            ("mode", inputs.mode),
            ("target_length", inputs.target_length),
            ("guide_weight", inputs.guide_weight),
            ("guide_band", inputs.guide_band),
            ("relax", inputs.relax),
            ("tau", inputs.tau),
            ("symmetry", inputs.symmetry),
        )
        return {
            "geometry": {
                "outer": _canonical_curve(inputs.outer_boundary),
                "inners": [_canonical_curve(loop) for loop in (inputs.inner_boundaries or [])],
                "guides": [_canonical_curve(curve) for curve in inputs.guides],
            },
            "params": dict((name, _plain(value)) for name, value in params),
        }


def field_provenance(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """``FieldInputs.provenance`` of the inputs given, which are
    ``CrossField.from_boundary``'s."""
    return FieldInputs(*args, **kwargs).provenance()


def wrap_to_period(delta: float) -> float:
    """Fold an angle difference into (-pi/4, +pi/4], the cross's half-period.
    Two crosses never differ by more than 45 degrees."""
    return delta - PERIOD * round(delta / PERIOD)


class CrossField(Data):
    """A boundary-aligned cross field on a planar triangulation.

    Attributes
    ----------
    background : BackgroundMesh
    u : dict[int, complex]
        ``exp(i*4*theta)`` per vertex; ``|u| = 1`` only when ``relaxed``.
    theta : dict[int, float]
        A representative cross angle per vertex, in (-pi/4, pi/4].
    relaxed : bool
        Which solver produced this field.
    iterations, residual : int, float or None
        Diffusion steps taken and the final ``max|u_k+1 - u_k|``, when relaxed.
    guides : list
        The guide curves the field was solved with. Densification spends
        element quality on alignment only when there are some.
    symmetry : Symmetry or None
        The symmetry group of the input.
    singularity_points : dict[int, [x, y, z]]
        Where ``symmetry.snap_singularities`` put each singularity; empty when
        no symmetry was applied.
    snap_report : dict
        What the snapping did.
    inputs : dict or None
        ``FieldInputs.provenance`` of what this field was solved from;
        ``None`` for a field built through ``solve`` directly.
    """

    def __init__(
        self,
        background: BackgroundMesh,
        u: dict[int, complex],
        relaxed: bool = False,
        iterations: int | None = None,
        residual: float | None = None,
    ) -> None:
        super(CrossField, self).__init__()
        self.background = background
        self.u = u
        self.theta = {vkey: phase(value) / 4.0 for vkey, value in u.items()}
        self.relaxed = relaxed
        self.iterations = iterations
        self.residual = residual
        self._singularities = None
        self.guides = []
        self.symmetry = None
        self.singularity_points = {}
        self.snap_report = {}
        self.inputs = None
        self._locator = None  # derived, never pickled or saved

    def __getstate__(self) -> dict[str, Any]:
        """Everything but the point locator, for ``pickle`` and ``deepcopy``."""
        state = dict(self.__dict__)
        state["_locator"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._locator = None

    # ------------------------------------------------------------------
    # the solve
    # ------------------------------------------------------------------

    @classmethod
    def solve(
        cls,
        background: BackgroundMesh,
        constraints: list[Constraint] | None = None,
        relax: bool = False,
        tau: float | None = None,
        tol: float = 1e-9,
        max_iterations: int = 20000,
    ) -> CrossField:
        """Smoothest cross field satisfying ``constraints``, with uniform edge weights.

        Parameters
        ----------
        background : BackgroundMesh
        constraints : list[Constraint], optional
            Defaults to ``from_boundary(background)``: tangent to every wall.
        relax : bool, optional
            Keep ``|u| = 1`` by diffusion + normalisation. See the module
            docstring.
        tau : float, optional
            Diffusion time per step, ``relax`` only. Defaults to
            ``(domain diagonal / 25) ** 2`` -- a property of the domain, so the
            steady state does not move when the background is refined.
        tol, max_iterations : optional
            Steady-state test on ``max|u_k+1 - u_k|``, ``relax`` only.

        Returns
        -------
        CrossField
        """
        if constraints is None:
            constraints = from_boundary(background)

        mesh = background.mesh
        vkeys = list(mesh.vertices())
        index = {vkey: i for i, vkey in enumerate(vkeys)}
        n = len(vkeys)

        hard = {}
        soft = {}
        for c in constraints:
            if c.vkey not in index:
                continue
            if c.weight is None:
                hard[index[c.vkey]] = representation(c.direction)
            else:
                # soft constraints on one vertex accumulate, e.g. where guides overlap
                w, acc = soft.get(index[c.vkey], (0.0, 0j))
                soft[index[c.vkey]] = (w + c.weight, acc + c.weight * representation(c.direction))

        rows, cols, vals = [], [], []
        for u_key, v_key in mesh.edges():
            i, j = index[u_key], index[v_key]
            rows += [i, j, i, j]
            cols += [i, j, j, i]
            vals += [1.0, 1.0, -1.0, -1.0]
        for i, (w, _) in soft.items():
            rows.append(i)
            cols.append(i)
            vals.append(w)
        A = coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr().astype(complex)

        b = np.zeros(n, dtype=complex)
        for i, (_, acc) in soft.items():
            b[i] = acc

        free = [i for i in range(n) if i not in hard]
        if not free:
            raise ValueError("every vertex is hard-constrained; nothing to solve")
        fixed = sorted(hard)
        x_fixed = np.array([hard[i] for i in fixed], dtype=complex)

        A_ff = A[free][:, free]
        A_fh = A[free][:, fixed]
        rhs = b[free] - A_fh.dot(x_fixed)

        x_free = spsolve(A_ff.tocsc(), rhs)

        iterations = residual = None
        if relax:
            x_free, iterations, residual = cls._relax(A, b, free, fixed, x_fixed, x_free, background, tau, tol, max_iterations)

        values = np.zeros(n, dtype=complex)
        values[free] = x_free
        values[fixed] = x_fixed

        return cls(background, {vkey: complex(values[index[vkey]]) for vkey in vkeys}, relaxed=bool(relax), iterations=iterations, residual=residual)

    @staticmethod
    def _relax(
        A: Any,
        b: np.ndarray,
        free: list[int],
        fixed: list[int],
        x_fixed: np.ndarray,
        x_start: np.ndarray,
        background: BackgroundMesh,
        tau: float | None,
        tol: float,
        max_iterations: int,
    ) -> tuple[np.ndarray, int, float]:
        """Diffusion + normalisation to steady state (Dai/Qiao/Wang Alg. 4.1), implicit and factorised once."""
        from scipy.sparse import identity

        if tau is None:
            tau = (background.diagonal / 25.0) ** 2

        n = A.shape[0]
        step = (identity(n, format="csr", dtype=complex) + tau * A).tocsr()
        solve_ff = factorized(step[free][:, free].tocsc())
        offset = tau * (b[free] - A[free][:, fixed].dot(x_fixed))

        u = np.asarray(x_start, dtype=complex).copy()
        u /= np.maximum(np.abs(u), 1e-12)

        residual = float("inf")
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            nxt = solve_ff(u + offset)
            nxt /= np.maximum(np.abs(nxt), 1e-12)
            residual = float(np.max(np.abs(nxt - u)))
            u = nxt
            if residual < tol:
                break
        return u, iterations, residual

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------

    def directions(self, vkey: int) -> list[list[float]]:
        """The four arms of the cross at a vertex, as unit vectors."""
        t = self.theta[vkey]
        return [[cos(t + k * PERIOD), sin(t + k * PERIOD), 0.0] for k in range(4)]

    def magnitude(self, vkey: int) -> float:
        """``|u|``. Near 0 means the vertex is close to a singularity."""
        return abs(self.u[vkey])

    def angle_in_face(
        self,
        fkey: int,
        bary: tuple[float, float, float],
        reference: float | None = None,
    ) -> float:
        """The cross angle inside a triangle, interpolating ``u`` rather than angles.

        Parameters
        ----------
        fkey : int
        bary : (float, float, float)
            Barycentric coordinates, in the order of ``face_vertices(fkey)``.
        reference : float, optional
            Return the arm nearest this angle.

        Returns
        -------
        float
        """
        vkeys = self.background.mesh.face_vertices(fkey)
        value = 0j
        for w, vkey in zip(bary, vkeys):
            value += w * self.u[vkey]
        if value == 0:
            # exactly on the zero: fall back to the nearest vertex
            value = self.u[max(zip(bary, vkeys))[1]]
        theta = phase(value) / 4.0
        if reference is not None:
            theta += PERIOD * round((reference - theta) / PERIOD)
        return theta

    # ------------------------------------------------------------------
    # singularities
    # ------------------------------------------------------------------

    def singularities(self) -> list[tuple[int, int]]:
        """Singular faces and their indices in quarter-turns (``+1`` valence 3, ``-1`` valence 5).

        Returns
        -------
        list[(int, int)]
            ``(fkey, index)`` for every face with a non-zero index.
        """
        if self._singularities is None:
            out = []
            mesh = self.background.mesh
            for fkey in mesh.faces():
                a, b, c = mesh.face_vertices(fkey)
                total = wrap_to_period(self.theta[b] - self.theta[a]) + wrap_to_period(self.theta[c] - self.theta[b]) + wrap_to_period(self.theta[a] - self.theta[c])
                k = int(round(total / PERIOD))
                if k:
                    out.append((fkey, k))
            self._singularities = out
        return self._singularities

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def _oriented_boundary_loops(self) -> list[list[int]]:
        """Boundary loops with the domain on the left: outer CCW, holes CW."""
        mesh = self.background.mesh
        loops = []
        for loop in mesh.boundaries():
            area = signed_area([mesh.vertex_coordinates(v) for v in loop])
            loops.append((abs(area), area, loop))
        loops.sort(reverse=True)  # largest is the outer one
        out = []
        for i, (_, area, loop) in enumerate(loops):
            want_ccw = i == 0
            out.append(loop if (area > 0) == want_ccw else list(reversed(loop)))
        return out

    def poincare_hopf(self) -> dict[str, Any]:
        """Check that the interior indices sum to the field's measured boundary winding.

        Returns
        -------
        dict
            ``interior_index_sum``, ``boundary_winding``, ``residual_radians``,
            ``singular_faces`` and ``ok``.
        """
        winding = 0.0
        for loop in self._oriented_boundary_loops():
            n = len(loop)
            for i in range(n):
                winding += wrap_to_period(self.theta[loop[(i + 1) % n]] - self.theta[loop[i]])

        interior = sum(k for _, k in self.singularities())
        expected = int(round(winding / PERIOD))
        return {
            "interior_index_sum": interior,
            "boundary_winding": expected,
            "residual_radians": winding - expected * PERIOD,
            "singular_faces": len(self.singularities()),
            "ok": interior == expected,
        }

    def report(self) -> dict[str, Any]:
        """``poincare_hopf`` plus the solve's diagnostics.

        ``min_magnitude`` and ``mean_magnitude`` are only informative when the
        field is not relaxed -- relaxation pins every magnitude to 1.
        """
        mags = [abs(v) for v in self.u.values()]
        info = dict(self.poincare_hopf())
        info["min_magnitude"] = min(mags)
        info["mean_magnitude"] = sum(mags) / len(mags)
        info["relaxed"] = self.relaxed
        info["iterations"] = self.iterations
        info["residual"] = self.residual
        return info

    # ------------------------------------------------------------------
    # solving from a domain
    # ------------------------------------------------------------------

    @classmethod
    def from_boundary(
        cls,
        outer_boundary: Any,
        inner_boundaries: Any = None,
        guides: Any = None,
        mode: str = "perpendicular",
        target_length: float | None = None,
        guide_weight: float | None = 1.0,
        guide_band: float | None = None,
        relax: bool = False,
        tau: float | None = None,
        symmetry: Symmetry | str | None = "auto",
    ) -> CrossField:
        """A field over a domain: background, constraints, solve and symmetry, without tracing.

        Parameters
        ----------
        outer_boundary : list[[x, y, z]]
        inner_boundaries : list[list[[x, y, z]]], optional
        guides : list[list[[x, y, z]]], optional
            Guide curves -- cables, force lines. A LIST of curves; a lone curve
            is accepted too.
        mode : {'perpendicular', 'tangent'}, optional
            See ``constraints.from_curves``.
        target_length : float, optional
            Background spacing -- not the quad size.
        guide_weight : float, optional
            Weight of the guide constraints, ``None`` for hard.
        guide_band : float, optional
            How far from a guide it constrains the field. Defaults to the
            background spacing.
        relax : bool, optional
            Use the diffusion + normalisation solver; see the module docstring.
        tau : float, optional
            Diffusion time per step, ``relax`` only.
        symmetry : Symmetry or 'auto' or None, optional
            ``'auto'`` detects the input's symmetry group and keeps it through
            the background, the field and its singularities. ``None`` ignores
            symmetry. An asymmetric input is unaffected either way.

        Returns
        -------
        CrossField
            With ``guides``, ``symmetry``, ``singularity_points``,
            ``snap_report`` and ``inputs`` set.
        """
        # imported here: ``background`` and ``symmetry`` import this module
        from compas_singular.framefield.background import BackgroundMesh
        from compas_singular.framefield.constraints import from_curves
        from compas_singular.framefield.symmetry import snap_singularities
        from compas_singular.framefield.symmetry import symmetrise

        inputs = FieldInputs(outer_boundary, inner_boundaries, guides, mode, target_length, guide_weight, guide_band, relax, tau, symmetry).resolved()
        symmetry = inputs.symmetry

        background = BackgroundMesh.from_boundary(outer_boundary, inner_boundaries, target_length=target_length, symmetry=symmetry)

        constraints = from_boundary(background)
        if inputs.guides:
            constraints += from_curves(background, inputs.guides, mode=mode, weight=guide_weight, band=guide_band)

        field = cls.solve(background, constraints, relax=relax, tau=tau)

        points = {}
        snap_report = {}
        if symmetry is not None and symmetry.enabled("field"):
            # project onto the symmetric subspace: exact, since rotations act
            # trivially on exp(i*4*theta) and reflections conjugate it
            field = symmetrise(field, symmetry)
        if symmetry is not None and symmetry.enabled("singularities"):
            # take each singularity off the arbitrary face that won the winding
            # tie, onto the field's own |u| minimum
            field, points, snap_report = snap_singularities(field)

        field.guides = list(inputs.guides or [])
        field.symmetry = symmetry
        field.singularity_points = points
        field.snap_report = snap_report
        field.inputs = inputs.provenance()
        return field

    # ------------------------------------------------------------------
    # provenance -- has the domain moved under this field?
    # ------------------------------------------------------------------

    def mismatch(self, *args: Any, **kwargs: Any) -> str | None:
        """Why this field does not belong to these inputs, or ``None``.

        Ask it of every field read back from disk before densifying with it.

        Returns
        -------
        str or None
            The first difference, as a sentence.
        """
        if self.inputs is None:
            return "this field carries no record of its inputs -- it was built through CrossField.solve rather than from_boundary, so whether it matches cannot be answered"

        other = field_provenance(*args, **kwargs)
        mine = self.inputs.get("geometry") or {}
        theirs = other["geometry"]
        labels = {"outer": "outer boundary", "inners": "holes", "guides": "guide curves"}
        for role in ("outer", "inners", "guides"):
            if mine.get(role) != theirs.get(role):
                return "the {} changed".format(labels[role])

        if (self.inputs.get("params") or {}) != other["params"]:
            return "solver settings changed -- {}".format(_describe_difference(self.inputs.get("params"), other["params"]))
        return None

    def matches(self, *args: Any, **kwargs: Any) -> bool:
        """Whether this field was solved from these inputs. ``False`` when it
        carries no record of its inputs: unknown is not equal."""
        return self.mismatch(*args, **kwargs) is None

    # ------------------------------------------------------------------
    # serialisation
    # ------------------------------------------------------------------

    def save_to_json(self, filepath: str, pretty: bool = False) -> str:
        """Write this field to a JSON file, making the folder if needed. Returns ``filepath``.

        Returns
        -------
        str
            ``filepath``.
        """
        import os

        import compas

        folder = os.path.dirname(os.path.abspath(filepath))
        if folder and not os.path.isdir(folder):
            os.makedirs(folder)
        compas.json_dump(self.__jsondata__(), filepath, pretty=pretty)
        return filepath

    @property
    def __data__(self) -> dict[str, Any]:
        """compas ``Data``: the ``save_to_json`` payload, so a field can
        travel inside a larger document (a session) and come back exactly."""
        return self.__jsondata__()

    @classmethod
    def __from_data__(cls, data: dict[str, Any]) -> CrossField:
        return cls.__from_jsondata__(data)

    def __jsondata__(self) -> dict[str, Any]:
        """The payload ``save_to_json`` writes: plain types plus one Mesh."""
        symmetry = None
        if self.symmetry is not None:
            symmetry = {"centre": [float(c) for c in self.symmetry.centre], "names": list(self.symmetry.names()), "steps": list(self.symmetry.steps)}

        return {
            "type": JSON_TYPE,
            "version": JSON_VERSION,
            "inputs": self.inputs,
            "background": {
                "mesh": self.background.mesh,
                "outer": [[float(c) for c in p] for p in self.background.outer],
                "inners": [[[float(c) for c in p] for p in loop] for loop in self.background.inners],
                "target_length": float(self.background.target_length),
            },
            # re/im as floats round-trip exactly
            "u": {str(vkey): [value.real, value.imag] for vkey, value in self.u.items()},
            "relaxed": bool(self.relaxed),
            "iterations": self.iterations,
            "residual": self.residual,
            # stored: snap_singularities overwrites them, and u cannot recover it
            "singularities": (None if self._singularities is None else [[int(fkey), int(index)] for fkey, index in self._singularities]),
            "guides": [[[float(c) for c in p] for p in curve] for curve in (self.guides or [])],
            "symmetry": symmetry,
            "singularity_points": {str(fkey): [float(c) for c in point] for fkey, point in (self.singularity_points or {}).items()},
            "snap_report": self.snap_report or {},
        }

    @classmethod
    def load_from_json(cls, filepath: str, default: Any = None) -> CrossField | Any:
        """**Read a field back.** The inverse of ``save_to_json``.

        Does NOT check that the field still belongs to your domain -- ask
        ``mismatch`` straight afterwards, every time.

        Parameters
        ----------
        filepath : str
        default : optional
            Returned when the file does not exist.

        Raises
        ------
        ValueError
            If the file is not a field, or is newer than this version reads.
        """
        import os

        import compas

        if not os.path.isfile(filepath):
            return default
        return cls.__from_jsondata__(compas.json_load(filepath))

    @classmethod
    def __from_jsondata__(cls, data: dict[str, Any]) -> CrossField:
        """Rebuild from ``__jsondata__``'s payload."""
        from compas_singular.framefield.background import BackgroundMesh
        from compas_singular.framefield.symmetry import ELEMENTS
        from compas_singular.framefield.symmetry import Symmetry

        found = data.get("type") if isinstance(data, dict) else type(data).__name__
        if found != JSON_TYPE:
            raise ValueError("not a CrossField file: expected type {!r}, found {!r}".format(JSON_TYPE, found))
        version = data.get("version")
        if version is None or version > JSON_VERSION:
            raise ValueError("this file is version {!r}; this compas_singular understands up to {}. Re-solve the field, or update.".format(version, JSON_VERSION))

        bg = data["background"]
        background = BackgroundMesh(bg["mesh"], bg["outer"], bg["inners"], bg["target_length"])

        # JSON object keys are strings; vertex and face keys are ints
        u = {int(vkey): complex(value[0], value[1]) for vkey, value in data["u"].items()}

        field = cls(background, u, relaxed=data.get("relaxed", False), iterations=data.get("iterations"), residual=data.get("residual"))

        singularities = data.get("singularities")
        if singularities is not None:
            field._singularities = [(int(fkey), int(index)) for fkey, index in singularities]
        field.guides = [[list(p) for p in curve] for curve in data.get("guides") or []]
        field.singularity_points = {int(fkey): list(point) for fkey, point in (data.get("singularity_points") or {}).items()}
        field.snap_report = data.get("snap_report") or {}
        field.inputs = data.get("inputs")

        symmetry = data.get("symmetry")
        if symmetry is not None:
            by_name = {element[5]: element for element in ELEMENTS}
            field.symmetry = Symmetry(symmetry["centre"], [by_name[name] for name in symmetry["names"]], symmetry["steps"])
        return field

    # ------------------------------------------------------------------
    # densifying with it
    # ------------------------------------------------------------------

    def locator(self) -> PointLocator:
        """**Point location on this field's background.** Built once, cached,
        never saved -- it is derived from the background alone."""
        if self._locator is None:
            from compas_singular.framefield.locator import PointLocator

            self._locator = PointLocator(self.background)
        return self._locator

    def densify(
        self,
        coarse: Any,
        edges_to_curves: dict[tuple[int, int], list[list[float]]] | None = None,
        spend: bool | None = None,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """**Densify a coarse layout with this field steering patch interiors.**

        What ``coarse.densification(field=field)`` calls; call it directly to
        get the statistics as well.

        Parameters
        ----------
        coarse : CoarseQuadMesh or CoarsePseudoQuadMesh
            Strips collected and densities set.
        edges_to_curves : dict, optional
            ``{(u, v): polyline}``, as ``densification`` takes it.
        spend : bool, optional
            Whether element quality may be spent on alignment. Defaults to
            whether this field has guides.
        **kwargs
            Passed to ``densify.field_densification``.

        Returns
        -------
        (QuadMesh, dict)
            The dense mesh (also set on ``coarse``) and the per-patch counts.
        """
        from compas_singular.framefield.densify import field_densification

        if spend is None:
            spend = bool(self.guides)
        return field_densification(coarse, self, self.locator(), edges_to_curves=edges_to_curves, spend=spend, **kwargs)
