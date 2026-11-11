from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from copy import deepcopy
from math import ceil
from math import floor
from typing import Any
from typing import Callable

from compas.geometry import Polyline
from compas.geometry import discrete_coons_patch
from compas.geometry import vector_average
from compas.itertools import linspace
from compas.itertools import pairwise
from compas.topology import connected_components
from compas.topology import vertex_adjacency_from_edges
from compas_singular.datastructures.mesh import Mesh
from compas_singular.datastructures.mesh import meshes_join_and_weld
from compas_singular.datastructures.mesh_quad import QuadMesh
from compas_singular.datastructures.mesh_quad_coarse.patterns import PATTERNS

__all__ = ["CoarseQuadMesh"]


def _int_key(key: Any) -> Any:
    """A key JSON stringified back to the integer it was; anything else as is."""
    if isinstance(key, str) and key.lstrip("-").isdigit():
        return int(key)
    return key


class CoarseQuadMesh(QuadMesh):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(CoarseQuadMesh, self).__init__(*args, **kwargs)
        self.attributes["strips_density"] = {}
        self.attributes["dense_pattern"] = {}
        self.attributes["vertex_coarse_to_dense"] = {}
        self.attributes["edge_coarse_to_dense"] = {}
        self.attributes["quad_mesh"] = None
        self.attributes["polygonal_mesh"] = None
        # The SHAPE of each coarse edge, when it is known. See ``edges_to_curves``.
        self.attributes["edges_to_curves"] = []
        self.attributes["decomposition_type"] = None

    @property
    def __data__(self) -> dict[str, Any]:
        """Everything but the derived dense meshes this layout last produced."""
        data = super(CoarseQuadMesh, self).__data__
        data["attributes"] = dict(data["attributes"], quad_mesh=None, polygonal_mesh=None)
        return data

    @classmethod
    def __from_data__(cls, data: dict[str, Any]) -> "CoarseQuadMesh":
        # JSON object keys are always strings, so a saved layout comes back with
        # strip keys '0', '1', ... -- and nothing fails loudly: strip and density
        # lookups still agree with each other, but ``get_face_pattern(0)`` misses
        # '0' and silently densifies every patch as ortho, and ``add_strip``'s
        # ``max(strips) + 1`` raises. Restore the integer keys, and the edge
        # tuples JSON turned into lists, as ``PseudoQuadMesh`` does for its poles.
        mesh = super(CoarseQuadMesh, cls).__from_data__(data)
        attributes = mesh.attributes
        attributes["strips"] = {_int_key(skey): [tuple(edge) for edge in edges] for skey, edges in (attributes.get("strips") or {}).items()}
        attributes["strips_density"] = {_int_key(skey): d for skey, d in (attributes.get("strips_density") or {}).items()}
        attributes["dense_pattern"] = {_int_key(fkey): pattern for fkey, pattern in (attributes.get("dense_pattern") or {}).items()}
        return mesh

    # --------------------------------------------------------------------------
    # constructors
    # --------------------------------------------------------------------------

    @classmethod
    def from_quad_mesh(
        cls, quad_mesh: QuadMesh, collect_strips: bool = True, collect_polyedges: bool = True, attribute_density: bool = True, strict: bool = False
    ) -> "CoarseQuadMesh":
        """Build coarse quad mesh from quad mesh with density and child-parent element data.

        Parameters
        ----------
        quad_mesh : QuadMesh
            A quad mesh.
        attribute_density : bool, optional
            Keep density data of dense quad mesh and inherit it as aatribute.
        strict : bool, optional
            Passed to ``QuadMesh.singularity_polyedge_decomposition``. Default is
            False. Setting it to True changes the resulting coarse layout.

        Returns
        ----------
        coarse_quad_mesh : CoarseQuadMesh
            A coarse quad mesh with density data.
        """
        polyedges = quad_mesh.singularity_polyedge_decomposition(strict=strict)

        # vertex data
        vertices = {vkey: quad_mesh.vertex_coordinates(vkey) for vkey in quad_mesh.vertices()}
        coarse_vertices_children = {vkey: vkey for polyedge in polyedges for vkey in [polyedge[0], polyedge[-1]]}
        coarse_vertices = {vkey: quad_mesh.vertex_coordinates(vkey) for vkey in coarse_vertices_children}

        # edge data
        coarse_edges_children = {(polyedge[0], polyedge[-1]): polyedge for polyedge in polyedges}
        singularity_edges = [(x, y) for polyedge in polyedges for u, v in pairwise(polyedge) for x, y in [(u, v), (v, u)]]

        # face data
        faces = {fkey: quad_mesh.face_vertices(fkey) for fkey in quad_mesh.faces()}
        adj_edges = {(f1, f2) for f1 in quad_mesh.faces() for f2 in quad_mesh.face_neighbors(f1) if f1 < f2 and quad_mesh.face_adjacency_halfedge(f1, f2) not in singularity_edges}
        coarse_faces_children = {}
        for i, connected_faces in enumerate(connected_components(vertex_adjacency_from_edges(adj_edges))):
            mesh = Mesh.from_vertices_and_faces(vertices, [faces[face] for face in connected_faces])
            coarse_faces_children[i] = [vkey for vkey in reversed(mesh.boundaries()[0]) if mesh.vertex_degree(vkey) == 2]

        coarse_quad_mesh = cls.from_vertices_and_faces(coarse_vertices, coarse_faces_children)

        # attribute relation child-parent element between coarse and dense quad meshes
        coarse_quad_mesh.attributes["vertex_coarse_to_dense"] = coarse_vertices_children
        coarse_quad_mesh.attributes["edge_coarse_to_dense"] = {u: {} for u in coarse_quad_mesh.vertices()}
        for (u, v), polyedge in coarse_edges_children.items():
            coarse_quad_mesh.attributes["edge_coarse_to_dense"][u][v] = polyedge
            coarse_quad_mesh.attributes["edge_coarse_to_dense"][v][u] = list(reversed(polyedge))

        # collect strip and polyedge attributes
        if collect_strips:
            coarse_quad_mesh.collect_strips()
        if collect_polyedges:
            coarse_quad_mesh.collect_polyedges()

        # store density attribute from input dense quad mesh
        if attribute_density:
            coarse_quad_mesh.set_strips_density(1)
            for skey in coarse_quad_mesh.strips():
                u, v = coarse_quad_mesh.strip_edges(skey)[0]
                d = len(coarse_edges_children.get((u, v), coarse_edges_children.get((v, u), [])))
                coarse_quad_mesh.set_strip_density(skey, d)

        # store quad mesh and use as polygonal mesh
        coarse_quad_mesh.set_quad_mesh(quad_mesh)
        coarse_quad_mesh.set_polygonal_mesh(deepcopy(quad_mesh))

        return coarse_quad_mesh

    # --------------------------------------------------------------------------
    # meshes getters and setters
    # --------------------------------------------------------------------------

    def get_quad_mesh(self) -> QuadMesh | None:
        return self.attributes["quad_mesh"]

    def set_quad_mesh(self, quad_mesh: QuadMesh) -> None:
        self.attributes["quad_mesh"] = quad_mesh

    def get_polygonal_mesh(self) -> Mesh | None:
        return self.attributes["polygonal_mesh"]

    def set_polygonal_mesh(self, polygonal_mesh: Mesh) -> None:
        self.attributes["polygonal_mesh"] = polygonal_mesh

    # --------------------------------------------------------------------------
    # edge curvature getter and setter
    # --------------------------------------------------------------------------

    def edges_to_curves(self) -> dict[tuple[int, int], list[list[float]]]:
        """``{(u, v): polyline}``: the stored shape of every coarse edge that has one, one direction per edge.

        Returns
        -------
        dict[tuple[int, int], list[[x, y, z]]]
            Keyed one way round per edge. ``densification`` looks up ``(v, u)`` and
            reverses when ``(u, v)`` is absent, so both directions are covered.
        """
        return {(u, v): points for u, v, points in self.attributes.get("edges_to_curves") or []}

    def set_edges_to_curves(self, edges_to_curves: dict[tuple[int, int], list[list[float]]] | None) -> None:
        """Remember the shape of each coarse edge, stored JSON-safe as ``[u, v, points]``. ``None`` or ``{}`` clears it.

        Parameters
        ----------
        edges_to_curves : dict[tuple[int, int], list[[x, y, z]]] or None
        """
        self.attributes["edges_to_curves"] = [[u, v, [list(point) for point in points]] for (u, v), points in (edges_to_curves or {}).items()]

    def shape_polylines(self) -> list[list[list[float]]]:
        """The polylines the coarse edges take their shape from (separatrices, branches, drawn curves). ``[]`` if none."""
        return [[list(point) for point in polyline] for polyline in self.attributes.get("shape_polylines") or []]

    def set_shape_polylines(self, polylines: list[list[list[float]]] | None) -> None:
        """Remember the polylines the edges take their shape from. ``None`` clears them."""
        self.attributes["shape_polylines"] = [[list(point) for point in polyline] for polyline in (polylines or [])]

    def _filtered_edges_to_curves(self, boundary_curvature: bool, skeleton_curvature: bool) -> dict[tuple[int, int], list[list[float]]]:
        """The stored ``edges_to_curves``, filtered by the boundary and skeleton curvature toggles.

        Parameters
        ----------
        boundary_curvature : bool
            Keep a stored curve for an edge on the layout boundary.
        skeleton_curvature : bool
            Keep a stored curve for an edge that is not -- the interior
            edges a skeleton or field decomposition traced as separatrices.

        Returns
        -------
        dict[tuple[int, int], list[[x, y, z]]]
        """
        stored = self.edges_to_curves()
        if not stored or (boundary_curvature and skeleton_curvature):
            return stored
        return {(u, v): curve for (u, v), curve in stored.items() if (boundary_curvature if self.is_edge_on_boundary(u, v) else skeleton_curvature)}

    def _create_patch_edge(self, u: int, v: int, d: int, edges_to_curves: dict[tuple[int, int], list[list[float]]] | None) -> list[list[float]]:
        """The ``d + 1`` points densifying edge ``(u, v)``, from its curve or else the chord.

        Parameters
        ----------
        u, v : hashable
            The edge, in the direction it is being densified.
        d : int
            The strip density -- ``d + 1`` points are returned, matching
            ``edge_point``.
        edges_to_curves : dict or None

        Returns
        -------
        list[[x, y, z]]
        """
        if edges_to_curves:
            if (u, v) in edges_to_curves:
                curve = Polyline(edges_to_curves[u, v])
                return [curve.point_at(t) for t in linspace(0, 1, d + 1)]
            if (v, u) in edges_to_curves:
                curve = Polyline(edges_to_curves[v, u])
                return [curve.point_at(t) for t in linspace(0, 1, d + 1)][::-1]
        curve = Polyline([self.vertex_coordinates(u), self.vertex_coordinates(v)])
        return [curve.point_at(t) for t in linspace(0, 1, d + 1)]

    # --------------------------------------------------------------------------
    # element child-parent relation getters
    # --------------------------------------------------------------------------

    def coarse_edge_dense_edges(self, u: int, v: int) -> list[int]:
        """Return the child edges, or polyedge, in the dense quad mesh from a parent edge in the coarse quad mesh."""
        return self.attributes["edge_coarse_to_dense"][u][v]

    # --------------------------------------------------------------------------
    # density getters and setters
    # --------------------------------------------------------------------------

    def get_strip_density(self, skey: int) -> int:
        """Get the density of a strip.

        Parameters
        ----------
        skey : hashable
            A strip key.

        Returns
        ----------
        int
            The strip density.
        """
        return self.attributes["strips_density"][skey]

    def get_strip_densities(self) -> dict[int, int]:
        """Get the density of a strip.

        Returns
        ----------
        dict
            The dictionary of the strip densities.
        """
        return self.attributes["strips_density"]

    # --------------------------------------------------------------------------
    # density setters
    # --------------------------------------------------------------------------

    def has_densities(self) -> bool:
        """Whether every strip already carries a density; a partial table counts as none.

        Returns
        -------
        bool
        """
        table = self.attributes.get("strips_density") or {}
        return bool(table) and all(skey in table for skey in self.strips())

    def set_strip_density(self, skey: int, d: int) -> None:
        """Set the densty of one strip.

        Parameters
        ----------
        skey : hashable
            A strip key.
        d : int
            A density parameter.
        """
        self.attributes["strips_density"][skey] = d

    def set_strips_density(self, d: int, skeys: list[int] | None = None) -> None:
        """Set the same density to all strips.

        Parameters
        ----------
        d : int
            A density parameter.
        skeys : list, None
            The keys of strips to set density. If is None, all strips are considered.
        """
        if skeys is None:
            skeys = self.strips()
        for skey in skeys:
            self.set_strip_density(skey, d)

    def set_strip_density_target(self, skey: int, t: float) -> None:
        """Set the strip densities based on a target length and the average length of the strip edges.

        Parameters
        ----------
        skey : hashable
            A strip key.
        t : float
            A target length.
        """
        self.set_strip_density(skey, int(ceil(vector_average([self.edge_length(u, v) for u, v in self.strip_edges(skey) if u != v]) / t)))

    def set_strip_density_func(self, skey: int, func: Callable, func_args: Any) -> None:
        """Set the strip densities based on a function.

        Parameters
        ----------
        skey : hashable
            A strip key.
        """
        self.set_strip_density(skey, int(func(skey, func_args)))

    def set_strips_density_target(self, t: float, skeys: list[int] | None = None) -> None:
        """Set the strip densities based on a target length and the average length of the strip edges.

        Parameters
        ----------
        t : float
            A target length.
        skeys : list, None
            The keys of strips to set density. If is None, all strips are considered.
        """
        if skeys is None:
            skeys = self.strips()
        for skey in skeys:
            self.set_strip_density_target(skey, t)

    def set_strips_density_func(self, func: Callable, func_args: Any, skeys: list[int] | None = None) -> None:
        """Set the strip densities based on a function.

        Parameters
        ----------
        skeys : list, None
            The keys of strips to set density. If is None, all strips are considered.
        """
        if skeys is None:
            skeys = self.strips()
        for skey in skeys:
            self.set_strip_density_func(skey, func, func_args)

    def set_mesh_density_face_target(self, nb_faces: int) -> None:
        """Set equal strip densities based on a target number of faces.

        Parameters
        ----------
        nb_faces : int
            The target number of faces.
        """
        n = (nb_faces / self.number_of_faces()) ** 0.5
        if ceil(n) - n > n - floor(n):
            n = int(floor(n))
        else:
            n = int(ceil(n))
        self.set_strips_density(n)

    # --------------------------------------------------------------------------
    # dense pattern setters and getters
    # --------------------------------------------------------------------------

    def dense_patterns(self) -> dict[int, str]:
        if self.attributes["dense_pattern"] == {}:
            self.attributes["dense_pattern"] = {fkey: "ortho" for fkey in self.faces()}
        return self.attributes["dense_pattern"]

    def get_face_pattern(self, fkey: int) -> str:
        fkeys = list(self.faces())
        if fkey not in fkeys:
            raise ValueError(f"The face key does not correspond to any face of the mesh. Allowed fkeys are: {fkeys}")

        face_patterns = self.attributes["dense_pattern"]
        if fkey not in face_patterns:
            self.attributes["dense_pattern"][fkey] = "ortho"
            print("This face did not have a patterns assigned to it yet. Defaulting to ortho.")
        return self.attributes["dense_pattern"][fkey]

    def get_faces_with_pattern(self, pattern: str) -> list[int]:
        if pattern not in PATTERNS:
            raise ValueError(f"This is not an allowed pattern type. Possible patterns are: {PATTERNS}")
        face_patterns = self.dense_patterns()
        return [fkey for fkey, face_pattern in face_patterns.items() if face_pattern == pattern]

    def set_face_pattern(self, fkey: int, pattern: str) -> None:
        fkeys = list(self.faces())
        if fkey not in fkeys:
            raise ValueError(f"The face key does not correspond to any face of the mesh. Allowed fkeys are: {fkeys}")
        if pattern not in PATTERNS:
            raise ValueError(f"This is not an allowed pattern type. Possible patterns are: {PATTERNS}")
        self.attributes["dense_pattern"][fkey] = pattern

    def set_global_face_pattern(self, pattern: str) -> None:
        if pattern not in PATTERNS:
            raise ValueError(f"This is not an allowed pattern type. Possible patterns are: {PATTERNS}")
        fkeys = self.faces()
        for fkey in fkeys:
            self.set_face_pattern(fkey, pattern)

    # --------------------------------------------------------------------------
    # densification
    # --------------------------------------------------------------------------

    def densification(
        self,
        boundary_curvature: bool = True,
        skeleton_curvature: bool = True,
        overwrite_edges_to_curves: dict[tuple[int, int], list[list[float]]] | None = None,
        field: Any = None,
    ) -> QuadMesh:
        """Generate a denser quad mesh from the coarse quad mesh and its strip densities.

        Parameters
        ----------
        boundary_curvature : bool, optional
            Use the shape ``edges_to_curves`` has stored for edges on the
            layout's own boundary, instead of chording them. Defaults to True.
            Ignored -- treated as True -- when ``overwrite_edges_to_curves`` is
            given.
        skeleton_curvature : bool, optional
            Same, for the edges that are NOT on the boundary -- the interior
            edges a skeleton or field decomposition traced as separatrices.
            Defaults to True. Ignored -- treated as True -- when
            ``overwrite_edges_to_curves`` is given.
        overwrite_edges_to_curves : dict, optional
            A dictionary with edges (u, v) pointing to a curve for
            densification, overriding whatever ``edges_to_curves`` has
            stored -- for every edge, regardless of ``boundary_curvature`` /
            ``skeleton_curvature``. The curves are lists of XYZ points.
        field : optional
            A ``CrossField`` to steer patch interiors; boundaries stay fixed. Any
            layout on the same walls works, including a skeleton one.

        Returns
        -------
        QuadMesh
            The dense mesh, also stored on this one -- ``get_quad_mesh()``.
        """
        if overwrite_edges_to_curves is not None:
            edges_to_curves = overwrite_edges_to_curves
        else:
            edges_to_curves = self._filtered_edges_to_curves(boundary_curvature, skeleton_curvature)

        if field is not None:
            # The field owns this: it carries its own background and builds its
            # own point locator, so a layout from ANY source -- a skeleton
            # decomposition, a hand-built mesh, one read back out of a document
            # -- can be densified with patch interiors that follow it, instead of
            # the bilinear blend of its own four sides that ``discrete_coons_patch``
            # gives and that never consults a field.
            #
            # Imported here and not at module scope: ``framefield.densify``
            # imports ``PseudoQuadMesh`` and ``meshes_join_and_weld`` from this
            # package, so a top-level import is a circular one -- this module is
            # reached while ``compas_singular.datastructures`` is still being
            # initialised. Nothing about ``field`` is type-checked, so any object
            # offering ``densify(coarse, edges_to_curves=...)`` works.
            dense, _stats = field.densify(self, edges_to_curves=edges_to_curves)
            self.set_quad_mesh(dense)
            return self.get_quad_mesh()

        edge_strip = {}
        for skey, edges in self.strips(data=True):
            for edge in edges:
                edge_strip[edge] = skey
                edge_strip[tuple(reversed(edge))] = skey

        face_meshes = {}
        for fkey in self.faces():
            polylines = []
            for u, v in self.face_halfedges(fkey):
                d = self.get_strip_density(edge_strip[(u, v)])
                polylines.append(self._create_patch_edge(u, v, d, edges_to_curves))
            ab, bc, cd, da = polylines
            vertices, faces = discrete_coons_patch(ab, bc, list(reversed(cd)), list(reversed(da)))
            face_meshes[fkey] = QuadMesh.from_vertices_and_faces(vertices, faces)

        self.set_quad_mesh(meshes_join_and_weld(list(face_meshes.values())))
        return self.get_quad_mesh()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass

    # import compas
    # from compas_singular.datastructures.mesh_quad.mesh_quad import QuadMesh
    # from compas.datastructures.mesh import mesh_smooth_centroid
    # from compas_plotters.meshplotter import MeshPlotter

    # #mesh = QuadMesh.from_obj(compas.get('faces.obj'))
    # # mesh = QuadMesh.from_json('/Users/Robin/Desktop/json/debug.json')
    # # mesh.collect_strips()
    # # mesh.collect_polyedges()

    # mesh_0 = CoarseQuadMesh.from_quad_mesh(QuadMesh.from_obj(compas.get('faces.obj')))
    # mesh_0.collect_strips()
    # # mesh_0.collect_polyedges()
    # # print(mesh_0.is_quadmesh())
    # # print(mesh_0.number_of_strips())

    # # vertices = [
    # [12.97441577911377, 24.33094596862793, 0.0], [18.310085296630859, 8.467333793640137, 0.0],
    # [30.052173614501953, 18.846050262451172, 0.0], [17.135400772094727, 16.750551223754883, 0.0],
    # [16.661802291870117, 22.973459243774414, 0.0], [14.180665969848633, 26.949295043945313, 0.0],
    # [36.052761077880859, 26.372636795043945, 0.0], [26.180931091308594, 21.778648376464844, 0.0],
    # [19.647378921508789, 12.288106918334961, 0.0], [9.355668067932129, 16.475896835327148, 0.0],
    # [18.929227828979492, 16.271940231323242, 0.0], [7.34525203704834, 12.111981391906738, 0.0],
    # [13.31309986114502, 14.699410438537598, 0.0], [18.699434280395508, 19.613750457763672, 0.0],
    # [11.913931846618652, 10.593378067016602, 0.0], [17.163223266601563, 26.870658874511719, 0.0],
    # [26.110898971557617, 26.634754180908203, 0.0], [22.851469039916992, 9.81414794921875, 0.0],
    # [21.051292419433594, 7.556171894073486, 0.0], [22.1370792388916, 19.089054107666016, 0.0]]
    # # faces = [
    # [15, 5, 0, 4], [0, 9, 12, 4], [9, 11, 14, 12], [14, 1, 8, 12], [1, 18, 17, 8], [17, 2, 7, 8],
    # [2, 6, 16, 7], [16, 15, 4, 7], [13, 19, 7, 4], [19, 10, 8, 7], [10, 3, 12, 8], [3, 13, 4, 12]]
    # # mesh = QuadMesh.from_vertices_and_faces(vertices, faces)
    # # mesh.collect_strips()
    # # mesh.collect_polyedges()

    # # mesh_0 = CoarseQuadMesh.from_quad_mesh(mesh)
    # # mesh_0.collect_strips()
    # # mesh_0.collect_polyedges()

    # # print(mesh_0.number_of_strips())
    # # print(mesh_0.attributes['vertex_coarse_to_dense'])
    # # print(mesh_0.attributes['edge_coarse_to_dense'])

    # # mesh_0.set_strips_density(1)
    # # mesh_0.set_strip_density(0, 6)
    # # mesh_0.set_strip_density(1, 5)
    # # mesh_0.densification()
    # # mesh_0.get_strip_densities()
    # #mesh_smooth_centroid(mesh_0.quad_mesh, kmax = 10)

    # # mesh_0.set_strips_density(10)
    # # mesh_0.densification()

    # plotter = MeshPlotter(mesh_0, figsize=(10, 10))
    # plotter.draw_edges()
    # plotter.draw_vertices()
    # plotter.draw_faces()
    # plotter.show()
