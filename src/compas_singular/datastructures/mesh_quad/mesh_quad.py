from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from math import floor
from operator import itemgetter
from typing import Any
from typing import Callable
from typing import Iterator

from compas.geometry import Brep
from compas.geometry import Point
from compas.geometry import Polygon
from compas.geometry import Polyline
from compas.geometry import centroid_points
from compas.itertools import pairwise
from compas_singular.datastructures.mesh import Mesh
from compas_singular.datastructures.mesh_quad.grammar.add_strip import add_strip
from compas_singular.datastructures.mesh_quad.grammar.add_strip import add_strips
from compas_singular.datastructures.mesh_quad.grammar.delete_strip import delete_strip
from compas_singular.datastructures.mesh_quad.grammar.delete_strip import delete_strips
from compas_singular.utilities import list_split

__all__ = ["QuadMesh"]


class QuadMesh(Mesh):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(QuadMesh, self).__init__(*args, **kwargs)
        self.attributes["strips"] = {}
        self.attributes["polyedges"] = {}

    def strips(self, data: bool = False) -> Iterator[int] | Iterator[tuple[int, list[tuple[int, int]]]]:
        if not bool(self.attributes["strips"]):
            self.collect_strips()
        else:
            pass
            # print("Using earlier collected strips. Pay attention that the mesh has not changed since.")
        for skey in self.attributes["strips"]:
            if data:
                yield skey, self.attributes["strips"][skey]
            else:
                yield skey

    def strip_map(self, view: bool = False) -> list[tuple[int, Brep]]:
        skeys = self.strips()
        strips = []
        for skey in skeys:
            fkeys = self.strip_faces(skey)
            strip_polygons = []
            for fkey in fkeys:
                vkeys = self.face_vertices(fkey)
                face_pts = [Point(*self.vertex_coordinates(vkey)) for vkey in vkeys]
                strip_polygons.append(Polygon(face_pts))

            strip = Brep.from_polygons(strip_polygons)
            strips.append((skey, strip))

        if view:
            from compas_viewer.scene import Tag
            from compas_viewer.viewer import Viewer

            viewer = Viewer()
            group = viewer.scene.add_group("Strip map")

            for strip in strips:
                skey = strip[0]
                strip = strip[1]
                # Brep.centroid is not implemented yet.
                position = centroid_points(strip.points)
                tag = Tag(text=str(skey), position=position)
                group.add(strip, name="Strip: " + str(skey))
                group.add(tag)

            viewer.show()
            viewer.scene.clear()

        return strips

    def polyedges(self, data: bool = False) -> Iterator[int] | Iterator[tuple[int, list[int]]]:
        if not bool(self.attributes["polyedges"]):
            self.collect_polyedges()
        else:
            pass
            # print("Using earlier collected polyedges. Pay attention that the mesh has not changed since.")
        for key in self.attributes["polyedges"]:
            if data:
                yield key, self.attributes["polyedges"][key]
            else:
                yield key

    def polyedge_map(self, view: bool = False) -> list[tuple[int, Polyline]]:
        pkeys = self.polyedges()
        polyedges = []
        for pkey in pkeys:
            vkeys = self.polyedge_vertices(pkey)
            polyedge_pts = []
            for vkey in vkeys:
                pt = Point(*self.vertex_coordinates(vkey))
                polyedge_pts.append(pt)

            polyedge = Polyline(polyedge_pts)
            polyedges.append((pkey, polyedge))

        if view:
            from compas_viewer.scene import Tag
            from compas_viewer.viewer import Viewer

            viewer = Viewer()
            group = viewer.scene.add_group("Polyedge map")

            for polyedge in polyedges:
                pkey = polyedge[0]
                polyedge = polyedge[1]
                # Brep.centroid is not implemented yet.
                position = centroid_points(polyedge.points)
                tag = Tag(text=str(pkey), position=position)
                group.add(polyedge, name="Polyedge: " + str(pkey))
                group.add(tag)

            viewer.show()
            viewer.scene.clear()

        return polyedges

    # --------------------------------------------------------------------------
    # opposite elements
    # --------------------------------------------------------------------------

    def is_strip_face(self, fkey: int) -> bool:
        """Whether a strip can cross this face: it has four sides."""
        return len(self.face_vertices(fkey)) == 4

    def face_opposite_edge(self, u: int, v: int) -> tuple[int, int] | None:
        """Returns the opposite edge in the quad face, or ``None`` on a boundary or a non-quad face.

        Parameters
        ----------
        u : int
            The identifier of the edge start.
        v : int
            The identifier of the edge end.

        Returns
        -------
        tuple or None
            The opposite edge ``(w, x)``.
            None if (u, v) is a boundary halfedge, i.e. has no face, or if its face
            is not a quad.
        """

        fkey = self.halfedge[u][v]
        if fkey is None or not self.is_strip_face(fkey):
            return None
        w = self.face_vertex_descendant(fkey, v)
        x = self.face_vertex_descendant(fkey, w)
        return (w, x)

    def vertex_opposite_vertex(self, u: int, v: int, strict: bool = False) -> int | None:
        """Returns the opposite vertex to u accross vertex v.

        Parameters
        ----------
        u : hashable
            A vertex key.
        v : hashable
            A vertex key.
        strict : bool, optional
            Decide the crossing on edges rather than vertices. Default is False,
            the historical behaviour; switching it changes decompositions.

        Returns
        -------
        hashable, None
            The opposite vertex.
            None if v is a singularity or if (u, v) leads outwards.

        """

        if strict:
            nbrs = self.vertex_neighbors(v, ordered=True)
            n = len(nbrs)

            # regular interior vertex: cross straight over
            if n == 4 and not self.is_vertex_on_boundary(v):
                return nbrs[nbrs.index(u) - 2]

            # regular boundary vertex reached along the boundary: follow the boundary
            if n == 3 and self.is_edge_on_boundary(u, v):
                for nbr in nbrs:
                    if nbr != u and self.is_edge_on_boundary(v, nbr):
                        return nbr

            return None

        if self.is_vertex_singular(v):
            return None

        elif self.is_vertex_on_boundary(v):
            if not self.is_vertex_on_boundary(u):
                return None

            else:
                return [nbr for nbr in self.vertex_neighbors(v) if nbr != u and self.is_vertex_on_boundary(nbr)][0]

        else:
            nbrs = self.vertex_neighbors(v, ordered=True)
            return nbrs[nbrs.index(u) - 2]

    # --------------------------------------------------------------------------
    # singularities
    # --------------------------------------------------------------------------

    def is_vertex_singular(self, vkey: int) -> bool:
        """Output whether a vertex is quad mesh singularity.

        Parameters
        ----------
        vkey : int
            The vertex key.

        Returns
        -------
        bool
            True if the vertex is a quad mesh singularity. False otherwise.

        """

        if (self.is_vertex_on_boundary(vkey) and self.vertex_degree(vkey) != 3) or (not self.is_vertex_on_boundary(vkey) and self.vertex_degree(vkey) != 4):
            return True

        else:
            return False

    def singularities(self) -> list[int]:
        """Returns all the singularity indices in the quad mesh.

        Returns
        -------
        list
            The list of vertex indices that are quad mesh singularities.

        """
        return [vkey for vkey in self.vertices() if self.is_vertex_singular(vkey)]

    def vertex_topo_index(self, vkey: int) -> float:
        """Compute vertex index.

        Parameters
        ----------
        vkey : int
            The vertex key.

        Returns
        -------
        int
            Vertex index.

        """

        if self.vertex_degree(vkey) == 0:
            return 0

        regular_valency = 4 if not self.is_vertex_on_boundary(vkey) else 3

        return (regular_valency - self.vertex_degree(vkey)) / 4

    # --------------------------------------------------------------------------
    # polyedges
    # --------------------------------------------------------------------------

    def collect_polyedge(self, u0: int, v0: int, both_sides: bool = True, oriented: bool = False, strict: bool = False) -> list[int]:
        """Collect all the edges in the polyedge of the input edge.

        Parameters
        ----------
        u0 : int
            The identifier of the edge start.
        v0 : int
            The identifier of the edge end.
        both_sides : bool, optional
            Whether to walk in both directions from the seed halfedge. Default is
            True. With False the walk stops at the first extremity, which is what
            a directional tracer wants.
        oriented : bool, optional
            Whether to return the polyedge in the direction of the seed halfedge.
            Default is False, which returns the reversed polyedge whenever the walk
            reaches an extremity on the first side, i.e. the seed direction is lost.
        strict : bool, optional
            Passed to ``vertex_opposite_vertex``. Default is False.

        Returns
        -------
        polyedge : list
            The list of the vertices in polyedge.
        """

        flipped = False
        polyedge = [u0, v0]

        while len(polyedge) <= self.number_of_vertices():
            # end if closed loop
            if polyedge[0] == polyedge[-1]:
                break

            # get next vertex accros four-valent vertex
            w = self.vertex_opposite_vertex(*polyedge[-2:], strict=strict)

            # flip if end of first extremity
            if w is None:
                if not both_sides:
                    break
                polyedge = list(reversed(polyedge))
                flipped = True
                # stop if end of second extremity
                w = self.vertex_opposite_vertex(*polyedge[-2:], strict=strict)
                if w is None:
                    break

            # add next vertex
            polyedge.append(w)

        if oriented and flipped:
            polyedge = list(reversed(polyedge))

        return polyedge

    def collect_polyedges(self, strict: bool = False) -> Iterator[tuple[int, list[int]]]:
        """Collect the polyedges accross four-valent vertices between boundaries and/or singularities and store it in the mesh data attributes.

        Parameters
        ----------
        strict : bool, optional
            Passed to ``vertex_opposite_vertex``. Default is False.

        Returns
        -------
        polyedges : list
            List of quad polyedges as list of vertices.

        """

        # the list fixes the seed order (deterministic, so polyedge keys are stable
        # across runs); the set is only for O(1) membership and removal.
        edges = list(self.edges())
        remaining = set(edges)

        nb_polyedges = -1
        for u0, v0 in reversed(edges):
            if (u0, v0) not in remaining:
                # already consumed by an earlier polyedge
                continue

            nb_polyedges += 1

            # collect new polyedge
            polyedge = self.collect_polyedge(u0, v0, strict=strict)
            self.attributes["polyedges"].update({nb_polyedges: polyedge})

            # remove collected edges
            for u, v in pairwise(polyedge):
                remaining.discard((u, v))
                remaining.discard((v, u))

        return self.polyedges(data=True)

    def is_polyedge_closed(self, pkey: int) -> bool:
        """Output whether a polyedge is closed.

        Parameters
        ----------
        pkey : hashable
            A strip key.

        Returns
        -------
        bool
            True if the polyedge is closed. False otherwise.
        """

        return self.attributes["polyedges"][pkey][0] == self.attributes["polyedges"][pkey][-1]

    def number_of_polyedges(self) -> int:
        """Count the number of polyedges in the mesh."""
        return len(list(self.polyedges()))

    def polyedge_vertices(self, pkey: int) -> list[int]:
        """Return the vertices of a polyedge.

        Parameters
        ----------
        pkey : hashable
            A polyedge key.

        Returns
        -------
        list
            The vertices of the polyedge.
        """

        return self.attributes["polyedges"][pkey]

    def polyedge_edges(self, pkey: int) -> list[tuple[int, int]]:
        """Return the edges of a polyedge.

        Parameters
        ----------
        pkey : hashable
            A polyedge key.

        Returns
        -------
        list
            The edges of the polyedge, as pairs of vertex keys.
        """

        return list(pairwise(self.polyedge_vertices(pkey)))

    def polyedge_midpoint(self, pkey: int) -> Point:
        """Return the point at mid-length of a polyedge.

        Parameters
        ----------
        pkey : hashable
            A polyedge key.

        Returns
        -------
        Point
            The midpoint.
        """

        return Polyline(self.polyline(pkey)).point_at(0.5)

    def polyedge_length(self, pkey: int) -> float:
        """Return the length of a polyedge.

        Parameters
        ----------
        pkey : hashable
            A polyedge key.

        Returns
        -------
        float
            The sum of the lengths of the edges of the polyedge.
        """

        return sum([self.edge_length(u, v) for u, v in self.polyedge_edges(pkey)])

    def singularity_polyedges(self) -> list[list[int]]:
        """Collect the polyedges connected to singularities.

        Returns
        -------
        list
            The polyedges connected to singularities.

        """

        # keep only polyedges connected to singularities or along the boundary
        polyedges = [
            polyedge
            for key, polyedge in self.polyedges(data=True)
            if self.is_vertex_singular(polyedge[0]) or self.is_vertex_singular(polyedge[-1]) or self.is_edge_on_boundary(polyedge[0], polyedge[1])
        ]

        # get intersections between polyedges for split
        vertices = [vkey for polyedge in polyedges for vkey in set(polyedge)]
        split_vertices = [vkey for vkey in self.vertices() if vertices.count(vkey) > 1]

        # split singularity polyedges
        return [split_polyedge for polyedge in polyedges for split_polyedge in list_split(polyedge, [polyedge.index(vkey) for vkey in split_vertices if vkey in polyedge])]

    def singularity_polyedge_decomposition(self, strict: bool = False) -> list[list[int]]:
        """Returns a quad patch decomposition of the mesh based on the singularity polyedges, including boundaries and additionnal splits on the boundaries.

        Parameters
        ----------
        strict : bool, optional
            Passed to ``vertex_opposite_vertex``. Default is False.

        Returns
        -------
        list
            The polyedges forming the decomposition.

        """
        if self.attributes["polyedges"] == {}:
            self.collect_polyedges(strict=strict)

        polyedges = [
            polyedge
            for key, polyedge in self.polyedges(data=True)
            if (self.is_vertex_singular(polyedge[0]) or self.is_vertex_singular(polyedge[-1])) and not self.is_edge_on_boundary(polyedge[0], polyedge[1])
        ]

        # split boundaries
        all_splits = list(set([vkey for polyedge in polyedges for vkey in polyedge] + self.singularities()))

        for boundary in self.boundaries():
            splits = [vkey for vkey in boundary if vkey in all_splits]
            new_splits = []

            if len(splits) == 0:
                new_splits += [vkey for vkey in list(itemgetter(0, int(floor(len(boundary) / 3)), int(floor(len(boundary) * 2 / 3)))(boundary))]

            elif len(splits) == 1:
                i = boundary.index(splits[0])
                new_splits += list(itemgetter(i - int(floor(len(boundary) * 2 / 3)), i - int(floor(len(boundary) / 3)))(boundary))

            elif len(splits) == 2:
                one, two = list_split(boundary + boundary[:1], [boundary.index(vkey) for vkey in splits])
                half = one if len(one) > len(two) else two
                new_splits.append(half[int(floor(len(half) / 2))])

            for vkey in new_splits:
                for nbr in self.vertex_neighbors(vkey):
                    if not self.is_edge_on_boundary(vkey, nbr):
                        new_polyedge = self.collect_polyedge(vkey, nbr, strict=strict)
                        polyedges.append(new_polyedge)
                        all_splits = list(set(all_splits + new_polyedge))
                        break

        # add boundaries
        polyedges += [polyedge for key, polyedge in self.polyedges(data=True) if self.is_edge_on_boundary(polyedge[0], polyedge[1])]

        # get intersections between polyedges for split
        vertices = [vkey for polyedge in polyedges for vkey in set(polyedge)]
        split_vertices = [vkey for vkey in self.vertices() if vertices.count(vkey) > 1]

        # split singularity polyedges
        return [split_polyedge for polyedge in polyedges for split_polyedge in list_split(polyedge, [polyedge.index(vkey) for vkey in split_vertices if vkey in polyedge])]

    # --------------------------------------------------------------------------
    # polylines
    # --------------------------------------------------------------------------

    def polyedge_graph(self, legacy: bool = True) -> tuple[dict[int, list[float]], list[tuple[int, int]]]:
        """Compute the vertices and edges of the graph representing the polyedge connectivity,
        where each graph vertex is a mesh polyedge and each graph edge a non-compas_singular mesh vertex representing the crossing of two polyedges.
        Polyedges connected by their extremities, which are singularities, do not count as overlapping.

        Parameters
        ----------
        legacy : bool, optional
            Use the historical implementation, about half of whose edges are
            self-loops. Default is True; ``False`` gives one edge per crossing.

        Returns
        -------
        tuple
            A tuple of two objects, the dictionary of mesh polyedge indices pointing to their centroid coordinates, and the list of edges between graph vertices.
        """

        if legacy:
            vertices = {key: centroid_points([self.vertex_coordinates(vkey) for vkey in polyedge]) for key, polyedge in self.polyedges(data=True)}
            edges = []
            for key, polyedge in self.polyedges(data=True):
                for vkey in polyedge:
                    if not self.is_vertex_singular(vkey):
                        for key_2, polyedge_2 in self.polyedges(data=True):
                            if vkey in polyedge_2:
                                edges.append((key, key_2))
                                break
            return vertices, edges

        vertices = {pkey: centroid_points(self.polyline(pkey)) for pkey in self.polyedges()}

        vkey_to_pkeys = {vkey: set() for vkey in self.vertices()}
        for pkey, polyedge in self.polyedges(data=True):
            for vkey in polyedge:
                if not self.is_vertex_singular(vkey):
                    vkey_to_pkeys[vkey].add(pkey)

        # a vertex that two polyedges pass through is a crossing; anything else
        # (a singularity, an extremity) is not an adjacency and is skipped
        edges = [tuple(pkeys) for pkeys in vkey_to_pkeys.values() if len(pkeys) == 2]

        return vertices, edges

    # --------------------------------------------------------------------------
    # polylines
    # --------------------------------------------------------------------------

    def polyline(self, pkey: int) -> list[list[float]]:
        """Return the coordinates of the vertices of a polyedge.

        Parameters
        ----------
        pkey : hashable
            A polyedge key.

        Returns
        -------
        list
            The polyline as a list of XYZ points.
        """

        return [self.vertex_coordinates(vkey) for vkey in self.polyedge_vertices(pkey)]

    def polylines(self) -> list[list[list[float]]]:
        """Return the polylines of the quad mesh.

        Returns
        -------
        list
            The polylines.
        """

        return [[self.vertex_coordinates(vkey) for vkey in polyedge] for key, polyedge in self.polyedges(data=True)]

    def singularity_polylines(self) -> list[list[list[float]]]:
        """Return the polylines connected to singularities.

        Returns
        -------
        list
            The polylines connected to singularities.

        """
        return [[self.vertex_coordinates(vkey) for vkey in polyedge] for polyedge in self.singularity_polyedges()]

    def singularity_polyline_decomposition(self) -> list[list[list[float]]]:
        """Return the polylines forming a quad patch decomposition of the mesh.

        Returns
        -------
        list
            The polylines connected to singularities.

        """
        return [[self.vertex_coordinates(vkey) for vkey in polyedge] for polyedge in self.singularity_polyedge_decomposition()]

    # --------------------------------------------------------------------------
    # strips
    # --------------------------------------------------------------------------

    def number_of_strips(self) -> int:
        """Count the number of strips in the mesh."""
        return len(list(self.strips()))

    def collect_strip(self, u0: int, v0: int, both_sides: bool = True) -> list[tuple[int, int]]:
        """Returns all the edges in the strip of the input edge; a non-quad face ends it like the boundary.

        Parameters
        ----------
        u0 : int
            The identifier of the edge start.
        v0 : int
            The identifier of the edge end.
        both_sides : bool, optional
            Whether to walk in both directions from the seed halfedge. Default is
            True. With False the walk stops at the first extremity.

        Returns
        -------
        strip : list
            The list of the edges in strip.
        """

        if not self._crosses(u0, v0):
            if not both_sides:
                return [(u0, v0)]
            u0, v0 = v0, u0

        edges = [(u0, v0)]

        count = self.number_of_edges()
        while count > 0:
            count -= 1

            u, v = edges[-1]
            opposite = self.face_opposite_edge(u, v)
            if opposite is None:
                break
            w, x = opposite

            if (x, w) == edges[0]:
                break

            edges.append((x, w))

            if not self._crosses(x, w):
                if not both_sides:
                    break
                edges = [(v, u) for u, v in reversed(edges)]
                u, v = edges[-1]
                if not self._crosses(u, v):
                    break

        return edges

    def _crosses(self, u: int, v: int) -> bool:
        """Whether a strip walk can continue across the halfedge ``(u, v)``."""
        if v not in self.halfedge[u]:
            return False
        fkey = self.halfedge[u][v]
        return fkey is not None and self.is_strip_face(fkey)

    def collect_strips(self) -> Iterator[tuple[int, list[tuple[int, int]]]]:
        """Collect the strip data and store it in the mesh data attributes.

        Returns
        -------
        strips : dict
            The strip data.
        """

        # CLEAR FIRST. ``update`` only adds and overwrites, so re-collecting a mesh
        # that has LOST a strip left the old keys behind -- and worse, a key that was
        # deleted and is now re-assigned lands at the END of the dict's insertion
        # order. Measured on a 4x4 grid: delete strip 3, re-collect, and the order is
        # [0, 1, 2, 4, 5, 6, 7, 3] -- so ``list(strips())[-1]`` is 3 while the maximum
        # is 7, and naming a new strip ``last + 1`` would then pick 4,
        # which already exists and is silently overwritten. Stale entries are also
        # read as real by ``is_strip_closed``, which only ever looks at
        # ``strips[skey][0]``.
        self.attributes["strips"].clear()

        # see collect_polyedges: list for a stable seed order, set for O(1) removal
        edges = [(u, v) if self.halfedge[u][v] is not None else (v, u) for u, v in self.edges()]
        remaining = set(edges)

        nb_strip = -1
        for u0, v0 in reversed(edges):
            if (u0, v0) not in remaining:
                continue

            nb_strip += 1

            strip_edges = self.collect_strip(u0, v0)
            self.attributes["strips"].update({nb_strip: strip_edges})
            for u, v in strip_edges:
                remaining.discard((u, v))
                remaining.discard((v, u))

        return self.strips(data=True)

    def add_strip(self, polyedge: list[int], open_strip: bool = True, project: Callable[[list[float]], list[float]] | None = None) -> tuple[int, dict[int, tuple[int, int]]]:
        """Add a strip along ``polyedge``. ``(new strip key, {old vertex: pair})``.

        See ``compas_singular.datastructures.mesh_quad.grammar.add_strip``.
        """
        return add_strip(self, polyedge, open_strip=open_strip, project=project)

    def add_strips(self, polyedges: list[list[int]], open_strip: bool = True, project: Callable[[list[float]], list[float]] | None = None) -> list[int]:
        """Add a strip along each polyedge. The new strip keys.

        ``open_strip`` and ``project`` as in ``add_strip``.
        """
        return add_strips(self, polyedges, open_strip=open_strip, project=project)

    def delete_strip(self, skey: int) -> dict[int, int]:
        """Delete the strip ``skey``, welding its sides. ``{old vertex: the vertex it merged into}``.

        See ``compas_singular.datastructures.mesh_quad.grammar.delete_strip``.
        """
        return delete_strip(self, skey)

    def delete_strips(self, skeys: list[int]) -> None:
        """Delete several strips. See ``delete_strip``."""
        return delete_strips(self, skeys)

    def is_strip_closed(self, skey: int) -> bool:
        """Output whether a strip is closed.

        Parameters
        ----------
        skey : hashable
            A strip key.

        Returns
        -------
        bool
            True if the strip is closed. False otherwise.
        """

        return not self.is_edge_on_boundary(*self.strip_edges(skey)[0])

    def strip_edges(self, skey: int) -> list[tuple[int, int]]:
        """Return the edges of a strip.

        Parameters
        ----------
        skey : hashable
            A strip key.
        Returns
        -------
        list
            The edges of the strip.

        """

        return self.attributes["strips"][skey]

    def edge_strip(self, edge: tuple[int, int]) -> int | None:
        """Return the strip of an edge.

        Parameters
        ----------
        edge : tuple
            An edge as two vertex keys.

        Returns
        -------
        strip
            The strip of the edge.
        """

        for skey, edges in self.strips(data=True):
            if edge in edges or tuple(reversed(edge)) in edges:
                return skey

    def strip_faces(self, skey: int) -> list[int]:
        """Return the faces of a strip.

        Parameters
        ----------
        skey : hashable
            A strip key.

        Returns
        -------
        list
            The faces of the strip.

        """

        # ``is_strip_face``: the last edge of a strip that ENDS at a polygon has that
        # polygon ahead of it, and it is not one of the strip's faces.
        return [self.halfedge[u][v] for u, v in self.strip_edges(skey) if self.halfedge[u][v] is not None and self.is_strip_face(self.halfedge[u][v])]

    def face_strips(self, fkey: int) -> list[int | None]:
        """Return the two strips of a face.

        Parameters
        ----------
        fkey : hashable

        Returns
        -------
        list
            The two strips of the face.
        """

        return [self.edge_strip((u, v)) for u, v in list(self.face_halfedges(fkey))[:2]]

    # --------------------------------------------------------------------------
    # strip data operations
    # --------------------------------------------------------------------------

    def substitute_vertex_in_strips(self, old_vkey: int, new_vkey: int, strips: list[int] | None = None) -> None:
        """Substitute a vertex by another one.

        Parameters
        ----------
        old_vkey : hashable
            The old vertex key.
        new_vkey : hashable
            The new vertex key.
        strips : list
            List of specific strip keys. Per default None, i.e. all.

        """

        if strips is None:
            strips = list(self.strips())
        self.attributes["strips"].update({skey: [tuple([new_vkey if vkey == old_vkey else vkey for vkey in list(edge)]) for edge in self.strip_edges(skey)] for skey in strips})

    def delete_face_in_strips(self, fkey: int) -> None:
        """Delete face in strips.

        Parameters
        ----------
        fkey : int
            The face to remove from every strip.

        """

        self.attributes["strips"] = {skey: [(u, v) for u, v in self.strip_edges(skey) if self.halfedge[u][v] != fkey] for skey in self.strips()}

    # --------------------------------------------------------------------------
    # strip graph
    # --------------------------------------------------------------------------

    def strip_graph(self) -> tuple[dict[int, list[float]], list[tuple[int | None, int | None]]]:
        """Compute the vertices and edges of the graph representing the strip connectivity,
        where each graph vertex is a mesh strip and each graph edge a mesh face representing the crossing of two strips.
        Potentially includes loop edges (u, u) or multiple arallel edges (u, v) and/or (v, u).

        Returns
        -------
        tuple
            A tuple of two objects, the dictionary of mesh strip keys pointing to graph vertex coordinates,
            and the list of edges between graph vertices.
        """

        vertices = {
            skey: centroid_points(self.strip_edge_midpoint_polyline(skey) if not self.is_strip_closed(skey) else self.strip_edge_midpoint_polyline(skey)[:-1])
            for skey in self.strips()
        }
        edges = [tuple(self.face_strips(fkey)) for fkey in self.faces()]
        return vertices, edges

    # --------------------------------------------------------------------------
    # strip polyedges
    # --------------------------------------------------------------------------

    def strip_side_polyedges(self, skey: int) -> tuple[list[int], list[int]]:
        """Return the two side polyedges of a strip.

        Parameters
        ----------
        skey : hashable
            A strip key.

        Returns
        -------
        tuple
            The pair of polyedges on the side of the strip.
        """

        strip_edges = self.strip_edges(skey)

        starts = [edge[0] for edge in strip_edges]
        ends = [edge[1] for edge in strip_edges]

        if self.is_strip_closed(skey):
            starts += starts[:1]
            ends += ends[:1]

        return (starts, ends)

    # --------------------------------------------------------------------------
    # strip polylines
    # --------------------------------------------------------------------------

    def strip_edge_midpoint_polyline(self, skey: int) -> list[Point]:
        """Return the strip polyline connecting edge midpoints.

        Parameters
        ----------
        skey : hashable
            A strip key.

        Returns
        -------
        list
            The edge midpoint polyline.
        """

        polyline = [self.edge_midpoint(u, v) for u, v in self.strip_edges(skey)]

        if self.is_strip_closed(skey):
            return polyline + polyline[:1]

        else:
            return polyline

    def strip_face_centroid_polyline(self, skey: int) -> list[Point]:
        """Return the strip polyline connecting face centroids.

        Parameters
        ----------
        skey : hashable
            A strip key.

        Returns
        -------
        list
            The face centroid polyline.
        """

        polyline = [self.face_centroid(fkey) for fkey in self.strip_faces(skey)]

        if self.is_strip_closed(skey):
            return polyline + polyline[:1]

        else:
            return polyline

    def strip_side_polylines(self, skey: int) -> tuple[list[list[float]], list[list[float]]]:
        """Return the two side polylines of a strip.

        Parameters
        ----------
        skey : hashable
            A strip key.

        Returns
        -------
        tuple
            The pair of polylines on the side of the strip.
        """

        starts, ends = self.strip_side_polyedges(skey)
        return ([self.vertex_coordinates(vkey) for vkey in starts], [self.vertex_coordinates(vkey) for vkey in ends])


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass

    # import compas
    # from compas_plotters.meshplotter import MeshPlotter

    # # mesh = QuadMesh.from_obj(compas.get('faces.obj'))
    # # mesh = QuadMesh.from_json('/Users/Robin/Desktop/json/debug.json')

    # # mesh.collect_strips()
    # # mesh.collect_polyedges()

    # # print(mesh.singularities())
    # # print(len(list(mesh.strips())))
    # # print(len(list(mesh.polyedges())))

    # # print(len(mesh.singularity_polyedge_decomposition()))

    # # print(mesh.strip_graph())
    # # print(mesh.polyedge_graph())

    # #plotter = MeshPlotter(mesh, figsize=(20, 20))
    # #plotter.draw_vertices(radius=0.4, text='key')
    # # plotter.draw_edges()
    # # plotter.draw_faces()
    # # plotter.show()
