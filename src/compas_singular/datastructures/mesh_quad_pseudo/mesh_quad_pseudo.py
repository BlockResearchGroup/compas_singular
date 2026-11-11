from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any
from typing import Iterator

from compas.tolerance import TOL
from compas_singular.datastructures.mesh_quad import QuadMesh
from compas_singular.utilities import list_split

__all__ = ["PseudoQuadMesh"]


class PseudoQuadMesh(QuadMesh):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super(PseudoQuadMesh, self).__init__(*args, **kwargs)
        self.attributes["face_pole"] = {}

    @classmethod
    def __from_data__(cls, data: dict[str, Any]) -> "PseudoQuadMesh":
        # COMPAS 2.x serialises ``attributes`` natively, but JSON object keys
        # are always strings. Restore the integer face keys of the pole map.
        mesh = super(PseudoQuadMesh, cls).__from_data__(data)
        face_pole = mesh.attributes.get("face_pole") or {}
        mesh.attributes["face_pole"] = {int(fkey): vkey for fkey, vkey in face_pole.items()}
        return mesh

    @classmethod
    def from_vertices_and_faces_with_poles(
        cls, vertices: "dict[int, list[float]] | list[list[float]]", faces: "dict[int, list[int]] | list[list[int]]", poles: list[list[float]] = []
    ) -> "PseudoQuadMesh":
        pole_map = tuple([TOL.geometric_key(pole) for pole in poles])
        mesh = cls.from_vertices_and_faces(vertices, faces)
        for fkey in mesh.faces():
            face_vertices = mesh.face_vertices(fkey)
            if len(face_vertices) == 3:
                mesh.attributes["face_pole"][fkey] = face_vertices[0]
                for vkey in face_vertices:
                    if TOL.geometric_key(mesh.vertex_coordinates(vkey)) in pole_map:
                        mesh.attributes["face_pole"].update({fkey: vkey})
                        break
        return mesh

    @classmethod
    def from_vertices_and_faces_with_face_poles(
        cls, vertices: "dict[int, list[float]] | list[list[float]]", faces: "dict[int, list[int]] | list[list[int]]", face_poles: dict[int, int] = {}
    ) -> "PseudoQuadMesh":
        mesh = cls.from_vertices_and_faces(vertices, faces)
        mesh.attributes["face_pole"] = face_poles
        return mesh

    def poles(self) -> list[int]:
        return list(set(self.attributes["face_pole"].values()))

    def is_pole(self, vkey: int) -> bool:
        return vkey in set(self.poles())

    def is_face_pseudo_quad(self, fkey: int) -> bool:
        return fkey in set(self.attributes["face_pole"].keys())

    def is_vertex_pole(self, vkey: int) -> bool:
        return vkey in set(self.attributes["face_pole"].values())

    def is_vertex_full_pole(self, vkey: int) -> bool:
        return all([self.is_face_pseudo_quad(fkey) for fkey in self.vertex_faces(vkey)])

    def is_vertex_partial_pole(self, vkey: int) -> bool:
        return self.is_vertex_pole(vkey) and not self.is_vertex_full_pole(vkey)

    def vertex_pole_faces(self, vkey: int) -> list[int]:
        return [fkey for fkey, pole in self.attributes["face_pole"].items() if pole == vkey]

    def is_strip_face(self, fkey: int) -> bool:
        """A quad, or a triangle registered as a pseudo-quad in ``face_pole``."""
        degree = len(self.face_vertices(fkey))
        return degree == 4 or (degree == 3 and fkey in self.attributes["face_pole"])

    def face_opposite_edge(self, u: int, v: int) -> tuple[int, int] | None:
        """Returns the opposite edge in the quad face.

        Parameters
        ----------
        u : int
            The identifier of the edge start.
        v : int
            The identifier of the edge end.

        Returns
        -------
        (w, x) : tuple
            The opposite edge. None on the boundary, and for a face a strip
            cannot cross -- see ``is_strip_face``.

        """

        fkey = self.halfedge[u][v]
        if fkey is None or not self.is_strip_face(fkey):
            return None
        # if quad
        if len(self.face_vertices(fkey)) == 4:
            w = self.face_vertex_descendant(fkey, v)
            x = self.face_vertex_descendant(fkey, w)
            return (w, x)
        # if pseudo quad
        if len(self.face_vertices(fkey)) == 3:
            pole = self.attributes["face_pole"][fkey]
            w = self.face_vertex_descendant(fkey, v)
            if u == pole:
                return (w, u)
            if v == pole:
                return (v, w)
            else:
                return (pole, pole)

    def collect_strip(self, u0: int, v0: int) -> list[tuple[int, int]]:
        """Returns all the edges in the strip of the input edge.

        Parameters
        ----------
        u0 : int
            The identifier of the edge start.
        v0 : int
            The identifier of the edge end.

        Returns
        -------
        strip : list
            The list of the edges in strip.
        """

        if not self._crosses(u0, v0):
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

            if w == x or not self._crosses(x, w):
                edges = [(v, u) for u, v in reversed(edges)]
                u, v = edges[-1]
                if u == v or not self._crosses(u, v):
                    break

        return edges

    def collect_strips(self) -> Iterator[tuple[int, list[tuple[int, int]]]]:
        """Collect the strip data.

        Returns
        -------
        strip : int
            The number of strips.

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

        # see QuadMesh.collect_polyedges: list for a stable seed order, set for O(1) removal
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
                if u != v:
                    remaining.discard((u, v))
                    remaining.discard((v, u))

        return self.strips(data=True)

    def has_strip_poles(self, skey: int) -> bool:
        return self.attributes["strips"][skey][0][0] == self.attributes["strips"][skey][0][1] or self.attributes["strips"][skey][-1][0] == self.attributes["strips"][skey][-1][1]

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

        return not self.has_strip_poles(skey) and not self.is_edge_on_boundary(*self.attributes["strips"][skey][0])

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

        if self.is_vertex_pole(vkey):
            return True
        elif (self.is_vertex_on_boundary(vkey) and self.vertex_degree(vkey) != 3) or (not self.is_vertex_on_boundary(vkey) and self.vertex_degree(vkey) != 4):
            return True

        else:
            return False

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

        if self.is_vertex_pole(vkey):
            if self.is_vertex_full_pole(vkey):
                if self.is_vertex_on_boundary(vkey):
                    return 1.0 / 2.0
                else:
                    return 1.0
            else:
                adapted_valency = sum([not self.is_face_pseudo_quad(fkey) for fkey in self.vertex_faces(vkey)])
                if self.is_vertex_on_boundary(vkey):
                    adapted_valency += 1
                regular_valency = 4.0 if not self.is_vertex_on_boundary(vkey) else 3.0
                return (regular_valency - adapted_valency) / 4.0
        else:
            regular_valency = 4.0 if not self.is_vertex_on_boundary(vkey) else 3.0
            return (regular_valency - self.vertex_degree(vkey)) / 4.0

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

        faces = []
        edges = self.strip_edges(skey)
        for i, (u, v) in enumerate(edges):
            if i == 0 and u == v:
                x, w = edges[1]
                faces.append(self.halfedge[w][x])
            elif i == len(edges) - 1 and u == v:
                pass
            else:
                if self.halfedge[u][v] is not None and self.is_strip_face(self.halfedge[u][v]):
                    faces.append(self.halfedge[u][v])
        return faces

    def face_strips(self, fkey: int) -> list[int]:
        """Return the two strips of a face.

        Parameters
        ----------
        fkey : hashable

        Returns
        -------
        list
            The two strips of the face.
        """

        if self.is_face_pseudo_quad(fkey):
            pole = self.attributes["face_pole"][fkey]
            # print(pole, fkey, self.face_vertices(fkey))
            u = self.face_vertex_descendant(fkey, pole)
            v = self.face_vertex_descendant(fkey, u)
            return [self.edge_strip((pole, u)), self.edge_strip((u, v))]
        else:
            return [self.edge_strip((u, v)) for u, v in list(self.face_halfedges(fkey))[:2]]

    def delete_face_in_strips(self, fkey: int) -> None:
        """Delete face in strips.

        Parameters
        ----------
        fkey : int
            The face to remove from every strip.

        """

        self.attributes["strips"] = {
            skey: [(u, v) for u, v in self.strip_edges(skey) if u == v or (self.halfedge[u][v] != fkey and self.halfedge[v][u] != fkey)] for skey in self.strips()
        }

    def singularity_polyedges(self) -> list[list[int]]:
        """Collect the polyedges connected to singularities.

        Returns
        -------
        list
            The polyedges connected to singularities.

        """

        # poles = set(self.poles())
        # keep only polyedges connected to singularities or along the boundary
        polyedges = [
            polyedge
            for key, polyedge in self.polyedges(data=True)
            if (self.is_vertex_singular(polyedge[0]) and not self.is_pole(polyedge[0]))
            or (self.is_vertex_singular(polyedge[-1]) and not self.is_pole(polyedge[-1]))
            or self.is_edge_on_boundary(polyedge[0], polyedge[1])
        ]

        # get intersections between polyedges for split
        vertices = [vkey for polyedge in polyedges for vkey in set(polyedge)]
        split_vertices = [vkey for vkey in self.vertices() if vertices.count(vkey) > 1]

        # split singularity polyedges
        return [split_polyedge for polyedge in polyedges for split_polyedge in list_split(polyedge, [polyedge.index(vkey) for vkey in split_vertices if vkey in polyedge])]

    # def add_face(self, vertices, fkey=None, attr_dict=None, **kwattr):
    #     """Add a face to the mesh object. Allow [a, b, c, c] faces.

    #     Parameters
    #     ----------
    #     vertices : list
    #         A list of vertex keys.
    #         For every vertex that does not yet exist, a new vertex is created.
    #     attr_dict : dict, optional
    #         Face attributes.
    #     kwattr : dict, optional
    #         Additional named face attributes.
    #         Named face attributes overwrite corresponding attributes in the
    #         attribute dict (``attr_dict``).

    #     Returns
    #     -------
    #     int
    #         The key of the face.
    #         The key is an integer, if no key was provided.
    #     hashable
    #         The key of the face.
    #         Any hashable object may be provided as identifier for the face.
    #         Provided keys are returned unchanged.

    #     Raises
    #     ------
    #     TypeError
    #         If the provided face key is of an unhashable type.

    #     Notes
    #     -----
    #     If no key is provided for the face, one is generated
    #     automatically. An automatically generated key is an integer that increments
    #     the highest integer value of any key used so far by 1.

    #     If a key with an integer value is provided that is higher than the current
    #     highest integer key value, then the highest integer value is updated accordingly.

    #     See Also
    #     --------
    #     * ``add_vertex``
    #     * ``add_edge``

    #     Examples
    #     --------
    #     >>>

    #     """
    #     attr = self._compile_fattr(attr_dict, kwattr)

    #     # remove clean vertices to allow [a, b, c, c] faces
    #     #self._clean_vertices(vertices)

    #     if len(vertices) < 3:
    #         return

    #     keys = []
    #     for key in vertices:
    #         if key not in self.vertex:
    #             key = self.add_vertex(key)
    #         keys.append(key)

    #     fkey = self._get_face_key(fkey)

    #     self.face[fkey] = keys
    #     self.facedata[fkey] = attr

    #     for u, v in self._cycle_keys(keys):
    #         self.halfedge[u][v] = fkey
    #         if u not in self.halfedge[v]:
    #             self.halfedge[v][u] = None

    #     return fkey

    # def delete_face(self, fkey):
    #     """Delete a face from the mesh object. Valid for [a, b, c, c] faces.

    #     Parameters
    #     ----------
    #     fkey : hashable
    #         The identifier of the face.

    #     Examples
    #     --------
    #     .. plot::
    #         :include-source:

    #         import compas
    #         from compas.datastructures import Mesh
    #         from compas.plotters import MeshPlotter

    #         mesh = Mesh.from_obj(compas.get('faces.obj'))

    #         mesh.delete_face(12)

    #         plotter = MeshPlotter(mesh)
    #         plotter.draw_vertices()
    #         plotter.draw_faces()
    #         plotter.show()

    #     """
    #     def check_validity(self):
    #         for u, v in self.edges():
    #             if v not in self.halfedge[u] or u not in self.halfedge[v]:
    #                 return (u, v)
    #     #if check_validity is not None:
    #     #    print('!!!')
    #     for u, v in self.face_halfedges(fkey):
    #         self.halfedge[u][v] = None
    #         if self.halfedge[v][u] is None:
    #             del self.halfedge[u][v]
    #             del self.halfedge[v][u]
    #     del self.face[fkey]

    # def delete_vertex(self, key):
    #     """Delete a vertex from the mesh and everything that is attached to it.

    #     Parameters
    #     ----------
    #     key : hashable
    #         The identifier of the vertex.

    #     Examples
    #     --------
    #     .. plot::
    #         :include-source:

    #         import compas
    #         from compas.datastructures import Mesh
    #         from compas_plotters import MeshPlotter

    #         mesh = Mesh.from_obj(compas.get('faces.obj'))

    #         mesh.delete_vertex(17)

    #         color = {key: '#ff0000' for key in mesh.vertices() if mesh.vertex_degree(key) == 2}

    #         plotter = MeshPlotter(mesh)
    #         plotter.draw_vertices(facecolor=color)
    #         plotter.draw_faces()
    #         plotter.show()

    #     In some cases, disconnected vertices can remain after application of this
    #     method. To remove these vertices as well, combine this method with vertex
    #     culling (``cull_vertices``).

    #     .. plot::
    #         :include-source:

    #         import compas
    #         from compas.datastructures import Mesh
    #         from compas_plotters import MeshPlotter

    #         mesh = Mesh.from_obj(compas.get('faces.obj'))

    #         mesh.delete_vertex(17)
    #         mesh.delete_vertex(18)
    #         mesh.delete_vertex(0)
    #         mesh.cull_vertices()

    #         color = {key: '#ff0000' for key in mesh.vertices() if mesh.vertex_degree(key) == 2}

    #         plotter = MeshPlotter(mesh)
    #         plotter.draw_vertices(facecolor=color)
    #         plotter.draw_faces()
    #         plotter.show()

    #     """
    #     nbrs = self.vertex_neighbors(key)
    #     for nbr in nbrs:
    #         fkey = self.halfedge[key][nbr]
    #         if fkey is None:
    #             continue
    #         for u, v in self.face_halfedges(fkey):
    #             self.halfedge[u][v] = None
    #         del self.face[fkey]
    #     for nbr in nbrs:
    #         print(nbr, key)
    #         del self.halfedge[nbr][key]
    #     for nbr in nbrs:
    #         for n in self.vertex_neighbors(nbr):
    #             if self.halfedge[nbr][n] is None and self.halfedge[n][nbr] is None:
    #                 del self.halfedge[nbr][n]
    #                 del self.halfedge[n][nbr]
    #     del self.halfedge[key]
    #     del self.vertex[key]

    #     del self.face[fkey]


#     def edges(self, data=False):
#         """Iterate over the edges of the mesh.

#         Parameters
#         ----------
#         data : bool, optional
#             Return the edge data as well as the edge vertex keys.

#         Yields
#         ------
#         2-tuple
#             The next edge as a (u, v) tuple, if ``data`` is false.
#         3-tuple
#             The next edge as a (u, v, data) tuple, if ``data`` is true.

#         Note
#         ----
#         Mesh edges have no topological meaning. They are only used to store data.
#         Edges are not automatically created when vertices and faces are added to
#         the mesh. Instead, they are created when data is stored on them, or when
#         they are accessed using this method.

#         This method yields the directed edges of the mesh.
#         Unless edges were added explicitly using ``add_edge`` the order of
#         edges is *as they come out*. However, as long as the toplogy remains
#         unchanged, the order is consistent.

#         Example
#         -------
#         .. code-block:: python

#             import compas
#             from compas.datastructures import Mesh
#             from compas.plotters import MeshPlotter

#             mesh = Mesh.from_obj(compas.get('faces.obj'))

#             for index, (u, v, attr) in enumerate(mesh.edges(True)):
#                 attr['index1'] = index

#             for index, (u, v, attr) in enumerate(mesh.edges(True)):
#                 attr['index2'] = index

#             plotter = MeshPlotter(mesh)

#             text = {(u, v): '{}-{}'.format(a['index1'], a['index2']) for u, v, a in mesh.edges(True)}

#             plotter.draw_vertices()
#             plotter.draw_faces()
#             plotter.draw_edges(text=text)
#             plotter.show()

#         """
#         edges = set()

#         for u in self.halfedge:
#             for v in self.halfedge[u]:

#                 if (u, v) in edges or (v, u) in edges:
#                     continue

#                 edges.add((u, v))
#                 edges.add((v, u))

#                 if (u, v) not in self.edgedata:
#                     self.edgedata[u, v] = self.default_edge_attributes.copy()

#                     if u != v:
#                         if (v, u) in self.edgedata:
#                             self.edgedata[u, v].update(self.edgedata[v, u])
#                             del self.edgedata[v, u]

#                         self.edgedata[v, u] = self.edgedata[u, v]

#                 if data:
#                     yield u, v, self.edgedata[u, v]
#                 else:
#                     yield u, v

#     def to_mesh_2(self):
#         vertices = [self.vertex_coordinates(vkey) for vkey in self.vertices()]
#         face_vertices = []
#         # remove consecutive duplicates in pseudo quad faces
#         for fkey in self.faces():
#             non_pseudo_face = []
#             pseudo_face = self.face_vertices(fkey)
#             for i, vkey in enumerate(pseudo_face):
#                 if vkey != pseudo_face[i - 1]:
#                     non_pseudo_face.append(vkey)
#             face_vertices.append(non_pseudo_face)
#         mesh = Mesh.from_vertices_and_faces(vertices, face_vertices)
#         return mesh

#     def to_mesh(self):

#         vertices = [self.vertex_coordinates(vkey) for vkey in self.vertices()]
#         vertex_remap = list(self.vertices())
#         faces = []
#         for fkey in self.faces():
#             face_vertices = []
#             for vkey in self.face_vertices(fkey):
#                 vkey_idx = vertex_remap.index(vkey)
#                 if vkey_idx not in face_vertices:
#                     face_vertices.append(vkey_idx)
#             faces.append(face_vertices)

#         mesh = Mesh.from_vertices_and_faces(vertices, faces)
#         return mesh

# def pqm_from_mesh(mesh, poles):
#     """Converts a mesh into a pseuod-quad mesh with poles inducing face of the type [a, b, c, c].

#     Parameters
#     ----------
#     mesh : Mesh
#         A mesh.
#     poles: list
#         List of pole coordinates.

#     Returns
#     -------
#     vertices, new_face_vertices: list
#         The vertices with the new face_vertices.

#     Raises
#     ------
#     -

#     """

#     vertices = [mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()]
#     vertex_conversion = {vkey: i for i, vkey in enumerate(mesh.vertices())}
#     new_face_vertices = []

#     poles = [TOL.geometric_key(pole) for pole in poles]

#     for fkey in mesh.faces():
#         face_vertices = mesh.face_vertices(fkey)[:]
#         if len(face_vertices) == 3:
#             # find pole location
#             pole = None
#             for vkey in face_vertices:
#                 geom_key = TOL.geometric_key(mesh.vertex_coordinates(vkey))
#                 if geom_key in poles:
#                     pole = vkey
#                     break
#             # modify face
#             if pole is not None:
#                 idx = face_vertices.index(vkey)
#                 face_vertices.insert(idx, vkey)

#         # store new face
#         new_face_vertices.append([vertex_conversion[vkey] for vkey in face_vertices])

#     return vertices, new_face_vertices

# def is_face_pseudo_quad(mesh, fkey):

#     if fkey is None or len(mesh.face_vertices(fkey)) != 4:
#         return 'invalid'

#     face_vertices = mesh.face_vertices(fkey)
#     for vkey in face_vertices:
#         if face_vertices.count(vkey) > 1:
#             return True

#     return False

# def vertex_topo_index(mesh, vkey):
#     """Return the index of a vertex in a coarse quad mesh with potential poles stored in pseudo-quad faces.

#     Parameters
#     ----------
#     mesh : Mesh
#         A mesh.
#     vkey: int
#         Key of a vertex

#     Returns
#     -------
#     index: float
#         The index of the vertex.

#     Raises
#     ------
#     -

#     """

#     if not mesh.is_quadmesh():
#         return None
#     if len(mesh.vertex_neighbors(vkey)) == 0:
#         return None

#     valency = float(len(mesh.vertex_neighbors(vkey)))
#     boundary = mesh.is_vertex_on_boundary(vkey)
#     if vkey in mesh.vertex_neighbors(vkey):
#         pole = True
#     else:
#         pole = False
#     #pole = True if vkey in mesh.vertex_neighbors(vkey) else False
#     partial_pole = False
#     #print valency, mesh.vertex_neighbors(vkey), boundary, pole, partial_pole
#     #return 0
#     # if pole:
#     #     vertex_faces = mesh.vertex_faces(vkey)
#     #     for fkey in vertex_faces:
#     #         if fkey is not None and not is_face_pseudo_quad(mesh, fkey):
#     #             partial_pole = True
#     #             break
#     #     if partial_pole:
#     #         pseudo_valency = sum([1 - is_face_pseudo_quad(mesh, fkey) for fkey in vertex_faces])
#     #         if boundary:
#     #             pseudo_valency += 1

#     if pole:
#         if partial_pole:
#             if not boundary:
#                 #print 'partial pole', 1. / 4. * (4. - pseudo_valency)
#                 return 1. / 4. * (4. - pseudo_valency)
#             else:
#                 #print 'boundary partial pole', 1. / 4. * (3. - pseudo_valency)
#                 return 1. / 4. * (3. - pseudo_valency)
#         else:
#             if not boundary:
#                 #print 'pole', 1.
#                 return 1.
#             else:
#                 #print 'boundary pole', 1. / 2.
#                 return 1. / 2.
#     else:
#         if not boundary:
#             #print 'classic', 1. / 4. * (4. - valency)
#             return 1. / 4. * (4. - valency)
#         else:
#             #print 'boundary classic', 1. / 4. * (3. - valency)
#             return 1. / 4. * (3. - valency)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass
