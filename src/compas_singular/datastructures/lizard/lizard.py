from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import TYPE_CHECKING

from compas_singular.datastructures.mesh_quad.grammar.add_strip import add_strip
from compas_singular.datastructures.mesh_quad.grammar.delete_strip import delete_strip

if TYPE_CHECKING:
    from compas_singular.datastructures import QuadMesh

__all__ = ["Lizard"]


class Lizard:
    def __init__(self, quad_mesh: "QuadMesh") -> None:
        self.lizard = None
        self.grow = False
        self.mesh = quad_mesh

    def initiate(self, tail: int | None = None, head: int | None = None) -> None:
        if tail in self.mesh.vertex and head in self.mesh.vertex and tail in self.mesh.vertex[head]:
            self.lizard = [head, tail]
        else:
            tail = list(self.mesh.vertices())[0]
            head = self.mesh.vertex_neighbors(tail)[0]
            self.lizard = [head, tail]

    def turn(self) -> None:
        nbrs = self.mesh.vertex_neighbors(self.lizard[0], ordered=True)
        i = nbrs.index(self.lizard[1])
        # COMPAS 2.x orders vertex neighbors with the opposite winding to the
        # 0.x/1.x API this turtle was written for. Stepping to the previous
        # neighbor keeps the turtle turning in the original direction.
        new_head = nbrs[i - 1]
        self.lizard.insert(0, new_head)
        if not self.grow:
            del self.lizard[-1]

    def pivot(self) -> None:
        nbrs = self.mesh.vertex_neighbors(self.lizard[1], ordered=True)
        i = nbrs.index(self.lizard[0])
        new_head = nbrs[i - 1]  # opposite neighbor winding in COMPAS 2.x (see turn)
        self.lizard[0] = new_head

    def add(self) -> None:
        if self.grow:
            n, old_vkeys_to_new_vkeys = add_strip(self.mesh, self.lizard[1:])
            head = old_vkeys_to_new_vkeys[self.lizard[0]] if self.lizard[0] in old_vkeys_to_new_vkeys else self.lizard[0]
            tail = old_vkeys_to_new_vkeys[self.lizard[1]]
            self.lizard = [head, tail]

        self.grow = not self.grow

    def delete(self) -> None:
        if self.grow:
            # use update polyedge
            print("Not Implemented")
            pass

        skey = self.mesh.edge_strip((self.lizard[0], self.lizard[1]))
        self.pivot()  # check that pivot ends along strip
        old_vkeys_to_new_vkeys = delete_strip(self.mesh, skey)
        self.lizard = [old_vkeys_to_new_vkeys[i] for i in self.lizard]  # specify one when old_vkeys_to_new_vkeys[i] is a tuple

    def from_vector_to_string(self, vector: list[int]) -> list[str]:
        string = []
        for i in range(len(vector) // 2):
            x, y = vector[2 * i : 2 * i + 2]
            if (x, y) == (0, 0):
                string.append("t")
            elif (x, y) == (0, 1):
                string.append("p")
            elif (x, y) == (1, 0):
                string.append("a")
            elif (x, y) == (1, 1):
                string.append("d")
        return string

    def from_string_to_vector(self, string: list[str]) -> list[int]:
        vector = []
        for k in string:
            if k == "t":
                vector += [0, 0]
            elif k == "p":
                vector += [0, 1]
            elif k == "a":
                vector += [1, 0]
            elif k == "d":
                vector += [1, 1]
        return vector

    def from_string_to_rules(self, string: list[str]) -> None:
        for k in string:
            if k == "t":
                self.turn()
            elif k == "p":
                self.pivot()
            elif k == "a":
                self.add()
            elif k == "d":
                self.delete()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import random as rd
    from math import pi

    from compas_plotters.meshplotter import MeshPlotter

    from compas.geometry import add_vectors
    from compas.numerical import fd_numpy
    from compas_singular.datastructures import CoarseQuadMesh
    from compas_singular.datastructures import QuadMesh
    from compas_singular.datastructures.mesh.operations import mesh_move_vertex_to
    from compas_singular.datastructures.mesh_quad.grammar.add_strip import add_strip
    from compas_singular.geometry import circle_evaluate

    def fix_boundaries(mesh: "QuadMesh") -> None:
        n = len(mesh.vertices_on_boundaries())
        for i, vkey in enumerate(mesh.boundaries()[0]):
            xyz = add_vectors(mesh.vertex_centroid(), circle_evaluate(2.0 * pi * i / n, 10))
            mesh_move_vertex_to(mesh, xyz, vkey)

    def find_form(mesh: "QuadMesh", total_load: float) -> None:
        vertices = [mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()]
        edges = list(mesh.edges())
        fixed = mesh.vertices_on_boundaries()
        q = [1.0] * len(edges)
        # total_area = mesh.area()
        loads = [[0.0, 0.0, total_load / mesh.number_of_vertices()]] * mesh.number_of_vertices()
        xyz, q, f, lengths, r = fd_numpy(vertices, edges, fixed, q, loads)
        for vkey, coordinates in zip(mesh.vertices(), xyz):
            mesh_move_vertex_to(mesh, coordinates, vkey)

    vertices_1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    faces_1 = [[0, 1, 3, 2]]

    vertices_2 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [0.0, 2.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0]]
    faces_2 = [[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]

    vertices_3 = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [1.0, 2.0, 0.0], [2.0, 2.0, 0.0], [0.0, 3.0, 0.0], [3.0, 3.0, 0.0]]
    faces_3 = [[0, 1, 3, 2], [1, 7, 5, 3], [7, 6, 4, 5], [6, 0, 2, 4], [2, 3, 5, 4]]

    mesh_0 = CoarseQuadMesh.from_vertices_and_faces(vertices_3, faces_3)
    mesh_0.collect_strips()

    n = 10
    for k in range(n):
        print(k + 1, "/", n)
        mesh = mesh_0.copy()

        lizard = Lizard(mesh)
        lizard.initiate()
        # vector = lizard.from_string_to_vector('atta') # ata, atta, attta, attta...d
        bin1 = [rd.randint(0, 1) for _ in range(rd.randint(0, 20))]
        bin2 = {0: [0, 0], 1: [0, 1]}
        vector = [1, 0] + [b2 for b1 in bin1 for b2 in bin2[b1]] + [1, 0]
        # vector = [rd.randint(0,1) for _ in range(2 * rd.randint(0,10))]
        # print(vector)
        string = lizard.from_vector_to_string(vector)
        # print(string)
        try:
            lizard.from_string_to_rules(string)
            # print(lizard.lizard)

            orders = [rd.randint(0, 1) for _ in range(5)]
            orders = [2] + orders + [2]
            orders = [2, 1, 3, 0, 3, 2, 1, 3, 2, 2]

            # print('boundaries 1/2')
            fix_boundaries(mesh)
            # print('densification')
            mesh.set_mesh_density_face_target(300)
            mesh.densification()
            # print('boundaries 2/2')
            fix_boundaries(mesh.get_quad_mesh())
            # print('form finding')
            find_form(mesh.get_quad_mesh(), 100.0)

            plotter = MeshPlotter(mesh.get_quad_mesh(), figsize=(10, 10))
            plotter.draw_vertices(radius=0.1)  # , text='key')
            plotter.draw_edges()
            plotter.draw_faces()
            name = "".join([str(i) for i in vector])
            # plotter.save('../../../../data/3/' + name)
            plotter.show()
            # print('done')
        except Exception:
            pass
