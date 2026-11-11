from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

import itertools
from typing import Any
from typing import Iterator

from compas.topology import vertex_adjacency_from_edges
from compas_singular.datastructures import QuadMesh
from compas_singular.datastructures import collateral_strip_deletions
from compas_singular.datastructures import delete_strips
from compas_singular.datastructures import split_strips
from compas_singular.datastructures import strips_to_split_to_prevent_boundary_collapse
from compas_singular.datastructures import total_boundary_deletions
from compas_singular.topology import is_adjacency_two_colorable
from compas_singular.utilities import are_items_in_list


def delete_strips_preserving_boundaries(mesh: QuadMesh, skeys: list[int]) -> None:
    """Delete strips, refining first whatever would let a boundary collapse."""
    to_split = strips_to_split_to_prevent_boundary_collapse(mesh, skeys)
    # Falsy covers both states that mean "do not split": ``None`` -- no strip
    # survives on that boundary to refine -- and ``{}`` -- nothing is at risk.
    if to_split:
        split_strips(mesh, to_split, open_strip=False)
    delete_strips(mesh, skeys)


__all__ = [
    "TwoColourableProjection",
]


class TwoColourableProjection(object):
    def __init__(self, quad_mesh: QuadMesh) -> None:
        self.quad_mesh = quad_mesh
        self.results: Any = None
        self.times: Any = None

    def projection_4(self, kmax: int = 1) -> bool | None:
        """Projection of a coarse quad mesh to the closest two-colourable sub-spaces.

        Parameters
        ----------
        mesh : CoarseQuadMesh
            A coarse quad mesh.

        Returns
        -------
        results : dict
            The combination pointing to the its result.
            If the combination is valid, the result is a tuple of the the two-colourable mesh,
            the two-colourable network, and the network vertex colors.

        References
        ----------
        .. [1] Oval et al., *Topology Finding of Two-Colourable Quad-Mesh Patterns in Structural Design*. Submitted.

        """

        mesh = self.quad_mesh
        vertices, edges = mesh.strip_graph()
        if is_adjacency_two_colorable(vertex_adjacency_from_edges(edges)) is not None:
            self.results = True
            return True

        results = {}

        k = 0

        current_pool = [[skey] for skey in mesh.strips()]

        while k < kmax:
            k += 1
            next_pool = []

            for combination in current_pool:
                if len(combination) > 1:
                    combination = list(set([i for item in combination for i in item]))
                else:
                    combination = list(set(combination))

                if len(collateral_strip_deletions(mesh, combination)) > 0:
                    continue
                if len(total_boundary_deletions(mesh, combination)) > 0:
                    continue

                # delete strips in mesh and check validity
                copy_mesh = mesh.copy()
                delete_strips_preserving_boundaries(copy_mesh, combination)
                topological_validity = copy_mesh.is_manifold() and copy_mesh.euler() == mesh.euler()
                if not topological_validity:
                    pass

                # delete strip vertices in network and check colourability
                else:
                    new_vertices = {vkey: xyz for vkey, xyz in vertices.items() if vkey not in combination}
                    new_edges = [(u, v) for u, v in edges if u not in combination and v not in combination]
                    two_colourability = is_adjacency_two_colorable(vertex_adjacency_from_edges(new_edges))
                    if not two_colourability:
                        next_pool.append(combination)
                    else:
                        results[tuple(combination)] = (copy_mesh, (new_vertices, new_edges), two_colourability)

            current_pool = itertools.combinations(next_pool, 2)

        self.results = results

    def projection_1(self, kmax: int = 1) -> bool | None:
        """Projection of a coarse quad mesh to the closest two-colourable sub-spaces.

        Parameters
        ----------
        mesh : CoarseQuadMesh
            A coarse quad mesh.

        Returns
        -------
        results : dict
            The combination pointing to the its result.
            If the combination is valid, the result is a tuple of the the two-colourable mesh,
            the two-colourable network, and the network vertex colors.

        References
        ----------
        .. [1] Oval et al., *Topology Finding of Two-Colourable Quad-Mesh Patterns in Structural Design*. Submitted.

        """

        mesh = self.quad_mesh
        n = mesh.number_of_strips()

        vertices, edges = mesh.strip_graph()
        if is_adjacency_two_colorable(vertex_adjacency_from_edges(edges)) is not None:
            self.results = True
            return True

        relation = {}
        for k in range(n):
            for combination in itertools.combination(mesh.strips(), k):
                other_strips = list(mesh.strips())
                for skey in combination:
                    del other_strips[skey]
                # downsteam_combinations = itertools.combination(mesh.strips(), k)
                relation[combination] = []

        results = {}

        k = 0

        current_pool = [[skey] for skey in mesh.strips()]

        while k < kmax:
            k += 1
            next_pool = []

            for combination in current_pool:
                if len(combination) > 1:
                    combination = list(set([i for item in combination for i in item]))
                else:
                    combination = list(set(combination))

                if len(collateral_strip_deletions(mesh, combination)) > 0:
                    continue
                if len(total_boundary_deletions(mesh, combination)) > 0:
                    continue

                # delete strips in mesh and check validity
                copy_mesh = mesh.copy()
                delete_strips_preserving_boundaries(copy_mesh, combination)
                topological_validity = copy_mesh.is_manifold() and copy_mesh.euler() == mesh.euler()
                if not topological_validity:
                    pass

                # delete strip vertices in network and check colourability
                else:
                    new_vertices = {vkey: xyz for vkey, xyz in vertices.items() if vkey not in combination}
                    new_edges = [(u, v) for u, v in edges if u not in combination and v not in combination]
                    two_colourability = is_adjacency_two_colorable(vertex_adjacency_from_edges(new_edges))
                    if not two_colourability:
                        next_pool.append(combination)
                    else:
                        results[tuple(combination)] = (copy_mesh, (new_vertices, new_edges), two_colourability)

            current_pool = itertools.combinations(next_pool, 2)

        self.results = results

    def projection_2(self, kmax: int = 1) -> bool | None:
        """Projection of a coarse quad mesh to the closest two-colourable sub-spaces.

        Parameters
        ----------
        mesh : CoarseQuadMesh
            A coarse quad mesh.

        Returns
        -------
        results : dict
            The combination pointing to the its result.
            If the combination is valid, the result is a tuple of the the two-colourable mesh,
            the two-colourable network, and the network vertex colors.

        References
        ----------
        .. [1] Oval et al., *Topology Finding of Two-Colourable Quad-Mesh Patterns in Structural Design*. Submitted.

        """

        mesh = self.quad_mesh

        # result for input mesh
        vertices, edges = mesh.strip_graph()
        if is_adjacency_two_colorable(vertex_adjacency_from_edges(edges)) is not None:
            self.results = True
            return True

        results = {}

        # guarantee valid kmax
        n = mesh.number_of_strips()
        if kmax < 1 or kmax > n:
            kmax = n

        # start iteration
        k = 0
        discarding_combination = []
        while k < kmax:
            k += 1
            to_continue = False
            # at_least_one_valid_k = False
            # test all combinations of (n k) strips
            for combination in itertools.combinations(mesh.strips(), k):
                set_combi = set(combination)
                # check results from potential previous sub-combinations
                for disc_comb in discarding_combination:
                    if disc_comb.issubset(set_combi):
                        break

                if len(collateral_strip_deletions(mesh, combination)) > 0:
                    to_continue = True
                    continue

                if len(total_boundary_deletions(mesh, combination)) > 0:
                    discarding_combination.append(set(combination))
                    continue

                # delete strips in mesh and check validity
                copy_mesh = mesh.copy()
                delete_strips_preserving_boundaries(copy_mesh, combination)
                topological_validity = copy_mesh.is_manifold() and copy_mesh.euler() == mesh.euler()
                if not topological_validity:
                    discarding_combination.append(set(combination))

                # delete strip vertices in network and check colourability
                else:
                    new_vertices = {vkey: xyz for vkey, xyz in vertices.items() if vkey not in combination}
                    new_edges = [(u, v) for u, v in edges if u not in combination and v not in combination]
                    two_colourability = is_adjacency_two_colorable(vertex_adjacency_from_edges(new_edges))
                    if not two_colourability:
                        to_continue = True
                    else:
                        results[combination] = (copy_mesh, (new_vertices, new_edges), two_colourability)
                        discarding_combination.append(set(combination))

            if not to_continue:
                break

        self.results = results

    def projection(self, kmax: int = 1) -> Any:
        """Projection of a coarse quad mesh to the closest two-colourable sub-spaces.

        Parameters
        ----------
        mesh : CoarseQuadMesh
            A coarse quad mesh.

        Returns
        -------
        results : dict
            The combination pointing to the its result.
            If the combination is valid, the result is a tuple of the the two-colourable mesh,
            the two-colourable network, and the network vertex colors.

        References
        ----------
        .. [1] Oval et al., *Topology Finding of Two-Colourable Quad-Mesh Patterns in Structural Design*. Submitted.

        """

        mesh = self.quad_mesh

        # # result for input mesh
        # vertices, edges = mesh.strip_graph()
        # if is_adjacency_two_colorable(vertex_adjacency_from_edges(edges)) is not None:
        # 	self.results = True
        # 	return True

        results = {}

        # guarantee valid kmax
        n = mesh.number_of_strips()
        if kmax < 1 or kmax > n:
            kmax = n

        # start iteration
        k = 0
        while k < kmax:
            k += 1
            to_continue = False
            # test all combinations of (n k) strips
            for combination in itertools.combinations(mesh.strips(), k):
                # check results from potential previous sub-combinations
                for previous_combination in results:
                    if are_items_in_list(previous_combination, combination):
                        # if a sub-combination yielded an invalid topology do not pursue
                        if results[previous_combination] == "invalid shape topology":
                            results[combination] = "already invalid shape topology"
                            break
                        elif isinstance(results[previous_combination], tuple):
                            results[combination] = "already two-colourable"
                            break
                if combination in results:
                    continue

                if len(collateral_strip_deletions(mesh, combination)) > 0:
                    to_continue = True
                    continue

                if len(total_boundary_deletions(mesh, combination)) > 0:
                    results[combination] = "invalid shape topology"
                    continue

                # delete strips in mesh and check validity
                copy_mesh = mesh.copy()
                copy_mesh.collect_strips()
                delete_strips_preserving_boundaries(copy_mesh, combination)
                topological_validity = copy_mesh.is_manifold() and copy_mesh.euler() == mesh.euler()
                if not topological_validity:
                    results[combination] = "invalid shape topology"

                # delete strip vertices in network and check colourability
                else:
                    vertices, edges = copy_mesh.strip_graph()
                    two_colourability = is_adjacency_two_colorable(vertex_adjacency_from_edges(edges))
                    if not two_colourability:
                        results[combination] = "not two-colourable"
                        to_continue = True
                    else:
                        results[combination] = (copy_mesh, (vertices, edges), two_colourability)

            if not to_continue:
                break

        return self.results

    def projection_0(self, kmax: int = 1) -> bool | None:
        """Projection of a coarse quad mesh to the closest two-colourable sub-spaces.

        Parameters
        ----------
        mesh : CoarseQuadMesh
            A coarse quad mesh.

        Returns
        -------
        results : dict
            The combination pointing to the its result.
            If the combination is valid, the result is a tuple of the the two-colourable mesh,
            the two-colourable network, and the network vertex colors.

        References
        ----------
        .. [1] Oval et al., *Topology Finding of Two-Colourable Quad-Mesh Patterns in Structural Design*. Submitted.

        """

        mesh = self.quad_mesh

        # result for input mesh
        vertices, edges = mesh.strip_graph()
        if is_adjacency_two_colorable(vertex_adjacency_from_edges(edges)) is not None:
            self.results = True
            return True

        results = {}

        # guarantee valid kmax
        n = mesh.number_of_strips()
        if kmax < 1 or kmax > n:
            kmax = n

        total_valid = 0
        # start iteration
        k = 0
        discarding_combination = []
        discarding_combination_type = {}
        while k < kmax:
            k += 1
            to_continue = False
            # test all combinations of (n k) strips
            for combination in itertools.combinations(mesh.strips(), k):
                set_combi = set(combination)
                # check results from potential previous sub-combinations
                for disc_comb in discarding_combination:
                    if disc_comb.issubset(set_combi):
                        # if are_items_in_list(previous_combination, combination):
                        # if a sub-combination yielded an invalid topology do not pursue
                        if discarding_combination_type[tuple(disc_comb)] == "invalid shape topology":
                            results[combination] = "already invalid shape topology"
                            break
                        elif discarding_combination_type[tuple(disc_comb)] == "two-colourable":
                            results[combination] = "already two-colourable"
                            break

                if len(collateral_strip_deletions(mesh, combination)) > 0:
                    results[combination] = "collateral deletions"

                if len(total_boundary_deletions(mesh, combination)) > 0:
                    results[combination] = "invalid shape topology"
                    discarding_combination.append(set(combination))
                    discarding_combination_type[tuple(combination)] = "invalid shape topology"

                if combination in results:
                    continue

                # delete strips in mesh and check validity
                copy_mesh = mesh.copy()
                # copy_mesh.collect_strips()
                delete_strips_preserving_boundaries(copy_mesh, combination)
                topological_validity = copy_mesh.is_manifold() and copy_mesh.euler() == mesh.euler()
                if not topological_validity:
                    results[combination] = "invalid shape topology"
                    discarding_combination.append(set(combination))
                    discarding_combination_type[tuple(combination)] = "invalid shape topology"

                # delete strip vertices in network and check colourability
                else:
                    # vertices, edges = copy_mesh.strip_graph()
                    new_vertices = {vkey: xyz for vkey, xyz in vertices.items() if vkey not in combination}
                    new_edges = [(u, v) for u, v in edges if u not in combination and v not in combination]
                    two_colourability = is_adjacency_two_colorable(vertex_adjacency_from_edges(new_edges))
                    if not two_colourability:
                        results[combination] = "not two-colourable"
                        to_continue = True
                    else:
                        results[combination] = (copy_mesh, (new_vertices, new_edges), two_colourability)
                        discarding_combination.append(set(combination))
                        discarding_combination_type[tuple(combination)] = "two-colourable"
                        total_valid += 1

            if not to_continue:
                break

        self.results = results

    # --------------------------------------------------------------------------
    # results
    # --------------------------------------------------------------------------

    def get_results(self) -> Any:
        return self.results

    def two_coloured_meshes(self, kmax: int = 1) -> Iterator[QuadMesh]:
        self.projection_4(kmax)
        for strips, results in self.get_results().items():
            yield results[0]

    def strip_deletions_yielding_two_colourability(self) -> list[tuple[int, ...]]:
        out = []
        for combination, result in self.get_results().items():
            if isinstance(result, tuple):
                out.append(combination)
        return out
