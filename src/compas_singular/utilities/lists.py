from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Any

__all__ = ["list_split", "sublist_from_to_items_in_closed_list", "are_items_in_list", "common_items", "remove_isomorphism_in_integer_list"]


def list_split(lst: list[Any], indices: list[int]) -> list[list[Any]]:
    """Split list at given indices.
    Closed lists have the same first and last elements.
    If the list is closed, splitting wraps around if the first or last index is not in the indices to split.


    Parameters
    ----------
    lst : list
            A list.
    indices : list
            A list of indices to split.

    Returns
    -------
    split_lists : list
            Nest lists from splitting the list at the given indices.

    """

    n = len(lst)

    if lst[0] == lst[-1]:
        closed = True
        if n - 1 in indices:
            indices.remove(n - 1)
            if 0 not in indices:
                indices.append(0)
    else:
        closed = False

    indices = list(sorted(set(indices)))

    split_lists = []
    current_list = []
    for index, item in enumerate(lst):
        current_list.append(item)
        if (index in indices and index != 0) or index == n - 1:
            split_lists.append(current_list)
            current_list = [item]

    if closed:
        if 0 not in indices:
            start = split_lists.pop(0)[1:]
            split_lists[-1] += start

    return split_lists


def sublist_from_to_items_in_closed_list(lst: list[Any], from_item: Any, to_item: Any) -> list[Any] | None:
    """Return sublist between oe item to another.

    Parameters
    ----------
    lst : list
            A list.
    from_item
            An item to be found in the list. The beginning of the sublist.
    to_item
            An item to be found in the list. The end of the sublist.

    Returns
    -------
    sublist : list
            A sublist from the input list, between from_item and to_item.
    """

    if from_item == to_item:
        return [from_item]
    if lst[0] != lst[-1]:
        lst.append(lst[0])
    from_idx = lst.index(from_item)
    to_idx = lst.index(to_item)
    sublists = list_split(lst, [from_idx, to_idx])

    for sublist in sublists:
        if sublist[0] == from_item:
            return sublist


def are_items_in_list(items: list[Any], lst: list[Any]) -> bool:
    """Check if items are in a list.

    Parameters
    ----------
    items : list
            A list of items (order does not matter).
    lst : list
            A list.

    Returns
    -------
    bool
            True if all items are in the list. False otherwise.
    """

    for i in items:
        if i not in lst:
            return False
    return True


def common_items(l1: list[Any], l2: list[Any]) -> list[Any]:
    """Return common items in two lists.

    Parameters
    ----------
    l1 : list
            A list.
    l2 : list
            A list.

    Returns
    -------
    list
            The common items.
    """

    return [item for item in l1 if item in l2]


def remove_isomorphism_in_integer_list(lst: list[int]) -> list[int]:
    # remove isomorphisms in list (open or closed)
    # interpreted as a polyedge

    if len(lst) < 2:
        return lst

    # if closed: min value first, and its minimum neighbour value second
    if lst[0] == lst[-1]:
        lst = lst[:-1]
        candidates = []

        start = min(lst)
        for i, key in enumerate(lst):
            # collect all candidates, there may be multiple minimum values and multiple minimum neighbours
            if key == start:
                candidate = lst[i:] + lst[:i] + [lst[i]]
                candidates.append(candidate)
                candidates.append(list(reversed(candidate)))
        for k in range(1, len(lst) + 1):
            n = len(candidates)
            if n == 1:
                break
            # get minimum-sum sub-list
            min_x = None
            for candidate in candidates:
                x = sum(candidate[:k])
                if min_x is None or x < min_x:
                    min_x = x
            # compare to minimum-sum sub-list
            for i, candidate in enumerate(reversed(candidates)):
                if sum(candidate[:k]) > min_x:
                    del candidates[n - i - 1]
        # potentially multiple canidates left due to symmetry in list, but no isomorphism left
        lst = candidates[0]

    # if open: minimum value extremmity at the start
    else:
        if lst[0] > lst[-1]:
            lst = list(reversed(lst))

    return lst


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    pass

    # import compas

    # print(list_split(list(range(20)) + [0], [0, 8, 9, 12, 13]))

    # print(sublist_from_to_items_in_closed_list(range(20), 13, 13))
