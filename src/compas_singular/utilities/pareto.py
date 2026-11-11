from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function

from typing import Iterable

__all__ = ["extract_pareto_indices", "is_dominating"]


def extract_pareto_indices(data: list[Iterable[float]], k: float = 1.0) -> list[int]:
    """Get the Pareto front from data. The performances are to be minimised.
    If the metrics must be maximised, take the opposite X <- -X and the inverse k <- 1/k.

    Parameters
    ----------
    data : list
        A list of iterables storing the performance data.
    k : float, optional
        Parameter for weak domination. Default is 1.0.

    Returns
    -------
    list
        List of indices of Pareto front in data.

    References
    ----------
    1. Wikipedia. *Multi-objective optimization*.
           Available at: https://en.wikipedia.org/wiki/Multi-objective_optimization.
    """

    return [i for i, Xi in enumerate(data) if len([Xj for Xj in data if is_dominating(Xj, Xi, 1 / k)]) == 0]


def is_dominating(X1: Iterable[float], X2: Iterable[float], k: float = 1.0) -> bool:
    """Check if design X1 dominates X2: all metrics below or equal and one strictly below.

    ``k`` below 1.0 allows weak domination.

    Parameters
    ----------
    X1 : iterable
        A design to test if is dominating.
    X2 : iterable
        A design to test if is dominated.
    k : float, optional
        Parameter for weak domination. Default if 1.0.

    Returns
    -------
    bool
        True if is dominating, False otherwise

    References
    ----------
    1. Wikipedia. *Multi-objective optimization*.
           Available at: https://en.wikipedia.org/wiki/Multi-objective_optimization.
    """

    return all([k * X1k <= X2k for X1k, X2k in zip(X1, X2)]) and any([k * X1k < X2k for X1k, X2k in zip(X1, X2)])
