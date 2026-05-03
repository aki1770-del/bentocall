"""Symbolic planner: chooses (k*, τ*, depth) before execution."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Plan:
    k_star: int       # branching factor at every level
    tau_star: int     # leaf threshold (tokens)
    depth: int        # recursion depth
    expected_calls: int  # k_star ** depth (number of leaves)


def plan(n_tokens: int, K: int, max_branching: int = 16) -> Plan:
    """Choose (k*, τ*, d) for an input of n tokens with leaf budget K.

    Strategy: τ* = K. Pick smallest k* ≥ 2 such that one level of split
    fits each chunk into K; if not, increase depth.
    """
    if n_tokens <= K:
        return Plan(k_star=1, tau_star=K, depth=0, expected_calls=1)

    for d in range(1, 6):
        k = math.ceil((n_tokens / K) ** (1.0 / d))
        k = max(k, 2)
        if k <= max_branching:
            return Plan(k_star=k, tau_star=K, depth=d, expected_calls=k ** d)

    raise ValueError(f"input too large: n={n_tokens}, K={K}")
