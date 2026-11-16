"""Compiled cell draws with the original Fenwick arithmetic and NumPy RNG stream.

Called only by the rank's planning process. Readers never execute this kernel.
Weights are private scratch space and become zero at selected positions.
"""

import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def draw_without_replacement(weights, cells, epsilon, rng):
    size = len(weights)
    tree = np.empty(size + 1, dtype=np.float64)
    tree[0] = 0.
    tree[1:] = weights
    # Preserve the Python implementation's addition order for exact replay.
    for index in range(1, size + 1):
        parent = index + (index & -index)
        if parent <= size:
            tree[parent] += tree[index]
    total = 0.
    index = size
    while index > 0:
        total += tree[index]
        index -= index & -index
    highest_bit = 1
    while highest_bit * 2 <= size:
        highest_bit *= 2
    available = np.arange(size)
    positions = available.copy()
    remaining = size
    selected = np.empty(cells, dtype=np.int64)
    for slot in range(cells):
        if rng.random() < epsilon:
            cell = available[rng.integers(0, remaining)]
        else:
            if not np.isfinite(total) or total <= 0.:
                raise RuntimeError("Cannot sample a Fenwick tree with zero total weight")
            cumulative = rng.random() * total
            if not np.isfinite(cumulative) or cumulative < 0. or cumulative >= total:
                raise ValueError("cumulative mass must be in [0, total)")
            cell = 0
            bit = highest_bit
            while bit:
                candidate = cell + bit
                if candidate <= size and tree[candidate] <= cumulative:
                    cumulative -= tree[candidate]
                    cell = candidate
                bit >>= 1
            if cell >= size:
                raise RuntimeError("Fenwick search exceeded the tree")
        selected[slot] = cell
        difference = -weights[cell]
        weights[cell] = 0.
        total += difference
        position = cell + 1
        while position <= size:
            tree[position] += difference
            position += position & -position
        position = positions[cell]
        remaining -= 1
        last = available[remaining]
        available[position] = last
        positions[last] = position
    return selected
