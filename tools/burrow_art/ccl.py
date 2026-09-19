"""Run-based 8-connected component labelling.

scipy is not available in the build image, and a per-pixel flood fill over a
1800x1500 canvas is too slow in pure Python. Rows are encoded as runs instead,
and runs of neighbouring rows are unioned, which keeps the work proportional to
the number of runs rather than the number of pixels.
"""

from __future__ import annotations

import numpy as np


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) spans of True in a 1-D boolean row."""
    padded = np.concatenate(([False], row, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


class _UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def component_areas(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label 8-connected components of `mask`.

    Returns `(labels, areas)` where `labels` is an int32 array with 0 for
    background and 1..n for components, and `areas[i]` is the pixel count of
    label `i + 1`.
    """
    height = mask.shape[0]
    uf = _UnionFind()
    runs_per_row: list[list[tuple[int, int, int]]] = []
    previous: list[tuple[int, int, int]] = []

    for y in range(height):
        current: list[tuple[int, int, int]] = []
        for start, end in _row_runs(mask[y]):
            run_id = uf.add()
            # 8-connectivity: runs touch when they overlap or meet diagonally.
            for p_start, p_end, p_id in previous:
                if p_start <= end and start <= p_end:
                    uf.union(run_id, p_id)
            current.append((start, end, run_id))
        runs_per_row.append(current)
        previous = current

    roots = {}
    labels = np.zeros(mask.shape, dtype=np.int32)
    areas: list[int] = []

    for y, runs in enumerate(runs_per_row):
        for start, end, run_id in runs:
            root = uf.find(run_id)
            label = roots.get(root)
            if label is None:
                areas.append(0)
                label = len(areas)
                roots[root] = label
            labels[y, start:end] = label
            areas[label - 1] += end - start

    return labels, np.array(areas, dtype=np.int64)
