"""Narrow adaptation of VCSL TN, adding matched paths.

Upstream: alipay/VCSL vcsl/vta.py @ 29ce63909ce605a73335ce48d1e21f395b4b503c
Copyright (c) 2021 vcsl-owner, MIT. See THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np


@dataclass
class TNPath:
    box: tuple[int, int, int, int]
    points: list[tuple[int, int]]


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    if not len(others):
        return np.asarray([0.0])
    lt = np.maximum(box[None, :2], others[:, :2])
    rb = np.minimum(box[None, 2:], others[:, 2:])
    wh = np.maximum(rb - lt + 1, 0)
    inter = wh[:, 0] * wh[:, 1]
    areas = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    other_areas = (others[:, 2] - others[:, 0] + 1) * (others[:, 3] - others[:, 1] + 1)
    return inter / (areas + other_areas - inter)


def temporal_network(
    sims: np.ndarray,
    max_step: int = 5,
    top_k: int = 5,
    max_paths: int = 10,
    min_sim: float = 0.2,
    min_length: int = 4,
    max_iou: float = 0.3,
) -> list[TNPath]:
    if sims.ndim != 2 or not sims.size or min(sims.shape) < 2:
        return []
    top = min(top_k, sims.shape[1])
    indices = np.argsort(-sims, axis=1)[:, :top]
    values = np.take_along_axis(sims, indices, axis=1)
    graph = nx.DiGraph()
    pairs: dict[int, tuple[int, int]] = {0: (-1, -1)}
    ids: dict[tuple[int, int], int] = {(-1, -1): 0}
    graph.add_node(0)
    node = 1
    for qi in range(sims.shape[0]):
        for ri in indices[qi]:
            pairs[node] = (qi, int(ri))
            ids[(qi, int(ri))] = node
            graph.add_node(node)
            node += 1
    for qi in range(sims.shape[0]):
        ri = indices[qi]
        intermediate = np.empty((0,), dtype=np.int32)
        for qj in range(qi + 1, min(sims.shape[0], qi + max_step)):
            rj = indices[qj]
            diff = rj[:, None] - ri
            c2 = (diff > 0) & (diff < max_step)
            if len(intermediate):
                c3 = (
                    np.sum(
                        (intermediate[None, None, :] < rj[:, None, None])
                        & (intermediate[None, None, :] > ri[None, :, None]),
                        axis=-1,
                    )
                    == 0
                )
            else:
                c3 = np.ones_like(c2, dtype=bool)
            score = np.repeat(values[qj][:, None], top, axis=1)
            rows, cols = np.where(c2 & c3 & (score >= min_sim))
            intermediate = np.unique(np.concatenate([intermediate, rj[rows]]))
            for row, col in zip(rows, cols):
                graph.add_edge(
                    ids[(qi, int(ri[col]))], ids[(qj, int(rj[row]))], weight=float(score[row, col])
                )
    results: list[TNPath] = []
    for _ in range(max_paths):
        path_ids = nx.algorithms.dag.dag_longest_path(graph, weight="weight")
        path_ids = [i for i in path_ids if i != 0]
        if not path_ids:
            break
        points = [pairs[i] for i in path_ids]
        for a, b in zip(path_ids, path_ids[1:]):
            graph[a][b]["weight"] = 0.0
        q = np.asarray([p[0] for p in points])
        r = np.asarray([p[1] for p in points])
        score = float(sims[q, r].sum())
        box = np.asarray([q.min(), r.min(), q.max(), r.max()], dtype=int)
        ave_length = max(1.0, ((box[2] - box[0]) + (box[3] - box[1])) / 2)
        if (
            score / ave_length > min_sim
            and min(box[2] - box[0], box[3] - box[1]) > min_length
            and _iou(box, np.asarray([x.box for x in results])).max() < max_iou
        ):
            results.append(TNPath(tuple(int(x) for x in box), points))
    return results
