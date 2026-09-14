from __future__ import annotations

import numpy as np


def search_raw(query: np.ndarray, refs: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw inner product search; deliberately does not normalize inputs."""
    if not len(query) or not len(refs):
        return np.empty((len(query), 0), np.float32), np.empty((len(query), 0), np.int64)
    k = min(top_k, len(refs))
    try:
        import faiss

        index = faiss.IndexFlatIP(refs.shape[1])
        index.add(np.ascontiguousarray(refs, dtype=np.float32))
        return index.search(np.ascontiguousarray(query, dtype=np.float32), k)
    except ImportError:
        sims = query @ refs.T
        ids = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, ids, axis=1), ids


def candidate_assets(
    query: np.ndarray, refs: list[tuple[int, np.ndarray]], top_k: int = 50
) -> list[int]:
    if not refs:
        return []
    vectors = np.concatenate([x[1] for x in refs])
    owners = np.concatenate([np.full(len(x[1]), x[0], np.int64) for x in refs])
    _, ids = search_raw(query, vectors, top_k)
    support: dict[int, set[int]] = {}
    for qi, row in enumerate(ids):
        for vector_id in row:
            support.setdefault(int(owners[vector_id]), set()).add(qi)
    return [k for k, _ in sorted(support.items(), key=lambda x: (-len(x[1]), x[0]))]
