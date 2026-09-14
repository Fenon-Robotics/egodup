import numpy as np

from egodup.vendor.temporal_network import temporal_network


def test_empty_and_small_matrices():
    assert temporal_network(np.empty((0, 0))) == []
    assert temporal_network(np.ones((1, 1))) == []


def test_path_and_coordinate_ordering():
    sim = np.full((20, 24), -0.5, np.float32)
    for q in range(2, 17):
        sim[q, q + 3] = 0.95
    paths = temporal_network(sim, max_step=5, top_k=1, min_length=4)
    assert paths
    assert paths[0].box == (2, 5, 16, 19)
    assert paths[0].points[0] == (2, 5)
    assert paths[0].points[-1] == (16, 19)
