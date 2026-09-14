from pathlib import Path

import numpy as np
import pytest

from egodup.config import FeatureProfile, MatchingProfile
from egodup.features import load_features, store_features
from egodup.localization import fit_time_relation, interval_union, row_normalize
from egodup.media import sha256_file
from egodup.retrieval import search_raw
from egodup.schemas import Result


def test_sha256_streaming(tmp_path: Path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"egodup")
    assert sha256_file(path) == "c8e6c6ff44b7e1c746f92e4440409eebb838588c9bc1ef626f4605fb7f0d2235"


def test_profile_identity_is_stable_and_sensitive():
    assert FeatureProfile().identity == FeatureProfile().identity
    assert FeatureProfile(sample_interval_seconds=2).identity != FeatureProfile().identity
    assert MatchingProfile(frame_top_k=2).identity != MatchingProfile().identity


def test_raw_inner_product_is_not_cosine():
    q = np.asarray([[2, 0]], np.float32)
    refs = np.asarray([[1, 0], [3, 0]], np.float32)
    scores, ids = search_raw(q, refs, 2)
    assert ids.tolist() == [[1, 0]]
    assert scores.tolist() == [[6.0, 2.0]]
    assert row_normalize(q).tolist() == [[1.0, 0.0]]


def test_faiss_numpy_parity():
    rng = np.random.default_rng(3)
    q = rng.normal(size=(4, 8)).astype("float32")
    refs = rng.normal(size=(9, 8)).astype("float32")
    scores, ids = search_raw(q, refs, 3)
    expected_ids = np.argsort(-(q @ refs.T), axis=1)[:, :3]
    assert np.array_equal(ids, expected_ids)
    assert np.allclose(scores, np.take_along_axis(q @ refs.T, expected_ids, axis=1))


def test_time_relation_speed_convention():
    q = np.asarray([0, 1, 2, 3], float)
    r = np.asarray([7, 9, 11, 13], float)
    relation = fit_time_relation(q, r)
    assert relation.a == pytest.approx(2.0)
    assert relation.b == pytest.approx(7.0)


def test_interval_union_does_not_charge_gaps():
    assert interval_union([(0, 4), (2, 5), (9, 11)]) == 7


def test_atomic_feature_artifact(tmp_path: Path):
    raw = np.ones((3, 512), np.float32)
    times = np.arange(3, dtype=float)
    path = store_features(tmp_path, "profile", "a" * 64, raw, times, np.zeros(3), {"test": True})
    loaded, loaded_times = load_features(path)
    assert np.array_equal(loaded, raw)
    assert np.array_equal(loaded_times, times)
    assert not any((tmp_path / "staging").iterdir())


def test_schema_rejects_nan():
    with pytest.raises(Exception):
        Result.model_validate(
            {
                "schema_version": "1.0",
                "run_id": "r",
                "query": {
                    "submission_id": "s",
                    "path": "x",
                    "sha256": "a",
                    "duration_seconds": float("nan"),
                },
                "execution_status": "completed",
                "decision": "no_match_found",
                "review_required": False,
                "search": {},
                "feature_profile_sha256": "a",
                "matching_profile_sha256": "b",
                "device": "cpu",
            }
        )
