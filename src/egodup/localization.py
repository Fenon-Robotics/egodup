from __future__ import annotations

import numpy as np

from .config import MatchingProfile
from .schemas import Correspondence, Segment, TimeRelation
from .vendor.temporal_network import temporal_network


def row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize a zero descriptor")
    return x / norms


def fit_time_relation(q: np.ndarray, r: np.ndarray) -> TimeRelation:
    if len(q) < 2 or np.ptp(q) == 0:
        a, b = 1.0, float(np.median(r - q))
    else:
        a, b = np.polyfit(q, r, 1)
    residual = np.abs(r - (a * q + b))
    return TimeRelation(
        a=float(a), b=float(b), median_absolute_residual_seconds=float(np.median(residual))
    )


def _cells_duration(times: np.ndarray, indices: np.ndarray, interval: float = 1.0) -> float:
    if not len(indices):
        return 0.0
    cells = [
        (max(0.0, times[i] - interval / 2), times[i] + interval / 2) for i in sorted(set(indices))
    ]
    start, end = cells[0]
    total = 0.0
    for a, b in cells[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    return total + end - start


def localize(
    q_raw: np.ndarray,
    r_raw: np.ndarray,
    q_times: np.ndarray,
    r_times: np.ndarray,
    profile: MatchingProfile | None = None,
    verify_samples: int = 0,
) -> list[Segment]:
    profile = profile or MatchingProfile()
    sims = row_normalize(q_raw) @ row_normalize(r_raw).T
    paths = temporal_network(
        sims, max_step=profile.tn_max_step, top_k=profile.tn_top_k, min_length=profile.tn_min_length
    )
    segments: list[Segment] = []
    for item in paths:
        points = sorted(set(item.points))
        q_idx = np.asarray([p[0] for p in points], dtype=int)
        r_idx = np.asarray([p[1] for p in points], dtype=int)
        q, r = q_times[q_idx], r_times[r_idx]
        scores = sims[q_idx, r_idx]
        q_start, q_end = float(q.min()), float(q.max() + 1.0)
        r_start, r_end = float(r.min()), float(r.max() + 1.0)
        supported = _cells_duration(q_times, q_idx)
        span = max(1e-9, q_end - q_start)
        gaps = np.diff(np.unique(q))
        largest_gap = float(gaps.max()) if len(gaps) else 0.0
        segment = Segment(
            query_interval_seconds=(q_start, q_end),
            reference_interval_seconds=(r_start, r_end),
            timestamp_uncertainty_seconds=0.5,
            matched_query_samples=len(set(q_idx.tolist())),
            supported_duration_seconds=supported,
            support_coverage=min(1.0, supported / span),
            median_cosine=float(np.median(scores)),
            p10_cosine=float(np.percentile(scores, 10)),
            largest_internal_gap_seconds=largest_gap,
            time_relation=fit_time_relation(q, r),
            correspondences=[
                Correspondence(query_seconds=float(a), reference_seconds=float(b), cosine=float(c))
                for a, b, c in zip(q, r, scores)
            ],
            verification_status="uncalibrated" if verify_samples else "not_requested",
        )
        if passes_gates(segment, profile):
            segments.append(segment)
    return sorted(segments, key=lambda s: s.query_interval_seconds)


def passes_gates(segment: Segment, profile: MatchingProfile) -> bool:
    return (
        segment.supported_duration_seconds >= profile.min_supported_seconds
        and segment.matched_query_samples >= profile.min_distinct_samples
        and segment.support_coverage >= profile.min_coverage
        and segment.median_cosine >= profile.median_cosine
        and segment.p10_cosine >= profile.p10_cosine
        and segment.largest_internal_gap_seconds <= profile.max_internal_gap_seconds
    )


def interval_union(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    total = 0.0
    start, end = sorted(intervals)[0]
    for a, b in sorted(intervals)[1:]:
        if a <= end:
            end = max(end, b)
        else:
            total += end - start
            start, end = a, b
    return total + end - start
