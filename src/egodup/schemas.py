from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TimeRelation(StrictModel):
    equation: str = "t_reference = a * t_query + b"
    a: float
    b: float
    median_absolute_residual_seconds: float


class Correspondence(StrictModel):
    query_seconds: float
    reference_seconds: float
    cosine: float


class Segment(StrictModel):
    query_interval_seconds: tuple[float, float]
    reference_interval_seconds: tuple[float, float]
    boundary_precision: Literal["sampled"] = "sampled"
    timestamp_uncertainty_seconds: float
    matched_query_samples: int
    supported_duration_seconds: float
    support_coverage: float
    median_cosine: float
    p10_cosine: float
    largest_internal_gap_seconds: float
    time_relation: TimeRelation
    correspondences: list[Correspondence]
    verification_status: Literal["not_requested", "uncalibrated", "inconclusive"] = "not_requested"


class Match(StrictModel):
    reference_submission_id: str
    reference_path: str
    reference_sha256: str
    exact_file_match: bool
    segments: list[Segment] = Field(default_factory=list)


class QueryInfo(StrictModel):
    submission_id: str
    path: str
    sha256: str
    duration_seconds: float | None = None


class SearchInfo(StrictModel):
    reference_generation: str | None = None
    within_batch: bool = False
    completed_within_policy: bool = True
    candidate_budget_exhausted: bool = False


class CalibrationInfo(StrictModel):
    id: str = "development-v1"
    provisional: bool = True


class Result(StrictModel):
    schema_version: str = "1.0"
    run_id: str
    query: QueryInfo
    execution_status: Literal["completed", "partial", "failed"]
    decision: (
        Literal["exact_duplicate", "suspected_copy", "no_match_found", "insufficient_evidence"]
        | None
    )
    review_required: bool
    search: SearchInfo
    calibration: CalibrationInfo = Field(default_factory=CalibrationInfo)
    feature_profile_sha256: str
    matching_profile_sha256: str
    device: str
    matches: list[Match] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    stage_timings_seconds: dict[str, float] = Field(default_factory=dict)
