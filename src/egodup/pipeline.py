from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from filelock import FileLock

from .config import FeatureProfile, MatchingProfile, resolve_device
from .db import add_asset, connect, references
from .features import feature_dir, load_features, store_features
from .localization import localize
from .media import probe, sample_video, sha256_file
from .model import infer, model_metadata
from .retrieval import candidate_assets
from .schemas import Match, QueryInfo, Result, SearchInfo


PROVISIONAL_WARNING = "Thresholds are provisional; manual review required."


def extract(path: Path, device: str) -> tuple[str, dict, np.ndarray, np.ndarray, dict]:
    t0 = time.perf_counter()
    sha = sha256_file(path)
    hashing = time.perf_counter() - t0
    t0 = time.perf_counter()
    metadata = probe(path)
    probing = time.perf_counter() - t0
    t0 = time.perf_counter()
    sampled = sample_video(path)
    decoding = time.perf_counter() - t0
    t0 = time.perf_counter()
    raw, batch = infer(sampled.frames, device)
    inference = time.perf_counter() - t0
    details = {
        "effective_batch_size": batch,
        "frame_ids": sampled.frame_ids.tolist(),
        "sample_errors": sampled.errors.tolist(),
        "diagnostics": sampled.diagnostics,
    }
    timings = {
        "hashing": hashing,
        "probing": probing,
        "decoding_sampling": decoding,
        "inference": inference,
    }
    return sha, metadata, raw, sampled.timestamps, {**details, "timings": timings}


def compare(
    query_path: Path, reference_path: Path, device: str = "auto", verify_samples: int = 0
) -> Result:
    device = resolve_device(device)
    run_id = f"run-{uuid.uuid4().hex}"
    feature_profile, matching = FeatureProfile(), MatchingProfile()
    started = time.perf_counter()
    q_sha = sha256_file(query_path)
    r_sha = sha256_file(reference_path)
    if q_sha == r_sha:
        warnings = []
        try:
            duration = probe(query_path).get("duration_seconds")
        except Exception as exc:
            duration = None
            warnings.append(f"Media probing failed after valid exact-file match: {exc}")
        return _result(
            run_id,
            query_path,
            q_sha,
            duration,
            device,
            "exact_duplicate",
            [
                Match(
                    reference_submission_id=_submission(reference_path),
                    reference_path=str(reference_path),
                    reference_sha256=r_sha,
                    exact_file_match=True,
                )
            ],
            feature_profile,
            matching,
            {"total": time.perf_counter() - started},
            extra_warnings=warnings,
        )
    q_sha, q_meta, q_raw, q_times, q_details = extract(query_path, device)
    r_sha, _, r_raw, r_times, r_details = extract(reference_path, device)
    t0 = time.perf_counter()
    segments = localize(
        q_raw[:240], r_raw[:240], q_times[:240], r_times[:240], matching, verify_samples
    )
    align = time.perf_counter() - t0
    matches = (
        [
            Match(
                reference_submission_id=_submission(reference_path),
                reference_path=str(reference_path),
                reference_sha256=r_sha,
                exact_file_match=False,
                segments=segments,
            )
        ]
        if segments
        else []
    )
    timings = _merge_timings(q_details["timings"], r_details["timings"])
    timings.update({"alignment": align, "total": time.perf_counter() - started})
    return _result(
        run_id,
        query_path,
        q_sha,
        q_meta.get("duration_seconds"),
        device,
        "suspected_copy" if matches else _negative(q_times),
        matches,
        feature_profile,
        matching,
        timings,
    )


def index(paths: list[Path], db_root: Path, device: str = "auto") -> dict:
    device = resolve_device(device)
    feature_profile = FeatureProfile()
    con = connect(db_root)
    completed, cached, failed = 0, 0, []
    with FileLock(str(db_root / ".writer.lock")):
        for path in paths:
            try:
                sha = sha256_file(path)
                dest = feature_dir(db_root, feature_profile.identity, sha)
                if dest.exists():
                    raw, ts = load_features(dest)
                    metadata = probe(path)
                    cached += 1
                else:
                    sha, metadata, raw, ts, details = extract(path, device)
                    quality = np.asarray(details["sample_errors"], dtype=np.float32)
                    dest = store_features(
                        db_root,
                        feature_profile.identity,
                        sha,
                        raw,
                        ts,
                        quality,
                        {
                            "source_path": str(path),
                            "model": model_metadata(),
                            "profile": feature_profile.identity,
                        },
                    )
                add_asset(
                    con,
                    sha,
                    path.stat().st_size,
                    metadata.get("duration_seconds"),
                    dest,
                    feature_profile.identity,
                )
                completed += 1
            except Exception as exc:
                failed.append({"path": str(path), "error": str(exc)})
    return {
        "indexed": completed,
        "cache_hits": cached,
        "failed": failed,
        "feature_profile_sha256": feature_profile.identity,
        "device": device,
    }


def scan(
    paths: list[Path],
    db_root: Path,
    device: str = "auto",
    within_batch: bool = False,
    verify_samples: int = 0,
) -> list[Result]:
    device = resolve_device(device)
    fp, mp = FeatureProfile(), MatchingProfile()
    con = connect(db_root)
    rows = references(con, fp.identity)
    ref_data: list[tuple[object, np.ndarray, np.ndarray]] = []
    for row in rows:
        raw, ts = load_features(Path(row["feature_path"]))
        ref_data.append((row, raw, ts))
    batch_refs: list[tuple[dict, np.ndarray, np.ndarray]] = []
    results: list[Result] = []
    run_id = f"run-{uuid.uuid4().hex}"
    for path in paths:
        try:
            started = time.perf_counter()
            sha = sha256_file(path)
            exact = []
            for row, _, _ in ref_data:
                if row["sha256"] == sha:
                    exact.append(
                        Match(
                            reference_submission_id=row["submission_id"],
                            reference_path=row["path"],
                            reference_sha256=sha,
                            exact_file_match=True,
                        )
                    )
            for row, _, _ in batch_refs:
                if row["sha256"] == sha:
                    exact.append(
                        Match(
                            reference_submission_id=row["submission_id"],
                            reference_path=row["path"],
                            reference_sha256=sha,
                            exact_file_match=True,
                        )
                    )
            if exact:
                warnings = []
                try:
                    duration = probe(path).get("duration_seconds")
                except Exception as exc:
                    duration = None
                    warnings.append(f"Media probing failed after valid exact-file match: {exc}")
                result = _result(
                    run_id,
                    path,
                    sha,
                    duration,
                    device,
                    "exact_duplicate",
                    exact,
                    fp,
                    mp,
                    {"total": time.perf_counter() - started},
                    within_batch,
                    extra_warnings=warnings,
                )
                results.append(result)
                if within_batch:
                    # Preserve aliases while sharing cached descriptors when available.
                    source = next(((r, a, t) for r, a, t in batch_refs if r["sha256"] == sha), None)
                    if source:
                        batch_refs.append(
                            (
                                {
                                    "sha256": sha,
                                    "submission_id": _submission(path),
                                    "path": str(path),
                                },
                                source[1],
                                source[2],
                            )
                        )
                continue
            sha, metadata, q_raw, q_times, details = extract(path, device)
            candidates: list[tuple[object, np.ndarray, np.ndarray]] = []
            combined = ref_data + batch_refs
            ids = candidate_assets(
                q_raw, [(i, np.asarray(x[1])) for i, x in enumerate(combined)], mp.frame_top_k
            )
            candidates.extend(combined[i] for i in ids[:20])
            matches = []
            for row, r_raw, r_times in candidates:
                segs = localize(
                    q_raw[:240],
                    np.asarray(r_raw)[:240],
                    q_times[:240],
                    np.asarray(r_times)[:240],
                    mp,
                    verify_samples,
                )
                if segs:
                    matches.append(
                        Match(
                            reference_submission_id=row["submission_id"],
                            reference_path=row["path"],
                            reference_sha256=row["sha256"],
                            exact_file_match=False,
                            segments=segs,
                        )
                    )
            decision = "suspected_copy" if matches else _negative(q_times)
            results.append(
                _result(
                    run_id,
                    path,
                    sha,
                    metadata.get("duration_seconds"),
                    device,
                    decision,
                    matches,
                    fp,
                    mp,
                    {**details["timings"], "total": time.perf_counter() - started},
                    within_batch,
                )
            )
            if within_batch:
                batch_refs.append(
                    (
                        {"sha256": sha, "submission_id": _submission(path), "path": str(path)},
                        q_raw,
                        q_times,
                    )
                )
        except Exception as exc:
            results.append(
                Result(
                    run_id=run_id,
                    query=QueryInfo(submission_id=_submission(path), path=str(path), sha256=""),
                    execution_status="failed",
                    decision=None,
                    review_required=False,
                    search=SearchInfo(within_batch=within_batch, completed_within_policy=False),
                    feature_profile_sha256=fp.identity,
                    matching_profile_sha256=mp.identity,
                    device=device,
                    errors=[str(exc)],
                )
            )
    return results


def _result(
    run_id: str,
    path: Path,
    sha: str,
    duration: float | None,
    device: str,
    decision: str,
    matches: list[Match],
    fp: FeatureProfile,
    mp: MatchingProfile,
    timings: dict,
    within_batch: bool = False,
    extra_warnings: list[str] | None = None,
) -> Result:
    return Result(
        run_id=run_id,
        query=QueryInfo(
            submission_id=_submission(path), path=str(path), sha256=sha, duration_seconds=duration
        ),
        execution_status="completed",
        decision=decision,
        review_required=decision in {"exact_duplicate", "suspected_copy", "insufficient_evidence"},
        search=SearchInfo(within_batch=within_batch),
        feature_profile_sha256=fp.identity,
        matching_profile_sha256=mp.identity,
        device=device,
        matches=matches,
        warnings=[PROVISIONAL_WARNING, *(extra_warnings or [])],
        stage_timings_seconds=timings,
    )


def _negative(times: np.ndarray) -> str:
    return (
        "no_match_found"
        if len(times) >= MatchingProfile().min_distinct_samples
        else "insufficient_evidence"
    )


def _submission(path: Path) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(path.resolve())))


def _merge_timings(*values: dict[str, float]) -> dict[str, float]:
    keys = set().union(*values)
    return {k: sum(x.get(k, 0.0) for x in values) for k in keys}
