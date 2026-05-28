from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_ingest.models import AudioWindow, ReplayCandidate, SceneCut


@dataclass(frozen=True)
class SkillFrame:
    timestamp_s: float
    frame_index: int
    mean_absdiff: float
    green_ratio: float
    central_motion_ratio: float
    edge_density: float
    motion_x: float | None
    motion_y: float | None
    activity_score: float
    is_skill_like: bool

    def to_dict(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s,
            "frame_index": self.frame_index,
            "mean_absdiff": self.mean_absdiff,
            "green_ratio": self.green_ratio,
            "central_motion_ratio": self.central_motion_ratio,
            "edge_density": self.edge_density,
            "motion_x": self.motion_x,
            "motion_y": self.motion_y,
            "activity_score": self.activity_score,
            "is_skill_like": self.is_skill_like,
        }


def detect_skill_candidates(
    video_path: Path,
    output_dir: Path,
    audio_windows: list[AudioWindow] | None = None,
    scene_cuts: list[SceneCut] | None = None,
    sample_interval_s: float = 2.0,
    min_duration_s: float = 6.0,
    max_duration_s: float = 18.0,
    merge_gap_s: float = 2.0,
    ignore_before_s: float = 0.0,
    max_candidates: int = 20,
) -> list[ReplayCandidate]:
    frames = analyze_skill_frames(
        video_path=video_path,
        sample_interval_s=sample_interval_s,
        ignore_before_s=ignore_before_s,
    )
    write_skill_frames(frames, output_dir)

    intervals = build_skill_intervals(
        frames=frames,
        sample_interval_s=sample_interval_s,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        merge_gap_s=merge_gap_s,
    )
    candidates = [
        interval_to_candidate(
            interval=interval,
            sample_interval_s=sample_interval_s,
            audio_windows=audio_windows or [],
            scene_cuts=scene_cuts or [],
        )
        for interval in intervals
    ]
    candidates = suppress_overlapping_candidates(
        rank_skill_candidates(candidates),
        max_overlap_ratio=0.20,
    )[:max_candidates]
    candidates.sort(key=lambda candidate: candidate.start_s)
    write_skill_candidates(candidates, output_dir)
    return candidates


def analyze_skill_frames(
    video_path: Path,
    sample_interval_s: float,
    ignore_before_s: float,
) -> list[SkillFrame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(int(round(fps * sample_interval_s)), 1)
    start_frame = max(int(round(ignore_before_s * fps)), 0)

    previous_gray = None
    rows: list[SkillFrame] = []
    frame_index = start_frame
    started_at = time.perf_counter()
    sample_index = 0

    while frame_index < frame_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break

        actual_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        timestamp_s = actual_frame_index / fps
        resized = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        green_ratio = field_green_ratio(resized)
        edge_density = frame_edge_density(gray)

        if previous_gray is None:
            mean_absdiff = 0.0
            central_motion_ratio = 0.0
            motion_x = None
            motion_y = None
        else:
            diff = cv2.absdiff(gray, previous_gray)
            mean_absdiff = float(diff.mean())
            central_motion_ratio = central_motion(diff)
            motion_x, motion_y = motion_centroid(diff)

        activity_score = skill_activity_score(
            mean_absdiff=mean_absdiff,
            green_ratio=green_ratio,
            central_motion_ratio=central_motion_ratio,
            edge_density=edge_density,
        )
        is_skill_like = (
            0.18 <= green_ratio <= 0.72
            and 4.5 <= mean_absdiff <= 34.0
            and central_motion_ratio >= 0.42
            and edge_density >= 0.045
            and activity_score >= 0.48
        )
        rows.append(
            SkillFrame(
                timestamp_s=timestamp_s,
                frame_index=actual_frame_index,
                mean_absdiff=mean_absdiff,
                green_ratio=green_ratio,
                central_motion_ratio=central_motion_ratio,
                edge_density=edge_density,
                motion_x=motion_x,
                motion_y=motion_y,
                activity_score=activity_score,
                is_skill_like=is_skill_like,
            )
        )
        sample_index += 1
        if sample_index == 1 or sample_index % 200 == 0:
            progress = f" ({frame_index / frame_count:.0%})" if frame_count else ""
            elapsed = time.perf_counter() - started_at
            print(
                f"[{time.strftime('%H:%M:%S')}] Skill scan sampled {sample_index} frames"
                f"{progress}; video time {timestamp_s:.0f}s; elapsed {elapsed:.1f}s",
                flush=True,
            )
        previous_gray = gray
        frame_index += frame_step

    cap.release()
    return rows


def field_green_ratio(frame) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 35, 35])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(mask.mean() / 255.0)


def frame_edge_density(gray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float(edges.mean() / 255.0)


def central_motion(diff) -> float:
    h, w = diff.shape
    full_motion = float(diff.sum())
    if full_motion <= 0:
        return 0.0
    central = diff[int(h * 0.18) : int(h * 0.84), int(w * 0.18) : int(w * 0.82)]
    return float(central.sum() / full_motion)


def motion_centroid(diff, threshold: int = 18) -> tuple[float | None, float | None]:
    mask = diff > threshold
    if int(mask.sum()) < 12:
        return None, None
    y_indices, x_indices = np.nonzero(mask)
    h, w = diff.shape
    return float(x_indices.mean() / max(w - 1, 1)), float(y_indices.mean() / max(h - 1, 1))


def skill_activity_score(
    mean_absdiff: float,
    green_ratio: float,
    central_motion_ratio: float,
    edge_density: float,
) -> float:
    motion_score = min(mean_absdiff / 18.0, 1.0)
    pitch_score = 1.0 - min(abs(green_ratio - 0.45) / 0.35, 1.0)
    central_score = min(central_motion_ratio / 0.65, 1.0)
    edge_score = min(edge_density / 0.12, 1.0)
    return (
        0.34 * motion_score
        + 0.24 * central_score
        + 0.22 * pitch_score
        + 0.20 * edge_score
    )


def build_skill_intervals(
    frames: list[SkillFrame],
    sample_interval_s: float,
    min_duration_s: float,
    max_duration_s: float,
    merge_gap_s: float,
) -> list[list[SkillFrame]]:
    raw: list[list[SkillFrame]] = []
    current: list[SkillFrame] = []

    for frame in frames:
        if frame.is_skill_like:
            current.append(frame)
        elif current:
            raw.append(current)
            current = []
    if current:
        raw.append(current)

    intervals = [
        interval
        for interval in raw
        if interval_duration(interval, sample_interval_s) >= min_duration_s
    ]
    if not intervals:
        return []

    merged = [intervals[0]]
    for interval in intervals[1:]:
        previous = merged[-1]
        if interval[0].timestamp_s - previous[-1].timestamp_s <= merge_gap_s:
            merged[-1] = previous + interval
        else:
            merged.append(interval)

    split: list[list[SkillFrame]] = []
    for interval in merged:
        chunk: list[SkillFrame] = []
        chunk_start = interval[0].timestamp_s
        for frame in interval:
            if frame.timestamp_s - chunk_start > max_duration_s and chunk:
                if interval_duration(chunk, sample_interval_s) >= min_duration_s:
                    split.append(chunk)
                chunk = []
                chunk_start = frame.timestamp_s
            chunk.append(frame)
        if chunk and interval_duration(chunk, sample_interval_s) >= min_duration_s:
            split.append(chunk)
    return split


def interval_to_candidate(
    interval: list[SkillFrame],
    sample_interval_s: float,
    audio_windows: list[AudioWindow],
    scene_cuts: list[SceneCut],
) -> ReplayCandidate:
    raw_start_s = interval[0].timestamp_s
    raw_end_s = interval[-1].timestamp_s + sample_interval_s
    start_s = max(raw_start_s - 3.0, 0.0)
    end_s = raw_end_s + 5.0
    activity_mean = sum(frame.activity_score for frame in interval) / len(interval)
    motion_mean = sum(frame.mean_absdiff for frame in interval) / len(interval)
    direction_changes = count_motion_direction_changes(interval)
    nearby_cuts = [
        cut
        for cut in scene_cuts
        if start_s - 8.0 <= cut.timestamp_s <= end_s + 12.0
    ]
    nearby_audio = [
        window
        for window in audio_windows
        if start_s - 8.0 <= window.timestamp_s <= end_s + 12.0
    ]
    best_audio = max(nearby_audio, key=lambda item: item.spike_score, default=None)

    confidence = 0.45 + activity_mean * 0.20
    if direction_changes >= 2:
        confidence += 0.10
    if len(nearby_cuts) >= 2:
        confidence += 0.06
    if best_audio and best_audio.is_spike:
        confidence += 0.06

    return ReplayCandidate(
        event="review_required",
        candidate_type="skill_segment",
        timestamp_s=start_s,
        start_s=start_s,
        end_s=end_s,
        confidence=min(confidence, 0.82),
        evidence={
            "skill_candidate_detected": True,
            "duration_s": end_s - start_s,
            "sample_count": len(interval),
            "activity_score_mean": activity_mean,
            "motion_mean_absdiff": motion_mean,
            "direction_change_count": direction_changes,
            "scene_cut_count_nearby": len(nearby_cuts),
            "best_audio": best_audio.to_dict() if best_audio else None,
            "label_hints": [
                "aerial_duel",
                "foul",
                "nutmeg",
                "skill_dribble",
                "solo_run",
                "tackle",
                "trick",
            ],
            "review_required_reason": "motion-rich on-pitch segment; possible skill, dribble, trick, duel, tackle, or foul",
        },
    )


def count_motion_direction_changes(interval: list[SkillFrame]) -> int:
    centroids = [
        (frame.motion_x, frame.motion_y)
        for frame in interval
        if frame.motion_x is not None and frame.motion_y is not None
    ]
    if len(centroids) < 3:
        return 0

    changes = 0
    previous_vector = None
    for previous, current in zip(centroids, centroids[1:]):
        vector = (current[0] - previous[0], current[1] - previous[1])
        magnitude = (vector[0] ** 2 + vector[1] ** 2) ** 0.5
        if magnitude < 0.015:
            continue
        if previous_vector is not None:
            dot = previous_vector[0] * vector[0] + previous_vector[1] * vector[1]
            if dot < 0:
                changes += 1
        previous_vector = vector
    return changes


def rank_skill_candidates(candidates: list[ReplayCandidate]) -> list[ReplayCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.confidence,
            candidate.evidence.get("direction_change_count", 0),
            candidate.evidence.get("activity_score_mean", 0),
        ),
        reverse=True,
    )


def suppress_overlapping_candidates(
    candidates: list[ReplayCandidate],
    max_overlap_ratio: float,
) -> list[ReplayCandidate]:
    kept: list[ReplayCandidate] = []
    for candidate in candidates:
        if any(
            overlap_ratio(candidate, existing) > max_overlap_ratio
            for existing in kept
        ):
            continue
        kept.append(candidate)
    return kept


def overlap_ratio(left: ReplayCandidate, right: ReplayCandidate) -> float:
    overlap = max(0.0, min(left.end_s, right.end_s) - max(left.start_s, right.start_s))
    if overlap <= 0:
        return 0.0
    shorter = min(left.end_s - left.start_s, right.end_s - right.start_s)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def interval_duration(interval: list[SkillFrame], sample_interval_s: float) -> float:
    return interval[-1].timestamp_s - interval[0].timestamp_s + sample_interval_s


def write_skill_frames(frames: list[SkillFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [frame.to_dict() for frame in frames]
    pd.DataFrame(rows).to_csv(output_dir / "skill_frames.csv", index=False)


def write_skill_candidates(candidates: list[ReplayCandidate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [candidate.to_dict() for candidate in candidates]
    pd.DataFrame(rows).to_csv(output_dir / "skill_candidates.csv", index=False)
    with (output_dir / "skill_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
