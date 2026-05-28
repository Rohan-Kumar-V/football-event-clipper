from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_ingest.models import AudioWindow, GoalCandidate, ReplayCandidate, SceneCut
from football_ingest.skills import (
    central_motion,
    field_green_ratio,
    frame_edge_density,
    motion_centroid,
    suppress_overlapping_candidates,
)


@dataclass(frozen=True)
class ChanceFrame:
    timestamp_s: float
    frame_index: int
    mean_absdiff: float
    green_ratio: float
    central_motion_ratio: float
    edge_density: float
    white_ratio: float
    motion_x: float | None
    motion_y: float | None
    pressure_score: float
    is_chance_like: bool

    def to_dict(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s,
            "frame_index": self.frame_index,
            "mean_absdiff": self.mean_absdiff,
            "green_ratio": self.green_ratio,
            "central_motion_ratio": self.central_motion_ratio,
            "edge_density": self.edge_density,
            "white_ratio": self.white_ratio,
            "motion_x": self.motion_x,
            "motion_y": self.motion_y,
            "pressure_score": self.pressure_score,
            "is_chance_like": self.is_chance_like,
        }


def detect_chance_candidates(
    video_path: Path,
    output_dir: Path,
    audio_windows: list[AudioWindow] | None = None,
    scene_cuts: list[SceneCut] | None = None,
    goal_candidates: list[GoalCandidate] | None = None,
    sample_interval_s: float = 2.0,
    min_duration_s: float = 6.0,
    max_duration_s: float = 24.0,
    merge_gap_s: float = 4.0,
    ignore_before_s: float = 0.0,
    goal_exclusion_s: float = 55.0,
    max_candidates: int = 18,
) -> list[ReplayCandidate]:
    frames = analyze_chance_frames(
        video_path=video_path,
        sample_interval_s=sample_interval_s,
        ignore_before_s=ignore_before_s,
    )
    write_chance_frames(frames, output_dir)

    intervals = build_chance_intervals(
        frames=frames,
        sample_interval_s=sample_interval_s,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        merge_gap_s=merge_gap_s,
    )
    goal_times = [candidate.timestamp_s for candidate in goal_candidates or []]
    candidates = [
        interval_to_candidate(
            interval=interval,
            sample_interval_s=sample_interval_s,
            audio_windows=audio_windows or [],
            scene_cuts=scene_cuts or [],
        )
        for interval in intervals
        if not overlaps_goal(interval, sample_interval_s, goal_times, goal_exclusion_s)
    ]
    candidates = suppress_overlapping_candidates(
        rank_chance_candidates(candidates),
        max_overlap_ratio=0.20,
    )[:max_candidates]
    candidates.sort(key=lambda candidate: candidate.start_s)
    write_chance_candidates(candidates, output_dir)
    return candidates


def analyze_chance_frames(
    video_path: Path,
    sample_interval_s: float,
    ignore_before_s: float,
) -> list[ChanceFrame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(int(round(fps * sample_interval_s)), 1)
    frame_index = max(int(round(ignore_before_s * fps)), 0)
    previous_gray = None
    rows: list[ChanceFrame] = []
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
        white_ratio = field_white_ratio(resized)

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

        pressure_score = chance_pressure_score(
            mean_absdiff=mean_absdiff,
            green_ratio=green_ratio,
            central_motion_ratio=central_motion_ratio,
            edge_density=edge_density,
            white_ratio=white_ratio,
        )
        is_chance_like = (
            0.20 <= green_ratio <= 0.82
            and 3.5 <= mean_absdiff <= 38.0
            and central_motion_ratio >= 0.34
            and edge_density >= 0.035
            and pressure_score >= 0.46
        )
        rows.append(
            ChanceFrame(
                timestamp_s=timestamp_s,
                frame_index=actual_frame_index,
                mean_absdiff=mean_absdiff,
                green_ratio=green_ratio,
                central_motion_ratio=central_motion_ratio,
                edge_density=edge_density,
                white_ratio=white_ratio,
                motion_x=motion_x,
                motion_y=motion_y,
                pressure_score=pressure_score,
                is_chance_like=is_chance_like,
            )
        )
        sample_index += 1
        if sample_index == 1 or sample_index % 200 == 0:
            progress = f" ({frame_index / frame_count:.0%})" if frame_count else ""
            elapsed = time.perf_counter() - started_at
            print(
                f"[{time.strftime('%H:%M:%S')}] Chance scan sampled {sample_index} frames"
                f"{progress}; video time {timestamp_s:.0f}s; elapsed {elapsed:.1f}s",
                flush=True,
            )
        previous_gray = gray
        frame_index += frame_step

    cap.release()
    return rows


def field_white_ratio(frame) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 1] <= 55) & (hsv[:, :, 2] >= 150)
    return float(mask.mean())


def chance_pressure_score(
    mean_absdiff: float,
    green_ratio: float,
    central_motion_ratio: float,
    edge_density: float,
    white_ratio: float,
) -> float:
    motion_score = min(mean_absdiff / 20.0, 1.0)
    pitch_score = 1.0 - min(abs(green_ratio - 0.48) / 0.38, 1.0)
    central_score = min(central_motion_ratio / 0.62, 1.0)
    line_score = min((edge_density * 0.75 + white_ratio * 0.25) / 0.11, 1.0)
    return (
        0.34 * motion_score
        + 0.26 * central_score
        + 0.22 * pitch_score
        + 0.18 * line_score
    )


def build_chance_intervals(
    frames: list[ChanceFrame],
    sample_interval_s: float,
    min_duration_s: float,
    max_duration_s: float,
    merge_gap_s: float,
) -> list[list[ChanceFrame]]:
    raw: list[list[ChanceFrame]] = []
    current: list[ChanceFrame] = []

    for frame in frames:
        if frame.is_chance_like:
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

    split: list[list[ChanceFrame]] = []
    for interval in merged:
        chunk: list[ChanceFrame] = []
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
    interval: list[ChanceFrame],
    sample_interval_s: float,
    audio_windows: list[AudioWindow],
    scene_cuts: list[SceneCut],
) -> ReplayCandidate:
    raw_start_s = interval[0].timestamp_s
    raw_end_s = interval[-1].timestamp_s + sample_interval_s
    start_s = max(raw_start_s - 6.0, 0.0)
    end_s = raw_end_s + 8.0
    pressure_mean = sum(frame.pressure_score for frame in interval) / len(interval)
    motion_mean = sum(frame.mean_absdiff for frame in interval) / len(interval)
    white_ratio_mean = sum(frame.white_ratio for frame in interval) / len(interval)
    next_cuts = [
        cut
        for cut in scene_cuts
        if raw_end_s <= cut.timestamp_s <= raw_end_s + 25.0
    ]
    nearby_cuts = [
        cut
        for cut in scene_cuts
        if start_s - 6.0 <= cut.timestamp_s <= end_s + 15.0
    ]
    nearby_audio = [
        window
        for window in audio_windows
        if start_s - 8.0 <= window.timestamp_s <= end_s + 18.0
    ]
    best_audio = max(nearby_audio, key=lambda item: item.spike_score, default=None)

    confidence = 0.42 + pressure_mean * 0.20
    if len(next_cuts) >= 2:
        confidence += 0.10
    if len(nearby_cuts) >= 4:
        confidence += 0.05
    if best_audio and best_audio.is_spike:
        confidence += 0.08
    if white_ratio_mean >= 0.12:
        confidence += 0.04

    return ReplayCandidate(
        event="review_required",
        candidate_type="chance_segment",
        timestamp_s=start_s,
        start_s=start_s,
        end_s=end_s,
        confidence=min(confidence, 0.84),
        evidence={
            "chance_candidate_detected": True,
            "duration_s": end_s - start_s,
            "sample_count": len(interval),
            "pressure_score_mean": pressure_mean,
            "motion_mean_absdiff": motion_mean,
            "white_ratio_mean": white_ratio_mean,
            "scene_cut_count_nearby": len(nearby_cuts),
            "scene_cut_count_after": len(next_cuts),
            "best_audio": best_audio.to_dict() if best_audio else None,
            "label_hints": [
                "big_chance",
                "blocked_shot",
                "counter_attack",
                "cross",
                "goalkeeper_save",
                "key_pass",
                "shot_off_target",
                "shot_on_target",
                "through_ball",
                "woodwork_hit",
            ],
            "review_required_reason": "attacking-pressure segment; possible shot, save, blocked shot, cross, key pass, or big chance",
        },
    )


def overlaps_goal(
    interval: list[ChanceFrame],
    sample_interval_s: float,
    goal_times: list[float],
    goal_exclusion_s: float,
) -> bool:
    if not goal_times:
        return False
    start_s = interval[0].timestamp_s
    end_s = interval[-1].timestamp_s + sample_interval_s
    return any(start_s - goal_exclusion_s <= goal <= end_s + goal_exclusion_s for goal in goal_times)


def rank_chance_candidates(candidates: list[ReplayCandidate]) -> list[ReplayCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.confidence,
            candidate.evidence.get("scene_cut_count_after", 0),
            candidate.evidence.get("pressure_score_mean", 0),
        ),
        reverse=True,
    )


def interval_duration(interval: list[ChanceFrame], sample_interval_s: float) -> float:
    return interval[-1].timestamp_s - interval[0].timestamp_s + sample_interval_s


def write_chance_frames(frames: list[ChanceFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [frame.to_dict() for frame in frames]
    pd.DataFrame(rows).to_csv(output_dir / "chance_frames.csv", index=False)


def write_chance_candidates(candidates: list[ReplayCandidate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [candidate.to_dict() for candidate in candidates]
    pd.DataFrame(rows).to_csv(output_dir / "chance_candidates.csv", index=False)
    with (output_dir / "chance_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
