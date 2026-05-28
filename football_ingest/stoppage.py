from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_ingest.models import AudioWindow, ReplayCandidate, SceneCut, ScoreboardRead


@dataclass(frozen=True)
class StoppageFrame:
    timestamp_s: float
    frame_index: int
    mean_absdiff: float
    green_ratio: float
    is_low_motion: bool
    is_non_field_view: bool

    def to_dict(self) -> dict:
        return {
            "timestamp_s": self.timestamp_s,
            "frame_index": self.frame_index,
            "mean_absdiff": self.mean_absdiff,
            "green_ratio": self.green_ratio,
            "is_low_motion": self.is_low_motion,
            "is_non_field_view": self.is_non_field_view,
        }


def detect_stoppage_candidates(
    video_path: Path,
    output_dir: Path,
    audio_windows: list[AudioWindow] | None = None,
    scene_cuts: list[SceneCut] | None = None,
    sample_interval_s: float = 1.0,
    low_motion_threshold: float = 7.5,
    non_field_green_threshold: float = 0.18,
    ignore_before_s: float = 0.0,
    min_duration_s: float = 12.0,
    max_duration_s: float = 50.0,
    merge_gap_s: float = 8.0,
    max_candidates: int = 24,
) -> list[ReplayCandidate]:
    frames = analyze_stoppage_frames(
        video_path=video_path,
        sample_interval_s=sample_interval_s,
        low_motion_threshold=low_motion_threshold,
        non_field_green_threshold=non_field_green_threshold,
    )
    write_stoppage_frames(frames, output_dir)

    intervals = build_stoppage_intervals(
        frames=frames,
        sample_interval_s=sample_interval_s,
        min_duration_s=min_duration_s,
        ignore_before_s=ignore_before_s,
        max_duration_s=max_duration_s,
        merge_gap_s=merge_gap_s,
    )
    candidates = [
        interval_to_candidate(
            interval=interval,
            audio_windows=audio_windows or [],
            scene_cuts=scene_cuts or [],
        )
        for interval in intervals
    ]
    candidates = rank_stoppage_candidates(candidates)[:max_candidates]
    candidates.sort(key=lambda candidate: candidate.start_s)
    write_stoppage_candidates(candidates, output_dir)
    return candidates


def infer_match_start_from_reads(
    reads: list[ScoreboardRead],
    preroll_s: float = 10.0,
) -> float:
    parsed_timestamps = [read.timestamp_s for read in reads if read.parsed]
    if not parsed_timestamps:
        return 0.0
    return max(min(parsed_timestamps) - preroll_s, 0.0)


def analyze_stoppage_frames(
    video_path: Path,
    sample_interval_s: float,
    low_motion_threshold: float,
    non_field_green_threshold: float,
) -> list[StoppageFrame]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_s = frame_count / fps if frame_count else 0.0
    previous_gray = None
    rows: list[StoppageFrame] = []
    timestamp_s = 0.0
    started_at = time.perf_counter()
    sample_index = 0

    while timestamp_s <= duration_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break

        frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        resized = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        diff = float(cv2.absdiff(gray, previous_gray).mean()) if previous_gray is not None else 999.0
        green_ratio = field_green_ratio(resized)

        rows.append(
            StoppageFrame(
                timestamp_s=timestamp_s,
                frame_index=frame_index,
                mean_absdiff=diff,
                green_ratio=green_ratio,
                is_low_motion=diff <= low_motion_threshold,
                is_non_field_view=green_ratio <= non_field_green_threshold,
            )
        )
        sample_index += 1
        if sample_index == 1 or sample_index % 300 == 0:
            progress = f" ({timestamp_s / duration_s:.0%})" if duration_s else ""
            elapsed = time.perf_counter() - started_at
            print(
                f"[{time.strftime('%H:%M:%S')}] Stoppage scan sampled {sample_index} frames"
                f"{progress}; video time {timestamp_s:.0f}s; elapsed {elapsed:.1f}s",
                flush=True,
            )
        previous_gray = gray
        timestamp_s += sample_interval_s

    cap.release()
    return rows


def field_green_ratio(frame) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 35, 35])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(mask.mean() / 255.0)


def build_stoppage_intervals(
    frames: list[StoppageFrame],
    sample_interval_s: float,
    min_duration_s: float,
    ignore_before_s: float,
    max_duration_s: float,
    merge_gap_s: float,
) -> list[list[StoppageFrame]]:
    raw: list[list[StoppageFrame]] = []
    current: list[StoppageFrame] = []

    for frame in frames:
        if frame.timestamp_s < ignore_before_s:
            if current:
                raw.append(current)
                current = []
            continue

        is_candidate_frame = frame.is_non_field_view or (
            frame.is_low_motion and frame.green_ratio <= 0.45
        )
        if is_candidate_frame:
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
    split: list[list[StoppageFrame]] = []
    for interval in merged:
        chunk: list[StoppageFrame] = []
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


def interval_duration(interval: list[StoppageFrame], sample_interval_s: float) -> float:
    return interval[-1].timestamp_s - interval[0].timestamp_s + sample_interval_s


def interval_to_candidate(
    interval: list[StoppageFrame],
    audio_windows: list[AudioWindow],
    scene_cuts: list[SceneCut],
) -> ReplayCandidate:
    start_s = max(interval[0].timestamp_s - 4.0, 0.0)
    end_s = interval[-1].timestamp_s + 6.0
    low_motion_ratio = sum(frame.is_low_motion for frame in interval) / len(interval)
    non_field_ratio = sum(frame.is_non_field_view for frame in interval) / len(interval)
    nearby_cuts = [
        cut
        for cut in scene_cuts
        if start_s - 8.0 <= cut.timestamp_s <= end_s + 8.0
    ]
    nearby_audio = [
        window
        for window in audio_windows
        if start_s - 10.0 <= window.timestamp_s <= end_s + 10.0
    ]
    best_audio = max(nearby_audio, key=lambda item: item.spike_score, default=None)
    label_hints = infer_stoppage_label_hints(low_motion_ratio, non_field_ratio, nearby_cuts)

    confidence = 0.45
    if low_motion_ratio >= 0.65:
        confidence += 0.10
    if non_field_ratio >= 0.50:
        confidence += 0.12
    if len(nearby_cuts) >= 3:
        confidence += 0.08
    if best_audio and not best_audio.is_spike:
        confidence += 0.03

    return ReplayCandidate(
        event="review_required",
        candidate_type="stoppage_segment",
        timestamp_s=start_s,
        start_s=start_s,
        end_s=end_s,
        confidence=min(confidence, 0.78),
        evidence={
            "stoppage_detected": True,
            "duration_s": end_s - start_s,
            "sample_count": len(interval),
            "low_motion_ratio": low_motion_ratio,
            "non_field_view_ratio": non_field_ratio,
            "scene_cut_count_nearby": len(nearby_cuts),
            "best_audio": best_audio.to_dict() if best_audio else None,
            "label_hints": label_hints,
            "review_required_reason": "stoppage-like segment; possible card, substitution, injury, referee discussion, or argument",
        },
    )


def infer_stoppage_label_hints(
    low_motion_ratio: float,
    non_field_ratio: float,
    nearby_cuts: list[SceneCut],
) -> list[str]:
    hints = [
        "yellow_card",
        "red_card",
        "second_yellow_red",
        "substitution",
        "injury_stoppage",
        "medical_treatment",
        "referee_discussion",
        "player_argument",
    ]
    if non_field_ratio >= 0.65 and len(nearby_cuts) >= 3:
        hints.extend(["manager_reaction", "crowd_reaction"])
    if low_motion_ratio >= 0.80:
        hints.append("free_kick")
    return sorted(set(hints))


def rank_stoppage_candidates(candidates: list[ReplayCandidate]) -> list[ReplayCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.confidence,
            candidate.evidence.get("non_field_view_ratio", 0),
            candidate.evidence.get("scene_cut_count_nearby", 0),
        ),
        reverse=True,
    )


def write_stoppage_frames(frames: list[StoppageFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [frame.to_dict() for frame in frames]
    pd.DataFrame(rows).to_csv(output_dir / "stoppage_frames.csv", index=False)


def write_stoppage_candidates(candidates: list[ReplayCandidate], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [candidate.to_dict() for candidate in candidates]
    pd.DataFrame(rows).to_csv(output_dir / "stoppage_candidates.csv", index=False)
    with (output_dir / "stoppage_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
