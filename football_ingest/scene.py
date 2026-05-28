from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from football_ingest.models import AudioWindow, ReplayCandidate, SceneCut


def detect_scene_cuts(
    video_path: Path,
    output_dir: Path,
    sample_interval_s: float = 0.5,
    diff_threshold: float = 22.0,
    hist_threshold: float = 0.70,
) -> list[SceneCut]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(int(round(fps * sample_interval_s)), 1)
    cuts: list[SceneCut] = []
    previous_gray = None
    previous_hist = None
    previous_hashes: list[tuple[float, int]] = []
    frame_index = 0
    sampled_frames = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    started_at = time.perf_counter()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_step != 0:
            frame_index += 1
            continue

        timestamp_s = frame_index / fps
        gray, hist, hash_value = frame_features(frame)
        sampled_frames += 1
        if sampled_frames == 1 or sampled_frames % 50 == 0:
            progress = f" ({frame_index / total_frames:.0%})" if total_frames else ""
            elapsed = time.perf_counter() - started_at
            print(
                f"[{time.strftime('%H:%M:%S')}] Scene scan sampled {sampled_frames} frames"
                f"{progress}; video time {timestamp_s:.0f}s; cuts {len(cuts)}; elapsed {elapsed:.1f}s",
                flush=True,
            )

        if previous_gray is not None and previous_hist is not None:
            mean_absdiff = float(cv2.absdiff(gray, previous_gray).mean())
            hist_correlation = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_CORREL))
            if mean_absdiff >= diff_threshold and hist_correlation <= hist_threshold:
                repeat_ts, repeat_distance = nearest_repeat_match(
                    timestamp_s,
                    hash_value,
                    previous_hashes,
                )
                cuts.append(
                    SceneCut(
                        timestamp_s=timestamp_s,
                        frame_index=frame_index,
                        mean_absdiff=mean_absdiff,
                        hist_correlation=hist_correlation,
                        hash_value=hash_value,
                        repeat_match_timestamp_s=repeat_ts,
                        repeat_match_distance=repeat_distance,
                    )
                )

        previous_gray = gray
        previous_hist = hist
        previous_hashes.append((timestamp_s, hash_value))
        if len(previous_hashes) > int(240 / sample_interval_s):
            previous_hashes = previous_hashes[-int(240 / sample_interval_s) :]

        frame_index += 1

    cap.release()
    write_scene_cuts(cuts, output_dir)
    return cuts


def detect_replay_candidates(
    cuts: list[SceneCut],
    audio_windows: list[AudioWindow],
    output_dir: Path,
    cluster_window_s: float = 14.0,
    min_cuts_per_cluster: int = 4,
    merge_gap_s: float = 8.0,
    clip_padding_before_s: float = 3.0,
    clip_padding_after_s: float = 5.0,
    near_audio_s: float = 35.0,
    max_candidates: int = 20,
) -> list[ReplayCandidate]:
    clusters = cluster_scene_cuts(cuts, cluster_window_s, min_cuts_per_cluster)
    clusters = merge_clusters(clusters, merge_gap_s)
    candidates: list[ReplayCandidate] = []

    for cluster in clusters:
        start_s = max(cluster[0].timestamp_s - clip_padding_before_s, 0.0)
        end_s = cluster[-1].timestamp_s + clip_padding_after_s
        best_audio = best_audio_window(audio_windows, start_s - near_audio_s, end_s + near_audio_s)
        repeat_matches = [cut for cut in cluster if cut.repeat_match_timestamp_s is not None]
        near_audio_spike = bool(best_audio and best_audio.is_spike)
        visual_repeat = bool(repeat_matches)

        if not near_audio_spike and not visual_repeat:
            continue

        confidence = 0.50
        if near_audio_spike:
            confidence += 0.15
        if visual_repeat:
            confidence += 0.15
        if len(cluster) >= min_cuts_per_cluster + 2:
            confidence += 0.05

        evidence = {
            "scene_cut_cluster": True,
            "cut_count": len(cluster),
            "cut_timestamps_s": [cut.timestamp_s for cut in cluster],
            "mean_absdiff_max": max(cut.mean_absdiff for cut in cluster),
            "hist_correlation_min": min(cut.hist_correlation for cut in cluster),
            "visual_similarity_to_recent_play": visual_repeat,
            "repeat_matches": [
                {
                    "timestamp_s": cut.timestamp_s,
                    "matched_timestamp_s": cut.repeat_match_timestamp_s,
                    "hash_distance": cut.repeat_match_distance,
                }
                for cut in repeat_matches[:5]
            ],
            "near_audio_spike": near_audio_spike,
            "best_audio_spike": best_audio.to_dict() if best_audio else None,
            "review_required_reason": "replay-like cut cluster; not a final event label",
        }
        candidates.append(
            ReplayCandidate(
                event="review_required",
                candidate_type="replay_segment",
                timestamp_s=start_s,
                start_s=start_s,
                end_s=end_s,
                confidence=min(confidence, 0.85),
                evidence=evidence,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.evidence.get("near_audio_spike", False),
            candidate.confidence,
            candidate.evidence.get("cut_count", 0),
        ),
        reverse=True,
    )
    candidates = candidates[:max_candidates]
    candidates.sort(key=lambda candidate: candidate.start_s)
    write_replay_candidates(candidates, output_dir)
    return candidates


def frame_features(frame):
    resized = cv2.resize(frame, (160, 90))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return gray, hist, dhash(gray)


def dhash(gray) -> int:
    small = cv2.resize(gray, (9, 8))
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def nearest_repeat_match(
    timestamp_s: float,
    hash_value: int,
    previous_hashes: list[tuple[float, int]],
    min_age_s: float = 8.0,
    max_age_s: float = 180.0,
    max_distance: int = 8,
) -> tuple[float | None, int | None]:
    best_ts = None
    best_distance = max_distance + 1
    for previous_ts, previous_hash in previous_hashes:
        age = timestamp_s - previous_ts
        if age < min_age_s or age > max_age_s:
            continue
        distance = hamming_distance(hash_value, previous_hash)
        if distance < best_distance:
            best_ts = previous_ts
            best_distance = distance

    if best_ts is None or best_distance > max_distance:
        return None, None
    return best_ts, best_distance


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def cluster_scene_cuts(
    cuts: list[SceneCut],
    cluster_window_s: float,
    min_cuts_per_cluster: int,
) -> list[list[SceneCut]]:
    clusters: list[list[SceneCut]] = []
    index = 0
    while index < len(cuts):
        start = cuts[index].timestamp_s
        cluster = [cut for cut in cuts[index:] if cut.timestamp_s <= start + cluster_window_s]
        if len(cluster) >= min_cuts_per_cluster:
            clusters.append(cluster)
            index += len(cluster)
        else:
            index += 1
    return clusters


def merge_clusters(clusters: list[list[SceneCut]], merge_gap_s: float) -> list[list[SceneCut]]:
    if not clusters:
        return []

    merged = [clusters[0]]
    for cluster in clusters[1:]:
        previous = merged[-1]
        if cluster[0].timestamp_s - previous[-1].timestamp_s <= merge_gap_s:
            merged[-1] = previous + cluster
        else:
            merged.append(cluster)
    return merged


def best_audio_window(
    windows: list[AudioWindow],
    start_s: float,
    end_s: float,
) -> AudioWindow | None:
    nearby = [window for window in windows if start_s <= window.timestamp_s <= end_s]
    return max(nearby, key=lambda window: window.spike_score, default=None)


def write_scene_cuts(cuts: list[SceneCut], output_dir: Path) -> None:
    rows = [cut.to_dict() for cut in cuts]
    pd.DataFrame(rows).to_csv(output_dir / "scene_cuts.csv", index=False)
    with (output_dir / "scene_cuts.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def write_replay_candidates(candidates: list[ReplayCandidate], output_dir: Path) -> None:
    rows = [candidate.to_dict() for candidate in candidates]
    pd.DataFrame(rows).to_csv(output_dir / "replay_candidates.csv", index=False)
    with (output_dir / "replay_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
