from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from football_ingest.clips import find_ffmpeg
from football_ingest.models import AudioWindow, GoalCandidate


def analyze_audio_energy(
    video_path: Path,
    output_dir: Path,
    sample_rate: int = 16_000,
    window_seconds: float = 1.0,
    hop_seconds: float = 0.5,
    spike_threshold: float = 3.0,
) -> list[AudioWindow]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found. Install ffmpeg or imageio-ffmpeg.")

    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "s16le",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while extracting audio:\n{completed.stderr.decode(errors='ignore')}"
        )

    samples = np.frombuffer(completed.stdout, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        windows: list[AudioWindow] = []
        write_audio_timeline(windows, output_dir)
        return windows

    samples /= 32768.0
    frame_len = max(int(sample_rate * window_seconds), 1)
    hop_len = max(int(sample_rate * hop_seconds), 1)

    raw_windows: list[tuple[float, float, float]] = []
    for start in range(0, max(samples.size - frame_len + 1, 1), hop_len):
        chunk = samples[start : start + frame_len]
        if chunk.size == 0:
            continue
        rms = float(np.sqrt(np.mean(np.square(chunk))))
        db = float(20.0 * np.log10(rms + 1e-8))
        raw_windows.append((start / sample_rate, rms, db))

    db_values = np.array([window[2] for window in raw_windows], dtype=np.float32)
    median_db = float(np.median(db_values))
    mad = float(np.median(np.abs(db_values - median_db)))
    robust_scale = max(mad * 1.4826, 1.0)
    high_db = float(np.percentile(db_values, 90))

    windows = [
        AudioWindow(
            timestamp_s=timestamp_s,
            rms=rms,
            db=db,
            spike_score=max((db - median_db) / robust_scale, 0.0),
            is_spike=(db >= high_db and (db - median_db) / robust_scale >= spike_threshold),
        )
        for timestamp_s, rms, db in raw_windows
    ]
    write_audio_timeline(windows, output_dir)
    return windows


def attach_audio_evidence(
    candidates: list[GoalCandidate],
    windows: list[AudioWindow],
    search_before_s: float = 45.0,
    search_after_s: float = 20.0,
) -> list[GoalCandidate]:
    enriched: list[GoalCandidate] = []

    for candidate in candidates:
        nearby = [
            window
            for window in windows
            if candidate.timestamp_s - search_before_s
            <= window.timestamp_s
            <= candidate.timestamp_s + search_after_s
        ]
        best = max(nearby, key=lambda window: window.spike_score, default=None)
        evidence = dict(candidate.evidence)
        evidence["audio"] = {
            "searched_before_s": search_before_s,
            "searched_after_s": search_after_s,
            "nearby_window_count": len(nearby),
            "best_spike": best.to_dict() if best else None,
            "crowd_spike_near_score_change": bool(best and best.is_spike),
        }
        enriched.append(replace(candidate, evidence=evidence))

    return enriched


def write_audio_timeline(windows: list[AudioWindow], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [window.to_dict() for window in windows]
    pd.DataFrame(rows).to_csv(output_dir / "audio_energy.csv", index=False)
    with (output_dir / "audio_energy.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
