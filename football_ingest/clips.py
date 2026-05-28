from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from football_ingest.models import ClipResult, GoalCandidate, ReplayCandidate


def clip_goal_candidates(
    video_path: Path,
    output_dir: Path,
    candidates: list[GoalCandidate],
    pre_seconds: float = 20.0,
    post_seconds: float = 35.0,
    dry_run: bool = False,
) -> list[ClipResult]:
    results: list[ClipResult] = []

    for index, candidate in enumerate(candidates, start=1):
        start_s = max(candidate.timestamp_s - pre_seconds, 0.0)
        end_s = candidate.timestamp_s + post_seconds
        event_dir = output_dir / candidate.event
        event_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{index:06d}_{candidate.event}_{int(candidate.timestamp_s)}s"
        output_video = event_dir / f"{stem}.mp4"
        metadata_path = event_dir / f"{stem}.json"

        result = ClipResult(
            event=candidate.event,
            source_video=video_path,
            output_video=output_video,
            metadata_path=metadata_path,
            start_s=start_s,
            end_s=end_s,
            confidence=candidate.confidence,
            evidence=candidate.evidence,
        )

        if not dry_run:
            cut_clip(video_path, output_video, start_s, end_s)

        write_clip_metadata(result)
        results.append(result)

    return results


def clip_goal_celebrations(
    video_path: Path,
    output_dir: Path,
    candidates: list[GoalCandidate],
    pre_seconds: float = 4.0,
    post_seconds: float = 32.0,
    dry_run: bool = False,
) -> list[ClipResult]:
    results: list[ClipResult] = []

    for index, candidate in enumerate(candidates, start=1):
        start_s = max(candidate.timestamp_s - pre_seconds, 0.0)
        end_s = candidate.timestamp_s + post_seconds
        event = "goal_celebration"
        event_dir = output_dir / event
        event_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{index:06d}_{event}_{int(candidate.timestamp_s)}s"
        output_video = event_dir / f"{stem}.mp4"
        metadata_path = event_dir / f"{stem}.json"
        evidence = dict(candidate.evidence)
        evidence["derived_from_event"] = "goal"
        evidence["clip_policy"] = {
            "reason": "scoreboard-confirmed goal aftermath",
            "pre_seconds": pre_seconds,
            "post_seconds": post_seconds,
        }

        result = ClipResult(
            event=event,
            source_video=video_path,
            output_video=output_video,
            metadata_path=metadata_path,
            start_s=start_s,
            end_s=end_s,
            confidence=candidate.confidence,
            evidence=evidence,
        )

        if not dry_run:
            cut_clip(video_path, output_video, start_s, end_s)

        write_clip_metadata(result)
        results.append(result)

    return results


def clip_replay_candidates(
    video_path: Path,
    output_dir: Path,
    candidates: list[ReplayCandidate],
    dry_run: bool = False,
) -> list[ClipResult]:
    results: list[ClipResult] = []

    for index, candidate in enumerate(candidates, start=1):
        event_dir = output_dir / candidate.event
        event_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{index:06d}_{candidate.candidate_type}_{int(candidate.start_s)}s"
        output_video = event_dir / f"{stem}.mp4"
        metadata_path = event_dir / f"{stem}.json"
        evidence = dict(candidate.evidence)
        evidence["candidate_type"] = candidate.candidate_type

        result = ClipResult(
            event=candidate.event,
            source_video=video_path,
            output_video=output_video,
            metadata_path=metadata_path,
            start_s=candidate.start_s,
            end_s=candidate.end_s,
            confidence=candidate.confidence,
            evidence=evidence,
        )

        if not dry_run:
            cut_clip(video_path, output_video, candidate.start_s, candidate.end_s)

        write_clip_metadata(result)
        results.append(result)

    return results


def cut_clip(video_path: Path, output_path: Path, start_s: float, end_s: float) -> None:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg or imageio-ffmpeg to write clips."
        )

    duration = max(end_s - start_s, 0.1)
    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed while creating {output_path}:\n{completed.stderr.strip()}"
        )


def find_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    return imageio_ffmpeg.get_ffmpeg_exe()


def write_clip_metadata(result: ClipResult) -> None:
    result.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with result.metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)
