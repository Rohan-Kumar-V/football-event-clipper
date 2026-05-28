from __future__ import annotations

import json
from pathlib import Path

from football_ingest.models import (
    AudioWindow,
    ClipResult,
    GoalCandidate,
    ReplayCandidate,
    SceneCut,
    ScoreboardRead,
)


def write_manifest(
    output_dir: Path,
    source_video: Path,
    reads: list[ScoreboardRead],
    candidates: list[GoalCandidate],
    clips: list[ClipResult],
    dry_run: bool,
    audio_windows: list[AudioWindow] | None = None,
    scene_cuts: list[SceneCut] | None = None,
    replay_candidates: list[ReplayCandidate] | None = None,
    broadcast_text_candidates: list[ReplayCandidate] | None = None,
    stoppage_candidates: list[ReplayCandidate] | None = None,
    skill_candidates: list[ReplayCandidate] | None = None,
    chance_candidates: list[ReplayCandidate] | None = None,
) -> None:
    manifest = {
        "source_video": str(source_video),
        "dry_run": dry_run,
        "ocr_reads": len(reads),
        "parsed_ocr_reads": sum(1 for read in reads if read.parsed),
        "audio_windows": len(audio_windows or []),
        "audio_spikes": sum(1 for window in audio_windows or [] if window.is_spike),
        "scene_cuts": len(scene_cuts or []),
        "replay_candidates": [
            candidate.to_dict() for candidate in replay_candidates or []
        ],
        "broadcast_text_candidates": [
            candidate.to_dict() for candidate in broadcast_text_candidates or []
        ],
        "stoppage_candidates": [
            candidate.to_dict() for candidate in stoppage_candidates or []
        ],
        "skill_candidates": [
            candidate.to_dict() for candidate in skill_candidates or []
        ],
        "chance_candidates": [
            candidate.to_dict() for candidate in chance_candidates or []
        ],
        "goal_candidates": [candidate.to_dict() for candidate in candidates],
        "clips": [clip.to_dict() for clip in clips],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
