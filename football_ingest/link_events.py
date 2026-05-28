from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LinkedClip:
    path: str
    role: str
    label: str
    confidence: float
    start_s: float | None = None
    end_s: float | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "label": self.label,
            "confidence": self.confidence,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "evidence": self.evidence,
        }


@dataclass
class LinkedEvent:
    event_id: str
    event_type: str
    canonical_timestamp_s: float
    canonical_clip: str | None
    confidence: float
    score_change: dict[str, Any] | None = None
    clips: list[LinkedClip] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "canonical_timestamp_s": self.canonical_timestamp_s,
            "canonical_clip": self.canonical_clip,
            "confidence": self.confidence,
            "score_change": self.score_change,
            "clips": [clip.to_dict() for clip in self.clips],
        }


def link_events(
    output_dir: Path,
    goal_link_window_s: float = 90.0,
) -> list[LinkedEvent]:
    manifest = read_json(output_dir / "manifest.json")
    classifications = read_json(
        output_dir / "validation" / "classification_manifest.json",
        default=[],
    )

    events = build_goal_events(output_dir, manifest)
    attach_goal_celebrations(output_dir, manifest, events, goal_link_window_s)
    attach_classified_clips(classifications, events, goal_link_window_s)
    attach_standalone_classifications(classifications, events)

    events.sort(key=lambda event: event.canonical_timestamp_s)
    write_linked_events(output_dir, events)
    return events


def build_goal_events(output_dir: Path, manifest: dict[str, Any]) -> list[LinkedEvent]:
    goal_candidates = manifest.get("goal_candidates", [])
    clips = manifest.get("clips", [])
    events: list[LinkedEvent] = []

    for index, candidate in enumerate(goal_candidates, start=1):
        timestamp_s = float(candidate["timestamp_s"])
        canonical_clip = find_clip_for_event(clips, "goal", timestamp_s)
        score_change = {
            "previous_score": candidate.get("previous_score"),
            "new_score": candidate.get("new_score"),
            "scoring_side": candidate.get("scoring_side"),
        }
        event = LinkedEvent(
            event_id=f"goal_{index:04d}_{int(timestamp_s)}s",
            event_type="goal",
            canonical_timestamp_s=timestamp_s,
            canonical_clip=canonical_clip,
            confidence=float(candidate.get("confidence", 0.0)),
            score_change=score_change,
        )
        if canonical_clip:
            event.clips.append(
                LinkedClip(
                    path=canonical_clip,
                    role="canonical_scoreboard_clip",
                    label="goal",
                    confidence=float(candidate.get("confidence", 0.0)),
                    start_s=clip_start(clips, canonical_clip),
                    end_s=clip_end(clips, canonical_clip),
                    evidence=candidate.get("evidence", {}),
                )
            )
        events.append(event)

    return events


def attach_goal_celebrations(
    output_dir: Path,
    manifest: dict[str, Any],
    events: list[LinkedEvent],
    goal_link_window_s: float,
) -> None:
    for clip in manifest.get("clips", []):
        if clip.get("event") != "goal_celebration":
            continue
        event = nearest_event(events, float(clip["start_s"]), goal_link_window_s)
        if not event:
            continue
        event.clips.append(
            LinkedClip(
                path=clip["output_video"],
                role="derived_celebration_clip",
                label="goal_celebration",
                confidence=float(clip.get("confidence", 0.0)),
                start_s=float(clip["start_s"]),
                end_s=float(clip["end_s"]),
                evidence=clip.get("evidence", {}),
            )
        )


def attach_classified_clips(
    classifications: list[dict[str, Any]],
    events: list[LinkedEvent],
    goal_link_window_s: float,
) -> None:
    for classification in classifications:
        if not classification.get("should_promote"):
            continue
        if classification.get("label") != "goal":
            continue

        timestamp = timestamp_from_path(
            classification.get("source_clip", "")
        ) or timestamp_from_path(classification.get("promoted_clip", ""))
        if timestamp is None:
            continue
        event = nearest_event(events, timestamp, goal_link_window_s)
        if not event:
            continue
        event.clips.append(
            LinkedClip(
                path=classification.get("promoted_clip")
                or classification.get("source_clip"),
                role="vlm_promoted_replay_clip",
                label="goal",
                confidence=float(classification.get("confidence", 0.0)),
                evidence={
                    "visible_evidence": classification.get("visible_evidence", []),
                    "source_clip": classification.get("source_clip"),
                    "uncertainty": classification.get("uncertainty"),
                },
            )
        )


def attach_standalone_classifications(
    classifications: list[dict[str, Any]],
    events: list[LinkedEvent],
) -> None:
    existing_paths = {
        clip.path
        for event in events
        for clip in event.clips
        if clip.path
    }
    standalone_index = 1
    for classification in classifications:
        if not classification.get("should_promote"):
            continue
        label = classification.get("label")
        if label == "goal":
            continue
        promoted_clip = classification.get("promoted_clip")
        if not promoted_clip or promoted_clip in existing_paths:
            continue
        timestamp = timestamp_from_path(promoted_clip) or 0.0
        event = LinkedEvent(
            event_id=f"{label}_{standalone_index:04d}_{int(timestamp)}s",
            event_type=label,
            canonical_timestamp_s=timestamp,
            canonical_clip=promoted_clip,
            confidence=float(classification.get("confidence", 0.0)),
        )
        event.clips.append(
            LinkedClip(
                path=promoted_clip,
                role="vlm_promoted_clip",
                label=label,
                confidence=float(classification.get("confidence", 0.0)),
                evidence={
                    "visible_evidence": classification.get("visible_evidence", []),
                    "source_clip": classification.get("source_clip"),
                    "uncertainty": classification.get("uncertainty"),
                },
            )
        )
        events.append(event)
        standalone_index += 1


def find_clip_for_event(
    clips: list[dict[str, Any]],
    event_type: str,
    timestamp_s: float,
) -> str | None:
    best_clip = None
    best_distance = float("inf")
    for clip in clips:
        if clip.get("event") != event_type:
            continue
        output_video = clip.get("output_video")
        clip_timestamp = timestamp_from_path(output_video or "")
        if clip_timestamp is None:
            continue
        distance = abs(clip_timestamp - timestamp_s)
        if distance < best_distance:
            best_clip = output_video
            best_distance = distance
    return best_clip


def clip_start(clips: list[dict[str, Any]], path: str) -> float | None:
    for clip in clips:
        if clip.get("output_video") == path:
            return float(clip["start_s"])
    return None


def clip_end(clips: list[dict[str, Any]], path: str) -> float | None:
    for clip in clips:
        if clip.get("output_video") == path:
            return float(clip["end_s"])
    return None


def nearest_event(
    events: list[LinkedEvent],
    timestamp_s: float,
    max_distance_s: float,
) -> LinkedEvent | None:
    if not events:
        return None
    event = min(events, key=lambda item: abs(item.canonical_timestamp_s - timestamp_s))
    if abs(event.canonical_timestamp_s - timestamp_s) <= max_distance_s:
        return event
    return None


def timestamp_from_path(path: str) -> float | None:
    match = re.search(r"_(\d+)s(?:\.mp4)?$", Path(path).name)
    if not match:
        return None
    return float(match.group(1))


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_linked_events(output_dir: Path, events: list[LinkedEvent]) -> None:
    rows = [event.to_dict() for event in events]
    with (output_dir / "linked_events.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    linked_dir = output_dir / "linked_events"
    linked_dir.mkdir(parents=True, exist_ok=True)
    for event in events:
        event_dir = linked_dir / event.event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        with (event_dir / "event.json").open("w", encoding="utf-8") as handle:
            json.dump(event.to_dict(), handle, indent=2)
