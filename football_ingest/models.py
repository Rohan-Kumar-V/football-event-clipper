from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CropBox:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def parse(cls, value: str) -> "CropBox":
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError("Crop must be formatted as x,y,width,height")
        return cls(*parts)


@dataclass(frozen=True)
class ScoreboardRead:
    timestamp_s: float
    frame_index: int
    raw_text: str
    confidence: float
    team1: str | None = None
    team2: str | None = None
    score1: int | None = None
    score2: int | None = None
    parsed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalCandidate:
    event: str
    timestamp_s: float
    scoring_side: str
    previous_score: tuple[int, int]
    new_score: tuple[int, int]
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioWindow:
    timestamp_s: float
    rms: float
    db: float
    spike_score: float
    is_spike: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SceneCut:
    timestamp_s: float
    frame_index: int
    mean_absdiff: float
    hist_correlation: float
    hash_value: int
    repeat_match_timestamp_s: float | None = None
    repeat_match_distance: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayCandidate:
    event: str
    candidate_type: str
    timestamp_s: float
    start_s: float
    end_s: float
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClipClassification:
    source_clip: Path
    label: str
    confidence: float
    should_promote: bool
    visible_evidence: list[str]
    uncertainty: str
    raw_response: str
    validation_profile: str = "general"
    promoted_clip: Path | None = None
    metadata_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_clip"] = str(self.source_clip)
        data["promoted_clip"] = str(self.promoted_clip) if self.promoted_clip else None
        data["metadata_path"] = str(self.metadata_path) if self.metadata_path else None
        return data


@dataclass(frozen=True)
class ClipResult:
    event: str
    source_video: Path
    output_video: Path
    metadata_path: Path
    start_s: float
    end_s: float
    confidence: float
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_video"] = str(self.source_video)
        data["output_video"] = str(self.output_video)
        data["metadata_path"] = str(self.metadata_path)
        return data
