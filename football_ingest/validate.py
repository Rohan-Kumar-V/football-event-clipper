from __future__ import annotations

import base64
import json
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import requests
import os

from football_ingest.labels import (
    LABEL_GUIDANCE,
    PROMOTABLE_LABELS,
    VALIDATOR_ONLY_LABELS,
    VALIDATION_LABELS,
    label_guidance_text,
    validation_label_text,
)
from football_ingest.models import ClipClassification


ALLOWED_VALIDATION_LABELS = set(VALIDATION_LABELS)
PROMOTABLE_LABEL_SET = set(PROMOTABLE_LABELS)

PROFILE_LABELS = {
    "card_decision": [
        "yellow_card",
        "red_card",
        "second_yellow_red",
        "foul",
        "player_argument",
        "referee_discussion",
    ],
    "substitution": [
        "substitution",
        "manager_reaction",
        "crowd_reaction",
    ],
    "injury_medical": [
        "injury_stoppage",
        "medical_treatment",
        "foul",
        "player_argument",
        "referee_discussion",
    ],
    "stoppage_family": [
        "yellow_card",
        "red_card",
        "second_yellow_red",
        "substitution",
        "injury_stoppage",
        "medical_treatment",
        "foul",
        "offside",
        "handball",
        "player_argument",
        "referee_discussion",
        "manager_reaction",
        "crowd_reaction",
        "goal_celebration",
        "aerial_duel",
    ],
    "var_referee": [
        "var_review",
        "referee_monitor_review",
        "referee_discussion",
        "player_argument",
        "penalty_kick",
        "offside",
        "foul",
        "handball",
    ],
    "celebration_reaction": [
        "goal_celebration",
        "crowd_reaction",
        "manager_reaction",
        "player_argument",
        "referee_discussion",
    ],
    "skill_dribble": [
        "skill_dribble",
        "nutmeg",
        "solo_run",
        "trick",
        "tackle",
        "aerial_duel",
        "foul",
    ],
    "attacking_replay": [
        "goal",
        "goal_celebration",
        "penalty_kick",
        "penalty_goal",
        "penalty_saved",
        "penalty_missed",
        "shot_on_target",
        "shot_off_target",
        "blocked_shot",
        "woodwork_hit",
        "goalkeeper_save",
        "big_chance",
        "cross",
        "through_ball",
        "key_pass",
        "counter_attack",
        "skill_dribble",
        "aerial_duel",
    ],
    "set_piece": [
        "penalty_kick",
        "penalty_goal",
        "penalty_saved",
        "penalty_missed",
        "corner_kick",
        "free_kick",
        "direct_free_kick_shot",
        "throw_in",
        "offside",
        "foul",
        "handball",
    ],
    "general": PROMOTABLE_LABELS,
}

PROFILE_RULES = {
    "card_decision": [
        "Promote yellow_card/red_card/second_yellow_red only when a physical card, card-color gesture, or explicit card graphic is visible.",
        "If players surround or speak to the referee but no card is visible, prefer referee_discussion or player_argument.",
        "Do not infer a card from a foul, stoppage, or angry body language alone.",
    ],
    "substitution": [
        "Promote substitution only when a substitution board, substitution graphic, or clear entering/leaving player swap is visible.",
        "A bench, coach, or lineup graphic alone is not a substitution.",
    ],
    "injury_medical": [
        "Promote medical_treatment only when medical staff or treatment is visible.",
        "Promote injury_stoppage when a player is down/in pain and play appears stopped.",
        "Do not infer injury from a normal foul replay unless the player remains down or treatment/stoppage is visible.",
    ],
    "stoppage_family": [
        "Use this profile for broad stoppage candidates where the exact incident is not known yet.",
        "Promote cards only when a card or card graphic is visible.",
        "Promote substitution only when a board, graphic, or player swap is visible.",
        "Promote injury/medical labels only when the player-down or treatment evidence is visible.",
        "If the clip only shows discussion/confrontation, prefer referee_discussion or player_argument.",
    ],
    "var_referee": [
        "Promote var_review only for VAR graphics/room/check presentation.",
        "Promote referee_monitor_review only when the referee is clearly at the pitchside monitor.",
        "If only discussion with the referee is visible, prefer referee_discussion.",
    ],
    "celebration_reaction": [
        "Promote goal_celebration only when players or fans are clearly celebrating a goal-like moment.",
        "Do not label ordinary pre-match ceremony, lineup, or generic crowd shots as a celebration.",
    ],
    "skill_dribble": [
        "Promote skill_dribble only when the ball carrier clearly beats or evades a defender while controlling the ball.",
        "Promote nutmeg only when the ball visibly passes through an opponent's legs.",
        "Promote trick only for a clear flair move; ordinary running with the ball is not enough.",
    ],
    "attacking_replay": [
        "Promote goal only when the ball entering the goal or a scoreboard-confirmed goal moment is visible.",
        "For shots, require a clear strike and visible outcome or goalkeeper action.",
        "For big_chance, require a clearly dangerous chance, not generic attacking possession.",
    ],
    "set_piece": [
        "Promote set-piece labels only when the restart setup, decision, or taking of the restart is visible.",
        "Do not infer a penalty/free kick/corner from players standing around unless the restart context is clear.",
    ],
    "general": [
        "Use the broad label list, but keep the same visible-evidence standard.",
    ],
}


VALIDATOR_PROMPT = f"""You are validating a short football broadcast clip from sampled frames.

Choose exactly one label from:
{validation_label_text()}.

Reliability rules:
- Only label what is clearly visible in the frames.
- Prefer the most specific visible label.
- Do not infer events from score/audio/context alone.
- Use no_event when nothing meaningful is visible.
- Use uncertain if the event is plausible but not clearly visible.

Label guidance:
{label_guidance_text()}

Return JSON only:
{{
  "label": "...",
  "confidence": 0.0,
  "should_promote": false,
  "visible_evidence": ["..."],
  "uncertainty": "..."
}}
"""


def validate_review_clips(
    output_dir: Path,
    api_base: str = "http://localhost:1234/v1",
    model: str = "qwen/qwen3-vl-4b",
    min_confidence: float = 0.78,
    frames_per_clip: int = 6,
    dry_run: bool = False,
    api_key_env: str = "",
) -> list[ClipClassification]:
    review_dir = output_dir / "review_required"
    clips = sorted(review_dir.glob("*.mp4"))
    classifications: list[ClipClassification] = []

    for index, clip_path in enumerate(clips, start=1):
        started_at = time.perf_counter()
        print(
            f"[{time.strftime('%H:%M:%S')}] Validating clip {index}/{len(clips)}: {clip_path.name}",
            flush=True,
        )
        metadata = read_clip_metadata(clip_path)
        validation_profile = validation_profile_from_metadata(metadata)
        prompt = build_validator_prompt(metadata, validation_profile)
        frames = extract_representative_frames(clip_path, frames_per_clip)
        raw_response = classify_frames_with_openai_compatible_api(
            frames=frames,
            api_base=api_base,
            model=model,
            prompt=prompt,
            api_key_env=api_key_env,
        )
        parsed = parse_model_json(raw_response)
        classification = classification_from_payload(
            clip_path=clip_path,
            payload=parsed,
            raw_response=raw_response,
            min_confidence=min_confidence,
            validation_profile=validation_profile,
        )
        if classification.should_promote:
            classification = promote_classification(
                output_dir=output_dir,
                classification=classification,
                dry_run=dry_run,
        )
        write_classification_metadata(output_dir, classification)
        classifications.append(classification)
        elapsed = time.perf_counter() - started_at
        print(
            f"[{time.strftime('%H:%M:%S')}] Classified {clip_path.name}: "
            f"{classification.label} confidence={classification.confidence:.2f} "
            f"promote={classification.should_promote} ({elapsed:.1f}s)",
            flush=True,
        )

    write_classification_manifest(output_dir, classifications)
    return classifications


def extract_representative_frames(clip_path: Path, frames_per_clip: int) -> list[bytes]:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip for validation: {clip_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise RuntimeError(f"Clip has no readable frames: {clip_path}")

    positions = [
        int(round((index + 1) * frame_count / (frames_per_clip + 1)))
        for index in range(frames_per_clip)
    ]
    encoded_frames: list[bytes] = []
    for position in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(position, frame_count - 1))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.resize(frame, (640, 360))
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            encoded_frames.append(encoded.tobytes())

    cap.release()
    if not encoded_frames:
        raise RuntimeError(f"Could not extract frames from clip: {clip_path}")
    return encoded_frames


def classify_frames_with_openai_compatible_api(
    frames: list[bytes],
    api_base: str,
    model: str,
    prompt: str = VALIDATOR_PROMPT,
    timeout_s: int = 180,
    api_key_env: str = "",
) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        b64 = base64.b64encode(frame).decode("ascii")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": 500,
    }
    
    headers = {}
    if api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    
    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        json=payload,
        headers=headers or None,
        timeout=timeout_s,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def read_clip_metadata(clip_path: Path) -> dict[str, Any]:
    metadata_path = clip_path.with_suffix(".json")
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validation_profile_from_metadata(metadata: dict[str, Any]) -> str:
    evidence = metadata.get("evidence", {}) if isinstance(metadata, dict) else {}
    if not isinstance(evidence, dict):
        evidence = {}
    candidate_type = str(evidence.get("candidate_type") or "").strip()
    label_hints = normalize_hint_list(evidence.get("label_hints", []))
    return select_validation_profile(candidate_type, label_hints)


def build_validator_prompt(metadata: dict[str, Any], profile: str | None = None) -> str:
    evidence = metadata.get("evidence", {}) if isinstance(metadata, dict) else {}
    if not isinstance(evidence, dict):
        evidence = {}
    candidate_type = str(evidence.get("candidate_type") or "").strip()
    label_hints = normalize_hint_list(evidence.get("label_hints", []))
    profile = profile or select_validation_profile(candidate_type, label_hints)
    labels = sorted(set(PROFILE_LABELS[profile] + VALIDATOR_ONLY_LABELS))
    rules = "\n".join(f"- {rule}" for rule in PROFILE_RULES[profile])
    guidance = "\n".join(
        f"- {label}: {LABEL_GUIDANCE.get(label, '')}"
        for label in labels
        if label in LABEL_GUIDANCE
    )

    hint_text = ", ".join(label_hints) if label_hints else "none"
    return f"""You are validating a short football broadcast clip from sampled frames.

Validation profile: {profile}
Candidate source: {candidate_type or "unknown"}
Candidate label hints: {hint_text}

Candidate source and hints are only routing context. They are not proof.

Choose exactly one label from:
{", ".join(labels)}.

Reliability rules:
- Only label what is clearly visible in the frames.
- Prefer the most specific visible label.
- Do not infer events from score/audio/context alone.
- Use no_event when nothing meaningful is visible.
- Use uncertain if the event is plausible but not clearly visible.
{rules}

Label guidance:
{guidance}

Return JSON only:
{{
  "label": "...",
  "confidence": 0.0,
  "should_promote": false,
  "visible_evidence": ["..."],
  "uncertainty": "..."
}}
"""


def normalize_hint_list(values) -> list[str]:
    if not isinstance(values, list):
        values = [values]
    return [str(value).strip() for value in values if str(value).strip()]


def select_validation_profile(candidate_type: str, label_hints: list[str]) -> str:
    hints = set(label_hints)
    if candidate_type == "stoppage_segment":
        return "stoppage_family"
    if candidate_type == "skill_segment":
        return "skill_dribble"
    if candidate_type == "chance_segment":
        return "attacking_replay"
    if hints & {"yellow_card", "red_card", "second_yellow_red"}:
        return "card_decision"
    if hints & {"substitution"}:
        return "substitution"
    if hints & {"injury_stoppage", "medical_treatment"}:
        return "injury_medical"
    if hints & {"var_review", "referee_monitor_review", "referee_discussion"}:
        return "var_referee"
    if hints & {"goal_celebration", "crowd_reaction", "manager_reaction"}:
        return "celebration_reaction"
    if hints & {"skill_dribble", "nutmeg", "solo_run", "trick"}:
        return "skill_dribble"
    if hints & {
        "penalty_kick",
        "penalty_goal",
        "penalty_saved",
        "penalty_missed",
        "corner_kick",
        "free_kick",
        "direct_free_kick_shot",
        "throw_in",
        "offside",
    }:
        return "set_piece"
    if candidate_type == "replay_segment":
        return "attacking_replay"
    return "general"


def parse_model_json(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {
                "label": "uncertain",
                "confidence": 0.0,
                "should_promote": False,
                "visible_evidence": [],
                "uncertainty": "Model did not return parseable JSON.",
            }
        return json.loads(match.group(0))


def classification_from_payload(
    clip_path: Path,
    payload: dict[str, Any],
    raw_response: str,
    min_confidence: float,
    validation_profile: str = "general",
) -> ClipClassification:
    label = str(payload.get("label", "uncertain")).strip()
    if label not in ALLOWED_VALIDATION_LABELS:
        label = "uncertain"

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    evidence = payload.get("visible_evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    evidence = [str(item) for item in evidence if str(item).strip()]

    model_should_promote = bool(payload.get("should_promote", False))
    should_promote = (
        model_should_promote
        and confidence >= min_confidence
        and label in PROMOTABLE_LABEL_SET
        and len(evidence) > 0
    )

    return ClipClassification(
        source_clip=clip_path,
        label=label,
        confidence=confidence,
        should_promote=should_promote,
        visible_evidence=evidence,
        uncertainty=str(payload.get("uncertainty", "")),
        raw_response=raw_response,
        validation_profile=validation_profile,
    )


def promote_classification(
    output_dir: Path,
    classification: ClipClassification,
    dry_run: bool,
) -> ClipClassification:
    event_dir = output_dir / classification.label
    event_dir.mkdir(parents=True, exist_ok=True)

    promoted_clip = event_dir / classification.source_clip.name.replace(
        "replay_segment", classification.label
    )
    metadata_path = promoted_clip.with_suffix(".json")

    if dry_run:
        return classification

    shutil.copy2(classification.source_clip, promoted_clip)

    promoted = replace(
        classification,
        promoted_clip=promoted_clip,
        metadata_path=metadata_path,
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(promoted.to_dict(), handle, indent=2)
    return promoted


def write_classification_metadata(
    output_dir: Path,
    classification: ClipClassification,
) -> None:
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = validation_dir / f"{classification.source_clip.stem}.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(classification.to_dict(), handle, indent=2)


def write_classification_manifest(
    output_dir: Path,
    classifications: list[ClipClassification],
) -> None:
    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    rows = [classification.to_dict() for classification in classifications]
    with (validation_dir / "classification_manifest.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(rows, handle, indent=2)
