from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import requests
import os

from football_ingest.labels import PROMOTABLE_LABELS
from football_ingest.models import AudioWindow, ReplayCandidate


STRONG_KEYWORD_LABELS = {
    "VAR": "var_review",
    "VIDEO ASSISTANT": "var_review",
    "VAR CHECK": "var_review",
    "CHECKING": "var_review",
    "ON-FIELD REVIEW": "referee_monitor_review",
    "REFEREE REVIEW AREA": "referee_monitor_review",
    "PENALTY": "penalty_kick",
    "YELLOW CARD": "yellow_card",
    "RED CARD": "red_card",
    "SECOND YELLOW": "second_yellow_red",
    "SUBSTITUTION": "substitution",
    "SUBSTITUTIONS": "substitution",
    "REPLACED BY": "substitution",
    "OFFSIDE": "offside",
    "HALF TIME": "half_time",
    "HALF-TIME": "half_time",
    "FULL TIME": "full_time",
    "FULL-TIME": "full_time",
    "ADDED TIME": "injury_stoppage",
    "INJURY": "injury_stoppage",
    "GOAL": "goal",
}

WEAK_CONTEXT_KEYWORDS = {
    "FIFA WORLD CUP",
    "QATAR 2022",
    "THE FINAL",
    "FINAL",
    "LUSAIL",
    "STADIUM",
    "TEAM LINEUP",
    "LINEUP",
    "FORMATION",
    "COACH",
    "FIFAPLUS",
    "QATAR AIRWAYS",
    "VISA",
    "HYUNDAI",
}

VLM_HINTS_ALLOWED_WITH_SUPPORT = {
    "injury_stoppage",
    "medical_treatment",
    "referee_discussion",
    "player_argument",
    "manager_reaction",
    "crowd_reaction",
    "goal_celebration",
    "goal",
    "penalty_kick",
    "yellow_card",
    "red_card",
    "substitution",
    "offside",
    "foul",
}


PROMPT = """You are reading visible broadcast overlay text from one football video frame.

Return JSON only:
{
  "visible_text": ["short text exactly visible on screen"],
  "label_hints": ["one or more football event labels if the text clearly indicates them"],
  "confidence": 0.0,
  "reason": "short visible evidence"
}

Rules:
- Only report text that is visibly overlaid or clearly readable.
- Do not infer from gameplay.
- Use label_hints only when visible text/graphics clearly support them.
- Allowed label_hints:
""" + ", ".join(PROMOTABLE_LABELS)


def detect_broadcast_text_candidates(
    video_path: Path,
    output_dir: Path,
    api_base: str = "http://localhost:1234/v1",
    model: str = "qwen/qwen3-vl-4b",
    sample_interval_s: float = 8.0,
    start_s: float = 0.0,
    end_s: float | None = None,
    max_frames: int = 120,
    min_confidence: float = 0.70,
    merge_gap_s: float = 20.0,
    support_candidates: list[ReplayCandidate] | None = None,
    audio_windows: list[AudioWindow] | None = None,
    api_key_env: str = "",  
) -> list[ReplayCandidate]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_s = frame_count / fps if frame_count else 0.0
    final_s = end_s if end_s is not None else duration_s

    raw_candidates: list[ReplayCandidate] = []
    timestamps = build_sample_timestamps(start_s, final_s, sample_interval_s, max_frames)
    started_at = time.perf_counter()
    for index, timestamp_s in enumerate(timestamps, start=1):
        print(
            f"[{time.strftime('%H:%M:%S')}] Broadcast text scan frame {index}/{len(timestamps)} "
            f"at {timestamp_s:.0f}s",
            flush=True,
        )
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_s * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue

        payload = read_broadcast_text_with_vlm(frame, api_base=api_base, model=model, api_key_env=api_key_env)
        text_items = [str(item) for item in payload.get("visible_text", [])]
        combined_text = " ".join(text_items).upper()
        vlm_hints = normalize_label_hints(payload.get("label_hints", []))
        confidence = safe_float(payload.get("confidence", 0.0))
        gate = gate_broadcast_text_candidate(
            combined_text=combined_text,
            vlm_hints=vlm_hints,
            timestamp_s=timestamp_s,
            support_candidates=support_candidates or [],
            audio_windows=audio_windows or [],
        )

        if not gate["label_hints"] or confidence < min_confidence:
            elapsed = time.perf_counter() - started_at
            print(
                f"[{time.strftime('%H:%M:%S')}] Broadcast text rejected frame {index}/{len(timestamps)} "
                f"confidence={confidence:.2f}; elapsed {elapsed:.1f}s",
                flush=True,
            )
            continue

        raw_candidates.append(
            ReplayCandidate(
                event="review_required",
                candidate_type="broadcast_text",
                timestamp_s=timestamp_s,
                start_s=max(timestamp_s - 8.0, 0.0),
                end_s=timestamp_s + 14.0,
                confidence=min(confidence, 0.90),
                evidence={
                    "broadcast_text_detected": True,
                    "visible_text": text_items,
                    "combined_text": combined_text,
                    "label_hints": gate["label_hints"],
                    "vlm_label_hints": vlm_hints,
                    "keyword_label_hints": gate["keyword_label_hints"],
                    "support_signals": gate["support_signals"],
                    "filter_reason": gate["filter_reason"],
                    "reason": payload.get("reason", ""),
                    "review_required_reason": "broadcast text/graphic label hint; not a final event label",
                },
            )
        )
        elapsed = time.perf_counter() - started_at
        print(
            f"[{time.strftime('%H:%M:%S')}] Broadcast text kept frame {index}/{len(timestamps)} "
            f"hints={','.join(gate['label_hints'])}; elapsed {elapsed:.1f}s",
            flush=True,
        )

    cap.release()
    candidates = merge_text_candidates(raw_candidates, merge_gap_s)
    write_broadcast_text_candidates(candidates, output_dir)
    return candidates


def build_sample_timestamps(
    start_s: float,
    end_s: float,
    sample_interval_s: float,
    max_frames: int,
) -> list[float]:
    timestamps = []
    current = start_s
    while current <= end_s and len(timestamps) < max_frames:
        timestamps.append(current)
        current += sample_interval_s
    return timestamps


def read_broadcast_text_with_vlm(frame, api_base: str, model: str, api_key_env: str = "") -> dict[str, Any]:
    resized = cv2.resize(frame, (768, 432))
    ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return {}
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

    headers = {}
    if api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    response = requests.post(
        f"{api_base.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 400,
        },
        headers=headers or None,
        timeout=120,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    return parse_json_response(raw)


def parse_json_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        return json.loads(match.group(0))


def normalize_label_hints(values) -> list[str]:
    if not isinstance(values, list):
        values = [values]
    allowed = set(PROMOTABLE_LABELS)
    return [str(value).strip() for value in values if str(value).strip() in allowed]


def keyword_label_hints(text: str) -> list[str]:
    hints = []
    for keyword, label in STRONG_KEYWORD_LABELS.items():
        if keyword_in_text(text, keyword):
            hints.append(label)
    return hints


def keyword_in_text(text: str, keyword: str) -> bool:
    pattern = r"(?<![A-Z0-9])" + re.escape(keyword).replace(r"\ ", r"\s+") + r"(?![A-Z0-9])"
    return re.search(pattern, text) is not None


def gate_broadcast_text_candidate(
    combined_text: str,
    vlm_hints: list[str],
    timestamp_s: float,
    support_candidates: list[ReplayCandidate],
    audio_windows: list[AudioWindow],
) -> dict[str, Any]:
    keyword_hints = sorted(set(keyword_label_hints(combined_text)))
    support_signals = nearby_support_signals(
        timestamp_s=timestamp_s,
        support_candidates=support_candidates,
        audio_windows=audio_windows,
    )
    scoreboard_only = is_scoreboard_or_context_only(combined_text)

    kept_hints = list(keyword_hints)
    supported_vlm_hints = [
        hint
        for hint in vlm_hints
        if hint in VLM_HINTS_ALLOWED_WITH_SUPPORT and support_signals
    ]
    kept_hints.extend(supported_vlm_hints)

    kept_hints = sorted(set(kept_hints))
    if keyword_hints:
        reason = "strong visible broadcast keyword"
    elif kept_hints and support_signals:
        reason = "vlm hint supported by nearby non-text signal"
    elif scoreboard_only:
        reason = "discarded scoreboard/branding context without event keyword"
    else:
        reason = "discarded weak broadcast text hint"

    return {
        "label_hints": kept_hints,
        "keyword_label_hints": keyword_hints,
        "support_signals": support_signals,
        "filter_reason": reason,
    }


def nearby_support_signals(
    timestamp_s: float,
    support_candidates: list[ReplayCandidate],
    audio_windows: list[AudioWindow],
    candidate_window_s: float = 25.0,
    audio_window_s: float = 20.0,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    for candidate in support_candidates:
        if candidate.start_s - candidate_window_s <= timestamp_s <= candidate.end_s + candidate_window_s:
            signals.append(
                {
                    "type": candidate.candidate_type,
                    "start_s": candidate.start_s,
                    "end_s": candidate.end_s,
                    "confidence": candidate.confidence,
                }
            )

    best_audio = max(
        (
            window
            for window in audio_windows
            if abs(window.timestamp_s - timestamp_s) <= audio_window_s and window.is_spike
        ),
        key=lambda window: window.spike_score,
        default=None,
    )
    if best_audio:
        signals.append(
            {
                "type": "audio_spike",
                "timestamp_s": best_audio.timestamp_s,
                "spike_score": best_audio.spike_score,
            }
        )
    return signals


def is_scoreboard_or_context_only(text: str) -> bool:
    compact = text.strip()
    if not compact:
        return True

    without_scores = re.sub(r"\b[A-Z]{2,4}\s*\d(?:\s*-\s*\d\s*[A-Z]{2,4})?\b", " ", compact)
    without_times = re.sub(r"\b\d{1,2}:\d{2}\b", " ", without_scores)
    without_digits = re.sub(r"\b\d+\b", " ", without_times)
    without_context = without_digits
    for keyword in WEAK_CONTEXT_KEYWORDS:
        without_context = without_context.replace(keyword, " ")
    remaining_tokens = re.findall(r"[A-Z]{3,}", without_context)

    if not remaining_tokens:
        return True
    if all(token in {"ARG", "FRA"} for token in remaining_tokens):
        return True
    return False


def merge_text_candidates(
    candidates: list[ReplayCandidate],
    merge_gap_s: float,
) -> list[ReplayCandidate]:
    if not candidates:
        return []

    candidates.sort(key=lambda candidate: candidate.start_s)
    merged: list[ReplayCandidate] = [candidates[0]]
    for candidate in candidates[1:]:
        previous = merged[-1]
        previous_hints = set(previous.evidence.get("label_hints", []))
        current_hints = set(candidate.evidence.get("label_hints", []))
        if candidate.start_s - previous.end_s <= merge_gap_s and previous_hints & current_hints:
            evidence = dict(previous.evidence)
            evidence["visible_text"] = sorted(
                set(evidence.get("visible_text", []))
                | set(candidate.evidence.get("visible_text", []))
            )
            evidence["combined_text"] = (
                str(evidence.get("combined_text", ""))
                + " "
                + str(candidate.evidence.get("combined_text", ""))
            ).strip()
            evidence["label_hints"] = sorted(previous_hints | current_hints)
            merged[-1] = ReplayCandidate(
                event=previous.event,
                candidate_type=previous.candidate_type,
                timestamp_s=previous.timestamp_s,
                start_s=previous.start_s,
                end_s=max(previous.end_s, candidate.end_s),
                confidence=max(previous.confidence, candidate.confidence),
                evidence=evidence,
            )
        else:
            merged.append(candidate)
    return merged


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def write_broadcast_text_candidates(
    candidates: list[ReplayCandidate],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [candidate.to_dict() for candidate in candidates]
    pd.DataFrame(rows).to_csv(output_dir / "broadcast_text_candidates.csv", index=False)
    with (output_dir / "broadcast_text_candidates.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
