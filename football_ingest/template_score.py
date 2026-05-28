from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import pandas as pd

from football_ingest.models import ScoreboardRead


@dataclass(frozen=True)
class DigitBox:
    x: int
    y: int
    width: int
    height: int

    @classmethod
    def parse_pair(cls, value: str) -> tuple["DigitBox", "DigitBox"]:
        boxes = []
        for raw_box in value.split(";"):
            parts = [int(part.strip()) for part in raw_box.split(",")]
            if len(parts) != 4:
                raise ValueError("Digit boxes must be formatted as x,y,width,height;x,y,width,height")
            boxes.append(cls(*parts))
        if len(boxes) != 2:
            raise ValueError("Exactly two score digit boxes are required")
        return boxes[0], boxes[1]


@dataclass(frozen=True)
class TemplateScoreConfig:
    template_dir: Path
    digit_boxes: tuple[DigitBox, DigitBox]
    frame_skip: int = 200
    min_confidence: float = 0.80
    team1: str = "TEAM1"
    team2: str = "TEAM2"


class TemplateScoreExtractor:
    def __init__(self, config: TemplateScoreConfig):
        self.config = config
        self.templates = load_digit_templates(config.template_dir)

    def extract(self, video_path: Path, output_dir: Path) -> list[ScoreboardRead]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        reads: list[ScoreboardRead] = []
        frame_index = 0
        sampled_frames = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        started_at = time.perf_counter()
        last_logged_score = None

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % self.config.frame_skip != 0:
                frame_index += 1
                continue

            timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            score1, conf1 = classify_digit(frame, self.config.digit_boxes[0], self.templates)
            score2, conf2 = classify_digit(frame, self.config.digit_boxes[1], self.templates)
            confidence = min(conf1, conf2)
            parsed = score1 is not None and score2 is not None and confidence >= self.config.min_confidence

            raw_text = (
                f"{self.config.team1} {score1 if score1 is not None else '?'}-"
                f"{score2 if score2 is not None else '?'} {self.config.team2}"
            )
            if parsed:
                score = (score1, score2)
                if score != last_logged_score:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] OCR score read at "
                        f"{timestamp_s:.1f}s frame {frame_index}: "
                        f"{score1}-{score2} confidence={confidence:.2f}",
                        flush=True,
                    )
                    last_logged_score = score
            reads.append(
                ScoreboardRead(
                    timestamp_s=timestamp_s,
                    frame_index=frame_index,
                    raw_text=raw_text,
                    confidence=confidence,
                    team1=self.config.team1 if parsed else None,
                    team2=self.config.team2 if parsed else None,
                    score1=score1 if parsed else None,
                    score2=score2 if parsed else None,
                    parsed=parsed,
                )
            )
            sampled_frames += 1
            if sampled_frames == 1 or sampled_frames % 25 == 0:
                progress = ""
                if total_frames:
                    progress = f" ({frame_index / total_frames:.0%})"
                elapsed = time.perf_counter() - started_at
                print(
                    f"[{time.strftime('%H:%M:%S')}] OCR sampled {sampled_frames} frames"
                    f"{progress}; video time {timestamp_s:.0f}s; elapsed {elapsed:.1f}s",
                    flush=True,
                )
            frame_index += 1

        cap.release()
        write_timeline(reads, output_dir)
        return reads


def load_digit_templates(template_dir: Path) -> dict[int, list]:
    templates: dict[int, list] = {}
    for path in template_dir.glob("*.png"):
        digit_text = path.stem.split("_", 1)[0]
        if not digit_text.isdigit():
            continue
        digit = int(digit_text)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        templates.setdefault(digit, []).append(preprocess_digit(image))

    if not templates:
        raise RuntimeError(f"No digit templates found in {template_dir}")
    return templates


def classify_digit(frame, box: DigitBox, templates: dict[int, list]) -> tuple[int | None, float]:
    crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
    if crop.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    prepared = preprocess_digit(gray)
    best_digit = None
    best_score = -1.0

    for digit, digit_templates in templates.items():
        for template in digit_templates:
            resized = cv2.resize(prepared, (template.shape[1], template.shape[0]))
            score = float(cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)[0][0])
            if score > best_score:
                best_score = score
                best_digit = digit

    return best_digit, max(best_score, 0.0)


def preprocess_digit(gray):
    return cv2.GaussianBlur(gray, (3, 3), 0)


def write_timeline(reads: list[ScoreboardRead], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [read.to_dict() for read in reads]
    pd.DataFrame(rows).to_csv(output_dir / "timeline_ocr.csv", index=False)
    with (output_dir / "timeline_ocr.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
