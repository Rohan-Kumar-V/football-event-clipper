from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from football_ingest.template_score import DigitBox


@dataclass(frozen=True)
class Calibration:
    video_path: str
    team1: str
    team2: str
    template_dir: str
    digit_boxes: str
    sample_times_s: list[float]
    notes: str = ""
    ocr_engine: str = "template"

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(**data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)


def create_calibration(
    video_path: Path,
    output_dir: Path,
    digit_boxes: str,
    team1: str,
    team2: str,
    template_specs: list[str],
    sample_times_s: list[float],
    notes: str = "",
) -> Calibration:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "sample_frames"
    template_dir = output_dir / "score_digit_templates"
    samples_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = {}
    for time_s in sample_times_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_path = samples_dir / f"sample_{int(time_s)}s.jpg"
        cv2.imwrite(str(frame_path), frame)
        frames[int(time_s)] = frame

    boxes = DigitBox.parse_pair(digit_boxes)
    for spec in template_specs:
        template = parse_template_spec(spec)
        frame = frames.get(int(template["time_s"]))
        if frame is None:
            raise RuntimeError(f"No sampled frame available for template spec: {spec}")
        box = boxes[int(template["side"])]
        crop = frame[box.y : box.y + box.height, box.x : box.x + box.width]
        if crop.size == 0:
            raise RuntimeError(f"Empty crop for template spec: {spec}")
        filename = (
            f"{template['digit']}_"
            f"{team1 if int(template['side']) == 0 else team2}_"
            f"{int(template['time_s'])}s.png"
        )
        cv2.imwrite(str(template_dir / filename), crop)

    cap.release()

    calibration = Calibration(
        video_path=str(video_path),
        team1=team1,
        team2=team2,
        template_dir=str(template_dir),
        digit_boxes=digit_boxes,
        sample_times_s=sample_times_s,
        notes=notes,
    )
    calibration.save(output_dir / "calibration.json")
    return calibration


def parse_template_spec(spec: str) -> dict[str, float | int]:
    parts = [part.strip() for part in spec.split(":")]
    if len(parts) != 3:
        raise ValueError(
            "Template specs must be formatted as time_s:side:digit, "
            "where side is 0 for left score or 1 for right score."
        )
    time_s = float(parts[0])
    side = int(parts[1])
    digit = int(parts[2])
    if side not in {0, 1}:
        raise ValueError("Template spec side must be 0 or 1")
    if digit < 0 or digit > 9:
        raise ValueError("Template spec digit must be 0-9")
    return {"time_s": time_s, "side": side, "digit": digit}
