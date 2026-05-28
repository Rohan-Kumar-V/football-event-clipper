from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from football_ingest.models import CropBox, ScoreboardRead
from football_ingest.template_score import DigitBox


DEFAULT_SCOREBOARD_CROP = CropBox(x=110, y=40, width=203, height=25)


def configure_paddle_runtime(output_dir: Path) -> None:
    runtime_home = output_dir.resolve().parent / ".runtime_home"
    runtime_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(runtime_home)
    os.environ["USERPROFILE"] = str(runtime_home)
    os.environ.setdefault("XDG_CACHE_HOME", str(runtime_home / ".cache"))
    os.environ.setdefault("PADDLE_HOME", str(runtime_home / ".cache" / "paddle"))
    os.environ.setdefault("PADDLE_EXTENSION_DIR", str(runtime_home / ".cache" / "paddle_extension"))
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")
    cache_dir = runtime_home / "paddlex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


@dataclass(frozen=True)
class OcrConfig:
    crop: CropBox = DEFAULT_SCOREBOARD_CROP
    frame_skip: int = 200
    min_confidence: float = 0.90
    language: str = "en"
    save_debug_crops: bool = False


class ScoreboardOcrExtractor:
    def __init__(self, config: OcrConfig):
        self.config = config

    def extract(self, video_path: Path, output_dir: Path) -> list[ScoreboardRead]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for video frame extraction. Install with: "
                "python -m pip install -e ."
            ) from exc

        configure_paddle_runtime(output_dir)
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is required for scoreboard OCR. Install with: "
                "python -m pip install -e .[ocr]"
            ) from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        debug_dir = output_dir / "debug_scoreboard_crops"
        if self.config.save_debug_crops:
            debug_dir.mkdir(parents=True, exist_ok=True)

        ocr = create_paddle_ocr(self.config.language)
        reads: list[ScoreboardRead] = []
        frame_index = 0
        last_logged_score = None

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % self.config.frame_skip != 0:
                frame_index += 1
                continue

            timestamp_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            crop = crop_frame(frame, self.config.crop)
            if self.config.save_debug_crops:
                crop_name = debug_dir / f"scoreboard_{frame_index:08d}_{int(timestamp_s)}s.jpg"
                cv2.imwrite(str(crop_name), crop)

            for raw_text, confidence in run_ocr(ocr, crop):
                parsed = parse_scoreboard_text(raw_text, confidence, self.config.min_confidence)
                score = (
                    parsed.get("score1"),
                    parsed.get("score2"),
                ) if parsed else None
                if parsed and score != last_logged_score:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] OCR score read at "
                        f"{timestamp_s:.1f}s frame {frame_index}: "
                        f"{parsed.get('score1')}-{parsed.get('score2')} "
                        f"confidence={confidence:.2f}",
                        flush=True,
                    )
                    last_logged_score = score
                reads.append(
                    ScoreboardRead(
                        timestamp_s=timestamp_s,
                        frame_index=frame_index,
                        raw_text=raw_text,
                        confidence=confidence,
                        team1=parsed.get("team1"),
                        team2=parsed.get("team2"),
                        score1=parsed.get("score1"),
                        score2=parsed.get("score2"),
                        parsed=bool(parsed),
                    )
                )

            frame_index += 1

        cap.release()
        write_timeline(reads, output_dir)
        return reads


@dataclass(frozen=True)
class DigitBoxOcrConfig:
    digit_boxes: tuple[DigitBox, DigitBox]
    frame_skip: int = 200
    min_confidence: float = 0.80
    language: str = "en"
    team1: str = "TEAM1"
    team2: str = "TEAM2"
    save_debug_crops: bool = False


class DigitBoxOcrExtractor:
    def __init__(self, config: DigitBoxOcrConfig):
        self.config = config

    def extract(self, video_path: Path, output_dir: Path) -> list[ScoreboardRead]:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required for video frame extraction. Install with: "
                "python -m pip install -e ."
            ) from exc

        configure_paddle_runtime(output_dir)
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "PaddleOCR is required for direct scoreboard OCR. Install with: "
                "python -m pip install -e .[ocr]"
            ) from exc

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        debug_dir = output_dir / "debug_score_digit_crops"
        if self.config.save_debug_crops:
            debug_dir.mkdir(parents=True, exist_ok=True)

        ocr = create_paddle_ocr(self.config.language)
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
            left_digit, left_conf, left_text = read_digit_box(
                ocr,
                frame,
                self.config.digit_boxes[0],
                debug_dir / f"{frame_index:08d}_{int(timestamp_s)}s_left"
                if self.config.save_debug_crops
                else None,
            )
            right_digit, right_conf, right_text = read_digit_box(
                ocr,
                frame,
                self.config.digit_boxes[1],
                debug_dir / f"{frame_index:08d}_{int(timestamp_s)}s_right"
                if self.config.save_debug_crops
                else None,
            )
            confidence = min(left_conf, right_conf)
            parsed = (
                left_digit is not None
                and right_digit is not None
                and confidence >= self.config.min_confidence
            )
            raw_text = (
                f"{self.config.team1} {left_text or '?'}-"
                f"{right_text or '?'} {self.config.team2}"
            )
            reads.append(
                ScoreboardRead(
                    timestamp_s=timestamp_s,
                    frame_index=frame_index,
                    raw_text=raw_text,
                    confidence=confidence,
                    team1=self.config.team1 if parsed else None,
                    team2=self.config.team2 if parsed else None,
                    score1=left_digit if parsed else None,
                    score2=right_digit if parsed else None,
                    parsed=parsed,
                )
            )
            if parsed:
                score = (left_digit, right_digit)
                if score != last_logged_score:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] OCR score read at "
                        f"{timestamp_s:.1f}s frame {frame_index}: "
                        f"{left_digit}-{right_digit} confidence={confidence:.2f}",
                        flush=True,
                    )
                    last_logged_score = score

            sampled_frames += 1
            if sampled_frames == 1 or sampled_frames % 25 == 0:
                progress = f" ({frame_index / total_frames:.0%})" if total_frames else ""
                elapsed = time.perf_counter() - started_at
                print(
                    f"[{time.strftime('%H:%M:%S')}] Direct OCR sampled {sampled_frames} frames"
                    f"{progress}; video time {timestamp_s:.0f}s; parsed {sum(read.parsed for read in reads)}; "
                    f"elapsed {elapsed:.1f}s",
                    flush=True,
                )

            frame_index += 1

        cap.release()
        write_timeline(reads, output_dir)
        return reads


def crop_frame(frame, crop: CropBox):
    y1 = max(crop.y, 0)
    y2 = max(crop.y + crop.height, y1 + 1)
    x1 = max(crop.x, 0)
    x2 = max(crop.x + crop.width, x1 + 1)
    return frame[y1:y2, x1:x2]


def create_paddle_ocr(language: str):
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=language,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def crop_digit_box(frame, box: DigitBox):
    y1 = max(box.y, 0)
    y2 = max(box.y + box.height, y1 + 1)
    x1 = max(box.x, 0)
    x2 = max(box.x + box.width, x1 + 1)
    return frame[y1:y2, x1:x2]


def read_digit_box(ocr, frame, box: DigitBox, debug_stem: Path | None = None) -> tuple[int | None, float, str]:
    try:
        import cv2
    except ImportError:
        return None, 0.0, ""

    crop = crop_digit_box(frame, box)
    if crop.size == 0:
        return None, 0.0, ""

    variants = preprocess_digit_crop(crop)
    best_digit = None
    best_confidence = 0.0
    best_text = ""

    for index, image in enumerate(variants):
        if debug_stem:
            cv2.imwrite(str(debug_stem.with_name(f"{debug_stem.name}_{index}.png")), image)
        for raw_text, confidence in run_ocr(ocr, image):
            digit = parse_single_digit(raw_text)
            if digit is not None and confidence > best_confidence:
                best_digit = digit
                best_confidence = confidence
                best_text = raw_text

    return best_digit, best_confidence, best_text


def preprocess_digit_crop(crop):
    import cv2

    variants = []
    scale = max(4, int(round(80 / max(crop.shape[:2]))))
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    padded = cv2.copyMakeBorder(enlarged, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    variants.append(padded)

    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    variants.append(gray)

    normalized = cv2.equalizeHist(gray)
    variants.append(normalized)

    _, otsu = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    variants.append(255 - otsu)
    return variants


def parse_single_digit(raw_text: str) -> int | None:
    text = raw_text.strip()
    replacements = {
        "O": "0",
        "o": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
    }
    text = "".join(replacements.get(char, char) for char in text)
    match = re.search(r"\d", text)
    if not match:
        return None
    return int(match.group(0))


def run_ocr(ocr, image) -> list[tuple[str, float]]:
    try:
        result = ocr.ocr(image, cls=True, det=False)
        return parse_paddle_ocr_result(result)
    except TypeError:
        pass
    except Exception:
        return []

    try:
        result = ocr.predict(image)
        return parse_paddle_ocr_result(result)
    except Exception:
        return []


def parse_paddle_ocr_result(result) -> list[tuple[str, float]]:
    reads: list[tuple[str, float]] = []
    for line in result or []:
        if hasattr(line, "json"):
            try:
                payload = line.json
                if callable(payload):
                    payload = payload()
                data = payload.get("res", payload) if isinstance(payload, dict) else {}
                texts = data.get("rec_texts") or data.get("text") or []
                scores = data.get("rec_scores") or data.get("scores") or []
                if isinstance(texts, str):
                    texts = [texts]
                if not isinstance(scores, list):
                    scores = [scores]
                for text, score in zip(texts, scores or [1.0] * len(texts)):
                    reads.append((str(text).strip(), float(score)))
                continue
            except Exception:
                pass
        if isinstance(line, dict):
            texts = line.get("rec_texts") or line.get("text") or []
            scores = line.get("rec_scores") or line.get("scores") or []
            if isinstance(texts, str):
                texts = [texts]
            if not isinstance(scores, list):
                scores = [scores]
            for text, score in zip(texts, scores or [1.0] * len(texts)):
                try:
                    reads.append((str(text).strip(), float(score)))
                except (TypeError, ValueError):
                    continue
            continue
        for item in line or []:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            text, confidence = item
            try:
                reads.append((str(text).strip(), float(confidence)))
            except (TypeError, ValueError):
                continue
    return reads


def parse_scoreboard_text(
    raw_text: str,
    confidence: float,
    min_confidence: float,
) -> dict[str, int | str]:
    if confidence < min_confidence:
        return {}

    text = normalize_scoreboard_text(raw_text)
    patterns = [
        re.compile(
            r"^(?P<team1>[A-Z]{2,4})\s*(?P<score1>\d{1,2})\s*[-: ]\s*"
            r"(?P<score2>\d{1,2})\s*(?P<team2>[A-Z]{2,4})$"
        ),
        re.compile(
            r"^(?P<team1>[A-Z]{2,4})(?P<score1>\d{1,2})"
            r"(?P<score2>\d{1,2})(?P<team2>[A-Z]{2,4})$"
        ),
    ]

    for pattern in patterns:
        match = pattern.match(text)
        if match:
            return {
                "team1": match.group("team1"),
                "team2": match.group("team2"),
                "score1": int(match.group("score1")),
                "score2": int(match.group("score2")),
            }

    groups = re.findall(r"[A-Z]{2,4}|\d{1,2}", text)
    if len(groups) >= 4:
        team1, score1, score2, team2 = groups[:4]
        if team1.isalpha() and score1.isdigit() and score2.isdigit() and team2.isalpha():
            return {
                "team1": team1,
                "team2": team2,
                "score1": int(score1),
                "score2": int(score2),
            }

    return {}


def normalize_scoreboard_text(raw_text: str) -> str:
    text = raw_text.upper().strip()
    replacements = {
        "|": " ",
        "_": " ",
        ".": " ",
        "—": "-",
        "–": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text)


def write_timeline(reads: list[ScoreboardRead], output_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError(
            "pandas is required for writing the OCR timeline. Install with: "
            "python -m pip install -e ."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [read.to_dict() for read in reads]
    pd.DataFrame(rows).to_csv(output_dir / "timeline_ocr.csv", index=False)
    with (output_dir / "timeline_ocr.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
