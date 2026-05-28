from __future__ import annotations

import argparse
from contextlib import contextmanager
import time
from pathlib import Path

import os
import requests

from football_ingest.audio import analyze_audio_energy, attach_audio_evidence
from football_ingest.broadcast_text import detect_broadcast_text_candidates
from football_ingest.calibration import Calibration
from football_ingest.chances import detect_chance_candidates
from football_ingest.clips import (
    clip_goal_candidates,
    clip_goal_celebrations,
    clip_replay_candidates,
)
from football_ingest.config import config_path, config_value
from football_ingest.goals import detect_goal_candidates
from football_ingest.link_events import link_events
from football_ingest.manifest import write_manifest
from football_ingest.ocr import DigitBoxOcrConfig, DigitBoxOcrExtractor
from football_ingest.reporting import generate_report
from football_ingest.scene import detect_replay_candidates, detect_scene_cuts
from football_ingest.skills import detect_skill_candidates
from football_ingest.stoppage import (
    detect_stoppage_candidates,
    infer_match_start_from_reads,
)
from football_ingest.taxonomy import ensure_event_folders
from football_ingest.template_score import DigitBox, TemplateScoreConfig, TemplateScoreExtractor
from football_ingest.validate import validate_review_clips


DEFAULT_VIDEO = config_path("ingestion.default_video", "input_videos/match.mp4")
DEFAULT_OUTPUT = config_path("ingestion.default_output", Path("clips") / "match_final")
DEFAULT_TEMPLATE_DIR = config_path("ingestion.default_template_dir", Path("clips") / "templates")
DEFAULT_DIGIT_BOXES = str(config_value("ingestion.default_digit_boxes", "222,36,30,31;274,36,30,31"))
DEFAULT_MODEL = str(config_value("vlm.model", "qwen/qwen3-vl-4b"))
DEFAULT_API_BASE = str(config_value("vlm.api_base", "http://localhost:1234/v1"))
DEFAULT_API_KEY_ENV = str(config_value("vlm.api_key_env", ""))
DEFAULT_OCR_ENGINE = str(config_value("ingestion.ocr_engine", "template"))


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@contextmanager
def timed_step(label: str):
    started_at = time.perf_counter()
    log(f"START {label}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started_at
        log(f"DONE  {label} ({elapsed:.1f}s)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full football clip ingestion pipeline end-to-end.",
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE_DIR)
    parser.add_argument("--digit-boxes", default=DEFAULT_DIGIT_BOXES)
    parser.add_argument("--ocr-engine", choices=["template", "paddle-digits"], default=DEFAULT_OCR_ENGINE)
    parser.add_argument("--team1", default=str(config_value("ingestion.team1", "ARG")))
    parser.add_argument("--team2", default=str(config_value("ingestion.team2", "FRA")))
    parser.add_argument("--frame-skip", type=int, default=int(config_value("ingestion.frame_skip", 200)))
    parser.add_argument("--min-ocr-confidence", type=float, default=float(config_value("ingestion.min_ocr_confidence", 0.90)))
    parser.add_argument("--max-replay-candidates", type=int, default=int(config_value("ingestion.max_replay_candidates", 12)))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-linking", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument("--skip-stoppage", action="store_true")
    parser.add_argument("--skip-skills", action="store_true")
    parser.add_argument("--skip-chances", action="store_true")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-broadcast-text", action="store_true")
    parser.add_argument("--broadcast-text-sample-interval", type=float, default=float(config_value("broadcast_text.sample_interval", 12.0)))
    parser.add_argument("--broadcast-text-max-frames", type=int, default=int(config_value("broadcast_text.max_frames", 120)))
    parser.add_argument("--validation-min-confidence", type=float, default=float(config_value("validation.min_confidence", 0.78)))
    parser.add_argument("--validation-frames-per-clip", type=int, default=int(config_value("validation.frames_per_clip", 6)))
    parser.add_argument("--max-stoppage-candidates", type=int, default=int(config_value("ingestion.max_stoppage_candidates", 24)))
    parser.add_argument("--max-skill-candidates", type=int, default=int(config_value("ingestion.max_skill_candidates", 20)))
    parser.add_argument("--max-chance-candidates", type=int, default=int(config_value("ingestion.max_chance_candidates", 18)))
    parser.add_argument("--stoppage-ignore-before", type=float)
    parser.add_argument("--stoppage-max-duration", type=float, default=float(config_value("ingestion.stoppage_max_duration", 50.0)))
    parser.add_argument("--goal-pre-seconds", type=float, default=float(config_value("ingestion.goal_pre_seconds", 20.0)))
    parser.add_argument("--goal-post-seconds", type=float, default=float(config_value("ingestion.goal_post_seconds", 35.0)))
    parser.add_argument("--celebration-pre-seconds", type=float, default=float(config_value("ingestion.celebration_pre_seconds", 4.0)))
    parser.add_argument("--celebration-post-seconds", type=float, default=float(config_value("ingestion.celebration_post_seconds", 32.0)))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.calibration:
        calibration = Calibration.load(args.calibration.resolve())
        if args.video == DEFAULT_VIDEO:
            args.video = Path(calibration.video_path)
        args.template_dir = Path(calibration.template_dir)
        args.digit_boxes = calibration.digit_boxes
        args.team1 = calibration.team1
        args.team2 = calibration.team2
        args.ocr_engine = calibration.ocr_engine

    video_path = args.video.resolve()
    output_dir = args.output.resolve()
    template_dir = args.template_dir.resolve()

    if not video_path.exists():
        raise SystemExit(f"Video file not found: {video_path}")
    if args.ocr_engine == "template" and not template_dir.exists():
        raise SystemExit(f"Template directory not found: {template_dir}")
    
    vlm_needed = not (args.skip_validation and args.skip_broadcast_text)
    if vlm_needed:
        log("Checking VLM API configuration...")
        preflight_vlm_api(
            api_base=args.api_base,
            model=args.model,
            api_key_env=args.api_key_env,
        )
        log("VLM API check passed.")

    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_event_folders(output_dir)

    log(f"Video: {video_path}")
    log(f"Output: {output_dir}")
    log(f"Frame skip: {args.frame_skip}")

    with timed_step("[1/4] OCR scoreboard timeline and goal detection"):
        if args.ocr_engine == "paddle-digits":
            extractor = DigitBoxOcrExtractor(
                DigitBoxOcrConfig(
                    digit_boxes=DigitBox.parse_pair(args.digit_boxes),
                    frame_skip=args.frame_skip,
                    min_confidence=args.min_ocr_confidence,
                    team1=args.team1,
                    team2=args.team2,
                )
            )
        else:
            extractor = TemplateScoreExtractor(
                TemplateScoreConfig(
                    template_dir=template_dir,
                    digit_boxes=DigitBox.parse_pair(args.digit_boxes),
                    frame_skip=args.frame_skip,
                    min_confidence=args.min_ocr_confidence,
                    team1=args.team1,
                    team2=args.team2,
                )
            )
        reads = extractor.extract(video_path, output_dir)
        goal_candidates = detect_goal_candidates(
            reads,
            min_confidence=args.min_ocr_confidence,
        )
        log(f"OCR reads: {len(reads)} parsed: {sum(read.parsed for read in reads)}")
        log(f"Goal candidates: {len(goal_candidates)}")

    audio_windows = []
    if not args.skip_audio:
        with timed_step("[2/4] Audio energy analysis"):
            audio_windows = analyze_audio_energy(video_path, output_dir)
            goal_candidates = attach_audio_evidence(goal_candidates, audio_windows)
            log(f"Audio windows: {len(audio_windows)}")
    else:
        log("[2/4] Skipping audio.")

    clips = []
    with timed_step("Clip scoreboard goal and celebration candidates"):
        clips.extend(
            clip_goal_candidates(
                video_path=video_path,
                output_dir=output_dir,
                candidates=goal_candidates,
                pre_seconds=args.goal_pre_seconds,
                post_seconds=args.goal_post_seconds,
                dry_run=args.dry_run,
            )
        )
        clips.extend(
            clip_goal_celebrations(
                video_path=video_path,
                output_dir=output_dir,
                candidates=goal_candidates,
                pre_seconds=args.celebration_pre_seconds,
                post_seconds=args.celebration_post_seconds,
                dry_run=args.dry_run,
            )
        )
        log(f"Goal-family clips: {len(clips)}")

    scene_cuts = []
    replay_candidates = []
    broadcast_text_candidates = []
    stoppage_candidates = []
    skill_candidates = []
    chance_candidates = []
    if not args.skip_replay:
        with timed_step("[3/4] Replay-like candidate detection"):
            scene_cuts = detect_scene_cuts(video_path, output_dir)
            replay_candidates = detect_replay_candidates(
                cuts=scene_cuts,
                audio_windows=audio_windows,
                output_dir=output_dir,
                max_candidates=args.max_replay_candidates,
            )
            clips.extend(
                clip_replay_candidates(
                    video_path=video_path,
                    output_dir=output_dir,
                    candidates=replay_candidates,
                    dry_run=args.dry_run,
                )
            )
            log(f"Scene cuts: {len(scene_cuts)}")
            log(f"Replay candidates: {len(replay_candidates)}")
    else:
        log("[3/4] Skipping replay detection.")

    if not args.skip_stoppage:
        with timed_step("[3/4] Stoppage-family candidate detection"):
            stoppage_ignore_before = (
                args.stoppage_ignore_before
                if args.stoppage_ignore_before is not None
                else infer_match_start_from_reads(reads)
            )
            stoppage_candidates = detect_stoppage_candidates(
                video_path=video_path,
                output_dir=output_dir,
                audio_windows=audio_windows,
                scene_cuts=scene_cuts,
                ignore_before_s=stoppage_ignore_before,
                max_duration_s=args.stoppage_max_duration,
                max_candidates=args.max_stoppage_candidates,
            )
            clips.extend(
                clip_replay_candidates(
                    video_path=video_path,
                    output_dir=output_dir,
                    candidates=stoppage_candidates,
                    dry_run=args.dry_run,
                )
            )
            log(f"Stoppage candidates: {len(stoppage_candidates)}")
    else:
        log("[3/4] Skipping stoppage detection.")

    if not args.skip_skills:
        with timed_step("[3/4] Skill/dribble candidate detection"):
            skill_ignore_before = infer_match_start_from_reads(reads)
            skill_candidates = detect_skill_candidates(
                video_path=video_path,
                output_dir=output_dir,
                audio_windows=audio_windows,
                scene_cuts=scene_cuts,
                ignore_before_s=skill_ignore_before,
                max_candidates=args.max_skill_candidates,
            )
            clips.extend(
                clip_replay_candidates(
                    video_path=video_path,
                    output_dir=output_dir,
                    candidates=skill_candidates,
                    dry_run=args.dry_run,
                )
            )
            log(f"Skill candidates: {len(skill_candidates)}")
    else:
        log("[3/4] Skipping skill/dribble detection.")

    if not args.skip_chances:
        with timed_step("[3/4] Shot/chance candidate detection"):
            chance_candidates = detect_chance_candidates(
                video_path=video_path,
                output_dir=output_dir,
                audio_windows=audio_windows,
                scene_cuts=scene_cuts,
                goal_candidates=goal_candidates,
                ignore_before_s=infer_match_start_from_reads(reads),
                max_candidates=args.max_chance_candidates,
            )
            clips.extend(
                clip_replay_candidates(
                    video_path=video_path,
                    output_dir=output_dir,
                    candidates=chance_candidates,
                    dry_run=args.dry_run,
                )
            )
            log(f"Chance candidates: {len(chance_candidates)}")
    else:
        log("[3/4] Skipping shot/chance detection.")

    if not args.skip_broadcast_text:
        with timed_step("[3/4] Broadcast text overlay scanning"):
            broadcast_text_candidates = detect_broadcast_text_candidates(
                video_path=video_path,
                output_dir=output_dir,
                api_base=args.api_base,
                model=args.model,
                sample_interval_s=args.broadcast_text_sample_interval,
                max_frames=args.broadcast_text_max_frames,
                support_candidates=replay_candidates + stoppage_candidates + skill_candidates + chance_candidates,
                audio_windows=audio_windows,
                api_key_env=args.api_key_env,
            )
            clips.extend(
                clip_replay_candidates(
                    video_path=video_path,
                    output_dir=output_dir,
                    candidates=broadcast_text_candidates,
                    dry_run=args.dry_run,
                )
            )
            log(f"Broadcast text candidates: {len(broadcast_text_candidates)}")
    else:
        log("[3/4] Skipping broadcast text scanning.")

    with timed_step("Write ingest manifest"):
        write_manifest(
            output_dir=output_dir,
            source_video=video_path,
            reads=reads,
            candidates=goal_candidates,
            clips=clips,
            dry_run=args.dry_run,
            audio_windows=audio_windows,
            scene_cuts=scene_cuts,
            replay_candidates=replay_candidates,
            broadcast_text_candidates=broadcast_text_candidates,
            stoppage_candidates=stoppage_candidates,
            skill_candidates=skill_candidates,
            chance_candidates=chance_candidates,
        )

    classifications = []
    if not args.skip_validation and (
        replay_candidates
        or broadcast_text_candidates
        or stoppage_candidates
        or skill_candidates
        or chance_candidates
    ):
        with timed_step("[4/4] VLM validation"):
            classifications = validate_review_clips(
                output_dir=output_dir,
                api_base=args.api_base,
                model=args.model,
                min_confidence=args.validation_min_confidence,
                frames_per_clip=args.validation_frames_per_clip,
                dry_run=args.dry_run,
                api_key_env=args.api_key_env,
            )
            log(
                "Validated clips: "
                f"{len(classifications)} promoted: "
                f"{sum(item.should_promote for item in classifications)}"
            )
    else:
        log("[4/4] Skipping VLM validation.")

    linked_events = []
    if not args.skip_linking:
        with timed_step("Link promoted clips into events"):
            linked_events = link_events(output_dir)
            log(f"Linked events: {len(linked_events)}")

    if not args.skip_report:
        with timed_step("Generate HTML report"):
            generate_report(output_dir)

    print("")
    log("Done.")
    print(f"Output: {output_dir}")
    print(f"OCR reads: {len(reads)} parsed: {sum(read.parsed for read in reads)}")
    print(f"Goals: {len(goal_candidates)}")
    print(f"Audio windows: {len(audio_windows)}")
    print(f"Scene cuts: {len(scene_cuts)}")
    print(f"Replay candidates: {len(replay_candidates)}")
    print(f"Stoppage candidates: {len(stoppage_candidates)}")
    print(f"Skill candidates: {len(skill_candidates)}")
    print(f"Chance candidates: {len(chance_candidates)}")
    print(f"Broadcast text candidates: {len(broadcast_text_candidates)}")
    print(
        "Validated clips: "
        f"{len(classifications)} promoted: "
        f"{sum(item.should_promote for item in classifications)}"
    )
    print(f"Linked events: {len(linked_events)}")
    if not args.skip_report:
        print(f"Report: {output_dir / 'run_report.html'}")

def preflight_vlm_api(api_base: str, model: str, api_key_env: str) -> None:
    headers = {}
    if api_key_env:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise SystemExit(
                f"VLM is enabled, but environment variable {api_key_env!r} is not set. "
                "Set it or choose Skip VLM."
            )
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with OK.",
            }
        ],
        "temperature": 0,
        "max_tokens": 8,
    }

    try:
        response = requests.post(
            f"{api_base.rstrip('/')}/chat/completions",
            json=payload,
            headers=headers or None,
            timeout=30,
        )
    except requests.RequestException as exc:
        raise SystemExit(
            "VLM is enabled, but the configured API is not reachable. "
            f"API base: {api_base}. Start the server, fix config.json, or choose Skip VLM. "
            f"Details: {exc}"
        ) from exc

    if response.status_code in {401, 403}:
        raise SystemExit(
            "VLM is enabled, but the API rejected authentication. "
            "Check config.json api_key_env and make sure the environment variable is set correctly, "
            "or choose Skip VLM."
        )

    if response.status_code == 404:
        raise SystemExit(
            "VLM is enabled, but the API endpoint or model was not found. "
            f"API base: {api_base}, model: {model}. Fix config.json or choose Skip VLM."
        )

    if not response.ok:
        raise SystemExit(
            "VLM is enabled, but the API preflight failed. "
            f"Status {response.status_code}: {response.text[:500]}"
        )


if __name__ == "__main__":
    main()
