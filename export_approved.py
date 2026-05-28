from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


DEFAULT_OUTPUT = Path("clips") / "match_replay"
DEFAULT_EXPORT = Path("final_clips")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export approved reviewed football clips.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--export", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument(
        "--include-needs-review",
        action="store_true",
        help="Also export events marked needs_review.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output.resolve()
    export_dir = args.export.resolve()

    linked_events = read_json(output_dir / "linked_events.json", default=[])
    decisions = read_json(output_dir / "review_decisions.json", default={})
    candidate_decisions = read_json(output_dir / "candidate_decisions.json", default={})

    exported = export_events(
        output_dir=output_dir,
        linked_events=linked_events,
        decisions=decisions,
        candidate_decisions=candidate_decisions,
        export_dir=export_dir,
        include_needs_review=args.include_needs_review,
    )
    print(f"Exported {len(exported)} events to {export_dir}")


def export_events(
    output_dir: Path,
    linked_events: list[dict],
    decisions: dict,
    candidate_decisions: dict,
    export_dir: Path,
    include_needs_review: bool = False,
) -> list[dict]:
    export_dir.mkdir(parents=True, exist_ok=True)
    exported_events = []

    for event in linked_events:
        decision = decisions.get(event["event_id"])
        if not should_export(decision, include_needs_review):
            continue

        final_type = decision.get("event_type") or event["event_type"]
        event_dir = export_dir / safe_name(final_type) / safe_name(event["event_id"])
        event_dir.mkdir(parents=True, exist_ok=True)

        exported_clips = []
        for index, clip in enumerate(event.get("clips", []), start=1):
            source = Path(clip["path"])
            if not source.exists():
                continue
            filename = f"{index:02d}_{safe_name(clip['role'])}.mp4"
            target = event_dir / filename
            shutil.copy2(source, target)
            exported_clips.append(
                {
                    "source": str(source),
                    "exported": str(target),
                    "role": clip.get("role"),
                    "label": clip.get("label"),
                    "confidence": clip.get("confidence"),
                }
            )

        exported_event = {
            "event_id": event["event_id"],
            "original_event_type": event["event_type"],
            "final_event_type": final_type,
            "review_status": decision.get("status"),
            "review_notes": decision.get("notes", ""),
            "canonical_timestamp_s": event.get("canonical_timestamp_s"),
            "score_change": event.get("score_change"),
            "clips": exported_clips,
        }
        write_json(event_dir / "event.json", exported_event)
        exported_events.append(exported_event)

    exported_events.extend(
        export_candidate_decisions(
            output_dir=output_dir,
            candidate_decisions=candidate_decisions,
            export_dir=export_dir,
            include_needs_review=include_needs_review,
        )
    )

    manifest = {
        "exported_event_count": len(exported_events),
        "events": exported_events,
    }
    write_json(export_dir / "export_manifest.json", manifest)
    return exported_events


def export_candidate_decisions(
    output_dir: Path,
    candidate_decisions: dict,
    export_dir: Path,
    include_needs_review: bool,
) -> list[dict]:
    exported_events = []
    for candidate_id, decision in candidate_decisions.items():
        if not should_export(decision, include_needs_review):
            continue

        source = Path(decision.get("source_clip", ""))
        if not source.exists():
            fallback = output_dir / "review_required" / f"{candidate_id}.mp4"
            source = fallback if fallback.exists() else source
        if not source.exists():
            continue

        final_type = decision.get("event_type") or "manual_event"
        event_id = f"manual_{safe_name(final_type)}_{safe_name(candidate_id)}"
        event_dir = export_dir / safe_name(final_type) / event_id
        event_dir.mkdir(parents=True, exist_ok=True)
        target = event_dir / f"01_manual_candidate_{safe_name(candidate_id)}.mp4"
        shutil.copy2(source, target)

        metadata = read_json(source.with_suffix(".json"), default={})
        exported_event = {
            "event_id": event_id,
            "original_event_type": "manual_candidate",
            "final_event_type": final_type,
            "review_status": decision.get("status"),
            "review_notes": decision.get("notes", ""),
            "canonical_timestamp_s": metadata.get("start_s"),
            "score_change": None,
            "candidate_id": candidate_id,
            "candidate_type": decision.get("candidate_type"),
            "clips": [
                {
                    "source": str(source),
                    "exported": str(target),
                    "role": "manual_candidate_clip",
                    "label": final_type,
                    "confidence": metadata.get("confidence"),
                }
            ],
        }
        write_json(event_dir / "event.json", exported_event)
        exported_events.append(exported_event)
    return exported_events


def should_export(decision: dict | None, include_needs_review: bool) -> bool:
    if not decision:
        return False
    status = decision.get("status")
    return status == "approved" or (include_needs_review and status == "needs_review")


def safe_name(value: str) -> str:
    clean = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)
    return clean.strip("_") or "item"


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
