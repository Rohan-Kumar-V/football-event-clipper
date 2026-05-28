from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


def generate_report(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    manifest = read_json(output_dir / "manifest.json", default={})
    linked_events = read_json(output_dir / "linked_events.json", default=[])
    decisions = read_json(output_dir / "review_decisions.json", default={})
    candidate_decisions = read_json(output_dir / "candidate_decisions.json", default={})
    classifications = read_json(
        output_dir / "validation" / "classification_manifest.json",
        default=[],
    )
    export_manifest = read_json(
        output_dir.parent / "final_clips" / "export_manifest.json",
        default=None,
    )

    report = {
        "output_dir": str(output_dir),
        "source_video": manifest.get("source_video"),
        "counts": {
            "ocr_reads": manifest.get("ocr_reads", 0),
            "parsed_ocr_reads": manifest.get("parsed_ocr_reads", 0),
            "audio_windows": manifest.get("audio_windows", 0),
            "audio_spikes": manifest.get("audio_spikes", 0),
            "scene_cuts": manifest.get("scene_cuts", 0),
            "goal_candidates": len(manifest.get("goal_candidates", [])),
            "replay_candidates": len(manifest.get("replay_candidates", [])),
            "stoppage_candidates": len(manifest.get("stoppage_candidates", [])),
            "skill_candidates": len(manifest.get("skill_candidates", [])),
            "chance_candidates": len(manifest.get("chance_candidates", [])),
            "broadcast_text_candidates": len(
                manifest.get("broadcast_text_candidates", [])
            ),
            "clips": len(manifest.get("clips", [])),
            "linked_events": len(linked_events),
            "manual_candidate_approvals": sum(
                1
                for item in candidate_decisions.values()
                if item.get("status") == "approved"
            ),
            "validated_clips": len(classifications),
            "promoted_clips": sum(
                1 for item in classifications if item.get("should_promote")
            ),
        },
        "event_types": dict(Counter(event.get("event_type") for event in linked_events)),
        "review_status": review_status_counts(linked_events, decisions),
        "events": build_event_rows(linked_events, decisions),
        "warnings": build_warnings(manifest, linked_events, classifications),
        "export": export_manifest,
    }

    write_json(output_dir / "run_report.json", report)
    write_html(output_dir / "run_report.html", render_html(report))
    return report


def build_event_rows(linked_events: list[dict], decisions: dict) -> list[dict]:
    rows = []
    for event in linked_events:
        decision = decisions.get(event.get("event_id"), {})
        rows.append(
            {
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "final_event_type": decision.get("event_type", event.get("event_type")),
                "review_status": decision.get("status", "pending"),
                "review_notes": decision.get("notes", ""),
                "timestamp_s": event.get("canonical_timestamp_s"),
                "confidence": event.get("confidence"),
                "clip_count": len(event.get("clips", [])),
                "clips": event.get("clips", []),
                "score_change": event.get("score_change"),
            }
        )
    return rows


def review_status_counts(linked_events: list[dict], decisions: dict) -> dict[str, int]:
    counts = Counter()
    for event in linked_events:
        decision = decisions.get(event.get("event_id"), {})
        counts[decision.get("status", "pending")] += 1
    return dict(counts)


def build_warnings(
    manifest: dict,
    linked_events: list[dict],
    classifications: list[dict],
) -> list[str]:
    warnings = []
    if not manifest:
        warnings.append("No manifest.json found.")
    if not linked_events:
        warnings.append("No linked events found.")
    if manifest and manifest.get("parsed_ocr_reads", 0) == 0:
        warnings.append("No parsed scoreboard reads.")
    if manifest and len(manifest.get("broadcast_text_candidates", [])) == 0:
        warnings.append("No broadcast text candidates found.")
    uncertain = [
        item
        for item in classifications
        if item.get("label") in {"uncertain", "no_event"}
    ]
    if classifications and len(uncertain) / len(classifications) > 0.5:
        warnings.append("More than half of validated clips were uncertain/no_event.")
    return warnings


def render_html(report: dict[str, Any]) -> str:
    counts = report["counts"]
    rows = "\n".join(render_event_card(event) for event in report["events"])
    warnings = "".join(f"<li>{esc(item)}</li>" for item in report["warnings"])
    count_cards = "".join(
        f"<div class='metric'><span>{esc(key.replace('_', ' '))}</span><strong>{value}</strong></div>"
        for key, value in counts.items()
    )
    review_cards = "".join(
        f"<div class='metric'><span>{esc(key)}</span><strong>{value}</strong></div>"
        for key, value in report["review_status"].items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Football Run Report</title>
  <style>
    body {{ margin: 0; font-family: Inter, system-ui, Segoe UI, sans-serif; background: #f6f7f9; color: #17202a; }}
    header {{ background: #fff; border-bottom: 1px solid #d9e0e7; padding: 18px 24px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    main {{ padding: 18px; display: grid; gap: 18px; }}
    section {{ background: #fff; border: 1px solid #d9e0e7; border-radius: 8px; padding: 14px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 10px; }}
    .metric {{ border: 1px solid #d9e0e7; border-radius: 8px; padding: 10px; background: #fbfcfd; }}
    .metric span {{ display: block; color: #647181; font-size: 12px; }}
    .metric strong {{ display: block; font-size: 22px; margin-top: 4px; }}
    .event {{ border-top: 1px solid #d9e0e7; padding: 12px 0; }}
    .event:first-child {{ border-top: 0; }}
    .pill {{ display: inline-flex; border: 1px solid #d9e0e7; border-radius: 999px; padding: 3px 9px; margin: 2px; color: #647181; font-size: 12px; }}
    a {{ color: #0d6b57; }}
    ul {{ margin-top: 8px; }}
  </style>
</head>
<body>
  <header>
    <h1>Football Run Report</h1>
    <div>{esc(report.get("source_video") or "")}</div>
    <div>{esc(report["output_dir"])}</div>
  </header>
  <main>
    <section>
      <h2>Counts</h2>
      <div class="grid">{count_cards}</div>
    </section>
    <section>
      <h2>Review Status</h2>
      <div class="grid">{review_cards or "<p>No review decisions yet.</p>"}</div>
    </section>
    <section>
      <h2>Warnings</h2>
      {f"<ul>{warnings}</ul>" if warnings else "<p>No warnings.</p>"}
    </section>
    <section>
      <h2>Linked Events</h2>
      {rows or "<p>No events.</p>"}
    </section>
  </main>
</body>
</html>"""


def render_event_card(event: dict[str, Any]) -> str:
    clips = "".join(
        f"<li><a href='{file_uri(clip.get('path'))}'>{esc(clip.get('role', 'clip'))}</a> "
        f"<span class='pill'>{esc(clip.get('label', ''))}</span></li>"
        for clip in event.get("clips", [])
    )
    score_change = event.get("score_change")
    score_text = ""
    if score_change:
        score_text = (
            f"<span class='pill'>score {score_change.get('previous_score')} -> "
            f"{score_change.get('new_score')}</span>"
        )
    return f"""<div class="event">
  <h3>{esc(event.get("event_id", ""))}</h3>
  <div>
    <span class="pill">{esc(event.get("event_type", ""))}</span>
    <span class="pill">final {esc(event.get("final_event_type", ""))}</span>
    <span class="pill">review {esc(event.get("review_status", ""))}</span>
    <span class="pill">{round(float(event.get("timestamp_s") or 0))}s</span>
    <span class="pill">confidence {float(event.get("confidence") or 0):.2f}</span>
    {score_text}
  </div>
  <ul>{clips}</ul>
</div>"""


def file_uri(path: str | None) -> str:
    if not path:
        return "#"
    return Path(path).resolve().as_uri()


def esc(value: Any) -> str:
    return html.escape(str(value))


def read_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_html(path: Path, html_text: str) -> None:
    path.write_text(html_text, encoding="utf-8")
