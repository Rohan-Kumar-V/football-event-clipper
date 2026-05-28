"""Deterministic event folder taxonomy."""

from football_ingest.labels import EVENT_LABELS


EVENT_FOLDERS = EVENT_LABELS


def ensure_event_folders(output_dir):
    for folder in EVENT_FOLDERS:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)
