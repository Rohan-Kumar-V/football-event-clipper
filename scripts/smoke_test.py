import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from football_ingest.goals import detect_goal_candidates
from football_ingest.broadcast_text import keyword_label_hints
from football_ingest.models import ReplayCandidate, ScoreboardRead
from football_ingest.ocr import parse_scoreboard_text
from football_ingest.skills import suppress_overlapping_candidates
from football_ingest.validate import validation_profile_from_metadata


def main() -> None:
    assert parse_scoreboard_text("MCI 1-0 ARS", 0.95, 0.90) == {
        "team1": "MCI",
        "team2": "ARS",
        "score1": 1,
        "score2": 0,
    }
    assert parse_scoreboard_text("MCI10ARS", 0.95, 0.90) == {
        "team1": "MCI",
        "team2": "ARS",
        "score1": 1,
        "score2": 0,
    }
    assert parse_scoreboard_text("MCI 1-0 ARS", 0.50, 0.90) == {}

    reads = [
        ScoreboardRead(10, 1, "MCI 0-0 ARS", 0.96, "MCI", "ARS", 0, 0, True),
        ScoreboardRead(70, 2, "MCI 1-0 ARS", 0.97, "MCI", "ARS", 1, 0, True),
    ]
    candidates = detect_goal_candidates(reads)
    assert len(candidates) == 1
    assert candidates[0].event == "goal"
    assert candidates[0].scoring_side == "team1"
    assert candidates[0].previous_score == (0, 0)
    assert candidates[0].new_score == (1, 0)

    assert keyword_label_hints("PENALTY CHECK") == ["penalty_kick"]
    assert keyword_label_hints("FRANCE VARANE KOUNDE") == []

    assert (
        validation_profile_from_metadata(
            {
                "evidence": {
                    "candidate_type": "skill_segment",
                    "label_hints": ["skill_dribble", "nutmeg"],
                }
            }
        )
        == "skill_dribble"
    )
    assert (
        validation_profile_from_metadata(
            {
                "evidence": {
                    "candidate_type": "chance_segment",
                    "label_hints": ["shot_on_target", "big_chance"],
                }
            }
        )
        == "attacking_replay"
    )

    overlapping = [
        ReplayCandidate("review_required", "skill_segment", 10, 10, 20, 0.8, {}),
        ReplayCandidate("review_required", "skill_segment", 12, 12, 21, 0.7, {}),
        ReplayCandidate("review_required", "skill_segment", 40, 40, 50, 0.6, {}),
    ]
    assert len(suppress_overlapping_candidates(overlapping, 0.2)) == 2

    print("smoke_test passed")


if __name__ == "__main__":
    main()
