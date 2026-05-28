from __future__ import annotations

from collections.abc import Iterable

from football_ingest.models import GoalCandidate, ScoreboardRead


def detect_goal_candidates(
    reads: Iterable[ScoreboardRead],
    min_confidence: float = 0.90,
    min_seconds_between_goals: float = 45.0,
) -> list[GoalCandidate]:
    valid_reads = [
        read
        for read in reads
        if read.parsed
        and read.score1 is not None
        and read.score2 is not None
        and read.confidence >= min_confidence
    ]
    valid_reads.sort(key=lambda read: read.timestamp_s)

    candidates: list[GoalCandidate] = []
    previous_score: tuple[int, int] | None = None
    previous_read: ScoreboardRead | None = None
    last_goal_timestamp = -10_000.0

    for read in valid_reads:
        current_score = (int(read.score1), int(read.score2))

        if previous_score is None:
            previous_score = current_score
            previous_read = read
            continue

        if current_score == previous_score:
            previous_read = read
            continue

        if is_valid_score_increase(previous_score, current_score):
            if read.timestamp_s - last_goal_timestamp >= min_seconds_between_goals:
                scoring_side = "team1" if current_score[0] > previous_score[0] else "team2"
                candidates.append(
                    GoalCandidate(
                        event="goal",
                        timestamp_s=read.timestamp_s,
                        scoring_side=scoring_side,
                        previous_score=previous_score,
                        new_score=current_score,
                        confidence=min(0.99, read.confidence),
                        evidence={
                            "scoreboard_changed": True,
                            "previous_timestamp_s": previous_read.timestamp_s
                            if previous_read
                            else None,
                            "current_timestamp_s": read.timestamp_s,
                            "previous_raw_text": previous_read.raw_text
                            if previous_read
                            else None,
                            "current_raw_text": read.raw_text,
                            "team1": read.team1,
                            "team2": read.team2,
                            "ocr_confidence": read.confidence,
                        },
                    )
                )
                last_goal_timestamp = read.timestamp_s

            previous_score = current_score
            previous_read = read
            continue

        previous_read = read

    return candidates


def is_valid_score_increase(previous_score: tuple[int, int], current_score: tuple[int, int]) -> bool:
    delta1 = current_score[0] - previous_score[0]
    delta2 = current_score[1] - previous_score[1]
    if delta1 < 0 or delta2 < 0:
        return False
    if delta1 == 0 and delta2 == 0:
        return False
    if delta1 > 0 and delta2 > 0:
        return False
    return max(delta1, delta2) <= 3
