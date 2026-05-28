"""Shared football event labels and validator-only outcomes."""

EVENT_LABELS = [
    "goal",
    "goal_celebration",
    "penalty_kick",
    "penalty_goal",
    "penalty_saved",
    "penalty_missed",
    "shootout_penalty",
    "own_goal",
    "var_review",
    "referee_monitor_review",
    "yellow_card",
    "red_card",
    "second_yellow_red",
    "substitution",
    "injury_stoppage",
    "medical_treatment",
    "foul",
    "handball",
    "offside",
    "corner_kick",
    "free_kick",
    "direct_free_kick_shot",
    "throw_in",
    "kickoff",
    "half_time",
    "full_time",
    "shot_on_target",
    "shot_off_target",
    "blocked_shot",
    "woodwork_hit",
    "goalkeeper_save",
    "goalkeeper_claim",
    "goalkeeper_punch",
    "clearance",
    "goal_line_clearance",
    "tackle",
    "slide_tackle",
    "interception",
    "aerial_duel",
    "header",
    "cross",
    "through_ball",
    "key_pass",
    "counter_attack",
    "pressing_sequence",
    "skill_dribble",
    "nutmeg",
    "solo_run",
    "trick",
    "big_chance",
    "crowd_reaction",
    "manager_reaction",
    "player_argument",
    "referee_discussion",
    "review_required",
]

VALIDATOR_ONLY_LABELS = [
    "no_event",
    "uncertain",
]

VALIDATION_LABELS = EVENT_LABELS + VALIDATOR_ONLY_LABELS

PROMOTABLE_LABELS = [
    label
    for label in EVENT_LABELS
    if label != "review_required"
]


LABEL_GUIDANCE = {
    "goal": "Use only when the ball enters the goal or a scoreboard-confirmed goal moment is clearly visible.",
    "goal_celebration": "Use when players or fans are clearly celebrating a goal.",
    "penalty_kick": "Use when a penalty setup, run-up, kick, or immediate penalty result is clearly visible.",
    "penalty_goal": "Use when a penalty kick clearly results in a goal.",
    "penalty_saved": "Use when a goalkeeper clearly saves a penalty.",
    "penalty_missed": "Use when a penalty is clearly missed or hits the frame and does not go in.",
    "shootout_penalty": "Use only for a penalty shootout kick, not an in-match penalty.",
    "own_goal": "Use only when the defending team clearly scores into its own goal.",
    "var_review": "Use when VAR room, VAR graphic, VAR check, or VAR review presentation is visible.",
    "referee_monitor_review": "Use when the referee is clearly using the pitchside monitor.",
    "yellow_card": "Use when a yellow card is clearly shown or a yellow-card graphic is visible.",
    "red_card": "Use when a red card is clearly shown or a red-card graphic is visible.",
    "second_yellow_red": "Use when a second-yellow dismissal is clearly shown or indicated.",
    "substitution": "Use when a substitution board, player swap, or substitution graphic is visible.",
    "injury_stoppage": "Use when play is stopped for an apparent injury.",
    "medical_treatment": "Use when medical staff are treating a player.",
    "foul": "Use when a foul contact or foul decision is clearly visible.",
    "handball": "Use when a handball incident or decision is clearly visible.",
    "offside": "Use when an offside decision, flag, or graphic is clearly visible.",
    "corner_kick": "Use when a corner kick setup or taking of the corner is visible.",
    "free_kick": "Use when a free-kick setup or taking of the free kick is visible.",
    "direct_free_kick_shot": "Use when a free kick is clearly shot directly at goal.",
    "throw_in": "Use when a throw-in is clearly taken.",
    "kickoff": "Use for the start/restart from the center spot.",
    "half_time": "Use when half-time is clearly shown or indicated.",
    "full_time": "Use when full-time is clearly shown or indicated.",
    "shot_on_target": "Use when a shot clearly forces a save, enters the goal, or is goal-bound.",
    "shot_off_target": "Use when a shot clearly misses the goal.",
    "blocked_shot": "Use when a shot is clearly blocked by a defender.",
    "woodwork_hit": "Use when the ball clearly hits post or crossbar.",
    "goalkeeper_save": "Use when the goalkeeper clearly saves a shot.",
    "goalkeeper_claim": "Use when the goalkeeper clearly catches/claims the ball.",
    "goalkeeper_punch": "Use when the goalkeeper clearly punches the ball away.",
    "clearance": "Use when a defender clearly clears the ball from danger.",
    "goal_line_clearance": "Use when a defender clearly prevents a goal near/on the line.",
    "tackle": "Use when a player clearly dispossesses/challenges an opponent.",
    "slide_tackle": "Use when the tackle is clearly a sliding challenge.",
    "interception": "Use when a player clearly cuts out a pass.",
    "aerial_duel": "Use when players clearly contest an airborne ball.",
    "header": "Use when a player clearly heads the ball.",
    "cross": "Use when a ball is clearly crossed into the penalty area.",
    "through_ball": "Use when a pass clearly splits defenders into attacking space.",
    "key_pass": "Use when a pass clearly creates a shot or major chance.",
    "counter_attack": "Use when a fast attacking transition is clearly visible.",
    "pressing_sequence": "Use when coordinated pressing clearly forces a turnover or mistake.",
    "skill_dribble": "Use when a player clearly beats one or more defenders while controlling the ball.",
    "nutmeg": "Use when the ball clearly goes through an opponent's legs.",
    "solo_run": "Use when one player carries the ball over distance and beats/evades defenders.",
    "trick": "Use when a clear trick or flair move is visible.",
    "big_chance": "Use when a high-quality chance is clearly shown, even if no goal occurs.",
    "crowd_reaction": "Use when the clip is primarily a crowd reaction.",
    "manager_reaction": "Use when the clip is primarily a manager/coach reaction.",
    "player_argument": "Use when players are clearly arguing or confronting each other.",
    "referee_discussion": "Use when players/staff are clearly discussing a decision with the referee.",
}


def validation_label_text() -> str:
    return ", ".join(VALIDATION_LABELS)


def label_guidance_text() -> str:
    return "\n".join(
        f"- {label}: {guidance}"
        for label, guidance in LABEL_GUIDANCE.items()
    )
