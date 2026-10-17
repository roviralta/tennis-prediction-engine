from dataclasses import dataclass
import re


_SCORE_TOKEN = re.compile(
    r"\[(?P<match_a>\d{1,2})-(?P<match_b>\d{1,2})\]|"
    r"(?P<a>\d{1,2})-(?P<b>\d{1,2})(?:\((?P<tiebreak>\d{1,2})\))?"
)
_STATUS_SUFFIX = re.compile(
    r"\s+(?P<status>RET|DEF|ABD|W/O|WO|DEFAULT|WALKOVER|PLAYED AND ABANDONED)\s*$",
    re.IGNORECASE,
)
_STATUS_ONLY = {
    "RET": "retired",
    "DEF": "default",
    "ABD": "abandoned",
    "W/O": "walkover",
    "WO": "walkover",
    "DEFAULT": "default",
    "WALKOVER": "walkover",
    "PLAYED AND ABANDONED": "abandoned",
}


@dataclass(frozen=True)
class ParsedMatchScore:
    """Parsed score data; derived result values are populated only when complete."""

    status: str
    set_scores: tuple[tuple[int, int], ...] = ()
    winner_sets: int | None = None
    loser_sets: int | None = None
    winner_games: int | None = None
    loser_games: int | None = None
    total_games: int | None = None
    set_score: str | None = None
    has_match_tiebreak: bool = False


def _score_side_is_complete_set(a: int, b: int) -> bool:
    return (max(a, b) >= 6 and abs(a - b) >= 2) or sorted((a, b)) == [6, 7]


def parse_match_score(score: object, best_of: object) -> ParsedMatchScore:
    """Parse an ATP score conservatively, rejecting incomplete or ambiguous results.

    The archive records the match winner's score first. A score is accepted only
    when that side wins the required number of sets for a best-of-three or
    best-of-five match. Retirements, defaults, walkovers, malformed scores, and
    nonstandard match formats do not produce market targets.
    """

    if score is None:
        return ParsedMatchScore(status="missing")
    text = str(score).strip()
    if not text:
        return ParsedMatchScore(status="missing")

    marker = _STATUS_SUFFIX.search(text)
    outcome_status: str | None = None
    if marker:
        outcome_status = _STATUS_ONLY[marker.group("status").upper()]
        text = text[: marker.start()].strip()
    elif text.upper() in _STATUS_ONLY:
        return ParsedMatchScore(status=_STATUS_ONLY[text.upper()])

    tokens = list(_SCORE_TOKEN.finditer(text))
    if not tokens:
        return ParsedMatchScore(status=outcome_status or "unparsed")

    previous_end = 0
    set_scores: list[tuple[int, int]] = []
    match_tiebreaks: list[bool] = []
    invalid_token = False
    for token in tokens:
        if text[previous_end : token.start()].strip():
            invalid_token = True
            break
        is_match_tiebreak = token.group("match_a") is not None
        a = int(token.group("match_a" if is_match_tiebreak else "a"))
        b = int(token.group("match_b" if is_match_tiebreak else "b"))
        if a == b:
            invalid_token = True
            break
        if is_match_tiebreak:
            if max(a, b) < 10 or abs(a - b) < 2:
                invalid_token = True
                break
        elif token.group("tiebreak") is not None and sorted((a, b)) != [6, 7]:
            invalid_token = True
            break
        set_scores.append((a, b))
        match_tiebreaks.append(is_match_tiebreak)
        previous_end = token.end()

    if invalid_token or text[previous_end:].strip():
        return ParsedMatchScore(status=outcome_status or "unparsed")
    if outcome_status:
        return ParsedMatchScore(
            status=outcome_status,
            set_scores=tuple(set_scores),
            has_match_tiebreak=any(match_tiebreaks),
        )
    if not all(
        is_match_tiebreak or _score_side_is_complete_set(a, b)
        for (a, b), is_match_tiebreak in zip(set_scores, match_tiebreaks, strict=True)
    ):
        return ParsedMatchScore(status="partial_score", set_scores=tuple(set_scores))

    try:
        best_of_value = int(best_of)
    except (TypeError, ValueError):
        return ParsedMatchScore(status="invalid_best_of", set_scores=tuple(set_scores))
    if best_of_value not in (3, 5):
        return ParsedMatchScore(status="unsupported_best_of", set_scores=tuple(set_scores))

    sets_to_win = best_of_value // 2 + 1
    winner_sets = sum(a > b for a, b in set_scores)
    loser_sets = sum(a < b for a, b in set_scores)
    if (
        winner_sets != sets_to_win
        or loser_sets >= sets_to_win
        or len(set_scores) > (2 * sets_to_win) - 1
        or winner_sets + loser_sets != len(set_scores)
    ):
        return ParsedMatchScore(status="score_result_mismatch", set_scores=tuple(set_scores))

    has_match_tiebreak = any(match_tiebreaks)
    winner_games = None if has_match_tiebreak else sum(a for a, _ in set_scores)
    loser_games = None if has_match_tiebreak else sum(b for _, b in set_scores)
    return ParsedMatchScore(
        status="complete_with_match_tiebreak" if has_match_tiebreak else "complete",
        set_scores=tuple(set_scores),
        winner_sets=winner_sets,
        loser_sets=loser_sets,
        winner_games=winner_games,
        loser_games=loser_games,
        total_games=(winner_games + loser_games) if winner_games is not None else None,
        set_score=f"{winner_sets}-{loser_sets}",
        has_match_tiebreak=has_match_tiebreak,
    )
