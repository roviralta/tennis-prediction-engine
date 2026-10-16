import re
from collections.abc import Sequence

from tennis_prediction.datasets.audit import FeatureDefinition


_STATIC_FEATURE_DEFINITIONS = {
    "surface": (
        "ATP tournament metadata",
        "Tournament surface is recorded before the target match",
    ),
    "draw_size": (
        "ATP tournament metadata",
        "Tournament draw size is known before the target match",
    ),
    "tourney_level": (
        "ATP tournament metadata",
        "Tournament level is known before the target match",
    ),
    "best_of": (
        "ATP tournament metadata",
        "Match format is known before the target match",
    ),
    "round": (
        "ATP tournament metadata",
        "Scheduled tournament round is known before the target match",
    ),
    "player_a_rank": (
        "ATP ranking archive",
        "Latest ranking with a ranking date on or before the tournament date",
    ),
    "player_b_rank": (
        "ATP ranking archive",
        "Latest ranking with a ranking date on or before the tournament date",
    ),
    "player_a_rank_points": (
        "ATP ranking archive",
        "Latest ranking points with a ranking date on or before the tournament date",
    ),
    "player_b_rank_points": (
        "ATP ranking archive",
        "Latest ranking points with a ranking date on or before the tournament date",
    ),
    "rank_difference": (
        "ATP ranking archive",
        "Difference between each player's latest ranking on or before the tournament date",
    ),
    "rank_points_difference": (
        "ATP ranking archive",
        "Difference between each player's latest ranking points on or before the tournament date",
    ),
    "player_a_age": (
        "ATP player birth-date metadata",
        "Age derived from date of birth and target tournament date",
    ),
    "player_b_age": (
        "ATP player birth-date metadata",
        "Age derived from date of birth and target tournament date",
    ),
    "head_to_head_matches": (
        "Historical ATP match results",
        "Count only meetings with tournament dates strictly before the target date",
    ),
    "player_a_head_to_head_wins": (
        "Historical ATP match results",
        "Count only meetings won by player A with tournament dates strictly before the target date",
    ),
    "player_b_head_to_head_wins": (
        "Historical ATP match results",
        "Count only meetings won by player B with tournament dates strictly before the target date",
    ),
    "player_a_head_to_head_win_rate": (
        "Historical ATP match results",
        "Win rate over meetings with tournament dates strictly before the target date",
    ),
    "player_b_head_to_head_win_rate": (
        "Historical ATP match results",
        "Win rate over meetings with tournament dates strictly before the target date",
    ),
    "head_to_head_win_rate_difference": (
        "Historical ATP match results",
        "Difference between head-to-head rates from meetings strictly before the target date",
    ),
}

_GENERAL_FORM_PATTERN = re.compile(
    r"(?:player_[ab]_win_rate|win_rate_difference)_last_(\d+)"
)
_SURFACE_FORM_PATTERN = re.compile(
    r"(?:player_[ab]_surface_win_rate|surface_win_rate_difference)_last_(\d+)"
)


def candidate_pre_match_feature_columns(window: int = 10) -> tuple[str, ...]:
    """Return the current reviewed candidate feature set for a chosen form window."""

    if window < 1:
        raise ValueError("window must be greater than zero")

    columns = [
        "surface",
        "draw_size",
        "tourney_level",
        "best_of",
        "round",
        "player_a_age",
        "player_b_age",
        "player_a_rank",
        "player_b_rank",
        "player_a_rank_points",
        "player_b_rank_points",
        "rank_difference",
        "rank_points_difference",
        f"player_a_win_rate_last_{window}",
        f"player_b_win_rate_last_{window}",
        f"win_rate_difference_last_{window}",
        f"player_a_surface_win_rate_last_{window}",
        f"player_b_surface_win_rate_last_{window}",
        f"surface_win_rate_difference_last_{window}",
        "head_to_head_matches",
        "player_a_head_to_head_wins",
        "player_b_head_to_head_wins",
        "player_a_head_to_head_win_rate",
        "player_b_head_to_head_win_rate",
        "head_to_head_win_rate_difference",
    ]
    feature_definitions_for(columns)
    return tuple(columns)


def feature_definitions_for(
    feature_columns: Sequence[str],
) -> tuple[FeatureDefinition, ...]:
    """Return reviewed pre-match definitions for supported feature names.

    Features based on player hand or height are intentionally not catalogued:
    the local player profile has no effective-date history to establish that
    those values were known as recorded for every historical match.
    """

    definitions = []
    for name in feature_columns:
        if name in _STATIC_FEATURE_DEFINITIONS:
            source, temporal_rule = _STATIC_FEATURE_DEFINITIONS[name]
        elif match := _GENERAL_FORM_PATTERN.fullmatch(name):
            if int(match.group(1)) < 1:
                raise ValueError(f"Recent-form window must be positive for feature {name!r}")
            source = "Historical ATP match results"
            temporal_rule = (
                "Uses the configured number of prior matches with tournament dates "
                "strictly before the target date"
            )
        elif match := _SURFACE_FORM_PATTERN.fullmatch(name):
            if int(match.group(1)) < 1:
                raise ValueError(f"Recent-form window must be positive for feature {name!r}")
            source = "Historical ATP match results and tournament surface metadata"
            temporal_rule = (
                "Uses prior matches on the target surface with tournament dates "
                "strictly before the target date"
            )
        else:
            raise ValueError(f"No reviewed pre-match feature definition for {name!r}")

        definitions.append(
            FeatureDefinition(
                name=name,
                source=source,
                temporal_rule=temporal_rule,
                available_before_match=True,
            )
        )

    if len({definition.name for definition in definitions}) != len(definitions):
        raise ValueError("feature_columns cannot contain duplicates")
    return tuple(definitions)
