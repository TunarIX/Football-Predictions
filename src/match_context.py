"""Competition and tournament categorisation helpers."""

from __future__ import annotations

from enum import StrEnum
import pandas as pd


class CompetitionType(StrEnum):
    CLUB = "club"
    INTERNATIONAL = "international"


class TournamentCategory(StrEnum):
    WORLD_CUP = "World Cup"
    CONTINENTAL = "Continental tournament"
    QUALIFIER = "Qualifier"
    NATIONS_LEAGUE = "Nations League"
    FRIENDLY = "Friendly"
    OTHER = "Other"


INTERNATIONAL_SELECTORS = {"fifa world cup", "international matches"}
CONTINENTAL_TERMS = (
    "uefa euro",
    "european championship",
    "copa america",
    "africa cup of nations",
    "afcon",
    "asian cup",
    "gold cup",
    "concacaf championship",
    "oceania nations cup",
)


def is_international_competition_name(competition: object) -> bool:
    text = "" if pd.isna(competition) else str(competition).strip().lower()
    return text in INTERNATIONAL_SELECTORS or any(
        term in text
        for term in (
            "world cup",
            "international",
            "nations league",
            "friendly",
            "qualifier",
            *CONTINENTAL_TERMS,
        )
    )


def competition_type(competition: object, configured_match_type: object | None = None) -> str:
    configured = "" if configured_match_type is None or pd.isna(configured_match_type) else str(configured_match_type).strip().lower()
    if configured in {CompetitionType.CLUB.value, CompetitionType.INTERNATIONAL.value}:
        return configured
    return CompetitionType.INTERNATIONAL.value if is_international_competition_name(competition) else CompetitionType.CLUB.value


def tournament_category(competition: object) -> str:
    text = "" if pd.isna(competition) else str(competition).strip().lower()
    if "world cup" in text and "qual" not in text:
        return TournamentCategory.WORLD_CUP.value
    if "nations league" in text:
        return TournamentCategory.NATIONS_LEAGUE.value
    if "qual" in text:
        return TournamentCategory.QUALIFIER.value
    if any(term in text for term in CONTINENTAL_TERMS):
        return TournamentCategory.CONTINENTAL.value
    if "friendly" in text or "friendlies" in text:
        return TournamentCategory.FRIENDLY.value
    return TournamentCategory.OTHER.value


def international_training_weight(competition: object, selected_competition: object | None = None) -> float:
    category = tournament_category(competition)
    selected = "" if selected_competition is None or pd.isna(selected_competition) else str(selected_competition).strip().lower()
    if selected == "fifa world cup":
        return {
            TournamentCategory.WORLD_CUP.value: 3.0,
            TournamentCategory.CONTINENTAL.value: 2.0,
            TournamentCategory.QUALIFIER.value: 2.0,
            TournamentCategory.NATIONS_LEAGUE.value: 2.0,
            TournamentCategory.FRIENDLY.value: 0.6,
            TournamentCategory.OTHER.value: 1.0,
        }.get(category, 1.0)
    return 1.0
