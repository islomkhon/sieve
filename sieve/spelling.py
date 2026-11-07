"""Deterministic spelling recovery for the Sieve prototype.

There is no language model here. A misspelling is corrected only when exactly
one known word of the expected grammatical category is close enough. The
category matters: "hat" can become "what" in a question position, but it is not
allowed to become an arbitrary noun.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import english
from . import lexicon as lx


COMMON_MISSPELLINGS = {
    "accomodate": "accommodate",
    "acheive": "achieve",
    "adress": "address",
    "arguement": "argument",
    "becuase": "because",
    "beleive": "believe",
    "calender": "calendar",
    "comming": "coming",
    "definately": "definitely",
    "definiton": "definition",
    "enviroment": "environment",
    "existance": "existence",
    "freind": "friend",
    "goverment": "government",
    "independant": "independent",
    "liason": "liaison",
    "maintainance": "maintenance",
    "neccessary": "necessary",
    "occassion": "occasion",
    "occured": "occurred",
    "ocurrence": "occurrence",
    "posession": "possession",
    "priviledge": "privilege",
    "recieve": "receive",
    "seperate": "separate",
    "succesful": "successful",
    "teh": "the",
    "thier": "their",
    "tommorow": "tomorrow",
    "truely": "truly",
    "untill": "until",
    "wierd": "weird",
    "writting": "writing",
}


CATEGORY_WORDS = {
    "action": lx.ACTION_VERBS,
    "question": lx.QUESTION_STARTS,
    "color": {"color"},
    "determiner": lx.DETERMINERS | lx.PRONOUNS,
    "adjective": lx.ADJECTIVES,
    "noun": lx.NOUNS | {"one"},
    "relation": lx.PREPOSITION_STARTS | {"there", "back", "here"},
    "auxiliary": lx.AUXILIARIES,
    "to": {"to"},
    "of": {"of"},
    "correction": {"actually", "no", "not"},
    "context": lx.DISCOURSE_WORDS | {"there", "back"},
    "control": {
        "show",
        "help",
        "why",
        "reset",
        "cancel",
        "state",
        "explain",
        "context",
        "memory",
        "remember",
        "forget",
    },
    "any": lx.ALL_WORDS,
}

GENERAL_POSITIONS = {
    "action": ("verb",),
    "adjective": ("adjective",),
    "noun": ("noun",),
    "any": tuple(english.CATEGORIES),
}


@dataclass(frozen=True)
class Correction:
    original: str
    corrected: str
    distance: int


def max_distance_for(word: str) -> int:
    if len(word) <= 6:
        return 1
    if len(word) <= 9:
        return 2
    return 3


def damerau_levenshtein(a: str, b: str, limit: int | None = None) -> int:
    """Damerau-Levenshtein distance, with an optional early cutoff."""

    if a == b:
        return 0
    if limit is not None and abs(len(a) - len(b)) > limit:
        return limit + 1

    previous_previous: list[int] | None = None
    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        row_min = current[0]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
            if (
                previous_previous is not None
                and i > 1
                and j > 1
                and char_a == b[j - 2]
                and a[i - 2] == char_b
            ):
                current[j] = min(current[j], previous_previous[j - 2] + cost)
            row_min = min(row_min, current[j])

        if limit is not None and row_min > limit:
            return limit + 1
        previous_previous, previous = previous, current

    return previous[-1]


def correct_word_candidates(
    token: str,
    categories: set[str],
    extra_words: set[str] | None = None,
) -> list[Correction]:
    """Return the closest corrections in the expected categories."""

    token = token.lower()
    if token in COMMON_MISSPELLINGS:
        corrected = COMMON_MISSPELLINGS[token]
        return [Correction(token, corrected, max_distance_for(token))]
    candidates: set[str] = set()
    for category in categories:
        candidates |= CATEGORY_WORDS.get(category, set())

    if extra_words and ("noun" in categories or "any" in categories):
        candidates |= extra_words

    if token in candidates:
        return [Correction(token, token, 0)]
    if not candidates:
        return []

    limit = max_distance_for(token)
    scored = _score_candidates(token, candidates, limit)
    if scored:
        return scored

    positions: list[str] = []
    for category in categories:
        for position in GENERAL_POSITIONS.get(category, ()):
            if position not in positions:
                positions.append(position)
    if not positions:
        return []

    general = _general_candidates(token, tuple(sorted(positions)), limit)
    if not general:
        return []
    best_distance = general[0].distance
    return [item for item in general if item.distance == best_distance]


@lru_cache(maxsize=4096)
def open_word_options(token: str, limit: int = 10) -> tuple[Correction, ...]:
    """Nearby open-vocabulary spellings, excluding the token itself.

    These are candidates, not decisions. The dictionary path tests them in a
    phrase context and accepts a correction only when it yields one entry.
    """

    token = token.lower()
    if token in COMMON_MISSPELLINGS:
        return (Correction(token, COMMON_MISSPELLINGS[token], max_distance_for(token)),)
    positions = tuple(sorted(english.CATEGORIES))
    candidates = _general_candidates(token, positions, max_distance_for(token))
    return tuple(item for item in candidates if item.corrected != token)[:limit]


def correct_domain_word(
    token: str,
    categories: set[str],
    extra_words: set[str] | None = None,
) -> Correction | None:
    """Correct only against the declared world vocabulary."""

    token = token.lower()
    candidates: set[str] = set()
    for category in categories:
        candidates |= CATEGORY_WORDS.get(category, set())
    if extra_words and ("noun" in categories or "any" in categories):
        candidates |= extra_words
    if token in candidates:
        return Correction(token, token, 0)
    if not candidates:
        return None
    scored = _score_candidates(token, candidates, max_distance_for(token))
    return scored[0] if len(scored) == 1 else None


def _score_candidates(
    token: str, candidates: set[str], limit: int
) -> list[Correction]:
    scored: list[Correction] = []
    for candidate in candidates:
        distance = damerau_levenshtein(token, candidate, limit)
        if distance <= limit:
            scored.append(Correction(token, candidate, distance))
    if not scored:
        return []
    scored.sort(
        key=lambda item: (
            item.distance,
            abs(len(token) - len(item.corrected)),
            item.corrected,
        )
    )
    best_distance = scored[0].distance
    return [item for item in scored if item.distance == best_distance]


@lru_cache(maxsize=4096)
def _general_candidates(
    token: str, positions: tuple[str, ...], limit: int
) -> tuple[Correction, ...]:
    scored: list[Correction] = []
    for candidate in english.lexicon().words_for_positions(positions):
        if abs(len(token) - len(candidate)) > limit:
            continue
        distance = damerau_levenshtein(token, candidate, limit)
        if distance <= limit:
            scored.append(Correction(token, candidate, distance))
    scored.sort(
        key=lambda item: (
            item.distance,
            abs(len(token) - len(item.corrected)),
            item.corrected,
        )
    )
    return tuple(scored)


def correct_word(
    token: str,
    categories: set[str],
    extra_words: set[str] | None = None,
) -> Correction | None:
    """Return a correction only when the best candidate is unique."""

    candidates = correct_word_candidates(token, categories, extra_words)
    if len(candidates) == 1:
        return candidates[0]
    return None
