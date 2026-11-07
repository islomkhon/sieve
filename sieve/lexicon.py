"""A tiny, explicit lexicon.

Every word below is a hand-declared constraint, not a statistical association.
Unknown words are reported instead of guessed. That is deliberate: the system
never pretends to understand a token it has not been given a rule for.
"""

from __future__ import annotations

from . import english
from .world import CONTAINER_KINDS, ROOM_KINDS, SUPPORT_KINDS, Entity


DETERMINERS = {
    "a",
    "an",
    "another",
    "any",
    "each",
    "every",
    "its",
    "my",
    "some",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "your",
}
PRONOUNS = {"it", "them", "they"}
POLITE_WORDS = {"please", "kindly"}

COLOR_WORDS = {"red", "blue", "green", "yellow"}
SIZE_WORDS = {"small", "medium", "large"}
STATE_WORDS = {"open", "closed", "full", "empty"}
LOCATION_WORDS = {"living"}
CONTEXT_ADJECTIVES = {"other", "same"}

COLOR_ALIASES = {
    "amber": "yellow",
    "azure": "blue",
    "cerulean": "blue",
    "crimson": "red",
    "emerald": "green",
    "gold": "yellow",
    "golden": "yellow",
    "jade": "green",
    "lime": "green",
    "maroon": "red",
    "navy": "blue",
    "ruby": "red",
    "scarlet": "red",
    "turquoise": "blue",
    "vermilion": "red",
}
SIZE_ALIASES = {
    "average": "medium",
    "big": "large",
    "enormous": "large",
    "huge": "large",
    "little": "small",
    "normal": "medium",
    "tiny": "small",
}
STATE_ALIASES = {
    "opened": "open",
    "shut": "closed",
}
ADJECTIVE_ALIASES = COLOR_ALIASES | SIZE_ALIASES | STATE_ALIASES
ADJECTIVES = (
    COLOR_WORDS
    | SIZE_WORDS
    | STATE_WORDS
    | LOCATION_WORDS
    | CONTEXT_ADJECTIVES
    | set(ADJECTIVE_ALIASES)
)

DOMAIN_NOUNS = {
    "apple",
    "box",
    "cup",
    "floor",
    "fridge",
    "kitchen",
    "object",
    "robot",
    "room",
    "shelf",
    "table",
    "thing",
}
NOUN_ALIASES = {
    "automaton": "robot",
    "beaker": "cup",
    "bot": "robot",
    "carton": "box",
    "chamber": "room",
    "cooler": "fridge",
    "crate": "box",
    "desk": "table",
    "goblet": "cup",
    "ground": "floor",
    "icebox": "fridge",
    "mug": "cup",
    "rack": "shelf",
    "refrigerator": "fridge",
    "tumbler": "cup",
    "worktable": "table",
}
GENERIC_NOUNS = {
    "appliance",
    "container",
    "furniture",
    "item",
    "location",
    "place",
    "receptacle",
    "support",
    "surface",
    "vessel",
}
NOUNS = DOMAIN_NOUNS | set(NOUN_ALIASES) | GENERIC_NOUNS

# These are true names for concepts, but they are too broad to serve as a
# grounding rule. Binding "hat is a thing" would make every non-room object a
# possible hat, producing an ambiguity instead of a usable definition.
GENERIC_BINDING_KINDS = GENERIC_NOUNS | {
    "artifact",
    "being",
    "category",
    "entity",
    "kind",
    "living thing",
    "matter",
    "object",
    "phenomenon",
    "physical object",
    "stuff",
    "substance",
    "thing",
    "type",
    "whole",
}

PUT_VERBS = {"put", "place", "move", "set"}
TAKE_VERBS = {"take", "grab", "pick"}
GIVE_VERBS = {"give", "hand"}
OPEN_VERBS = {"open", "close"}
ACTION_ALIASES = {
    "bring": "put",
    "carry": "put",
    "deposit": "put",
    "drop": "put",
    "insert": "put",
    "lay": "put",
    "leave": "put",
    "lower": "put",
    "position": "put",
    "relocate": "put",
    "return": "put",
    "shift": "put",
    "situate": "put",
    "stow": "put",
    "collect": "take",
    "fetch": "take",
    "get": "take",
    "grab": "take",
    "hold": "take",
    "lift": "take",
    "pick": "take",
    "remove": "take",
    "retrieve": "take",
    "seize": "take",
    "withdraw": "take",
    "deliver": "give",
    "hand": "give",
    "offer": "give",
    "pass": "give",
    "send": "give",
    "lock": "close",
    "shut": "close",
    "unlock": "open",
}
ACTION_SYNONYMS = set(ACTION_ALIASES)
ACTION_VERBS = PUT_VERBS | TAKE_VERBS | GIVE_VERBS | OPEN_VERBS | ACTION_SYNONYMS

AUXILIARIES = {"is", "are", "was", "were"}
QUESTION_STARTS = {
    "what",
    "where",
    "which",
    "who",
    "how",
    "is",
    "are",
    "was",
    "were",
}
RELATION_ALIASES = {
    "alongside": "next_to",
    "atop": "on",
    "beside": "next_to",
    "inside": "in",
    "into": "in",
    "near": "next_to",
    "onto": "on",
    "towards": "to",
    "upon": "on",
    "within": "in",
}
PREPOSITION_STARTS = {
    "on",
    "onto",
    "in",
    "inside",
    "to",
    "at",
    "from",
    "next",
    "under",
    "above",
    "atop",
    "upon",
    "into",
    "within",
    "beside",
    "near",
    "alongside",
    "towards",
    "off",
}

# Discourse markers are not content words. They are explicit instructions for
# extending or repairing the previous turn in the conversation.
DISCOURSE_WORDS = {
    "about",
    "again",
    "also",
    "and",
    "context",
    "did",
    "do",
    "forget",
    "go",
    "happened",
    "here",
    "how",
    "just",
    "more",
    "memory",
    "now",
    "once",
    "or",
    "other",
    "rather",
    "remember",
    "repeat",
    "same",
    "then",
    "was",
    "went",
}

ALL_WORDS = (
    DETERMINERS
    | PRONOUNS
    | POLITE_WORDS
    | ADJECTIVES
    | NOUNS
    | ACTION_VERBS
    | QUESTION_STARTS
    | PREPOSITION_STARTS
    | DISCOURSE_WORDS
    | {
        "one",
        "of",
        "color",
        "can",
        "could",
        "would",
        "will",
        "you",
        "me",
        "actually",
        "no",
        "not",
        "there",
        "back",
    }
)

IRREGULAR_PLURALS = {
    "shelves": "shelf",
    "fridges": "fridge",
    "boxes": "box",
    "glasses": "glass",
}


def normalize_token(token: str) -> str:
    """Map a few common plurals onto the demo's singular vocabulary."""

    token = token.lower()
    if token in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[token]
    if token in ALL_WORDS:
        return token
    if token.endswith("es") and token[:-2] in NOUNS:
        return canonical_noun(token[:-2])
    if token.endswith("s") and token[:-1] in NOUNS:
        return canonical_noun(token[:-1])
    if token in NOUN_ALIASES:
        return NOUN_ALIASES[token]
    return token


def noun_matches(entity: Entity, noun: str) -> bool:
    if noun == "one":
        return True
    if noun in {"object", "thing", "item"}:
        return entity.kind != "room"
    noun = canonical_noun(noun)
    if noun == "room":
        return entity.kind == "room"
    if entity.kind == "room":
        return entity.id == noun or entity.id.replace("_", " ") == noun
    if noun in {"container", "receptacle", "vessel"}:
        return entity.kind in CONTAINER_KINDS or bool(entity.attrs.get("container"))
    if noun in {"support", "surface"}:
        return entity.kind in SUPPORT_KINDS or bool(entity.attrs.get("support"))
    if noun == "furniture":
        return entity.kind in {"table", "shelf"}
    if noun == "appliance":
        return entity.kind == "fridge"
    if noun in {"location", "place"}:
        return (
            entity.kind in SUPPORT_KINDS
            or entity.kind in CONTAINER_KINDS
            or entity.kind in ROOM_KINDS
        )
    return entity.kind == noun or noun in entity.aliases


def adjective_matches(entity: Entity, adjective: str) -> bool:
    adjective = canonical_adjective(adjective)
    if adjective == "living":
        return entity.id == "living_room"
    if adjective == "open":
        return bool(entity.attrs.get("open"))
    if adjective == "closed":
        return entity.attrs.get("open") is False
    if adjective == "full":
        return bool(entity.attrs.get("full"))
    if adjective == "empty":
        return entity.attrs.get("full") is False
    if adjective in COLOR_WORDS:
        return entity.attrs.get("color") == adjective
    if adjective in SIZE_WORDS:
        return entity.attrs.get("size") == adjective
    if adjective == entity.kind:
        return True
    return bool(entity.attrs.get(adjective))


def is_known_word(token: str) -> bool:
    normalized = normalize_token(token)
    return normalized in ALL_WORDS or english.is_known(normalized)


def canonical_noun(token: str) -> str:
    return NOUN_ALIASES.get(token, token)


def canonical_adjective(token: str) -> str:
    return ADJECTIVE_ALIASES.get(token, token)


def canonical_relation(token: str) -> str:
    return RELATION_ALIASES.get(token, token)


def canonical_action(token: str) -> str:
    direct = normalize_token(token)
    if direct in ACTION_ALIASES:
        return ACTION_ALIASES[direct]
    if direct in ACTION_VERBS:
        return direct
    # Inflected forms can have more than one possible base ("putting" can
    # derive from "put" or "putt"). Prefer the base that names an action this
    # world can actually perform instead of refusing the verb.
    for lemma in english.base_forms(direct, "verb"):
        action = ACTION_ALIASES.get(lemma, lemma)
        if action in ACTION_VERBS:
            return action
    lemma = english.lemma(direct, "verb")
    if lemma is None:
        return direct
    return ACTION_ALIASES.get(lemma, lemma)


def general_adjective(token: str) -> str | None:
    if token in ADJECTIVES:
        return canonical_adjective(token)
    lemma = english.lemma(token, "adjective")
    if lemma is None:
        return None
    return canonical_adjective(lemma)


def general_noun(token: str) -> str | None:
    if token in NOUNS:
        return canonical_noun(token)
    lemma = english.lemma(token, "noun")
    if lemma is None:
        return None
    return canonical_noun(lemma)


def is_general_verb(token: str) -> bool:
    return bool(english.base_forms(token, "verb"))


def is_adverb(token: str) -> bool:
    return bool(english.base_forms(token, "adverb"))
