"""Explicit open-vocabulary knowledge for SIEVE.

The bundled data is a compact index of WordNet 3.0: an authored lexical
database, not a trained model. It supplies spelling, parts of speech, and short
definitions. WordNet knowledge is deliberately kept separate from the small
grounded world; a known English word still needs a world binding before it can
be moved or inspected.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DATA_PATH = Path(__file__).with_name("data") / "english_lexicon.json.gz"
SUPPLEMENT_PATH = Path(__file__).with_name("data") / "common_english.json"
CATEGORIES = (
    "noun",
    "verb",
    "adjective",
    "adverb",
    "conjunction",
    "determiner",
    "pronoun",
    "preposition",
    "interjection",
    "numeral",
)


@dataclass(frozen=True)
class Lexeme:
    word: str
    categories: tuple[str, ...]
    definition: str | None = None

    @property
    def pos_text(self) -> str:
        labels = {
            "noun": "noun",
            "verb": "verb",
            "adjective": "adjective",
            "adverb": "adverb",
            "conjunction": "conjunction",
            "determiner": "determiner",
            "pronoun": "pronoun",
            "preposition": "preposition",
            "interjection": "interjection",
            "numeral": "numeral",
        }
        return ", ".join(labels.get(category, category) for category in self.categories)


class EnglishLexicon:
    def __init__(self, path: Path = DATA_PATH) -> None:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            payload = json.load(source)
        self.source: str = payload.get("source", "WordNet")
        self._entries: dict[str, dict[str, object]] = payload["entries"]
        self._irregular: dict[str, str] = payload["irregular"]
        self._position_words: dict[str, tuple[str, ...]] = {}
        self._merge_supplement()

    def _merge_supplement(self) -> None:
        """Merge the authored closed-class and modern-English supplement."""

        if not SUPPLEMENT_PATH.exists():
            return
        with SUPPLEMENT_PATH.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        for word, description in payload.get("entries", {}).items():
            target = self._entries.setdefault(word, {"p": [], "d": ""})
            categories = target["p"]
            assert isinstance(categories, list)
            for category in description.get("p", []):
                if category not in categories:
                    categories.append(str(category))
            if not target.get("d") and description.get("d"):
                target["d"] = str(description["d"])

    @staticmethod
    def clean(word: str) -> str:
        return word.strip().lower().replace("’", "'").replace("_", " ")

    def exact(self, word: str) -> Lexeme | None:
        key = self.clean(word)
        description = self._entries.get(key)
        if description is None:
            return None
        categories = tuple(str(item) for item in description["p"])
        definition = str(description.get("d") or "")
        return Lexeme(key, categories, definition or None)

    def base_forms(self, word: str, category: str | None = None) -> tuple[str, ...]:
        """Return possible lemmas, preferring an exact category-compatible form."""

        key = self.clean(word)
        forms: list[str] = []

        irregular = self._irregular.get(key)
        if irregular is not None:
            entry = self.exact(irregular)
            if category is None or (entry is not None and category in entry.categories):
                forms.append(irregular)

        exact = self.exact(key)
        if exact is not None and (
            category is None or category in exact.categories
        ):
            forms.append(key)

        if category != "adverb":
            for candidate in self._suffix_bases(key):
                entry = self.exact(candidate)
                if entry is None:
                    continue
                if category is None or category in entry.categories:
                    forms.append(candidate)

        unique: list[str] = []
        for form in forms:
            if form not in unique:
                unique.append(form)
        return tuple(unique)

    @staticmethod
    def _suffix_bases(word: str) -> tuple[str, ...]:
        candidates: list[str] = []

        def add(value: str) -> None:
            if value and len(value) >= 2 and value not in candidates:
                candidates.append(value)

        if word.endswith("ies") and len(word) > 4:
            add(word[:-3] + "y")
            add(word[:-1])
        if word.endswith("ves") and len(word) > 4:
            add(word[:-3] + "f")
            add(word[:-3] + "fe")
        if word.endswith("es") and len(word) > 3:
            add(word[:-2])
            add(word[:-1])
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            add(word[:-1])

        if word.endswith("ied") and len(word) > 4:
            add(word[:-3] + "y")
        if word.endswith("ed") and len(word) > 3:
            add(word[:-2])
            add(word[:-1])
            if word[-3] == word[-4]:
                add(word[:-3])
        if word.endswith("ing") and len(word) > 4:
            stem = word[:-3]
            add(stem)
            add(stem + "e")
            if len(stem) > 2 and stem[-1] == stem[-2]:
                add(stem[:-1])

        if word.endswith("est") and len(word) > 4:
            add(word[:-3])
            add(word[:-2])
            if word[-4] == word[-5] if len(word) > 4 else False:
                add(word[:-4])
        if word.endswith("er") and len(word) > 3:
            add(word[:-2])
            add(word[:-1])
        if word.endswith("ly") and len(word) > 4:
            stem = word[:-2]
            add(stem)
            if stem.endswith("i"):
                add(stem[:-1] + "y")

        return tuple(candidates)

    def lemma(self, word: str, category: str | None = None) -> str | None:
        forms = self.base_forms(word, category)
        if len(forms) == 1:
            return forms[0]
        return None

    def categories(self, word: str) -> tuple[str, ...]:
        categories: list[str] = []
        for form in self.base_forms(word):
            entry = self.exact(form)
            if entry is None:
                continue
            for category in entry.categories:
                if category not in categories:
                    categories.append(category)
        return tuple(categories)

    def words_for_positions(self, positions: tuple[str, ...]) -> tuple[str, ...]:
        key = ",".join(sorted(positions))
        existing = self._position_words.get(key)
        if existing is not None:
            return existing
        selected: list[str] = []
        for word, description in self._entries.items():
            if " " in word:
                continue
            categories = {str(item) for item in description["p"]}
            if categories.intersection(positions):
                selected.append(word)
        self._position_words[key] = tuple(sorted(selected))
        return self._position_words[key]

    def has_word(self, word: str) -> bool:
        return bool(self.base_forms(word))

    def stats(self) -> dict[str, int]:
        """Count the bundled lexical universe by useful category."""

        counts = {
            "entries": len(self._entries),
            "definitions": sum(
                1
                for description in self._entries.values()
                if str(description.get("d") or "").strip()
            ),
        }
        for category in ("noun", "verb", "adjective", "adverb"):
            counts[category + "s"] = len(self.words_for_positions((category,)))
        return counts


@lru_cache(maxsize=1)
def lexicon() -> EnglishLexicon:
    return EnglishLexicon()


def lookup(word: str) -> Lexeme | None:
    entry = lexicon().exact(word)
    if entry is not None:
        return entry
    # Some inflections have several possible bases ("analyses" can suggest
    # "analysis" and "analyse"). Prefer the first full lexical entry rather
    # than losing the word merely because the base is not unique.
    for form in lexicon().base_forms(word):
        entry = lexicon().exact(form)
        if entry is not None:
            return entry
    return None


def lemma(word: str, category: str | None = None) -> str | None:
    return lexicon().lemma(word, category)


def base_forms(word: str, category: str | None = None) -> tuple[str, ...]:
    return lexicon().base_forms(word, category)


def words_for_positions(positions: tuple[str, ...]) -> tuple[str, ...]:
    return lexicon().words_for_positions(positions)


def is_known(word: str, category: str | None = None) -> bool:
    return bool(lexicon().base_forms(word, category))


def definition_text(word: str) -> str | None:
    entry = lookup(word)
    if entry is None:
        return None
    body = entry.definition or "No short definition is recorded."
    return f"{entry.word} ({entry.pos_text}): {body}"
