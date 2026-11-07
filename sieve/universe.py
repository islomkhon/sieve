"""The open vocabulary concept catalog.

WordNet supplies thousands of noun concepts. They are not all physical objects
in the demo room: a concept becomes a grounded entity only when the user adds
one with "add ...". This keeps the grounded world honest while making the
recognized universe as broad as the bundled lexicon.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import english
from .world import Entity, World


@dataclass(frozen=True)
class Concept:
    word: str
    categories: tuple[str, ...]
    definition: str | None


class ConceptUniverse:
    """Lazy access to the bundled English concept inventory."""

    def lookup(self, word: str) -> Concept | None:
        entry = english.lookup(word)
        if entry is None:
            return None
        return Concept(entry.word, entry.categories, entry.definition)

    def is_noun(self, word: str) -> bool:
        concept = self.lookup(word)
        return concept is not None and "noun" in concept.categories

    def canonical_noun(self, word: str) -> str:
        entry = english.lookup(word)
        if entry is None:
            return word
        if "noun" in entry.categories:
            return entry.word
        lemma = english.lemma(word, "noun")
        return lemma or entry.word

    def stats(self) -> dict[str, int]:
        return english.lexicon().stats()

    def materialize(
        self,
        world: World,
        noun: str,
        *,
        adjectives: tuple[str, ...] = (),
        location: str | None = None,
        behavior: dict[str, object] | None = None,
    ) -> Entity:
        from . import lexicon as lx

        kind = self.canonical_noun(noun) or noun
        base = re.sub(r"[^a-z0-9]+", "_", kind.lower()).strip("_") or "object"
        entity_id = base
        suffix = 2
        while entity_id in world.entities:
            entity_id = f"{base}_{suffix}"
            suffix += 1

        attrs: dict[str, object] = dict(behavior or {})
        for adjective in adjectives:
            canonical = lx.canonical_adjective(adjective)
            if canonical in lx.COLOR_WORDS:
                attrs["color"] = canonical
            elif canonical in lx.SIZE_WORDS:
                attrs["size"] = canonical
            elif canonical in lx.STATE_WORDS:
                attrs[canonical] = True
            else:
                attrs[canonical] = True

        entity = Entity(
            id=entity_id,
            kind=kind,
            attrs=attrs,
            location=location,
            movable=bool(attrs.get("movable", True)),
        )
        world.entities[entity.id] = entity
        return entity
