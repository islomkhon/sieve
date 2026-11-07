"""Deterministic phrase and frame parser.

This is deliberately small. The larger Sieve idea treats syntax as constraints;
this prototype uses frame assembly as a fast special case for a bounded world.
No statistical model, embedding, or learned parameter is involved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from . import english
from . import lexicon as lx
from . import normalize as nm
from . import spelling


_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


class ParseError(Exception):
    """The utterance is well-formed English for this demo, but not supported."""


class UnknownWord(ParseError):
    def __init__(self, word: str, role: str = "token"):
        self.word = word
        self.role = role
        super().__init__(f"I don't know the word {word!r}.")


@dataclass(frozen=True)
class NP:
    det: str | None
    adjectives: tuple[str, ...]
    noun: str
    text: str
    start: int
    end: int
    modifiers: tuple["PP", ...] = ()


@dataclass(frozen=True)
class PP:
    mode: str
    target: NP


@dataclass(frozen=True)
class ActionDraft:
    verb: str
    theme: NP
    destination: PP | None = None
    source: PP | None = None
    recipient: NP | None = None


@dataclass(frozen=True)
class QueryDraft:
    kind: str
    subject: NP | None = None
    destination: PP | None = None
    attribute: str | None = None
    noun: str | None = None
    value: str | None = None
    adjectives: tuple[str, ...] = ()


def tokenize(text: str) -> list[str]:
    expanded = nm.expand_contractions(text)
    tokens = [lx.normalize_token(match.group(0)) for match in _WORD_RE.finditer(expanded)]
    tokens = [token for token in tokens if token not in lx.POLITE_WORDS]
    tokens = nm.rewrite_phrases(tokens)
    tokens = nm.strip_polite_prefixes(tokens)
    # Rewrite before and after polite-prefix removal. The first pass preserves
    # requests such as "show me the red cup"; the second pass sees the payload
    # exposed when a prefix such as "could you" is removed.
    tokens = nm.rewrite_phrases(tokens)
    tokens = nm.strip_polite_prefixes(tokens)
    tokens = nm.rewrite_phrases(tokens)
    return tokens


class Parser:
    def __init__(self) -> None:
        self.extra_nouns: set[str] = set()
        self.learned_aliases: dict[str, str] = {}
        self.corrections: list[spelling.Correction] = []

    def add_noun(self, word: str) -> None:
        self.extra_nouns.add(word)

    def _apply_aliases(self, tokens: list[str]) -> list[str]:
        """Apply explicitly learned word-to-word rules before parsing."""

        if not self.learned_aliases:
            return tokens
        return [self.learned_aliases.get(token, token) for token in tokens]

    def correct_token(self, tokens: list[str], i: int, categories: set[str]) -> str:
        return self._correct_token(tokens, i, categories)

    def _correct_token(self, tokens: list[str], i: int, categories: set[str]) -> str:
        if i >= len(tokens):
            return ""
        token = tokens[i]
        if (
            token in lx.ALL_WORDS
            or token in self.extra_nouns
            or lx.is_known_word(token)
        ):
            return token
        correction = spelling.correct_word(token, categories, self.extra_nouns)
        if correction is None or correction.corrected == token:
            return token
        self._apply_correction(tokens, i, correction)
        return correction.corrected

    def _apply_correction(
        self, tokens: list[str], i: int, correction: spelling.Correction
    ) -> None:
        if correction.corrected == tokens[i]:
            return
        tokens[i] = correction.corrected
        self.corrections.append(correction)

    def _correct_first_token(self, tokens: list[str], categories: set[str]) -> None:
        token = tokens[0]
        if token in lx.ALL_WORDS or token in self.extra_nouns:
            return

        # Do not rewrite a valid word merely because a nearby supported word
        # is in the expected category. "why" must not become "who", and a
        # definitional subject such as "hat" must not become "what". A known
        # noun/verb in an action slot can still be a misspelling ("pout" ->
        # "put"), so lexical knowledge alone is not enough; sentence cues and
        # already-supported grammar decide.
        if lx.is_known_word(token):
            question_words = {
                "how",
                "what",
                "when",
                "where",
                "which",
                "who",
                "whom",
                "whose",
                "why",
            }
            entry = english.lookup(token)
            noun_only_single = (
                len(tokens) == 1
                and entry is not None
                and "noun" in entry.categories
            )
            definition_subject = (
                len(tokens) > 1 and tokens[1] in {"are", "is", "means", "was", "were"}
            )
            if (
                token in question_words
                or token in lx.AUXILIARIES
                or token in {"cancel", "context", "explain", "forget", "help", "reset", "show", "state"}
                or lx.canonical_action(token) in lx.ACTION_VERBS
                or noun_only_single
                or definition_subject
            ):
                return
            if entry is not None and not ({"noun", "verb"} & set(entry.categories)):
                return

        candidates = spelling.correct_word_candidates(
            token, categories, self.extra_nouns
        )
        if not candidates:
            return
        if len(candidates) == 1:
            self._apply_correction(tokens, 0, candidates[0])
            return

        original_corrections = self.corrections
        plausible: list[spelling.Correction] = []
        for candidate in candidates:
            trial = list(tokens)
            trial[0] = candidate.corrected
            self.corrections = []
            try:
                self._dispatch(trial)
            except UnknownWord:
                plausible.append(candidate)
            except ParseError:
                continue
            else:
                plausible.append(candidate)
        self.corrections = original_corrections

        if len(plausible) == 1:
            self._apply_correction(tokens, 0, plausible[0])

    def _dispatch(self, tokens: list[str]) -> ActionDraft | QueryDraft:
        first = tokens[0]
        action = lx.canonical_action(first)
        if action in lx.ACTION_VERBS:
            return self._parse_action(tokens)
        if first in lx.QUESTION_STARTS:
            return self._parse_question(tokens)
        if first in {
            "show",
            "help",
            "why",
            "reset",
            "cancel",
            "never",
            "nevermind",
            "context",
            "memory",
            "remember",
            "forget",
        }:
            raise ParseError("That is a control word, not a sentence.")
        general_noun = lx.general_noun(first)
        if general_noun is not None:
            raise ParseError(
                f"I recognize {general_noun!r} as a noun, but this turn needs an "
                "instruction or a question."
            )
        if lx.is_general_verb(first):
            raise ParseError(
                f"I recognize {first!r} as a verb, but this world has no action for it."
            )
        if not lx.is_known_word(first) and first not in self.extra_nouns:
            raise UnknownWord(first, role="token")
        raise ParseError(
            f"I recognize {first!r}, but I don't know how it fits this turn."
        )

    def starts_new_utterance(self, text: str) -> bool:
        tokens = self._apply_aliases(tokenize(text))
        if not tokens:
            return False
        while tokens and tokens[0] in {"and", "then", "also"}:
            tokens = tokens[1:]
        if not tokens:
            return False
        self._correct_first_token(tokens, {"action", "question", "control"})
        first = tokens[0]
        if lx.canonical_action(first) in lx.ACTION_VERBS or first in lx.QUESTION_STARTS:
            return True
        return first in {
            "show",
            "help",
            "why",
            "reset",
            "cancel",
            "never",
            "nevermind",
            "context",
            "memory",
            "remember",
            "forget",
        }

    def parse(self, text: str) -> ActionDraft | QueryDraft:
        self.corrections = []
        tokens = self._apply_aliases(tokenize(text))
        if not tokens:
            raise ParseError("I didn't hear any words.")

        self._correct_first_token(tokens, {"action", "question"})
        return self._dispatch(tokens)

    def parse_reference(self, text: str) -> NP:
        self.corrections = []
        tokens = self._apply_aliases(tokenize(text))
        self._correct_token(tokens, 0, {"correction"})
        while tokens and tokens[0] in {"actually", "no", "not", "and", "then", "also"}:
            tokens = tokens[1:]
        if not tokens:
            raise ParseError("I need a thing to choose.")

        for index in range(len(tokens)):
            self._correct_token(tokens, index, {"adjective"})
        if all(token in lx.ADJECTIVES for token in tokens):
            return NP(None, tuple(tokens), "one", " ".join(tokens), 0, len(tokens))

        self._correct_token(tokens, 0, {"relation"})
        if tokens[0] in lx.PREPOSITION_STARTS:
            pp, i = self._parse_pp(tokens, 0, allowed=None)
            self._expect_end(tokens, i)
            return NP(None, (), "one", "one " + " ".join(tokens), 0, 0, (pp,))

        np, i = self._parse_np(tokens, 0, allow_bare=True, allow_modifier=True)
        self._expect_end(tokens, i)
        return np

    def _parse_action(self, tokens: list[str]) -> ActionDraft:
        verb = lx.canonical_action(self._correct_token(tokens, 0, {"action"}))

        if verb in lx.PUT_VERBS:
            theme, i = self._parse_np(tokens, self._skip_adverbs(tokens, 1))
            i = self._skip_adverbs(tokens, i)
            i = self._skip_action_particles(tokens, i, verb)
            destination, i = self._parse_pp(
                tokens, i, allowed={"on", "in", "next_to", "to", "at"}
            )
            self._expect_end(tokens, i)
            return ActionDraft(verb=verb, theme=theme, destination=destination)

        if verb in lx.TAKE_VERBS:
            theme, i = self._parse_np(tokens, self._skip_adverbs(tokens, 1))
            source = None
            i = self._skip_adverbs(tokens, i)
            i = self._skip_action_particles(tokens, i, verb)
            if i < len(tokens):
                source, i = self._parse_pp(tokens, i, allowed={"from", "on", "in", "at"})
            self._expect_end(tokens, i)
            return ActionDraft(verb=verb, theme=theme, source=source)

        if verb in lx.GIVE_VERBS:
            theme, i = self._parse_np(tokens, self._skip_adverbs(tokens, 1))
            i = self._skip_adverbs(tokens, i)
            i = self._skip_action_particles(tokens, i, verb)
            recipient_pp, i = self._parse_pp(tokens, i, allowed={"to"})
            self._expect_end(tokens, i)
            return ActionDraft(verb=verb, theme=theme, recipient=recipient_pp.target)

        if verb in lx.OPEN_VERBS:
            theme, i = self._parse_np(tokens, self._skip_adverbs(tokens, 1))
            self._expect_end(tokens, i)
            return ActionDraft(verb=verb, theme=theme)

        raise ParseError(f"I don't know how to use the verb {verb!r}.")

    def _parse_question(self, tokens: list[str]) -> QueryDraft:
        first = tokens[0]

        if first in lx.AUXILIARIES:
            existential = len(tokens) > 1 and tokens[1] == "there"
            i = 2 if existential else 1
            subject, i = self._parse_np(tokens, i)
            if existential:
                destination = None
                if i < len(tokens):
                    destination, i = self._parse_pp(
                        tokens, i, allowed={"on", "in", "next_to", "to", "at"}
                    )
                self._expect_end(tokens, i)
                return QueryDraft(
                    kind="exists",
                    noun=subject.noun,
                    adjectives=subject.adjectives,
                    destination=destination,
                )
            if i < len(tokens):
                self._correct_token(tokens, i, {"adjective"})
            if i < len(tokens) and tokens[i] in lx.ADJECTIVES:
                value = tokens[i]
                i += 1
                self._expect_end(tokens, i)
                return QueryDraft(kind="property", subject=subject, value=value)
            destination, i = self._parse_pp(
                tokens, i, allowed={"on", "in", "next_to", "to", "at"}
            )
            self._expect_end(tokens, i)
            return QueryDraft(kind="yes_no", subject=subject, destination=destination)

        if first == "where":
            i = 1
            if i < len(tokens):
                self._correct_token(tokens, i, {"auxiliary"})
            if i < len(tokens) and tokens[i] in lx.AUXILIARIES:
                i += 1
            subject, i = self._parse_np(tokens, i, allow_modifier=True)
            self._expect_end(tokens, i)
            return QueryDraft(kind="where", subject=subject)

        if first == "what":
            i = 1
            if i < len(tokens):
                self._correct_token(tokens, i, {"auxiliary"})
            if i < len(tokens) and tokens[i] in lx.AUXILIARIES:
                i += 1

            if i < len(tokens):
                self._correct_token(tokens, i, {"color", "determiner"})
            if i < len(tokens) and tokens[i] == "color":
                i += 1
                if i < len(tokens):
                    self._correct_token(tokens, i, {"auxiliary"})
                if i < len(tokens) and tokens[i] in lx.AUXILIARIES:
                    i += 1
                if i < len(tokens):
                    self._correct_token(tokens, i, {"of"})
                if i < len(tokens) and tokens[i] == "of":
                    i += 1
                subject, i = self._parse_np(tokens, i, allow_modifier=True)
                self._expect_end(tokens, i)
                return QueryDraft(kind="color", subject=subject)

            if i + 1 < len(tokens) and tokens[i] == "the":
                self._correct_token(tokens, i + 1, {"color"})
            if i + 1 < len(tokens) and tokens[i] == "the" and tokens[i + 1] == "color":
                i += 2
                if i < len(tokens):
                    self._correct_token(tokens, i, {"of"})
                if i < len(tokens) and tokens[i] == "of":
                    i += 1
                subject, i = self._parse_np(tokens, i, allow_modifier=True)
                self._expect_end(tokens, i)
                return QueryDraft(kind="color", subject=subject)

            if i < len(tokens) and tokens[i] == "size":
                i += 1
                if i < len(tokens):
                    self._correct_token(tokens, i, {"auxiliary"})
                if i < len(tokens) and tokens[i] in lx.AUXILIARIES:
                    i += 1
                if i < len(tokens):
                    self._correct_token(tokens, i, {"of"})
                if i < len(tokens) and tokens[i] == "of":
                    i += 1
                subject, i = self._parse_np(tokens, i, allow_modifier=True)
                self._expect_end(tokens, i)
                return QueryDraft(kind="size", subject=subject)

            if i + 1 < len(tokens) and tokens[i] == "the" and tokens[i + 1] == "size":
                i += 2
                if i < len(tokens):
                    self._correct_token(tokens, i, {"of"})
                if i < len(tokens) and tokens[i] == "of":
                    i += 1
                subject, i = self._parse_np(tokens, i, allow_modifier=True)
                self._expect_end(tokens, i)
                return QueryDraft(kind="size", subject=subject)

            if i < len(tokens) and tokens[i] in {"everywhere", "everyplace"}:
                i += 1
                self._expect_end(tokens, i)
                return QueryDraft(kind="inventory")

            # "What is the red cup?" asks for a description of a grounded
            # object. The dictionary path deliberately declines phrases that
            # begin with a determiner, so this frame gets the grounded case.
            if i < len(tokens) and (
                tokens[i] in lx.DETERMINERS or tokens[i] in lx.PRONOUNS
            ):
                subject, i = self._parse_np(tokens, i, allow_modifier=True)
                self._expect_end(tokens, i)
                return QueryDraft(kind="describe", subject=subject)

            # "What is X?" with a bare X is a description or definition
            # request, not a location question. The dictionary path handles
            # known bare words; this branch gives an unknown bare word the
            # normal noun-repair dialogue instead of a misleading location
            # error.
            if (
                i < len(tokens)
                and tokens[i] not in lx.PREPOSITION_STARTS
                and tokens[i] not in {"here", "there"}
            ):
                subject, i = self._parse_np(tokens, i, allow_modifier=True)
                self._expect_end(tokens, i)
                return QueryDraft(kind="describe", subject=subject)

            destination, i = self._parse_pp(
                tokens, i, allowed={"on", "in", "next_to", "at"}
            )
            self._expect_end(tokens, i)
            return QueryDraft(kind="list", destination=destination)

        if first == "which":
            noun_np, i = self._parse_np(tokens, 1)
            if i >= len(tokens):
                return QueryDraft(
                    kind="list_kind",
                    noun=noun_np.noun,
                    adjectives=noun_np.adjectives,
                )
            if i < len(tokens):
                self._correct_token(tokens, i, {"auxiliary"})
            if i < len(tokens) and tokens[i] in lx.AUXILIARIES:
                i += 1
            destination, i = self._parse_pp(
                tokens, i, allowed={"on", "in", "next_to", "at"}
            )
            self._expect_end(tokens, i)
            return QueryDraft(kind="which", noun=noun_np.noun, destination=destination)

        if first == "how":
            if tokens[1:2] != ["many"]:
                raise ParseError("I understand 'how many ...' questions here.")
            noun_np, i = self._parse_np(tokens, 2)
            if i < len(tokens):
                self._correct_token(tokens, i, {"auxiliary"})
            if i < len(tokens) and tokens[i] in lx.AUXILIARIES:
                i += 1
            if i < len(tokens) and tokens[i] == "there":
                i += 1
            destination = None
            if i < len(tokens):
                destination, i = self._parse_pp(
                    tokens, i, allowed={"on", "in", "next_to", "at"}
                )
            self._expect_end(tokens, i)
            return QueryDraft(
                kind="count",
                noun=noun_np.noun,
                destination=destination,
                adjectives=noun_np.adjectives,
            )

        if first == "who":
            raise ParseError("I don't answer 'who' questions in this demo.")

        raise ParseError(f"I don't know how to answer a question starting with {first!r}.")

    def _parse_np(
        self,
        tokens: list[str],
        i: int,
        *,
        allow_bare: bool = False,
        allow_modifier: bool = False,
    ) -> tuple[NP, int]:
        if i >= len(tokens):
            raise ParseError("I expected a thing, for example 'the red cup'.")

        start = i
        det: str | None = None

        if tokens[i] not in lx.PRONOUNS and tokens[i] not in lx.DETERMINERS:
            self._correct_token(tokens, i, {"determiner"})

        if tokens[i] in lx.PRONOUNS:
            noun = tokens[i]
            i += 1
            return NP(None, (), noun, " ".join(tokens[start:i]), start, i), i

        if tokens[i] in lx.DETERMINERS:
            det = tokens[i]
            i += 1
        if det in {"this", "that", "these", "those"} and (
            i >= len(tokens) or tokens[i] in lx.PREPOSITION_STARTS
        ):
            np = NP(None, (), det, " ".join(tokens[start:i]), start, i)
            if allow_modifier:
                modifiers: list[PP] = []
                while i < len(tokens) and tokens[i] in lx.PREPOSITION_STARTS:
                    pp, i = self._parse_pp(tokens, i, allowed=None)
                    modifiers.append(pp)
                if modifiers:
                    np = replace(np, modifiers=tuple(modifiers))
            return np, i

        adjectives: list[str] = []
        while i < len(tokens):
            if tokens[i] == "one":
                break
            if (
                lx.is_adverb(tokens[i])
                and i + 1 < len(tokens)
                and (
                    tokens[i + 1] in lx.ADJECTIVES
                    or lx.general_adjective(tokens[i + 1]) is not None
                )
            ):
                i += 1
                continue
            if tokens[i] not in lx.ADJECTIVES:
                if spelling.correct_domain_word(
                    tokens[i], {"noun"}, self.extra_nouns
                ) is not None:
                    break
                self._correct_token(tokens, i, {"adjective"})
            if tokens[i] in lx.ADJECTIVES:
                adjectives.append(lx.canonical_adjective(tokens[i]))
                i += 1
                continue
            general_adjective = lx.general_adjective(tokens[i])
            if general_adjective is not None:
                adjectives.append(general_adjective)
                i += 1
                continue
            if self._is_compound_modifier(tokens, i):
                adjectives.append(tokens[i])
                i += 1
                continue
            break

        if (
            i < len(tokens)
            and tokens[i] not in lx.NOUNS
            and tokens[i] not in self.extra_nouns
            and tokens[i] != "one"
        ):
            self._correct_token(tokens, i, {"noun"})

        if i < len(tokens) and (tokens[i] in lx.NOUNS or tokens[i] in self.extra_nouns):
            noun = lx.canonical_noun(tokens[i])
            i += 1
        elif i < len(tokens) and tokens[i] == "one":
            noun = "one"
            i += 1
        elif i < len(tokens) and lx.general_noun(tokens[i]) is not None:
            raise UnknownWord(lx.general_noun(tokens[i]) or tokens[i], role="noun")
        elif allow_bare and adjectives:
            noun = "one"
        elif i < len(tokens):
            word = tokens[i]
            if word not in lx.ALL_WORDS and word not in self.extra_nouns:
                raise UnknownWord(word, role="noun")
            previous = tokens[i - 1] if i else "the start of the sentence"
            raise ParseError(f"I expected a noun after {previous!r}.")
        else:
            raise ParseError("I expected a noun.")

        np = NP(det, tuple(adjectives), noun, " ".join(tokens[start:i]), start, i)

        if allow_modifier:
            modifiers: list[PP] = []
            while i < len(tokens):
                if tokens[i] not in lx.PREPOSITION_STARTS:
                    self._correct_token(tokens, i, {"relation"})
                if tokens[i] not in lx.PREPOSITION_STARTS:
                    break
                pp, i = self._parse_pp(tokens, i, allowed=None)
                modifiers.append(pp)
            if modifiers:
                np = replace(np, modifiers=tuple(modifiers))

        return np, i

    def _parse_pp(
        self,
        tokens: list[str],
        i: int,
        *,
        allowed: set[str] | None = None,
    ) -> tuple[PP, int]:
        if i >= len(tokens):
            raise ParseError("I expected a location, for example 'on the table'.")

        if tokens[i] not in lx.PREPOSITION_STARTS:
            self._correct_token(tokens, i, {"relation"})

        if tokens[i] in {"there", "back", "here"}:
            word = tokens[i]
            target = NP(None, (), word, word, i, i + 1)
            return PP(word, target), i + 1

        if tokens[i] == "next":
            if i + 1 < len(tokens):
                self._correct_token(tokens, i + 1, {"to"})
            if i + 1 >= len(tokens) or tokens[i + 1] != "to":
                raise ParseError("I expected 'next to'.")
            mode = "next_to"
            i += 2
        elif tokens[i] == "on" and i + 2 < len(tokens) and tokens[i + 1] == "top" and tokens[i + 2] == "of":
            mode = "on"
            i += 3
        else:
            mapping = {
                "on": "on",
                "onto": "on",
                "in": "in",
                "inside": "in",
                "into": "in",
                "within": "in",
                "to": "to",
                "towards": "to",
                "at": "at",
                "from": "from",
                "off": "from",
                "under": "under",
                "above": "above",
                "atop": "on",
                "upon": "on",
                "beside": "next_to",
                "near": "next_to",
                "alongside": "next_to",
                "next_to": "next_to",
            }
            relation = lx.canonical_relation(tokens[i])
            if relation not in mapping:
                if tokens[i] not in lx.ALL_WORDS:
                    raise UnknownWord(tokens[i], role="relation")
                raise ParseError(f"I didn't understand the location word {tokens[i]!r}.")
            mode = mapping[relation]
            i += 1

        if allowed is not None and mode not in allowed:
            raise ParseError(f"The relation {mode!r} doesn't fit here.")

        target, i = self._parse_np(tokens, i)
        return PP(mode, target), i

    def _expect_end(self, tokens: list[str], i: int) -> None:
        if i >= len(tokens):
            return
        word = tokens[i]
        if lx.is_known_word(word):
            raise ParseError(f"I recognize {word!r}, but I don't know how it fits here.")
        if word not in lx.ALL_WORDS:
            raise UnknownWord(word, role="token")
        raise ParseError(f"I didn't understand {word!r} in that position.")

    def _skip_adverbs(self, tokens: list[str], i: int) -> int:
        while i < len(tokens):
            if not lx.is_adverb(tokens[i]):
                break
            if tokens[i] in lx.ALL_WORDS or tokens[i] in lx.PREPOSITION_STARTS:
                break
            if lx.general_noun(tokens[i]) is not None:
                break
            i += 1
        return i

    def _skip_action_particles(self, tokens: list[str], i: int, verb: str) -> int:
        if verb in lx.PUT_VERBS:
            particles = {"down"}
        elif verb in lx.TAKE_VERBS:
            particles = {"down", "up"}
        elif verb in lx.GIVE_VERBS:
            particles = {"back"}
        else:
            return i
        while i < len(tokens) and tokens[i] in particles:
            i += 1
        return i

    def _is_compound_modifier(self, tokens: list[str], i: int) -> bool:
        if i + 1 >= len(tokens):
            return False
        if lx.general_noun(tokens[i]) is None:
            return False
        following = tokens[i + 1]
        return (
            following in lx.NOUNS
            or following in self.extra_nouns
            or following == "one"
        )
