"""The Sieve interpreter.

The central loop is:

    utterance -> candidate readings -> constraints -> survivors
    survivors == 1  -> act or answer
    survivors > 1   -> ask the question that best splits the survivors
    survivors == 0  -> report the violated constraint

There is no probability, embedding, training, or hidden model state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from itertools import product
from typing import TypeAlias

from . import english
from . import lexicon as lx
from . import spelling
from .memory import LearningMemory
from .parser import (
    ActionDraft,
    NP,
    PP,
    ParseError,
    Parser,
    QueryDraft,
    UnknownWord,
    tokenize,
)
from .universe import ConceptUniverse
from .world import Entity, World, initial_world


class ResolutionError(Exception):
    """A well-formed utterance could not be grounded in the world."""


@dataclass(frozen=True)
class Destination:
    mode: str
    target: Entity


@dataclass(frozen=True)
class Action:
    verb: str
    theme: Entity
    destination: Destination | None = None
    source: Destination | None = None
    recipient: Entity | None = None


@dataclass(frozen=True)
class Query:
    kind: str
    subject: Entity | None = None
    destination: Destination | None = None
    attribute: str | None = None
    noun: str | None = None
    value: str | None = None
    adjectives: tuple[str, ...] = ()
    exclude_ids: tuple[str, ...] = ()


Candidate: TypeAlias = Action | Query


_REPEAT_PATTERNS = {
    ("again",),
    ("do", "it", "again"),
    ("do", "that", "again"),
    ("do", "the", "same"),
    ("do", "the", "same", "thing"),
    ("once", "more"),
    ("repeat",),
    ("repeat", "that"),
    ("same",),
    ("same", "thing"),
}

_CONTROL_WORDS = {
    "cancel",
    "context",
    "forget",
    "help",
    "memory",
    "never",
    "nevermind",
    "remember",
    "reset",
    "show",
    "why",
}


@dataclass
class Pending:
    candidates: list[Candidate]
    slot: int
    attr: str
    original: str
    question: str


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str
    result_kind: str
    is_question: bool
    is_meta: bool = False


@dataclass
class Context:
    last_entities: list[Entity] = field(default_factory=list)
    last_action: Action | None = None
    last_query: Query | None = None
    last_destination: Destination | None = None
    last_locations: dict[str, str | None] = field(default_factory=dict)
    last_mode: str | None = None
    last_results: list[Entity] = field(default_factory=list)
    other_entities: list[Entity] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    last_rejection: str | None = None


@dataclass
class Result:
    text: str
    kind: str = "answer"
    meta: bool = False

    def __str__(self) -> str:
        return self.text


class Interpreter:
    def __init__(
        self,
        world: World | None = None,
        memory: LearningMemory | None = None,
    ):
        self.world = world or initial_world()
        self.parser = Parser()
        self.context = Context()
        self.pending: Pending | None = None
        self.definition_word: str | None = None
        self.definition_original: str | None = None
        self.noun_aliases: dict[str, str] = {}
        self.learned_definitions: dict[str, str] = {}
        self.learned_entity_aliases: dict[str, str] = {}
        self.learned_behaviors: dict[str, dict[str, object]] = {}
        self.created_entity_ids: set[str] = set()
        self.universe = ConceptUniverse()
        self.memory = memory
        if self.memory is not None:
            self._restore_memory()

    def reset(self) -> None:
        self.world = initial_world()
        self.context = Context()
        self.pending = None
        self.definition_word = None
        self.definition_original = None
        self.created_entity_ids.clear()
        self.parser.extra_nouns.clear()
        self.parser.extra_nouns.update(self.noun_aliases)
        self.parser.extra_nouns.update(self.learned_entity_aliases)
        self.parser.extra_nouns.update(self.learned_definitions)
        for word, entity_id in list(self.learned_entity_aliases.items()):
            entity = self.world.get(entity_id)
            if entity is None:
                self.learned_entity_aliases.pop(word, None)
                continue
            entity.aliases.add(word)
            self.parser.add_noun(word)
        if self.memory is not None:
            self.memory.world_state.clear()
            self.memory.created_entities.clear()
        self._save_memory()

    def _restore_memory(self, *, restore_turns: bool = True) -> None:
        """Apply learned rules, entities, world state, and history."""

        memory = self.memory
        if memory is None:
            return

        self.noun_aliases.update(memory.noun_aliases)
        self.parser.extra_nouns.update(memory.noun_aliases)
        self.parser.learned_aliases.update(memory.token_aliases)
        self.learned_definitions.update(memory.definitions)
        self.parser.extra_nouns.update(memory.definitions)
        self.learned_behaviors.update(
            {word: dict(attrs) for word, attrs in memory.behaviors.items()}
        )

        for entity_id, record in memory.created_entities.items():
            if self.world.get(entity_id) is not None:
                continue
            kind = str(record.get("kind") or "object")
            attrs = record.get("attrs")
            entity = Entity(
                id=entity_id,
                kind=kind,
                attrs=dict(attrs) if isinstance(attrs, dict) else {},
                location=str(record["location"]) if record.get("location") else None,
                movable=bool(record.get("movable", True)),
            )
            self.world.entities[entity.id] = entity
            self.created_entity_ids.add(entity.id)
            self.parser.add_noun(entity.kind)

        for entity_id, state in memory.world_state.items():
            entity = self.world.get(entity_id)
            if entity is None:
                continue
            if "location" in state:
                location = state.get("location")
                entity.location = str(location) if location else None
            attrs = state.get("attrs")
            if isinstance(attrs, dict):
                entity.attrs = dict(attrs)

        for word, entity_id in memory.entity_aliases.items():
            entity = self.world.get(entity_id)
            if entity is None:
                continue
            entity.aliases.add(word)
            self.learned_entity_aliases[word] = entity_id
            self.parser.add_noun(word)

        if restore_turns:
            self.context.turns = [
                Turn(
                    user=str(item.get("user", "")),
                    assistant=str(item.get("assistant", "")),
                    result_kind=str(item.get("result_kind", "answer")),
                    is_question=bool(item.get("is_question", False)),
                    is_meta=bool(item.get("is_meta", False)),
                )
                for item in memory.turns
            ]

    def _save_memory(self) -> None:
        """Persist explicit rules, current world state, and recent history."""

        memory = self.memory
        if memory is None:
            return

        memory.noun_aliases = dict(self.noun_aliases)
        memory.token_aliases = dict(self.parser.learned_aliases)
        memory.entity_aliases = dict(self.learned_entity_aliases)
        memory.definitions = dict(self.learned_definitions)
        memory.behaviors = {
            word: dict(attrs) for word, attrs in self.learned_behaviors.items()
        }
        memory.created_entities = {
            entity_id: {
                "kind": entity.kind,
                "attrs": dict(entity.attrs),
                "location": entity.location,
                "movable": entity.movable,
            }
            for entity_id in self.created_entity_ids
            if (entity := self.world.get(entity_id)) is not None
        }
        memory.world_state = {
            entity.id: {
                "location": entity.location,
                "attrs": dict(entity.attrs),
            }
            for entity in self.world.all()
        }
        memory.turns = [
            {
                "user": turn.user,
                "assistant": turn.assistant,
                "result_kind": turn.result_kind,
                "is_question": turn.is_question,
                "is_meta": turn.is_meta,
            }
            for turn in self.context.turns[-200:]
        ]
        memory.save()

    def _clean_transcript_line(self, text: str) -> str | None:
        """Remove shell/CLI wrappers from a pasted transcript line.

        Users often paste output that includes a Windows prompt, a ``you>``
        prefix, or a startup banner. Those are transport noise, not user
        utterances. A pasted assistant line is ignored; a pasted user line is
        unwrapped so its content can still be interpreted normally.
        """

        raw = text.strip()
        if re.match(r"^(?:sieve|assistant)>\s*", raw, re.IGNORECASE):
            return None
        raw = re.sub(
            r"^(?:you|user)>\s*", "", raw, flags=re.IGNORECASE
        ).strip()
        raw = re.sub(
            r"^(?:[A-Za-z]:[\\/][^>\r\n]*>|PS [^>\r\n]*>)\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        ).strip()
        if re.match(
            r"^python(?:\.exe)?\s+-m\s+sieve(?:\.cli)?(?:\s+--[\w-]+)*\s*$",
            raw,
            re.IGNORECASE,
        ):
            return None
        compact = " ".join(raw.split())
        if (
            compact.lower().startswith("sieve demo")
            or compact == "No model, no training, no embeddings."
            or compact.lower().startswith("learning memory:")
        ):
            return None
        return raw

    def handle(self, text: str, *, allow_compound: bool = True) -> Result:
        """Interpret one user turn and record it in conversation history.

        Recording happens only here, not in internal retries. That keeps one
        history entry per user message even when a definition causes the
        original request to be replayed.
        """

        raw = text.strip()
        cleaned = self._clean_transcript_line(raw)
        if cleaned is None:
            return Result(
                "Ignored a pasted terminal prompt or assistant transcript line.",
                "info",
                meta=True,
            )
        raw = cleaned
        if not raw:
            return Result("Say something.", "info")
        result = self._handle(raw, allow_compound=allow_compound)
        self.context.turns.append(
            Turn(
                user=raw,
                assistant=result.text,
                result_kind=result.kind,
                is_question=self._looks_like_question(raw),
                is_meta=result.meta,
            )
        )
        self._save_memory()
        return result

    def _handle(self, text: str, *, allow_compound: bool = True) -> Result:
        raw = text.strip()
        if not raw:
            return Result("Say something.", "info")

        asks_for_more = bool(
            re.match(r"^\s*(?:and\s+)?(?:what|anything)\s+else\b", raw, re.IGNORECASE)
        )

        # Conversational and meta intent must see the whole sentence. Otherwise
        # "what is X and how do I bind it?" can be split into two clauses before
        # the binding-help rule has a chance to run.
        conversational = self._conversational_response(raw)
        if conversational is not None:
            return conversational

        if allow_compound:
            clauses = self._split_compound(raw)
            if clauses:
                results: list[Result] = []
                for clause in clauses:
                    result = self._handle(clause, allow_compound=False)
                    results.append(result)
                    if result.kind != "answer":
                        return result
                return Result(" ".join(result.text for result in results))

        low = raw.lower()
        if low in {"help", "?"}:
            return Result(self.help_text(), "info", meta=True)
        if low in {"show", "state"}:
            return Result(self.world.describe_state(), "info", meta=True)
        if low in {"why", "explain"}:
            return Result(self.explain(), "info")
        if low == "reset":
            self.reset()
            return Result(
                "Reset the demo world. Learned rules are kept; use "
                "'forget learning' to clear them.",
                "info",
            )
        if low in {"context", "memory", "remember"}:
            return Result(self._context_summary(), "info", meta=True)
        if low in {
            "history",
            "chat history",
            "show history",
            "conversation history",
            "show the conversation",
            "summarize the conversation",
            "what did we talk about",
            "what have we talked about",
        }:
            return Result(self._history_text(), "info", meta=True)
        if low in {"knowledge", "learned", "show knowledge", "what have you learned"}:
            return Result(self._knowledge_summary(), "info", meta=True)
        if low in {"universe", "catalog", "world size", "universe size"}:
            return Result(self._universe_summary(), "info", meta=True)
        if low in {"forget learning", "clear learning", "forget everything", "clear memory"}:
            self._forget_learning()
            return Result(
                "Forgot all learned rules, created entities, world changes, and history. "
                "The demo world has been reset.",
                "info",
                meta=True,
            )
        if low in {"forget", "clear context"}:
            self.context = Context()
            self.pending = None
            self.definition_word = None
            self.definition_original = None
            return Result(
                "I forgot the conversation context. The world itself is unchanged.",
                "info",
                meta=True,
            )
        if low in {"cancel", "never mind", "nevermind"}:
            if self.definition_word is not None:
                word = self.definition_word
                self.definition_word = None
                self.definition_original = None
                return Result(
                    f"Okay, I cancelled the definition of {word!r}.",
                    "info",
                    meta=True,
                )
            if self.pending is not None:
                self.pending = None
                return Result(
                    "Okay, I cancelled the pending choice.", "info", meta=True
                )
            return Result(
                "There is no pending choice to cancel.", "info", meta=True
            )

        learning = self._learning_response(raw)
        if learning is not None:
            return learning

        implicit = self._implicit_definition_response(raw)
        if implicit is not None:
            return implicit

        world_build = self._world_build_response(raw)
        if world_build is not None:
            return world_build

        dictionary = self._dictionary_response(raw)
        if dictionary is not None:
            return dictionary

        if self.definition_word is not None:
            definition_result = self._try_definition(raw)
            if definition_result is not None:
                return definition_result
            if self.parser.starts_new_utterance(raw):
                self.definition_word = None
                self.definition_original = None
            elif self._starts_independent_reference(raw):
                self.definition_word = None
                self.definition_original = None
            else:
                return Result(self._definition_prompt(), "clarify")

        if self.pending is not None:
            pending_tokens = tuple(tokenize(raw))
            if self._is_repeat(raw) or pending_tokens in {("which", "one"), ("what",)}:
                return Result(self.pending.question, "clarify")
            exclude = self._is_negative_answer(raw)
            answer_text = self._strip_answer_prefixes(raw)
            if not answer_text:
                return Result("I need a thing to choose.", "error")
            if answer_text != raw:
                if self.parser.starts_new_utterance(answer_text):
                    self.pending = None
                    raw = answer_text
                else:
                    return self._handle_pending(answer_text, exclude=exclude)
            elif self.parser.starts_new_utterance(raw):
                self.pending = None
            else:
                return self._handle_pending(raw, exclude=exclude)

        if self.pending is None and self.definition_word is None:
            bare_reference = self._bare_reference_response(raw)
            if bare_reference is not None:
                return bare_reference

        contextual = self._normalize_contextual_phrases(raw)
        if isinstance(contextual, Result):
            return contextual
        raw = contextual

        try:
            draft = self.parser.parse(raw)
        except UnknownWord as exc:
            if exc.role == "noun":
                if exc.word in {
                    "am",
                    "are",
                    "be",
                    "been",
                    "being",
                    "can",
                    "could",
                    "did",
                    "do",
                    "does",
                    "had",
                    "has",
                    "have",
                    "is",
                    "may",
                    "might",
                    "must",
                    "shall",
                    "should",
                    "was",
                    "were",
                    "will",
                    "would",
                }:
                    return Result(
                        f"I couldn't interpret {exc.word!r} as a noun here. "
                        "Try a command such as 'put the red cup on the shelf' or "
                        "a question such as 'what is on the table?'.",
                        "error",
                    )
                entry = english.lookup(exc.word)
                if entry is not None and "noun" not in entry.categories:
                    return Result(
                        f"I recognize {exc.word!r} as English {entry.pos_text}, "
                        "but it is not usable as a noun here. Try rephrasing the "
                        "question or command.",
                        "error",
                    )
                self.definition_word = exc.word
                self.definition_original = raw
                return Result(self._definition_prompt(), "clarify")
            suggestions = [
                correction.corrected
                for correction in spelling.open_word_options(exc.word, 5)
                if correction.corrected != exc.word
            ]
            hint = ""
            if suggestions:
                hint = " Closest known English words: " + self._join_or(suggestions) + "."
            return Result(
                f"I don't know the word {exc.word!r} yet. Every word needs an "
                "explicit rule. Teach me with "
                f"'learn that {exc.word} means a box'."
                + hint,
                "error",
            )
        except ParseError as exc:
            return Result(str(exc), "error")

        try:
            if (
                isinstance(draft, QueryDraft)
                and draft.subject is not None
                and draft.subject.noun == "they"
            ):
                return self._answer_plural_query(draft)
            candidates = self._instantiate(draft)
        except ResolutionError as exc:
            return Result(str(exc), "error")
        if asks_for_more and self.context.last_results:
            excluded = tuple(entity.id for entity in self.context.last_results)
            candidates = [
                replace(
                    candidate,
                    exclude_ids=tuple(
                        dict.fromkeys((*candidate.exclude_ids, *excluded))
                    ),
                )
                if isinstance(candidate, Query)
                else candidate
                for candidate in candidates
            ]
        return self._decide(candidates, raw)

    def _split_compound(self, raw: str) -> list[str]:
        parts = [
            part.strip()
            for part in re.split(
                r"\s*;\s*|\s*,?\s*(?:and\s+)?then\s+", raw, flags=re.IGNORECASE
            )
            if part.strip()
        ]
        if len(parts) == 1:
            parts = [
                part.strip()
                for part in re.split(r"\s+and\s+", raw, flags=re.IGNORECASE)
                if part.strip()
            ]
        if len(parts) < 2:
            return []
        if all(self.parser.starts_new_utterance(part) for part in parts):
            return parts
        if len(parts) == 2 and self.parser.starts_new_utterance(parts[0]):
            tokens = tokenize(parts[0])
            if tokens:
                verb = lx.canonical_action(tokens[0])
                if verb in lx.ACTION_VERBS:
                    inherited = f"{verb} {parts[1]}"
                    try:
                        self.parser.parse(inherited)
                    except ParseError:
                        return []
                    return [parts[0], inherited]
        return []

    def _normalize_contextual_phrases(self, raw: str) -> str | Result:
        """Turn explicit discourse markers into complete utterances.

        This is not statistical coreference. Each marker is a rule that either
        copies a typed slot from the previous turn, restores a saved location,
        or repeats the previous action/query.
        """

        raw = self._strip_temporal_fillers(raw)
        raw = self._strip_trailing_fillers(raw)
        if self._is_repeat(raw):
            return self._repeat_last()

        meta = self._contextual_meta_response(raw)
        if isinstance(meta, Result):
            return meta
        if meta is not None:
            raw = meta

        stripped, rest = self._strip_context_prefix(raw)
        if not stripped:
            if self._is_bare_context_reference(raw):
                return self._extend_last_turn(raw)
            return raw
        if not rest:
            return Result("Tell me what to extend from the previous turn.", "error")
        if self._is_repeat(rest):
            return self._repeat_last()

        if self.parser.starts_new_utterance(rest):
            tokens = self._correct_context_tokens(tokenize(rest))
            if tokens and tokens[0] in _CONTROL_WORDS:
                return self._handle(rest)
            return rest

        return self._extend_last_turn(rest)

    def _correct_context_tokens(self, tokens: list[str]) -> list[str]:
        corrected = list(tokens)
        for index, token in enumerate(corrected):
            if token in lx.ALL_WORDS:
                continue
            categories = {"context"}
            if index == 0:
                categories.add("question")
            correction = spelling.correct_word(token, categories)
            if correction is not None:
                corrected[index] = correction.corrected
        return corrected

    def _strip_trailing_fillers(self, raw: str) -> str:
        tokens = tokenize(raw)
        while tokens and tokens[-1] == "now":
            tokens.pop()
        if not tokens:
            return raw
        return " ".join(tokens)

    def _strip_temporal_fillers(self, raw: str) -> str:
        """Remove discourse-time words after dictionary dispatch has run."""

        tokens = tokenize(raw)
        cleaned: list[str] = []
        index = 0
        while index < len(tokens):
            if tokens[index] == "right" and tokens[index + 1 : index + 2] == ["now"]:
                index += 2
                continue
            if tokens[index] in {"currently", "now", "still"}:
                index += 1
                continue
            cleaned.append(tokens[index])
            index += 1
        if len(cleaned) == len(tokens):
            return raw
        return " ".join(cleaned)

    def _contextual_meta_response(self, raw: str) -> str | Result | None:
        tokens = tokenize(raw)
        if not tokens:
            return None
        probe = self._correct_context_tokens(tokens)

        if len(probe) >= 2 and probe[0] == "where" and probe[1] == "did":
            rest = probe[2:]
            if rest and rest[0] == "you":
                rest = rest[1:]
            if rest and rest[0] not in lx.ACTION_VERBS:
                correction = spelling.correct_word(rest[0], {"action"})
                if correction is not None:
                    rest[0] = correction.corrected
            if rest and rest[0] in lx.ACTION_VERBS:
                rest = rest[1:]
            while rest and rest[-1] in {"go", "went"}:
                rest = rest[:-1]
            if not rest:
                return Result("Tell me which thing you mean.", "error")
            return "where is " + " ".join(rest)

        if probe[:2] == ["how", "many"] and len(probe) >= 3:
            previous = self.context.last_query
            rest = probe[2:]
            if (
                previous is not None
                and previous.noun is not None
                and rest[:1] in (["are"], ["is"])
                and (
                    len(rest) == 1
                    or rest[1] in {"at", "in", "next", "on", "there", "under"}
                )
            ):
                modifiers = " ".join(previous.adjectives)
                noun = f"{modifiers} {previous.noun}".strip()
                return f"how many {noun} {' '.join(rest)}"

        if (
            probe[:4] == ["what", "did", "you", "do"]
            or probe[:5] == ["what", "did", "you", "just", "do"]
            or probe in (["what", "happened"], ["what", "was", "that"])
        ):
            return self._describe_last_turn()

        if len(probe) >= 4 and probe[:3] == ["what", "did", "you"]:
            rest = probe[3:]
            if rest and rest[0] not in lx.ACTION_VERBS:
                correction = spelling.correct_word(rest[0], {"action"})
                if correction is not None:
                    rest = [correction.corrected, *rest[1:]]
            action = self.context.last_action
            if action is not None and rest:
                verb = lx.canonical_action(rest[0])
                rest = rest[1:]
                if verb == action.verb:
                    if not rest:
                        return Result(
                            f"I {action.verb} the {action.theme.label()}."
                        )
                    if action.destination is not None:
                        expected = tokenize(
                            self.world.relation_phrase(
                                action.destination.mode,
                                action.destination.target,
                            )
                        )
                        if tokenize(" ".join(rest)) == expected:
                            return Result(
                                f"I {action.verb} the {action.theme.label()} "
                                f"{self.world.relation_phrase(action.destination.mode, action.destination.target)}."
                            )

        if len(probe) >= 4 and probe[:2] == ["what", "is"] and "made" in probe:
            if probe.index("made") + 1 < len(probe) and probe[probe.index("made") + 1] == "of":
                index = probe.index("made")
                subject = " ".join(probe[2:index])
                return Result(
                    f"I do not know what material {subject} is made of in this world.",
                    "info",
                )
        return None

    def _is_repeat(self, text: str) -> bool:
        tokens = self._correct_context_tokens(tokenize(text))
        return tuple(tokens) in _REPEAT_PATTERNS

    def _repeat_last(self) -> Result:
        if self.context.last_action is not None:
            return self._finalize(self.context.last_action)
        if self.context.last_query is not None:
            return self._finalize(self.context.last_query)
        return Result("There is no previous turn to repeat.", "info")

    def _describe_last_turn(self) -> Result:
        if self.context.last_action is not None:
            return Result(f"I {self._describe_candidate(self.context.last_action)}.")
        if self.context.last_query is not None:
            return Result(
                "My last turn was "
                + self._describe_candidate(self.context.last_query)
                + "."
            )
        return Result("I haven't done anything yet.", "info")

    def _strip_context_prefix(self, raw: str) -> tuple[bool, str]:
        tokens = tokenize(raw)
        if not tokens:
            return False, raw
        probe = self._correct_context_tokens(tokens)
        if (
            len(probe) >= 2
            and probe[1] == "about"
            and probe[0] not in {"what", "how"}
        ):
            candidates = spelling.correct_word_candidates(tokens[0], {"question"})
            if any(candidate.corrected == "what" for candidate in candidates):
                probe[0] = "what"
        index = 0
        stripped = False
        while index < len(probe):
            if probe[index] in {"and", "then", "also", "or"}:
                index += 1
                stripped = True
                continue
            if (
                probe[index] in {"what", "how"}
                and index + 1 < len(probe)
                and probe[index + 1] == "about"
            ):
                index += 2
                stripped = True
                continue
            break
        if not stripped:
            return False, raw
        return True, " ".join(tokens[index:])

    def _strip_answer_prefixes(self, raw: str) -> str:
        tokens = tokenize(raw)
        if not tokens:
            return raw
        probe = self._correct_context_tokens(tokens)
        index = 0
        while index < len(probe):
            if probe[index] in {
                "actually",
                "also",
                "and",
                "no",
                "not",
                "or",
                "rather",
                "then",
            }:
                index += 1
                continue
            if (
                probe[index] in {"what", "how"}
                and index + 1 < len(probe)
                and probe[index + 1] == "about"
            ):
                index += 2
                continue
            break
        if index == 0:
            return raw
        return " ".join(tokens[index:])

    def _is_negative_answer(self, raw: str) -> bool:
        tokens = tokenize(raw)
        if not tokens:
            return False
        probe = self._correct_context_tokens(tokens)
        index = 0
        while index < len(probe) and probe[index] in {"actually", "no"}:
            index += 1
        return index < len(probe) and probe[index] == "not"

    def _is_bare_context_reference(self, raw: str) -> bool:
        if not (
            self.context.last_action is not None
            or self.context.last_query is not None
            or bool(self.context.last_results)
        ):
            return False
        tokens = tokenize(raw)
        if not tokens:
            return False
        probe = self._correct_context_tokens(tokens)
        if (
            probe[0] in lx.ACTION_VERBS
            or probe[0] in lx.QUESTION_STARTS
            or probe[0] in _CONTROL_WORDS
        ):
            return False
        reference_starts = (
            lx.DETERMINERS
            | lx.PRONOUNS
            | lx.ADJECTIVES
            | lx.NOUNS
            | {"one", "other", "same"}
        )
        return probe[0] in reference_starts

    def _extend_last_turn(self, rest: str) -> str | Result:
        reference = self._contextual_np_text(rest)
        if isinstance(reference, Result):
            return reference

        if self.context.last_query is not None:
            return self._extend_query(self.context.last_query, reference)
        if self.context.last_action is not None:
            return self._extend_action(self.context.last_action, reference)
        if self.context.last_results:
            return f"where is {reference}"
        return Result(
            "I don't have a previous question or command to extend yet.", "info"
        )

    def _answer_plural_query(self, draft: QueryDraft) -> Result:
        """Answer a question about 'they' over the current salient set."""

        referents = self._dedupe_entities(
            self.context.last_entities or self.context.last_results
        )
        if not referents:
            return Result("I do not know what 'they' refers to yet.", "error")
        destinations = self._query_destination_options(draft.destination)
        answers = [
            self._answer_query(
                Query(
                    kind=draft.kind,
                    subject=entity,
                    destination=destination,
                    attribute=draft.attribute,
                    noun=draft.noun,
                    value=draft.value,
                    adjectives=draft.adjectives,
                )
            )
            for entity, destination in product(referents, destinations)
        ]
        return Result(" ".join(answers))

    def _contextual_np_text(self, rest: str) -> str | Result:
        tokens = tokenize(rest)
        if not tokens:
            return Result("I need a thing to refer to.", "error")

        # "what about in the kitchen" and "and on the shelf" carry a location
        # relation. Drop that relation here; _extend_query chooses the right
        # one for the location entity.
        if tokens[:2] == ["next", "to"]:
            tokens = tokens[2:]
        elif tokens and tokens[0] in {
            "at",
            "in",
            "inside",
            "on",
            "onto",
            "to",
            "within",
        }:
            tokens = tokens[1:]
        if not tokens:
            return Result("I need a thing to refer to.", "error")

        probe = list(tokens)
        for index, token in enumerate(probe):
            if token in lx.ALL_WORDS:
                continue
            correction = spelling.correct_word(token, {"adjective", "context"})
            if correction is not None:
                probe[index] = correction.corrected

        core = [
            token
            for token in probe
            if token not in {"a", "an", "the", "this", "that"}
        ]
        if core in (["other"], ["other", "one"]):
            return self._other_reference_text()
        if core in (["same"], ["same", "one"], ["same", "thing"]):
            referents = self.context.last_entities or self.context.last_results
            if not referents:
                return Result("I don't know which thing was the same one.", "error")
            return "it" if len(referents) == 1 else "them"
        if tokens in (["that"], ["this"]):
            referents = self.context.last_entities or self.context.last_results
            if not referents:
                return Result("I don't know what you mean by that.", "error")
            return "it" if len(referents) == 1 else "them"
        if core and all(token in lx.ADJECTIVES for token in core):
            return "the " + " ".join(core) + " one"
        return " ".join(probe)

    def _more_about_last(self) -> Result:
        referents = self.context.last_entities or self.context.last_results
        if not referents:
            if self.context.last_action is not None:
                return Result(
                    f"I {self._describe_candidate(self.context.last_action)}."
                )
            return Result("I don't have a current subject to elaborate on yet.", "info")

        if len(referents) > 1:
            descriptions = [self._describe_entity(entity) for entity in referents[:3]]
            return Result(" ".join(descriptions), "info")

        entity = referents[0]
        text = self._describe_entity(entity)
        entry = english.lookup(entity.kind)
        if entry is not None and entry.definition:
            text += (
                f" More generally, {self._article(entity.kind)} {entity.kind} "
                f"is {entry.definition}."
            )
        return Result(text, "info")

    def _other_reference_text(self) -> str | Result:
        referents = self.context.last_entities or self.context.last_results
        if not referents:
            return Result("I don't know which thing 'the other one' refers to.", "error")

        anchor = referents[0]
        excluded = {entity.id for entity in referents}
        alternatives = [
            entity
            for entity in self.world.all()
            if entity.kind == anchor.kind and entity.id not in excluded
        ]
        if len(alternatives) == 1:
            return f"the {alternatives[0].label()}"
        if not alternatives:
            return Result(f"I don't see another {anchor.kind} in this world.", "error")
        labels = [f"the {entity.label()}" for entity in alternatives]
        return Result(
            "Which other one do you mean: " + self._join_or(labels) + "?", "clarify"
        )

    def _single_location(self, text: str) -> Entity | None:
        try:
            reference = self.parser.parse_reference(text)
            entities = self._resolve_np(reference)
        except (ParseError, ResolutionError):
            return None
        if len(entities) == 1 and self.world.is_location(entities[0]):
            return entities[0]
        return None

    def _extend_query(self, query: Query, reference: str) -> str:
        if query.kind == "inventory":
            location = self._single_location(reference)
            if location is not None:
                destination = self._location_destination(location)
                return f"what is {self._pp_text(destination)}"
            return f"where is {reference}"

        if query.kind == "count" and query.noun is not None:
            location = self._single_location(reference)
            if location is not None:
                destination = self._location_destination(location)
                modifiers = " ".join(query.adjectives)
                noun = f"{modifiers} {query.noun}".strip()
                return f"how many {noun} are {self._pp_text(destination)}"
            return f"where is {reference}"

        if query.kind == "exists" and query.noun is not None:
            location = self._single_location(reference)
            if location is not None:
                destination = self._location_destination(location)
                modifiers = " ".join(query.adjectives)
                noun = f"{modifiers} {query.noun}".strip()
                return f"is there a {noun} {self._pp_text(destination)}"
            return f"where is {reference}"

        if query.kind == "where":
            return f"where is {reference}"
        if query.kind == "color":
            return f"what color is {reference}"
        if query.kind == "property" and query.value is not None:
            return f"is {reference} {query.value}"

        if query.kind == "yes_no" and query.destination is not None:
            assert query.subject is not None
            location = self._single_location(reference)
            if location is not None:
                destination = self._location_destination(location)
                return (
                    f"is the {query.subject.label()} "
                    f"{self._pp_text(destination)}"
                )
            return f"is {reference} {self._pp_text(query.destination)}"

        if query.kind in {"list", "which"} and query.destination is not None:
            location = self._single_location(reference)
            if location is not None:
                destination = self._location_destination(location)
                if query.kind == "which" and query.noun is not None:
                    return f"which {query.noun} is {self._pp_text(destination)}"
                return f"what is {self._pp_text(destination)}"
            return f"where is {reference}"

        return f"where is {reference}"

    def _extend_action(self, action: Action, reference: str) -> str | Result:
        if action.verb in lx.PUT_VERBS:
            if action.destination is None:
                return Result(
                    "The previous command did not name a destination to reuse.", "error"
                )
            return f"{action.verb} {reference} {self._pp_text(action.destination)}"

        if action.verb in lx.TAKE_VERBS:
            text = f"{action.verb} {reference}"
            if action.source is not None:
                text += " " + self._source_pp_text(action.source)
            return text

        if action.verb in lx.GIVE_VERBS:
            assert action.recipient is not None
            return (
                f"{action.verb} {reference} to the {action.recipient.label()}"
            )

        if action.verb in lx.OPEN_VERBS:
            return f"{action.verb} {reference}"

        return Result(f"I don't know how to extend {action.verb!r}.", "error")

    def _context_summary(self) -> str:
        lines = ["Conversation context:"]
        if self.context.last_action is not None:
            lines.append(
                "  last action: " + self._describe_candidate(self.context.last_action)
            )
        elif self.context.last_query is not None:
            lines.append(
                "  last question: " + self._describe_candidate(self.context.last_query)
            )
        if self.context.last_entities:
            labels = ", ".join(
                f"the {entity.label()}" for entity in self.context.last_entities
            )
            lines.append(f"  salient referents: {labels}")
        if self.context.last_destination is not None:
            destination = self.context.last_destination
            lines.append(
                "  last destination: "
                + self.world.relation_phrase(destination.mode, destination.target)
            )
        if self.context.last_locations:
            restorable: list[str] = []
            for entity_id, location_id in self.context.last_locations.items():
                entity = self.world.get(entity_id)
                location = self.world.get(location_id) if location_id else None
                if entity is None:
                    continue
                phrase = f"the {location.label()}" if location else "nowhere"
                restorable.append(f"the {entity.label()} -> {phrase}")
            if restorable:
                lines.append("  restorable locations: " + "; ".join(restorable))
        if len(lines) == 1:
            lines.append("  no prior turn has been interpreted")
        return "\n".join(lines)

    def _history_text(self) -> str:
        turns = [turn for turn in self.context.turns if turn.user]
        if not turns:
            return "Chat history is empty."
        lines = [f"Chat history ({len(turns)} turns):"]
        for index, turn in enumerate(turns[-20:], start=max(1, len(turns) - 19)):
            lines.append(f"  {index}. you> {turn.user}")
            lines.append(f"     sieve> {self._brief_text(turn.assistant)}")
        return "\n".join(lines)

    def _knowledge_summary(self) -> str:
        lines = ["Learned knowledge:"]
        if self.noun_aliases:
            rules = "; ".join(
                f"{word} -> kind {kind}" for word, kind in sorted(self.noun_aliases.items())
            )
            lines.append("  noun kinds: " + rules)
        if self.learned_entity_aliases:
            rules = []
            for word, entity_id in sorted(self.learned_entity_aliases.items()):
                entity = self.world.get(entity_id)
                label = f"the {entity.label()}" if entity is not None else entity_id
                rules.append(f"{word} -> {label}")
            lines.append("  object names: " + "; ".join(rules))
        if self.parser.learned_aliases:
            rules = "; ".join(
                f"{word} -> {target}"
                for word, target in sorted(self.parser.learned_aliases.items())
            )
            lines.append("  word aliases: " + rules)
        if self.learned_definitions:
            rules = "; ".join(
                f"{word}: {self._brief_text(text, 100)}"
                for word, text in sorted(self.learned_definitions.items())
            )
            lines.append("  definitions: " + rules)
        if self.created_entity_ids:
            labels = []
            for entity_id in sorted(self.created_entity_ids):
                entity = self.world.get(entity_id)
                if entity is not None:
                    labels.append(entity.label())
            lines.append("  added objects: " + ", ".join(labels))
        if len(lines) == 1:
            lines.append("  no learned rules yet")
        if self.memory is not None and self.memory.path is not None:
            lines.append(f"  memory file: {self.memory.path}")
        else:
            lines.append("  storage: this session only")
        return "\n".join(lines)

    def _universe_summary(self) -> str:
        stats = self.universe.stats()
        grounded = len([entity for entity in self.world.all() if entity.kind != "room"])
        entries = stats.get("entries", 0)
        definitions = stats.get("definitions", 0)
        coverage = round(definitions / entries * 100) if entries else 0
        return (
            "Concept universe:\n"
            f"  lexical entries: {entries:,}\n"
            f"  definitions: {definitions:,} ({coverage}% coverage)\n"
            f"  noun concepts:  {stats.get('nouns', 0):,}\n"
            f"  verb concepts:  {stats.get('verbs', 0):,}\n"
            f"  adjective concepts: {stats.get('adjectives', 0):,}\n"
            f"  grounded objects: {grounded:,} ({len(self.created_entity_ids):,} added by you)\n"
            "Concepts are recognized and defined immediately. Add one to the "
            "physical world with, for example, 'add a red elephant on the table'."
        )

    def _forget_learning(self) -> None:
        if self.memory is not None:
            self.memory.clear()
        self.noun_aliases.clear()
        self.learned_definitions.clear()
        self.learned_entity_aliases.clear()
        self.learned_behaviors.clear()
        self.created_entity_ids.clear()
        self.parser.learned_aliases.clear()
        self.reset()

    def _learning_response(self, raw: str) -> Result | None:
        tokens = tokenize(raw)
        if not tokens or tokens[0] not in {"learn", "teach", "remember"}:
            return None
        if tokens[0] == "remember" and len(tokens) == 1:
            return None

        body = tokens[1:]
        if body[:1] == ["me"]:
            body = body[1:]
        if body[:1] == ["that"]:
            body = body[1:]
        if body[:2] in (["the", "word"], ["the", "term"]):
            body = body[2:]
        elif body[:1] in (["word"], ["term"]):
            body = body[1:]
        if body[:1] in (["a"], ["an"], ["the"]):
            body = body[1:]

        if not body:
            return Result(
                "Teach me a rule such as 'learn that car means a box' or "
                "'learn that snag means take'.",
                "error",
            )

        word = lx.normalize_token(body[0])
        rest = body[1:]
        if not rest or rest[0] not in {
            "is",
            "means",
            "refers",
            "represents",
            "denotes",
        }:
            return Result(
                "Use a form such as 'learn that car means a box', "
                "'learn that car is a kind of box', or "
                "'learn that snag means take'.",
                "error",
            )
        tail = rest[1:]
        if not tail:
            return Result(f"Tell me what {word!r} should mean.", "error")
        return self._learn_word(word, tail)

    def _implicit_definition_response(self, raw: str) -> Result | None:
        """Accept a direct definition such as 'car is a box'."""

        if self.definition_word is not None:
            return None
        tokens = tokenize(raw)
        if len(tokens) < 3:
            return None
        if tokens[0] in {
            "am",
            "are",
            "can",
            "could",
            "do",
            "does",
            "how",
            "is",
            "should",
            "was",
            "were",
            "what",
            "when",
            "where",
            "which",
            "who",
            "why",
            "will",
            "would",
        }:
            return None
        if tokens[0] in lx.ACTION_VERBS or tokens[0] in _CONTROL_WORDS:
            return None

        if tokens[0] in {"a", "an", "the"}:
            if len(tokens) < 4:
                return None
            word = tokens[1]
            rest = tokens[2:]
            tail = rest[1:] if rest and rest[0] in {"is", "means", "refers"} else []
        else:
            word = tokens[0]
            rest = tokens[1:]
            tail = rest[1:] if rest and rest[0] in {"is", "means", "refers"} else []
        if not tail:
            return None
        if (
            word in lx.DETERMINERS
            or word in lx.PRONOUNS
            or word in lx.ADJECTIVES
            or word in lx.PREPOSITION_STARTS
            or word in self.noun_aliases
            or self._entities_for_noun(word)
        ):
            return None
        if tail[0] not in lx.DETERMINERS and not self._is_noun_word(tail[0]):
            return None
        return self._learn_word(word, tail)

    def _learn_word(self, word: str, tail: list[str]) -> Result:
        word = lx.normalize_token(word)
        if not word or word in lx.DETERMINERS or word in lx.PRONOUNS:
            return Result(f"I can't learn a rule for {word!r}.", "error")

        raw_tail = list(tail)
        body = list(tail)
        if body[:1] in (["a"], ["an"], ["the"]):
            body = body[1:]
        if not body:
            return Result(f"Tell me what {word!r} should mean.", "error")

        if body[:2] in (["kind", "of"], ["type", "of"], ["sort", "of"]):
            body = body[2:]
        elif body[:1] == ["like"]:
            body = body[1:]
        if body[:1] in (["a"], ["an"], ["the"]):
            body = body[1:]

        if len(body) == 1:
            action = lx.canonical_action(body[0])
            if action in lx.ACTION_VERBS:
                return self._learn_token_alias(word, action, "action")

            adjective = lx.canonical_adjective(body[0])
            if adjective in lx.ADJECTIVES:
                return self._learn_token_alias(word, adjective, "adjective")

            relation = lx.canonical_relation(body[0])
            if relation in {
                "on",
                "in",
                "to",
                "at",
                "from",
                "next_to",
                "under",
                "above",
            }:
                return self._learn_token_alias(word, relation, "relation")

        if raw_tail[:1] == ["the"]:
            entity_result = self._learn_entity_alias(word, " ".join(raw_tail))
            if entity_result is not None:
                return entity_result

        if len(body) == 1 and self._is_noun_word(body[0]):
            return self._learn_kind_alias(word, body[0])

        if (
            len(body) >= 2
            and body[0] in lx.NOUNS
            and body[0] not in lx.GENERIC_BINDING_KINDS
            and body[1] in {
            "for",
            "with",
            "that",
            "which",
            }
        ):
            kind = body[0]
            self.learned_definitions[word] = " ".join(body)
            return self._learn_kind_alias(word, kind, definition=" ".join(body))

        behavior = self._infer_behavior(body)
        self.learned_definitions[word] = " ".join(body)
        self.learned_entity_aliases.pop(word, None)
        self.noun_aliases.pop(word, None)
        self.parser.add_noun(word)
        if behavior:
            self.learned_behaviors[word] = behavior
        return Result(
            f"Learned: {word!r} means {self._brief_text(' '.join(body))}. "
            f"{self._learning_note()} Add a physical one with 'add a {word}'.",
            "info",
            meta=True,
        )

    def _learn_token_alias(self, word: str, target: str, category: str) -> Result:
        if word == target:
            return Result(f"{word!r} already means {target!r}.", "info", meta=True)
        self.parser.learned_aliases[word] = target
        self.learned_definitions[word] = f"the {category} {target}"
        return Result(
            f"Learned: {word!r} means the {category} {target!r}. "
            f"{self._learning_note()}",
            "info",
            meta=True,
        )

    def _learn_kind_alias(
        self,
        word: str,
        kind: str,
        *,
        definition: str | None = None,
    ) -> Result:
        kind = self._resolve_kind(lx.canonical_noun(lx.normalize_token(kind)))
        kind_phrase = kind.replace("_", " ")
        if kind_phrase in lx.GENERIC_BINDING_KINDS:
            explanation = (
                f"The kind {kind_phrase!r} is too broad to ground {word!r}. "
                f"Pick a narrower kind such as box, table, shelf, or fridge."
            )
            self.context.last_rejection = explanation
            return Result(explanation, "error")
        if word == kind:
            return Result(f"{word!r} is already its own kind.", "info", meta=True)

        self.noun_aliases[word] = kind
        self.learned_entity_aliases.pop(word, None)
        self.parser.add_noun(word)
        behavior = self._behavior_for_kind(kind)
        if behavior:
            self.learned_behaviors[word] = behavior
        self.learned_definitions[word] = definition or f"a kind of {kind}"
        return Result(
            f"Learned: {word!r} behaves like the kind {kind!r}. "
            f"{self._learning_note()}",
            "info",
            meta=True,
        )

    def _learn_entity_alias(self, word: str, reference_text: str) -> Result | None:
        try:
            reference = self.parser.parse_reference(reference_text)
            entities = self._resolve_np(reference)
        except (ParseError, ResolutionError):
            return None
        if len(entities) != 1:
            return Result(
                f"That description names {len(entities)} objects; make it more specific.",
                "error",
            )
        entity = entities[0]
        self.noun_aliases.pop(word, None)
        entity.aliases.add(word)
        self.learned_entity_aliases[word] = entity.id
        self.parser.add_noun(word)
        self.learned_definitions[word] = f"the {entity.label()}"
        return Result(
            f"Learned: {word!r} refers to the {entity.label()}. "
            f"{self._learning_note()}",
            "info",
            meta=True,
        )

    def _is_noun_word(self, word: str) -> bool:
        return (
            word in lx.NOUNS
            or word in self.parser.extra_nouns
            or word in self.noun_aliases
            or word in self.learned_definitions
            or self.universe.is_noun(word)
        )

    def _behavior_for_kind(self, kind: str) -> dict[str, object]:
        resolved = self._resolve_kind(kind)
        behavior: dict[str, object] = {}
        if resolved in lx.CONTAINER_KINDS:
            behavior["container"] = True
        if resolved in lx.SUPPORT_KINDS:
            behavior["support"] = True
        return behavior

    def _infer_behavior(self, body: list[str]) -> dict[str, object]:
        words = set(body)
        behavior: dict[str, object] = {}
        if words & {
            "container",
            "contain",
            "contains",
            "hold",
            "holds",
            "receptacle",
            "vessel",
        }:
            behavior["container"] = True
        if words & {"surface", "support", "supports", "tabletop"}:
            behavior["support"] = True
        return behavior

    def _learning_note(self) -> str:
        if self.memory is None:
            return "Remembered for this session."
        if self.memory.path is not None:
            return f"Saved to {self.memory.path}."
        return "Remembered in memory."

    def _world_build_response(self, raw: str) -> Result | None:
        tokens = tokenize(raw)
        if not tokens or tokens[0] not in {
            "add",
            "create",
            "materialize",
            "spawn",
            "summon",
        }:
            return None

        body = tokens[1:]
        while body and body[0] in {"a", "an", "the", "new"}:
            body = body[1:]
        if not body:
            return Result(
                "Tell me what to add, for example 'add a red elephant on the table'.",
                "error",
            )

        noun = None
        for token in body:
            if token in lx.DETERMINERS or token in lx.ADJECTIVES:
                continue
            if token in lx.PREPOSITION_STARTS or token in {"next", "there", "here"}:
                break
            noun = token
            break
        if noun is None:
            return Result("Tell me which kind of object to add.", "error")

        if not self._is_noun_word(noun):
            self.definition_word = noun
            self.definition_original = None
            return Result(
                f"I don't recognize {noun!r}. Teach me first with "
                f"'learn that {noun} means a box', then say 'add a {noun}'.",
                "clarify",
            )

        self.parser.add_noun(noun)
        try:
            reference = self.parser.parse_reference(" ".join(body))
        except (ParseError, ResolutionError) as exc:
            return Result(f"I couldn't add that object: {exc}", "error")

        if reference.noun == "one" or not reference.noun:
            return Result("Tell me which kind of object to add.", "error")

        location = None
        if reference.modifiers:
            modifier = reference.modifiers[0]
            try:
                targets = self._resolve_np(modifier.target)
            except ResolutionError as exc:
                return Result(str(exc), "error")
            if len(targets) != 1:
                return Result(
                    f"That location names {len(targets)} objects; make it more specific.",
                    "error",
                )
            location = targets[0].id

        noun = self.universe.canonical_noun(reference.noun)
        behavior = dict(self.learned_behaviors.get(noun, {}))
        behavior.update(self._behavior_for_kind(noun))
        entity = self.universe.materialize(
            self.world,
            noun,
            adjectives=reference.adjectives,
            location=location,
            behavior=behavior,
        )
        self.created_entity_ids.add(entity.id)
        self.parser.add_noun(entity.kind)
        self.context.last_entities = [entity]
        self.context.last_results = [entity]
        self.context.last_action = None
        self.context.last_query = None
        self.context.other_entities = []

        where = self.world.location_phrase(entity)
        saved = (
            "It is saved."
            if self.memory is not None
            else "It is remembered for this session."
        )
        if location is None:
            return Result(
                f"Added the {entity.label()} to the universe. It has no physical "
                f"location yet. {saved}",
                "answer",
            )
        return Result(f"Added the {entity.label()}; it is {where}. {saved}", "answer")

    def _pp_text(self, destination: Destination) -> str:
        mode = destination.mode
        target = destination.target
        if mode in {"on", "in", "next_to"}:
            return self.world.relation_phrase(mode, target)
        if mode == "at":
            return f"at the {target.label()}"
        if mode == "to":
            return f"to the {target.label()}"
        if mode == "from":
            return f"from the {target.label()}"
        return f"{mode} the {target.label()}"

    def _source_pp_text(self, source: Destination) -> str:
        if source.mode in {"from", "on", "in", "next_to", "at"}:
            return self._pp_text(source)
        return f"from the {source.target.label()}"

    def _conversational_response(self, raw: str) -> Result | None:
        feedback = self._feedback_response(raw)
        if feedback is not None:
            return feedback
        memory_meta = self._memory_meta_response(raw)
        if memory_meta is not None:
            return memory_meta
        clarification = self._clarification_response(raw)
        if clarification is not None:
            return clarification
        about = self._about_response(raw)
        if about is not None:
            return about
        comparison = self._comparison_response(raw)
        if comparison is not None:
            return comparison
        history = self._history_response(raw)
        if history is not None:
            return history
        binding = self._binding_help_response(raw)
        if binding is not None:
            return binding
        why = self._why_response(raw)
        if why is not None:
            return why

        compact = " ".join(re.findall(r"[a-z]+", raw.lower()))
        acknowledgement = self._acknowledgement_response(compact)
        if acknowledgement is not None:
            return acknowledgement
        if compact in {"help me", "help", "what can you do", "what do you do"}:
            return Result(self.help_text(), "info")
        if compact in {"demo", "run demo", "show demo", "run the demo"}:
            return Result(
                "The scripted demo runs from the shell: python -m sieve.cli --demo. "
                "In this session you can type 'help' for examples.",
                "info",
                meta=True,
            )
        tokens = tuple(tokenize(raw))
        if not tokens:
            return None

        greetings = {
            ("hello",),
            ("hi",),
            ("hey",),
            ("good", "morning"),
            ("good", "afternoon"),
            ("good", "evening"),
        }
        if tokens in greetings:
            return Result(
                "Hello. I am SIEVE. You can ask about this world or define an English word.",
                "info",
            )

        if tokens in {
            ("how", "are", "you"),
            ("how", "is", "it", "going"),
            ("how", "are", "things"),
        }:
            return Result(
                "I am running deterministically: no model, no training, and no hidden state.",
                "info",
            )

        if tokens in {
            ("who", "are", "you"),
            ("what", "are", "you"),
            ("what", "is", "your", "name"),
            ("what", "is", "your", "purpose"),
            ("what", "is", "your", "origin"),
            ("who", "made", "you"),
            ("who", "created", "you"),
            ("who", "built", "you"),
            ("who", "wrote", "you"),
            ("who", "is", "your", "creator"),
            ("where", "do", "you", "come", "from"),
        }:
            return Result(
                "I am SIEVE, a model-free interpreter. I ground language in explicit world constraints.",
                "info",
            )

        if tokens in {
            ("are", "you", "alive"),
            ("are", "you", "conscious"),
            ("are", "you", "human"),
            ("are", "you", "sentient"),
            ("are", "you", "real"),
        }:
            return Result(
                "No. I am deterministic software running explicit rules. I can "
                "model this conversation and this small world, but I am not alive "
                "or conscious.",
                "info",
            )

        if tokens in {
            ("are", "you", "an", "ai"),
            ("are", "you", "a", "robot"),
            ("are", "you", "a", "language", "model"),
            ("are", "you", "using", "a", "model"),
        }:
            return Result(
                "I am a deterministic constraint interpreter, not a neural language model.",
                "info",
            )

        if tokens in {
            ("what", "can", "you", "do"),
            ("what", "do", "you", "do"),
            ("how", "can", "you", "help"),
            ("help", "me"),
        }:
            return Result(self.help_text(), "info")

        if tokens in {
            ("thank", "you"),
            ("thank", "you", "very", "much"),
            ("thanks",),
            ("many", "thanks"),
        }:
            return Result("You are welcome.", "info")

        if tokens in {
            ("goodbye",),
            ("bye",),
            ("see", "you"),
            ("good", "night"),
        }:
            return Result("Goodbye.", "info")

        if tokens in {
            ("where", "are", "you"),
            ("where", "am", "i"),
            ("what", "room", "am", "i", "in"),
            ("which", "room", "am", "i", "in"),
        }:
            robot = self.world.get("robot")
            if robot is None:
                return Result("I do not know where I am.", "info")
            return Result(f"I am {self.world.location_phrase(robot)}.", "info")

        if compact in {
            "tell me more",
            "tell me more about it",
            "tell me more about that",
            "can you tell me more",
            "more",
            "go on",
            "continue",
            "keep going",
            "what else do you know",
            "what else can you tell me",
            "what else can you say",
        }:
            return self._more_about_last()
        return None

    def _memory_meta_response(self, raw: str) -> Result | None:
        """Answer direct questions about learning and remembered turns."""

        compact = " ".join(re.findall(r"[a-z0-9]+", raw.lower()))
        bulk_request = bool(
            re.search(r"\b(?:learn|teach|memorize)\b", compact)
            and (
                (
                    re.search(r"\b(?:all|every)\b", compact)
                    and re.search(r"\b(?:word|words|meanings?)\b", compact)
                )
                or re.search(r"\bdictionary\b", compact)
            )
        )
        if bulk_request:
            stats = self.universe.stats()
            entries = stats.get("entries", 0)
            definitions = stats.get("definitions", 0)
            return Result(
                f"All {entries:,} bundled lexical entries already have "
                f"definitions ({definitions:,} definitions, 100% coverage). "
                "There is nothing useful to copy with 'learn that': ask "
                "'define X' or 'what is X' for any bundled word. Use "
                "'learn that' for a word outside the catalog, a correction, "
                "or a world behavior, for example 'learn that snag means take'. "
                "Bulk-copying every definition into ~/.sieve/memory.json would "
                "only duplicate the lexicon; it would not add understanding.",
                "info",
                meta=True,
            )

        teach_questions = {
            "are you trainable",
            "can i teach you",
            "can you be trained",
            "can you learn",
            "can you remember",
            "can you remember things",
            "do you learn",
            "how can i teach you",
            "how do i teach you",
            "how do you learn",
        }
        if compact in teach_questions:
            return Result(
                "Yes, but not by gradient training. I learn explicit rules that "
                "you can inspect and erase. Say 'learn that car means a box', "
                "'learn that snag means take', or define a new noun such as "
                "'florp is a small container for tools'. The CLI saves those "
                "rules in ~/.sieve/memory.json and reloads them on restart.",
                "info",
                meta=True,
            )

        last_said_questions = {
            "do you remember my last message",
            "do you remember what i asked",
            "do you remember what i said",
            "what did i say last",
            "what do you remember me saying",
            "what was my last message",
        }
        if compact in last_said_questions:
            turns = [
                turn
                for turn in self.context.turns
                if not turn.is_meta and turn.user
            ]
            if not turns:
                return Result(
                    "I do not have a prior message to remember yet.",
                    "info",
                    meta=True,
                )
            turn = turns[-1]
            text = f"You last said: {turn.user!r}."
            if turn.assistant:
                text += f"\nI replied: {self._brief_text(turn.assistant)}"
            return Result(text, "info", meta=True)

        remember_questions = {
            "do you remember",
            "do you remember anything",
            "show me what you remember",
            "what do you know",
            "what do you remember",
            "what do you remember about me",
            "what have you remembered",
        }
        if compact not in remember_questions:
            return None

        learned_words = (
            set(self.learned_definitions)
            | set(self.noun_aliases)
            | set(self.parser.learned_aliases)
            | set(self.learned_entity_aliases)
        )
        turns = [
            turn for turn in self.context.turns if not turn.is_meta and turn.user
        ]
        if not learned_words and not turns:
            return Result(
                "I have no explicit rules and no earlier turns yet. Teach me "
                "with 'learn that car means a box', or ask 'history' later.",
                "info",
                meta=True,
            )
        pieces: list[str] = []
        if learned_words:
            count = len(learned_words)
            noun = "rule" if count == 1 else "rules"
            pieces.append(f"{count} explicit learned {noun}")
        if turns:
            count = len(turns)
            noun = "turn" if count == 1 else "turns"
            pieces.append(f"{count} earlier chat {noun}")
        return Result(
            "I remember " + " and ".join(pieces) + ". I do not store hidden "
            "weights or a hidden summary. Say 'knowledge' to inspect learned "
            "rules or 'history' to inspect the transcript.",
            "info",
            meta=True,
        )

    def _clarification_response(self, raw: str) -> Result | None:
        compact = " ".join(re.findall(r"[a-z]+", raw.lower()))
        clarification_requests = {
            "can you clarify",
            "can you clarify that",
            "can you explain",
            "can you explain that",
            "can you explain that again",
            "can you rephrase that",
            "clarify that",
            "could you clarify",
            "could you clarify that",
            "could you explain",
            "could you explain that",
            "explain it",
            "explain that",
            "explain that again",
            "explain this",
            "i am confused",
            "i am lost",
            "i do not understand",
            "i don t understand",
            "i do not follow",
            "i do not get it",
            "i don t get it",
            "i don t follow",
            "i m confused",
            "i m lost",
            "i dont understand",
            "i dont follow",
            "i dont get it",
            "not sure what you mean",
            "please explain",
            "please explain that",
            "rephrase that",
            "what do you mean by that",
            "what did you mean",
            "what do you mean",
            "what does it mean",
            "what does that mean",
            "what is that",
            "what s that",
            "what was that",
            "what was that again",
        }
        restate_requests = {
            "come again",
            "huh",
            "pardon",
            "repeat that",
            "say that again",
            "what did you say",
            "what did you say again",
            "what was that again",
        }
        if compact not in clarification_requests and compact not in restate_requests:
            return None

        if compact in restate_requests:
            turns = [
                turn
                for turn in self.context.turns
                if not turn.is_meta and turn.assistant
            ]
            if turns:
                turn = turns[-1]
                return Result(
                    f"My last answer was: {self._brief_text(turn.assistant)}\n"
                    f"You said: {turn.user!r}",
                    "info",
                    meta=True,
                )
            return Result(
                "I have not answered anything yet. Ask about the world or define a word.",
                "info",
                meta=True,
            )

        referents = self.context.last_entities or self.context.last_results
        if referents:
            return self._more_about_last()
        if self.context.last_action is not None or self.context.last_query is not None:
            return self._describe_last_turn()
        turns = [turn for turn in self.context.turns if not turn.is_meta and turn.assistant]
        if turns:
            turn = turns[-1]
            return Result(
                f"My last answer was: {self._brief_text(turn.assistant)}\n"
                f"You said: {turn.user!r}",
                "info",
                meta=True,
            )
        return Result(
            "I have not answered anything yet. Ask about the world or define a word.",
            "info",
            meta=True,
        )

    def _about_response(self, raw: str) -> Result | None:
        """Resolve 'tell me about X', 'describe X', and similar requests."""

        lower = raw.strip().lower()
        prefixes = (
            "tell me about ",
            "what do you mean by ",
            "what does that mean by ",
            "describe ",
            "explain ",
        )
        body_text: str | None = None
        matched = ""
        for prefix in prefixes:
            if lower.startswith(prefix):
                body_text = raw.strip()[len(prefix) :].strip(" ?.!")
                matched = prefix
                break
        if body_text is None:
            return None
        if not body_text:
            if matched == "explain ":
                return None
            return self._more_about_last()
        body = tokenize(body_text)
        if len(body) == 1 and body[0] in {"it", "that", "this"}:
            return self._more_about_last()

        text = body_text
        try:
            reference = self.parser.parse_reference(text)
            entities = self._resolve_np(reference)
        except (ParseError, ResolutionError):
            entities = []
        if entities:
            return self._entity_choice_response(entities, raw)

        definition = self._dictionary_response("define " + text)
        if definition is not None:
            return definition
        return Result(
            f"I have no grounded or lexical entry for {text!r}.",
            "info",
        )

    def _acknowledgement_response(self, compact: str) -> Result | None:
        acknowledgements = {
            "alright",
            "all right",
            "awesome",
            "cool",
            "excellent",
            "great",
            "got it",
            "got that",
            "i get it",
            "i see",
            "i understand",
            "makes sense",
            "nice",
            "ok",
            "okay",
            "perfect",
            "right",
            "sounds good",
            "sure",
            "that is clear",
            "that s clear",
            "that makes sense",
            "understood",
            "yes i see",
        }
        courtesy = {
            "a",
            "again",
            "and",
            "lot",
            "much",
            "please",
            "so",
            "thank",
            "thanks",
            "very",
            "you",
        }
        words = compact.split()
        while words and words[-1] in courtesy:
            words.pop()
        core = " ".join(words)
        if compact in acknowledgements or core in acknowledgements:
            return Result(
                "Good. Ask another question whenever you are ready.",
                "info",
                meta=True,
            )
        return None

    def _comparison_response(self, raw: str) -> Result | None:
        """Answer explicit lexical comparison questions."""

        text = raw.strip().lower().rstrip("?.! ").strip()
        patterns = (
            r"^(?:what\s+is|what's|whats|what\s+are)\s+(?:the\s+)?differences?\s+between\s+(.+?)\s+and\s+(.+?)$",
            r"^(?:difference|differences)\s+between\s+(.+?)\s+and\s+(.+?)$",
            r"^(?:compare|contrast)\s+(.+?)\s+(?:and|with|to)\s+(.+?)$",
            r"^how\s+(?:are|is)\s+(.+?)\s+and\s+(.+?)\s+different$",
            r"^how\s+(?:are|is)\s+(.+?)\s+different\s+from\s+(.+?)$",
            r"^how\s+do\s+(.+?)\s+and\s+(.+?)\s+differ$",
            r"^how\s+does\s+(.+?)\s+differ\s+from\s+(.+?)$",
            r"^what\s+makes\s+(.+?)\s+different\s+from\s+(.+?)$",
            r"^(?:are|is)\s+(.+?)\s+different\s+from\s+(.+?)$",
            r"^(?:are|is)\s+(.+?)\s+and\s+(.+?)\s+the\s+same$",
            r"^(?:is|are)\s+(.+?)\s+the\s+same\s+as\s+(.+?)$",
            r"^do\s+(.+?)\s+and\s+(.+?)\s+mean\s+the\s+same$",
            r"^(.+?)\s+(?:vs\.?|versus)\s+(.+?)$",
        )
        left = right = ""
        for pattern in patterns:
            match = re.match(pattern, text)
            if match:
                left, right = match.group(1), match.group(2)
                break
        if not left or not right:
            return None
        return self._compare_terms(left, right)

    def _compare_terms(self, left: str, right: str) -> Result:
        def detail(term: str) -> tuple[str, english.Lexeme | None, str | None, list[Entity]]:
            tokens = [token for token in tokenize(term) if token not in {"a", "an", "the"}]
            phrase = " ".join(tokens) or term
            entry = english.lookup(phrase)
            learned = self.learned_definitions.get(phrase)
            entities: list[Entity] = []
            try:
                reference = self.parser.parse_reference(term)
                entities = self._resolve_np(reference)
            except (ParseError, ResolutionError):
                pass
            return phrase, entry, learned, entities

        left_phrase, left_entry, left_learned, left_entities = detail(left)
        right_phrase, right_entry, right_learned, right_entities = detail(right)

        if left_entry is not None and right_entry is not None:
            same = left_entry.word == right_entry.word or (
                left_entry.definition is not None
                and left_entry.definition == right_entry.definition
            )
            lead = (
                "They are synonyms or share the same definition."
                if same
                else "They are separate lexical entries."
            )
            left_text = (
                f"{left_entry.word} ({left_entry.pos_text}): "
                f"{left_entry.definition or 'No short definition is recorded.'}"
            )
            right_text = (
                f"{right_entry.word} ({right_entry.pos_text}): "
                f"{right_entry.definition or 'No short definition is recorded.'}"
            )
            return Result(f"{lead}\n{left_text}\n{right_text}")

        if left_learned is not None or right_learned is not None:
            left_text = left_learned or (
                left_entry.definition if left_entry is not None else "no learned definition"
            )
            right_text = right_learned or (
                right_entry.definition if right_entry is not None else "no learned definition"
            )
            return Result(
                f"{left_phrase}: {left_text}\n{right_phrase}: {right_text}"
            )

        if len(left_entities) == 1 and len(right_entities) == 1:
            return Result(
                f"{self._describe_entity(left_entities[0])}\n"
                f"{self._describe_entity(right_entities[0])}"
            )

        return Result(
            f"I have no lexical or grounded entries for "
            f"{left_phrase!r} and {right_phrase!r}.",
            "info",
        )

    def _looks_like_question(self, text: str) -> bool:
        tokens = tokenize(text)
        if not tokens:
            return False
        if text.strip().endswith("?"):
            return True
        return tokens[0] in {
            "am",
            "are",
            "can",
            "could",
            "how",
            "is",
            "should",
            "was",
            "were",
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "will",
            "would",
        }

    def _brief_text(self, text: str, limit: int = 320) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."

    def _history_tokens(self, raw: str) -> list[str]:
        """Tokenize a history query, repairing context-word misspellings."""

        raw = re.sub(r"\b1st\b", "first", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\b2nd\b", "second", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\b3rd\b", "third", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\b4th\b", "fourth", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\b5th\b", "fifth", raw, flags=re.IGNORECASE)
        history_vocabulary = {
            "answer",
            "asked",
            "before",
            "first",
            "last",
            "message",
            "messages",
            "my",
            "previous",
            "prior",
            "question",
            "questions",
            "recent",
            "said",
            "say",
            "second",
            "third",
            "was",
            "were",
            "what",
        }
        corrected: list[str] = []
        for token in tokenize(raw):
            if token in lx.ALL_WORDS or lx.is_known_word(token):
                corrected.append(token)
                continue
            correction = spelling.correct_word(token, {"context"})
            if correction is not None:
                corrected.append(correction.corrected)
                continue
            options = [
                option.corrected
                for option in spelling.open_word_options(token, 30)
                if option.corrected in history_vocabulary
            ]
            corrected.append(options[0] if len(options) == 1 else token)
        return corrected

    def _history_response(self, raw: str) -> Result | None:
        tokens = self._history_tokens(raw)
        if not tokens:
            return None
        token_set = set(tokens)
        if not ({"what", "which", "repeat"} & token_set):
            return None

        asks_question = bool({"question", "questions", "ask", "asked"} & token_set)
        asks_assistant = bool(
            {"you", "your"} & token_set
            and {
                "answer",
                "answered",
                "reply",
                "replied",
                "response",
                "responses",
                "said",
                "say",
            }
            & token_set
        )
        if "answer" in token_set and not ({"i", "my"} & token_set):
            asks_assistant = True

        number_one = bool(re.search(r"\b(?:1|one)\b", raw.lower())) and bool(
            {"message", "messages", "question", "questions", "answer"} & token_set
        )
        ordinal_cues = {
            "before",
            "beginning",
            "earliest",
            "first",
            "initial",
            "last",
            "latest",
            "prior",
            "previous",
            "recent",
            "start",
            "second",
            "third",
            "fourth",
            "fifth",
        }
        first_topic = bool(
            {"message", "messages", "question", "questions", "answer"} & token_set
            and ordinal_cues & token_set
        )
        if number_one:
            token_set.add("first")
            first_topic = True
        if not ({"i", "my"} & token_set or asks_assistant or first_topic):
            return None

        topic_words = {
            "answer",
            "answered",
            "ask",
            "asked",
            "line",
            "lines",
            "message",
            "messages",
            "prompt",
            "question",
            "questions",
            "reply",
            "replied",
            "response",
            "responses",
            "said",
            "say",
            "sentence",
            "sentences",
            "type",
            "typed",
            "utterance",
            "utterances",
        }
        if not (topic_words & token_set):
            return None

        bare_ask = bool({"ask", "asked"} & token_set) and "did" in token_set
        bare_assistant = asks_assistant and bool({"did", "was", "were"} & token_set)

        stop_words = {
            "a",
            "about",
            "an",
            "and",
            "answer",
            "answered",
            "are",
            "ask",
            "asked",
            "at",
            "beginning",
            "be",
            "been",
            "before",
            "did",
            "do",
            "does",
            "earliest",
            "fifth",
            "first",
            "fourth",
            "i",
            "in",
            "initial",
            "is",
            "last",
            "latest",
            "line",
            "lines",
            "me",
            "message",
            "messages",
            "my",
            "of",
            "on",
            "one",
            "previous",
            "prior",
            "prompt",
            "question",
            "questions",
            "recent",
            "remember",
            "repeat",
            "reply",
            "replied",
            "response",
            "responses",
            "said",
            "say",
            "second",
            "sentence",
            "sentences",
            "start",
            "that",
            "the",
            "thing",
            "things",
            "third",
            "to",
            "type",
            "typed",
            "utterance",
            "utterances",
            "was",
            "were",
            "what",
            "which",
            "you",
            "your",
        }
        keywords = [token for token in tokens if token not in stop_words]
        if (
            not (ordinal_cues & token_set)
            and not bare_ask
            and not bare_assistant
            and not first_topic
            and not keywords
        ):
            return None

        eligible = [
            turn
            for turn in self.context.turns
            if not turn.is_meta and (turn.is_question or not asks_question)
        ]
        topic = "question" if asks_question else "message"

        if keywords:
            matched = [
                turn
                for turn in eligible
                if any(
                    keyword in f"{turn.user} {turn.assistant}".lower()
                    for keyword in keywords
                )
            ]
            if not matched:
                return Result(
                    f"I found no earlier turn about {self._join_or(keywords)}.",
                    "info",
                    meta=True,
                )
            eligible = matched

        if not eligible:
            prefix = (
                f"You have not asked a {topic} before this one."
                if asks_question
                else "You have not sent a message before this one."
            )
            earlier = [turn for turn in self.context.turns if not turn.is_meta]
            if asks_question and earlier:
                first = earlier[0]
                prefix += f" Your first message was: {first.user!r}."
            return Result(prefix, "info", meta=True)

        ordinal_index = -1
        ordinal_label = "previous"
        if token_set & {"beginning", "earliest", "first", "initial", "start"}:
            ordinal_index = 0
            ordinal_label = "first"
        else:
            for word, index in (
                ("second", 1),
                ("third", 2),
                ("fourth", 3),
                ("fifth", 4),
            ):
                if word in token_set:
                    ordinal_index = index
                    ordinal_label = word
                    break
        if keywords:
            ordinal_index = -1
        if ordinal_index >= len(eligible):
            return Result(
                f"There are only {len(eligible)} matching turns in the history.",
                "info",
                meta=True,
            )
        turn = eligible[ordinal_index]

        if asks_assistant:
            text = f"My {ordinal_label} answer was: {self._brief_text(turn.assistant)}"
            text += f"\nYou said: {turn.user!r}"
            return Result(text, "info", meta=True)
        if keywords:
            text = (
                f"The most recent turn about {self._join_or(keywords)} was: "
                f"{turn.user!r}."
            )
            if turn.assistant:
                text += f"\nI replied: {self._brief_text(turn.assistant)}"
            return Result(text, "info", meta=True)
        text = f"Your {ordinal_label} {topic} was: {turn.user!r}."
        if turn.assistant:
            text += f"\nI replied: {self._brief_text(turn.assistant)}"
        return Result(text, "info", meta=True)

    def _feedback_response(self, raw: str) -> Result | None:
        tokens = set(tokenize(raw))
        harsh = {
            "awful",
            "bad",
            "dumb",
            "garbage",
            "idiot",
            "nonsense",
            "stupid",
            "terrible",
            "useless",
        }
        makes_no_sense = {"makes", "no", "sense"} <= tokens or {
            "does",
            "not",
            "make",
            "sense",
        } <= tokens
        you_are_wrong = {"you", "are", "wrong"} <= tokens or {
            "you",
            "re",
            "wrong",
        } <= tokens
        addresses_agent = bool(
            {"assistant", "sieve", "you", "your", "yourself"} & tokens
        )
        points_at_reply = bool(
            {"it", "that", "this"} & tokens
            and {"are", "is", "was", "were"} & tokens
        )
        standalone = tokens in (
            {"dumb"},
            {"garbage"},
            {"nonsense"},
            {"stupid"},
            {"useless"},
        )
        if not (
            (harsh & tokens and (addresses_agent or points_at_reply or standalone))
            or makes_no_sense
            or you_are_wrong
        ):
            return None
        return Result(
            "That is a fair criticism. Some replies were mechanically correct "
            "but conversationally wrong: valid words were treated as typos, "
            "and a missing world rule was answered with a template. I cannot "
            "infer around a missing rule without a model, but I can expose the "
            "rule that fired. Ask 'why' for the constraint, 'what did I ask "
            "first?' for the conversation history, or 'how do I bind this?' "
            "for the missing vocabulary rule. If an answer is wrong, name the "
            "answer you expected and I will make the rule explicit.",
            "info",
        )

    def _binding_help_response(self, raw: str) -> Result | None:
        tokens = tokenize(raw)
        if not tokens:
            return None
        token_set = set(tokens)
        if not ({"bind", "binding", "bound"} & token_set):
            return None
        if tokens[0] == "why":
            return None
        if not (
            {"how", "what", "help", "explain", "can", "could", "should"}
            & token_set
        ):
            return None

        word = self._binding_target_word(tokens)
        if word is not None and self.definition_word is None:
            self.definition_word = word
            self.definition_original = None
        return Result(self._binding_help_text(word), "info")

    def _binding_target_word(self, tokens: list[str]) -> str | None:
        if self.definition_word is not None:
            return self.definition_word
        skipped = {
            "a",
            "an",
            "are",
            "as",
            "bind",
            "binding",
            "bound",
            "can",
            "could",
            "do",
            "does",
            "how",
            "i",
            "is",
            "it",
            "kind",
            "me",
            "object",
            "please",
            "should",
            "that",
            "the",
            "this",
            "to",
            "what",
            "why",
            "word",
            "would",
            "you",
        }
        for token in reversed(tokens):
            if token in skipped:
                continue
            entry = english.lookup(token)
            if entry is not None and "noun" in entry.categories:
                return entry.word
        return None

    def _binding_help_text(self, word: str | None) -> str:
        if word is not None:
            entry = english.lookup(word)
            if entry is not None:
                meaning = entry.definition or "No short definition is recorded."
                lead = (
                    f"{word!r} is recognized as English {entry.pos_text}: {meaning} "
                    "It is not bound in this world yet."
                )
            else:
                lead = f"{word!r} is not bound in this world yet."
        else:
            lead = "No word is currently waiting for a binding."
        example = word or "word"
        return (
            lead
            + "\nBinding is an explicit rule, not model training and not a guess "
            "about the real world. Choose one form:\n"
            f"  kind:   {example} is a box\n"
            f"  object: call the red box a {example}\n"
            "A kind binding gives the word the same behavior as that kind in "
            "this demo; it does not claim that every example really is a box. "
            "An object binding makes one existing object answer to the word. "
            "Say 'cancel' to drop it."
        )

    def _why_response(self, raw: str) -> Result | None:
        tokens = tokenize(raw)
        if not tokens or tokens[0] != "why":
            return None
        token_set = set(tokens)
        if {"bind", "binding", "bound", "box", "kind"} & token_set:
            return Result(self._binding_explanation(), "info")
        return Result(self.explain(), "info")

    def _binding_explanation(self) -> str:
        word = self.definition_word
        subject = f" {word!r}" if word is not None else " that word"
        return (
            f"You do not have to call{subject} a box. In this demo a kind "
            "binding is only an explicit way to import behavior the interpreter "
            "already understands: boxes can hold things, while tables and "
            "shelves can support them. It does not assert that every hat is a "
            "box in the real world. If you want a literal name for one object, "
            "use 'call the red box a hat'. If neither is what you mean, say "
            "'cancel' and describe the object with a kind already in this world."
        )

    def _known_word_binding_text(self, entry: english.Lexeme) -> str:
        meaning = entry.definition or "No short definition is recorded."
        return (
            f"{entry.word} ({entry.pos_text}): {meaning}\n"
            f"No {entry.word} is bound in this world. To use it, choose an "
            f"explicit rule: '{entry.word} is a box' treats {entry.word} as a "
            f"kind with box-like behavior, or 'call the red box a {entry.word}' "
            "makes one existing object answer to the word. Say 'cancel' to drop it."
        )

    def _starts_independent_reference(self, raw: str) -> bool:
        tokens = tokenize(raw)
        if not tokens:
            return False
        first = tokens[0]
        if (
            first in lx.ACTION_VERBS
            or first in lx.QUESTION_STARTS
            or first in lx.PREPOSITION_STARTS
            or first in _CONTROL_WORDS
        ):
            return False
        if first in lx.DETERMINERS or first in lx.ADJECTIVES or first in lx.NOUNS:
            return True
        entry = english.lookup(first)
        return entry is not None and "noun" in entry.categories

    def _bare_reference_response(self, raw: str) -> Result | None:
        tokens = tokenize(raw)
        if not tokens:
            return None
        if any(
            token in {"it", "one", "other", "same", "that", "them", "these", "they", "this", "those"}
            for token in tokens
        ):
            return None
        first = tokens[0]
        if (
            first in lx.ACTION_VERBS
            or first in lx.QUESTION_STARTS
            or first in lx.PREPOSITION_STARTS
            or first in _CONTROL_WORDS
        ):
            return None
        if any(token in lx.ACTION_VERBS for token in tokens):
            return None
        if not (
            first in lx.DETERMINERS
            or first in lx.ADJECTIVES
            or first in lx.NOUNS
            or lx.is_known_word(first)
        ):
            return None

        if len(tokens) == 1:
            entry = english.lookup(tokens[0])
            if entry is None:
                return None
            noun = lx.general_noun(tokens[0])
            if noun is None and "noun" not in entry.categories:
                return None
            matches = self._entities_for_noun(noun or tokens[0])
            if matches:
                return self._entity_choice_response(matches, raw)
            self.definition_word = entry.word
            self.definition_original = None
            return Result(
                self._known_word_binding_text(entry), "clarify", meta=True
            )

        try:
            reference = self.parser.parse_reference(raw)
            entities = self._resolve_np(reference)
        except (ParseError, ResolutionError):
            return None
        return self._entity_choice_response(entities, raw)

    def _entity_choice_response(self, entities: list[Entity], original: str) -> Result:
        if len(entities) == 1:
            return Result(self._describe_entity(entities[0]))
        if len(entities) > 8:
            kind_counts: dict[str, int] = {}
            for entity in entities:
                kind_counts[entity.kind] = kind_counts.get(entity.kind, 0) + 1
            kinds = [
                f"{count} {self._plural_noun(kind)}" for kind, count in kind_counts.items()
            ]
            question = (
                f"That description fits {len(entities)} objects across "
                + self._join_or(kinds)
                + ". Which kind do you mean?"
            )
        else:
            labels = [f"the {entity.label()}" for entity in entities]
            question = (
                f"That description fits {self._join_or(labels)}. "
                "Which one do you mean?"
            )
        candidates: list[Candidate] = [
            Query(kind="describe", subject=entity) for entity in entities
        ]
        self.pending = Pending(candidates, 0, "candidate", original, question)
        return Result(question, "clarify")

    def help_text(self) -> str:
        return (
            "Sieve demo. Meaning is what survives constraints.\n"
            "Try these, one per line:\n"
            "  put the red cup on the shelf\n"
            "  put the cup on the table            (asks which cup)\n"
            "  what is on the table\n"
            "  where is the red apple\n"
            "  is the fridge open\n"
            "  open the fridge\n"
            "  put the cup in the box              (two questions, then action)\n"
            "  what color is the apple             (asks which apple)\n"
            "  what color is the car               (offers to bind 'car')\n"
            "  car is a box                        (kind definition)\n"
            "  call the red box a car              (object definition)\n"
            "  pout the rde cuup onn the shelff    (spelling recovery)\n"
            "  clock                                (define a bare known word)\n"
            "  the red thing                        (ask among grounded matches)\n"
            "  hat is a thing                       (rejected: kind too broad)\n"
            "Broad English:\n"
            "  define serendipity                  (dictionary definition)\n"
            "  what is an elephant                 (open-vocabulary lookup)\n"
            "  define elephnat                     (cue-aware typo repair)\n"
            "  what is the red cup                 (grounded description)\n"
            "  show me everything in the fridge    (inventory)\n"
            "  how many cups are there on the table\n"
            "Context can be carried across turns:\n"
            "  where is it                         (pronoun from the last turn)\n"
            "  put it there                        (reuse the last destination)\n"
            "  put it back                         (restore its previous location)\n"
            "  what about the blue cup             (extend the last action/question)\n"
            "  where did you put it                (ask about the last move)\n"
            "  what did you do                     (describe the last turn)\n"
            "  do it again                         (repeat the last turn)\n"
            "  the other one                       (the alternative of the same kind)\n"
            "  how do I bind it?                   (binding rules, not model training)\n"
            "  why should I call it a box?         (explain what a kind binding does)\n"
            "  what was my first question?         (typed conversation history)\n"
            "  what did I type first?              (history paraphrase)\n"
            "  history                              (show recent chat history)\n"
            "  what did I say about the cup?        (search chat history)\n"
            "  what did you say about the cup?      (search past replies)\n"
            "  can you clarify / I'm confused       (explain the last answer)\n"
            "  what do you remember                 (inspect explicit memory)\n"
            "  can you learn                        (explain persistent learning)\n"
            "  what is the difference between phone and telephone\n"
            "  why are you so stupid               (handled as feedback, not a typo)\n"
            "Explicit learning (saved in ~/.sieve/memory.json by the CLI):\n"
            "  learn that car means a box           (persistent kind rule)\n"
            "  learn that snag means take           (persistent word alias)\n"
            "  learn that florp means a container   (freeform definition)\n"
            "  add a red elephant on the table      (ground an English concept)\n"
            "  knowledge                            (show learned rules)\n"
            "  universe                             (catalog size and definition coverage)\n"
            "  forget learning                      (erase learned rules and history)\n"
            "Other controls: show, why, context, forget, cancel, reset, help.\n"
            "No model, no gradient training, no embeddings."
        )

    def explain(self) -> str:
        lines: list[str] = []
        if self.pending is None:
            if self.context.last_rejection is not None:
                lines.append("The last request was rejected:")
                lines.append("  " + self.context.last_rejection)
            else:
                lines.append(
                    "No ambiguity is pending. The current reading survived every constraint."
                )
        else:
            lines.append(f"I am holding {len(self.pending.candidates)} live interpretations:")
            for candidate in self.pending.candidates:
                lines.append(f"  - {self._describe_candidate(candidate)}")
            lines.append(
                f"They differ on {self.pending.attr!r}; the question chooses the smallest split."
            )
        if self.parser.corrections:
            seen: set[tuple[str, str]] = set()
            notes: list[str] = []
            for correction in self.parser.corrections:
                key = (correction.original, correction.corrected)
                if key not in seen:
                    seen.add(key)
                    notes.append(f"{correction.original!r} -> {correction.corrected!r}")
            lines.append("Spelling corrections: " + ", ".join(notes))
        return "\n".join(lines)

    def _handle_pending(self, raw: str, *, exclude: bool = False) -> Result:
        pending = self.pending
        assert pending is not None
        pool = self._slot_entities(pending)

        try:
            reference = self.parser.parse_reference(raw)
            resolved = self._resolve_np(reference, pool=pool)
        except (ParseError, ResolutionError) as exc:
            return Result(str(exc), "error")

        resolved_ids = {entity.id for entity in resolved}
        if exclude:
            if len(resolved) == len(pool):
                return Result(
                    "That excludes every remaining option. " + pending.question,
                    "clarify",
                )
            filtered = [
                candidate
                for candidate in pending.candidates
                if self._slot_entity(candidate, pending.slot).id not in resolved_ids
            ]
        else:
            if len(resolved) == len(pool):
                return Result(
                    "That description doesn't distinguish the remaining options. "
                    + pending.question,
                    "clarify",
                )
            filtered = [
                candidate
                for candidate in pending.candidates
                if self._slot_entity(candidate, pending.slot).id in resolved_ids
            ]
        if not filtered:
            return Result("None of the remaining readings match that description.", "error")

        self.pending = None
        return self._decide(filtered, pending.original)

    def _definition_prompt(self) -> str:
        word = self.definition_word or "that word"
        entry = english.lookup(word)
        if entry is not None:
            meaning = entry.definition or "No short definition is recorded."
            return (
                f"I recognize {word!r} as English {entry.pos_text}: {meaning} "
                "It is not bound in this world yet.\n"
                f"Bind it to a kind with '{word} is a box' only if it should have "
                "box-like behavior here; a kind binding imports behavior, it does "
                "not claim every such thing really is a box.\n"
                f"Or bind one existing object with 'call the red box a {word}'. "
                f"You can also add a physical one with 'add a {word} on the table'.\n"
                "Explicit rules are remembered across restarts. Say 'cancel' to drop it."
            )
        suggestions = [
            correction.corrected
            for correction in spelling.open_word_options(word, 3)
            if correction.corrected != word
        ]
        hint = ""
        if suggestions:
            hint = " Did you mean " + self._join_or(suggestions) + "?"
        return (
            f"I don't know what {self._article(word)} {word!r} is in this world."
            + hint
            + f" Bind it to a kind, for example '{word} is a box', or to a specific "
            f"object, for example 'call the red box a {word}'. "
            f"You can also add one with 'add a {word} on the table'. "
            "Say 'cancel' to drop it."
        )

    def _repair_dictionary_phrase(
        self, word_tokens: list[str]
    ) -> tuple[english.Lexeme | None, list[tuple[str, str]]]:
        """Look up a phrase, repairing misspelled components in context."""

        base = list(word_tokens)
        common_changes: list[tuple[str, str]] = []
        for index, token in enumerate(base):
            fixed = spelling.COMMON_MISSPELLINGS.get(token)
            if fixed is not None and fixed != token:
                base[index] = fixed
                common_changes.append((token, fixed))

        phrase = " ".join(base)
        entry = english.lookup(phrase)
        if entry is not None:
            return entry, common_changes

        option_sets: list[list[tuple[str, tuple[str, str] | None, int]]] = []
        for token in base:
            options: list[tuple[str, tuple[str, str] | None, int]] = [(token, None, 0)]
            if token not in lx.ALL_WORDS:
                exact = english.lookup(token)
                if exact is None or len(token) <= 4:
                    limit = 50 if len(token) <= 4 else 20
                    for fix in spelling.open_word_options(token, limit):
                        if fix.corrected != token:
                            options.append(
                                (fix.corrected, (token, fix.corrected), fix.distance)
                            )
            option_sets.append(options)

        matches: dict[str, tuple[english.Lexeme, list[tuple[str, str]], int]] = {}
        tested = 0
        for combination in product(*option_sets):
            tested += 1
            if tested > 512:
                break
            candidate_tokens = [item[0] for item in combination]
            candidate_phrase = " ".join(candidate_tokens)
            if candidate_phrase == phrase:
                continue
            candidate = english.lookup(candidate_phrase)
            if candidate is None:
                continue
            changes = [item[1] for item in combination if item[1] is not None]
            score = sum(item[2] for item in combination) + len(changes)
            previous = matches.get(candidate.word)
            if previous is None or score < previous[2]:
                matches[candidate.word] = (candidate, changes, score)

        if not matches:
            return None, common_changes
        ranked = sorted(matches.values(), key=lambda item: (item[2], item[0].word))
        if len(ranked) > 1 and ranked[0][2] == ranked[1][2]:
            return None, common_changes
        entry, changes, _ = ranked[0]
        return entry, [*common_changes, *changes]

    def _known_component_answer(self, word_tokens: list[str]) -> Result | None:
        """Explain a non-headword phrase using its recognized components."""

        skipped = lx.DETERMINERS | lx.PREPOSITION_STARTS | {
            "and",
            "or",
            "to",
            "for",
            "with",
        }
        parts: list[str] = []
        seen: set[str] = set()
        for token in word_tokens:
            if token in skipped:
                continue
            entry = english.lookup(token)
            if entry is None:
                fixed = spelling.COMMON_MISSPELLINGS.get(token)
                entry = english.lookup(fixed) if fixed is not None else None
            if entry is None or entry.word in seen:
                continue
            seen.add(entry.word)
            meaning = entry.definition or "No short definition is recorded."
            parts.append(f"{entry.word} ({entry.pos_text}): {meaning}")
            if len(parts) == 4:
                break
        if not parts:
            return None
        phrase = " ".join(word_tokens)
        return Result(
            f"No single English entry is recorded for {phrase!r}. "
            "Recognized components: " + " | ".join(parts),
            "info",
        )

    def _spelling_response(self, tokens: list[str]) -> Result | None:
        target: list[str] | None = None
        if tokens[:1] == ["spell"] and len(tokens) > 1:
            target = tokens[1:]
        elif tokens[:4] == ["how", "do", "i", "spell"]:
            target = tokens[4:]
        elif tokens[:4] == ["how", "do", "you", "spell"]:
            target = tokens[4:]
        elif tokens[:4] == ["how", "does", "one", "spell"]:
            target = tokens[4:]
        elif tokens[:2] in (["how", "is"], ["how", "are"]) and tokens[-1:] == ["spelled"]:
            target = tokens[2:-1]
        elif tokens[:3] == ["spelling", "of"]:
            target = tokens[3:]
        elif tokens[:5] == ["what", "is", "the", "spelling", "of"]:
            target = tokens[5:]
        if target is None:
            return None

        while target and target[0] in {"a", "an", "the"}:
            target = target[1:]
        if not target:
            return Result("Tell me which word to spell.", "error")

        entry, changes = self._repair_dictionary_phrase(target)
        if entry is None:
            suggestions: list[str] = []
            for fix in spelling.open_word_options(target[-1], 3):
                if fix.corrected not in suggestions:
                    suggestions.append(fix.corrected)
            if suggestions:
                return Result(
                    "I found no dictionary headword for that spelling. "
                    "Closest forms: " + self._join_or(suggestions) + ".",
                    "info",
                )
            return Result("I found no English dictionary headword for that spelling.", "info")

        if " " in entry.word:
            spelled = repr(entry.word)
        else:
            spelled = "-".join(entry.word)
        note = ""
        if changes:
            note = f" I read {' '.join(target)!r} as {entry.word!r}."
        return Result(f"{entry.word} is spelled {spelled}.{note}")

    def _dictionary_response(self, raw: str) -> Result | None:
        tokens = tokenize(raw)
        if not tokens:
            return None

        spelling_result = self._spelling_response(tokens)
        if spelling_result is not None:
            return spelling_result

        word_tokens: list[str] | None = None
        explicit = False
        if tokens[0] in {"define", "definition"}:
            word_tokens = tokens[1:]
            explicit = True
        elif tokens[:2] == ["meaning", "of"]:
            word_tokens = tokens[2:]
            explicit = True
        elif tokens[:5] in (
            ["what", "is", "the", "definition", "of"],
            ["what", "is", "the", "meaning", "of"],
        ):
            word_tokens = tokens[5:]
            explicit = True
        elif tokens[:4] in (
            ["what", "is", "the", "word"],
            ["what", "is", "the", "term"],
        ):
            word_tokens = tokens[4:]
            explicit = True
        elif tokens[:4] == ["what", "is", "the", "meaning"]:
            if tokens[4:5] == ["of"]:
                word_tokens = tokens[5:]
            explicit = True
        elif tokens[:2] == ["what", "does"] and "mean" in tokens[2:]:
            mean_index = tokens.index("mean", 2)
            tail = [*tokens[2:mean_index], *tokens[mean_index + 1 :]]
            tail = [
                token
                for token in tail
                if token not in {"be", "it", "that", "them", "this", "to"}
            ]
            word_tokens = tail
            explicit = True
        elif (
            tokens[:2] == ["what", "is"]
            and tokens[2:3] == ["the"]
            and "of" in tokens[3:]
            and tokens[3]
            not in {
                "color",
                "colour",
                "condition",
                "content",
                "contents",
                "location",
                "size",
                "state",
                "status",
                "whereabouts",
            }
        ):
            word_tokens = tokens[3:]
            explicit = True
        elif tokens[:2] == ["what", "is"] and len(tokens) >= 3:
            tail = tokens[2:]
            if tail[0] in {"a", "an"}:
                tail = tail[1:]
            deictic = {
                "everyplace",
                "everywhere",
                "here",
                "now",
                "there",
                "today",
            }
            if tail and tail[0] not in deictic and tail[0] not in lx.DETERMINERS and tail[0] not in (
                lx.PREPOSITION_STARTS
                | lx.ACTION_VERBS
                | lx.QUESTION_STARTS
                | lx.PRONOUNS
            ):
                word_tokens = tail

        if not word_tokens:
            return None

        if explicit:
            filters = {"a", "an", "for", "of", "the", "term", "word"}
            while (
                word_tokens
                and spelling.COMMON_MISSPELLINGS.get(word_tokens[0], word_tokens[0])
                in filters
            ):
                word_tokens = word_tokens[1:]
            while (
                word_tokens
                and spelling.COMMON_MISSPELLINGS.get(word_tokens[-1], word_tokens[-1])
                in filters
            ):
                word_tokens = word_tokens[:-1]
        if not word_tokens:
            return Result("Tell me which word to define.", "error")

        phrase = " ".join(word_tokens)
        learned = self.learned_definitions.get(phrase)
        if learned is not None:
            return Result(f"{phrase} (learned): {learned}")
        entry, changes = self._repair_dictionary_phrase(word_tokens)
        if entry is None:
            components = self._known_component_answer(word_tokens)
            if components is not None:
                return components
            if explicit:
                return Result(
                    f"I found no English lexical entry for {phrase!r}.", "info"
                )
            return None
        meaning = entry.definition or "No short definition is recorded."
        note = ""
        if changes:
            note = f" (read {phrase!r} as {entry.word!r})"
        return Result(f"{entry.word} ({entry.pos_text}){note}: {meaning}")

    def _try_definition(self, raw: str) -> Result | None:
        word = self.definition_word
        if word is None:
            return None

        tokens = tokenize(raw)
        if not tokens:
            return None

        if tokens[0] in {"it", "that", "this", "one"}:
            if len(tokens) >= 2 and tokens[1] in {"is", "means", "refers"}:
                return self._bind_from_definition_tail(tokens[2:])
            if len(tokens) >= 2:
                return self._learn_word(word, tokens[1:])

        if tokens[0] in {"a", "an"} and len(tokens) >= 2 and self._is_noun_word(tokens[1]):
            if len(tokens) == 2:
                return self._finish_definition(kind=self._resolve_kind(tokens[1]))
            return self._bind_from_definition_tail(tokens[1:])

        if len(tokens) == 1 and self._is_noun_word(tokens[0]):
            return self._finish_definition(kind=self._resolve_kind(tokens[0]))

        if tokens[0] == "call":
            if len(tokens) < 3 or tokens[-1] != word:
                return None
            if tokens[-2] in {"a", "an"}:
                target_tokens = tokens[1:-2]
            else:
                target_tokens = tokens[1:-1]
            if not target_tokens:
                return Result(
                    f"Tell me which object should be called {word!r}.", "error"
                )
            return self._bind_word_to_reference(word, target_tokens)

        if tokens[0] == word:
            if len(tokens) >= 3 and tokens[1] in {"is", "means"}:
                return self._bind_from_definition_tail(tokens[2:])

        if (
            len(tokens) >= 4
            and tokens[0] in {"the", "a", "an"}
            and tokens[1] == word
            and tokens[2] == "is"
        ):
            return self._bind_from_definition_tail(tokens[3:])

        return None

    def _bind_word_to_reference(self, word: str, target_tokens: list[str]) -> Result:
        text = " ".join(target_tokens)
        try:
            reference = self.parser.parse_reference(text)
            entities = self._resolve_np(reference)
        except (ParseError, ResolutionError) as exc:
            return Result(f"I couldn't bind {word!r}: {exc}", "error")

        if len(entities) != 1:
            return Result(
                f"That description names {len(entities)} objects; make it more specific.",
                "error",
            )
        return self._finish_definition(entity=entities[0])

    def _bind_from_definition_tail(self, tail: list[str]) -> Result:
        word = self.definition_word
        assert word is not None

        body = list(tail)
        if body and body[0] in {"a", "an"}:
            body = body[1:]
        if not body:
            return Result(f"Tell me what {word!r} should mean.", "error")

        kind_phrase = " ".join(body).lower()
        if kind_phrase in lx.GENERIC_BINDING_KINDS or (
            len(body) == 1 and body[0] in lx.GENERIC_BINDING_KINDS
        ):
            count = sum(1 for entity in self.world.all() if entity.kind != "room")
            explanation = (
                f"The kind {kind_phrase!r} is too broad to ground {word!r}: it "
                f"would match all {count} non-room objects, so the original "
                "request would branch into many readings instead of one. Pick a "
                f"narrower kind, for example '{word} is a box', or bind one "
                f"object with 'call the red box a {word}'. I did not change the world."
            )
            self.context.last_rejection = explanation
            return Result(explanation, "error")

        if body[0] not in lx.NOUNS and body[0] not in self.noun_aliases:
            self.parser.correct_token(body, 0, {"noun"})

        if len(body) == 1 and self._is_noun_word(body[0]):
            kind = self._resolve_kind(body[0])
            return self._finish_definition(kind=kind)

        if len(body) > 1:
            try:
                reference = self.parser.parse_reference(" ".join(body))
                entities = self._resolve_np(reference)
            except (ParseError, ResolutionError):
                entities = []
            if len(entities) == 1:
                return self._finish_definition(entity=entities[0])

        learned = self._learn_word(word, body)
        if learned.kind == "error":
            return learned

        self.definition_word = None
        self.definition_original = None
        self.context.last_rejection = None
        return learned

    def _finish_definition(self, *, entity: Entity | None = None, kind: str | None = None) -> Result:
        word = self.definition_word
        original = self.definition_original
        assert word is not None

        if entity is not None:
            self.noun_aliases.pop(word, None)
            entity.aliases.add(word)
            self.learned_entity_aliases[word] = entity.id
            self.learned_definitions[word] = f"the {entity.label()}"
            self.parser.add_noun(word)
        elif kind is not None:
            self.noun_aliases[word] = self._resolve_kind(kind)
            self.learned_entity_aliases.pop(word, None)
            behavior = self._behavior_for_kind(kind)
            if behavior:
                self.learned_behaviors[word] = behavior
            self.learned_definitions[word] = f"a kind of {self._resolve_kind(kind)}"
            self.parser.add_noun(word)
        else:
            raise ValueError("A definition needs an entity or a kind.")

        self.definition_word = None
        self.definition_original = None
        self.context.last_rejection = None
        if original:
            return self._handle(original)
        if entity is not None:
            return Result(f"Okay. {word!r} now refers to the {entity.label()}.", "info")
        return Result(f"Okay. {word!r} now means the kind {kind!r}.", "info")

    def _resolve_kind(self, noun: str) -> str:
        seen: set[str] = set()
        while noun in self.noun_aliases and noun not in seen:
            seen.add(noun)
            noun = self.noun_aliases[noun]
        return noun

    def _noun_matches(self, entity: Entity, noun: str) -> bool:
        if lx.noun_matches(entity, noun):
            return True
        resolved = self._resolve_kind(noun)
        return resolved != noun and lx.noun_matches(entity, resolved)

    def _entities_for_noun(self, noun: str) -> list[Entity]:
        """Prefer exact entities, then fall back to a learned kind alias."""

        exact = [
            entity for entity in self.world.all() if lx.noun_matches(entity, noun)
        ]
        if exact:
            return exact
        resolved = self._resolve_kind(noun)
        if resolved == noun:
            return []
        return [
            entity
            for entity in self.world.all()
            if lx.noun_matches(entity, resolved)
        ]

    def _instantiate(self, draft: ActionDraft | QueryDraft) -> list[Candidate]:
        if isinstance(draft, ActionDraft):
            return self._instantiate_action(draft)
        return self._instantiate_query(draft)

    def _instantiate_action(self, draft: ActionDraft) -> list[Action]:
        themes = self._resolve_np(draft.theme)

        recipients: list[Entity | None] = [None]
        if draft.recipient is not None:
            recipients = self._resolve_np(draft.recipient)

        valid: list[Action] = []
        violations: list[str] = []
        for theme in themes:
            destinations = self._destination_options(draft.destination, theme)
            sources = self._source_options(draft.source, theme)
            for destination, source, recipient in product(
                destinations, sources, recipients
            ):
                action = Action(
                    verb=draft.verb,
                    theme=theme,
                    destination=destination,
                    source=source,
                    recipient=recipient,
                )
                violation = self._validate_action(action)
                if violation is None:
                    valid.append(action)
                else:
                    violations.append(violation)

        if not valid:
            if violations:
                raise ResolutionError(violations[0])
            raise ResolutionError("No interpretation survived the action constraints.")
        return valid

    def _instantiate_query(self, draft: QueryDraft) -> list[Query]:
        subjects: list[Entity | None] = [None]
        if draft.subject is not None:
            subjects = list(self._resolve_np(draft.subject))

        destinations = self._query_destination_options(draft.destination)

        return [
            Query(
                kind=draft.kind,
                subject=subject,
                destination=destination,
                attribute=draft.attribute,
                noun=draft.noun,
                value=draft.value,
                adjectives=draft.adjectives,
            )
            for subject, destination in product(subjects, destinations)
        ]

    def _destination_options(
        self, pp: PP | None, theme: Entity
    ) -> list[Destination | None]:
        if pp is None:
            return [None]
        mode = pp.mode
        if mode == "there":
            return [self._there_destination()]
        if mode == "back":
            return [self._back_destination(theme)]
        if mode == "here":
            return [self._here_destination()]
        targets = self._resolve_np(pp.target)
        return [Destination(mode, target) for target in targets]

    def _source_options(
        self, pp: PP | None, theme: Entity
    ) -> list[Destination | None]:
        if pp is None:
            return [None]
        mode = pp.mode
        if mode == "there":
            return [self._there_destination()]
        if mode == "back":
            return [self._back_destination(theme)]
        if mode == "here":
            return [self._here_destination()]
        targets = self._resolve_np(pp.target)
        return [Destination(mode, target) for target in targets]

    def _query_destination_options(
        self, pp: PP | None
    ) -> list[Destination | None]:
        if pp is None:
            return [None]
        mode = pp.mode
        if mode == "there":
            return [self._there_destination()]
        if mode == "here":
            return [self._here_destination()]
        if mode == "back":
            raise ResolutionError(
                "I need to know which object's previous place you mean by 'back'."
            )
        targets = self._resolve_np(pp.target)
        return [Destination(mode, target) for target in targets]

    def _there_destination(self) -> Destination:
        if self.context.last_destination is None:
            raise ResolutionError("I don't know where 'there' refers to yet.")
        return self.context.last_destination

    def _back_destination(self, theme: Entity) -> Destination:
        if theme.id not in self.context.last_locations:
            raise ResolutionError(
                f"I don't know where to put the {theme.label()} back yet."
            )
        location_id = self.context.last_locations[theme.id]
        location = self.world.get(location_id) if location_id is not None else None
        if location is None:
            raise ResolutionError(
                f"I don't know where the {theme.label()} came from."
            )
        return self._location_destination(location)

    def _here_destination(self) -> Destination:
        robot = self.world.get("robot")
        location = (
            self.world.get(robot.location)
            if robot is not None and robot.location is not None
            else None
        )
        if location is None:
            raise ResolutionError("I don't know where 'here' is yet.")
        return self._location_destination(location)

    def _location_destination(self, location: Entity) -> Destination:
        if self.world.is_support(location):
            return Destination("on", location)
        if self.world.is_container(location):
            return Destination("in", location)
        if self.world.is_room(location):
            return Destination("in", location)
        return Destination("at", location)

    def _resolve_np(self, np: NP, pool: list[Entity] | None = None) -> list[Entity]:
        fallback_candidates: list[Entity] | None = None
        if np.noun in {"it", "them", "they", "this", "that", "these", "those"}:
            candidates = list(pool if pool is not None else self.context.last_entities)
            if not candidates:
                raise ResolutionError(
                    f"I do not know what {np.noun!r} refers to yet."
                )
        elif np.noun == "one":
            if pool is not None:
                candidates = list(pool)
            else:
                referents = self.context.last_entities or self.context.last_results
                kinds = {entity.kind for entity in referents}
                if len(kinds) == 1:
                    kind = next(iter(kinds))
                    candidates = [
                        entity
                        for entity in self.world.all()
                        if entity.kind == kind
                    ]
                    fallback_candidates = [
                        entity for entity in self.world.all() if entity.movable
                    ]
                else:
                    candidates = [
                        entity for entity in self.world.all() if entity.movable
                    ]
        else:
            candidates = self._entities_for_noun(np.noun)
            if pool is not None:
                pool_ids = {entity.id for entity in pool}
                candidates = [entity for entity in candidates if entity.id in pool_ids]
            if not candidates:
                raise ResolutionError(self._missing_noun_message(np.noun))

        candidates = self._filter_np_candidates(np, candidates)
        if not candidates and fallback_candidates is not None:
            candidates = self._filter_np_candidates(np, fallback_candidates)
        if not candidates:
            raise ResolutionError(self._missing_noun_message(np.noun, self._np_phrase(np)))

        return self._dedupe_entities(candidates)

    def _missing_noun_message(self, noun: str, phrase: str | None = None) -> str:
        learned = self.learned_definitions.get(noun)
        if learned is not None and noun not in self.noun_aliases:
            return (
                f"I know what a {noun} is ({self._brief_text(learned)}), but no "
                f"{noun} is grounded in this world yet. Add one with "
                f"'add a {noun} on the table'."
            )
        return f"I don't see {phrase or 'a ' + noun} in this world."

    def _filter_np_candidates(self, np: NP, candidates: list[Entity]) -> list[Entity]:
        for adjective in np.adjectives:
            if adjective in {"other", "same"}:
                referents = self.context.last_entities or self.context.last_results
                if adjective == "same":
                    referent_ids = {entity.id for entity in referents}
                    candidates = [
                        entity for entity in candidates if entity.id in referent_ids
                    ]
                else:
                    excluded = {entity.id for entity in referents}
                    anchor_kind = referents[0].kind if referents else None
                    candidates = [
                        entity
                        for entity in candidates
                        if entity.id not in excluded
                        and (anchor_kind is None or entity.kind == anchor_kind)
                    ]
            else:
                candidates = [
                    entity
                    for entity in candidates
                    if lx.adjective_matches(entity, adjective)
                ]
        for modifier in np.modifiers:
            targets = self._resolve_np(modifier.target)
            candidates = [
                entity
                for entity in candidates
                if any(self._relation(entity, modifier.mode, target) for target in targets)
            ]
        return candidates

    def _relation(self, entity: Entity, mode: str, target: Entity) -> bool:
        if mode in {"to", "from"}:
            return entity.location == target.id or (
                self.world.is_room(target) and self.world.is_within(entity, target)
            )
        return self.world.relation_holds(mode, entity, target)

    def _validate_action(self, action: Action) -> str | None:
        if action.verb in lx.OPEN_VERBS:
            if not self.world.is_container(action.theme):
                return (
                    f"You can only open or close containers; the {action.theme.label()} "
                    "is not one."
                )
            return None

        if not action.theme.movable:
            return f"The {action.theme.label()} can't be moved."

        if action.verb in lx.PUT_VERBS:
            destination = action.destination
            if destination is None:
                return "Tell me where to put it."
            target = destination.target
            if target.id == action.theme.id:
                return "An object can't be placed in or on itself."
            if destination.mode == "on":
                if not self.world.is_support(target):
                    return f"The {target.label()} is not a surface you can put something on."
            elif destination.mode == "in":
                if not self.world.is_container(target) and not self.world.is_room(target):
                    return f"The {target.label()} is not a container."
            elif destination.mode == "next_to":
                if self.world.is_room(target):
                    return "You can't stand next to a whole room here."
            elif destination.mode in {"to", "at"}:
                if not self.world.is_location(target) and target.location is None:
                    return f"The {target.label()} is not a place."
            else:
                return f"I don't know how to place something {destination.mode} a {target.label()}."

        if action.verb in lx.TAKE_VERBS and action.source is not None:
            if not self.world.is_location(action.source.target):
                return f"You can't take something from the {action.source.target.label()}."

        if action.verb in lx.GIVE_VERBS:
            if action.recipient is None:
                return "Tell me who to give it to."
            if not (action.recipient.kind == "robot" or action.recipient.attrs.get("animate")):
                return f"The {action.recipient.label()} can't receive things."

        return None

    def _decide(self, candidates: list[Candidate], original: str) -> Result:
        candidates = self._dedupe_candidates(candidates)
        if not candidates:
            return Result("I couldn't form a grounded interpretation.", "error")
        if len(candidates) == 1:
            self.pending = None
            return self._finalize(candidates[0])

        self.pending = self._make_pending(candidates, original)
        return Result(self.pending.question, "clarify")

    def _make_pending(self, candidates: list[Candidate], original: str) -> Pending:
        slots = [self._focus_slots(candidate) for candidate in candidates]
        slot = 0
        for index in range(len(slots[0])):
            ids = {slots[row][index][1].id for row in range(len(slots))}
            if len(ids) > 1:
                slot = index
                break
        else:
            return Pending(
                candidates,
                0,
                "candidate",
                original,
                "Which reading did you mean: "
                + self._join_or([self._describe_candidate(c) for c in candidates])
                + "?",
            )

        attribute = self._choose_attribute(candidates, slot)
        question = self._question_for(candidates, slot, attribute)
        return Pending(candidates, slot, attribute, original, question)

    def _choose_attribute(self, candidates: list[Candidate], slot: int) -> str:
        priorities = {"kind": 0, "color": 1, "size": 2, "location": 3}
        usable: dict[str, list[object]] = {}
        for attribute in priorities:
            values: list[object] = []
            for candidate in candidates:
                entity = self._slot_entity(candidate, slot)
                if attribute == "location":
                    values.append(entity.location)
                elif attribute == "kind":
                    values.append(entity.kind)
                else:
                    values.append(entity.attrs.get(attribute))
            if all(value is not None for value in values) and len(set(values)) > 1:
                usable[attribute] = values
        if not usable:
            return "candidate"
        return min(
            usable,
            key=lambda attribute: (
                len(set(usable[attribute])),
                priorities[attribute],
            ),
        )

    def _question_for(self, candidates: list[Candidate], slot: int, attribute: str) -> str:
        entities = [self._slot_entity(candidate, slot) for candidate in candidates]

        if attribute == "kind":
            values = [entity.label() for entity in entities]
            return "Which one do you mean: " + self._join_or(
                [f"the {value}" for value in values]
            ) + "?"

        if attribute == "color":
            kinds = {entity.kind for entity in entities}
            if len(kinds) == 1:
                kind = entities[0].kind
                values = [str(entity.attrs.get("color")) for entity in entities]
                return f"Which {kind} do you mean: " + self._join_or(
                    [f"the {value} one" for value in values]
                ) + "?"
            values = [entity.label() for entity in entities]
            return "Which one do you mean: " + self._join_or(
                [f"the {value}" for value in values]
            ) + "?"

        if attribute == "size":
            kinds = {entity.kind for entity in entities}
            if len(kinds) == 1:
                kind = entities[0].kind
                values = [str(entity.attrs.get("size")) for entity in entities]
                return f"Which {kind} do you mean: " + self._join_or(
                    [f"the {value} one" for value in values]
                ) + "?"
            values = [entity.label() for entity in entities]
            return "Which one do you mean: " + self._join_or(
                [f"the {value}" for value in values]
            ) + "?"

        if attribute == "location":
            kinds = {entity.kind for entity in entities}
            if len(kinds) == 1:
                kind = entities[0].kind
                phrases = [self.world.location_phrase(entity) for entity in entities]
                return f"Which {kind} do you mean: " + self._join_or(
                    [f"the one {phrase}" for phrase in phrases]
                ) + "?"
            values = [
                f"{entity.label()} {self.world.location_phrase(entity)}"
                for entity in entities
            ]
            return "Which one do you mean: " + self._join_or(
                [f"the {value}" for value in values]
            ) + "?"

        return "Which reading did you mean: " + self._join_or(
            [self._describe_candidate(candidate) for candidate in candidates]
        ) + "?"

    def _finalize(self, candidate: Candidate) -> Result:
        self.context.last_rejection = None
        if isinstance(candidate, Action):
            previous_location = candidate.theme.location
            text = self._execute_action(candidate)
            if candidate.theme.location != previous_location:
                locations = dict(self.context.last_locations)
                locations[candidate.theme.id] = previous_location
                self.context.last_locations = locations
            self.context.last_entities = [candidate.theme]
            self.context.last_results = [candidate.theme]
            self.context.other_entities = []
            self.context.last_action = candidate
            self.context.last_query = None
            self.context.last_mode = candidate.verb
            self.context.last_destination = candidate.destination
            return Result(text)

        text = self._answer_query(candidate)
        results = self._query_results(candidate)
        self.context.last_entities = results
        self.context.last_results = results
        self.context.other_entities = []
        self.context.last_action = None
        self.context.last_query = candidate
        self.context.last_mode = candidate.kind
        if candidate.destination is not None:
            self.context.last_destination = candidate.destination
        elif candidate.kind == "where" and candidate.subject is not None:
            location = self.world.get(candidate.subject.location or "")
            if location is not None:
                self.context.last_destination = self._location_destination(location)
        return Result(text)

    def _execute_action(self, action: Action) -> str:
        if action.verb in lx.PUT_VERBS:
            destination = action.destination
            assert destination is not None
            target = destination.target

            if destination.mode in {"on", "in"}:
                if action.theme.location == target.id:
                    return (
                        f"The {action.theme.label()} is already "
                        f"{self.world.relation_phrase(destination.mode, target)}."
                    )
                action.theme.location = target.id
                return (
                    f"Done. The {action.theme.label()} is now "
                    f"{self.world.relation_phrase(destination.mode, target)}."
                )

            if destination.mode == "next_to":
                if self.world.relation_holds("next_to", action.theme, target):
                    return (
                        f"The {action.theme.label()} is already next to the {target.label()}."
                    )
                action.theme.location = target.location
                return f"Done. The {action.theme.label()} is now next to the {target.label()}."

            if destination.mode in {"to", "at"}:
                if target.kind == "robot":
                    action.theme.location = target.id
                    return (
                        f"Done. The {action.theme.label()} is now with the "
                        f"{target.label()}."
                    )
                if self.world.is_room(target):
                    action.theme.location = target.id
                    return f"Done. The {action.theme.label()} is now in the {target.label()}."
                if self.world.is_support(target):
                    action.theme.location = target.id
                    return f"Done. The {action.theme.label()} is now on the {target.label()}."
                if self.world.is_container(target):
                    action.theme.location = target.id
                    return f"Done. The {action.theme.label()} is now in the {target.label()}."
                action.theme.location = target.location
                return f"Done. The {action.theme.label()} is now next to the {target.label()}."

        if action.verb in lx.TAKE_VERBS:
            if action.source is not None:
                source = action.source
                holds = self._relation(action.theme, source.mode, source.target)
                if not holds:
                    return (
                        f"The {action.theme.label()} is not "
                        f"{self.world.relation_phrase(source.mode, source.target)}."
                    )
            action.theme.location = "robot"
            return f"Done. You are now holding the {action.theme.label()}."

        if action.verb in lx.GIVE_VERBS:
            assert action.recipient is not None
            action.theme.location = action.recipient.id
            return (
                f"Done. The {action.theme.label()} is now with the "
                f"{action.recipient.label()}."
            )

        if action.verb in lx.OPEN_VERBS:
            state = action.verb == "open"
            action.theme.attrs["open"] = state
            return (
                f"Done. The {action.theme.label()} is now "
                f"{'open' if state else 'closed'}."
            )

        return f"I don't know how to perform {action.verb!r}."

    def _query_results(self, query: Query) -> list[Entity]:
        excluded = set(query.exclude_ids)

        def keep(items: list[Entity]) -> list[Entity]:
            return [entity for entity in items if entity.id not in excluded]

        if query.kind in {"count", "exists"} and query.noun is not None:
            matches = [
                entity
                for entity in self._entities_for_noun(query.noun)
                if all(
                    lx.adjective_matches(entity, adjective)
                    for adjective in query.adjectives
                )
            ]
            if query.destination is not None:
                destination = query.destination
                matches = [
                    entity
                    for entity in matches
                    if entity.id != destination.target.id
                    and self.world.relation_holds(
                        destination.mode, entity, destination.target
                    )
                ]
            return keep(matches)
        if query.kind == "inventory":
            return keep(
                [entity for entity in self.world.all() if entity.kind != "room"]
            )
        if query.kind == "list_kind" and query.noun is not None:
            return keep(
                [
                    entity
                    for entity in self._entities_for_noun(query.noun)
                    if all(
                        lx.adjective_matches(entity, adjective)
                        for adjective in query.adjectives
                    )
                ]
            )
        if query.kind == "list" and query.destination is not None:
            destination = query.destination
            return keep(
                [
                    entity
                    for entity in self.world.all()
                    if entity.id != destination.target.id
                    and self.world.relation_holds(
                        destination.mode, entity, destination.target
                    )
                ]
            )
        if query.kind == "which" and query.destination is not None and query.noun is not None:
            destination = query.destination
            return keep(
                [
                    entity
                    for entity in self._entities_for_noun(query.noun)
                    if self.world.relation_holds(
                        destination.mode, entity, destination.target
                    )
                ]
            )
        if (
            query.kind in {
                "yes_no",
                "where",
                "color",
                "size",
                "state",
                "property",
                "describe",
            }
            and query.subject is not None
        ):
            return keep([query.subject])
        return []

    def _answer_query(self, query: Query) -> str:
        if query.kind == "count":
            assert query.noun is not None
            items = self._query_results(query)
            count = len(items)
            noun = query.noun if count == 1 else self._plural_noun(query.noun)
            modifiers = " ".join(query.adjectives)
            noun_phrase = f"{modifiers} {noun}".strip()
            verb = "is" if count == 1 else "are"
            phrase = ""
            if query.destination is not None:
                phrase = " " + self.world.relation_phrase(
                    query.destination.mode, query.destination.target
                )
            return f"There {verb} {count} {noun_phrase}{phrase}."

        if query.kind == "exists":
            assert query.noun is not None
            items = self._query_results(query)
            if not items:
                phrase = ""
                if query.destination is not None:
                    phrase = " " + self.world.relation_phrase(
                        query.destination.mode, query.destination.target
                    )
                noun = self._plural_noun(query.noun)
                return f"No. There are no {noun}{phrase}."
            count = len(items)
            noun = query.noun if count == 1 else self._plural_noun(query.noun)
            verb = "is" if count == 1 else "are"
            phrase = ""
            if query.destination is not None:
                phrase = " " + self.world.relation_phrase(
                    query.destination.mode, query.destination.target
                )
            return f"Yes. There {verb} {count} {noun}{phrase}."

        if query.kind == "inventory":
            items = self._query_results(query)
            if not items:
                return "The world is empty."
            count = len(items)
            noun = "thing" if count == 1 else "things"
            return self._sentence(
                f"The world contains {count} {noun}: {self._list_phrase(items)}."
            )

        if query.kind == "list_kind":
            assert query.noun is not None
            items = self._query_results(query)
            count = len(items)
            noun = query.noun if count == 1 else self._plural_noun(query.noun)
            modifiers = " ".join(query.adjectives)
            noun_phrase = f"{modifiers} {noun}".strip()
            if not items:
                return self._sentence(f"There are no {noun_phrase} in this world.")
            verb = "is" if count == 1 else "are"
            return self._sentence(
                f"There {verb} {count} {noun_phrase}: {self._list_phrase(items)}."
            )

        if query.kind == "yes_no":
            assert query.subject is not None and query.destination is not None
            if self.world.relation_holds(
                query.destination.mode, query.subject, query.destination.target
            ):
                return (
                    f"Yes. The {query.subject.label()} is "
                    f"{self.world.relation_phrase(query.destination.mode, query.destination.target)}."
                )
            return (
                f"No. The {query.subject.label()} is "
                f"{self.world.location_phrase(query.subject)}."
            )

        if query.kind == "where":
            assert query.subject is not None
            return (
                f"The {query.subject.label()} is "
                f"{self.world.location_phrase(query.subject)}."
            )

        if query.kind == "list":
            assert query.destination is not None
            destination = query.destination
            items = self._query_results(query)
            phrase = self.world.relation_phrase(destination.mode, destination.target)
            if not items:
                return f"Nothing is {phrase}."
            verb = "is" if len(items) == 1 else "are"
            return self._sentence(f"{self._list_phrase(items)} {verb} {phrase}.")

        if query.kind == "color":
            assert query.subject is not None
            color = query.subject.attrs.get("color")
            if color is None:
                return f"The {query.subject.label()} has no color in this world."
            return f"The {query.subject.label()} is {color}."

        if query.kind == "size":
            assert query.subject is not None
            size = query.subject.attrs.get("size")
            if size is None:
                return f"The {query.subject.label()} has no size in this world."
            return f"The {query.subject.label()} is {size}."

        if query.kind == "describe":
            assert query.subject is not None
            return self._describe_entity(query.subject)

        if query.kind in {"state", "property"}:
            assert query.subject is not None
            return self._answer_property(query.subject, query.value or "")

        if query.kind == "which":
            assert query.destination is not None and query.noun is not None
            destination = query.destination
            items = self._query_results(query)
            phrase = self.world.relation_phrase(destination.mode, destination.target)
            if not items:
                return f"No {query.noun} is {phrase}."
            verb = "is" if len(items) == 1 else "are"
            return self._sentence(f"{self._list_phrase(items)} {verb} {phrase}.")

        return "I don't know how to answer that question."

    def _answer_property(self, subject: Entity, value: str) -> str:
        if value in {"open", "closed"}:
            is_open = bool(subject.attrs.get("open"))
            yes = is_open if value == "open" else not is_open
            if yes:
                return f"Yes. The {subject.label()} is {value}."
            return f"No. The {subject.label()} is {'open' if is_open else 'closed'}."

        if value in lx.COLOR_WORDS:
            actual = subject.attrs.get("color")
            if subject.attrs.get("color") == value:
                return f"Yes. The {subject.label()} is {value}."
            return f"No. The {subject.label()} is {actual}."

        if value in lx.SIZE_WORDS:
            actual = subject.attrs.get("size")
            if subject.attrs.get("size") == value:
                return f"Yes. The {subject.label()} is {value}."
            return f"No. The {subject.label()} is {actual}."

        if subject.attrs.get(value):
            return f"Yes. The {subject.label()} is {value}."
        return f"No. The {subject.label()} is not {value}."

    @staticmethod
    def _article(word: str) -> str:
        return "an" if word[:1].lower() in "aeiou" else "a"

    @staticmethod
    def _and_phrase(values: list[str]) -> str:
        if not values:
            return ""
        if len(values) == 1:
            return values[0]
        if len(values) == 2:
            return f"{values[0]} and {values[1]}"
        return ", ".join(values[:-1]) + f", and {values[-1]}"

    def _describe_entity(self, entity: Entity) -> str:
        sentences = [
            f"The {entity.label()} is {self._article(entity.kind)} {entity.kind}."
        ]
        traits: list[str] = []
        color = entity.attrs.get("color")
        if color:
            traits.append(str(color))
        size = entity.attrs.get("size")
        if size:
            traits.append(f"{size} in size")
        if entity.attrs.get("open") is True:
            traits.append("open")
        elif entity.attrs.get("open") is False:
            traits.append("closed")
        if entity.attrs.get("full") is True:
            traits.append("full")
        elif entity.attrs.get("full") is False:
            traits.append("empty")
        if traits:
            sentences.append(f"It is {self._and_phrase(traits)}.")
        if entity.location is not None:
            sentences.append(
                f"It is {self.world.location_phrase(entity)}."
            )
        return " ".join(sentences)

    def _describe_candidate(self, candidate: Candidate) -> str:
        if isinstance(candidate, Action):
            text = f"{candidate.verb} the {candidate.theme.label()}"
            if candidate.destination is not None:
                text += " " + self.world.relation_phrase(
                    candidate.destination.mode, candidate.destination.target
                )
            if candidate.source is not None:
                text += f" from the {candidate.source.target.label()}"
            if candidate.recipient is not None:
                text += f" to the {candidate.recipient.label()}"
            return text

        if candidate.kind == "where" and candidate.subject is not None:
            return f"where the {candidate.subject.label()} is"
        if candidate.kind == "color" and candidate.subject is not None:
            return f"the color of the {candidate.subject.label()}"
        if candidate.kind == "size" and candidate.subject is not None:
            return f"the size of the {candidate.subject.label()}"
        if candidate.kind == "describe" and candidate.subject is not None:
            return f"a description of the {candidate.subject.label()}"
        if candidate.kind == "inventory":
            return "an inventory of the world"
        if candidate.kind == "list_kind" and candidate.noun is not None:
            modifiers = " ".join(candidate.adjectives)
            noun = f"{modifiers} {candidate.noun}".strip()
            return f"a list of {self._plural_noun(noun)}"
        if candidate.kind in {"yes_no", "state", "property"} and candidate.subject is not None:
            return f"a question about the {candidate.subject.label()}"
        if candidate.kind == "list" and candidate.destination is not None:
            return "what is " + self.world.relation_phrase(
                candidate.destination.mode, candidate.destination.target
            )
        if candidate.kind == "which" and candidate.destination is not None:
            return f"which {candidate.noun} is " + self.world.relation_phrase(
                candidate.destination.mode, candidate.destination.target
            )
        if candidate.kind == "count" and candidate.noun is not None:
            text = f"how many {candidate.noun}"
            if candidate.destination is not None:
                text += " are " + self.world.relation_phrase(
                    candidate.destination.mode, candidate.destination.target
                )
            return text
        if candidate.kind == "exists" and candidate.noun is not None:
            text = f"whether there is a {candidate.noun}"
            if candidate.destination is not None:
                text += " " + self.world.relation_phrase(
                    candidate.destination.mode, candidate.destination.target
                )
            return text
        return candidate.kind

    def _plural_noun(self, noun: str) -> str:
        irregular = {
            "box": "boxes",
            "shelf": "shelves",
            "fridge": "fridges",
            "apple": "apples",
            "cup": "cups",
            "object": "objects",
            "item": "items",
            "thing": "things",
        }
        if noun in irregular:
            return irregular[noun]
        if noun.endswith(("s", "x", "z", "ch", "sh")):
            return noun + "es"
        if noun.endswith("y") and len(noun) > 1 and noun[-2] not in "aeiou":
            return noun[:-1] + "ies"
        return noun + "s"

    def _focus_slots(self, candidate: Candidate) -> list[tuple[str, Entity]]:
        if isinstance(candidate, Action):
            slots: list[tuple[str, Entity]] = [("theme", candidate.theme)]
            if candidate.destination is not None:
                slots.append(("destination", candidate.destination.target))
            if candidate.source is not None:
                slots.append(("source", candidate.source.target))
            if candidate.recipient is not None:
                slots.append(("recipient", candidate.recipient))
            return slots

        slots = []
        if candidate.subject is not None:
            slots.append(("subject", candidate.subject))
        if candidate.destination is not None:
            slots.append(("target", candidate.destination.target))
        return slots

    def _slot_entity(self, candidate: Candidate, slot: int) -> Entity:
        return self._focus_slots(candidate)[slot][1]

    def _slot_entities(self, pending: Pending) -> list[Entity]:
        unique: dict[str, Entity] = {}
        for candidate in pending.candidates:
            entity = self._slot_entity(candidate, pending.slot)
            unique[entity.id] = entity
        return list(unique.values())

    def _query_focus(self, query: Query) -> list[Entity]:
        if query.kind in {
            "yes_no",
            "where",
            "color",
            "size",
            "state",
            "property",
            "describe",
        }:
            return [query.subject] if query.subject is not None else []
        return []

    def _dedupe_entities(self, entities: list[Entity]) -> list[Entity]:
        unique: dict[str, Entity] = {}
        for entity in entities:
            unique[entity.id] = entity
        return list(unique.values())

    def _dedupe_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        unique: dict[tuple[object, ...], Candidate] = {}
        for candidate in candidates:
            unique[self._candidate_key(candidate)] = candidate
        return list(unique.values())

    def _candidate_key(self, candidate: Candidate) -> tuple[object, ...]:
        if isinstance(candidate, Action):
            destination = (
                None
                if candidate.destination is None
                else (candidate.destination.mode, candidate.destination.target.id)
            )
            source = (
                None
                if candidate.source is None
                else (candidate.source.mode, candidate.source.target.id)
            )
            recipient = None if candidate.recipient is None else candidate.recipient.id
            return (
                "action",
                candidate.verb,
                candidate.theme.id,
                destination,
                source,
                recipient,
            )

        destination = (
            None
            if candidate.destination is None
            else (candidate.destination.mode, candidate.destination.target.id)
        )
        subject = None if candidate.subject is None else candidate.subject.id
        return (
            "query",
            candidate.kind,
            subject,
            destination,
            candidate.attribute,
            candidate.noun,
            candidate.value,
            candidate.adjectives,
            candidate.exclude_ids,
        )

    def _np_phrase(self, np: NP) -> str:
        parts: list[str] = []
        if np.det:
            parts.append(np.det)
        parts.extend(np.adjectives)
        parts.append(np.noun)
        for modifier in np.modifiers:
            parts.append(f"{modifier.mode} {modifier.target.text}")
        return " ".join(parts)

    def _sentence(self, text: str) -> str:
        if not text:
            return text
        return text[0].upper() + text[1:]

    def _list_phrase(self, entities: list[Entity]) -> str:
        labels = [f"the {entity.label()}" for entity in entities]
        if len(labels) == 1:
            return labels[0]
        if len(labels) == 2:
            return f"{labels[0]} and {labels[1]}"
        return ", ".join(labels[:-1]) + f", and {labels[-1]}"

    def _join_or(self, values: list[str]) -> str:
        unique: list[str] = []
        for value in values:
            if value not in unique:
                unique.append(value)
        if len(unique) == 1:
            return unique[0]
        if len(unique) == 2:
            return f"{unique[0]} or {unique[1]}"
        return ", ".join(unique[:-1]) + f", or {unique[-1]}"
