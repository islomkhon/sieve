"""Persistent, inspectable learning memory for Sieve.

The memory is deliberately plain JSON. It records explicit rules supplied by
the user: word aliases, learned definitions, created entities, world state, and
recent conversation turns. No statistical parameter is stored or updated.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MEMORY_VERSION = 1
DEFAULT_MEMORY_PATH = Path.home() / ".sieve" / "memory.json"


@dataclass
class LearningMemory:
    """A small JSON-backed store for explicitly learned rules."""

    path: Path | None = None
    noun_aliases: dict[str, str] = field(default_factory=dict)
    token_aliases: dict[str, str] = field(default_factory=dict)
    entity_aliases: dict[str, str] = field(default_factory=dict)
    definitions: dict[str, str] = field(default_factory=dict)
    behaviors: dict[str, dict[str, Any]] = field(default_factory=dict)
    created_entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    world_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    turns: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path is not None:
            self.load()

    @classmethod
    def default_path(cls) -> Path:
        override = os.environ.get("SIEVE_MEMORY_PATH")
        return Path(override).expanduser() if override else DEFAULT_MEMORY_PATH

    @classmethod
    def load_default(cls) -> "LearningMemory":
        memory = cls(path=cls.default_path())
        memory.load()
        return memory

    def load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(payload, dict):
            return
        self.noun_aliases = self._string_map(payload.get("noun_aliases"))
        self.token_aliases = self._string_map(payload.get("token_aliases"))
        self.entity_aliases = self._string_map(payload.get("entity_aliases"))
        self.definitions = self._string_map(payload.get("definitions"))
        self.behaviors = self._dict_map(payload.get("behaviors"))
        self.created_entities = self._dict_map(payload.get("created_entities"))
        self.world_state = self._dict_map(payload.get("world_state"))
        turns = payload.get("turns")
        self.turns = (
            [dict(item) for item in turns if isinstance(item, dict)]
            if isinstance(turns, list)
            else []
        )

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": MEMORY_VERSION,
            "noun_aliases": self.noun_aliases,
            "token_aliases": self.token_aliases,
            "entity_aliases": self.entity_aliases,
            "definitions": self.definitions,
            "behaviors": self.behaviors,
            "created_entities": self.created_entities,
            "world_state": self.world_state,
            "turns": self.turns[-200:],
        }
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)

    def clear(self) -> None:
        self.noun_aliases.clear()
        self.token_aliases.clear()
        self.entity_aliases.clear()
        self.definitions.clear()
        self.behaviors.clear()
        self.created_entities.clear()
        self.world_state.clear()
        self.turns.clear()
        self.save()

    @staticmethod
    def _string_map(value: object) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {str(key): str(item) for key, item in value.items()}

    @staticmethod
    def _dict_map(value: object) -> dict[str, dict[str, Any]]:
        if not isinstance(value, dict):
            return {}
        return {
            str(key): dict(item)
            for key, item in value.items()
            if isinstance(item, dict)
        }
