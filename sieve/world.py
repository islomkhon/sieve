"""A small, inspectable world model for the Sieve prototype.

Nothing here is learned. Entities, attributes, and relations are explicit.
The engine interprets language by deciding which statements about this world
survive all of an utterance's constraints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SUPPORT_KINDS = {"table", "shelf", "floor", "counter"}
CONTAINER_KINDS = {"fridge", "box", "drawer", "basket"}
ROOM_KINDS = {"room"}


@dataclass(eq=False)
class Entity:
    """A concrete thing, place, or agent in the world."""

    id: str
    kind: str
    attrs: dict[str, Any] = field(default_factory=dict)
    location: str | None = None
    movable: bool = True
    aliases: set[str] = field(default_factory=set)

    @property
    def color(self) -> str | None:
        value = self.attrs.get("color")
        return value if isinstance(value, str) else None

    @property
    def size(self) -> str | None:
        value = self.attrs.get("size")
        return value if isinstance(value, str) else None

    def label(self) -> str:
        if self.kind == "room":
            return self.id.replace("_", " ")
        color = self.color
        if color:
            return f"{color} {self.kind}"
        return self.kind.replace("_", " ")

    def short_label(self) -> str:
        color = self.color
        if color:
            return f"{color} {self.kind}"
        return self.kind

    def __repr__(self) -> str:
        return f"Entity({self.id!r})"


@dataclass
class World:
    entities: dict[str, Entity]

    def all(self) -> list[Entity]:
        return list(self.entities.values())

    def get(self, entity_id: str) -> Entity | None:
        return self.entities.get(entity_id)

    def is_support(self, entity: Entity) -> bool:
        return entity.kind in SUPPORT_KINDS or bool(entity.attrs.get("support"))

    def is_container(self, entity: Entity) -> bool:
        return entity.kind in CONTAINER_KINDS or bool(entity.attrs.get("container"))

    def is_room(self, entity: Entity) -> bool:
        return entity.kind in ROOM_KINDS

    def is_location(self, entity: Entity) -> bool:
        return self.is_support(entity) or self.is_container(entity) or self.is_room(entity)

    def relation_holds(self, relation: str, subject: Entity, target: Entity) -> bool:
        if relation == "on":
            return subject.location == target.id and self.is_support(target)
        if relation == "in":
            if self.is_room(target):
                return self.is_within(subject, target)
            return subject.location == target.id and self.is_container(target)
        if relation == "next_to":
            return (
                subject.location is not None
                and subject.location == target.location
                and subject.id != target.id
            )
        if relation == "at":
            if subject.location == target.id:
                return True
            return self.is_room(target) and self.is_within(subject, target)
        return False

    def is_within(self, subject: Entity, target: Entity) -> bool:
        """Whether subject is directly in target or nested inside it."""

        current = subject.location
        seen: set[str] = set()
        while current is not None and current not in seen:
            if current == target.id:
                return True
            seen.add(current)
            location = self.get(current)
            current = location.location if location is not None else None
        return False

    def location_phrase(self, entity: Entity) -> str:
        if entity.location is None:
            return "nowhere"
        location = self.get(entity.location)
        if location is None:
            return "somewhere unknown"
        if location.kind == "robot":
            return f"with the {location.label()}"
        if self.is_support(location):
            return f"on the {location.label()}"
        if self.is_container(location):
            return f"in the {location.label()}"
        if self.is_room(location):
            return f"in the {location.label()}"
        return f"at the {location.label()}"

    def relation_phrase(self, relation: str, target: Entity) -> str:
        if relation == "on":
            return f"on the {target.label()}"
        if relation == "in":
            return f"in the {target.label()}"
        if relation == "next_to":
            return f"next to the {target.label()}"
        if relation == "at":
            if self.is_room(target):
                return f"in the {target.label()}"
            return f"at the {target.label()}"
        return f"{relation} the {target.label()}"

    def describe_state(self) -> str:
        lines = ["World state:"]
        ordered = sorted(
            (entity for entity in self.all() if entity.kind != "room"),
            key=lambda entity: (entity.location or "", entity.id),
        )
        for entity in ordered:
            lines.append(f"  {entity.label():<16} {self.location_phrase(entity)}")
        return "\n".join(lines)


def initial_world() -> World:
    """The fixed demo world. There is no pretrained knowledge here."""

    entities: dict[str, Entity] = {}

    def add(
        entity_id: str,
        kind: str,
        location: str | None = None,
        attrs: dict[str, Any] | None = None,
        *,
        movable: bool = True,
        aliases: set[str] | None = None,
    ) -> None:
        entities[entity_id] = Entity(
            id=entity_id,
            kind=kind,
            attrs=dict(attrs or {}),
            location=location,
            movable=movable,
            aliases=set(aliases or set()),
        )

    add("living_room", "room", movable=False)
    add("kitchen", "room", movable=False)

    add("table", "table", "living_room", movable=False)
    add("shelf", "shelf", "living_room", movable=False)
    add("floor", "floor", "living_room", movable=False)
    add("fridge", "fridge", "kitchen", {"open": False}, movable=False)
    add("robot", "robot", "living_room", {"animate": True}, movable=False)

    add("red_cup", "cup", "table", {"color": "red", "size": "small"})
    add("blue_cup", "cup", "table", {"color": "blue", "size": "small"})
    add("green_apple", "apple", "table", {"color": "green", "size": "small"})
    add("red_apple", "apple", "fridge", {"color": "red", "size": "small"})
    add("blue_box", "box", "floor", {"color": "blue", "size": "medium"})
    add("red_box", "box", "shelf", {"color": "red", "size": "medium"})

    return World(entities)
