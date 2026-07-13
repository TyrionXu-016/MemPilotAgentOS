from __future__ import annotations

from dataclasses import dataclass

from agentos.models import MemoryRecord


PROFILES = {
    "learning": [
        ("space", "gravity"),
        ("ocean", "pronunciation"),
        ("sports", "spelling"),
        ("music", "vocabulary"),
        ("adventure", "grammar"),
        ("science", "orbit"),
    ],
    "family_education": [
        ("story", "fraction"),
        ("visual", "geometry"),
        ("game", "multiplication"),
        ("stepwise", "word_problem"),
        ("encouraging", "division"),
        ("hands_on", "measurement"),
    ],
    "home_service": [
        ("gentle", "warm_water"),
        ("quiet", "medicine_reminder"),
        ("concise", "find_keys"),
        ("home", "bedtime_light"),
        ("voice", "door_check"),
        ("minimal", "room_temperature"),
    ],
}

STYLE_GROUP = {style: group for group, pairs in PROFILES.items() for style, _ in pairs}
ITEM_GROUP = {item: group for group, pairs in PROFILES.items() for _, item in pairs}

LEGACY_STYLE_ALIASES = {
    "space": ("太空",),
    "story": ("故事", "故事化"),
    "home": ("温水", "书桌", "家庭服务"),
}
LEGACY_ITEM_ALIASES = {
    "gravity": ("gravity",),
    "fraction": ("分数",),
    "warm_water": ("温水", "睡前", "水杯"),
}


@dataclass(frozen=True)
class Selection:
    scenario_group: str | None = None
    style: str | None = None
    item: str | None = None
    used_memory_ids: tuple[int, ...] = ()


def select_from_goal(goal: str) -> Selection:
    lowered = goal.lower()
    style = next((value for value in STYLE_GROUP if _matches(lowered, value, LEGACY_STYLE_ALIASES)), None)
    item = next((value for value in ITEM_GROUP if _matches(lowered, value, LEGACY_ITEM_ALIASES)), None)
    group = STYLE_GROUP.get(style) or ITEM_GROUP.get(item)
    if group is None:
        if any(token in goal for token in ("家庭作业", "数学", "孩子")):
            group = "family_education"
        elif any(token in goal for token in ("睡前", "饮水", "水杯", "生活辅助")):
            group = "home_service"
        else:
            group = "learning"
    return Selection(group, style, item)


def select_from_memories(memories: list[MemoryRecord]) -> Selection:
    group = None
    style = None
    item = None
    used: list[int] = []
    for memory in memories:
        memory_group = _tag_value(memory.tags, "domain:")
        memory_style = _tag_value(memory.tags, "style:")
        memory_item = _tag_value(memory.tags, "item:")
        if memory_style is None and memory.memory_type in {"preference", "profile"}:
            memory_style = _legacy_value(memory, STYLE_GROUP, LEGACY_STYLE_ALIASES)
        if memory_item is None and memory.memory_type in {"feedback", "task", "scene"}:
            memory_item = _legacy_value(memory, ITEM_GROUP, LEGACY_ITEM_ALIASES)
        selected = False
        if group is None and memory_group:
            group = memory_group
            selected = True
        if style is None and memory_style:
            style = memory_style
            selected = True
        if item is None and memory_item:
            item = memory_item
            selected = True
        if selected and memory.id is not None:
            used.append(memory.id)
    return Selection(group, style, item, tuple(dict.fromkeys(used)))


def _matches(text: str, value: str, aliases: dict[str, tuple[str, ...]]) -> bool:
    return value in text or any(alias.lower() in text for alias in aliases.get(value, ()))


def _tag_value(tags: list[str], prefix: str) -> str | None:
    return next((tag[len(prefix):] for tag in tags if tag.startswith(prefix)), None)


def _legacy_value(
    memory: MemoryRecord,
    values: dict[str, str],
    aliases: dict[str, tuple[str, ...]],
) -> str | None:
    haystack = " ".join([memory.content, *memory.tags]).lower()
    return next((value for value in values if _matches(haystack, value, aliases)), None)
