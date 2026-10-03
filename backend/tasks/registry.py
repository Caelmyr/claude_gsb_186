"""Named map/reduce function registry.

Jobs reference a mapper and reducer **by name**; the registry resolves names to
the callables in ``worker.mappers_reducers``.  This indirection is what lets the
Master dispatch a task with only ``{"mapper": "wordcount_mapper", ...}`` — no
code is shipped over the wire, and a subprocess can resolve the function by
importing this module and looking the name up.

Each registration also carries a *contract* used by the pre-submission
preflight (see ``master.preflight``):

* ``input_shape`` / ``emits`` — what kind of input records the function
  consumes and what shape the emitted ``(key, value)`` pairs have, so the
  preflight can verify the generated dataset matches the declared processing
  without waiting for a runtime crash;
* ``params`` — declared job parameters (required/optional) with type hints, so
  a missing or mistyped ``params.pattern`` is reported on the field that owns
  it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from backend.worker import mappers_reducers as _fn

Mapper = Callable[[list[Any], dict], list[tuple[str, Any]]]
Reducer = Callable[[str, list[Any], dict], dict]

# Record / value shape vocabulary shared with the preflight:
#   text       — a string line
#   kv         — a dict with at least {"key", "value"}
#   count      — additive numeric values (counts)
#   number     — arbitrary numeric values (stats / sum)
#   any        — unconstrained
_MAPPERS: dict[str, Mapper] = {}
_REDUCERS: dict[str, Reducer] = {}
_DESCRIPTIONS: dict[str, str] = {}
_META: dict[str, dict] = {}


@dataclass
class ParamSpec:
    """Declaration of one ``params`` entry consumed by a function."""
    name: str
    type: str = "string"             # string | number | bool
    required: bool = False
    default: Any = None
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name, "type": self.type, "required": self.required,
            "default": self.default, "description": self.description,
        }


@dataclass
class FunctionSpec:
    """Contract metadata attached to a registered mapper or reducer."""
    name: str
    kind: str                        # mapper | reducer
    description: str = ""
    input_shape: str = "any"
    emits: str = "any"
    accepts: str = "any"
    params: list[ParamSpec] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "description": self.description,
            "input_shape": self.input_shape,
            "emits": self.emits,
            "accepts": self.accepts,
            "params": [p.to_dict() for p in self.params],
        }


def _store(kind: str, table: dict, name: str, fn: Callable, description: str,
           input_shape: str, emits: str, accepts: str,
           params: list[ParamSpec]) -> None:
    table[name] = fn
    _DESCRIPTIONS[name] = description
    _META[name] = FunctionSpec(
        name=name, kind=kind, description=description,
        input_shape=input_shape, emits=emits, accepts=accepts, params=params,
    )


def register_mapper(name: str, fn: Mapper, description: str = "", *,
                    input_shape: str = "any", emits: str = "any",
                    params: list[ParamSpec] | None = None) -> None:
    _store("mapper", _MAPPERS, name, fn, description,
           input_shape=input_shape, emits=emits, accepts="any",
           params=params or [])


def register_reducer(name: str, fn: Reducer, description: str = "", *,
                     accepts: str = "any",
                     params: list[ParamSpec] | None = None) -> None:
    _store("reducer", _REDUCERS, name, fn, description,
           input_shape="any", emits="any", accepts=accepts,
           params=params or [])


def get_mapper(name: str) -> Mapper:
    if name not in _MAPPERS:
        raise KeyError(f"unknown mapper: {name!r}; available: {sorted(_MAPPERS)}")
    return _MAPPERS[name]


def get_reducer(name: str) -> Reducer:
    if name not in _REDUCERS:
        raise KeyError(f"unknown reducer: {name!r}; available: {sorted(_REDUCERS)}")
    return _REDUCERS[name]


def has_mapper(name: str) -> bool:
    return name in _MAPPERS


def has_reducer(name: str) -> bool:
    return name in _REDUCERS


def mapper_spec(name: str) -> FunctionSpec | None:
    return _META.get(name) if name in _MAPPERS else None


def reducer_spec(name: str) -> FunctionSpec | None:
    return _META.get(name) if name in _REDUCERS else None


def list_mappers() -> list[dict]:
    return [_META[n].to_dict() for n in sorted(_MAPPERS)]


def list_reducers() -> list[dict]:
    return [_META[n].to_dict() for n in sorted(_REDUCERS)]


def list_all() -> dict:
    return {"mappers": list_mappers(), "reducers": list_reducers()}


# ---------------------------------------------------------------------------
# Built-in registration
# ---------------------------------------------------------------------------
register_mapper(
    "wordcount_mapper", _fn.wordcount_mapper, "WordCount 词频统计",
    input_shape="text", emits="count",
)
register_mapper(
    "word_length_mapper", _fn.word_length_mapper, "Word length 词长统计",
    input_shape="text", emits="number",
)
register_mapper(
    "grep_mapper", _fn.grep_mapper, "Grep 关键词检索",
    input_shape="text", emits="count",
    params=[ParamSpec("pattern", "string", required=False, default="map",
                      description="每行需要匹配的子串 substring matched per line")],
)
register_mapper(
    "kv_mapper", _fn.kv_mapper, "Key-value 键值聚合",
    input_shape="kv", emits="number",
)
register_mapper(
    "distinct_mapper", _fn.distinct_mapper, "Distinct 去重枚举",
    input_shape="text", emits="count",
)

register_reducer(
    "count_reducer", _fn.count_reducer, "Count 求和计数",
    accepts="count",
)
register_reducer(
    "stats_reducer", _fn.stats_reducer, "Stats 统计(min/max/avg)",
    accepts="number",
)
register_reducer(
    "sum_reducer", _fn.sum_reducer, "Sum 求和聚合",
    accepts="number",
)
