"""Stages, and the graph that connects them.

A stage is one pure computation. It declares the model parameters it consumes, the
typed inputs it requires and the typed outputs it provides, and its ``compute`` method
turns parameters and inputs into outputs. Any object with these attributes is a stage:

.. code-block:: python

    @attrs.frozen
    class Smooth:
        name: str = "smooth"
        consumes = frozenset({"width"})
        requires = {"field": Field}
        provides = {"smoothed": Slot(Field, tag="smoothed")}

        def compute(self, params, field):
            return {"smoothed": field.smoothed(params["width"])}

Inputs and outputs are matched by :class:`Slot`, a type and a tag, never by the name
of the stage that produced them. :func:`resolve` links each requirement to the single
output that matches it, and reports every problem with the graph before anything is
computed.
"""

from __future__ import annotations

import graphlib
from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

import attrs

from .params import ParamSpace


@attrs.frozen
class Slot:
    """A typed input or output of a stage, matched by its type and tag.

    A required slot is satisfied by a provided slot with the same tag whose type is
    the required type or a subclass of it. Tags tell apart several outputs of the
    same type, such as the power spectra of two fields.
    """

    type: type = attrs.field(validator=attrs.validators.instance_of(type))
    tag: str = attrs.field(default="", validator=attrs.validators.instance_of(str))

    def satisfies(self, required: Slot) -> bool:
        """Whether this provided slot can feed ``required``."""
        return self.tag == required.tag and issubclass(self.type, required.type)

    def __str__(self) -> str:
        """Describe the slot for error messages."""
        name = self.type.__qualname__
        return f"{name}[{self.tag!r}]" if self.tag else name


def as_slot(slot) -> Slot:
    """Return ``slot`` as a :class:`Slot`; a bare type has the empty tag."""
    if isinstance(slot, Slot):
        return slot
    if isinstance(slot, type):
        return Slot(slot)
    raise TypeError(f"A slot must be a type or a Slot, got {slot!r}.")


@runtime_checkable
class Stage(Protocol):
    """The interface a stage must provide.

    ``requires`` and ``provides`` map slot names to slots (or bare types). The slot
    names of ``requires`` are the keyword arguments ``compute`` receives, and those of
    ``provides`` are the keys of the dictionary it returns. ``consumes`` lists the
    model parameters the stage accepts; ``compute`` receives those that the parameter
    space defines, and uses its own defaults for the rest.
    """

    name: str
    consumes: frozenset[str]
    requires: Mapping[str, Slot | type]
    provides: Mapping[str, Slot | type]

    def compute(self, params: Mapping[str, Any], **inputs) -> dict[str, Any]:
        """Compute the provided slots from parameters and required inputs."""
        ...


def _requires(stage) -> dict[str, Slot]:
    return {k: as_slot(v) for k, v in stage.requires.items()}


def _provides(stage) -> dict[str, Slot]:
    return {k: as_slot(v) for k, v in stage.provides.items()}


@attrs.frozen
class Link:
    """A required slot of one stage, fed by a provided slot of another."""

    stage: str
    slot: str
    provider: str
    provided: str


class GraphError(ValueError):
    """A stage graph that cannot be resolved; ``problems`` lists every reason."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__(
            "The stage graph is invalid:\n" + "\n".join(f"- {p}" for p in problems)
        )


class StageError(TypeError):
    """A stage whose outputs do not match what it declares."""


@attrs.frozen
class StageGraph:
    """Stages linked by their slots, in an order in which they can be computed.

    Build one with :func:`resolve`.
    """

    stages: tuple[Stage, ...]
    links: tuple[Link, ...]

    def inputs_of(self, name: str) -> dict[str, tuple[str, str]]:
        """Map each required slot of stage ``name`` to its ``(provider, slot)``."""
        return {
            link.slot: (link.provider, link.provided)
            for link in self.links
            if link.stage == name
        }


def _describe_param(p) -> str:
    return repr(p.name) if p.name == p.feeds else f"{p.name!r} (feeds {p.feeds!r})"


def _unused_params(stages, params: ParamSpace) -> list[str]:
    """Return the parameters that reach no stage, directly or through a tie."""
    consumed = set().union(*(s.consumes for s in stages))
    by_feeds = {p.feeds: p for p in params.params}
    used = set()
    todo = [f for f in by_feeds if f in consumed]
    while todo:
        feeds = todo.pop()
        if feeds in used:
            continue
        used.add(feeds)
        tie = by_feeds[feeds].tied
        if tie is not None:
            todo.extend(tie.inputs)
    return [_describe_param(p) for p in params.params if p.feeds not in used]


def _link(stages) -> tuple[list[Link], list[str]]:
    """Link every required slot to its provider, and describe the ones that fail."""
    provided = [(s.name, k, slot) for s in stages for k, slot in _provides(s).items()]
    links, problems = [], []
    for s in stages:
        for k, need in _requires(s).items():
            matches = [
                (name, slot) for name, slot, have in provided if have.satisfies(need)
            ]
            if len(matches) == 1:
                links.append(Link(s.name, k, *matches[0]))
                continue
            where = f"Stage {s.name!r} requires {need} (slot {k!r})"
            if matches:
                found = ", ".join(f"{n!r} (slot {m!r})" for n, m in matches)
                problems.append(
                    f"{where}, which is provided by more than one stage: {found}. "
                    "Give the outputs distinct tags."
                )
                continue
            other_tags = [
                f"{name!r} provides {have}"
                for name, _, have in provided
                if issubclass(have.type, need.type)
            ]
            hint = f" ({'; '.join(other_tags)})" if other_tags else ""
            problems.append(f"{where}, but no stage provides it{hint}.")
    return links, problems


def resolve(stages: Iterable[Stage], params: ParamSpace | None = None) -> StageGraph:
    """Link stages by their slots, and check the graph without computing anything.

    Raises
    ------
    GraphError
        Listing every problem found: duplicate stage names, a required slot that no
        stage or more than one stage provides, a parameter of ``params`` that no stage
        consumes (directly or through a tie), and cycles.
    """
    stages = tuple(stages)
    for s in stages:
        if not isinstance(s, Stage):
            raise TypeError(f"{s!r} is not a stage: see yabf.stages.Stage.")

    problems = []
    names = [s.name for s in stages]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append(
            f"Several stages are named {dupes}; stage names must be unique."
        )

    links, link_problems = _link(stages)
    problems += link_problems

    if params is not None:
        unused = _unused_params(stages, params)
        if unused:
            problems.append(f"No stage consumes the parameters {', '.join(unused)}.")

    # Stages are identified by name, so cycles can only be traced once names are unique.
    graph = {n: set() for n in names}
    for link in links:
        graph[link.stage].add(link.provider)
    try:
        order = tuple(graphlib.TopologicalSorter(graph).static_order())
    except graphlib.CycleError as e:
        if not dupes:
            problems.append(f"The stages form a cycle: {' -> '.join(e.args[1])}.")

    if problems:
        raise GraphError(problems)
    by_name = {s.name: s for s in stages}
    return StageGraph(stages=tuple(by_name[n] for n in order), links=tuple(links))


def call_stage(
    stage: Stage, params: Mapping[str, Any], inputs: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute ``stage`` and check that it returns exactly the slots it provides.

    ``params`` may hold more parameters than the stage consumes; it receives only
    those it consumes.

    Raises
    ------
    StageError
        If the outputs are not exactly the provided slots, or one has the wrong type.
    """
    consumed = {k: v for k, v in params.items() if k in stage.consumes}
    out = stage.compute(consumed, **inputs)
    provides = _provides(stage)
    if not isinstance(out, Mapping) or set(out) != set(provides):
        got = sorted(out) if isinstance(out, Mapping) else type(out).__name__
        raise StageError(
            f"Stage {stage.name!r} returned {got}, but provides {sorted(provides)}."
        )
    for k, slot in provides.items():
        if not isinstance(out[k], slot.type):
            raise StageError(
                f"Stage {stage.name!r} returned a {type(out[k]).__qualname__} for "
                f"slot {k!r}, which provides {slot}."
            )
    return dict(out)
