"""Stages, and the graph that connects them.

A stage is one pure computation. It declares the model parameters it consumes, each
with a :class:`ParamSpec`, the typed inputs it requires and the typed outputs it
provides, and its ``compute`` method turns parameters and inputs into outputs. Any
object with these attributes is a stage:

.. code-block:: python

    @attrs.frozen
    class Smooth:
        name: str = "smooth"
        consumes = {
            "width": ParamSpec(1.0, domain=(0, np.inf), description="Scale, in Mpc."),
        }
        requires = {"field": Field}
        provides = {"smoothed": Slot(Field, tag="smoothed")}

        def compute(self, params, field):
            return {"smoothed": field.smoothed(params["width"])}

Inputs and outputs are matched by :class:`Slot`, a type and a tag, never by the name
of the stage that produced them. :func:`resolve` links each requirement to the single
output that matches it, or, for an :class:`All` requirement, to every output that
matches it. It also checks the parameter space against what each stage accepts, and
reports every problem it finds before anything is computed.
"""

from __future__ import annotations

import graphlib
from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

import attrs
import numpy as np

from .params import ParamSpace


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


REQUIRED = _Required()
"""The default of a parameter that a stage cannot compute without."""


def _domain(value):
    if value is None:
        return None
    low, high = (float(v) for v in value)
    if not low <= high:
        raise ValueError(f"A domain needs low <= high, got ({low:g}, {high:g}).")
    return (low, high)


def _fmt(value) -> str:
    if np.ndim(value) == 0:
        return f"{float(value):g}"
    return np.array2string(np.asarray(value), precision=4)


@attrs.frozen
class ParamSpec:
    """A model parameter that a stage accepts: its default, domain and shape.

    This describes the model, not the inference: it says which values the stage can
    compute with, not which values are believed to be likely. Priors, fiducial values
    and labels belong to :class:`~yabf.params.Param`.

    Parameters
    ----------
    default
        The value the stage uses when the parameter space does not define the
        parameter, or :data:`REQUIRED` if the stage cannot compute without it.
    domain
        The closed interval ``(low, high)`` of values the stage accepts (the ends may
        be infinite), or ``None`` if it accepts any value.
    shape
        The shape of the value: ``()`` for a scalar. A stage whose parameters depend
        on its configuration builds its specs from it: a polynomial stage of order
        ``n`` consumes ``ParamSpec(shape=(n,))``.
    description
        What the parameter means, for people reading the stage.
    """

    default: Any = REQUIRED
    domain: tuple[float, float] | None = attrs.field(default=None, converter=_domain)
    shape: tuple[int, ...] = attrs.field(default=(), converter=tuple)
    description: str = ""

    def __attrs_post_init__(self):
        """Check that the default is a value the stage accepts."""
        if not self.required and (why := self.why_not(self.default)):
            raise ValueError(f"The default {_fmt(self.default)} {why}.")

    @property
    def required(self) -> bool:
        """Whether the stage cannot compute without a value for the parameter."""
        return self.default is REQUIRED

    def why_not(self, value) -> str | None:
        """Say why the stage does not accept ``value``, or return None if it does."""
        if np.shape(value) != self.shape:
            return f"has shape {np.shape(value)}, not {self.shape}"
        return self._outside_domain(value)

    def _outside_domain(self, value) -> str | None:
        if self.domain is not None:
            low, high = self.domain
            v = np.asarray(value, dtype=float)
            if not np.all((v >= low) & (v <= high)):
                return f"is outside [{low:g}, {high:g}]"
        return None


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


@attrs.frozen
class All:
    """A requirement for every output that matches a slot, from any number of stages.

    A stage that requires ``All(Spectrum, tag="model")`` receives a tuple of every
    ``Spectrum`` output tagged ``"model"``, in the order the stages were given, without
    naming the stages that provide them: a residual (data minus every model
    component) or a sum of components, say. A stage's own outputs never satisfy its
    ``All`` requirements.

    Parameters
    ----------
    type, tag
        The slot to match, as for :class:`Slot`.
    min
        The fewest matching outputs allowed. The default of one catches a mistyped
        tag; zero allows none.
    """

    type: type = attrs.field(validator=attrs.validators.instance_of(type))
    tag: str = attrs.field(default="", validator=attrs.validators.instance_of(str))
    min: int = attrs.field(default=1, kw_only=True, validator=attrs.validators.ge(0))

    @property
    def slot(self) -> Slot:
        """The slot that each matching output satisfies."""
        return Slot(self.type, self.tag)

    def __str__(self) -> str:
        """Describe the requirement for error messages."""
        return f"every {self.slot}"


def _as_requirement(req) -> Slot | All:
    return req if isinstance(req, All) else as_slot(req)


@runtime_checkable
class Stage(Protocol):
    """The interface a stage must provide.

    ``consumes`` maps the name of each model parameter the stage accepts to a
    :class:`ParamSpec`. ``compute`` receives every one of them: the value from the
    parameter space, or the default from the spec. It is read from the stage instance,
    so it can be a property that depends on the instance's configuration: the number
    of terms of a polynomial, say, which sets the length of its coefficient vector.

    ``requires`` and ``provides`` map slot names to slots (or bare types). The slot
    names of ``requires`` are the keyword arguments ``compute`` receives, and those of
    ``provides`` are the keys of the dictionary it returns. A requirement may also be
    :class:`All`, for which ``compute`` receives a tuple of every matching output.

    A stage may also define ``check(params)``, which raises ``ValueError`` if a set of
    parameter values is inconsistent in a way no single :class:`ParamSpec` can express
    (one parameter bounded by another, say). It must be cheap: :func:`resolve` calls it
    on a few points of the parameter space, without computing anything else.

    A stage may also set ``cacheable = True`` to have a :class:`~yabf.pipeline.Pipeline`
    keep its outputs across evaluations. This suits stages with small outputs, such as
    observables, and not simulators. A cached output must not refer to the stage's
    inputs (a NumPy view of an input field, say), or it keeps them alive.
    """

    name: str
    consumes: Mapping[str, ParamSpec]
    requires: Mapping[str, Slot | All | type]
    provides: Mapping[str, Slot | type]

    def compute(self, params: Mapping[str, Any], **inputs) -> dict[str, Any]:
        """Compute the provided slots from parameters and required inputs."""
        ...


def _requires(stage) -> dict[str, Slot | All]:
    return {k: _as_requirement(v) for k, v in stage.requires.items()}


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

    def inputs_of(self, name: str) -> dict[str, tuple[tuple[str, str], ...]]:
        """Map each required slot of stage ``name`` to the outputs that feed it.

        Outputs are ``(stage, slot)`` pairs: one for an ordinary requirement, any
        number for an :class:`All` requirement.
        """
        return dict(self._inputs[name])

    def gather(
        self, name: str, outputs: Mapping[tuple[str, str], Any]
    ) -> dict[str, Any]:
        """Collect the inputs of stage ``name`` from ``outputs``, keyed by slot name.

        ``outputs`` maps ``(stage, slot)`` pairs to computed values. An :class:`All`
        requirement receives a tuple of its values.
        """
        requires = self._requirements[name]
        return {
            k: tuple(outputs[src] for src in srcs)
            if isinstance(requires[k], All)
            else outputs[srcs[0]]
            for k, srcs in self._inputs[name].items()
        }

    # Lookups computed once per graph: a pipeline uses them on every evaluation.
    @cached_property
    def _requirements(self) -> dict[str, dict[str, Slot | All]]:
        return {s.name: _requires(s) for s in self.stages}

    @cached_property
    def _inputs(self) -> dict[str, dict[str, tuple[tuple[str, str], ...]]]:
        inputs = {s.name: dict.fromkeys(s.requires, ()) for s in self.stages}
        for link in self.links:
            inputs[link.stage][link.slot] += ((link.provider, link.provided),)
        return inputs


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


def _probes(params: ParamSpace):
    """Yield points of the parameter space to check the stages at, with a label.

    The points are the fiducial point, each sampled parameter near both edges of its
    prior (with the others at their fiducial values), and a few draws from the prior.
    """
    fiducial = params.fiducial
    yield "at the fiducial point", fiducial
    for i, p in enumerate(params.sampled):
        for side, q in (("lower", 1e-3), ("upper", 1 - 1e-3)):
            theta = fiducial.copy()
            theta[i] = p.prior.ppf(q)
            yield f"with {p.name!r} near the {side} edge of its prior", theta
    for theta in params.sample_prior(8, np.random.default_rng(0)):
        yield "at a prior draw", theta


def _param_problems(stages, params: ParamSpace) -> list[str]:
    """Check the parameter space against what each stage accepts, computing nothing."""
    found: dict[tuple[str, str], str] = {}  # the first problem per stage and parameter
    by_feeds: dict[str, list] = {}  # several parameters feed a vector's elements
    for p in params.params:
        by_feeds.setdefault(p.feeds, []).append(p)
    for s in stages:
        for name, spec in s.consumes.items():
            if name not in by_feeds and spec.required:
                found[s.name, name] = (
                    f"Stage {s.name!r} requires the parameter {name!r}, which the "
                    "parameter space does not define."
                )
            # For a monotonic transform (every built-in one), the support of a prior
            # maps onto the interval between the model values at its edges. Other
            # transforms are only checked at the points from _probes below. An
            # element of a vector is checked against the domain only; the vector's
            # shape is checked at those points.
            for p in by_feeds.get(name, []):
                if not p.is_sampled:
                    continue
                check = spec.why_not if p.index is None else spec._outside_domain
                for side, q in (("lower", 0.0), ("upper", 1.0)):
                    value = p.to_model(p.prior.ppf(q))
                    if (s.name, name) not in found and (why := check(value)):
                        found[s.name, name] = (
                            f"Stage {s.name!r} rejects {name!r} at the {side} edge of "
                            f"the prior of {p.name!r}: the value {_fmt(value)} {why}."
                        )

    for where, theta in _probes(params):
        values = params.to_model(theta)
        for s in stages:
            if any((s.name, name) in found for name in s.consumes):
                continue
            for name, spec in s.consumes.items():
                if name in values and (why := spec.why_not(values[name])):
                    found[s.name, name] = (
                        f"Stage {s.name!r} rejects {name!r} {where}: the value "
                        f"{_fmt(values[name])} {why}."
                    )
            check = getattr(s, "check", None)
            if check is None or (s.name, "") in found:
                continue
            try:
                check(
                    {k: values.get(k, spec.default) for k, spec in s.consumes.items()}
                )
            except ValueError as e:
                found[s.name, ""] = (
                    f"Stage {s.name!r} rejects the parameters {where}: {e}"
                )
    return list(found.values())


def _link(stages) -> tuple[list[Link], list[str]]:
    """Link every required slot to its provider, and describe the ones that fail."""
    provided = [(s.name, k, slot) for s in stages for k, slot in _provides(s).items()]
    links, problems = [], []
    for s in stages:
        for k, need in _requires(s).items():
            if isinstance(need, All):
                matches = [
                    (name, slot)
                    for name, slot, have in provided
                    if name != s.name and have.satisfies(need.slot)
                ]
                links.extend(Link(s.name, k, *m) for m in matches)
                if len(matches) < need.min:
                    found = "1 stage provides" if matches else "no stage provides"
                    if len(matches) > 1:
                        found = f"{len(matches)} stages provide"
                    problems.append(
                        f"Stage {s.name!r} requires {need} (slot {k!r}), at least "
                        f"{need.min}, but {found} one."
                    )
                continue
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
        stage or more than one stage provides, and cycles. Given ``params``, also a
        parameter that no stage consumes (directly or through a tie), a required
        parameter that ``params`` does not define, and a parameter whose values reach
        outside what a stage accepts, either at the edges of its prior or at a few
        points of the parameter space where each stage's ``check`` is also run.
    """
    stages = tuple(stages)
    for s in stages:
        if not isinstance(s, Stage):
            raise TypeError(f"{s!r} is not a stage: see yabf.stages.Stage.")
        if not isinstance(s.consumes, Mapping) or not all(
            isinstance(spec, ParamSpec) for spec in s.consumes.values()
        ):
            raise TypeError(
                f"Stage {s.name!r} must declare `consumes` as a mapping from parameter "
                f"names to ParamSpec, got {s.consumes!r}."
            )

    problems = []
    names = [s.name for s in stages]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        problems.append(
            f"Several stages are named {dupes}; stage names must be unique."
        )
    problems.extend(
        f"Stage {s.name!r} requires a slot named 'params', which is reserved for the "
        "parameters passed to compute(); rename the slot."
        for s in stages
        if "params" in s.requires
    )

    links, link_problems = _link(stages)
    problems += link_problems

    if params is not None:
        unused = _unused_params(stages, params)
        if unused:
            problems.append(f"No stage consumes the parameters {', '.join(unused)}.")
        problems += _param_problems(stages, params)

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

    The stage receives every parameter it consumes: the value in ``params``, or the
    default from its :class:`ParamSpec`. Other entries of ``params`` are ignored.

    Raises
    ------
    StageError
        If ``params`` lacks a required parameter, or the outputs are not exactly the
        provided slots, or one has the wrong type.
    """
    missing = sorted(
        k for k, spec in stage.consumes.items() if spec.required and k not in params
    )
    if missing:
        raise StageError(
            f"Stage {stage.name!r} requires the parameters {missing}, which were not "
            "given."
        )
    consumed = {k: params.get(k, spec.default) for k, spec in stage.consumes.items()}
    out = stage.compute(consumed, **inputs)
    provides = _provides(stage)
    if not isinstance(out, Mapping) or set(out) != set(provides):
        got = sorted(out, key=str) if isinstance(out, Mapping) else type(out).__name__
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
