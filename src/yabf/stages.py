"""Stages, and the graph that connects them.

A stage is one pure computation. It declares the model parameters it takes, each with
a :class:`ParamSpec`, the named inputs it requires and the named outputs it provides,
and its ``compute`` method turns parameters and inputs into outputs. Any object with
these attributes is a stage:

.. code-block:: python

    @attrs.frozen
    class Smooth:
        name: str = "smooth"
        parameters = {
            "width": ParamSpec(1.0, domain=(0, np.inf), description="Scale, in Mpc."),
        }
        requires = {"density": Field}
        provides = {"smoothed_density": Field}

        def compute(self, params: Mapping, density: Field) -> dict[str, Field]:
            '''Smooth the density field on the scale ``width``.'''
            return {"smoothed_density": density.smoothed(params["width"])}

Outputs have a name and a type. An input asks for an output by its name and type: here,
``Smooth`` takes the ``Field`` named ``"density"`` that some other stage provides,
whichever stage that is, and provides a ``Field`` named ``"smoothed_density"``. An input
can also ask for the only output of a type, whatever its name (:class:`One`), or for
every output that matches (:class:`All`). :func:`resolve` links each input to the
outputs that match it. It also checks the parameter space against what each stage
accepts, and reports every problem it finds before anything is computed.
"""

from __future__ import annotations

import graphlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import cached_property
from typing import Any, Protocol, runtime_checkable

import attrs
import numpy as np
from numpy.typing import ArrayLike

from .params import ParamSpace


class _Required:
    """The type of :data:`REQUIRED`."""

    def __repr__(self) -> str:
        """Show the sentinel by name."""
        return "REQUIRED"


REQUIRED = _Required()
"""The default of a parameter that a stage cannot compute without."""


def _domain(value: Iterable[float] | None) -> tuple[float, float] | None:
    """Return a domain as a ``(low, high)`` pair of floats, checking low <= high."""
    if value is None:
        return None
    low, high = (float(v) for v in value)
    if not low <= high:
        raise ValueError(f"A domain needs low <= high, got ({low:g}, {high:g}).")
    return (low, high)


def _fmt(value: ArrayLike) -> str:
    """Format a scalar or array value briefly, for error messages."""
    if np.ndim(value) == 0:
        return f"{float(value):g}"
    return np.array2string(
        np.asarray(value, dtype=float), formatter={"float_kind": lambda x: f"{x:g}"}
    )


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
        ``n`` takes ``ParamSpec(shape=(n,))``.
    description
        What the parameter means, for people reading the stage.
    random
        Whether the parameter is a source of randomness, such as a random seed, rather
        than a physical quantity. A posterior reports such inputs (see
        :attr:`~yabf.posterior.Posterior.random_inputs`), since how each is set (left
        at its default, fixed, or sampled) is an approximation the run makes.
    """

    default: Any = REQUIRED
    domain: tuple[float, float] | None = attrs.field(default=None, converter=_domain)
    shape: tuple[int, ...] = attrs.field(default=(), converter=tuple)
    description: str = ""
    random: bool = attrs.field(
        default=False, validator=attrs.validators.instance_of(bool), kw_only=True
    )

    def __attrs_post_init__(self) -> None:
        """Check that the default is a value the stage accepts."""
        if not self.required and (why := self.why_not(self.default)):
            raise ValueError(f"The default {_fmt(self.default)} {why}.")

    @property
    def required(self) -> bool:
        """Whether the stage cannot compute without a value for the parameter."""
        return self.default is REQUIRED

    def why_not(self, value: ArrayLike) -> str | None:
        """Say why the stage does not accept ``value``, or return None if it does."""
        if np.shape(value) != self.shape:
            return f"has shape {np.shape(value)}, not {self.shape}"
        if self.domain is not None:
            low, high = self.domain
            v = np.asarray(value, dtype=float)
            if not np.all((v >= low) & (v <= high)):
                return f"is outside [{low:g}, {high:g}]"
        return None


def _names(value: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """Return the output names a requirement accepts: None for any name."""
    if value is None:
        return None
    names = (value,) if isinstance(value, str) else tuple(value)
    if not names or not all(isinstance(n, str) for n in names):
        raise TypeError(f"Output names must be strings, got {value!r}.")
    return names


def _describe(kind: type, names: tuple[str, ...] | None) -> str:
    """Describe the outputs that a type and some names match, for messages."""
    what = kind.__qualname__
    if names is None:
        return f"{what} of any name"
    return f"{what} named " + " or ".join(repr(n) for n in names)


@attrs.frozen
class One:
    """A requirement for the single output that matches a type and, optionally, names.

    ``One(Curve)`` is the only ``Curve`` output in the graph, whatever its name: for a
    stage that needs an input of some kind, but not a particular one. ``One(Spectrum,
    name="observed")`` is the ``Spectrum`` named ``"observed"``, received under another
    argument name. In ``requires``, a bare type ``T`` under the key ``k`` is
    ``One(T, name=k)``. :func:`resolve` reports an error unless exactly one output, of
    another stage, matches.

    Parameters
    ----------
    type
        The type of the output; a subclass matches too.
    name
        The name of the output, or several names to accept any one of them; ``None``
        (the default) accepts any name.
    """

    type: type = attrs.field(validator=attrs.validators.instance_of(type))
    name: tuple[str, ...] | None = attrs.field(default=None, converter=_names)

    def matches(self, name: str, kind: type) -> bool:
        """Whether an output with this name and type satisfies the requirement."""
        return issubclass(kind, self.type) and (self.name is None or name in self.name)

    def __str__(self) -> str:
        """Describe the requirement for messages."""
        return f"a {_describe(self.type, self.name)}"


@attrs.frozen
class All:
    """A requirement for every output that matches a type and, optionally, names.

    A stage that requires ``All(Spectrum, name="component")`` receives a tuple of every
    ``Spectrum`` named ``"component"``, in the order the stages were given, without
    naming the stages that provide them: to sum the components of a model, say.
    ``All(Spectrum)`` receives every ``Spectrum``, whatever its name. A stage's own
    outputs never satisfy its requirements.

    Parameters
    ----------
    type
        The type of the outputs; subclasses match too.
    name
        The name of the outputs, or several names to accept any of them; ``None`` (the
        default) accepts any name.
    min
        The fewest matching outputs allowed. The default of one catches a mistyped
        name; zero allows none.
    """

    type: type = attrs.field(validator=attrs.validators.instance_of(type))
    name: tuple[str, ...] | None = attrs.field(default=None, converter=_names)
    min: int = attrs.field(default=1, kw_only=True, validator=attrs.validators.ge(0))

    def matches(self, name: str, kind: type) -> bool:
        """Whether an output with this name and type satisfies the requirement."""
        return issubclass(kind, self.type) and (self.name is None or name in self.name)

    def __str__(self) -> str:
        """Describe the requirement for messages."""
        return f"every {_describe(self.type, self.name)}"


def _as_requirement(key: str, req: One | All | type) -> One | All:
    """Return a requirement as :class:`One` or :class:`All`; a bare type is by name."""
    if isinstance(req, One | All):
        return req
    if isinstance(req, type):
        return One(req, name=key)
    raise TypeError(f"Input {key!r} must be a type, One(...) or All(...), got {req!r}.")


@runtime_checkable
class Stage(Protocol):
    """The interface a stage must provide.

    ``parameters`` maps the name of each model parameter the stage takes to a
    :class:`ParamSpec`. ``compute`` receives every one of them, in its first argument:
    the value from the parameter space, or the default from the spec. It is read from
    the stage instance, so it can be a property that depends on the instance's
    configuration: the number of terms of a polynomial, say, which sets the length of
    its coefficient vector.

    ``provides`` maps the name of each output to its type, and ``compute`` returns a
    dictionary with those keys. ``requires`` maps the name of each input, the keyword
    argument ``compute`` receives it as, to what it needs: a type, for the output of
    that name and type; :class:`One`, for the only output of a type, or of a type and
    other names; or :class:`All`, for a tuple of every matching output.

    A stage may also define ``check(params)``, which raises ``ValueError`` if a set of
    parameter values is inconsistent in a way no single :class:`ParamSpec` can express
    (one parameter bounded by another, say). It must be cheap: :func:`resolve` calls it
    on a few points of the parameter space, without computing anything else.

    A stage may also set ``cacheable = True`` to have a :class:`~yabf.pipeline.Pipeline`
    keep its outputs across evaluations. This suits stages with small outputs, such as
    observables, and not simulators. A cached output must not refer to the stage's
    inputs (a NumPy view of an input field, say), or it keeps them alive.

    A stage may set ``batched = True`` to compute many points in one call, as an
    emulator or a vectorised likelihood can. It is then called with a batch of ``n``
    points: each parameter has a leading axis of length ``n``, each input is a sequence
    of ``n`` values (what the stage would receive at each point), and each output must
    be a sequence of ``n`` values, one per point. A batched stage is also called this
    way, with ``n = 1``, when a single point is evaluated.
    """

    name: str
    parameters: Mapping[str, ParamSpec]
    requires: Mapping[str, One | All | type]
    provides: Mapping[str, type]

    def compute(self, params: Mapping[str, Any], **inputs: object) -> dict[str, Any]:
        """Compute the outputs from parameters and inputs."""
        ...


def _requires(stage: Stage) -> dict[str, One | All]:
    """The requirements of ``stage``, with bare types made into :class:`One`."""
    return {k: _as_requirement(k, v) for k, v in stage.requires.items()}


@attrs.frozen
class Link:
    """An input of one stage, fed by an output of another."""

    stage: str
    input: str
    provider: str
    output: str


class GraphError(ValueError):
    """A stage graph that cannot be resolved; ``problems`` lists every reason."""

    def __init__(self, problems: list[str]) -> None:
        """Collect ``problems`` into one message, one per line."""
        self.problems = problems
        super().__init__(
            "The stage graph is invalid:\n" + "\n".join(f"- {p}" for p in problems)
        )


class StageError(TypeError):
    """A stage whose outputs do not match what it declares."""


@attrs.frozen
class StageGraph:
    """Stages linked by their inputs and outputs, in an order they can be computed in.

    Build one with :func:`resolve`. Printing it shows what each stage takes and
    provides, and where each input comes from.
    """

    stages: tuple[Stage, ...]
    links: tuple[Link, ...]

    def inputs_of(self, name: str) -> dict[str, tuple[tuple[str, str], ...]]:
        """Map each input of stage ``name`` to the outputs that feed it.

        Outputs are ``(stage, output)`` pairs: one for :class:`One` (or a bare type),
        any number for :class:`All`.
        """
        return dict(self._inputs[name])

    def gather(
        self, name: str, outputs: Mapping[tuple[str, str], Any]
    ) -> dict[str, Any]:
        """Collect the inputs of stage ``name`` from ``outputs``, by input name.

        ``outputs`` maps ``(stage, output)`` pairs to computed values. An :class:`All`
        input receives a tuple of its values.
        """
        requires = self._requirements[name]
        return {
            k: tuple(outputs[src] for src in srcs)
            if isinstance(requires[k], All)
            else outputs[srcs[0]]
            for k, srcs in self._inputs[name].items()
        }

    def __str__(self) -> str:
        """Show each stage's inputs, with their sources, and outputs, in order."""
        lines = []
        for s in self.stages:
            lines.append(s.name)
            for k, req in self._requirements[s.name].items():
                sources = ", ".join(f"{p}.{o}" for p, o in self._inputs[s.name][k])
                lines.append(f"  takes     {k}: {req}, from {sources or 'nothing'}")
            for k, kind in s.provides.items():
                lines.append(f"  provides  {k}: {kind.__qualname__}")
        return "\n".join(lines)

    # Lookups computed once per graph: a pipeline uses them on every evaluation.
    @cached_property
    def _requirements(self) -> dict[str, dict[str, One | All]]:
        """The requirements of each stage, by stage name."""
        return {s.name: _requires(s) for s in self.stages}

    @cached_property
    def _inputs(self) -> dict[str, dict[str, tuple[tuple[str, str], ...]]]:
        """The outputs feeding each input, by stage name and input name."""
        inputs = {s.name: dict.fromkeys(s.requires, ()) for s in self.stages}
        for link in self.links:
            inputs[link.stage][link.input] += ((link.provider, link.output),)
        return inputs


def _unused_params(stages: Sequence[Stage], params: ParamSpace) -> list[str]:
    """Return the parameters that reach no stage, directly or through a derived one.

    Only the last parameter of each unused chain is named: if a sampled ``log10_x``
    derives ``x`` and no stage takes ``x``, only ``x`` is.
    """
    taken = set().union(*(s.parameters for s in stages))
    used = set().union(*(params.upstream(n) for n in taken if n in params))
    inputs = {i for p in params.derived for i in p.inputs}
    skip = used | inputs
    return [repr(p.name) for p in params.params if p.name not in skip]


def _edges(params: ParamSpace) -> Iterator[tuple[str, str | None, np.ndarray]]:
    """Yield the points to check exactly, with a label and the parameter moved.

    The points are the fiducial point, where no parameter is moved, and each sampled
    parameter at both edges of its prior, with the others at their fiducial values.
    """
    fiducial = params.fiducial
    yield "at the fiducial point", None, fiducial
    for i, p in enumerate(params.sampled):
        for side, q in (("lower", 0.0), ("upper", 1.0)):
            theta = fiducial.copy()
            theta[i] = p.prior.ppf(q)
            yield f"at the {side} edge of the prior of {p.name!r}", p.name, theta


def _probes(params: ParamSpace) -> Iterator[tuple[str, np.ndarray]]:
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


def _param_problems(stages: Sequence[Stage], params: ParamSpace) -> list[str]:
    """Check the parameter space against what each stage accepts, computing nothing."""
    found: dict[tuple[str, str], str] = {}  # the first problem per stage and parameter
    for s in stages:
        for name, spec in s.parameters.items():
            if name not in params and spec.required:
                found[s.name, name] = (
                    f"Stage {s.name!r} requires the parameter {name!r}, which the "
                    "parameter space does not define."
                )

    # The fiducial point, then each sampled parameter at the very edges of its prior
    # with the others at their fiducial values. For a monotonic function of one
    # sampled parameter (every built-in one), the support of the prior maps onto the
    # interval between the values at its edges, so this check is exact. Other
    # functions are also checked at the points from _probes below, and so is a value
    # that is undefined (NaN) at an infinite edge.
    for where, moved, theta in _edges(params):
        with np.errstate(all="ignore"):
            values = params.to_model(theta)
        for s in stages:
            for name, spec in s.parameters.items():
                if (
                    (s.name, name) in found
                    or name not in params
                    or (moved is not None and moved not in params.upstream(name))
                    or np.any(np.isnan(values[name]))
                ):
                    continue
                if why := spec.why_not(values[name]):
                    found[s.name, name] = (
                        f"Stage {s.name!r} rejects {name!r} {where}: the value "
                        f"{_fmt(values[name])} {why}."
                    )

    for where, theta in _probes(params):
        values = params.to_model(theta)
        for s in stages:
            if any((s.name, name) in found for name in s.parameters):
                continue
            for name, spec in s.parameters.items():
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
                    {k: values.get(k, spec.default) for k, spec in s.parameters.items()}
                )
            except ValueError as e:
                found[s.name, ""] = (
                    f"Stage {s.name!r} rejects the parameters {where}: {e}"
                )
    return list(found.values())


def _link(stages: Sequence[Stage]) -> tuple[list[Link], list[str]]:
    """Link every input to the outputs that match it, and describe the failures."""
    outputs = [(s.name, k, kind) for s in stages for k, kind in s.provides.items()]
    links, problems = [], []
    for s in stages:
        for k, need in _requires(s).items():
            matches = [
                (stage, name)
                for stage, name, kind in outputs
                if stage != s.name and need.matches(name, kind)
            ]
            where = f"Stage {s.name!r} requires {need} (input {k!r})"
            if isinstance(need, All):
                links.extend(Link(s.name, k, *m) for m in matches)
                if len(matches) < need.min:
                    problems.append(
                        f"{where}, at least {need.min}, but "
                        f"{len(matches) or 'no'} output{'s' * (len(matches) != 1)} "
                        "match."
                    )
                continue
            if len(matches) == 1:
                links.append(Link(s.name, k, *matches[0]))
                continue
            if matches:
                found = ", ".join(f"{stage}.{name}" for stage, name in matches)
                kind = need.type.__qualname__
                if need.name is None:
                    fix = f"Name the one to use, with One({kind}, name=...)."
                elif len(need.name) > 1:
                    fix = "Accept fewer names, so that only one output matches."
                else:
                    fix = "Give the outputs distinct names."
                problems.append(f"{where}, but several outputs match: {found}. {fix}")
                continue
            same_type = [
                f"{stage}.{name}"
                for stage, name, kind in outputs
                if stage != s.name and issubclass(kind, need.type)
            ]
            hint = (
                f" (outputs of that type: {', '.join(same_type)})" if same_type else ""
            )
            problems.append(f"{where}, but no stage provides it{hint}.")
    return links, problems


def _declaration_problems(stage: Stage) -> list[str]:
    """Describe what is malformed in the declarations of ``stage``."""
    problems = []
    if not isinstance(stage.parameters, Mapping) or not all(
        isinstance(spec, ParamSpec) for spec in stage.parameters.values()
    ):
        problems.append(
            f"Stage {stage.name!r} must declare `parameters` as a mapping from "
            f"parameter names to ParamSpec, got {stage.parameters!r}."
        )
    if not isinstance(stage.provides, Mapping) or not all(
        isinstance(k, str) and isinstance(kind, type)
        for k, kind in stage.provides.items()
    ):
        problems.append(
            f"Stage {stage.name!r} must declare `provides` as a mapping from output "
            f"names to types, got {stage.provides!r}."
        )
    if not isinstance(stage.requires, Mapping) or not all(
        isinstance(k, str) for k in stage.requires
    ):
        problems.append(
            f"Stage {stage.name!r} must declare `requires` as a mapping from input "
            f"names, got {stage.requires!r}."
        )
        return problems
    for k, req in stage.requires.items():
        try:
            _as_requirement(k, req)
        except TypeError as e:
            problems.append(f"Stage {stage.name!r}: {e}")
    if "params" in stage.requires:
        problems.append(
            f"Stage {stage.name!r} has an input named 'params', which is reserved for "
            "the parameters passed to compute(); rename the input."
        )
    return problems


def resolve(stages: Iterable[Stage], params: ParamSpace | None = None) -> StageGraph:
    """Link stages by their inputs and outputs, and check the graph, computing nothing.

    Raises
    ------
    TypeError
        If a stage is malformed: its ``parameters``, ``requires`` or ``provides`` are
        not what :class:`Stage` describes.
    GraphError
        Listing every problem found: duplicate stage names, an input that no output
        or (for :class:`One`) several outputs match, and cycles. Given ``params``,
        also a parameter that no stage takes (directly or through a tie), a required
        parameter that ``params`` does not define, and a parameter whose values reach
        outside what a stage accepts, either at the edges of its prior or at a few
        points of the parameter space where each stage's ``check`` is also run.
    """
    stages = tuple(stages)
    for s in stages:
        if not isinstance(s, Stage):
            raise TypeError(f"{s!r} is not a stage: see yabf.stages.Stage.")
    malformed = [p for s in stages for p in _declaration_problems(s)]
    if malformed:
        raise TypeError("\n".join(malformed))

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
            problems.append(f"No stage takes the parameters {', '.join(unused)}.")
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
    """Compute ``stage`` and check that it returns exactly the outputs it provides.

    The stage receives every parameter it takes: the value in ``params``, or the
    default from its :class:`ParamSpec`. Other entries of ``params`` are ignored.

    Raises
    ------
    StageError
        If ``params`` lacks a required parameter, or the outputs are not exactly the
        provided ones, or one has the wrong type.
    """
    missing = sorted(
        k for k, spec in stage.parameters.items() if spec.required and k not in params
    )
    if missing:
        raise StageError(
            f"Stage {stage.name!r} requires the parameters {missing}, which were not "
            "given."
        )
    taken = {k: params.get(k, spec.default) for k, spec in stage.parameters.items()}
    out = stage.compute(taken, **inputs)
    provides = stage.provides
    if not isinstance(out, Mapping) or set(out) != set(provides):
        got = sorted(out, key=str) if isinstance(out, Mapping) else type(out).__name__
        raise StageError(
            f"Stage {stage.name!r} returned {got}, but provides {sorted(provides)}."
        )
    for k, kind in provides.items():
        if not isinstance(out[k], kind):
            raise StageError(
                f"Stage {stage.name!r} returned a {type(out[k]).__qualname__} for "
                f"output {k!r}, which is a {kind.__qualname__}."
            )
    return dict(out)


def call_stage_batch(
    stage: Stage,
    params: Mapping[str, ArrayLike],
    inputs: Mapping[str, Sequence[object]],
    n: int,
) -> dict[str, Sequence[object]]:
    """Compute a batched ``stage`` at ``n`` points, checking what it returns.

    ``params`` maps parameter names to arrays with a leading axis of length ``n``, and
    ``inputs`` maps each input to a sequence of ``n`` values. A parameter the stage
    takes but ``params`` lacks has its default, repeated along the batch.

    Raises
    ------
    StageError
        If ``params`` lacks a required parameter, or the outputs are not exactly the
        provided ones, each a sequence of ``n`` values of the output's type.
    """
    missing = sorted(
        k for k, spec in stage.parameters.items() if spec.required and k not in params
    )
    if missing:
        raise StageError(
            f"Stage {stage.name!r} requires the parameters {missing}, which were not "
            "given."
        )
    taken = {
        k: np.asarray(params[k])
        if k in params
        else np.broadcast_to(spec.default, (n, *np.shape(spec.default)))
        for k, spec in stage.parameters.items()
    }
    out = stage.compute(taken, **inputs)
    provides = stage.provides
    if not isinstance(out, Mapping) or set(out) != set(provides):
        got = sorted(out, key=str) if isinstance(out, Mapping) else type(out).__name__
        raise StageError(
            f"Stage {stage.name!r} returned {got}, but provides {sorted(provides)}."
        )
    for k, kind in provides.items():
        values = out[k]
        if not isinstance(values, Sequence | np.ndarray) or len(values) != n:
            raise StageError(
                f"Batched stage {stage.name!r} must return a sequence of {n} values "
                f"for output {k!r}, one per point."
            )
        wrong = [type(v).__qualname__ for v in values if not isinstance(v, kind)]
        if wrong:
            raise StageError(
                f"Batched stage {stage.name!r} returned a {wrong[0]} for output "
                f"{k!r}, which is a {kind.__qualname__}."
            )
    return dict(out)
