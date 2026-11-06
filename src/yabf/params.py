"""Sampled, fixed and derived parameters, and the space they span.

A :class:`Param` is one parameter of an inference, sampled with a prior or fixed to a
value. A :class:`DerivedParam` is computed from other parameters by a function. It
lets the sampler work in a convenient space: for example, sample ``log10_L_X``
uniformly and derive ``L_X = 10**log10_L_X`` for the model. It also ties one parameter
to others, as in ``F_STAR7_MINI = F_STAR10 - 3 * ALPHA_STAR``.

A derived parameter can refer to its function by name, as a configuration file would,
once the function is registered with :func:`register_relation`. The functions
``identity``, ``pow10``, ``log10``, ``exp``, ``log`` and ``stack`` are registered
already.

A vector parameter is derived from its elements by ``stack``: :func:`param_vector`
builds the elements ``beta_0``, ``beta_1``, ... and the vector ``beta`` made of them.
Parameters can share a joint prior, such as a multivariate normal: each takes a
:class:`~yabf.priors.Marginal` of it as its prior.

A :class:`ParamSpace` is an ordered collection of parameters. It provides what samplers
need, vectorised over any leading batch dimensions: the prior transform from the unit
hypercube, the log prior density, draws from the prior and a Gaussian ball around the
fiducial point for initialising walkers. It also computes the value of every parameter,
derived ones included, at a point of the sampled space.
"""

from __future__ import annotations

import graphlib
import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cached_property

import attrs
import numpy as np
from numpy.typing import ArrayLike

from .priors import JointPrior, Marginal, Prior, as_prior

_RELATIONS: dict[str, Callable[..., np.ndarray]] = {}


def _origin(function: Callable) -> tuple[str | None, str | None]:
    """Return where ``function`` is defined: its module and qualified name."""
    return getattr(function, "__module__", None), getattr(
        function, "__qualname__", None
    )


def register_relation(
    function: Callable[..., np.ndarray] | None = None, /, *, name: str | None = None
) -> Callable:
    """Register a function, so that a :class:`DerivedParam` can refer to it by name.

    Use it as a decorator: ``@register_relation`` registers the function under its own
    name, and ``@register_relation(name="...")`` under the name given. The function is
    returned unchanged, so it can still be called directly.

    The function must be vectorised. It receives one array per input, with the same
    leading (batch) dimensions, and returns an array with those leading dimensions.

    A name can be registered once. Registering the same function again, as when a
    notebook cell is run twice, replaces it.

    Examples
    --------
    >>> @register_relation
    ... def radius(x, y):
    ...     return np.hypot(x, y)
    """

    def register(f: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        """Register ``f`` and return it."""
        key = f.__name__ if name is None else name
        if key in _RELATIONS and _origin(_RELATIONS[key]) != _origin(f):
            raise ValueError(f"A relation named '{key}' is already registered.")
        _RELATIONS[key] = f
        return f

    return register if function is None else register(function)


def get_relation(name: str) -> Callable[..., np.ndarray]:
    """Return the function registered as ``name``."""
    try:
        return _RELATIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown relation '{name}'. Registered relations: {sorted(_RELATIONS)}."
        ) from None


def _identity(x: np.ndarray) -> np.ndarray:
    """Return ``x`` unchanged."""
    return x


def _pow10(x: np.ndarray) -> np.ndarray:
    """Return ``10**x``."""
    return 10.0**x


def _stack(*elements: np.ndarray) -> np.ndarray:
    """Stack the elements of a vector along a new last axis."""
    return np.stack(np.broadcast_arrays(*elements), axis=-1)


register_relation(_identity, name="identity")
register_relation(_pow10, name="pow10")
register_relation(np.log10, name="log10")
register_relation(np.exp, name="exp")
register_relation(np.log, name="log")
register_relation(_stack, name="stack")


def _resolve(function: Callable[..., np.ndarray] | str) -> Callable[..., np.ndarray]:
    """Return ``function``, looking it up if it is the name of a registered one."""
    return get_relation(function) if isinstance(function, str) else function


def _check_function(
    inst: DerivedParam, att: attrs.Attribute, value: Callable | str
) -> None:
    """Check that ``function`` is callable, or names a registered relation."""
    if not callable(_resolve(value)):
        raise TypeError(
            f"The function of '{inst.name}' must be callable, or the name of a "
            f"registered relation; got {value!r}."
        )


def _signature(function: Callable) -> inspect.Signature | None:
    """Return the signature of ``function``, or None if it cannot be read."""
    try:
        return inspect.signature(function)
    except (TypeError, ValueError):
        return None


def _input_names(
    inputs: str | Param | DerivedParam | Iterable[str | Param | DerivedParam],
) -> tuple[str, ...]:
    """Return the names of ``inputs``: parameters, names, or a sequence of either."""
    if isinstance(inputs, str | Param | DerivedParam):
        inputs = (inputs,)
    return tuple(p if isinstance(p, str) else p.name for p in inputs)


@attrs.frozen
class DerivedParam:
    """A parameter computed from other parameters by a function.

    A derived parameter is not sampled, and has no prior of its own: its distribution
    follows from the priors of the parameters it is computed from.

    Parameters
    ----------
    name
        The name of the parameter.
    function
        A vectorised function, or the name of one registered with
        :func:`register_relation`. It is called with the values of ``inputs``, in order.
    inputs
        The parameters passed to ``function``: their names, or the parameters
        themselves. Each may be sampled, fixed or derived. Defaults to the names of the
        function's required arguments.
    latex
        A LaTeX label for plots. Defaults to ``name``.

    Examples
    --------
    Sample the logarithm of an amplitude, and give the model the amplitude itself:

    >>> log_amp = Param("log10_amp", prior=(2.0, 4.0))
    >>> amp = DerivedParam("amp", "pow10", log_amp)

    Derive a parameter from two others, taking the inputs from the function's
    arguments:

    >>> def radius(x, y):
    ...     return np.hypot(x, y)
    >>> r = DerivedParam("r", radius)
    >>> r.inputs
    ('x', 'y')

    Give the inputs when their names are not the function's arguments:

    >>> def f_star7_mini(f_star10, alpha_star):
    ...     return f_star10 - 3 * alpha_star
    >>> f7 = DerivedParam("F_STAR7_MINI", f_star7_mini, ["F_STAR10", "ALPHA_STAR"])
    """

    name: str = attrs.field(validator=attrs.validators.instance_of(str))
    function: Callable[..., np.ndarray] | str = attrs.field(validator=_check_function)
    inputs: tuple[str, ...] = attrs.field(converter=_input_names)
    latex: str = attrs.field(kw_only=True)

    @inputs.default
    def _inputs_default(self) -> tuple[str, ...]:
        """Default the inputs to the names of the function's required arguments."""
        sig = _signature(self.func)
        kinds = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        if sig is None or any(
            a.kind is inspect.Parameter.VAR_POSITIONAL for a in sig.parameters.values()
        ):
            raise TypeError(
                f"Give the inputs of '{self.name}': they cannot be read from the "
                f"arguments of {self.function!r}."
            )
        return tuple(
            a.name
            for a in sig.parameters.values()
            if a.kind in kinds and a.default is inspect.Parameter.empty
        )

    @latex.default
    def _latex_default(self) -> str:
        """Default the LaTeX label to the parameter's name."""
        return self.name

    def __attrs_post_init__(self) -> None:
        """Check that the function can take the inputs."""
        sig = _signature(self.func)
        if sig is None:
            return
        try:
            sig.bind(*self.inputs)
        except TypeError:
            raise TypeError(
                f"Parameter '{self.name}' passes {len(self.inputs)} input(s), "
                f"{list(self.inputs)}, to {self.function!r}, which does not accept "
                "them."
            ) from None

    @property
    def func(self) -> Callable[..., np.ndarray]:
        """The function computing the parameter."""
        return _resolve(self.function)

    def __call__(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        """Compute the parameter from the values of its inputs, taken by name."""
        return self.func(*(values[name] for name in self.inputs))


def _optional_prior(value: Prior | tuple[float, float] | None) -> Prior | None:
    """Convert ``value`` with :func:`~yabf.priors.as_prior`, keeping ``None``."""
    return None if value is None else as_prior(value)


@attrs.frozen
class Param:
    """A parameter of an inference, sampled with a prior or fixed to a value.

    Parameters
    ----------
    name
        The name of the parameter.
    prior
        The prior: a ``(low, high)`` tuple for a uniform prior, or any object with
        ``logpdf`` and ``ppf`` methods, such as a frozen scipy distribution. Give
        exactly one of a prior or a fixed value.
    fixed
        A fixed value. A fixed parameter is not sampled, but its value still reaches the
        model.
    fiducial
        A reference value. Defaults to the fixed value or the median of the prior.
    latex
        A LaTeX label for plots. Defaults to ``name``.
    """

    name: str = attrs.field(validator=attrs.validators.instance_of(str))
    prior: Prior | None = attrs.field(
        default=None, converter=_optional_prior, kw_only=True
    )
    fixed: float | None = attrs.field(
        default=None, converter=attrs.converters.optional(float), kw_only=True
    )
    fiducial: float = attrs.field(converter=float, kw_only=True)
    latex: str = attrs.field(kw_only=True)

    @fiducial.default
    def _fiducial_default(self) -> float:
        """Default the fiducial value to the fixed value or the prior median."""
        if self.fixed is not None:
            return self.fixed
        if self.prior is None:
            return np.nan  # rejected in __attrs_post_init__
        return self.prior.ppf(0.5)

    @latex.default
    def _latex_default(self) -> str:
        """Default the LaTeX label to the parameter's name."""
        return self.name

    def __attrs_post_init__(self) -> None:
        """Check that the parameter is fully and consistently specified."""
        if (self.prior is None) == (self.fixed is None):
            raise ValueError(
                f"Parameter '{self.name}' needs exactly one of a prior or a fixed "
                "value."
            )
        if self.prior is not None and not self.prior.logpdf(self.fiducial) > -np.inf:
            raise ValueError(
                f"The fiducial value of '{self.name}' ({self.fiducial}) is outside the "
                "support of its prior."
            )

    @property
    def is_fixed(self) -> bool:
        """Whether the parameter is fixed rather than sampled."""
        return self.fixed is not None

    @property
    def is_sampled(self) -> bool:
        """Whether the parameter is sampled."""
        return self.prior is not None


def _unique(kind: str, values: Sequence[str]) -> None:
    """Raise if any of ``values`` (parameter ``kind``, for the message) repeats."""
    seen, dupes = set(), set()
    for v in values:
        (dupes if v in seen else seen).add(v)
    if dupes:
        raise ValueError(f"Duplicate parameter {kind}: {sorted(dupes)}.")


def _joint_groups(sampled: Sequence[Param]) -> list[tuple[JointPrior, list[int]]]:
    """Group the sampled parameters that share a joint prior.

    Returns each joint prior with the positions of its parameters among ``sampled``,
    in the order of the joint prior's own parameters.
    """
    groups: dict[int, tuple[JointPrior, dict[int, int]]] = {}
    for col, p in enumerate(sampled):
        if isinstance(p.prior, Marginal):
            joint, cols = groups.setdefault(id(p.prior.joint), (p.prior.joint, {}))
            if p.prior.index in cols:
                raise ValueError(
                    f"Two parameters take marginal {p.prior.index} of the same joint "
                    f"prior, including '{p.name}'."
                )
            cols[p.prior.index] = col
    out = []
    for joint, cols in groups.values():
        if sorted(cols) != list(range(joint.dim)):
            names = [sampled[c].name for c in cols.values()]
            raise ValueError(
                f"The joint prior of {names} covers {joint.dim} parameters, so all of "
                f"them must be sampled; only its marginals {sorted(cols)} are."
            )
        out.append((joint, [cols[i] for i in range(joint.dim)]))
    return out


def _derivation_order(
    params: Sequence[Param | DerivedParam],
) -> tuple[DerivedParam, ...]:
    """Order the derived parameters, each after the parameters it is derived from."""
    derived = {p.name: p for p in params if isinstance(p, DerivedParam)}
    known = {p.name for p in params}
    graph = {}
    for name, p in derived.items():
        missing = [i for i in p.inputs if i not in known]
        if missing:
            raise ValueError(
                f"Parameter '{name}' is derived from {missing}, which the parameter "
                "space does not define."
            )
        graph[name] = {i for i in p.inputs if i in derived}
    try:
        return tuple(
            derived[n] for n in graphlib.TopologicalSorter(graph).static_order()
        )
    except graphlib.CycleError as e:
        raise ValueError(f"Derived parameters form a cycle: {e.args[1]}.") from None


@attrs.frozen
class ParamSpace:
    """An ordered collection of parameters.

    The sampled parameters, in order, define the coordinates the sampler sees. All
    methods taking ``theta`` or ``u`` accept arrays of shape ``(..., ndim)`` and are
    vectorised over the leading dimensions.

    A derived parameter's inputs must be parameters of the space, and derived
    parameters may depend on other derived parameters as long as they do not form a
    cycle. Every parameter of a joint prior must be in the space, and sampled.
    """

    params: tuple[Param | DerivedParam, ...] = attrs.field(converter=tuple)

    @params.validator
    def _params_validator(
        self, att: attrs.Attribute, value: tuple[Param | DerivedParam, ...]
    ) -> None:
        """Check that the parameters form a consistent space."""
        for p in value:
            if not isinstance(p, Param | DerivedParam):
                raise TypeError(f"Expected Param or DerivedParam instances, got {p!r}.")
        _unique("names", [p.name for p in value])
        _derivation_order(value)
        _joint_groups([p for p in value if isinstance(p, Param) and p.is_sampled])

    @cached_property
    def _joint(self) -> list[tuple[JointPrior, list[int]]]:
        """Each joint prior, with the columns of its parameters among the sampled."""
        return _joint_groups(self.sampled)

    @cached_property
    def _independent(self) -> list[int]:
        """The columns of the sampled parameters that have a prior of their own."""
        return [
            i for i, p in enumerate(self.sampled) if not isinstance(p.prior, Marginal)
        ]

    @cached_property
    def _order(self) -> tuple[DerivedParam, ...]:
        """The derived parameters, each after those it is derived from."""
        return _derivation_order(self.params)

    @cached_property
    def _by_name(self) -> dict[str, Param | DerivedParam]:
        """Every parameter, by name."""
        return {p.name: p for p in self.params}

    def __contains__(self, name: object) -> bool:
        """Whether the space has a parameter called ``name``."""
        return name in self._by_name

    @property
    def sampled(self) -> tuple[Param, ...]:
        """The sampled parameters, in order."""
        return tuple(p for p in self.params if isinstance(p, Param) and p.is_sampled)

    @property
    def fixed(self) -> tuple[Param, ...]:
        """The fixed parameters."""
        return tuple(p for p in self.params if isinstance(p, Param) and p.is_fixed)

    @property
    def derived(self) -> tuple[DerivedParam, ...]:
        """The derived parameters."""
        return tuple(p for p in self.params if isinstance(p, DerivedParam))

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the sampled parameters, in order."""
        return tuple(p.name for p in self.sampled)

    @property
    def ndim(self) -> int:
        """The number of sampled parameters."""
        return len(self.sampled)

    @property
    def fiducial(self) -> np.ndarray:
        """The fiducial values of the sampled parameters."""
        return np.array([p.fiducial for p in self.sampled])

    def upstream(self, name: str) -> frozenset[str]:
        """The parameter ``name`` and all the parameters it is derived from."""
        out, todo = set(), [name]
        while todo:
            n = todo.pop()
            if n not in out:
                out.add(n)
                p = self._by_name[n]
                if isinstance(p, DerivedParam):
                    todo.extend(p.inputs)
        return frozenset(out)

    def _check_shape(self, x: ArrayLike, what: str) -> np.ndarray:
        """Return ``x`` as a float array, checking its last axis has length ``ndim``."""
        x = np.asarray(x, dtype=float)
        if x.shape[-1:] != (self.ndim,):
            raise ValueError(
                f"{what} must have shape (..., {self.ndim}), got {x.shape}."
            )
        return x

    def prior_transform(self, u: ArrayLike) -> np.ndarray:
        """Map points of the unit hypercube to the sampled parameters."""
        u = self._check_shape(u, "u")
        out = np.empty(u.shape)
        for i in self._independent:
            out[..., i] = self.sampled[i].prior.ppf(u[..., i])
        for joint, cols in self._joint:
            out[..., cols] = joint.transform(u[..., cols])
        return out

    def log_prior(self, theta: ArrayLike) -> np.ndarray:
        """The log prior density of the sampled parameters (``-inf`` outside it)."""
        theta = self._check_shape(theta, "theta")
        total = np.zeros(theta.shape[:-1])
        for i in self._independent:
            total = total + self.sampled[i].prior.logpdf(theta[..., i])
        for joint, cols in self._joint:
            total = total + joint.logpdf(theta[..., cols])
        return total

    def to_model(self, theta: ArrayLike) -> dict[str, np.ndarray]:
        """Return the value of every parameter at ``theta``, by name.

        Sampled values are taken from ``theta``, fixed values are broadcast to its batch
        shape, and derived values are computed from the others. A model takes the values
        it needs from these by name.
        """
        theta = self._check_shape(theta, "theta")
        out = {p.name: theta[..., i] for i, p in enumerate(self.sampled)}
        for p in self.fixed:
            out[p.name] = np.broadcast_to(p.fixed, theta.shape[:-1])
        for p in self._order:
            out[p.name] = np.asarray(p(out))
        return out

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points from the prior, shape ``(n, ndim)``."""
        return self.prior_transform(rng.uniform(size=(n, self.ndim)))

    def gaussian_ball(
        self,
        n: int,
        rng: np.random.Generator,
        scale: float = 0.01,
        center: ArrayLike | None = None,
        max_tries: int = 100,
    ) -> np.ndarray:
        """Draw ``n`` points from a Gaussian ball inside the prior, shape ``(n, ndim)``.

        The ball is centred on ``center``, of shape ``(ndim,)`` (default: the fiducial
        point). Its width in each parameter is ``scale`` times that parameter's prior
        width, measured as half the central 68% interval. Points outside the prior are
        redrawn.
        """
        center = self.fiducial if center is None else np.asarray(center, dtype=float)
        if center.shape != (self.ndim,):
            raise ValueError(
                f"center must have shape ({self.ndim},), got {center.shape}."
            )
        if not self.log_prior(center) > -np.inf:
            raise ValueError("The centre of the Gaussian ball is outside the prior.")
        width = np.array(
            [
                (p.prior.ppf(0.8413447) - p.prior.ppf(0.1586553)) / 2
                for p in self.sampled
            ]
        )
        out = np.empty((n, self.ndim))
        todo = np.arange(n)
        for _ in range(max_tries):
            out[todo] = center + scale * width * rng.standard_normal(
                (len(todo), self.ndim)
            )
            todo = todo[~(self.log_prior(out[todo]) > -np.inf)]
            if not len(todo):
                return out
        raise RuntimeError(
            f"Could not draw {n} points inside the prior after {max_tries} tries; "
            "the ball is too wide for the prior or its centre is on the prior's edge."
        )


def _element(value: object, i: int) -> object:
    """Return element ``i`` of a per-element sequence, or ``value`` if it is shared."""
    return value[i] if np.ndim(value) == 1 else value


def param_vector(
    name: str,
    length: int | None = None,
    *,
    prior: Prior
    | tuple[float, float]
    | JointPrior
    | Sequence[Prior | tuple[float, float]]
    | None = None,
    fixed: Mapping[int, float] | Sequence[float] | None = None,
    fiducial: float | Sequence[float] | None = None,
    latex: str | None = None,
) -> tuple[Param | DerivedParam, ...]:
    """Build a vector parameter and the parameters of its elements.

    Element ``i`` is the parameter ``f"{name}_{i}"``, and the vector is the derived
    parameter ``name``, which stacks the elements along a last axis. The elements come
    first in the returned tuple, and the vector last.

    To sample a vector in a different space, derive a new vector from this one: for
    example, ``DerivedParam("beta", "pow10", "log10_beta")`` after
    ``param_vector("log10_beta", ...)``.

    Parameters
    ----------
    name
        The name of the vector.
    length
        The number of elements. It may be left out if another argument gives it.
    prior
        The prior of the sampled elements: one prior (or ``(low, high)`` tuple) for
        every element, a list with one per element, or a
        :class:`~yabf.priors.JointPrior` over all of them, such as a multivariate
        normal.
    fixed
        Fixed elements: a mapping from element index to value, the other elements
        being sampled, or a sequence fixing every element.
    fiducial
        The fiducial value of the sampled elements: one value, or one per element.
    latex
        A LaTeX label for the vector (default: ``name``). The elements get it with
        their index as a subscript.

    Examples
    --------
    A five-term polynomial whose second term is fixed at zero:

    >>> params = param_vector("p", 5, prior=(-1e4, 1e4), fixed={1: 0.0})
    """
    lengths = set()
    if length is not None:
        lengths.add(length)
    if isinstance(prior, JointPrior):
        lengths.add(prior.dim)
    if isinstance(prior, list):
        lengths.add(len(prior))
    if fixed is not None and not isinstance(fixed, Mapping):
        lengths.add(len(fixed))
    if np.ndim(fiducial) == 1:
        lengths.add(len(fiducial))
    if len(lengths) != 1:
        raise ValueError(
            f"The length of vector '{name}' must be given, or follow from its other "
            f"arguments, and be consistent; got {sorted(lengths) or 'none'}."
        )
    (n,) = lengths

    fixed = (
        {}
        if fixed is None
        else dict(fixed if isinstance(fixed, Mapping) else enumerate(fixed))
    )
    if isinstance(prior, JointPrior) and fixed:
        raise ValueError(
            f"A joint prior needs every element of '{name}' sampled; fix none of them."
        )

    params = []
    for i in range(n):
        kw = {"latex": f"{latex or name}_{{{i}}}"}
        if i in fixed:
            params.append(Param(f"{name}_{i}", fixed=fixed[i], **kw))
            continue
        if prior is None:
            raise ValueError(f"Element {i} of '{name}' needs a prior or a fixed value.")
        if isinstance(prior, JointPrior):
            element_prior = Marginal(prior, i)
        elif isinstance(prior, list):
            element_prior = prior[i]
        else:
            element_prior = prior
        if fiducial is not None:
            kw["fiducial"] = _element(fiducial, i)
        params.append(Param(f"{name}_{i}", prior=element_prior, **kw))
    vector = DerivedParam(name, "stack", params, latex=latex or name)
    return (*params, vector)
