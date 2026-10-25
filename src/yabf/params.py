"""Sampled and fixed parameters, and the space they span.

A :class:`Param` is one parameter of an inference: sampled with a prior, fixed, or
tied to other parameters. It feeds one model parameter, possibly through a named
transform, so that the sampler can work in a convenient space: for example, sample
``log10_L_X`` uniformly and feed ``L_X = 10**log10_L_X`` to the model.

A tied parameter is computed from other model parameters by a named relation, such as
``equal`` or ``linear``. Relations are named, like transforms, so that a parameter
space can be written to and read from a configuration file.

A model parameter can also be a vector: each of its elements is fed by its own
parameter, and :func:`param_vector` builds them, ``beta_0``, ``beta_1``, ... for a
vector ``beta``. Parameters can share a joint prior, such as a multivariate normal: each
takes a :class:`~yabf.priors.Marginal` of it as its prior.

A :class:`ParamSpace` is an ordered collection of parameters. It provides what samplers
need, vectorised over any leading batch dimensions: the prior transform from the unit
hypercube, the log prior density, draws from the prior and a Gaussian ball around the
fiducial point for initialising walkers.
"""

from __future__ import annotations

import graphlib
from collections.abc import Callable, Mapping, Sequence
from functools import cached_property

import attrs
import numpy as np

from .priors import JointPrior, Marginal, Prior, as_prior


@attrs.frozen
class Transform:
    """A named, vectorised map from sampled values to model values, with its inverse."""

    name: str
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]


_TRANSFORMS: dict[str, Transform] = {}


def register_transform(
    name: str,
    forward: Callable[[np.ndarray], np.ndarray],
    inverse: Callable[[np.ndarray], np.ndarray],
) -> Transform:
    """Register a transform under ``name``, so that a :class:`Param` can refer to it.

    Both functions must be vectorised, and ``inverse`` must undo ``forward``.
    """
    if name in _TRANSFORMS:
        raise ValueError(f"A transform named '{name}' is already registered.")
    transform = Transform(name=name, forward=forward, inverse=inverse)
    _TRANSFORMS[name] = transform
    return transform


def get_transform(name: str) -> Transform:
    """Return the registered transform called ``name``."""
    try:
        return _TRANSFORMS[name]
    except KeyError:
        raise ValueError(
            f"Unknown transform '{name}'. Registered transforms: {sorted(_TRANSFORMS)}."
        ) from None


def _identity(x):
    return x


def _pow10(x):
    return 10.0**x


register_transform("identity", _identity, _identity)
register_transform("pow10", _pow10, np.log10)
register_transform("log10", np.log10, _pow10)
register_transform("exp", np.exp, np.log)
register_transform("log", np.log, np.exp)


@attrs.frozen
class Relation:
    """A named, vectorised function computing a model parameter from others.

    The function is called with the input values as positional arguments, followed by
    the tie's keyword arguments.
    """

    name: str
    function: Callable[..., np.ndarray]


_RELATIONS: dict[str, Relation] = {}


def register_relation(name: str, function: Callable[..., np.ndarray]) -> Relation:
    """Register a relation under ``name``, so that a :class:`Tie` can refer to it.

    The function must be vectorised: it receives one array per input, all of the same
    shape, and returns an array of that shape.
    """
    if name in _RELATIONS:
        raise ValueError(f"A relation named '{name}' is already registered.")
    relation = Relation(name=name, function=function)
    _RELATIONS[name] = relation
    return relation


def get_relation(name: str) -> Relation:
    """Return the registered relation called ``name``."""
    try:
        return _RELATIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown relation '{name}'. Registered relations: {sorted(_RELATIONS)}."
        ) from None


def _equal(x):
    return x


def _linear(*xs, coefficients, offset=0.0):
    return offset + sum(c * x for c, x in zip(coefficients, xs, strict=True))


register_relation("equal", _equal)
register_relation("linear", _linear)


def _frozen_kwargs(kwargs) -> tuple[tuple[str, object], ...]:
    return tuple(
        (k, tuple(v) if isinstance(v, list) else v)
        for k, v in sorted(dict(kwargs).items())
    )


@attrs.frozen
class Tie:
    """A model parameter computed from other model parameters by a named relation.

    Parameters
    ----------
    relation
        The name of a registered relation.
    inputs
        The model parameters passed to the relation, in order.
    kwargs
        Keyword arguments passed to the relation. They are stored as sorted
        ``(name, value)`` pairs, with lists turned into tuples, so that a tie is
        immutable and hashable.

    Examples
    --------
    ``F_STAR7 = F_STAR10 - 3 * ALPHA_STAR``:

    >>> tie = Tie("linear", ("F_STAR10", "ALPHA_STAR"), {"coefficients": (1, -3)})
    """

    relation: str = attrs.field()
    inputs: tuple[str, ...] = attrs.field(converter=tuple)
    kwargs: tuple[tuple[str, object], ...] = attrs.field(
        factory=tuple, converter=_frozen_kwargs
    )

    @relation.validator
    def _relation_validator(self, att, value):
        get_relation(value)

    def __call__(self, values: Mapping[str, np.ndarray]) -> np.ndarray:
        """Evaluate the relation on the inputs taken from ``values``."""
        args = [values[name] for name in self.inputs]
        return get_relation(self.relation).function(*args, **dict(self.kwargs))


def as_tie(tie) -> Tie:
    """Return ``tie`` as a :class:`Tie`.

    A string ``"x"`` ties the parameter to equal the model parameter ``x``. A mapping
    ``{"x": a, "y": b}`` ties it to ``a * x + b * y``.
    """
    if isinstance(tie, Tie):
        return tie
    if isinstance(tie, str):
        return Tie("equal", (tie,))
    if isinstance(tie, Mapping):
        return Tie("linear", tuple(tie), {"coefficients": tuple(tie.values())})
    raise TypeError(
        "A tie must be a Tie, the name of a model parameter, or a mapping of model "
        f"parameters to coefficients; got {tie!r}."
    )


def _check_transform(inst, att, value):
    get_transform(value)


def _optional_prior(value):
    return None if value is None else as_prior(value)


def _optional_tie(value):
    return None if value is None else as_tie(value)


@attrs.frozen
class Param:
    """A parameter of an inference: sampled with a prior, fixed, or tied.

    Parameters
    ----------
    name
        The name of the parameter, as the sampler sees it.
    prior
        The prior: a ``(low, high)`` tuple for a uniform prior, or any object with
        ``logpdf`` and ``ppf`` methods, such as a frozen scipy distribution. Give
        exactly one of a prior, a fixed value or a tie.
    fixed
        A fixed value. A fixed parameter is not sampled, but still feeds the model.
    tied
        A :class:`Tie` computing this parameter's model value from other model
        parameters; see :func:`as_tie` for the short forms. A tied parameter is not
        sampled and takes no transform.
    fiducial
        A reference value, in the sampled space. Defaults to the fixed value or the
        median of the prior; not defined for a tied parameter.
    feeds
        The model parameter this one determines. Defaults to ``name``.
    index
        If given, this parameter determines element ``index`` of the vector model
        parameter ``feeds``, whose elements are all fed by parameters of the same
        space. A tied parameter cannot be an element.
    transform
        The name of a registered transform applied before the value reaches the model.
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
    tied: Tie | None = attrs.field(default=None, converter=_optional_tie, kw_only=True)
    fiducial: float = attrs.field(converter=float, kw_only=True)
    feeds: str = attrs.field(kw_only=True)
    index: int | None = attrs.field(
        default=None,
        kw_only=True,
        validator=attrs.validators.optional(attrs.validators.ge(0)),
    )
    transform: str = attrs.field(
        default="identity", validator=_check_transform, kw_only=True
    )
    latex: str = attrs.field(kw_only=True)

    @fiducial.default
    def _fiducial_default(self):
        if self.fixed is not None:
            return self.fixed
        if self.prior is None:
            return np.nan  # tied, or rejected in __attrs_post_init__
        return self.prior.ppf(0.5)

    @feeds.default
    def _feeds_default(self):
        return self.name

    @latex.default
    def _latex_default(self):
        return self.name

    def __attrs_post_init__(self):
        """Check that the parameter is fully and consistently specified."""
        given = [x is not None for x in (self.prior, self.fixed, self.tied)]
        if sum(given) != 1:
            raise ValueError(
                f"Parameter '{self.name}' needs exactly one of a prior, a fixed value "
                "or a tie."
            )
        if self.is_tied and self.transform != "identity":
            raise ValueError(
                f"Parameter '{self.name}' is tied, so its value comes from its "
                "relation and it takes no transform."
            )
        if self.is_tied and self.index is not None:
            raise ValueError(
                f"Parameter '{self.name}' is tied, so it cannot be an element of a "
                "vector; tie the whole vector instead."
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
    def is_tied(self) -> bool:
        """Whether the parameter is tied to other parameters."""
        return self.tied is not None

    @property
    def is_sampled(self) -> bool:
        """Whether the parameter is sampled."""
        return self.prior is not None

    def to_model(self, value):
        """Map a value in the sampled space to the value the model receives."""
        return get_transform(self.transform).forward(value)


def _unique(kind: str, values: Sequence[str]):
    seen, dupes = set(), set()
    for v in values:
        (dupes if v in seen else seen).add(v)
    if dupes:
        raise ValueError(f"Duplicate parameter {kind}: {sorted(dupes)}.")


def _check_feeds(params: Sequence[Param]):
    """Check that each model parameter is fed once, or element by element."""
    by_feeds: dict[str, list[Param]] = {}
    for p in params:
        by_feeds.setdefault(p.feeds, []).append(p)
    _unique("feeds", [p.feeds for p in params if p.index is None])
    for feeds, ps in by_feeds.items():
        indices = sorted(p.index for p in ps if p.index is not None)
        if indices and len(indices) < len(ps):
            raise ValueError(
                f"Model parameter '{feeds}' is fed both whole and element by element."
            )
        if indices and indices != list(range(len(ps))):
            raise ValueError(
                f"The elements of model parameter '{feeds}' must have the indices 0 to "
                f"{len(ps) - 1}, each once; got {indices}."
            )


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


def _tie_order(params: Sequence[Param]) -> tuple[Param, ...]:
    """Order the tied parameters so that each comes after the ties it depends on."""
    tied = {p.feeds: p for p in params if p.is_tied}
    known = {p.feeds for p in params}
    graph = {}
    for feeds, p in tied.items():
        missing = sorted(set(p.tied.inputs) - known)
        if missing:
            raise ValueError(
                f"Parameter '{p.name}' is tied to {missing}, which no parameter feeds."
            )
        graph[feeds] = {name for name in p.tied.inputs if name in tied}
    try:
        order = graphlib.TopologicalSorter(graph).static_order()
        return tuple(tied[feeds] for feeds in order)
    except graphlib.CycleError as e:
        raise ValueError(f"Parameter ties form a cycle: {e.args[1]}.") from None


@attrs.frozen
class ParamSpace:
    """An ordered collection of parameters.

    The sampled parameters, in order, define the coordinates the sampler sees. All
    methods taking ``theta`` or ``u`` accept arrays of shape ``(..., ndim)`` and are
    vectorised over the leading dimensions.

    A tie's inputs must be model parameters fed by parameters in the space, and ties
    may depend on other ties as long as they do not form a cycle. The elements of a
    vector model parameter must all be in the space, and so must every parameter of a
    joint prior, sampled.
    """

    params: tuple[Param, ...] = attrs.field(converter=tuple)

    @params.validator
    def _params_validator(self, att, value):
        for p in value:
            if not isinstance(p, Param):
                raise TypeError(f"Expected Param instances, got {p!r}.")
        _unique("names", [p.name for p in value])
        _check_feeds(value)
        _tie_order(value)
        _joint_groups([p for p in value if p.is_sampled])

    @cached_property
    def _joint(self) -> list[tuple[JointPrior, list[int]]]:
        return _joint_groups(self.sampled)

    @cached_property
    def _independent(self) -> list[int]:
        return [
            i for i, p in enumerate(self.sampled) if not isinstance(p.prior, Marginal)
        ]

    @property
    def sampled(self) -> tuple[Param, ...]:
        """The sampled parameters, in order."""
        return tuple(p for p in self.params if p.is_sampled)

    @property
    def fixed(self) -> tuple[Param, ...]:
        """The fixed parameters."""
        return tuple(p for p in self.params if p.is_fixed)

    @property
    def tied(self) -> tuple[Param, ...]:
        """The tied parameters."""
        return tuple(p for p in self.params if p.is_tied)

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

    def _check_shape(self, x, what: str) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[-1:] != (self.ndim,):
            raise ValueError(
                f"{what} must have shape (..., {self.ndim}), got {x.shape}."
            )
        return x

    def prior_transform(self, u) -> np.ndarray:
        """Map points of the unit hypercube to the sampled parameters."""
        u = self._check_shape(u, "u")
        out = np.empty(u.shape)
        for i in self._independent:
            out[..., i] = self.sampled[i].prior.ppf(u[..., i])
        for joint, cols in self._joint:
            out[..., cols] = joint.transform(u[..., cols])
        return out

    def log_prior(self, theta) -> np.ndarray | float:
        """The log prior density of the sampled parameters (``-inf`` outside it)."""
        theta = self._check_shape(theta, "theta")
        total = np.zeros(theta.shape[:-1])
        for i in self._independent:
            total = total + self.sampled[i].prior.logpdf(theta[..., i])
        for joint, cols in self._joint:
            total = total + joint.logpdf(theta[..., cols])
        return total

    def to_model(self, theta) -> dict[str, np.ndarray]:
        """Return the model parameters for ``theta``, keyed by the names they feed.

        Sampled values are transformed, fixed values are broadcast to the batch shape
        of ``theta``, the elements of each vector are stacked along a last axis, and
        tied values are computed from the others.
        """
        theta = self._check_shape(theta, "theta")
        out, elements = {}, {}

        def put(p, value):
            if p.index is None:
                out[p.feeds] = value
            else:
                elements.setdefault(p.feeds, {})[p.index] = value

        for i, p in enumerate(self.sampled):
            put(p, p.to_model(theta[..., i]))
        for p in self.fixed:
            put(p, np.broadcast_to(p.to_model(p.fixed), theta.shape[:-1]))
        for feeds, values in elements.items():
            out[feeds] = np.stack([values[i] for i in range(len(values))], axis=-1)
        for p in _tie_order(self.params):
            out[p.feeds] = p.tied(out)
        return out

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points from the prior, shape ``(n, ndim)``."""
        return self.prior_transform(rng.uniform(size=(n, self.ndim)))

    def gaussian_ball(
        self,
        n: int,
        rng: np.random.Generator,
        scale: float = 0.01,
        center=None,
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


def _element(value, i: int):
    """Return element ``i`` of a per-element sequence, or ``value`` if it is shared."""
    return value[i] if np.ndim(value) == 1 else value


def param_vector(
    name: str,
    length: int | None = None,
    *,
    prior=None,
    fixed=None,
    fiducial=None,
    feeds: str | None = None,
    transform: str = "identity",
    latex: str | None = None,
) -> tuple[Param, ...]:
    """Build the parameters that feed the elements of a vector model parameter.

    Element ``i`` is the parameter ``f"{name}_{i}"``, feeding element ``i`` of the
    model parameter ``feeds`` (default: ``name``).

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
    feeds, transform, latex
        As for :class:`Param`; ``latex`` gets the element index as a subscript.

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
        kw = {
            "feeds": feeds or name,
            "index": i,
            "transform": transform,
            "latex": f"{latex or name}_{{{i}}}",
        }
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
    return tuple(params)
