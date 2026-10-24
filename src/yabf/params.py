"""Sampled and fixed parameters, and the space they span.

A :class:`Param` is one parameter of an inference: either sampled, with a prior, or
fixed. It feeds one model parameter, possibly through a named transform, so that the
sampler can work in a convenient space: for example, sample ``log10_L_X`` uniformly
and feed ``L_X = 10**log10_L_X`` to the model.

A :class:`ParamSpace` is an ordered collection of parameters. It provides what samplers
need, vectorised over any leading batch dimensions: the prior transform from the unit
hypercube, the log prior density, draws from the prior and a Gaussian ball around the
fiducial point for initialising walkers.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import attrs
import numpy as np

from .priors import Prior, as_prior


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


def _check_transform(inst, att, value):
    get_transform(value)


def _optional_prior(value):
    return None if value is None else as_prior(value)


@attrs.frozen
class Param:
    """A parameter of an inference: sampled with a prior, or fixed.

    Parameters
    ----------
    name
        The name of the parameter, as the sampler sees it.
    prior
        The prior: a ``(low, high)`` tuple for a uniform prior, or any object with
        ``logpdf`` and ``ppf`` methods, such as a frozen scipy distribution. Give
        either a prior or a fixed value.
    fixed
        A fixed value. A fixed parameter is not sampled, but still feeds the model.
    fiducial
        A reference value, in the sampled space. Defaults to the fixed value or the
        median of the prior.
    feeds
        The model parameter this one determines. Defaults to ``name``.
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
    fiducial: float = attrs.field(converter=float, kw_only=True)
    feeds: str = attrs.field(kw_only=True)
    transform: str = attrs.field(
        default="identity", validator=_check_transform, kw_only=True
    )
    latex: str = attrs.field(kw_only=True)

    @fiducial.default
    def _fiducial_default(self):
        if self.fixed is not None:
            return self.fixed
        if self.prior is None:
            return np.nan  # rejected in __attrs_post_init__ with a clear message
        return self.prior.ppf(0.5)

    @feeds.default
    def _feeds_default(self):
        return self.name

    @latex.default
    def _latex_default(self):
        return self.name

    def __attrs_post_init__(self):
        """Check that the parameter is fully and consistently specified."""
        if (self.prior is None) == (self.fixed is None):
            raise ValueError(
                f"Parameter '{self.name}' needs exactly one of a prior or a fixed value."
            )
        if self.prior is not None and not np.isfinite(self.prior.logpdf(self.fiducial)):
            raise ValueError(
                f"The fiducial value of '{self.name}' ({self.fiducial}) is outside the "
                "support of its prior."
            )

    @property
    def is_fixed(self) -> bool:
        """Whether the parameter is fixed rather than sampled."""
        return self.fixed is not None

    def to_model(self, value):
        """Map a value in the sampled space to the value the model receives."""
        return get_transform(self.transform).forward(value)


def _unique(kind: str, values: Sequence[str]):
    seen, dupes = set(), set()
    for v in values:
        (dupes if v in seen else seen).add(v)
    if dupes:
        raise ValueError(f"Duplicate parameter {kind}: {sorted(dupes)}.")


@attrs.frozen
class ParamSpace:
    """An ordered collection of parameters.

    The sampled parameters, in order, define the coordinates the sampler sees. All
    methods taking ``theta`` or ``u`` accept arrays of shape ``(..., ndim)`` and are
    vectorised over the leading dimensions.
    """

    params: tuple[Param, ...] = attrs.field(converter=tuple)

    @params.validator
    def _params_validator(self, att, value):
        for p in value:
            if not isinstance(p, Param):
                raise TypeError(f"Expected Param instances, got {p!r}.")
        _unique("names", [p.name for p in value])
        _unique("feeds", [p.feeds for p in value])

    @property
    def sampled(self) -> tuple[Param, ...]:
        """The sampled parameters, in order."""
        return tuple(p for p in self.params if not p.is_fixed)

    @property
    def fixed(self) -> tuple[Param, ...]:
        """The fixed parameters."""
        return tuple(p for p in self.params if p.is_fixed)

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
        return np.stack(
            [p.prior.ppf(u[..., i]) for i, p in enumerate(self.sampled)], axis=-1
        )

    def log_prior(self, theta) -> np.ndarray | float:
        """The log prior density of the sampled parameters (``-inf`` outside it)."""
        theta = self._check_shape(theta, "theta")
        return sum(p.prior.logpdf(theta[..., i]) for i, p in enumerate(self.sampled))

    def to_model(self, theta) -> dict[str, np.ndarray]:
        """The model parameters for ``theta``: transformed sampled values and fixed ones.

        Fixed values are broadcast to the batch shape of ``theta``.
        """
        theta = self._check_shape(theta, "theta")
        out = {p.feeds: p.to_model(theta[..., i]) for i, p in enumerate(self.sampled)}
        for p in self.fixed:
            out[p.feeds] = np.broadcast_to(p.to_model(p.fixed), theta.shape[:-1])
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

        The ball is centred on ``center`` (default: the fiducial point). Its width in
        each parameter is ``scale`` times that parameter's prior width, measured as half
        the central 68% interval. Points outside the prior are redrawn.
        """
        center = self.fiducial if center is None else self._check_shape(center, "center")
        if not np.isfinite(self.log_prior(center)):
            raise ValueError("The centre of the Gaussian ball is outside the prior.")
        width = np.array(
            [(p.prior.ppf(0.8413447) - p.prior.ppf(0.1586553)) / 2 for p in self.sampled]
        )
        out = np.empty((n, self.ndim))
        todo = np.arange(n)
        for _ in range(max_tries):
            out[todo] = center + scale * width * rng.standard_normal((len(todo), self.ndim))
            todo = todo[~np.isfinite(self.log_prior(out[todo]))]
            if not len(todo):
                return out
        raise RuntimeError(
            f"Could not draw {n} points inside the prior after {max_tries} tries; "
            "the ball is too wide for the prior or its centre is on the prior's edge."
        )
