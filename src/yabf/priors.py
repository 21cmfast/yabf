"""Priors for sampled parameters.

A prior is any object with vectorised ``logpdf(x)`` and ``ppf(u)`` methods: the log
probability density, and the inverse cumulative distribution function that maps the
unit interval onto the parameter. Frozen :mod:`scipy.stats` distributions satisfy this
as they are, so any of them can be used directly. The functions here are short names
for the common cases, and :func:`as_prior` turns a ``(low, high)`` tuple into a
uniform prior.

A :class:`JointPrior` is a prior over several parameters at once, such as a
multivariate normal with a covariance matrix. Each of its parameters takes a
:class:`Marginal` of it as its prior.
"""

from __future__ import annotations

from functools import cached_property
from typing import Protocol, runtime_checkable

import attrs
import numpy as np
from scipy import linalg, stats


@runtime_checkable
class Prior(Protocol):
    """The interface a prior must provide."""

    def logpdf(self, x):
        """Return the log probability density at ``x``."""
        ...

    def ppf(self, q):
        """Return the parameter value below which a fraction ``q`` of the prior lies."""
        ...


def uniform(low: float, high: float) -> Prior:
    """A uniform prior on ``[low, high]``."""
    if not high > low:
        raise ValueError(f"A uniform prior needs high > low, got ({low}, {high}).")
    return stats.uniform(loc=low, scale=high - low)


def log_uniform(low: float, high: float) -> Prior:
    """A prior uniform in the logarithm of the parameter, on ``[low, high]``."""
    if not high > low > 0:
        raise ValueError(
            f"A log-uniform prior needs high > low > 0, got ({low}, {high})."
        )
    return stats.loguniform(low, high)


def normal(mean: float, sigma: float) -> Prior:
    """A normal prior with the given mean and standard deviation."""
    if not sigma > 0:
        raise ValueError(f"A normal prior needs sigma > 0, got {sigma}.")
    return stats.norm(loc=mean, scale=sigma)


def truncated_normal(
    mean: float, sigma: float, low: float = -np.inf, high: float = np.inf
) -> Prior:
    """A normal prior truncated to ``[low, high]``."""
    if not sigma > 0:
        raise ValueError(f"A truncated normal prior needs sigma > 0, got {sigma}.")
    if not high > low:
        raise ValueError(
            f"A truncated normal prior needs high > low, got ({low}, {high})."
        )
    return stats.truncnorm(
        (low - mean) / sigma, (high - mean) / sigma, loc=mean, scale=sigma
    )


@runtime_checkable
class JointPrior(Protocol):
    """The interface a prior over several parameters must provide.

    ``dim`` is the number of parameters. ``logpdf`` and ``transform`` act on arrays
    of shape ``(..., dim)``: ``transform`` maps points of the unit hypercube onto the
    parameters, so that uniform points become draws from the prior. ``marginals``
    are the one-parameter priors of each parameter on its own.
    """

    @property
    def dim(self) -> int:
        """The number of parameters."""
        ...

    @property
    def marginals(self) -> tuple[Prior, ...]:
        """The prior of each parameter on its own."""
        ...

    def logpdf(self, x):
        """Return the joint log probability density at ``x``, shape ``(..., dim)``."""
        ...

    def transform(self, u):
        """Map points of the unit hypercube, shape ``(..., dim)``, onto the prior."""
        ...


@attrs.frozen(eq=False)
class MultivariateNormal:
    """A multivariate normal prior, with a mean and a covariance matrix.

    Its transform is ``mean + L @ Phi^-1(u)``, with ``L`` the Cholesky factor of the
    covariance and ``Phi`` the standard normal distribution function.
    """

    mean: np.ndarray = attrs.field(
        converter=lambda x: np.atleast_1d(np.asarray(x, float))
    )
    cov: np.ndarray = attrs.field(
        converter=lambda x: np.atleast_2d(np.asarray(x, float))
    )

    def __attrs_post_init__(self):
        """Check that the covariance is a symmetric positive-definite matrix."""
        k = self.mean.size
        if self.mean.shape != (k,) or self.cov.shape != (k, k):
            raise ValueError(
                "A multivariate normal prior needs a 1D mean and a square covariance "
                f"of matching size, got shapes {self.mean.shape} and {self.cov.shape}."
            )
        if not np.allclose(self.cov, self.cov.T):
            raise ValueError(
                "A multivariate normal prior needs a symmetric covariance."
            )
        try:
            linalg.cholesky(self.cov, lower=True)
        except linalg.LinAlgError:
            raise ValueError(
                "A multivariate normal prior needs a positive-definite covariance."
            ) from None

    @cached_property
    def _cholesky(self) -> np.ndarray:
        return linalg.cholesky(self.cov, lower=True)

    @cached_property
    def _precision(self) -> np.ndarray:
        return linalg.cho_solve((self._cholesky, True), np.eye(self.dim))

    @property
    def dim(self) -> int:
        """The number of parameters."""
        return self.mean.size

    @cached_property
    def marginals(self) -> tuple[Prior, ...]:
        """The normal prior of each parameter on its own."""
        return tuple(
            stats.norm(loc=m, scale=np.sqrt(v))
            for m, v in zip(self.mean, np.diag(self.cov), strict=True)
        )

    def logpdf(self, x):
        """Return the joint log probability density at ``x``, shape ``(..., dim)``."""
        d = np.asarray(x, dtype=float) - self.mean
        chi2 = np.einsum("...i,ij,...j->...", d, self._precision, d)
        logdet = 2 * np.sum(np.log(np.diag(self._cholesky)))
        return -0.5 * (chi2 + logdet + self.dim * np.log(2 * np.pi))

    def transform(self, u):
        """Map points of the unit hypercube, shape ``(..., dim)``, onto the prior."""
        return self.mean + stats.norm.ppf(u) @ self._cholesky.T


def multivariate_normal(mean, cov) -> MultivariateNormal:
    """A multivariate normal prior with the given mean vector and covariance matrix."""
    return MultivariateNormal(mean, cov)


@attrs.frozen
class Marginal:
    """The prior of one parameter of a :class:`JointPrior`, on its own.

    Give each parameter of a joint prior a marginal of it as its prior; the parameter
    space then uses the joint prior for the parameters together.
    """

    joint: JointPrior
    index: int = attrs.field(validator=attrs.validators.ge(0))

    @index.validator
    def _index_validator(self, att, value):
        if value >= self.joint.dim:
            raise ValueError(
                f"A joint prior over {self.joint.dim} parameters has no parameter "
                f"{value}."
            )

    def logpdf(self, x):
        """Return the marginal log probability density at ``x``."""
        return self.joint.marginals[self.index].logpdf(x)

    def ppf(self, q):
        """Return the value below which a fraction ``q`` of the marginal lies."""
        return self.joint.marginals[self.index].ppf(q)

    def cdf(self, x):
        """Return the fraction of the marginal below ``x``."""
        return self.joint.marginals[self.index].cdf(x)


def as_prior(prior) -> Prior:
    """Return ``prior`` as a :class:`Prior`.

    A ``(low, high)`` tuple becomes a uniform prior. Any object that already has
    ``logpdf`` and ``ppf`` methods, such as a frozen scipy distribution, is returned
    unchanged.
    """
    if isinstance(prior, Prior):
        return prior
    if isinstance(prior, tuple) and len(prior) == 2:
        return uniform(*prior)
    if isinstance(prior, JointPrior):
        raise TypeError(
            "A joint prior covers several parameters: give each of them a "
            "Marginal(joint, i) as its prior, or use param_vector(..., prior=joint)."
        )
    raise TypeError(
        "A prior must be a (low, high) tuple or an object with 'logpdf' and 'ppf' "
        f"methods (such as a frozen scipy.stats distribution); got {prior!r}."
    )
