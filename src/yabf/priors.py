"""Priors for sampled parameters.

A prior is any object with vectorised ``logpdf(x)`` and ``ppf(u)`` methods: the log
probability density, and the inverse cumulative distribution function that maps the
unit interval onto the parameter. Frozen :mod:`scipy.stats` distributions satisfy this
as they are, so any of them can be used directly. The functions here are short names
for the common cases, and :func:`as_prior` turns a ``(low, high)`` tuple into a
uniform prior.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy import stats


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
    raise TypeError(
        "A prior must be a (low, high) tuple or an object with 'logpdf' and 'ppf' "
        f"methods (such as a frozen scipy.stats distribution); got {prior!r}."
    )
