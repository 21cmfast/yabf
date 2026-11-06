"""Likelihoods: the statistics that score a model against data, and simulate data.

A likelihood holds a noise model: a covariance, a number of degrees of freedom, a
bound on a systematic. It provides two methods, which describe one distribution of the
data given the model:

- ``loglike(model, data)``: the log probability density of ``data`` given ``model``;
- ``simulate(model, rng)``: a draw of data from that same distribution.

Because both come from one object, mock data and inference cannot disagree about the
noise. ``model`` and ``data`` are arrays of values, in the same units, or objects with a
``values`` attribute (such as an observable). A model may also carry a ``covariance``
attribute, its own uncertainty, which :class:`Gaussian` can add to the data
covariance.

A data value that is NaN marks a flagged channel: it is left out of the likelihood,
which is the likelihood of the other channels with the flagged one marginalised.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import attrs
import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg, special, stats


@runtime_checkable
class Likelihood(Protocol):
    """The interface every likelihood provides."""

    def loglike(self, model: object, data: object) -> float:
        """Return the log probability density of ``data`` given ``model``."""
        ...

    def simulate(self, model: object, rng: np.random.Generator) -> np.ndarray:
        """Draw data from the distribution that :meth:`loglike` scores."""
        ...


def _values(x: object) -> np.ndarray:
    """Return the values of a model or data: ``x.values`` if it has them, else ``x``."""
    return np.asarray(getattr(x, "values", x), dtype=float).ravel()


def _model_covariance(model: object) -> np.ndarray | None:
    """Return the covariance a model carries, or ``None``."""
    cov = getattr(model, "covariance", None)
    return None if cov is None else np.asarray(cov, dtype=float)


def _frozen(x: np.ndarray) -> np.ndarray:
    """Return a read-only copy of ``x``."""
    x = np.array(x, dtype=float)
    x.setflags(write=False)
    return x


def _matrix(x: ArrayLike) -> np.ndarray:
    """Return ``x`` as a read-only 2D float array."""
    return _frozen(np.atleast_2d(np.asarray(x, dtype=float)))


def _observed(data: np.ndarray) -> np.ndarray:
    """Return which data values are observed: those that are not flagged as NaN."""
    return ~np.isnan(data)


def _positive(x: ArrayLike) -> np.ndarray:
    """Return ``x`` as a 1D float array, checking that every entry is positive."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if not np.all(x > 0):
        raise ValueError(f"Standard deviations must be positive, got {x}.")
    return _frozen(x)


def _check_covariance(inst: object, att: attrs.Attribute, cov: np.ndarray) -> None:
    """Check that ``cov`` is a symmetric positive-definite matrix."""
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"A covariance must be a square matrix, got {cov.shape}.")
    if not np.allclose(cov, cov.T, rtol=0, atol=1e-10 * np.abs(cov).max()):
        raise ValueError("A covariance must be symmetric.")
    try:
        linalg.cholesky(cov, lower=True)
    except linalg.LinAlgError:
        raise ValueError("A covariance must be positive definite.") from None


def _check_size(n: int, model: np.ndarray, what: str = "model") -> None:
    """Check that ``model`` has one value per dimension of the noise model."""
    if model.size != n:
        raise ValueError(
            f"The likelihood describes {n} values, but the {what} has {model.size}."
        )


@attrs.frozen(eq=False)
class Gaussian:
    """A multivariate normal distribution of the data about the model.

    Parameters
    ----------
    covariance
        The covariance of the data, an ``(n, n)`` matrix.
    model_covariance
        ``"include"`` adds the covariance a model carries (as ``model.covariance``),
        such as an emulator's error, to the data covariance; ``"ignore"`` leaves it
        out. A model without a covariance is the same either way. The log-determinant
        of the total covariance is always included, so a covariance that depends on
        the model is accounted for correctly.
    """

    covariance: np.ndarray = attrs.field(converter=_matrix, validator=_check_covariance)
    model_covariance: Literal["include", "ignore"] = attrs.field(
        default="include", validator=attrs.validators.in_(("include", "ignore"))
    )

    @classmethod
    def from_sigma(cls, sigma: ArrayLike, **kwargs: object) -> Gaussian:
        """Build a Gaussian with independent errors of standard deviation ``sigma``."""
        return cls(np.diag(_positive(sigma) ** 2), **kwargs)

    def total_covariance(self, model: object) -> np.ndarray:
        """Return the covariance of the data given ``model``."""
        if self.model_covariance == "ignore":
            return self.covariance
        cov = _model_covariance(model)
        return self.covariance if cov is None else self.covariance + cov

    def loglike(self, model: object, data: object) -> float:
        """Return the log probability density of ``data`` given ``model``."""
        m, d = _values(model), _values(data)
        _check_size(len(self.covariance), m)
        _check_size(len(self.covariance), d, "data")
        keep = _observed(d)
        cov = self.total_covariance(model)[np.ix_(keep, keep)]
        chol = linalg.cho_factor(cov, lower=True)
        r = (d - m)[keep]
        chi2 = r @ linalg.cho_solve(chol, r)
        logdet = 2 * np.sum(np.log(np.diag(chol[0])))
        return float(-0.5 * (chi2 + logdet + r.size * np.log(2 * np.pi)))

    def simulate(self, model: object, rng: np.random.Generator) -> np.ndarray:
        """Draw data from the normal distribution about ``model``."""
        m = _values(model)
        _check_size(len(self.covariance), m)
        chol = linalg.cholesky(self.total_covariance(model), lower=True)
        return m + chol @ rng.standard_normal(m.size)


@attrs.frozen(eq=False)
class MultivariateT:
    """A multivariate Student's t distribution of the data about the model.

    Its tails are heavier than a Gaussian's, which tempers the pull of outliers.

    Parameters
    ----------
    shape
        The ``(n, n)`` shape matrix. The covariance of the data is
        ``shape * df / (df - 2)`` for ``df > 2``.
    df
        The number of degrees of freedom; the distribution tends to a Gaussian with
        covariance ``shape`` as ``df`` grows.
    """

    shape: np.ndarray = attrs.field(converter=_matrix, validator=_check_covariance)
    df: float = attrs.field(converter=float, validator=attrs.validators.gt(0))

    def loglike(self, model: object, data: object) -> float:
        """Return the log probability density of ``data`` given ``model``."""
        m, d = _values(model), _values(data)
        _check_size(len(self.shape), m)
        _check_size(len(self.shape), d, "data")
        keep = _observed(d)
        shape = self.shape[np.ix_(keep, keep)]
        dist = stats.multivariate_t(loc=m[keep], shape=shape, df=self.df)
        return float(dist.logpdf(d[keep]))

    def simulate(self, model: object, rng: np.random.Generator) -> np.ndarray:
        """Draw data from the t distribution about ``model``."""
        m = _values(model)
        _check_size(len(self.shape), m)
        dist = stats.multivariate_t(loc=m, shape=self.shape, df=self.df)
        return np.atleast_1d(dist.rvs(random_state=rng))


@attrs.frozen(eq=False)
class Poisson:
    """Independent Poisson counts, whose expected values are the model."""

    def loglike(self, model: object, data: object) -> float:
        """Return the log probability of counts ``data`` given means ``model``."""
        mu, k = _values(model), _values(data)
        _check_size(mu.size, k, "data")
        keep = _observed(k)
        mu, k = mu[keep], k[keep]
        if np.any(k < 0) or np.any(k != np.round(k)):
            raise ValueError("Poisson data must be non-negative whole numbers.")
        return float(np.sum(stats.poisson.logpmf(k, mu)))

    def simulate(self, model: object, rng: np.random.Generator) -> np.ndarray:
        """Draw counts with means ``model``."""
        return rng.poisson(_values(model)).astype(float)


@attrs.frozen(eq=False)
class UpperLimit:
    """Measurements that bound the model from above, as in a systematics-limited survey.

    Each value is measured as ``d = m + s + n``: the model ``m``, a non-negative
    systematic ``s`` with a uniform prior, and Gaussian noise ``n`` of standard
    deviation ``sigma``. Marginalising the systematic gives the likelihood. A model
    below the data is barely penalised, since a systematic can make up the difference;
    a model above it is penalised as by a Gaussian.

    Parameters
    ----------
    sigma
        The noise standard deviation of each value.
    systematic_bound
        The upper end of the systematic's uniform prior, in the units of the values.
        ``None`` takes the prior to be unbounded, as is usual for an upper limit. That
        prior is improper, so the log-likelihood is then defined only up to a constant,
        and data cannot be simulated: give a bound far above the data to simulate
        mocks that match the unbounded likelihood.
    """

    sigma: np.ndarray = attrs.field(converter=_positive)
    systematic_bound: float | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(float),
        validator=attrs.validators.optional(attrs.validators.gt(0)),
    )

    def loglike(self, model: object, data: object) -> float:
        """Return the log-likelihood of ``model`` given the measurements ``data``."""
        m, d = _values(model), _values(data)
        _check_size(m.size, d, "data")
        keep = _observed(d)
        sigma = np.broadcast_to(self.sigma, m.shape)[keep]
        a = (d - m)[keep] / sigma
        if self.systematic_bound is None:
            return float(np.sum(special.log_ndtr(a)))
        b = a - self.systematic_bound / sigma
        # log(Phi(a) - Phi(b)) for a > b, without cancellation: from the lower tails
        # when b < 0, and as Phi(-b) - Phi(-a) from the upper tails otherwise.
        hi, lo = np.where(b < 0, a, -b), np.where(b < 0, b, -a)
        log_mass = special.log_ndtr(hi) + np.log1p(
            -np.exp(special.log_ndtr(lo) - special.log_ndtr(hi))
        )
        return float(np.sum(log_mass - np.log(self.systematic_bound)))

    def simulate(self, model: object, rng: np.random.Generator) -> np.ndarray:
        """Draw measurements: the model, plus a uniform systematic, plus noise."""
        if self.systematic_bound is None:
            raise ValueError(
                "The systematic of an upper limit has an improper (unbounded) prior, "
                "which cannot be sampled; set systematic_bound to simulate data."
            )
        m = _values(model)
        sigma = np.broadcast_to(self.sigma, m.shape)
        systematic = rng.uniform(0, self.systematic_bound, m.shape)
        return m + systematic + sigma * rng.standard_normal(m.shape)


@attrs.frozen(eq=False)
class AsymmetricGaussian:
    """Independent errors with different widths below and above the model.

    Each value is drawn from a split normal distribution with its mode at the model: a
    half-Gaussian of width ``sigma_lower`` below it and of width ``sigma_upper`` above
    it, joined so that the density is continuous and integrates to one.
    """

    sigma_lower: np.ndarray = attrs.field(converter=_positive)
    sigma_upper: np.ndarray = attrs.field(converter=_positive)

    def loglike(self, model: object, data: object) -> float:
        """Return the log probability density of ``data`` given ``model``."""
        m, d = _values(model), _values(data)
        _check_size(m.size, d, "data")
        keep = _observed(d)
        m, d = m[keep], d[keep]
        lo = np.broadcast_to(self.sigma_lower, keep.shape)[keep]
        hi = np.broadcast_to(self.sigma_upper, keep.shape)[keep]
        sigma = np.where(d < m, lo, hi)
        norm = np.log(np.sqrt(np.pi / 2) * (lo + hi))
        return float(np.sum(-0.5 * ((d - m) / sigma) ** 2 - norm))

    def simulate(self, model: object, rng: np.random.Generator) -> np.ndarray:
        """Draw data from the split normal distribution about ``model``."""
        m = _values(model)
        lo = np.broadcast_to(self.sigma_lower, m.shape)
        hi = np.broadcast_to(self.sigma_upper, m.shape)
        below = rng.uniform(size=m.shape) < lo / (lo + hi)
        size = np.abs(rng.standard_normal(m.shape))
        return np.where(below, m - lo * size, m + hi * size)


def _tabulated_pdf(inst: TabulatedPDF, att: attrs.Attribute, pdf: np.ndarray) -> None:
    """Check that the tabulated density is non-negative and matches its grid."""
    if pdf.shape != inst.x.shape:
        raise ValueError(
            f"The density has shape {pdf.shape}, but the grid has {inst.x.shape}."
        )
    if np.any(pdf < 0) or not np.any(pdf > 0):
        raise ValueError("A tabulated density must be non-negative, and not all zero.")


def _grid(x: ArrayLike) -> np.ndarray:
    """Return ``x`` as a strictly increasing 1D float array."""
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("A tabulated grid must have at least two increasing points.")
    return _frozen(x)


@attrs.frozen(eq=False)
class TabulatedPDF:
    """Independent errors with a tabulated distribution, the same for every value.

    The table gives the density of the residual ``data - model`` at the points ``x``;
    it is interpolated linearly between them, is zero outside them, and is normalised
    to integrate to one. This describes any measured error distribution, such as one
    published as a histogram or a sampled posterior.
    """

    x: np.ndarray = attrs.field(converter=_grid)
    pdf: np.ndarray = attrs.field(
        converter=lambda p: _frozen(np.ravel(p)), validator=_tabulated_pdf
    )

    @property
    def _density(self) -> np.ndarray:
        """Return the tabulated density, normalised to integrate to one."""
        return self.pdf / np.trapezoid(self.pdf, self.x)

    def loglike(self, model: object, data: object) -> float:
        """Return the log probability density of ``data`` given ``model``."""
        m, d = _values(model), _values(data)
        _check_size(m.size, d, "data")
        keep = _observed(d)
        r = (d - m)[keep]
        density = np.interp(r, self.x, self._density, left=0.0, right=0.0)
        with np.errstate(divide="ignore"):
            return float(np.sum(np.log(density)))

    def simulate(self, model: object, rng: np.random.Generator) -> np.ndarray:
        """Draw data about ``model``, inverting the exact piecewise-quadratic CDF."""
        m = _values(model)
        p, x = self._density, self.x
        width = np.diff(x)
        slope = np.diff(p) / width
        cdf = np.concatenate([[0.0], np.cumsum((p[:-1] + p[1:]) / 2 * width)])
        u = rng.uniform(size=m.shape) * cdf[-1]
        i = np.clip(np.searchsorted(cdf, u, side="right") - 1, 0, len(width) - 1)
        # Within interval i the density is p[i] + slope[i] t, so the mass up to t is
        # p[i] t + slope[i] t^2 / 2; solve it for the remaining mass u - cdf[i].
        rest, a, b = u - cdf[i], slope[i] / 2, p[i]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                np.abs(a) > 1e-12 * np.abs(b) / width[i],
                (-b + np.sqrt(b**2 + 4 * a * rest)) / (2 * a),
                rest / b,
            )
        return m + x[i] + np.clip(t, 0, width[i])
