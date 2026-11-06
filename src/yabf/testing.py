"""Analytic posteriors, a toy simulator and a stage contract check, for testing.

These help test yabf itself, samplers and the adapters that drive them, and code built
on yabf, such as stages in other packages:

- :class:`GaussianMixture` is a distribution known exactly: its mean, covariance,
  density, marginal distributions and exact draws.
- :func:`correlated_gaussian` and :func:`bimodal` build a
  :class:`~yabf.posterior.Posterior` whose posterior is a known mixture, with a known
  evidence. Each returns an :class:`AnalyticPosterior`, which holds the posterior with
  its exact solution.
- :class:`ToySimulator` and :class:`GlobalSignal` stand in for a simulator and an
  observable: a large product, with a random seed, and a small summary of it. The
  summary is linear in the parameters and does not depend on the seed, so
  :func:`toy_simulation` builds a posterior through them that is known exactly too.
- :func:`assert_stage_contract` checks that a stage keeps the stage protocol.

A sampler recovers one of these posteriors if its samples have the mixture's moments
within their Monte Carlo error, and a nested sampler if its evidence agrees with
:attr:`AnalyticPosterior.log_evidence` within its stated error.

Every posterior here has a uniform prior on a box around the mixture, at least ten
standard deviations wide on every side of every component. The probability of the
mixture outside the box is then below ``1e-20``, so the posterior is the mixture, and
the evidence its stated value, to double precision.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import attrs
import numpy as np
from numpy.typing import ArrayLike
from scipy import special, stats

from .likelihoods import Gaussian
from .params import Param, param_vector
from .posterior import LikelihoodTerm, LogLikelihood, Posterior
from .stages import (
    One,
    ParamSpec,
    Stage,
    _declaration_problems,
    call_stage,
    call_stage_batch,
)

#: How many standard deviations the prior box extends beyond every component.
BOX_WIDTH = 10.0

#: The relative tolerance between a batch and its points, in
#: :func:`assert_stage_contract`.
BATCH_RTOL = 1e-10


def _readonly(value: ArrayLike) -> np.ndarray:
    """Return a read-only float copy of ``value``."""
    out = np.array(value, dtype=float)
    out.flags.writeable = False
    return out


@attrs.frozen(eq=False)
class GaussianMixture:
    """A mixture of multivariate normal distributions.

    Parameters
    ----------
    weights
        The weight of each of the ``k`` components, positive and summing to one.
    means
        The mean of each component, shape ``(k, d)``.
    covariances
        The covariance of each component, shape ``(k, d, d)``, symmetric and positive
        definite.
    """

    weights: np.ndarray = attrs.field(converter=_readonly)
    means: np.ndarray = attrs.field(converter=_readonly)
    covariances: np.ndarray = attrs.field(converter=_readonly)

    def __attrs_post_init__(self) -> None:
        """Check the shapes, the weights and the covariances."""
        k = len(self.weights)
        if self.weights.ndim != 1 or self.means.ndim != 2 or len(self.means) != k:
            raise ValueError(
                "Give one weight and one mean of shape (d,) per component."
            )
        d = self.means.shape[1]
        if self.covariances.shape != (k, d, d):
            raise ValueError(
                f"The covariances must have shape ({k}, {d}, {d}), got "
                f"{self.covariances.shape}."
            )
        if np.any(self.weights <= 0) or not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("The weights must be positive and sum to one.")
        for cov in self.covariances:
            if not np.allclose(cov, cov.T):
                raise ValueError("Each covariance must be symmetric.")
            np.linalg.cholesky(cov)  # raises if not positive definite

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return self.means.shape[1]

    @property
    def mean(self) -> np.ndarray:
        """The mean of the mixture."""
        return self.weights @ self.means

    @property
    def covariance(self) -> np.ndarray:
        """The covariance of the mixture: within and between the components."""
        second = np.einsum(
            "k,kij->ij",
            self.weights,
            self.covariances + np.einsum("ki,kj->kij", self.means, self.means),
        )
        return second - np.outer(self.mean, self.mean)

    def logpdf(self, x: ArrayLike) -> np.ndarray:
        """The log density at ``x``, of shape ``(..., d)``."""
        x = np.asarray(x, dtype=float)
        flat = x.reshape(-1, self.ndim)
        terms = [
            np.log(w) + np.atleast_1d(stats.multivariate_normal(m, c).logpdf(flat))
            for w, m, c in zip(self.weights, self.means, self.covariances, strict=True)
        ]
        return special.logsumexp(terms, axis=0).reshape(x.shape[:-1])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` points from the mixture, shape ``(n, d)``."""
        component = rng.choice(len(self.weights), size=n, p=self.weights)
        out = np.empty((n, self.ndim))
        for k, (m, c) in enumerate(zip(self.means, self.covariances, strict=True)):
            here = component == k
            out[here] = rng.multivariate_normal(m, c, size=here.sum())
        return out

    def marginal_cdf(self, x: ArrayLike, i: int) -> np.ndarray:
        """The cumulative distribution of coordinate ``i`` alone, at ``x``."""
        sigma = np.sqrt(self.covariances[:, i, i])
        z = (np.asarray(x, dtype=float)[..., None] - self.means[:, i]) / sigma
        return stats.norm.cdf(z) @ self.weights


@attrs.frozen(eq=False)
class AnalyticPosterior:
    """A posterior, with its exact solution.

    Parameters
    ----------
    posterior
        The posterior, built from yabf's parameters and stages.
    truth
        The posterior distribution of the sampled parameters, exactly.
    log_evidence
        The natural log of the evidence: the integral of the likelihood over the prior.
    """

    posterior: Posterior
    truth: GaussianMixture
    log_evidence: float


def _box(truth: GaussianMixture) -> tuple[np.ndarray, np.ndarray]:
    """Return the edges of a box :data:`BOX_WIDTH` deviations around every component."""
    sigma = np.sqrt(np.diagonal(truth.covariances, axis1=1, axis2=2))
    low = (truth.means - BOX_WIDTH * sigma).min(axis=0)
    high = (truth.means + BOX_WIDTH * sigma).max(axis=0)
    return low, high


def _box_params(name: str, truth: GaussianMixture) -> tuple[Param, ...]:
    """Return a vector of parameters, uniform on the box around ``truth``."""
    low, high = _box(truth)
    return param_vector(
        name, prior=[(lo, hi) for lo, hi in zip(low, high, strict=True)]
    )


def _log_volume(truth: GaussianMixture) -> float:
    """Return the log of the volume of the box around ``truth``."""
    low, high = _box(truth)
    return float(np.sum(np.log(high - low)))


@attrs.frozen(eq=False)
class MixtureLikelihood:
    """A stage whose log-likelihood of the parameters ``x`` is a mixture's log density.

    It is batched: it scores a whole batch of points in one call.
    """

    truth: GaussianMixture
    name: str = "likelihood"
    requires: ClassVar = {}
    provides: ClassVar = {"lnl": LogLikelihood}
    batched: ClassVar = True

    @property
    def parameters(self) -> dict[str, ParamSpec]:
        """The point ``x``, a vector of the mixture's dimension."""
        return {"x": ParamSpec(shape=(self.truth.ndim,))}

    def compute(self, params: Mapping[str, np.ndarray]) -> dict[str, list]:
        """The log density at each point of the batch."""
        return {"lnl": [LogLikelihood(v) for v in self.truth.logpdf(params["x"])]}


def _mixture_posterior(truth: GaussianMixture) -> AnalyticPosterior:
    """Return the posterior whose likelihood is ``truth``, on its prior box."""
    posterior = Posterior(_box_params("x", truth), [MixtureLikelihood(truth)])
    return AnalyticPosterior(posterior, truth, -_log_volume(truth))


def _correlation(ndim: int, rho: float) -> np.ndarray:
    """Return the correlation matrix ``rho**|i - j|``."""
    i = np.arange(ndim)
    return rho ** np.abs(i[:, None] - i[None, :])


def correlated_gaussian(ndim: int = 4, rho: float = 0.8) -> AnalyticPosterior:
    """A posterior that is a correlated multivariate normal.

    The mean of coordinate ``i`` is ``i``, its standard deviation grows from 0.5 to 2,
    and the correlation between coordinates ``i`` and ``j`` is ``rho**|i - j|``. The
    parameters are ``x_0``, ``x_1``, ...
    """
    sigma = np.linspace(0.5, 2.0, ndim)
    cov = _correlation(ndim, rho) * np.outer(sigma, sigma)
    truth = GaussianMixture([1.0], [np.arange(float(ndim))], [cov])
    return _mixture_posterior(truth)


def bimodal(
    ndim: int = 2, separation: float = 8.0, weights: Sequence[float] = (0.3, 0.7)
) -> AnalyticPosterior:
    """A posterior with two well-separated modes of different weights.

    Both modes are correlated normals (correlation 0.5 between neighbouring
    coordinates, unit variances), centred ``separation`` standard deviations apart
    along ``x_0``. A sampler that finds both should recover their weights.
    """
    cov = _correlation(ndim, 0.5)
    offset = np.zeros(ndim)
    offset[0] = separation / 2
    truth = GaussianMixture(weights, [-offset, offset], [cov, cov])
    return _mixture_posterior(truth)


def legendre_basis(frequencies: ArrayLike, n: int) -> np.ndarray:
    """The first ``n`` Legendre polynomials over the band, shape ``(len, n)``."""
    nu = np.asarray(frequencies, dtype=float)
    x = 2 * (nu - nu.min()) / (nu.max() - nu.min()) - 1
    return np.polynomial.legendre.legvander(x, n - 1)


@attrs.frozen(eq=False)
class ToySimulator:
    """A stand-in for a simulator: a seeded lightcone around a smooth signal.

    The lightcone has one row per frequency and one column per pixel. Each row is the
    signal ``basis @ theta`` at that frequency, plus noise of standard deviation
    ``scatter`` drawn from the seed and made to average to zero across the pixels. The
    lightcone is large and random, but its average over the pixels is exactly the
    signal.

    Parameters
    ----------
    basis
        The signal's basis functions, shape ``(n_freq, n_theta)``; see
        :func:`legendre_basis`.
    n_pix
        The number of pixels.
    scatter
        The standard deviation of the noise across pixels.
    batched
        Whether to compute a whole batch of points in one call.
    """

    basis: np.ndarray = attrs.field(converter=_readonly)
    n_pix: int = 1000
    scatter: float = 1.0
    batched: bool = False
    name: str = "simulator"
    requires: ClassVar = {}
    provides: ClassVar = {"lightcone": np.ndarray}

    @property
    def parameters(self) -> dict[str, ParamSpec]:
        """The signal's coefficients ``theta``, and the random ``seed``."""
        return {
            "theta": ParamSpec(shape=(self.basis.shape[1],)),
            "seed": ParamSpec(0, random=True, description="Seeds the pixel noise."),
        }

    def _lightcone(self, theta: np.ndarray, seed: int) -> np.ndarray:
        """The lightcone at one point."""
        noise = np.random.default_rng(int(seed)).normal(
            0.0, self.scatter, (len(self.basis), self.n_pix)
        )
        noise -= noise.mean(axis=1, keepdims=True)
        return (self.basis @ theta)[:, None] + noise

    def compute(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The lightcone, at one point or, if batched, at each point of the batch."""
        if not self.batched:
            return {"lightcone": self._lightcone(params["theta"], params["seed"])}
        thetas, seeds = (
            params["theta"],
            np.broadcast_to(params["seed"], len(params["theta"])),
        )
        return {
            "lightcone": [
                self._lightcone(t, s) for t, s in zip(thetas, seeds, strict=True)
            ]
        }


@attrs.frozen
class GlobalSignal:
    """A stand-in for an observable: the lightcone averaged over its pixels."""

    name: str = "global_signal"
    parameters: ClassVar = {}
    requires: ClassVar = {"lightcone": np.ndarray}
    provides: ClassVar = {"global_signal": np.ndarray}
    cacheable: ClassVar = True

    def compute(self, params: Mapping[str, Any], lightcone: np.ndarray) -> dict:
        """The average of the lightcone over its pixels, at each frequency."""
        return {"global_signal": lightcone.mean(axis=1)}


def _linear_gaussian(
    basis: np.ndarray, cov: np.ndarray, data: np.ndarray
) -> tuple[GaussianMixture, float]:
    """The posterior and log evidence of ``theta`` given ``data ~ N(basis theta, cov)``.

    The posterior is for a flat prior over all of parameter space: normal, with
    covariance ``F^-1``, where ``F = B^T C^-1 B``, and mean ``F^-1 B^T C^-1 d``. The
    returned log evidence is the log of the likelihood's integral over the parameters.
    """
    n, p = basis.shape
    cinv_b = np.linalg.solve(cov, basis)
    fisher = basis.T @ cinv_b
    b = cinv_b.T @ data
    post_cov = np.linalg.inv(fisher)
    post_mean = post_cov @ b
    chi2 = data @ np.linalg.solve(cov, data) - b @ post_mean
    log_integral = (
        -0.5 * (n - p) * np.log(2 * np.pi)
        - 0.5 * np.linalg.slogdet(cov)[1]
        - 0.5 * np.linalg.slogdet(fisher)[1]
        - 0.5 * chi2
    )
    return GaussianMixture([1.0], [post_mean], [post_cov]), float(log_integral)


def toy_simulation(
    n_theta: int = 3,
    n_freq: int = 40,
    n_pix: int = 1000,
    sigma: float = 0.1,
    seed: int = 0,
    batched: bool = False,
) -> AnalyticPosterior:
    """A posterior through a toy simulator, an observable and a likelihood term.

    The simulator (:class:`ToySimulator`) makes a lightcone from the coefficients
    ``theta`` of a smooth signal, the observable (:class:`GlobalSignal`) averages it
    over the pixels, and a :class:`~yabf.posterior.LikelihoodTerm` scores a mock of
    that signal with independent Gaussian noise of standard deviation ``sigma``. The
    signal is linear in ``theta``, so the posterior is normal and known exactly.

    Parameters
    ----------
    n_theta
        The number of coefficients, the parameters ``theta_0``, ``theta_1``, ...
    n_freq, n_pix
        The size of the lightcone.
    sigma
        The noise of the mock, in each frequency channel.
    seed
        Seeds the mock's noise. The simulator's own seed is left at its default.
    batched
        Whether the simulator computes a whole batch of points in one call.
    """
    frequencies = np.linspace(50.0, 100.0, n_freq)
    basis = legendre_basis(frequencies, n_theta)
    rng = np.random.default_rng(seed)
    true_theta = rng.normal(0.0, 1.0, n_theta)
    noise = Gaussian.from_sigma(np.full(n_freq, sigma))
    data = noise.simulate(basis @ true_theta, rng)
    truth, log_integral = _linear_gaussian(basis, noise.covariance, data)
    posterior = Posterior(
        _box_params("theta", truth),
        [ToySimulator(basis, n_pix=n_pix, batched=batched), GlobalSignal()],
        likelihoods=[
            LikelihoodTerm(
                "radiometer",
                noise,
                One(np.ndarray, name="global_signal"),
                data=data,
            )
        ],
    )
    return AnalyticPosterior(posterior, truth, log_integral - _log_volume(truth))


def _same(a: object, b: object, rtol: float = 0.0) -> bool:
    """Whether ``a`` and ``b`` are equal, NaN equal to NaN; numbers within ``rtol``."""
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        return a.keys() == b.keys() and all(_same(a[k], b[k], rtol) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(
            _same(x, y, rtol) for x, y in zip(a, b, strict=True)
        )
    if hasattr(a, "values") and hasattr(b, "values"):
        return type(a) is type(b) and _same(a.values, b.values, rtol)
    if isinstance(a, np.ndarray | float | int) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if rtol and a.dtype.kind in "fc" and b.dtype.kind in "fc":
            return bool(np.allclose(a, b, rtol=rtol, atol=0, equal_nan=True))
        return bool(np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    return bool(a == b)


def _require(condition: bool, message: str) -> None:
    """Raise ``AssertionError(message)`` unless ``condition`` holds."""
    if not condition:
        raise AssertionError(message)


def assert_stage_contract(
    stage: Stage,
    params: Mapping[str, Any],
    inputs: Mapping[str, Any] | None = None,
    n: int | None = None,
) -> None:
    """Check that ``stage`` keeps the stage protocol, at the values given.

    The checks are that:

    - the stage's declarations are well formed, as :func:`~yabf.stages.resolve` checks
      them;
    - ``compute`` returns exactly the outputs the stage provides, of their types;
    - calling it again with the same values gives the same outputs;
    - it changes none of the parameters or inputs it is given;
    - for a batched stage, the batch gives at each point what that point alone gives,
      to a relative :data:`BATCH_RTOL`, since vectorised arithmetic may round
      differently.

    Parameters
    ----------
    stage
        The stage to check.
    params
        Values of the stage's parameters: for a batched stage, with a leading axis of
        length ``n``.
    inputs
        The stage's inputs, by name: for a batched stage, a sequence of ``n`` values
        each.
    n
        The number of points, for a batched stage.

    Raises
    ------
    AssertionError
        If the stage breaks the protocol, saying how.
    """
    inputs = dict(inputs or {})
    problems = _declaration_problems(stage)
    _require(not problems, f"Stage {stage.name!r} is malformed: {problems}")
    batched = getattr(stage, "batched", False)
    if batched and n is None:
        raise TypeError(
            f"Stage {stage.name!r} is batched: give the number of points n."
        )

    def run() -> dict[str, Any]:
        """Compute the stage once at the given values."""
        if batched:
            return call_stage_batch(stage, params, inputs, n)
        return call_stage(stage, params, inputs)

    before = copy.deepcopy((dict(params), inputs))
    first, second = run(), run()
    _require(
        _same(before, (dict(params), inputs)),
        f"Stage {stage.name!r} changed its parameters or inputs.",
    )
    for k in first:
        _require(
            _same(first[k], second[k]),
            f"Stage {stage.name!r} gave a different {k!r} when called again with the "
            "same values.",
        )
    if batched:
        for i in range(n):
            one = call_stage_batch(
                stage,
                {k: np.asarray(v)[i : i + 1] for k, v in params.items()},
                {k: [v[i]] for k, v in inputs.items()},
                1,
            )
            for k in first:
                _require(
                    _same(first[k][i], one[k][0], rtol=BATCH_RTOL),
                    f"Batched stage {stage.name!r} gave a different {k!r} at point {i} "
                    "of the batch than at that point alone.",
                )
