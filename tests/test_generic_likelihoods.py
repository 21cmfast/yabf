"""Tests of the generic likelihoods.

Each likelihood must agree with scipy or a closed form, be a normalised density of the
data, and simulate data from the very distribution its log-likelihood describes.
"""

import attrs
import numpy as np
import pytest
from scipy import integrate, special, stats

from yabf.likelihoods import (
    AsymmetricGaussian,
    Gaussian,
    Likelihood,
    MultivariateT,
    Poisson,
    TabulatedPDF,
    UpperLimit,
)


@attrs.frozen
class Model:
    """A model with its own uncertainty, as an emulator's prediction has."""

    values: np.ndarray
    covariance: np.ndarray | None = None


def random_covariance(n, rng, scale=1.0):
    a = rng.standard_normal((n, n))
    return scale * (a @ a.T / n + np.eye(n))


RNG = np.random.default_rng(0)
C_DATA = random_covariance(4, RNG)
C_MODEL = random_covariance(4, RNG, 0.3)
M4 = np.array([1.0, -0.5, 2.0, 0.3])
D4 = M4 + np.array([0.4, -1.1, 0.2, 0.9])
X = np.linspace(-3, 3, 41)
TABLE = np.exp(-0.5 * X**2) * (1 + 0.6 * np.tanh(2 * X))  # skewed, non-negative


@pytest.mark.parametrize(
    ("likelihood", "model", "data", "expected"),
    [
        (
            Gaussian(C_DATA),
            M4,
            D4,
            stats.multivariate_normal(M4, C_DATA).logpdf(D4),
        ),
        (
            Gaussian(C_DATA),
            Model(M4, C_MODEL),
            D4,
            stats.multivariate_normal(M4, C_DATA + C_MODEL).logpdf(D4),
        ),
        (
            Gaussian(C_DATA, model_covariance="ignore"),
            Model(M4, C_MODEL),
            D4,
            stats.multivariate_normal(M4, C_DATA).logpdf(D4),
        ),
        (
            MultivariateT(C_DATA, df=4),
            M4,
            D4,
            stats.multivariate_t(M4, C_DATA, df=4).logpdf(D4),
        ),
        (
            Poisson(),
            [2.5, 0.3, 7.0],
            [3, 0, 5],
            stats.poisson.logpmf([3, 0, 5], [2.5, 0.3, 7.0]).sum(),
        ),
        (
            UpperLimit([0.5, 2.0]),
            [1.0, 3.0],
            [1.4, 2.0],
            special.log_ndtr(np.array([0.4 / 0.5, -1.0 / 2.0])).sum(),
        ),
        (
            # The systematic integrated out numerically: (1/S) int_0^S N(d | m + s).
            UpperLimit([0.5], systematic_bound=3.0),
            [1.0],
            [2.2],
            np.log(
                integrate.quad(lambda s: stats.norm.pdf(2.2, 1.0 + s, 0.5), 0, 3)[0] / 3
            ),
        ),
        (
            TabulatedPDF(X, TABLE),
            [0.5, -1.0],
            [0.75, -1.5],
            np.log(np.interp([0.25, -0.5], X, TABLE / np.trapezoid(TABLE, X))).sum(),
        ),
    ],
)
def test_log_likelihoods_match_scipy_and_closed_forms(
    likelihood, model, data, expected
):
    assert isinstance(likelihood, Likelihood)
    np.testing.assert_allclose(likelihood.loglike(model, data), expected, rtol=1e-10)


# One-dimensional likelihoods, each a density of the data around a model value.
DENSITIES_1D = {
    "gaussian": Gaussian.from_sigma([0.7]),
    "student t": MultivariateT([[0.5]], df=3),
    "upper limit": UpperLimit([0.5], systematic_bound=2.0),
    "asymmetric": AsymmetricGaussian([0.4], [1.3]),
    "tabulated": TabulatedPDF(X, TABLE),
}


@pytest.mark.parametrize("name", list(DENSITIES_1D))
def test_each_likelihood_is_a_normalised_density_of_the_data(name):
    lik = DENSITIES_1D[name]

    def density(d):
        return np.exp(lik.loglike([0.3], [d]))

    total = sum(
        integrate.quad(density, *limits, limit=200)[0]
        for limits in [(-np.inf, 0.3), (0.3, np.inf)]
    )
    np.testing.assert_allclose(total, 1.0, rtol=1e-6)


@pytest.mark.parametrize("name", list(DENSITIES_1D))
def test_simulated_data_follow_the_likelihood(name):
    """Draws from ``simulate`` pass a KS test against the CDF implied by ``loglike``."""
    lik = DENSITIES_1D[name]
    rng = np.random.default_rng(1)
    draws = np.array([lik.simulate([0.3], rng)[0] for _ in range(4000)])
    grid = np.linspace(-30, 30, 200_001)
    density = np.exp([lik.loglike([0.3], [d]) for d in grid[::10]])
    cdf = integrate.cumulative_trapezoid(density, grid[::10], initial=0)
    assert stats.kstest(draws, lambda d: np.interp(d, grid[::10], cdf)).pvalue > 1e-3


@pytest.mark.parametrize(
    ("lik", "model"),
    [
        (Gaussian(C_DATA), M4),
        (Gaussian(C_DATA), Model(M4, C_MODEL)),
        (Gaussian(C_DATA, model_covariance="ignore"), Model(M4, C_MODEL)),
    ],
)
def test_simulated_gaussian_residuals_have_chi2_distribution(lik, model):
    """With the covariance the likelihood uses, r^T C^-1 r is chi-squared, n dof."""
    rng = np.random.default_rng(2)
    cov = lik.total_covariance(model)
    inv = np.linalg.inv(cov)
    chi2 = []
    for _ in range(3000):
        r = lik.simulate(model, rng) - M4
        chi2.append(r @ inv @ r)
    assert stats.kstest(chi2, stats.chi2(4).cdf).pvalue > 1e-3


def test_simulated_t_residuals_have_an_f_distribution():
    """For a multivariate t, r^T S^-1 r / n follows F(n, df)."""
    rng = np.random.default_rng(3)
    lik = MultivariateT(C_DATA, df=5)
    inv = np.linalg.inv(C_DATA)
    stat = [(r := lik.simulate(M4, rng) - M4) @ inv @ r / 4 for _ in range(3000)]
    assert stats.kstest(stat, stats.f(4, 5).cdf).pvalue > 1e-3


def test_simulated_counts_follow_the_poisson_distribution():
    rng = np.random.default_rng(4)
    counts = Poisson().simulate(np.full(20_000, 3.2), rng)
    observed = np.bincount(counts.astype(int), minlength=12)[:12]
    expected = stats.poisson.pmf(np.arange(12), 3.2) * counts.size
    keep = expected > 5
    chi2 = np.sum((observed[keep] - expected[keep]) ** 2 / expected[keep])
    assert stats.chi2(keep.sum() - 1).sf(chi2) > 1e-3


def test_an_upper_limit_penalises_only_models_above_the_data():
    """Lowering the model never lowers the likelihood, and far below the data costs
    nothing, since a positive systematic can fill the gap."""
    lik = UpperLimit([1.0])
    models = np.linspace(-20, 10, 61)
    lnl = np.array([lik.loglike([m], [2.0]) for m in models])
    assert np.all(np.diff(lnl) <= 0)
    np.testing.assert_allclose(lnl[0], 0.0, atol=1e-12)
    # Far above the data, it falls off like a Gaussian in (m - d) / sigma.
    np.testing.assert_allclose(lnl[-1], stats.norm.logcdf(2.0 - 10.0), rtol=1e-10)


def test_an_unbounded_upper_limit_cannot_simulate():
    with pytest.raises(ValueError, match="improper"):
        UpperLimit([1.0]).simulate([0.0], np.random.default_rng(5))


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Gaussian([[1.0, 2.0], [2.0, 1.0]]), "positive definite"),
        (lambda: Gaussian([[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
        (lambda: Gaussian(np.eye(2), model_covariance="sometimes"), "must be in"),
        (lambda: Gaussian(np.eye(2)).loglike([1.0, 2.0, 3.0], [0.0, 0.0]), "3"),
        (lambda: Gaussian.from_sigma([1.0, 0.0]), "positive"),
        (lambda: MultivariateT(np.eye(2), df=0), "df"),
        (lambda: Poisson().loglike([1.0], [-1.0]), "non-negative whole"),
        (lambda: Poisson().loglike([1.0], [1.5]), "non-negative whole"),
        (lambda: UpperLimit([1.0], systematic_bound=-1.0), "systematic_bound"),
        (lambda: AsymmetricGaussian([1.0], [-1.0]), "positive"),
        (lambda: TabulatedPDF([0.0, 1.0], [1.0, -1.0]), "non-negative"),
        (lambda: TabulatedPDF([0.0, 1.0], [1.0]), "shape"),
        (lambda: TabulatedPDF([1.0, 0.0], [1.0, 1.0]), "increasing"),
    ],
)
def test_inconsistent_likelihoods_and_inputs_are_rejected(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def test_a_bounded_upper_limit_stays_finite_far_from_the_model():
    """Far above model + bound, the log-likelihood is very negative but finite: the
    Gaussian tail of the noise, not a numerical -inf."""
    lik = UpperLimit([1.0], systematic_bound=2.0)
    lnl = lik.loglike([0.0], [40.0])
    upper, lower = stats.norm.logsf(38.0), stats.norm.logsf(40.0)
    expected = upper + np.log1p(-np.exp(lower - upper)) - np.log(2.0)
    np.testing.assert_allclose(lnl, expected, rtol=1e-8)
