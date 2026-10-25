"""Tests of the prior distributions.

Every prior must have a ``logpdf`` and a ``ppf`` that describe the same distribution,
and the prior transform must turn uniform draws into draws from that distribution.
"""

import numpy as np
import pytest
from scipy import stats

from yabf import priors

PRIORS = {
    "uniform": priors.uniform(-3.0, 1.0),
    "tuple": priors.as_prior((2.0, 5.0)),
    "log_uniform": priors.log_uniform(1e-2, 1e3),
    "normal": priors.normal(0.5, 0.2),
    "truncated_normal": priors.truncated_normal(0.0, 1.0, low=-0.5, high=2.0),
    "scipy_beta": priors.as_prior(stats.beta(2.0, 5.0)),
}


@pytest.fixture(params=sorted(PRIORS), ids=sorted(PRIORS))
def prior(request):
    return PRIORS[request.param]


def test_logpdf_matches_derivative_of_ppf(prior):
    # Change of variables: x = ppf(u) with u ~ U(0, 1) has density 1 / (dx/du),
    # so logpdf(ppf(u)) must equal -log(d ppf / du).
    u = np.linspace(0.02, 0.98, 49)
    h = 1e-6
    dxdu = (prior.ppf(u + h) - prior.ppf(u - h)) / (2 * h)
    np.testing.assert_allclose(prior.logpdf(prior.ppf(u)), -np.log(dxdu), atol=1e-5)


def test_prior_transform_draws_follow_the_prior(prior):
    rng = np.random.default_rng(1)
    x = prior.ppf(rng.uniform(size=4000))
    cdf = getattr(prior, "cdf", None)
    if cdf is None:
        pytest.skip("prior has no cdf to compare against")
    assert stats.kstest(x, cdf).pvalue > 1e-3


def test_vectorised_matches_scalar(prior):
    u = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(prior.ppf(u), [prior.ppf(v) for v in u])
    x = prior.ppf(u)
    np.testing.assert_allclose(prior.logpdf(x), [prior.logpdf(v) for v in x])


@pytest.mark.parametrize(
    ("factory", "args"),
    [
        (priors.uniform, (1.0, 1.0)),
        (priors.log_uniform, (0.0, 1.0)),
        (priors.normal, (0.0, 0.0)),
        (priors.truncated_normal, (0.0, 1.0, 2.0, 1.0)),
    ],
)
def test_invalid_parameters_raise(factory, args):
    with pytest.raises(ValueError, match="needs"):
        factory(*args)


@pytest.mark.parametrize("bad", [(1.0,), [0.0, 1.0], "uniform", 3.0])
def test_as_prior_rejects_non_priors(bad):
    with pytest.raises(TypeError, match="logpdf"):
        priors.as_prior(bad)


MVN = priors.multivariate_normal(
    [1.0, -2.0, 0.5],
    [[1.0, 0.6, 0.2], [0.6, 2.0, -0.3], [0.2, -0.3, 0.5]],
)


def test_joint_logpdf_matches_the_jacobian_of_its_transform():
    # x = T(u) with u uniform on the unit cube has density 1 / |det dT/du|.
    rng = np.random.default_rng(0)
    h = 1e-6
    for u in rng.uniform(0.05, 0.95, size=(20, MVN.dim)):
        jac = np.stack(
            [
                (MVN.transform(u + h * e) - MVN.transform(u - h * e)) / (2 * h)
                for e in np.eye(MVN.dim)
            ],
            axis=-1,
        )
        _, logdet = np.linalg.slogdet(jac)
        np.testing.assert_allclose(MVN.logpdf(MVN.transform(u)), -logdet, atol=1e-5)


def test_joint_transform_draws_have_the_prior_mean_and_covariance():
    n = 50_000
    x = MVN.transform(np.random.default_rng(1).uniform(size=(n, MVN.dim)))
    se = np.sqrt(np.diag(MVN.cov) / n)
    assert np.all(np.abs(x.mean(axis=0) - MVN.mean) < 5 * se)
    # The sampling error of a covariance element is about sqrt(2/n) times its scale.
    np.testing.assert_allclose(np.cov(x.T), MVN.cov, atol=5 * np.sqrt(2 / n) * 2.0)


def test_joint_logpdf_matches_scipy_over_a_batch():
    x = np.random.default_rng(2).normal(size=(4, 5, MVN.dim))
    expected = stats.multivariate_normal(MVN.mean, MVN.cov).logpdf(x)
    np.testing.assert_allclose(MVN.logpdf(x), expected)


def test_marginals_follow_the_joint_draws():
    x = MVN.transform(np.random.default_rng(3).uniform(size=(4000, MVN.dim)))
    for i, marginal in enumerate(MVN.marginals):
        assert stats.kstest(x[:, i], marginal.cdf).pvalue > 1e-3


@pytest.mark.parametrize(
    ("mean", "cov", "match"),
    [
        ([0.0, 0.0], [[1.0, 0.5], [0.0, 1.0]], "symmetric"),
        ([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]], "positive-definite"),
        ([0.0, 0.0], np.eye(3), "matching size"),
    ],
    ids=["asymmetric", "not_positive_definite", "wrong_size"],
)
def test_invalid_multivariate_normals_are_rejected(mean, cov, match):
    with pytest.raises(ValueError, match=match):
        priors.multivariate_normal(mean, cov)


def test_a_joint_prior_is_not_a_single_parameter_prior():
    with pytest.raises(TypeError, match="Marginal"):
        priors.as_prior(MVN)
