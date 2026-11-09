"""Tests of likelihood terms: likelihoods scoring stage outputs in a posterior."""

from typing import ClassVar

import attrs
import numpy as np
import pytest
from scipy import stats

from yabf import priors
from yabf.likelihoods import Gaussian
from yabf.params import DerivedParam, Param, param_vector
from yabf.posterior import LikelihoodTerm, Posterior
from yabf.stages import REQUIRED, GraphError, One, ParamSpec

X = np.linspace(-1.0, 1.0, 6)
COV = np.diag(np.full(X.size, 0.3**2)) + 0.01


class Curve:
    """A model curve on the grid X."""

    def __init__(self, values):
        self.values = np.asarray(values)


@attrs.frozen(eq=False)
class Line:
    """``beta_0 + beta_1 x``, counting its runs; it may also take a random seed."""

    name: str = "line"
    calls: list = attrs.field(factory=list)
    seeded: bool = False
    requires: ClassVar = {}
    provides: ClassVar = {"model": Curve}

    @property
    def parameters(self):
        specs = {"beta": ParamSpec(shape=(2,))}
        if self.seeded:
            specs["seed"] = ParamSpec(0, random=True)
        return specs

    def compute(self, params):
        self.calls.append(1)
        return {"model": Curve(params["beta"][0] + params["beta"][1] * X)}


@attrs.frozen(eq=False)
class Gain:
    """An instrument stage scaling the model by a gain parameter."""

    name: str = "gain"
    parameters: ClassVar = {"gain": ParamSpec(1.0, domain=(0, np.inf))}
    requires: ClassVar = {"model": Curve}
    provides: ClassVar = {"observed": Curve}

    def compute(self, params, model):
        return {"observed": Curve(params["gain"] * model.values)}


OBSERVED = One(Curve, name="observed")


@attrs.frozen(eq=False)
class ScaledGaussian:
    """A Gaussian whose covariance is scaled by a sampled noise amplitude squared."""

    covariance: np.ndarray
    parameters: ClassVar = {"noise": ParamSpec(REQUIRED, domain=(0, np.inf))}

    def with_params(self, params):
        return Gaussian(params["noise"] ** 2 * self.covariance)

    def loglike(self, model, data):
        raise NotImplementedError("use with_params")

    def simulate(self, model, rng):
        raise NotImplementedError("use with_params")


BETA = param_vector("beta", 2, prior=priors.normal(0.0, 1.0))
DATA = 0.2 + 0.5 * X + np.random.default_rng(0).multivariate_normal(np.zeros(6), COV)


def mvn(mean, data=DATA, cov=COV):
    return stats.multivariate_normal(mean, cov).logpdf(data)


def test_a_term_scores_its_model_as_its_likelihood_does():
    post = Posterior(
        BETA,
        [Line()],
        likelihoods=[LikelihoodTerm("lk", Gaussian(COV), Curve, data=DATA)],
    )
    theta = np.array([0.1, 0.7])
    lnl, _ = post.log_likelihood(theta)
    np.testing.assert_allclose(lnl, mvn(0.1 + 0.7 * X), rtol=1e-12)


def test_two_terms_score_different_outputs_of_one_model_run():
    """One term scores the model as it is, the other through an instrument's gain."""
    line = Line()
    post = Posterior(
        [*BETA, Param("gain", prior=(0.5, 2.0))],
        [line, Gain()],
        likelihoods=[
            LikelihoodTerm("raw", Gaussian(COV), Curve, data=DATA),
            LikelihoodTerm("observed", Gaussian(COV), OBSERVED, data=DATA),
        ],
    )
    theta = np.array([0.1, 0.7, 1.3])
    m = 0.1 + 0.7 * X

    lnl, _ = post.log_likelihood(theta)
    np.testing.assert_allclose(lnl, mvn(m) + mvn(1.3 * m), rtol=1e-12)
    assert len(line.calls) == 1  # the model ran once, for both terms
    assert post.scored == {"raw": ("line", "model"), "observed": ("gain", "observed")}


def test_mocks_are_distributed_as_the_likelihood_assumes():
    """For a Gaussian term, mock residuals about the scored model give r C^-1 r
    distributed as chi-squared with one degree of freedom per value."""
    post = Posterior(
        [*BETA, Param("gain", fixed=1.5)],
        [Line(), Gain()],
        likelihoods=[LikelihoodTerm("lk", Gaussian(COV), OBSERVED)],
    )
    theta = np.array([0.1, 0.7])
    model = 1.5 * (0.1 + 0.7 * X)
    inv = np.linalg.inv(COV)
    rng = np.random.default_rng(1)
    chi2 = [
        (r := post.simulate(theta, rng)["lk"] - model) @ inv @ r for _ in range(2000)
    ]
    assert stats.kstest(chi2, stats.chi2(X.size).cdf).pvalue > 1e-3


def test_a_likelihood_with_its_own_parameters_gets_them_from_the_posterior():
    term = LikelihoodTerm("lk", ScaledGaussian(COV), Curve, data=DATA)
    post = Posterior(
        [*BETA, Param("noise", prior=(0.1, 5.0))], [Line()], likelihoods=[term]
    )
    for noise in [0.5, 1.0, 2.0]:
        lnl, _ = post.log_likelihood([0.1, 0.7, noise])
        np.testing.assert_allclose(
            lnl, mvn(0.1 + 0.7 * X, cov=noise**2 * COV), rtol=1e-12
        )


def test_the_parameters_of_a_likelihood_are_checked_against_the_prior():
    """A likelihood's own parameters are checked like any stage's: here, a prior
    reaching noise levels the likelihood rejects."""
    term = LikelihoodTerm("lk", ScaledGaussian(COV), Curve, data=DATA)
    with pytest.raises(GraphError, match="outside"):
        Posterior(
            [*BETA, Param("noise", prior=(-2.0, 2.0))], [Line()], likelihoods=[term]
        )


def test_the_noise_amplitude_of_a_mock_is_recovered():
    """Simulate one large mock from a posterior without data, add it as the data, and
    fit the noise amplitude: the maximum-likelihood value lies within a few standard
    errors of the truth, sigma / sqrt(2 n)."""
    n = 400

    @attrs.frozen(eq=False)
    class Flat:
        name: str = "flat"
        parameters: ClassVar = {}
        requires: ClassVar = {}
        provides: ClassVar = {"model": Curve}

        def compute(self, params):
            return {"model": Curve(np.zeros(n))}

    truth = 1.7
    model = Posterior(
        [Param("noise", prior=(0.1, 5.0))],
        [Flat()],
        likelihoods=[LikelihoodTerm("lk", ScaledGaussian(np.eye(n)), Curve)],
    )
    post = model.with_data(model.simulate([truth], np.random.default_rng(2)))
    grid = np.linspace(1.2, 2.2, 101)
    lnl, _ = post.log_likelihood(grid[:, None])
    best = grid[np.argmax(lnl)]
    assert abs(best - truth) < 4 * truth / np.sqrt(2 * n)


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        ([], "default"),
        ([Param("seed", fixed=42)], "fixed"),
        ([Param("seed", prior=(0, 1000))], "sampled"),
        (
            [
                Param("log_seed", prior=(0, 3)),
                DerivedParam("seed", "pow10", "log_seed"),
            ],
            "derived",
        ),
    ],
    ids=["default", "fixed", "sampled", "derived"],
)
def test_random_inputs_are_reported_with_how_they_are_set(seed, expected):
    params = [*BETA, *seed]
    post = Posterior(
        params,
        [Line(seeded=True)],
        likelihoods=[LikelihoodTerm("lk", Gaussian(COV), Curve, data=DATA)],
    )
    assert post.random_inputs == {"seed": expected}


@attrs.frozen(eq=False)
class Shift:
    """A stage providing the model shifted by a constant, under another name."""

    name: str = "shift"
    parameters: ClassVar = {}
    requires: ClassVar = {"model": Curve}
    provides: ClassVar = {"shifted": Curve}

    def compute(self, params, model):
        return {"shifted": Curve(model.values + 0.25)}


@pytest.mark.parametrize(
    ("model", "expected"),
    [(Curve, 0.0), (One(Curve, name="shifted"), 0.25)],
    ids=["model", "named"],
)
def test_a_term_scores_the_output_it_names(model, expected):
    """With two curves in the graph, a term scores the one it asks for by name."""
    post = Posterior(
        BETA,
        [Line(), Shift()],
        likelihoods=[LikelihoodTerm("lk", Gaussian(COV), model, data=DATA)],
    )
    lnl, _ = post.log_likelihood([0.1, 0.7])
    np.testing.assert_allclose(lnl, mvn(0.1 + 0.7 * X + expected), rtol=1e-12)


def test_the_model_to_score_must_be_a_requirement():
    with pytest.raises(TypeError, match="must be a type or One"):
        LikelihoodTerm("lk", Gaussian(COV), "model", data=DATA)


def test_a_batch_draws_each_mock_at_its_own_point():
    """Mocks at prior draws of the line and of a noise amplitude: each residual about
    its own point's model, in units of its own noise, gives r C^-1 r / noise^2
    distributed as chi-squared with one degree of freedom per value."""
    term = LikelihoodTerm("lk", ScaledGaussian(COV), Curve)
    post = Posterior(
        [*BETA, Param("noise", prior=(0.1, 5.0))], [Line()], likelihoods=[term]
    )
    rng = np.random.default_rng(3)
    theta = post.params.sample_prior(2000, rng)
    mocks = post.simulate(theta, rng)["lk"]
    assert mocks.shape == (len(theta), X.size)
    r = mocks - (theta[:, :1] + theta[:, 1:2] * X)
    chi2 = np.einsum("ni,ij,nj->n", r, np.linalg.inv(COV), r) / theta[:, 2] ** 2
    assert stats.kstest(chi2, stats.chi2(X.size).cdf).pvalue > 1e-3


def _gained(line=None, **kwargs):
    """A posterior of the line and an instrument gain, scored by one Gaussian term."""
    return Posterior(
        [*BETA, Param("gain", prior=(0.5, 1.5))],
        [line or Line(), Gain()],
        likelihoods=[LikelihoodTerm("lk", Gaussian(COV), OBSERVED, **kwargs)],
        derived=[("lk", "lnl"), ("gain", "observed")],
    )


def test_the_log_posterior_adds_the_prior_and_skips_the_likelihood_outside_it():
    line = Line()
    post = _gained(line, data=DATA)
    theta = np.array([[0.2, 0.5, 1.0], [0.2, 0.5, -1.0], [-0.3, 0.1, 1.4]])
    lp, blobs = post.log_posterior(theta)
    assert len(line.calls) == 2  # the model ran at the two points inside the prior
    inside = [0, 2]
    lnl, _ = post.log_likelihood(theta[inside])
    np.testing.assert_allclose(lp[inside], post.log_prior(theta[inside]) + lnl)
    assert lp[1] == -np.inf
    np.testing.assert_allclose(blobs["lk", "lnl"][inside], lnl)
    assert np.isnan(blobs["lk", "lnl"][1])
    assert blobs["gain", "observed"][1] is None
    np.testing.assert_allclose(
        blobs["gain", "observed"][2].values, 1.4 * (-0.3 + 0.1 * X)
    )
    # One point at a time gives the same.
    for i in inside:
        one, _ = post.log_posterior(theta[i])
        assert one == pytest.approx(lp[i])
    assert post.log_posterior(theta[1]) == (
        -np.inf,
        {("lk", "lnl"): None, ("gain", "observed"): None},
    )


@pytest.mark.parametrize(
    ("use", "match"),
    [
        (lambda post: post.log_likelihood(np.zeros(3) + 1), "have no data"),
        (lambda post: post.with_data({"other": DATA}), "no likelihood terms named"),
    ],
    ids=["no_data", "unknown_term"],
)
def test_a_posterior_without_data_can_only_simulate(use, match):
    with pytest.raises(ValueError, match=match):
        use(_gained())


def test_adding_data_gives_the_posterior_built_with_it():
    built = _gained(data=DATA)
    added = _gained().with_data({"lk": DATA})
    theta = np.array([[0.2, 0.5, 1.0], [-0.3, 0.1, 1.4]])
    np.testing.assert_array_equal(
        added.log_likelihood(theta)[0], built.log_likelihood(theta)[0]
    )
