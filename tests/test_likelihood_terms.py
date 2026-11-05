"""Tests of likelihood terms: likelihoods scoring stage outputs in a posterior."""

from typing import ClassVar

import attrs
import numpy as np
import pytest
from scipy import stats

from yabf import priors
from yabf.likelihoods import Gaussian
from yabf.params import Param, param_vector
from yabf.posterior import LikelihoodTerm, Posterior
from yabf.stages import REQUIRED, All, GraphError, ParamSpec

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
    domain: tuple = (0, np.inf)
    requires: dict = attrs.field(factory=lambda: {"model": Curve})
    provides: dict = attrs.field(factory=lambda: {"model": Curve})

    @property
    def parameters(self):
        return {"gain": ParamSpec(1.0, domain=self.domain)}

    def compute(self, params, model):
        return {"model": Curve(params["gain"] * model.values)}


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
        BETA, [Line()], likelihoods=[LikelihoodTerm("lk", Gaussian(COV), DATA, Curve)]
    )
    theta = np.array([0.1, 0.7])
    lnl, _ = post.log_likelihood(theta)
    np.testing.assert_allclose(lnl, mvn(0.1 + 0.7 * X), rtol=1e-12)


def test_terms_apply_their_own_instrument_chain_to_a_shared_model():
    """Two terms score one model run: one as it is, one through its own gain stage."""
    line = Line()
    post = Posterior(
        [*BETA, Param("gain", prior=(0.5, 2.0))],
        [line],
        likelihoods=[
            LikelihoodTerm("raw", Gaussian(COV), DATA, Curve),
            LikelihoodTerm("windowed", Gaussian(COV), DATA, Curve, forward=[Gain()]),
        ],
    )
    theta = np.array([0.1, 0.7, 1.3])
    m = 0.1 + 0.7 * X

    lnl, _ = post.log_likelihood(theta)
    out = post.forward(theta)

    np.testing.assert_allclose(lnl, mvn(m) + mvn(1.3 * m), rtol=1e-12)
    np.testing.assert_allclose(out["windowed", "scored"].values, 1.3 * m)
    np.testing.assert_allclose(out["raw", "scored"].values, m)
    assert len(line.calls) == 2  # once per evaluation, shared by both terms


def test_mocks_are_distributed_as_the_likelihood_assumes():
    """For a Gaussian term, mock residuals about the forwarded model give r C^-1 r
    distributed as chi-squared with one degree of freedom per value."""
    post = Posterior(
        [*BETA, Param("gain", fixed=1.5)],
        [Line()],
        likelihoods=[
            LikelihoodTerm("lk", Gaussian(COV), DATA, Curve, forward=[Gain()])
        ],
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
    term = LikelihoodTerm("lk", ScaledGaussian(COV), DATA, Curve)
    post = Posterior(
        [*BETA, Param("noise", prior=(0.1, 5.0))], [Line()], likelihoods=[term]
    )
    for noise in [0.5, 1.0, 2.0]:
        lnl, _ = post.log_likelihood([0.1, 0.7, noise])
        np.testing.assert_allclose(
            lnl, mvn(0.1 + 0.7 * X, cov=noise**2 * COV), rtol=1e-12
        )


@attrs.frozen(eq=False)
class Clip:
    """An instrument stage clipping the model to a band ``[low, high]``."""

    name: str = "clip"
    parameters: ClassVar = {"low": ParamSpec(-1.0), "high": ParamSpec(1.0)}
    requires: ClassVar = {"model": Curve}
    provides: ClassVar = {"model": Curve}

    def check(self, params):
        if params["low"] >= params["high"]:
            raise ValueError("the band must have low < high.")

    def compute(self, params, model):
        return {"model": Curve(np.clip(model.values, params["low"], params["high"]))}


@pytest.mark.parametrize(
    ("term", "param", "match"),
    [
        (LikelihoodTerm("lk", ScaledGaussian(COV), DATA, Curve), "noise", "outside"),
        (
            LikelihoodTerm("lk", Gaussian(COV), DATA, Curve, forward=[Clip()]),
            "low",
            "low < high",
        ),
    ],
    ids=["likelihood_domain", "forward_check"],
)
def test_term_parameters_are_checked_against_the_prior(term, param, match):
    """A term's parameters, from its likelihood or its forward stages, are checked
    like any stage's: here, a prior reaching values the term rejects."""
    with pytest.raises(GraphError, match=match):
        Posterior(
            [*BETA, Param(param, prior=(-2.0, 2.0))], [Line()], likelihoods=[term]
        )


def test_the_noise_amplitude_of_a_mock_is_recovered():
    """Fit the noise amplitude to one large mock: the maximum-likelihood value lies
    within a few standard errors of the truth, sigma / sqrt(2 n)."""
    n = 400

    @attrs.frozen(eq=False)
    class Flat:
        name: str = "flat"
        parameters: ClassVar = {}
        requires: ClassVar = {}
        provides: ClassVar = {"model": Curve}

        def compute(self, params):
            return {"model": Curve(np.zeros(n))}

    lik = ScaledGaussian(np.eye(n))
    truth = 1.7
    mock = lik.with_params({"noise": truth}).simulate(
        np.zeros(n), np.random.default_rng(2)
    )
    post = Posterior(
        [Param("noise", prior=(0.1, 5.0))],
        [Flat()],
        likelihoods=[LikelihoodTerm("lk", lik, mock, Curve)],
    )
    grid = np.linspace(1.2, 2.2, 101)
    lnl, _ = post.log_likelihood(grid[:, None])
    best = grid[np.argmax(lnl)]
    assert abs(best - truth) < 4 * truth / np.sqrt(2 * n)


@pytest.mark.parametrize(
    ("seed", "expected"),
    [
        (None, "default"),
        (Param("seed", fixed=42), "fixed"),
        (Param("seed", prior=(0, 1000)), "sampled"),
    ],
)
def test_random_inputs_are_reported_with_how_they_are_set(seed, expected):
    params = [*BETA] if seed is None else [*BETA, seed]
    post = Posterior(
        params,
        [Line(seeded=True)],
        likelihoods=[LikelihoodTerm("lk", Gaussian(COV), DATA, Curve)],
    )
    assert post.random_inputs == {"seed": expected}


@attrs.frozen(eq=False)
class TwoInputs:
    name: str = "two"
    parameters: ClassVar = {}
    requires: ClassVar = {"a": Curve, "b": Curve}
    provides: ClassVar = {"model": Curve}

    def compute(self, params, a, b):
        return {"model": a}


@pytest.mark.parametrize(
    ("forward", "error", "match"),
    [
        ([TwoInputs()], ValueError, "take one input"),
        ([Gain(), Gain(name="g2", domain=(1, 2))], ValueError, "twice"),
        ([Gain(requires={"model": All(Curve)})], ValueError, "take one input"),
        ([Gain(requires={"model": float})], TypeError, "maps a float"),
        ([Gain(provides={"model": float})], TypeError, "to a float"),
    ],
    ids=["two_inputs", "conflicting_specs", "all_input", "wrong_input", "wrong_output"],
)
def test_malformed_terms_are_rejected(forward, error, match):
    with pytest.raises(error, match=match):
        LikelihoodTerm("lk", Gaussian(COV), DATA, Curve, forward=forward)


def test_mocks_are_drawn_one_point_at_a_time():
    post = Posterior(
        BETA, [Line()], likelihoods=[LikelihoodTerm("lk", Gaussian(COV), DATA, Curve)]
    )
    with pytest.raises(ValueError, match="one point"):
        post.simulate(np.zeros((3, 2)), np.random.default_rng(3))
