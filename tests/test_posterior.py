"""Tests of the Posterior, on a linear model with a Gaussian likelihood."""

import weakref
from typing import ClassVar

import attrs
import numpy as np
import pytest

from yabf import priors
from yabf.params import param_vector
from yabf.posterior import LogLikelihood, Posterior
from yabf.stages import One, ParamSpec, StageError

X = np.linspace(-1.0, 1.0, 7)
SIGMA = 0.5


class Curve:
    """A model curve on the grid X (a plain class, so that weakrefs work)."""

    def __init__(self, values):
        self.values = values


def design(n):
    return np.vander(X, n, increasing=True)


@attrs.frozen(eq=False)
class Polynomial:
    """``sum_k beta_k x**k`` on the grid, optionally for a whole batch at once."""

    n: int
    batched: bool = False
    name: str = "poly"
    calls: list = attrs.field(factory=list)
    made: list = attrs.field(factory=list)
    requires: ClassVar = {}
    provides: ClassVar = {"model": Curve}

    @property
    def parameters(self):
        return {"beta": ParamSpec(shape=(self.n,))}

    def compute(self, params):
        self.calls.append(np.shape(params["beta"]))
        values = params["beta"] @ design(self.n).T
        curves = [Curve(v) for v in np.atleast_2d(values)]
        self.made.extend(weakref.ref(c) for c in curves)
        return {"model": curves if self.batched else curves[0]}


@attrs.frozen(eq=False)
class Gaussian:
    """A Gaussian likelihood with independent errors, and the chi-squared as a blob."""

    data: np.ndarray
    batched: bool = False
    name: str = "like"
    requires: dict = attrs.field(factory=lambda: {"model": Curve})
    parameters: ClassVar = {}
    provides: ClassVar = {"lnl": LogLikelihood, "chi2": float}

    def compute(self, params, model):
        models = np.array([m.values for m in model]) if self.batched else model.values
        chi2 = np.sum((self.data - models) ** 2, axis=-1) / SIGMA**2
        lnl = -0.5 * chi2 - X.size * np.log(np.sqrt(2 * np.pi) * SIGMA)
        if self.batched:
            return {"lnl": [LogLikelihood(v) for v in lnl], "chi2": list(chi2)}
        return {"lnl": LogLikelihood(lnl), "chi2": float(chi2)}


def closed_form(beta, data):
    chi2 = np.sum((data - beta @ design(beta.shape[-1]).T) ** 2, axis=-1) / SIGMA**2
    return -0.5 * chi2 - X.size * np.log(np.sqrt(2 * np.pi) * SIGMA), chi2


TAU = 1.0
BETA_TRUE = np.array([0.3, -0.8])
DATA = BETA_TRUE @ design(2).T + SIGMA * np.random.default_rng(1).standard_normal(
    X.size
)


def make_posterior(poly_batched=False, like_batched=False, derived=(("like", "chi2"),)):
    return Posterior(
        param_vector("beta", 2, prior=priors.normal(0.0, TAU)),
        [Polynomial(2, batched=poly_batched), Gaussian(DATA, batched=like_batched)],
        derived=derived,
    )


def test_log_likelihood_is_the_gaussian_closed_form():
    post = make_posterior()
    for theta in post.params.sample_prior(5, np.random.default_rng(2)):
        lnl, blobs = post.log_likelihood(theta)
        expected_lnl, expected_chi2 = closed_form(theta, DATA)
        np.testing.assert_allclose(lnl, expected_lnl, rtol=1e-12)
        np.testing.assert_allclose(blobs["like", "chi2"], expected_chi2, rtol=1e-12)


@pytest.mark.parametrize("poly_batched", [False, True])
@pytest.mark.parametrize("like_batched", [False, True])
def test_batched_and_point_by_point_evaluation_agree(poly_batched, like_batched):
    post = make_posterior(poly_batched, like_batched)
    theta = post.params.sample_prior(40, np.random.default_rng(3))

    lnl, blobs = post.log_likelihood(theta)
    singles = [post.log_likelihood(t) for t in theta]

    assert lnl.shape == (40,)
    np.testing.assert_allclose(lnl, [s[0] for s in singles], rtol=1e-12)
    np.testing.assert_allclose(
        blobs["like", "chi2"], [s[1]["like", "chi2"] for s in singles], rtol=1e-12
    )
    np.testing.assert_allclose(lnl, closed_form(theta, DATA)[0], rtol=1e-12)
    # A batched stage is called once for the batch (and once per single point, as a
    # batch of one); an unbatched one is called once per point.
    calls = post.stages[0].calls
    if poly_batched:
        assert calls[0] == (40, 2)
        assert calls[1:] == [(1, 2)] * 40
    else:
        assert calls == [(2,)] * 80


def test_likelihood_weighted_prior_draws_recover_the_analytic_posterior():
    """For a linear model with Gaussian prior and noise, the posterior is Gaussian.

    Weighting prior draws by their likelihood (importance sampling) must recover its
    mean, within the Monte Carlo error set by the effective sample size.
    """
    post = make_posterior(poly_batched=True, like_batched=True)
    theta = post.params.sample_prior(20_000, np.random.default_rng(4))
    lnl, _ = post.log_likelihood(theta)
    w = np.exp(lnl - lnl.max())
    w /= w.sum()
    ess = 1 / np.sum(w**2)

    a = design(2)
    precision = a.T @ a / SIGMA**2 + np.eye(2) / TAU**2
    cov = np.linalg.inv(precision)
    mean = cov @ a.T @ DATA / SIGMA**2

    assert ess > 1000
    np.testing.assert_array_less(
        np.abs(w @ theta - mean), 5 * np.sqrt(np.diag(cov) / ess)
    )


def test_forward_gives_every_output_and_matches_the_log_likelihood():
    post = make_posterior()
    theta = np.array([0.1, 0.2])

    out = post.forward(theta)

    assert set(out) == {("poly", "model"), ("like", "lnl"), ("like", "chi2")}
    np.testing.assert_allclose(out["poly", "model"].values, theta @ design(2).T)
    assert out["like", "lnl"] == post.log_likelihood(theta)[0]
    batch = post.forward(np.array([theta, 2 * theta]))
    assert [len(v) for v in batch.values()] == [2, 2, 2]
    assert batch["like", "lnl"][0] == out["like", "lnl"]


@attrs.frozen(eq=False)
class Smooth:
    """An unbatched stage between the simulator and the likelihood.

    It records how many of the simulator's curves are alive when it runs.
    """

    sim: Polynomial
    alive: list = attrs.field(factory=list)
    name: str = "smooth"
    parameters: ClassVar = {}
    requires: ClassVar = {"model": Curve}
    provides: ClassVar = {"smoothed": Curve}

    def compute(self, params, model):
        self.alive.append(sum(r() is not None for r in self.sim.made))
        return {"smoothed": Curve(model.values.copy())}


def test_unbatched_stages_release_each_point_before_the_next():
    """Running unbatched stages point by point keeps one simulation alive at a time,
    even when a batched stage comes after them."""
    sim = Polynomial(2)
    smooth = Smooth(sim)
    like = Gaussian(DATA, batched=True, requires={"model": One(Curve, name="smoothed")})
    post = Posterior(
        param_vector("beta", 2, prior=priors.normal(0.0, TAU)), [sim, smooth, like]
    )
    post.log_likelihood(post.params.sample_prior(10, np.random.default_rng(5)))
    assert smooth.alive == [1] * 10


def test_log_likelihood_terms_add_up():
    """Two likelihoods of the same model give the sum of their log-likelihoods, from
    a single run of the model."""
    other = DATA[::-1].copy()
    poly = Polynomial(2)
    post = Posterior(
        param_vector("beta", 2, prior=priors.normal(0.0, TAU)),
        [poly, Gaussian(DATA), Gaussian(other, name="like2")],
    )
    theta = np.array([0.4, -0.1])

    lnl, _ = post.log_likelihood(theta)

    assert post.loglike_outputs == (("like", "lnl"), ("like2", "lnl"))
    np.testing.assert_allclose(
        lnl, closed_form(theta, DATA)[0] + closed_form(theta, other)[0], rtol=1e-12
    )
    assert len(poly.calls) == 1


@pytest.mark.parametrize(
    ("make", "error", "match"),
    [
        (
            lambda: Posterior(
                param_vector("beta", 1, prior=priors.normal(0.0, TAU)), [Polynomial(1)]
            ),
            ValueError,
            "no likelihood",
        ),
        (lambda: make_posterior(derived=[("like", "nope")]), ValueError, "not stage"),
        (lambda: make_posterior().log_likelihood(np.zeros(3)), ValueError, "shape"),
        (
            lambda: make_posterior().log_likelihood(np.zeros((2, 2, 2))),
            ValueError,
            "shape",
        ),
    ],
)
def test_inconsistent_posteriors_and_points_are_rejected(make, error, match):
    with pytest.raises(error, match=match):
        make()


def test_a_batched_stage_must_return_one_value_per_point():
    @attrs.frozen(eq=False)
    class Short(Polynomial):
        def compute(self, params):
            return {"model": super().compute(params)["model"][:-1]}

    post = Posterior(
        param_vector("beta", 2, prior=priors.normal(0.0, TAU)),
        [Short(2, batched=True), Gaussian(DATA)],
    )
    with pytest.raises(StageError, match="sequence of 3 values"):
        post.log_likelihood(np.zeros((3, 2)))
