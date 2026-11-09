"""Tests of the analytic posteriors, the toy simulator and the stage contract check.

The posteriors here are the references that samplers are tested against, so the tests
check them exactly where possible: each posterior's log density against its stated
solution and evidence, and each mixture's moments against exact draws.
"""

from typing import ClassVar

import attrs
import emcee
import numpy as np
import pytest
from scipy import stats

from yabf import testing
from yabf.posterior import LogLikelihood
from yabf.stages import ParamSpec, StageError

FIXTURES = {
    "correlated_gaussian": testing.correlated_gaussian,
    "bimodal": testing.bimodal,
    "toy_simulation": testing.toy_simulation,
    "toy_simulation_batched": lambda: testing.toy_simulation(batched=True),
}


@pytest.fixture(params=FIXTURES.values(), ids=FIXTURES.keys())
def analytic(request):
    return request.param()


def test_the_log_posterior_is_the_truth_up_to_the_evidence(analytic):
    """log prior + log L = log p(theta | d) + log Z, at points across the posterior.

    This checks the stated posterior and evidence together: the difference is
    constant only if the posterior has the stated shape, and that constant is the log
    evidence only if the stated density is normalised.
    """
    theta = analytic.truth.sample(50, np.random.default_rng(0))
    lp, _ = analytic.posterior.log_posterior(theta)
    np.testing.assert_allclose(
        lp - analytic.truth.logpdf(theta), analytic.log_evidence, rtol=0, atol=1e-9
    )


def test_the_prior_box_holds_the_posterior(analytic):
    """Draws from the truth, far into its tails, are inside the prior."""
    theta = analytic.truth.sample(10_000, np.random.default_rng(1))
    assert np.all(np.isfinite(analytic.posterior.log_prior(theta)))


@pytest.mark.parametrize(
    "make", [testing.correlated_gaussian, testing.bimodal], ids=["gaussian", "bimodal"]
)
def test_a_mixture_has_the_moments_and_marginals_of_its_draws(make):
    """Exact draws have the stated mean (within 5 standard errors), covariance, and
    marginal distributions (KS tests)."""
    truth = make().truth
    n = 40_000
    draws = truth.sample(n, np.random.default_rng(2))
    se = np.sqrt(np.diag(truth.covariance) / n)
    assert np.all(np.abs(draws.mean(axis=0) - truth.mean) < 5 * se)
    # The variance of a sample covariance element is (S_ii S_jj + S_ij^2) / n for a
    # normal, and larger for a mixture; 5 of these is a loose but useful bound.
    cov = truth.covariance
    se_cov = np.sqrt((np.outer(np.diag(cov), np.diag(cov)) + cov**2) / n)
    assert np.all(np.abs(np.cov(draws.T) - cov) < 10 * se_cov)
    for i in range(truth.ndim):
        assert (
            stats.kstest(draws[:, i], lambda x, i=i: truth.marginal_cdf(x, i)).pvalue
            > 1e-3
        )


def test_a_bimodal_posterior_has_the_stated_weights():
    truth = testing.bimodal().truth
    draws = truth.sample(20_000, np.random.default_rng(3))
    upper = np.mean(draws[:, 0] > 0)
    assert abs(upper - 0.7) < 5 * np.sqrt(0.7 * 0.3 / len(draws))


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (([0.5, 0.6], [[0.0], [1.0]], [[[1.0]], [[1.0]]]), "sum to one"),
        (([1.0], [[0.0, 0.0]], [[[1.0, 0.0], [0.0, 1.0]]] * 2), "shape"),
        (([1.0], [[0.0, 0.0]], [[[1.0, 0.5], [0.4, 1.0]]]), "symmetric"),
    ],
    ids=["weights", "shapes", "asymmetric"],
)
def test_malformed_mixtures_are_rejected(args, match):
    with pytest.raises(ValueError, match=match):
        testing.GaussianMixture(*args)


def test_the_simulators_seed_changes_the_lightcone_but_not_its_average():
    analytic = testing.toy_simulation(n_pix=200)
    theta = analytic.truth.mean
    simulator, average = analytic.posterior.stages
    a = simulator.compute({"theta": theta, "seed": 1})["lightcone"]
    b = simulator.compute({"theta": theta, "seed": 2})["lightcone"]
    assert not np.allclose(a, b)
    np.testing.assert_allclose(
        average.compute({}, a)["global_signal"],
        average.compute({}, b)["global_signal"],
        rtol=0,
        atol=1e-12,
    )


def test_batched_and_unbatched_simulators_agree():
    slow = testing.toy_simulation()
    fast = testing.toy_simulation(batched=True)
    theta = slow.truth.sample(20, np.random.default_rng(4))
    np.testing.assert_array_equal(
        slow.posterior.log_likelihood(theta)[0], fast.posterior.log_likelihood(theta)[0]
    )


def _toy_stage_cases():
    """The toy stages, with values to check them at."""
    analytic = testing.toy_simulation(n_pix=50)
    simulator, average = analytic.posterior.stages
    batched = testing.ToySimulator(simulator.basis, n_pix=50, batched=True)
    theta = analytic.truth.sample(3, np.random.default_rng(5))
    lightcone = simulator.compute({"theta": theta[0], "seed": 0})["lightcone"]
    mixture = testing.bimodal()
    stage = mixture.posterior.stages[0]
    return {
        "simulator": (simulator, {"theta": theta[0], "seed": 7}, {}, None),
        "batched_simulator": (
            batched,
            {"theta": theta, "seed": np.array([1, 2, 3])},
            {},
            3,
        ),
        "global_signal": (average, {}, {"lightcone": lightcone}, None),
        "mixture_likelihood": (
            stage,
            {"x": mixture.truth.sample(4, np.random.default_rng(6))},
            {},
            4,
        ),
    }


TOY_STAGES = _toy_stage_cases()


@pytest.mark.parametrize(
    ("stage", "params", "inputs", "n"), TOY_STAGES.values(), ids=TOY_STAGES.keys()
)
def test_the_toy_stages_keep_the_stage_contract(stage, params, inputs, n):
    testing.assert_stage_contract(stage, params, inputs, n=n)


@attrs.frozen(eq=False)
class Faulty:
    """A stage with one fault, chosen by ``fault``."""

    fault: str
    name: str = "faulty"
    parameters: ClassVar = {"a": ParamSpec(1.0)}
    requires: ClassVar = {"field": np.ndarray}
    provides: ClassVar = {"out": np.ndarray}
    calls: list = attrs.field(factory=list)

    def compute(self, params, field):
        self.calls.append(1)
        if self.fault == "mutates":
            field *= 2
            return {"out": field}
        if self.fault == "random":
            return {"out": field + len(self.calls)}
        if self.fault == "wrong_type":
            return {"out": list(field)}
        return {"out": field * params["a"]}


@attrs.frozen(eq=False)
class InconsistentBatch:
    """A batched stage whose result at a point depends on the rest of the batch."""

    name: str = "inconsistent"
    parameters: ClassVar = {"a": ParamSpec(1.0)}
    requires: ClassVar = {}
    provides: ClassVar = {"lnl": LogLikelihood}
    batched: ClassVar = True

    def compute(self, params):
        a = np.asarray(params["a"])
        return {"lnl": [LogLikelihood(v) for v in a - a.mean()]}


@pytest.mark.parametrize(
    ("stage", "n", "error", "match"),
    [
        (Faulty("mutates"), None, AssertionError, "changed its parameters or inputs"),
        (Faulty("random"), None, AssertionError, "different 'out' when called again"),
        (Faulty("wrong_type"), None, StageError, "returned a list"),
        (InconsistentBatch(), 3, AssertionError, "at point 0 of the batch"),
    ],
    ids=["mutates", "random", "wrong_type", "inconsistent_batch"],
)
def test_the_contract_check_catches_a_faulty_stage(stage, n, error, match):
    if n is None:
        params, inputs = {"a": 2.0}, {"field": np.arange(4.0)}
    else:
        params, inputs = {"a": np.array([1.0, 2.0, 4.0])}, {}
    with pytest.raises(error, match=match):
        testing.assert_stage_contract(stage, params, inputs, n=n)
    if n is None:
        testing.assert_stage_contract(Faulty("none"), params, inputs)


def _emcee_moments(analytic, steps: int, seed: int):
    """Run emcee on ``analytic``; return the chain's mean, covariance and the number of
    effective samples."""
    post = analytic.posterior
    rng = np.random.default_rng(seed)
    nwalkers = 4 * post.ndim
    start = post.params.gaussian_ball(nwalkers, rng, center=analytic.truth.mean)
    sampler = emcee.EnsembleSampler(
        nwalkers,
        post.ndim,
        lambda theta: post.log_posterior(theta)[0],
        vectorize=True,
    )
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(start, steps, progress=False)
    tau = sampler.get_autocorr_time(discard=steps // 5, quiet=True).max()
    chain = sampler.get_chain(discard=steps // 5, flat=True)
    return chain.mean(axis=0), np.cov(chain.T), len(chain) / tau


def test_emcee_recovers_a_correlated_gaussian():
    """Sampler conformance, for emcee driven through Posterior.log_posterior.

    With n_eff effective samples, the mean must be within 5 Monte Carlo standard
    errors, and each covariance element within 5 of its standard errors,
    sqrt((S_ii S_jj + S_ij^2) / n_eff) for a normal posterior S.
    """
    analytic = testing.correlated_gaussian()
    mean, cov, n_eff = _emcee_moments(analytic, 3000, seed=7)
    truth = analytic.truth.covariance
    sd = np.sqrt(np.diag(truth))
    assert np.all(np.abs(mean - analytic.truth.mean) < 5 * sd / np.sqrt(n_eff))
    se = np.sqrt((np.outer(sd**2, sd**2) + truth**2) / n_eff)
    assert np.all(np.abs(cov - truth) < 5 * se)
