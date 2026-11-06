"""Tests of the sampler adapters, on posteriors whose answer is known exactly.

A sampler passes if it recovers the posterior's moments within their Monte Carlo
error and, for a nested sampler, the evidence within its error, roughly
1 / sqrt(n_eff). Batched and unbatched stages, and resumed and uninterrupted runs,
must give the same samples.
"""

import hashlib

import arviz_stats
import attrs
import nautilus
import numpy as np
import pytest

from yabf import sampling, testing
from yabf._nautilus import AtomicSampler

FAST_NAUTILUS = {"n_live": 300, "n_eff": 3000, "n_networks": 1}
SIGNAL = ("global_signal", "global_signal")


def _check_moments(samples, truth, n_sigma=5.0):
    """Assert the samples' mean is within ``n_sigma`` Monte Carlo standard errors."""
    sd = np.sqrt(np.diag(truth.covariance))
    z = (samples.mean - truth.mean) / (sd / np.sqrt(samples.n_effective))
    assert np.all(np.abs(z) < n_sigma), z


@pytest.mark.parametrize(
    "make", [testing.correlated_gaussian, testing.bimodal], ids=["gaussian", "bimodal"]
)
def test_nautilus_recovers_the_posterior_and_the_evidence(make):
    analytic = make()
    samples = sampling.run_nautilus(analytic.posterior, seed=1, **FAST_NAUTILUS)
    assert samples.finished
    _check_moments(samples, analytic.truth)
    error = 1 / np.sqrt(samples.n_effective)
    assert abs(samples.log_evidence - analytic.log_evidence) < 5 * error
    if make is testing.bimodal:
        # The mode at positive x_0 has weight 0.7; its estimate is binomial.
        upper = samples.weights[samples.points[:, 0] > 0].sum()
        assert abs(upper - 0.7) < 5 * np.sqrt(0.7 * 0.3 / samples.n_effective)


def test_emcee_recovers_the_posterior():
    analytic = testing.correlated_gaussian()
    samples = sampling.run_emcee(
        analytic.posterior, 2000, np.random.default_rng(1), discard=400
    )
    assert samples.sampler == "emcee"
    assert samples.log_evidence is None
    _check_moments(samples, analytic.truth)
    np.testing.assert_allclose(samples.weights, 1 / len(samples.points))


def _run(sampler, posterior, **kwargs):
    """Run ``sampler`` briefly on ``posterior``, with a fixed seed."""
    if sampler == "emcee":
        return sampling.run_emcee(posterior, 150, np.random.default_rng(3), **kwargs)
    return sampling.run_nautilus(posterior, seed=3, **FAST_NAUTILUS, **kwargs)


def _with_signal(analytic):
    """The toy posterior, returning the global signal as a derived output."""
    return attrs.evolve(analytic.posterior, derived=[SIGNAL])


@pytest.mark.parametrize("sampler", ["emcee", "nautilus"])
def test_a_batched_simulator_gives_the_same_samples(sampler):
    """The batched path computes the same posterior as the unbatched one, so the same
    seed gives the same samples and derived outputs.

    The toy's global signal is exactly ``basis @ theta``, so each sample's derived
    output can also be checked against the sample itself.
    """
    analytic = testing.toy_simulation(n_pix=50, batched=True)
    slow = _run(sampler, _with_signal(testing.toy_simulation(n_pix=50)))
    fast = _run(sampler, _with_signal(analytic))
    np.testing.assert_array_equal(slow.points, fast.points)
    np.testing.assert_array_equal(slow.weights, fast.weights)
    np.testing.assert_array_equal(slow.derived[SIGNAL], fast.derived[SIGNAL])

    basis = analytic.posterior.stages[0].basis
    np.testing.assert_allclose(
        fast.derived[SIGNAL], fast.points @ basis.T, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        fast.log_prior, analytic.posterior.log_prior(fast.points)
    )


def test_emcee_resumes_to_the_uninterrupted_chain(tmp_path):
    """Stopping at step 100 and resuming to step 200 gives the chain a 200-step run
    gives, derived outputs included: the checkpoint holds the random state."""
    analytic = testing.toy_simulation(n_pix=50, batched=True)
    posterior = _with_signal(analytic)
    checkpoint = tmp_path / "chain.h5"
    sampling.run_emcee(posterior, 100, np.random.default_rng(4), checkpoint=checkpoint)
    resumed = sampling.run_emcee(
        posterior, 200, np.random.default_rng(99), checkpoint=checkpoint
    )
    whole = sampling.run_emcee(posterior, 200, np.random.default_rng(4))
    np.testing.assert_array_equal(resumed.points, whole.points)
    np.testing.assert_array_equal(resumed.derived[SIGNAL], whole.derived[SIGNAL])


def test_nautilus_resumes_from_a_checkpoint(tmp_path):
    """A run stopped after a few thousand evaluations is unfinished; resumed, it
    finishes, recovers the posterior, and has the derived outputs of every sample."""
    analytic = testing.toy_simulation(n_pix=50, batched=True)
    posterior = _with_signal(analytic)
    checkpoint = tmp_path / "run.h5"
    first = sampling.run_nautilus(
        posterior, seed=5, checkpoint=checkpoint, n_like_max=2000, **FAST_NAUTILUS
    )
    assert not first.finished
    resumed = sampling.run_nautilus(
        posterior, seed=5, checkpoint=checkpoint, **FAST_NAUTILUS
    )
    assert resumed.finished
    _check_moments(resumed, analytic.truth)
    assert np.all(np.isfinite(resumed.derived[SIGNAL]))


@pytest.mark.parametrize("method", ["write", "write_shell_update"])
def test_a_failed_checkpoint_write_leaves_the_previous_checkpoint(
    tmp_path, monkeypatch, method
):
    """A write that dies part way, as when the process is killed, leaves the old
    checkpoint as it was, and a run can resume from it."""
    analytic = testing.correlated_gaussian()
    checkpoint = tmp_path / "run.h5"
    sampling.run_nautilus(
        analytic.posterior,
        seed=6,
        checkpoint=checkpoint,
        n_like_max=2000,
        **FAST_NAUTILUS,
    )
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    def dies_part_way(self, filepath, *args, **kwargs):
        """Leave a partial file behind, then fail, as a killed write would."""
        with open(filepath, "wb") as f:
            f.write(b"partial")
        raise OSError("killed during the write")

    monkeypatch.setattr(nautilus.Sampler, method, dies_part_way)
    sampler = AtomicSampler(
        analytic.posterior.prior_transform,
        lambda theta: analytic.posterior.log_likelihood(theta)[0],
        n_dim=analytic.posterior.ndim,
        vectorized=True,
        filepath=str(checkpoint),
        resume=True,
        n_live=FAST_NAUTILUS["n_live"],
        n_networks=1,
    )
    args = (checkpoint,) if method == "write" else (checkpoint, 0)
    kwargs = {"overwrite": True} if method == "write" else {}
    with pytest.raises(OSError, match="killed"):
        getattr(sampler, method)(*args, **kwargs)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
    monkeypatch.undo()
    # The checkpoint still loads: building a sampler on it resumes from it.
    reloaded = AtomicSampler(
        analytic.posterior.prior_transform,
        lambda theta: analytic.posterior.log_likelihood(theta)[0],
        n_dim=analytic.posterior.ndim,
        vectorized=True,
        filepath=str(checkpoint),
        resume=True,
        n_live=FAST_NAUTILUS["n_live"],
        n_networks=1,
    )
    assert reloaded.n_like == sampler.n_like > 0


def test_the_moments_and_getdist_use_the_weights():
    """Weighted moments: the weighted mean and covariance of samples whose weights
    are those of a 30/70 mixture of two points."""
    points = np.array([[0.0, 0.0], [1.0, 2.0]])
    samples = sampling.Samples(
        names=("a", "b"),
        points=points,
        weights=[0.3, 0.7],
        log_likelihood=np.zeros(2),
        log_prior=np.zeros(2),
        derived={},
        n_effective=2.0,
    )
    np.testing.assert_allclose(samples.mean, [0.7, 1.4])
    np.testing.assert_allclose(samples.covariance, 0.21 * np.outer([1, 2], [1, 2]))
    mc = samples.to_getdist()
    np.testing.assert_allclose(mc.getMeans()[:2], samples.mean)


def test_systematic_resampling_keeps_each_samples_share():
    """Each sample appears floor(n w) or ceil(n w) times among n resampled ones."""
    rng = np.random.default_rng(8)
    weights = rng.dirichlet(np.ones(50))
    samples = sampling.Samples(
        names=("i", "x"),
        points=np.column_stack([np.arange(50), rng.normal(size=50)]),
        weights=weights,
        log_likelihood=np.zeros(50),
        log_prior=np.zeros(50),
        derived={("stage", "out"): np.arange(50) * 2.0},
        n_effective=20.0,
    )
    n = 1000
    resampled = samples.resample(n, rng)
    counts = np.bincount(resampled.points[:, 0].astype(int), minlength=50)
    assert np.all(counts >= np.floor(n * weights))
    assert np.all(counts <= np.ceil(n * weights))
    assert resampled.equally_weighted
    np.testing.assert_array_equal(
        resampled.derived["stage", "out"], resampled.points[:, 0] * 2
    )
    assert resampled.n_effective == 20.0


def test_emcee_samples_keep_each_walkers_chain_for_arviz():
    """In ArviZ, each chain is one walker: a rejected move repeats the walker's last
    position, so consecutive draws of a chain are often equal, which they never are
    between different walkers. After a converged run, R-hat is close to one."""
    analytic = testing.correlated_gaussian()
    post = analytic.posterior
    rng = np.random.default_rng(9)
    start = post.params.gaussian_ball(16, rng, center=analytic.truth.mean)
    samples = sampling.run_emcee(post, 3000, rng, start=start, discard=600)
    assert samples.n_chains == 16

    tree = samples.to_arviz(params=post.params)
    x0 = tree.posterior["x_0"].values
    assert x0.shape == (16, 2400)
    np.testing.assert_array_equal(x0.ravel(), samples.points[:, 0])
    np.testing.assert_array_equal(
        tree.posterior["x"].values[..., 1], tree.posterior["x_1"]
    )
    repeats = np.mean(x0[:, 1:] == x0[:, :-1])
    assert 0.2 < repeats < 0.9
    np.testing.assert_allclose(
        tree.sample_stats["lp"].values.ravel(),
        samples.log_likelihood + samples.log_prior,
    )
    rhat = arviz_stats.summary(tree, var_names=list(post.names), kind="diagnostics")[
        "r_hat"
    ]
    assert np.all(rhat.astype(float) < 1.05)


def test_arviz_takes_derived_outputs_with_their_coordinates():
    analytic = testing.toy_simulation(n_pix=50, batched=True)
    samples = sampling.run_emcee(_with_signal(analytic), 150, np.random.default_rng(3))
    freq = np.linspace(50.0, 100.0, 40)
    tree = samples.to_arviz(
        dims={"global_signal.global_signal": ["freq"]}, coords={"freq": freq}
    )
    signal = tree.posterior["global_signal.global_signal"]
    assert signal.dims == ("chain", "draw", "freq")
    np.testing.assert_array_equal(signal["freq"], freq)
    np.testing.assert_array_equal(
        signal.values.reshape(-1, 40), samples.derived[SIGNAL]
    )
    assert tree.posterior.attrs["sampler"] == "emcee"


def test_arviz_refuses_weighted_samples():
    analytic = testing.bimodal()
    samples = sampling.run_nautilus(analytic.posterior, seed=7, **FAST_NAUTILUS)
    with pytest.raises(ValueError, match="resample these first"):
        samples.to_arviz()
    tree = samples.resample(2000, np.random.default_rng(10)).to_arviz()
    assert tree.posterior["x_0"].shape == (1, 2000)
    assert tree.posterior.attrs["log_evidence"] == samples.log_evidence


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"weights": np.full(3, 0.5)}, "sum to one"),
        ({"n_chains": 2}, "cannot be 2 chains"),
        ({"log_prior": np.zeros(2)}, "2 entries for 3 samples"),
        ({"points": np.zeros((3, 3))}, "shape"),
    ],
    ids=["weights", "chains", "length", "points"],
)
def test_inconsistent_samples_are_rejected(change, match):
    good = {
        "names": ("a", "b"),
        "points": np.zeros((3, 2)),
        "weights": np.full(3, 1 / 3),
        "log_likelihood": np.zeros(3),
        "log_prior": np.zeros(3),
        "derived": {},
        "n_effective": 3.0,
    }
    with pytest.raises(ValueError, match=match):
        sampling.Samples(**{**good, **change})
