"""Tests of Param and ParamSpace."""

import numpy as np
import pytest
from scipy import stats

from yabf import priors
from yabf.params import Param, ParamSpace, get_transform, register_transform


@pytest.fixture
def space():
    return ParamSpace(
        [
            Param("a", prior=(-1.0, 3.0)),
            Param("log10_b", prior=priors.normal(1.0, 0.5), feeds="b", transform="pow10"),
            Param("c", fixed=2.0),
            Param("d", prior=stats.gamma(3.0)),
        ]
    )


def test_fixed_parameters_are_not_sampled(space):
    assert space.names == ("a", "log10_b", "d")
    assert space.ndim == 3


@pytest.mark.parametrize("name", ["identity", "pow10", "log10", "exp", "log"])
def test_transforms_are_inverted_by_their_inverse(name):
    t = get_transform(name)
    x = np.linspace(0.1, 3.0, 13)
    np.testing.assert_allclose(t.inverse(t.forward(x)), x)


def test_prior_transform_draws_follow_each_marginal(space):
    rng = np.random.default_rng(0)
    theta = space.sample_prior(4000, rng)
    assert theta.shape == (4000, 3)
    for i, p in enumerate(space.sampled):
        assert stats.kstest(theta[:, i], p.prior.cdf).pvalue > 1e-3


def test_log_prior_is_the_sum_of_the_marginals(space):
    rng = np.random.default_rng(1)
    theta = space.sample_prior(50, rng).reshape(5, 10, 3)
    expected = sum(
        p.prior.logpdf(theta[..., i]) for i, p in enumerate(space.sampled)
    )
    np.testing.assert_allclose(space.log_prior(theta), expected)


def test_log_prior_is_minus_infinity_outside_the_support(space):
    inside = space.fiducial
    outside = inside.copy()
    outside[0] = 10.0  # 'a' is uniform on [-1, 3]
    assert np.isfinite(space.log_prior(inside))
    assert space.log_prior(outside) == -np.inf


def test_unit_cube_centre_maps_to_the_prior_medians(space):
    np.testing.assert_allclose(space.prior_transform(np.full(3, 0.5)), space.fiducial)


def test_to_model_applies_transforms_and_broadcasts_fixed_values(space):
    theta = np.array([[0.0, 2.0, 1.0], [1.0, -1.0, 4.0]])
    model = space.to_model(theta)
    assert set(model) == {"a", "b", "c", "d"}
    np.testing.assert_allclose(model["b"], [100.0, 0.1])
    np.testing.assert_allclose(model["c"], [2.0, 2.0])


def test_gaussian_ball_is_inside_the_prior_with_the_requested_width(space):
    rng = np.random.default_rng(2)
    n, scale = 20000, 0.05
    theta = space.gaussian_ball(n, rng, scale=scale)
    assert np.all(np.isfinite(space.log_prior(theta)))
    widths = np.array(
        [(p.prior.ppf(0.8413447) - p.prior.ppf(0.1586553)) / 2 for p in space.sampled]
    )
    sigma = scale * widths
    # The fiducial points are well inside these priors, so the ball is barely clipped:
    # its mean is the centre (to 5 standard errors) and its spread the requested width.
    z = (theta.mean(axis=0) - space.fiducial) / (sigma / np.sqrt(n))
    assert np.all(np.abs(z) < 5)
    np.testing.assert_allclose(theta.std(axis=0), sigma, rtol=0.03)


def test_gaussian_ball_redraws_points_outside_a_bounded_prior():
    # Centred on the edge-adjacent point of a uniform prior: about half the raw draws
    # fall outside and must be redrawn.
    space = ParamSpace([Param("x", prior=(0.0, 1.0), fiducial=1e-6)])
    theta = space.gaussian_ball(1000, np.random.default_rng(3), scale=0.1)
    assert np.all((theta >= 0) & (theta <= 1))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "exactly one"),
        ({"prior": (0.0, 1.0), "fixed": 0.5}, "exactly one"),
        ({"prior": (0.0, 1.0), "fiducial": 2.0}, "outside the support"),
        ({"prior": (0.0, 1.0), "transform": "nope"}, "Unknown transform"),
    ],
)
def test_invalid_params_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Param("x", **kwargs)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ([Param("x", prior=(0, 1)), Param("x", prior=(0, 2))], "names"),
        ([Param("x", prior=(0, 1)), Param("y", prior=(0, 1), feeds="x")], "feeds"),
    ],
)
def test_duplicate_names_or_feeds_raise(params, match):
    with pytest.raises(ValueError, match=match):
        ParamSpace(params)


def test_registered_transform_cannot_be_overwritten():
    with pytest.raises(ValueError, match="already registered"):
        register_transform("pow10", np.exp, np.log)
