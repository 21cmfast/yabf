"""Tests of Param and ParamSpace."""

import numpy as np
import pytest
from scipy import stats

from yabf import priors
from yabf.params import (
    Param,
    ParamSpace,
    Tie,
    get_transform,
    param_vector,
    register_relation,
    register_transform,
)
from yabf.priors import Marginal


@pytest.fixture
def space():
    return ParamSpace(
        [
            Param("a", prior=(-1.0, 3.0)),
            Param(
                "log10_b", prior=priors.normal(1.0, 0.5), feeds="b", transform="pow10"
            ),
            Param("c", fixed=2.0),
            Param("d", prior=stats.gamma(3.0)),
        ]
    )


def test_fixed_parameters_are_not_sampled(space):
    assert space.names == ("a", "log10_b", "d")
    assert space.ndim == 3


_REALS = np.linspace(-6.0, 6.0, 49)
_POSITIVE = np.logspace(-6.0, 6.0, 49)


@pytest.mark.parametrize(
    ("name", "domain"),
    [
        ("identity", _REALS),
        ("pow10", _REALS),
        ("log10", _POSITIVE),
        ("exp", _REALS),
        ("log", _POSITIVE),
    ],
)
def test_transforms_are_inverted_by_their_inverse(name, domain):
    t = get_transform(name)
    np.testing.assert_allclose(t.inverse(t.forward(domain)), domain, rtol=1e-12)
    # The image of the domain round-trips the other way too.
    image = t.forward(domain)
    np.testing.assert_allclose(t.forward(t.inverse(image)), image, rtol=1e-12)


def test_prior_transform_draws_follow_each_marginal(space):
    rng = np.random.default_rng(0)
    theta = space.sample_prior(4000, rng)
    assert theta.shape == (4000, 3)
    for i, p in enumerate(space.sampled):
        assert stats.kstest(theta[:, i], p.prior.cdf).pvalue > 1e-3


def test_log_prior_is_the_sum_of_the_marginals(space):
    rng = np.random.default_rng(1)
    theta = space.sample_prior(50, rng).reshape(5, 10, 3)
    expected = sum(p.prior.logpdf(theta[..., i]) for i, p in enumerate(space.sampled))
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


def test_points_of_infinite_density_are_inside_the_prior():
    # The arcsine distribution has infinite density at the edges of its support.
    space = ParamSpace([Param("x", prior=stats.beta(0.5, 0.5), fiducial=0.0)])
    assert space.log_prior([0.0]) == np.inf
    theta = space.gaussian_ball(100, np.random.default_rng(6), scale=0.1)
    assert np.all((theta >= 0) & (theta <= 1))


def test_gaussian_ball_needs_a_single_centre(space):
    with pytest.raises(ValueError, match="center must have shape"):
        space.gaussian_ball(4, np.random.default_rng(), center=np.zeros((4, 3)))


def test_a_space_with_nothing_sampled_has_an_empty_prior():
    space = ParamSpace([Param("a", fixed=1.0), Param("b", tied={"a": 2.0})])
    theta = space.sample_prior(5, np.random.default_rng(7))
    assert theta.shape == (5, 0)
    np.testing.assert_array_equal(space.log_prior(theta), np.zeros(5))
    np.testing.assert_array_equal(space.to_model(theta)["b"], np.full(5, 2.0))


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
        ({"prior": (0.0, 1.0), "tied": "y"}, "exactly one"),
        ({"tied": "y", "transform": "pow10"}, "no transform"),
        ({"tied": "y", "feeds": "v", "index": 0}, "cannot be an element"),
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
        ([Param("x", prior=(0, 1)), Param("y", tied="z")], "no parameter feeds"),
        ([Param("x", tied="y"), Param("y", tied={"x": 2.0})], "cycle"),
    ],
)
def test_inconsistent_param_spaces_raise(params, match):
    with pytest.raises(ValueError, match=match):
        ParamSpace(params)


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Param("x", prior=(0.0, 1.0), transform="nope"), "Unknown transform"),
        (lambda: Tie("nope", ("y",)), "Unknown relation"),
        (lambda: register_transform("pow10", np.exp, np.log), "already registered"),
        (lambda: register_relation("linear", np.exp), "already registered"),
    ],
    ids=[
        "unknown_transform",
        "unknown_relation",
        "transform_exists",
        "relation_exists",
    ],
)
def test_names_must_be_registered_once(make, match):
    with pytest.raises(ValueError, match=match):
        make()


def _astro_params(order):
    # 21cmFAST's dependent defaults for the minihalo parameters, in log10 units.
    params = {
        "F_STAR10_ACG": Param("F_STAR10_ACG", prior=(-3.0, 0.0)),
        "ALPHA_STAR_ACG": Param("ALPHA_STAR_ACG", prior=(-0.5, 1.0)),
        "F_STAR7_MCG": Param(
            "F_STAR7_MCG", tied={"F_STAR10_ACG": 1.0, "ALPHA_STAR_ACG": -3.0}
        ),
        "ALPHA_STAR_MCG": Param("ALPHA_STAR_MCG", tied="ALPHA_STAR_ACG"),
        "LX_OVER_SFR_ACG": Param("LX_OVER_SFR_ACG", fixed=40.0),
        "LX_OVER_SFR_MCG": Param("LX_OVER_SFR_MCG", tied="LX_OVER_SFR_ACG"),
        # A tie of a tie: resolved after the tie it depends on, whatever the order.
        "F_STAR7_COPY": Param("F_STAR7_COPY", tied="F_STAR7_MCG"),
    }
    return ParamSpace([params[k] for k in order])


_DEPENDENCIES_FIRST = [
    "F_STAR10_ACG",
    "ALPHA_STAR_ACG",
    "LX_OVER_SFR_ACG",
    "F_STAR7_MCG",
    "ALPHA_STAR_MCG",
    "LX_OVER_SFR_MCG",
    "F_STAR7_COPY",
]
ORDERS = {
    "dependencies_first": _DEPENDENCIES_FIRST,
    "dependencies_last": _DEPENDENCIES_FIRST[::-1],
}


@pytest.mark.parametrize("order", ORDERS.values(), ids=ORDERS.keys())
def test_ties_reproduce_the_dependent_defaults_of_21cmfast(order):
    space = _astro_params(order)
    assert space.ndim == 2
    assert len(space.tied) == 4

    theta = space.sample_prior(12, np.random.default_rng(4)).reshape(3, 4, 2)
    model = space.to_model(theta)
    f10, alpha = model["F_STAR10_ACG"], model["ALPHA_STAR_ACG"]

    np.testing.assert_allclose(model["F_STAR7_MCG"], f10 - 3 * alpha)
    np.testing.assert_array_equal(model["ALPHA_STAR_MCG"], alpha)
    np.testing.assert_array_equal(model["LX_OVER_SFR_MCG"], np.full((3, 4), 40.0))
    np.testing.assert_array_equal(model["F_STAR7_COPY"], model["F_STAR7_MCG"])


def test_ties_relate_model_values_not_sampled_values():
    # 'b' is sampled as log10_b; a parameter tied to 'b' receives 10**log10_b.
    space = ParamSpace(
        [
            Param("log10_b", prior=(-1.0, 1.0), feeds="b", transform="pow10"),
            Param("c", tied="b"),
        ]
    )
    model = space.to_model(np.array([[0.5], [-0.5]]))
    np.testing.assert_allclose(model["c"], 10 ** np.array([0.5, -0.5]))


def test_tied_parameters_do_not_change_the_prior():
    tied = _astro_params(ORDERS["dependencies_first"])
    untied = ParamSpace(tied.sampled)
    theta = tied.sample_prior(20, np.random.default_rng(5))
    np.testing.assert_array_equal(tied.log_prior(theta), untied.log_prior(theta))


def test_ties_are_immutable_and_hashable():
    kwargs = {"offset": 1.0, "coefficients": [2.0]}
    tie = Tie("linear", ("x",), kwargs)
    kwargs["offset"] = 5.0  # changing the caller's dict leaves the tie unchanged
    same = Tie("linear", ("x",), {"coefficients": (2.0,), "offset": 1.0})
    assert tie == same
    assert hash(tie) == hash(same)
    np.testing.assert_allclose(tie({"x": np.array([0.25])}), [1.5])


@pytest.mark.parametrize("batch", [(), (5,), (2, 3)])
def test_vector_elements_are_assembled_into_one_model_parameter(batch):
    space = ParamSpace(
        [
            *param_vector("p", 4, prior=(-10.0, 10.0), fixed={1: 0.0}),
            Param("x", prior=(0, 1)),
        ]
    )
    assert space.names == ("p_0", "p_2", "p_3", "x")
    theta = space.sample_prior(int(np.prod(batch)), np.random.default_rng(6))
    theta = theta.reshape((*batch, space.ndim))
    model = space.to_model(theta)
    assert model["p"].shape == (*batch, 4)
    np.testing.assert_array_equal(model["p"][..., [0, 2, 3]], theta[..., :3])
    np.testing.assert_array_equal(model["p"][..., 1], np.zeros(batch))
    np.testing.assert_array_equal(model["x"], theta[..., 3])


@pytest.mark.parametrize(
    ("params", "match"),
    [
        (
            [
                Param("p_0", prior=(0, 1), feeds="p", index=0),
                Param("p_2", prior=(0, 1), feeds="p", index=2),
            ],
            r"indices 0 to 1, each once; got \[0, 2\]",
        ),
        (
            [Param("p", prior=(0, 1)), Param("p_0", prior=(0, 1), feeds="p", index=0)],
            "both whole and element by element",
        ),
    ],
    ids=["missing_element", "whole_and_elements"],
)
def test_inconsistent_vectors_are_rejected(params, match):
    with pytest.raises(ValueError, match=match):
        ParamSpace(params)


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        (("p",), {"prior": (0, 1)}, "length of vector 'p' must be given"),
        (("p", 3), {"prior": [(0, 1), (0, 1)]}, "consistent"),
        (("p", 2), {}, "needs a prior or a fixed value"),
        (
            ("p",),
            {"prior": priors.multivariate_normal([0, 0], np.eye(2)), "fixed": {0: 1}},
            "fix none",
        ),
    ],
    ids=["no_length", "inconsistent_length", "no_prior", "joint_and_fixed"],
)
def test_param_vector_rejects_inconsistent_arguments(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        param_vector(*args, **kwargs)


COV = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, -0.5], [0.0, -0.5, 2.0]])


def test_a_joint_prior_couples_its_parameters_wherever_they_are_listed():
    mvn = priors.multivariate_normal([1.0, 2.0, 3.0], COV)
    b0, b1, b2 = param_vector("beta", prior=mvn)
    x = Param("x", prior=priors.normal(0.0, 1.0))
    space = ParamSpace([b2, x, b0, b1])
    np.testing.assert_allclose(space.fiducial, [3.0, 0.0, 1.0, 2.0])

    n = 40_000
    draws = space.sample_prior(n, np.random.default_rng(7))
    beta, xs = draws[:, [2, 3, 0]], draws[:, 1]
    np.testing.assert_allclose(np.cov(beta.T), COV, atol=5 * np.sqrt(2 / n) * 2.0)
    # x is independent of the vector.
    corr = [np.corrcoef(xs, beta[:, i])[0, 1] for i in range(3)]
    assert np.all(np.abs(corr) < 5 / np.sqrt(n))

    theta = draws[:10]
    expected = mvn.logpdf(theta[:, [2, 3, 0]]) + stats.norm.logpdf(theta[:, 1])
    np.testing.assert_allclose(space.log_prior(theta), expected)


@pytest.mark.parametrize(
    ("which", "match"),
    [("subset", "all of them must be sampled"), ("repeat", "Two parameters take")],
)
def test_joint_priors_must_be_used_whole_and_once(which, match):
    mvn = priors.multivariate_normal([0.0, 0.0, 0.0], COV)
    params = list(param_vector("beta", prior=mvn))
    if which == "subset":
        params = params[:2]
    else:
        params.append(Param("extra", prior=Marginal(mvn, 0)))
    with pytest.raises(ValueError, match=match):
        ParamSpace(params)
