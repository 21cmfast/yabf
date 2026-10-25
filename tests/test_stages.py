"""Tests of the stage protocol and the resolver."""

import copy
import itertools
from typing import ClassVar

import attrs
import numpy as np
import pytest

from yabf import priors
from yabf.params import Param, ParamSpace, param_vector
from yabf.stages import (
    REQUIRED,
    All,
    GraphError,
    ParamSpec,
    Slot,
    StageError,
    call_stage,
    resolve,
)


@attrs.frozen(eq=False)
class Field:
    values: np.ndarray


class LightconeField(Field):
    """A more specific field, which can feed anything that requires a Field."""


@attrs.frozen
class Variance:
    value: float


@attrs.frozen
class Simulator:
    """A white-noise field with standard deviation ``sigma``, on a fixed realisation."""

    name: str = "sim"
    tag: str = ""
    output: type = Field
    consumes: ClassVar = {
        "sigma": ParamSpec(REQUIRED, domain=(0, 10), description="Standard deviation."),
        "seed": ParamSpec(0, description="Seed of the realisation."),
    }
    requires: ClassVar = {}

    @property
    def provides(self):
        return {"field": Slot(self.output, self.tag)}

    def compute(self, params):
        noise = np.random.default_rng(int(params["seed"])).standard_normal(4096)
        return {"field": self.output(params["sigma"] * noise)}


@attrs.frozen
class Measure:
    """The variance of a field."""

    name: str = "measure"
    tag: str = ""
    slot: str = "field"
    consumes: ClassVar = {}
    provides: ClassVar = {"var": Variance}

    @property
    def requires(self):
        return {self.slot: Slot(Field, self.tag)}

    def compute(self, params, **inputs):
        (field,) = inputs.values()
        return {"var": Variance(float(np.var(field.values)))}


@attrs.frozen
class Likelihood:
    """A Gaussian likelihood for the measured variance."""

    name: str = "like"
    consumes: ClassVar = {"noise": ParamSpec(0.0, domain=(0, np.inf))}
    requires: ClassVar = {"var": Variance}
    provides: ClassVar = {"lnl": float}

    def compute(self, params, var):
        return {"lnl": -0.5 * (var.value + params["noise"] - 1.0) ** 2}


def _run(graph, params):
    """Compute every stage of a resolved graph in order."""
    outputs = {}
    for stage in graph.stages:
        out = call_stage(stage, params, graph.gather(stage.name, outputs))
        outputs.update({(stage.name, slot): value for slot, value in out.items()})
    return outputs


TOY = (Simulator(), Measure(), Likelihood())


@pytest.mark.parametrize(
    "stages",
    list(itertools.permutations(TOY)),
    ids=lambda s: "-".join(x.name for x in s),
)
def test_stages_are_ordered_after_their_providers_whatever_the_input_order(stages):
    graph = resolve(stages)
    assert [s.name for s in graph.stages] == ["sim", "measure", "like"]
    assert graph.inputs_of("measure") == {"field": (("sim", "field"),)}
    assert graph.inputs_of("like") == {"var": (("measure", "var"),)}


def test_the_toy_chain_recovers_the_variance_of_its_field():
    sigma = 2.0
    out = _run(resolve(TOY), {"sigma": sigma, "seed": 1})
    # The sample variance of n white-noise points has relative error sqrt(2/n).
    n = len(out["sim", "field"].values)
    z = (out["measure", "var"].value - sigma**2) / (sigma**2 * np.sqrt(2 / n))
    assert abs(z) < 5


def _state(x):
    return attrs.asdict(x) if attrs.has(type(x)) else x


@pytest.mark.parametrize("stage", TOY, ids=lambda s: s.name)
def test_toy_stages_obey_the_stage_contract(stage):
    params = {"sigma": 1.5, "seed": 3, "noise": 0.1, "unrelated": 7.0}
    inputs = {
        "measure": {"field": Field(np.arange(10.0))},
        "like": {"var": Variance(2.0)},
    }.get(stage.name, {})
    before = copy.deepcopy(inputs)

    first = call_stage(stage, params, inputs)
    second = call_stage(stage, params, inputs)

    # call_stage has checked the outputs against `provides`; the stage must also be
    # deterministic and leave its inputs as they were.
    for k in first:
        np.testing.assert_equal(_state(first[k]), _state(second[k]))
    for k, v in inputs.items():
        np.testing.assert_equal(_state(v), _state(before[k]))


@attrs.frozen
class Spy:
    seen: list = attrs.field(factory=list, eq=False)
    name: str = "spy"
    consumes: ClassVar = {"a": ParamSpec(REQUIRED), "b": ParamSpec(2.0)}
    requires: ClassVar = {}
    provides: ClassVar = {}

    def compute(self, params):
        self.seen.append(dict(params))
        return {}


def test_a_stage_receives_what_it_consumes_with_defaults_for_the_rest():
    spy = Spy()
    call_stage(spy, {"a": 1.0, "c": 3.0}, {})
    assert spy.seen == [{"a": 1.0, "b": 2.0}]


def test_a_missing_required_parameter_is_reported_before_computing():
    spy = Spy()
    with pytest.raises(StageError, match=r"requires the parameters \['a'\]"):
        call_stage(spy, {"b": 1.0}, {})
    assert spy.seen == []


@attrs.frozen
class Broken:
    name: str = "broken"
    returns: object = None
    consumes: ClassVar = {}
    requires: ClassVar = {}
    provides: ClassVar = {"var": Variance}

    def compute(self, params):
        return self.returns


@pytest.mark.parametrize(
    ("returns", "match"),
    [
        ({}, r"returned \[\], but provides \['var'\]"),
        ({"var": Variance(1.0), "extra": 1.0}, r"returned \['extra', 'var'\]"),
        ({"var": 1.0}, "returned a float for slot 'var', which provides Variance"),
        ({"var": Variance(1.0), 1: 1.0}, r"returned \[1, 'var'\]"),
        (Variance(1.0), "returned Variance"),
    ],
    ids=["missing", "extra", "wrong_type", "mixed_key_types", "not_a_mapping"],
)
def test_a_stage_that_breaks_its_contract_is_reported(returns, match):
    with pytest.raises(StageError, match=match):
        call_stage(Broken(returns=returns), {}, {})


def test_tags_pick_between_outputs_of_the_same_type():
    stages = [
        Simulator(name="sim_a", tag="a"),
        Simulator(name="sim_b", tag="b"),
        Measure(name="measure_b", tag="b"),
    ]
    graph = resolve(stages)
    assert graph.inputs_of("measure_b") == {"field": (("sim_b", "field"),)}


def test_a_subclass_satisfies_a_requirement_for_its_base_type():
    graph = resolve([Simulator(output=LightconeField), Measure()])
    assert graph.inputs_of("measure") == {"field": (("sim", "field"),)}


def _problems(stages, params=None):
    with pytest.raises(GraphError) as e:
        resolve(stages, params)
    return e.value.problems


@pytest.mark.parametrize(
    ("stages", "expected"),
    [
        (
            [Measure()],
            "Stage 'measure' requires Field (slot 'field'), but no stage provides it.",
        ),
        (
            [Simulator(tag="raw"), Measure()],
            (
                "Stage 'measure' requires Field (slot 'field'), but no stage provides "
                "it ('sim' provides Field['raw'])."
            ),
        ),
        (
            [Simulator(name="sim_a"), Simulator(name="sim_b"), Measure()],
            (
                "Stage 'measure' requires Field (slot 'field'), which is provided by "
                "more than one stage: 'sim_a' (slot 'field'), 'sim_b' (slot 'field'). "
                "Give the outputs distinct tags."
            ),
        ),
        (
            [Simulator(), Simulator(), Measure()],
            "Several stages are named ['sim']; stage names must be unique.",
        ),
        (
            [Measure(name="params_slot", slot="params")],
            (
                "Stage 'params_slot' requires a slot named 'params', which is reserved "
                "for the parameters passed to compute(); rename the slot."
            ),
        ),
    ],
    ids=[
        "missing",
        "missing_with_other_tag",
        "ambiguous",
        "duplicate_names",
        "reserved_slot_name",
    ],
)
def test_unresolvable_graphs_are_reported_with_readable_messages(stages, expected):
    assert expected in _problems(stages)


def test_a_cycle_is_reported():
    @attrs.frozen
    class Feedback:
        """Requires the variance and provides the field it is measured from."""

        name: str = "feedback"
        consumes: ClassVar = {}
        requires: ClassVar = {"var": Variance}
        provides: ClassVar = {"field": Field}

        def compute(self, params, var):
            return {"field": Field(np.full(4, var.value))}

    (problem,) = _problems([Measure(), Feedback()])
    assert problem.startswith("The stages form a cycle: ")
    assert "measure -> feedback" in problem or "feedback -> measure" in problem


def test_parameters_that_no_stage_consumes_are_reported():
    params = ParamSpace(
        [
            Param("sigma", prior=(0.5, 2.0)),
            Param("F_STAR10", prior=(-3.0, 0.0)),  # a typo or a deprecated name
            Param("log10_x", prior=(0.0, 1.0), feeds="x", transform="pow10"),
        ]
    )
    (problem,) = _problems(TOY, params)
    assert problem == (
        "No stage consumes the parameters 'F_STAR10', 'log10_x' (feeds 'x')."
    )


def test_a_parameter_used_through_a_tie_counts_as_consumed():
    # 'base' reaches the simulator only as the input of the tied 'sigma'.
    params = ParamSpace(
        [Param("base", prior=(0.5, 2.0)), Param("sigma", tied={"base": 2.0})]
    )
    resolve(TOY, params)


def test_every_problem_is_reported_at_once():
    params = ParamSpace([Param("unused", fixed=1.0)])
    stages = [Measure(name="m1"), Measure(name="m2"), Likelihood(), Likelihood()]
    problems = "\n".join(_problems(stages, params))
    for expected in [
        "named ['like']",
        "Stage 'm1' requires Field (slot 'field'), but no stage provides it",
        "Stage 'm2' requires Field (slot 'field'), but no stage provides it",
        "Stage 'like' requires Variance (slot 'var'), which is provided by more",
        "'unused'",
    ]:
        assert expected in problems


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"default": 2.0, "domain": (0, 1)}, r"default 2 is outside \[0, 1\]"),
        ({"default": 0.0, "shape": (3,)}, r"has shape \(\), not \(3,\)"),
        ({"domain": (1, 0)}, "low <= high"),
    ],
    ids=["default_outside_domain", "default_wrong_shape", "empty_domain"],
)
def test_inconsistent_param_specs_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ParamSpec(**kwargs)


def test_consumes_must_describe_each_parameter():
    @attrs.frozen
    class Bare:
        name: str = "bare"
        consumes: ClassVar = frozenset({"x"})
        requires: ClassVar = {}
        provides: ClassVar = {}

        def compute(self, params):
            return {}

    with pytest.raises(TypeError, match="mapping from parameter names to ParamSpec"):
        resolve([Bare()])


@attrs.frozen
class Bubbles:
    """Bubbles of radius up to ``r_max`` in a box of side ``box``, at most a third."""

    name: str = "bubbles"
    consumes: ClassVar = {
        "box": ParamSpec(100.0, domain=(0, np.inf)),
        "r_max": ParamSpec(15.0, domain=(0, np.inf)),
        "weights": ParamSpec((1.0, 1.0), shape=(2,)),
    }
    requires: ClassVar = {}
    provides: ClassVar = {}

    def check(self, params):
        if params["r_max"] > params["box"] / 3:
            raise ValueError("r_max must be at most a third of the box.")

    def compute(self, params):
        return {}


@pytest.mark.parametrize(
    ("stages", "params", "expected"),
    [
        (
            [Simulator()],
            [Param("seed", fixed=1)],
            (
                "Stage 'sim' requires the parameter 'sigma', which the parameter space "
                "does not define."
            ),
        ),
        (
            [Simulator()],
            [Param("sigma", fixed=-1.0)],
            (
                "Stage 'sim' rejects 'sigma' at the fiducial point: the value -1 is "
                "outside [0, 10]."
            ),
        ),
        (
            [Simulator()],
            [Param("sigma", prior=(5.0, 20.0))],
            (
                "Stage 'sim' rejects 'sigma' at the upper edge of the prior of "
                "'sigma': the value 20 is outside [0, 10]."
            ),
        ),
        (
            # A normal prior has infinite support, so it cannot fit a bounded domain.
            [Simulator()],
            [Param("sigma", prior=priors.normal(5.0, 1.0))],
            (
                "Stage 'sim' rejects 'sigma' at the lower edge of the prior of "
                "'sigma': the value -inf is outside [0, 10]."
            ),
        ),
        (
            # Domains apply to model values, after the transform.
            [Simulator()],
            [Param("log10_sigma", prior=(-1.0, 2.0), feeds="sigma", transform="pow10")],
            (
                "Stage 'sim' rejects 'sigma' at the upper edge of the prior of "
                "'log10_sigma': the value 100 is outside [0, 10]."
            ),
        ),
        (
            # Tied values are checked at points near the edges of their inputs' priors.
            [Simulator()],
            [Param("base", prior=(1.0, 4.0)), Param("sigma", tied={"base": 3.0})],
            "Stage 'sim' rejects 'sigma' with 'base' near the upper edge of its prior",
        ),
        (
            [Bubbles()],
            [Param("weights", fixed=1.0)],
            (
                "Stage 'bubbles' rejects 'weights' at the fiducial point: the value "
                "1 has shape (), not (2,)."
            ),
        ),
        (
            # The default box (100) is used, and r_max reaches past a third of it.
            [Bubbles()],
            [Param("r_max", prior=(5.0, 50.0))],
            (
                "Stage 'bubbles' rejects the parameters with 'r_max' near the upper "
                "edge of its prior: r_max must be at most a third of the box."
            ),
        ),
    ],
    ids=[
        "required_missing",
        "fixed_outside_domain",
        "prior_outside_domain",
        "unbounded_prior",
        "transformed_prior",
        "tied_outside_domain",
        "wrong_shape",
        "check_hook",
    ],
)
def test_parameters_a_stage_does_not_accept_are_reported(stages, params, expected):
    problems = _problems(stages, ParamSpace(params))
    assert any(p.startswith(expected) for p in problems), problems


@pytest.mark.parametrize(
    "params",
    [
        [Param("sigma", prior=(0.5, 10.0)), Param("seed", fixed=2)],
        [Param("log10_sigma", prior=(-1.0, 1.0), feeds="sigma", transform="pow10")],
        [Param("base", prior=(1.0, 3.0)), Param("sigma", tied={"base": 3.0})],
    ],
    ids=["bounded_prior", "transformed_prior", "tied"],
)
def test_parameter_spaces_inside_the_domains_resolve(params):
    resolve([Simulator()], ParamSpace(params))


def test_the_check_hook_passes_a_consistent_space():
    resolve([Bubbles()], ParamSpace([Param("r_max", prior=(5.0, 30.0))]))


FREQS = np.linspace(50.0, 100.0, 51)


@attrs.frozen(eq=False)
class Spectrum:
    values: np.ndarray


@attrs.frozen
class PowerLaw:
    """A power-law spectral component, with an amplitude parameter of its own."""

    name: str
    index: float = -2.5
    tag: str = "model"

    @property
    def consumes(self):
        return {f"{self.name}_amp": ParamSpec(REQUIRED)}

    requires: ClassVar = {}

    @property
    def provides(self):
        return {"spectrum": Slot(Spectrum, self.tag)}

    def compute(self, params):
        amp = params[f"{self.name}_amp"]
        return {"spectrum": Spectrum(amp * (FREQS / 75.0) ** self.index)}


@attrs.frozen
class Residual:
    """The data minus every model component, whichever stages provide them."""

    data: np.ndarray = attrs.field(eq=False)
    min_parts: int = 1
    name: str = "residual"
    consumes: ClassVar = {}
    provides: ClassVar = {"residual": Slot(Spectrum, "residual")}

    @property
    def requires(self):
        return {"parts": All(Spectrum, "model", min=self.min_parts)}

    def compute(self, params, parts):
        return {"residual": Spectrum(self.data - sum(p.values for p in parts))}


@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_a_residual_subtracts_every_model_component_without_naming_them(
    n_components,
):
    components = [PowerLaw(f"c{i}", index=-2.5 + 0.5 * i) for i in range(n_components)]
    params = {f"c{i}_amp": 100.0 * (i + 1) for i in range(n_components)}
    noise = np.random.default_rng(5).normal(0, 0.01, FREQS.size)
    model = sum(params[f"{c.name}_amp"] * (FREQS / 75) ** c.index for c in components)

    stages = [Residual(model + noise), *components]
    graph = resolve(stages)
    assert graph.inputs_of("residual") == {
        "parts": tuple((c.name, "spectrum") for c in components)
    }
    out = _run(graph, params)
    np.testing.assert_allclose(out["residual", "residual"].values, noise, atol=1e-10)


def test_a_stage_does_not_collect_its_own_outputs():
    @attrs.frozen
    class Total:
        """Sums the model components, and is itself a model component."""

        name: str = "total"
        consumes: ClassVar = {}
        requires: ClassVar = {"parts": All(Spectrum, "model")}
        provides: ClassVar = {"total": Slot(Spectrum, "model")}

        def compute(self, params, parts):
            return {"total": Spectrum(sum(p.values for p in parts))}

    graph = resolve([Total(), PowerLaw("fg")])
    assert graph.inputs_of("total") == {"parts": (("fg", "spectrum"),)}


@pytest.mark.parametrize(
    ("min_parts", "n_components", "expected"),
    [
        (
            1,
            0,
            (
                "Stage 'residual' requires every Spectrum['model'] (slot 'parts'), at "
                "least 1, but no stage provides one."
            ),
        ),
        (
            3,
            2,
            (
                "Stage 'residual' requires every Spectrum['model'] (slot 'parts'), at "
                "least 3, but 2 stages provide one."
            ),
        ),
    ],
    ids=["none", "too_few"],
)
def test_too_few_matches_for_all_are_reported(min_parts, n_components, expected):
    components = [PowerLaw(f"c{i}") for i in range(n_components)]
    problems = _problems([Residual(FREQS, min_parts=min_parts), *components])
    assert expected in problems


def test_all_with_min_zero_accepts_no_matches():
    out = _run(resolve([Residual(FREQS, min_parts=0)]), {})
    np.testing.assert_array_equal(out["residual", "residual"].values, FREQS)


@attrs.frozen
class Polynomial:
    """A polynomial with ``n`` coefficients, each between -100 and 100.

    The class works for any number of terms; each instance fixes its own.
    """

    n: int = 4
    name: str = "poly"
    requires: ClassVar = {}
    provides: ClassVar = {"spectrum": Slot(Spectrum, "model")}

    @property
    def consumes(self):
        return {"p": ParamSpec(REQUIRED, shape=(self.n,), domain=(-100, 100))}

    def compute(self, params):
        return {"spectrum": Spectrum(np.polyval(params["p"][::-1], FREQS / 75))}


def test_a_vector_of_elements_feeds_a_vector_parameter():
    params = ParamSpace(param_vector("p", 4, prior=(-10.0, 10.0), fixed={1: 0.0}))
    out = _run(resolve([Polynomial()], params), params.to_model(params.fiducial))
    expected = np.polyval([0.0, 0.0, 0.0, 0.0], FREQS / 75)
    np.testing.assert_allclose(out["poly", "spectrum"].values, expected)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (
            param_vector("p", 3, prior=(-10.0, 10.0)),
            (
                "Stage 'poly' rejects 'p' at the fiducial point: the value [0. 0. 0.] "
                "has shape (3,), not (4,)."
            ),
        ),
        (
            param_vector("p", 4, prior=[(-10, 10), (-10, 10), (-10, 200), (-10, 10)]),
            (
                "Stage 'poly' rejects 'p' at the upper edge of the prior of 'p_2': the "
                "value 200 is outside [-100, 100]."
            ),
        ),
        (
            # A multivariate normal has infinite support, so its elements leave the
            # bounded domain.
            param_vector("p", prior=priors.multivariate_normal(np.zeros(4), np.eye(4))),
            (
                "Stage 'poly' rejects 'p' at the lower edge of the prior of 'p_0': the "
                "value -inf is outside [-100, 100]."
            ),
        ),
    ],
    ids=["wrong_length", "element_prior_outside_domain", "joint_prior_unbounded"],
)
def test_vectors_a_stage_does_not_accept_are_reported(params, expected):
    assert expected in _problems([Polynomial()], ParamSpace(params))


@pytest.mark.parametrize("n", [2, 5])
def test_each_instance_of_a_stage_sets_the_length_of_its_vectors(n):
    params = ParamSpace(param_vector("p", n, prior=(-10.0, 10.0)))
    resolve([Polynomial(n)], params)
    other = Polynomial(n + 1)
    (problem,) = _problems([other], params)
    assert problem.endswith(f"has shape ({n},), not ({n + 1},).")
