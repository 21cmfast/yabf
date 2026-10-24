"""Tests of the stage protocol and the resolver."""

import copy
import itertools
from typing import ClassVar

import attrs
import numpy as np
import pytest

from yabf.params import Param, ParamSpace
from yabf.stages import GraphError, Slot, StageError, call_stage, resolve


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
    consumes: ClassVar = frozenset({"sigma", "seed"})
    requires: ClassVar = {}

    @property
    def provides(self):
        return {"field": Slot(self.output, self.tag)}

    def compute(self, params):
        noise = np.random.default_rng(params.get("seed", 0)).standard_normal(4096)
        return {"field": self.output(params.get("sigma", 1.0) * noise)}


@attrs.frozen
class Measure:
    """The variance of a field."""

    name: str = "measure"
    tag: str = ""
    consumes: ClassVar = frozenset()
    provides: ClassVar = {"var": Variance}

    @property
    def requires(self):
        return {"field": Slot(Field, self.tag)}

    def compute(self, params, field):
        return {"var": Variance(float(np.var(field.values)))}


@attrs.frozen
class Likelihood:
    """A Gaussian likelihood for the measured variance."""

    name: str = "like"
    consumes: ClassVar = frozenset({"noise"})
    requires: ClassVar = {"var": Variance}
    provides: ClassVar = {"lnl": float}

    def compute(self, params, var):
        return {"lnl": -0.5 * (var.value + params.get("noise", 0.0) - 1.0) ** 2}


def _run(graph, params):
    """Compute every stage of a resolved graph in order."""
    outputs = {}
    for stage in graph.stages:
        inputs = {
            slot: outputs[provider][provided]
            for slot, (provider, provided) in graph.inputs_of(stage.name).items()
        }
        outputs[stage.name] = call_stage(stage, params, inputs)
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
    assert graph.inputs_of("measure") == {"field": ("sim", "field")}
    assert graph.inputs_of("like") == {"var": ("measure", "var")}


def test_the_toy_chain_recovers_the_variance_of_its_field():
    sigma = 2.0
    out = _run(resolve(TOY), {"sigma": sigma, "seed": 1})
    # The sample variance of n white-noise points has relative error sqrt(2/n).
    n = len(out["sim"]["field"].values)
    z = (out["measure"]["var"].value - sigma**2) / (sigma**2 * np.sqrt(2 / n))
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


def test_a_stage_receives_only_the_parameters_it_consumes():
    seen = {}

    @attrs.frozen
    class Spy:
        name: str = "spy"
        consumes: ClassVar = frozenset({"a", "b"})
        requires: ClassVar = {}
        provides: ClassVar = {}

        def compute(self, params):
            seen.update(params)
            return {}

    call_stage(Spy(), {"a": 1.0, "c": 3.0}, {})
    assert seen == {"a": 1.0}


@attrs.frozen
class Broken:
    name: str = "broken"
    returns: object = None
    consumes: ClassVar = frozenset()
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
        (Variance(1.0), "returned Variance"),
    ],
    ids=["missing", "extra", "wrong_type", "not_a_mapping"],
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
    assert graph.inputs_of("measure_b") == {"field": ("sim_b", "field")}


def test_a_subclass_satisfies_a_requirement_for_its_base_type():
    graph = resolve([Simulator(output=LightconeField), Measure()])
    assert graph.inputs_of("measure") == {"field": ("sim", "field")}


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
            "Stage 'measure' requires Field (slot 'field'), but no stage provides it "
            "('sim' provides Field['raw']).",
        ),
        (
            [Simulator(name="sim_a"), Simulator(name="sim_b"), Measure()],
            "Stage 'measure' requires Field (slot 'field'), which is provided by more "
            "than one stage: 'sim_a' (slot 'field'), 'sim_b' (slot 'field'). Give the "
            "outputs distinct tags.",
        ),
        (
            [Simulator(), Simulator(), Measure()],
            "Several stages are named ['sim']; stage names must be unique.",
        ),
    ],
    ids=["missing", "missing_with_other_tag", "ambiguous", "duplicate_names"],
)
def test_unresolvable_graphs_are_reported_with_readable_messages(stages, expected):
    assert expected in _problems(stages)


def test_a_cycle_is_reported():
    @attrs.frozen
    class Feedback:
        """Requires the variance and provides the field it is measured from."""

        name: str = "feedback"
        consumes: ClassVar = frozenset()
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
    for kind in ["named ['like']", "no stage provides", "more than one", "'unused'"]:
        assert kind in problems
