"""Tests of pipeline evaluation: sharing, early release and caching."""

import weakref
from typing import ClassVar

import attrs
import numpy as np
import pytest

from yabf.params import Param, ParamSpace
from yabf.pipeline import Pipeline
from yabf.stages import REQUIRED, All, ParamSpec, Slot


@attrs.frozen(eq=False)
class Field:
    """A large simulation product."""

    values: np.ndarray


@attrs.frozen
class Variance:
    value: float


@attrs.frozen
class Simulator:
    """A white-noise field; keeps a weak reference to every field it makes."""

    made: list = attrs.field(factory=list, eq=False)
    name: str = "sim"
    consumes: ClassVar = {
        "sigma": ParamSpec(REQUIRED, domain=(0, np.inf)),
        "seed": ParamSpec(0),
    }
    requires: ClassVar = {}
    provides: ClassVar = {"field": Field}

    def compute(self, params):
        noise = np.random.default_rng(int(params["seed"])).standard_normal(10**5)
        field = Field(params["sigma"] * noise)
        self.made.append(weakref.ref(field))
        return {"field": field}


@attrs.frozen
class Measure:
    """The variance of the field: a small observable, cached across evaluations."""

    calls: list = attrs.field(factory=list, eq=False)
    name: str = "measure"
    cacheable: ClassVar = True
    consumes: ClassVar = {}
    requires: ClassVar = {"field": Field}
    provides: ClassVar = {"var": Variance}

    def compute(self, params, field):
        self.calls.append(weakref.ref(field))
        return {"var": Variance(float(np.var(field.values)))}


@attrs.frozen
class Instrument:
    """A fractional model error applied to the observable."""

    name: str = "instrument"
    consumes: ClassVar = {"model_error": ParamSpec(0.0, domain=(0, np.inf))}
    requires: ClassVar = {"var": Variance}
    provides: ClassVar = {"var": Slot(Variance, "observed")}

    def compute(self, params, var):
        return {"var": Variance(var.value * (1 + params["model_error"]))}


@attrs.frozen
class Likelihood:
    """A Gaussian likelihood for an observed variance of one."""

    name: str = "like"
    consumes: ClassVar = {}
    requires: ClassVar = {"var": Slot(Variance, "observed")}
    provides: ClassVar = {"lnl": float}

    def compute(self, params, var):
        return {"lnl": -0.5 * (var.value - 1.0) ** 2 / 0.01}


@attrs.frozen
class FieldsFreed:
    """Records, when it runs, whether every field the simulator made has been freed."""

    sim: Simulator
    seen: list = attrs.field(factory=list, eq=False)
    name: str = "probe"
    consumes: ClassVar = {}
    requires: ClassVar = {"var": Slot(Variance, "observed")}
    provides: ClassVar = {"ok": bool}

    def compute(self, params, var):
        self.seen.append(all(ref() is None for ref in self.sim.made))
        return {"ok": True}


PARAMS = {"sigma": 1.0, "seed": 1, "model_error": 0.0}


def _chain(*likelihoods, cache_size=128):
    sim, measure = Simulator(), Measure()
    stages = [sim, measure, Instrument(), *(likelihoods or [Likelihood()])]
    return Pipeline.build(stages, cache_size=cache_size), sim, measure


def test_two_likelihoods_share_one_simulation():
    pipe, sim, _ = _chain(Likelihood("like_a"), Likelihood("like_b"), cache_size=0)
    out = pipe(PARAMS)
    assert len(sim.made) == 1
    assert out["like_a", "lnl"] == out["like_b", "lnl"]


def test_outputs_are_passed_by_reference_not_copied():
    pipe, sim, measure = _chain()
    out = pipe(PARAMS, want=[("sim", "field"), ("like", "lnl")])
    made = sim.made[0]()
    assert out["sim", "field"] is made
    assert measure.calls[0]() is made


def test_the_field_is_freed_once_its_last_consumer_has_run():
    sim = Simulator()
    probe = FieldsFreed(sim)
    pipe = Pipeline.build([sim, Measure(), Instrument(), probe])
    pipe(PARAMS)
    # The probe runs after the measurement, the field's only consumer.
    assert probe.seen == [True]


def test_a_wanted_output_is_kept():
    pipe, sim, _ = _chain()
    out = pipe(PARAMS, want=[("sim", "field"), ("like", "lnl")])
    assert sim.made[0]() is out["sim", "field"]


def test_only_the_stages_an_output_depends_on_are_run():
    pipe, sim, measure = _chain()
    pipe(PARAMS, want=[("sim", "field")])
    assert len(sim.made) == 1
    assert measure.calls == []


@pytest.mark.parametrize(
    ("change", "simulations"),
    [
        ({}, 1),
        ({"model_error": 0.2}, 1),  # instrument only: the cached observable is reused
        ({"sigma": 2.0}, 2),
        ({"seed": 2}, 2),
    ],
    ids=["nothing", "instrument_only", "simulator_parameter", "seed"],
)
def test_only_changes_upstream_of_the_cached_observable_re_simulate(
    change, simulations
):
    pipe, sim, _ = _chain()
    pipe(PARAMS)
    out = pipe(PARAMS | change)
    assert len(sim.made) == simulations
    fresh, _, _ = _chain(cache_size=0)
    assert out == fresh(PARAMS | change)


def test_cached_results_equal_freshly_computed_ones():
    cached, _, _ = _chain()
    cached(PARAMS)
    params = PARAMS | {"model_error": 0.2}
    fresh, _, _ = _chain(cache_size=0)
    assert cached(params) == fresh(params)


def test_simulator_products_are_never_cached():
    pipe, sim, _ = _chain()
    pipe(PARAMS, want=[("sim", "field")])
    pipe(PARAMS, want=[("sim", "field")])
    assert len(sim.made) == 2


@pytest.mark.parametrize(("cache_size", "simulations"), [(0, 4), (1, 4), (2, 2)])
def test_the_cache_holds_at_most_cache_size_results(cache_size, simulations):
    pipe, sim, _ = _chain(cache_size=cache_size)
    for sigma in [1.0, 2.0, 1.0, 2.0]:
        pipe(PARAMS | {"sigma": sigma})
    assert len(sim.made) == simulations


def test_cache_keys_accept_values_from_a_parameter_space():
    # ParamSpace.to_model returns fixed values as zero-dimensional arrays.
    space = ParamSpace(
        [
            Param("sigma", prior=(0.5, 2.0)),
            Param("seed", fixed=3),
            Param("model_error", prior=(0.0, 0.5)),
        ]
    )
    pipe, sim, _ = _chain()
    fresh, _, _ = _chain(cache_size=0)
    points = [[1.2, 0.1], [1.2, 0.3], [1.5, 0.3], [1.2, 0.3]]
    for theta in points:
        values = space.to_model(np.array(theta))
        assert pipe(values) == fresh(values)
    # Re-simulated for each new sigma, but not when only model_error changes, nor
    # when an earlier point comes back.
    assert len(sim.made) == 2


def test_unknown_outputs_are_rejected():
    pipe, _, _ = _chain()
    with pytest.raises(ValueError, match=r"no outputs \[\('like', 'chi2'\)\]"):
        pipe(PARAMS, want=[("like", "chi2")])


FREQS = np.linspace(50.0, 100.0, 1001)


@attrs.frozen(eq=False)
class Spectrum:
    values: np.ndarray


@attrs.frozen
class Component:
    """A cached power-law component; keeps a weak reference to each spectrum."""

    name: str
    index: float = -2.5
    made: list = attrs.field(factory=list, eq=False)
    cacheable: ClassVar = True
    requires: ClassVar = {}
    provides: ClassVar = {"spectrum": Slot(Spectrum, "model")}

    @property
    def consumes(self):
        return {f"{self.name}_amp": ParamSpec(REQUIRED)}

    def compute(self, params):
        spectrum = Spectrum(params[f"{self.name}_amp"] * (FREQS / 75) ** self.index)
        self.made.append(weakref.ref(spectrum))
        return {"spectrum": spectrum}


@attrs.frozen
class Residual:
    """The data minus every model component."""

    data: np.ndarray = attrs.field(eq=False)
    cacheable: bool = False
    name: str = "residual"
    consumes: ClassVar = {}
    requires: ClassVar = {"parts": All(Spectrum, "model")}
    provides: ClassVar = {"chi2": float}

    def compute(self, params, parts):
        return {"chi2": float(np.sum((self.data - sum(p.values for p in parts)) ** 2))}


def _components():
    return [Component("fg", -2.5), Component("bias", 0.0), Component("tilt", 1.0)]


def test_changing_one_component_recomputes_only_that_component():
    # The residual is cached too, so its key must depend on every component.
    components = _components()
    pipe = Pipeline.build([Residual(np.zeros_like(FREQS), True), *components])
    params = {"fg_amp": 1000.0, "bias_amp": 1.0, "tilt_amp": 0.1}
    pipe(params)
    changed = params | {"tilt_amp": 0.2}
    out = pipe(changed)
    assert [len(c.made) for c in components] == [1, 1, 2]

    fresh = Pipeline.build(
        [Residual(np.zeros_like(FREQS)), *_components()], cache_size=0
    )
    assert out == fresh(changed)


def test_every_component_is_freed_once_the_residual_has_used_it():
    components = _components()
    pipe = Pipeline.build([Residual(np.zeros_like(FREQS)), *components], cache_size=0)
    pipe({"fg_amp": 1000.0, "bias_amp": 1.0, "tilt_amp": 0.1})
    assert all(ref() is None for c in components for ref in c.made)
