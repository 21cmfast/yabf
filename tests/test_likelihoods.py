import numpy as np
import pytest

from yabf import Param

from .shared_resources import SimpleComponent, SimpleLikelihood


@pytest.fixture(scope="module")
def inactive_lk():
    return SimpleLikelihood(components=[SimpleComponent(name="cmp")])


@pytest.fixture
def global_lk():
    return SimpleLikelihood(
        components=[SimpleComponent(name="cmp", params=(Param("x", fiducial=1.5),))]
    )


@pytest.fixture(scope="module")
def sub_lk():
    return SimpleLikelihood(
        components=[SimpleComponent(name="cmp", params=(Param("x", fiducial=1.5),))],
        params=(Param("y", fiducial=2),),
    )


def test_likelihood_properties(inactive_lk, global_lk, sub_lk):
    lk = inactive_lk

    assert not lk.in_active_mode
    assert "y" in lk.base_parameter_dct
    assert len(lk.child_base_parameters) == len(lk.child_base_parameter_dct) == 2
    assert len(lk.active_params) == 0
    assert lk.total_active_params == 0
    assert lk.fiducial_params["y"] == 0
    assert lk.logprior() == 0

    assert global_lk.in_active_mode
    assert global_lk.total_active_params == 1
    print(global_lk.child_active_params["x"])
    print(
        global_lk.child_active_params["x"].prior.a,
        global_lk.child_active_params["x"].prior.b,
    )
    assert global_lk.logprior() == np.log(1 / 20)
    assert len(global_lk.fiducial_params) == 2  # one param and one component

    assert sub_lk.in_active_mode
    assert sub_lk.total_active_params == 2
    print(sub_lk.fiducial_params)
    assert sub_lk.logprior() == np.log(1 / 200) + np.log(1 / 20)
    assert len(sub_lk.fiducial_params) == 2  # fidicual only cares about top-level
    assert len(sub_lk.fiducial_params["cmp"]) == 1


def test_generate_refs(inactive_lk, global_lk):
    lk = inactive_lk

    with pytest.raises(AttributeError):
        lk.generate_refs()

    lk = global_lk

    refs = lk.generate_refs()
    assert len(refs) == 1
    refs = lk.generate_refs(squeeze=True)
    assert -10 < refs[0] < 10
    refs = lk.generate_refs(n=10, squeeze=True, params=["x"])
    assert len(refs) == 1
    assert len(refs[0]) == 10


def test_fill_params(inactive_lk, global_lk, sub_lk):
    params = inactive_lk._fill_params()
    assert params["y"] == 0
    assert params["cmp"]["x"] == 0

    params = global_lk._fill_params()
    assert params["y"] == 0

    assert params["cmp"]["x"] == 1.5

    params = sub_lk._fill_params()
    assert params["y"] == 2
    assert params["cmp"]["x"] == 1.5


def test_parameter_list_to_dict(inactive_lk, global_lk, sub_lk):
    with pytest.raises(AttributeError):
        inactive_lk._parameter_list_to_dict([])

    with pytest.raises(ValueError):
        global_lk._parameter_list_to_dict([])

    p = global_lk._parameter_list_to_dict([7])
    assert p["cmp"]["x"] == 7

    p = sub_lk._parameter_list_to_dict([7, 10])
    assert p["y"] == 7
    assert p["cmp"]["x"] == 10
