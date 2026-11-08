"""Tests of the CosmoSIS driver, on posteriors whose answer is known exactly.

The MPI tests launch ``mpirun`` in a subprocess. They are skipped without CosmoSIS,
mpi4py or ``mpirun``.
"""

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import ClassVar

import attrs
import numpy as np
import pytest

from yabf import cosmosis, testing
from yabf.params import Param
from yabf.posterior import LogLikelihood, Posterior
from yabf.stages import ParamSpec

pytest.importorskip("cosmosis")

SIGNAL = ("global_signal", "global_signal")
NAUTILUS = {"n_live": 300, "n_networks": 1, "seed": 1}


def _toy():
    """The toy simulation, returning its global signal as a derived output."""
    analytic = testing.toy_simulation(n_pix=50)
    return analytic, attrs.evolve(analytic.posterior, derived=[SIGNAL])


def _check(samples, analytic, posterior):
    """Assert the samples recover the analytic posterior, with aligned outputs."""
    sd = np.sqrt(np.diag(analytic.truth.covariance))
    z = (samples.mean - analytic.truth.mean) / (sd / np.sqrt(samples.n_effective))
    assert np.all(np.abs(z) < 5), z
    basis = posterior.stages[0].basis
    np.testing.assert_allclose(
        samples.derived[SIGNAL], samples.points @ basis.T, rtol=0, atol=1e-8
    )
    some = samples.points[:: max(len(samples.points) // 20, 1)]
    lnl, _ = posterior.log_likelihood(some)
    np.testing.assert_allclose(
        samples.log_likelihood[:: max(len(samples.points) // 20, 1)], lnl, rtol=1e-8
    )


def test_nautilus_through_cosmosis_recovers_the_posterior(tmp_path):
    """The chain CosmoSIS writes, read back, has the posterior's moments and
    evidence, each sample's derived output, and its log-likelihood."""
    analytic, posterior = _toy()
    samples = cosmosis.run(posterior, "nautilus", tmp_path, options=NAUTILUS)
    assert samples.finished
    assert samples.sampler == "nautilus"
    _check(samples, analytic, posterior)
    error = 1 / np.sqrt(samples.n_effective)
    assert abs(samples.log_evidence - analytic.log_evidence) < 5 * error


def test_multinest_through_cosmosis_recovers_two_modes(tmp_path):
    """MultiNest, with importance nested sampling off by default, finds both modes of
    the bimodal posterior, with their weights, and its evidence."""
    analytic = testing.bimodal()
    samples = cosmosis.run(
        analytic.posterior,
        "multinest",
        tmp_path,
        options={"live_points": 400, "random_seed": 2},
    )
    upper = samples.weights[samples.points[:, 0] > 0].sum()
    n = samples.n_effective
    assert abs(upper - 0.7) < 5 * np.sqrt(0.21 / n)
    assert abs(samples.log_evidence - analytic.log_evidence) < 0.3


def test_a_dry_run_computes_writes_and_reads_back(tmp_path):
    _, posterior = _toy()
    report = cosmosis.dry_run(
        posterior, np.random.default_rng(0), n_draws=3, output=tmp_path
    )
    assert report.ok, str(report)
    assert report.points.shape == (4, posterior.ndim)
    np.testing.assert_allclose(
        report.points[0], posterior.prior_transform(np.full(posterior.ndim, 0.5))
    )
    assert np.all(report.seconds > 0)
    assert report.peak_memory > 0
    assert "OK" in str(report)
    assert (tmp_path / "chain.txt").exists()


@attrs.frozen
class Broken:
    """A likelihood stage that fails above ``x = 0.4``, in the way chosen."""

    how: str
    name: str = "broken"
    parameters: ClassVar = {"x": ParamSpec()}
    requires: ClassVar = {}
    provides: ClassVar = {"lnl": LogLikelihood}

    def compute(self, params):
        if params["x"] > 0.4:
            if self.how == "raises":
                raise RuntimeError("the simulation crashed")
            if self.how == "nan":
                return {"lnl": LogLikelihood(np.nan)}
            return {"lnl": 0.5}
        return {"lnl": LogLikelihood(-(params["x"] ** 2))}


@pytest.mark.parametrize(
    ("how", "message"),
    [
        ("raises", "RuntimeError: the simulation crashed"),
        ("nan", "lnL = nan"),
        ("wrong_type", "StageError"),
    ],
)
def test_a_dry_run_catches_a_broken_stage(how, message):
    """The stage fails at the centre of the prior (x = 0.5); the dry run reports it
    for that point and carries on to the others."""
    posterior = Posterior([Param("x", prior=(0.0, 1.0))], [Broken(how)])
    report = cosmosis.dry_run(posterior, np.random.default_rng(1), n_draws=5)
    assert not report.ok
    assert message in str(report).splitlines()[1]
    assert len(report.errors) == 6


@attrs.frozen
class TwoParameters:
    """A likelihood of two parameters whose names differ only in case and a dot."""

    name: str = "two"
    parameters: ClassVar = {"a.b": ParamSpec(), "A_b": ParamSpec()}
    requires: ClassVar = {}
    provides: ClassVar = {"lnl": LogLikelihood}

    def compute(self, params):
        return {"lnl": LogLikelihood(0.0)}


def test_names_that_clash_in_cosmosis_are_refused():
    posterior = Posterior(
        [Param("a.b", prior=(0.0, 1.0)), Param("A_b", prior=(0.0, 1.0))],
        [TwoParameters()],
    )
    with pytest.raises(ValueError, match="share the CosmoSIS names"):
        cosmosis.build_pipeline(posterior)


# MPI

MPIRUN = shutil.which("mpirun")
mpi = pytest.mark.skipif(
    MPIRUN is None or importlib.util.find_spec("mpi4py") is None,
    reason="needs mpirun and mpi4py",
)

SCRIPT = """
    import sys, time
    import attrs, numpy as np
    from yabf import cosmosis, testing

    class Slow(testing.GlobalSignal):
        def compute(self, params, lightcone):
            time.sleep({sleep})
            return super().compute(params, lightcone)

    analytic = testing.toy_simulation(n_pix=50)
    simulator, _ = analytic.posterior.stages
    posterior = attrs.evolve(
        analytic.posterior,
        stages=[simulator, Slow()],
        derived=[("global_signal", "global_signal")],
    )
    options = {options}
    {before}
    cosmosis.run(posterior, "nautilus", sys.argv[1], options=options, resume=True)
"""


def _mpirun(script: Path, output: Path, n: int = 2, use_mpi4py: bool = True) -> list:
    """The command running ``script`` on ``n`` MPI ranks."""
    command = [MPIRUN, "-n", str(n), "--oversubscribe"]
    if os.geteuid() == 0:
        command.append("--allow-run-as-root")
    command += [sys.executable]
    if use_mpi4py:
        command += ["-m", "mpi4py"]
    return [*command, str(script), str(output)]


def _clean_environment() -> dict:
    """The environment for an MPI job started from this process.

    Initialising MPI in this process (CosmoSIS's MultiNest does) sets variables in
    its C-level environment that ``os.environ`` does not show, and that make a
    child ``mpirun`` fail silently; passing ``os.environ`` explicitly leaves them
    out.
    """
    return dict(os.environ)


def _write(tmp_path, sleep=0.0, before="", options=None):
    """Write the MPI test script."""
    script = tmp_path / "run.py"
    script.write_text(
        textwrap.dedent(SCRIPT).format(
            sleep=sleep, before=before, options=repr(options or NAUTILUS)
        )
    )
    return script


@mpi
def test_a_run_killed_under_mpi_resumes_from_its_checkpoint(tmp_path):
    """Kill an MPI run once its checkpoint exists, resume it, and get the posterior,
    with the derived output of every sample."""
    script = _write(tmp_path, sleep=0.002)
    output = tmp_path / "out"
    run = subprocess.Popen(
        _mpirun(script, output),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        env=_clean_environment(),
    )
    deadline = time.monotonic() + 300
    while not list(output.glob("*.hdf5")):
        assert run.poll() is None, "the run finished before its first checkpoint"
        assert time.monotonic() < deadline, "no checkpoint after 300 s"
        time.sleep(0.5)
    time.sleep(2)
    assert run.poll() is None, "the run finished before it could be killed"
    os.killpg(run.pid, signal.SIGKILL)
    run.wait()

    subprocess.run(
        _mpirun(script, output), check=True, timeout=600, env=_clean_environment()
    )
    analytic, posterior = _toy()
    samples = cosmosis.read_chain(output / "chain.txt", posterior)
    assert samples.finished
    _check(samples, analytic, posterior)


@mpi
@pytest.mark.parametrize("use_mpi4py", [False, True], ids=["plain", "mpi4py"])
def test_a_failure_on_the_first_rank_ends_the_job(tmp_path, use_mpi4py):
    """The first rank fails while the others wait for work: the job ends within
    seconds, with an error, rather than waiting until its time runs out. This holds
    even when the script is launched without ``python -m mpi4py``."""
    # Only the first rank, which configures the sampler, sees the bad option.
    before = textwrap.dedent("""
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() == 0:
            options["n_live"] = "not a number"
    """).strip()
    script = _write(tmp_path, before=before)
    start = time.monotonic()
    result = subprocess.run(
        _mpirun(script, tmp_path / "out", n=3, use_mpi4py=use_mpi4py),
        timeout=120,
        check=False,
        capture_output=True,
        text=True,
        env=_clean_environment(),
    )
    assert result.returncode != 0
    assert "aborting all 3 ranks" in result.stderr
    assert time.monotonic() - start < 120


@pytest.mark.skipif(importlib.util.find_spec("mpi4py") is None, reason="needs mpi4py")
def test_importing_yabf_does_not_initialise_mpi():
    """A process that initialised MPI cannot launch MPI jobs (they fail silently), so
    importing yabf, even its CosmoSIS driver, must leave MPI alone."""
    code = "import sys, yabf, yabf.cosmosis; print('mpi4py.MPI' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=_clean_environment(),
    )
    assert result.stdout.strip() == "False"
