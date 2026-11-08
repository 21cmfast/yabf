"""Run a posterior through CosmoSIS, and check it with a dry run first.

`CosmoSIS <https://cosmosis.readthedocs.io>`_ drives many samplers (nautilus,
MultiNest, PolyChord, emcee, ...) the same way, with resume and MPI. It evaluates
one point per call: for batched stages, such as an emulator on a GPU, use
:mod:`yabf.sampling` instead. Both return :class:`~yabf.sampling.Samples`.

CosmoSIS samples the unit hypercube, and each point is mapped to the parameters by
:meth:`~yabf.posterior.Posterior.prior_transform` before the likelihood is computed.
The posterior and evidence are therefore exactly the posterior's own, whatever its
priors (joint, or any scipy distribution), and CosmoSIS needs to know nothing about
them. The chain it writes holds the sampled parameters, by name, and the derived
outputs, as derived values; :func:`read_chain` reads it back.

- :func:`run` runs a sampler and returns its samples.
- :func:`dry_run` computes the posterior at a few points, with no sampler, and
  writes and reads them through the same output, so that a broken stage, a slow
  one or one using too much memory is found before a long run starts.

Under MPI, launch the script on every rank with ``mpirun -n N python -m mpi4py
script.py``. The first rank drives the sampler and the others compute the posterior;
:func:`run` returns the samples on the first rank and None on the others. If the
first rank fails, it aborts the whole job, so that the other ranks do not wait for
work until the job's time runs out (``python -m mpi4py`` does the same for every
rank).

It needs the ``cosmosis`` package, and ``mpi4py`` for MPI.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import resource
import sys
import time
import traceback
import warnings
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np

from .pipeline import Output
from .posterior import Posterior
from .sampling import Samples, _values

if TYPE_CHECKING:
    from cosmosis import LikelihoodPipeline, MPIPool
    from cosmosis.datablock import DataBlock

#: The section of the CosmoSIS parameters, which are the unit-cube coordinates.
UNIT_CUBE = "unit_cube"


def _key(name: str) -> str:
    """Return ``name`` as a CosmoSIS key: lower case, letters, digits and _ only."""
    return re.sub(r"[^a-z0-9_]", "_", name.lower())


@attrs.frozen
class _Layout:
    """Where the parameters and derived outputs of a posterior go in CosmoSIS."""

    params: tuple[str, ...]
    outputs: tuple[tuple[Output, str, tuple[int, ...]], ...]

    @classmethod
    def of(cls, posterior: Posterior) -> _Layout:
        """The layout of ``posterior``'s parameters and derived outputs.

        The shape of each derived output is taken from one evaluation at the centre
        of the prior.
        """
        outputs = []
        if posterior.derived:
            centre = posterior.prior_transform(np.full(posterior.ndim, 0.5))
            _, blobs = posterior.log_likelihood(centre)
            outputs = [
                (d, _key(f"{d[0]}__{d[1]}"), _values(blobs[d]).shape)
                for d in posterior.derived
            ]
        layout = cls(tuple(posterior.names), tuple(outputs))
        keys = [_key(n) for n in layout.params] + [k for _, k, _ in layout.outputs]
        clashes = sorted({k for k in keys if keys.count(k) > 1})
        if clashes:
            raise ValueError(
                f"Parameters or derived outputs share the CosmoSIS names {clashes}, "
                "which ignore case and punctuation: rename them."
            )
        return layout

    @property
    def extra_output(self) -> str:
        """The derived values CosmoSIS saves with each sample."""
        saves = [f"derived/{_key(n)}" for n in self.params]
        for _, key, shape in self.outputs:
            size = int(np.prod(shape))
            saves.append(f"derived/{key}" if shape == () else f"derived/{key}#{size}")
        return " ".join(saves)


def build_pipeline(posterior: Posterior) -> LikelihoodPipeline:
    """Return a CosmoSIS pipeline computing ``posterior`` on the unit hypercube.

    Its parameters are the unit-cube coordinates ``u0``, ``u1``, ... in the section
    ``unit_cube``. Each point is mapped to the parameters with the posterior's prior
    transform, and the likelihood computed there; the parameters and the derived
    outputs are saved as derived values. An error in the posterior is raised, not
    turned into a zero likelihood.
    """
    from cosmosis import LikelihoodPipeline
    from cosmosis.runtime import logs, module

    layout = _Layout.of(posterior)
    n = posterior.ndim

    def setup(options: object) -> dict:
        """Nothing to set up."""
        return {}

    def execute(block: DataBlock, config: object) -> int:
        """Compute the posterior at the block's point of the unit cube."""
        u = np.array([block[UNIT_CUBE, f"u{i}"] for i in range(n)])
        theta = posterior.prior_transform(u)
        lnl, blobs = posterior.log_likelihood(theta)
        for name, value in zip(layout.params, theta, strict=True):
            block["derived", _key(name)] = float(value)
        for d, key, shape in layout.outputs:
            value = _values(blobs[d])
            block["derived", key] = float(value) if shape == () else value.ravel()
        block["likelihoods", "yabf_like"] = float(lnl)
        return 0

    config = {
        "pipeline": {
            "likelihoods": "yabf",
            "debug": "T",
            "extra_output": layout.extra_output,
        }
    }
    values = {UNIT_CUBE: {f"u{i}": "0.0 0.5 1.0" for i in range(n)}}
    # CosmoSIS prints its setup, and a table of the priors, which here are those of
    # the unit cube and would only confuse; keep them quiet.
    level = logs.get_level()
    logs.set_verbosity("muted")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return LikelihoodPipeline(
                config,
                values=values,
                modules=[module.FunctionModule("yabf", setup, execute)],
            )
    finally:
        logs.set_level(level)


def _meta(path: Path) -> tuple[list[str], dict[str, str]]:
    """Return the column names and the ``key=value`` metadata of a CosmoSIS chain."""
    names, meta = None, {}
    with path.open() as f:
        for line in f:
            if not line.startswith("#"):
                continue
            if names is None:
                names = line[1:].split()
            elif "=" in line:
                k, v = line[1:].strip().split("=", 1)
                meta[k.strip()] = v.split()[0] if v.split() else ""
    if names is None:
        raise ValueError(f"{path} has no header of column names.")
    return [n.lower() for n in names], meta


def read_chain(path: str | os.PathLike, posterior: Posterior) -> Samples:
    """Read a CosmoSIS text chain of ``posterior``, written by :func:`run`.

    The weights are those the sampler wrote (``log_weight`` or ``weight``), or equal.
    The log prior is the posterior's, at each sample; the log-likelihood is the
    chain's posterior minus its (unit-cube) prior. The number of effective samples is
    the sampler's own estimate if it wrote one, and otherwise that of the weights,
    ``1 / sum(w**2)``: for MCMC, that is the number of samples, an upper bound.
    """
    path = Path(path)
    names, meta = _meta(path)
    data = np.loadtxt(path, comments="#", ndmin=2)
    column = {n: i for i, n in enumerate(names)}
    layout = _Layout.of(posterior)

    def derived(key: str) -> np.ndarray:
        """The derived values called ``key``, one row per sample."""
        if f"derived--{key}" in column:
            return data[:, column[f"derived--{key}"]]
        cols = [column[n] for n in names if re.fullmatch(f"derived--{key}_\\d+", n)]
        return data[:, sorted(cols, key=lambda c: int(names[c].rsplit("_", 1)[1]))]

    points = np.column_stack([derived(_key(n)) for n in layout.params])
    if "log_weight" in column:
        log_w = data[:, column["log_weight"]]
        weights = np.exp(log_w - log_w.max())
    elif "weight" in column:
        weights = data[:, column["weight"]]
    else:
        weights = np.ones(len(data))
    weights = weights / weights.sum()
    prior = data[:, column["prior"]] if "prior" in column else 0.0
    log_evidence = float(meta["log_z"]) if "log_z" in meta else None
    return Samples(
        names=posterior.names,
        points=points,
        weights=weights,
        log_likelihood=data[:, column["post"]] - prior,
        log_prior=posterior.log_prior(points),
        derived={
            d: derived(key).reshape(len(data), *shape)
            for d, key, shape in layout.outputs
        },
        n_effective=float(meta["neff"]) if "neff" in meta else 1 / np.sum(weights**2),
        log_evidence=log_evidence,
        finished=meta.get("complete", "0") == "1",
        sampler=meta.get("sampler", ""),
    )


#: Environment variables that MPI launchers set for the processes they start.
_LAUNCHED_BY_MPI = (
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_RANK",
    "MPI_LOCALNRANKS",
    "MV2_COMM_WORLD_SIZE",
)


def _rank() -> tuple[int, int]:
    """Return the MPI rank of this process and the number of ranks: (0, 1) alone.

    MPI is only initialised (by importing mpi4py) when the process was started by an
    MPI launcher, or mpi4py is already in use: initialising it in a serial process
    is slow, and would confuse MPI jobs that the process starts.
    """
    launched = any(v in os.environ for v in _LAUNCHED_BY_MPI)
    if not launched and "mpi4py.MPI" not in sys.modules:
        return 0, 1
    try:
        from mpi4py import MPI
    except ImportError:
        return 0, 1
    return MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()


@contextlib.contextmanager
def _pool() -> Iterator[MPIPool | None]:
    """Yield a CosmoSIS MPI pool when running on several ranks, else None."""
    if _rank()[1] == 1:
        yield None
        return
    from cosmosis import MPIPool

    with MPIPool() as pool:
        yield pool


@contextlib.contextmanager
def _abort_on_failure() -> Iterator[None]:
    """Abort every rank if this one fails, when running on several of them."""
    try:
        yield
    except BaseException:
        rank, size = _rank()
        if size == 1:
            raise
        from mpi4py import MPI

        traceback.print_exc()
        sys.stderr.write(f"Rank {rank} failed: aborting all {size} ranks.\n")
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort(1)


def _sampler_options(
    sampler: str, output: Path, options: Mapping[str, object], resume: bool
) -> dict[str, str]:
    """Return a sampler's ini section, with this driver's defaults."""
    out = {k: str(v) for k, v in options.items()}
    if sampler == "nautilus":
        out.setdefault("resume", "T" if resume else "F")
    elif sampler == "multinest":
        out.setdefault("live_points", "500")
        out.setdefault("max_iterations", "1000000")
        out.setdefault("feedback", "F")
        out.setdefault("resume", "T" if resume else "F")
        out.setdefault("multinest_outfile_root", str(output / "multinest_"))
        # MultiNest's importance nested sampling (ins) biases the evidence of a
        # resumed run (by +0.4 to +1.05 in ln Z, in the M0.3 spike,
        # 21cmfast/21CMMC#96), so it is off unless asked for.
        out.setdefault("ins", "F")
        if resume and out["ins"].upper() in ("T", "TRUE", "Y", "YES", "1"):
            warnings.warn(
                "MultiNest with ins=T gives a biased evidence when resumed; the "
                "posterior is unaffected.",
                stacklevel=3,
            )
    return out


def run(
    posterior: Posterior,
    sampler: str,
    output: str | os.PathLike,
    *,
    options: Mapping[str, object] | None = None,
    resume: bool = False,
    verbosity: str = "quiet",
) -> Samples | None:
    """Sample ``posterior`` with one of CosmoSIS's samplers.

    Parameters
    ----------
    posterior
        The posterior to sample.
    sampler
        The name of a CosmoSIS sampler, such as ``"nautilus"`` or ``"multinest"``.
    output
        A directory for the chain (``chain.txt``) and the sampler's checkpoints.
    options
        The sampler's options, as in its section of a CosmoSIS ini file, such as
        ``{"n_live": 1000, "seed": 1}`` for nautilus. MultiNest's importance nested
        sampling (``ins``) is off unless asked for, because it biases the evidence
        of a resumed run.
    resume
        Whether to resume from the checkpoints in ``output``, if there are any.
    verbosity
        CosmoSIS's verbosity, such as ``"quiet"`` or ``"standard"``.

    Returns
    -------
    Samples or None
        The samples, read back from the chain, on the first MPI rank (or without
        MPI); None on the other ranks.
    """
    from cosmosis import run_cosmosis
    from cosmosis.runtime import logs

    from ._nautilus import atomic_checkpoints

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ini = {
        "runtime": {
            "sampler": sampler,
            "verbosity": verbosity,
            "resume": "T" if resume else "F",
        },
        "output": {"filename": str(output / "chain.txt"), "format": "text"},
        sampler: _sampler_options(sampler, output, options or {}, resume),
    }
    with _abort_on_failure():
        logs.set_verbosity(verbosity)
        pipeline = build_pipeline(posterior)
        with atomic_checkpoints(), _pool() as pool:
            run_cosmosis(ini=ini, pool=pool, pipeline=pipeline)
            if pool is not None and not pool.is_master():
                return None
        return read_chain(output / "chain.txt", posterior)


@attrs.frozen(eq=False)
class DryRunReport:
    """What a dry run found, at each point it computed.

    Parameters
    ----------
    points
        The points of the parameter space, shape ``(n, ndim)``: the centre of the
        prior, then draws from it.
    log_likelihood
        The log-likelihood at each point; NaN where it failed.
    seconds
        The time each evaluation took.
    errors
        The error at each point, or None where it succeeded.
    peak_memory
        The largest resident memory of this process during the run, in bytes.
    problems
        Problems found when writing the points to the chain and reading them back.
    """

    names: tuple[str, ...]
    points: np.ndarray
    log_likelihood: np.ndarray
    seconds: np.ndarray
    errors: tuple[str | None, ...]
    peak_memory: int
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every point was computed, finite, and written and read back."""
        return (
            all(e is None for e in self.errors)
            and bool(np.all(np.isfinite(self.log_likelihood)))
            and not self.problems
        )

    def __str__(self) -> str:
        """A table of the points, their log-likelihoods and times, and any errors."""
        lines = [
            (
                f"Dry run: {'OK' if self.ok else 'FAILED'}; peak memory "
                f"{self.peak_memory / 2**20:.0f} MiB."
            ),
        ]
        for i, (lnl, t, e) in enumerate(
            zip(self.log_likelihood, self.seconds, self.errors, strict=True)
        ):
            where = "centre of the prior" if i == 0 else f"prior draw {i}"
            status = f"lnL = {lnl:.6g}" if e is None else f"error: {e}"
            lines.append(f"  {where}: {status} ({t:.3g} s)")
        lines.extend(f"  {p}" for p in self.problems)
        return "\n".join(lines)


def _peak_memory() -> int:
    """The largest resident memory of this process so far, in bytes."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024


def dry_run(
    posterior: Posterior,
    rng: np.random.Generator,
    *,
    n_draws: int = 3,
    output: str | os.PathLike | None = None,
) -> DryRunReport:
    """Compute ``posterior`` at a few points, with no sampler, and report.

    The points are the centre of the prior and ``n_draws`` draws from it. Each is
    computed in turn, timed, and any error is caught and reported rather than
    raised. If ``output`` is given and every point succeeded, the points are then run
    through CosmoSIS's ``list`` sampler, which writes them to ``output/chain.txt``
    as a real run would; the chain is read back and checked against the direct
    results.
    """
    u = np.vstack(
        [np.full(posterior.ndim, 0.5), rng.uniform(size=(n_draws, posterior.ndim))]
    )
    points = posterior.prior_transform(u)
    lnl = np.full(len(points), np.nan)
    seconds = np.zeros(len(points))
    errors: list[str | None] = []
    blobs = []
    for i, theta in enumerate(points):
        start = time.perf_counter()
        try:
            lnl[i], b = posterior.log_likelihood(theta)
            blobs.append(b)
            errors.append(None)
        # Catch anything a stage raises: reporting it, at its point, is the job.
        except Exception as e:  # noqa: BLE001
            blobs.append(None)
            errors.append(f"{type(e).__name__}: {e}")
        seconds[i] = time.perf_counter() - start
    problems = []
    if output is not None and all(e is None for e in errors):
        problems = _check_output(posterior, Path(output), u, lnl, blobs)
    return DryRunReport(
        names=tuple(posterior.names),
        points=points,
        log_likelihood=lnl,
        seconds=seconds,
        errors=tuple(errors),
        peak_memory=_peak_memory(),
        problems=tuple(problems),
    )


def _check_output(
    posterior: Posterior,
    output: Path,
    u: np.ndarray,
    lnl: np.ndarray,
    blobs: Sequence[Mapping[Output, object]],
) -> list[str]:
    """Write the points through CosmoSIS's list sampler, read them back, compare."""
    from cosmosis import run_cosmosis
    from cosmosis.runtime import logs

    logs.set_verbosity("muted")

    output.mkdir(parents=True, exist_ok=True)
    points_file = output / "dry_run_points.txt"
    header = "\t".join(f"{UNIT_CUBE}--u{i}" for i in range(posterior.ndim))
    np.savetxt(points_file, u, header=header, comments="#", delimiter="\t")
    ini = {
        "runtime": {"sampler": "list", "verbosity": "quiet"},
        "output": {"filename": str(output / "chain.txt"), "format": "text"},
        "list": {"filename": str(points_file)},
    }
    run_cosmosis(ini=ini, pipeline=build_pipeline(posterior))
    chain = read_chain(output / "chain.txt", posterior)
    problems = []
    if not np.allclose(chain.log_likelihood, lnl, rtol=1e-10, atol=0):
        problems.append("The log-likelihoods read back differ from those computed.")
    for d in posterior.derived:
        direct = np.stack([_values(b[d]) for b in blobs])
        if not np.allclose(
            chain.derived[d], direct, rtol=1e-10, atol=0, equal_nan=True
        ):
            problems.append(
                f"The derived output {d} read back differs from the one computed."
            )
    return problems
