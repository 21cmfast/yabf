"""Run samplers on a posterior, and the samples they return.

Each function here drives one sampler directly on a :class:`~yabf.posterior.Posterior`
and returns its result as :class:`Samples`: the points with their weights, the
log-likelihood and log prior of each, the posterior's derived outputs at each point,
and the evidence when the sampler estimates it.

The samplers are driven in batches. The posterior is evaluated on many points at
once, so a batched stage, such as an emulator on a GPU, computes them in one call (see
:meth:`~yabf.pipeline.Pipeline.batch`); unbatched stages are called once per point.

- :func:`run_emcee` runs emcee, an ensemble MCMC sampler, with every walker in one
  batch.
- :func:`run_nautilus` runs nautilus, a nested sampler that also estimates the
  evidence. It needs the ``nautilus-sampler`` package.

Both write a checkpoint when given a file, and resume from it when called again with
the same file, so that a run can be stopped and continued. A resumed run stores the
derived outputs of every sample, like an uninterrupted one.

Derived outputs are stored as arrays of floats: an output with ``values``, such as an
observable, is stored as its values, and anything else must convert to a float array
of the same shape at every point.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np
from getdist import MCSamples
from numpy.typing import ArrayLike
from scipy import special

from .params import ParamSpace
from .pipeline import Output
from .posterior import Posterior

if TYPE_CHECKING:
    import xarray


def _readonly(value: ArrayLike) -> np.ndarray:
    """Return a read-only copy of ``value`` as a float array."""
    out = np.array(value, dtype=float)
    out.flags.writeable = False
    return out


def _readonly_mapping(value: Mapping[Output, ArrayLike]) -> dict[Output, np.ndarray]:
    """Return a mapping of read-only float arrays."""
    return {k: _readonly(v) for k, v in value.items()}


@attrs.frozen(eq=False)
class Samples:
    """Weighted samples of a posterior, with what was computed at each.

    Parameters
    ----------
    names
        The names of the sampled parameters, in the order of the columns of ``points``.
    points
        The samples, shape ``(n, ndim)``.
    weights
        The weight of each sample, summing to one. MCMC samples all have equal weights.
    log_likelihood, log_prior
        The log-likelihood and the log prior density of each sample.
    derived
        The posterior's derived outputs at each sample, by ``(stage, output)``, each
        with a leading axis of length ``n``.
    n_effective
        The number of effective samples: for MCMC, the number of samples divided by
        the integrated autocorrelation time; for nested sampling, the sampler's own
        estimate.
    log_evidence
        The natural log of the evidence, if the sampler estimates it.
    n_chains
        The number of chains the samples come from, all of the same length, one after
        the other: for emcee, one per walker. Nested samplers give one.
    finished
        Whether the sampler reached its stopping criterion. A run stopped early, by
        a limit on time or on likelihood evaluations, can be resumed from its
        checkpoint.
    sampler
        The name of the sampler.
    """

    names: tuple[str, ...] = attrs.field(converter=tuple)
    points: np.ndarray = attrs.field(converter=_readonly)
    weights: np.ndarray = attrs.field(converter=_readonly)
    log_likelihood: np.ndarray = attrs.field(converter=_readonly)
    log_prior: np.ndarray = attrs.field(converter=_readonly)
    derived: dict[Output, np.ndarray] = attrs.field(converter=_readonly_mapping)
    n_effective: float
    log_evidence: float | None = None
    n_chains: int = 1
    finished: bool = True
    sampler: str = ""

    def __attrs_post_init__(self) -> None:
        """Check that every array has one entry per sample."""
        n = len(self.points)
        if self.points.shape != (n, len(self.names)):
            raise ValueError(
                f"points must have shape (n, {len(self.names)}), got "
                f"{self.points.shape}."
            )
        for what, value in [
            ("weights", self.weights),
            ("log_likelihood", self.log_likelihood),
            ("log_prior", self.log_prior),
            *self.derived.items(),
        ]:
            if len(value) != n:
                raise ValueError(f"{what} has {len(value)} entries for {n} samples.")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("The weights must sum to one.")
        if self.n_chains < 1 or n % self.n_chains:
            raise ValueError(
                f"{n} samples cannot be {self.n_chains} chains of the same length."
            )

    @property
    def equally_weighted(self) -> bool:
        """Whether every sample has the same weight, as for MCMC."""
        return bool(np.allclose(self.weights, 1 / len(self.weights)))

    @property
    def mean(self) -> np.ndarray:
        """The weighted mean of the samples."""
        return self.weights @ self.points

    @property
    def covariance(self) -> np.ndarray:
        """The weighted covariance of the samples."""
        centred = self.points - self.mean
        return (centred.T * self.weights) @ centred

    def resample(self, n: int, rng: np.random.Generator) -> Samples:
        """Return ``n`` equally weighted samples, drawn by systematic resampling.

        Each sample appears ``floor(n * w)`` or ``ceil(n * w)`` times, for its weight
        ``w``: the least random way to turn weighted samples into equal ones. Use it
        for tools that only take equally weighted samples, such as ArviZ, with ``n``
        about the number of effective samples; the result has no more information
        than the weighted samples.
        """
        positions = (rng.random() + np.arange(n)) / n
        index = np.minimum(
            np.searchsorted(np.cumsum(self.weights), positions), len(self.weights) - 1
        )
        return attrs.evolve(
            self,
            points=self.points[index],
            weights=np.full(n, 1 / n),
            log_likelihood=self.log_likelihood[index],
            log_prior=self.log_prior[index],
            derived={k: v[index] for k, v in self.derived.items()},
            n_effective=min(float(n), self.n_effective),
            n_chains=1,
        )

    def to_arviz(
        self,
        params: ParamSpace | None = None,
        dims: Mapping[str, list[str]] | None = None,
        coords: Mapping[str, ArrayLike] | None = None,
    ) -> xarray.DataTree:
        """Return the samples as an ArviZ :class:`xarray.DataTree`.

        The ``posterior`` group holds each sampled parameter, each derived output (as
        ``"stage.output"``) and, if ``params`` is given, each of its derived
        parameters, with dimensions ``chain`` and ``draw`` first. The
        ``sample_stats`` group holds the log posterior ``lp``, the log-likelihood and
        the log prior of each sample. The sampler's name, the number of effective
        samples and the evidence are attributes of the ``posterior`` group. It needs
        the ``arviz-base`` package.

        ArviZ takes equally weighted samples only: :meth:`resample` weighted ones
        first.

        Parameters
        ----------
        params
            The parameter space sampled, to include its derived parameters.
        dims, coords
            The names of the dimensions of the variables after ``chain`` and
            ``draw``, and their coordinates, as for :func:`arviz_base.from_dict`:
            for example, ``dims={"signal.component": ["freq"]}`` and
            ``coords={"freq": frequencies}``.

        Raises
        ------
        ValueError
            If the samples are not equally weighted.
        """
        from arviz_base import from_dict

        if not self.equally_weighted:
            raise ValueError(
                "ArviZ takes equally weighted samples: resample these first, with "
                "samples.resample(n, rng)."
            )

        def chains(values: np.ndarray) -> np.ndarray:
            """Split values along the samples into (chain, draw, ...)."""
            return values.reshape(self.n_chains, -1, *values.shape[1:])

        posterior = {
            name: chains(self.points[:, i]) for i, name in enumerate(self.names)
        }
        if params is not None:
            model = params.to_model(self.points)
            posterior |= {p.name: chains(model[p.name]) for p in params.derived}
        posterior |= {f"{s}.{o}": chains(v) for (s, o), v in self.derived.items()}
        stats = {
            "lp": chains(self.log_likelihood + self.log_prior),
            "log_likelihood": chains(self.log_likelihood),
            "log_prior": chains(self.log_prior),
        }
        attrs_ = {"sampler": self.sampler, "n_effective": self.n_effective}
        if self.log_evidence is not None:
            attrs_["log_evidence"] = self.log_evidence
        return from_dict(
            {"posterior": posterior, "sample_stats": stats},
            dims=dict(dims or {}),
            coords=dict(coords or {}),
            attrs={"posterior": attrs_},
        )

    def to_getdist(
        self, labels: list[str] | None = None, **kwargs: object
    ) -> MCSamples:
        """Return the samples as a :class:`getdist.MCSamples`, for plots and tables.

        ``labels`` are LaTeX labels for the parameters, and ``kwargs`` are passed to
        :class:`getdist.MCSamples`.
        """
        return MCSamples(
            samples=self.points,
            weights=self.weights,
            loglikes=-(self.log_likelihood + self.log_prior),
            names=list(self.names),
            labels=labels,
            **kwargs,
        )


def _values(value: object) -> np.ndarray:
    """Return a derived output as a float array: its ``values``, if it has them."""
    return np.asarray(getattr(value, "values", value), dtype=float)


def _derived_dtype(posterior: Posterior) -> np.dtype | None:
    """The structured dtype storing the posterior's derived outputs, or None.

    The shapes are taken from one evaluation at the fiducial point.
    """
    if not posterior.derived:
        return None
    _, blobs = posterior.log_likelihood(posterior.params.fiducial)
    return np.dtype(
        [(f"{s}.{o}", float, _values(blobs[s, o]).shape) for s, o in posterior.derived]
    )


def _numeric(column: ArrayLike, shape: tuple[int, ...]) -> np.ndarray:
    """Return a column of derived outputs as floats, with NaN where one is missing."""
    column = list(column)
    out = np.full((len(column), *shape), np.nan)
    for i, value in enumerate(column):
        if value is not None:
            out[i] = _values(value)
    return out


def _unpack(
    posterior: Posterior, blobs: np.ndarray | None, n: int
) -> dict[Output, np.ndarray]:
    """Return a structured array of derived outputs as arrays, by output."""
    if blobs is None:
        return {}
    blobs = np.asarray(blobs).reshape(n)
    return {(s, o): blobs[f"{s}.{o}"] for s, o in posterior.derived}


def _by_walker(values: np.ndarray) -> np.ndarray:
    """Flatten emcee's (step, walker, ...) array to one walker's chain after another."""
    values = np.swapaxes(values, 0, 1)
    return values.reshape(-1, *values.shape[2:])


def _emcee_log_prob(
    posterior: Posterior, dtype: np.dtype | None
) -> Callable[[np.ndarray], np.ndarray | list[tuple]]:
    """Return the vectorised log probability emcee samples, with blobs if any."""

    def log_prob(theta: np.ndarray) -> np.ndarray | list[tuple]:
        """The log posterior of each walker, with its derived outputs."""
        lp, blobs = posterior.log_posterior(theta)
        if dtype is None:
            return lp
        columns = [
            _numeric(blobs[s, o], dtype[f"{s}.{o}"].shape) for s, o in posterior.derived
        ]
        return [(p, *row) for p, *row in zip(lp, *columns, strict=True)]

    return log_prob


def run_emcee(
    posterior: Posterior,
    nsteps: int,
    rng: np.random.Generator,
    *,
    nwalkers: int | None = None,
    start: ArrayLike | None = None,
    checkpoint: str | os.PathLike | None = None,
    discard: int = 0,
    thin: int = 1,
    **kwargs: object,
) -> Samples:
    """Sample ``posterior`` with emcee, evaluating all the walkers as one batch.

    Parameters
    ----------
    posterior
        The posterior to sample.
    nsteps
        The total number of steps of the chain. When resuming from ``checkpoint``,
        the run continues until the chain has this many steps.
    rng
        Seeds the walkers' starting points and emcee's own random state.
    nwalkers
        The number of walkers: by default, one per starting point if ``start`` is
        given, and otherwise four per parameter.
    start
        The starting points, shape ``(nwalkers, ndim)``. By default, they are drawn
        from the prior. They are ignored when resuming.
    checkpoint
        An HDF5 file to write the chain to as it runs. If it already holds a chain,
        the run resumes from its last step.
    discard, thin
        The number of steps to discard from the start of the chain, and the
        thinning, of the returned samples.
    kwargs
        Passed to :class:`emcee.EnsembleSampler`, such as ``moves``.
    """
    import emcee

    ndim = posterior.ndim
    if start is not None:
        start = np.asarray(start, dtype=float)
        nwalkers = nwalkers or len(start)
    nwalkers = nwalkers or 4 * ndim
    dtype = _derived_dtype(posterior)
    backend = None if checkpoint is None else emcee.backends.HDFBackend(checkpoint)
    resuming = backend is not None and Path(checkpoint).exists() and backend.iteration
    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        _emcee_log_prob(posterior, dtype),
        vectorize=True,
        blobs_dtype=dtype,
        backend=backend,
        **kwargs,
    )
    if resuming:
        sampler.run_mcmc(None, max(nsteps - backend.iteration, 0), progress=False)
    else:
        if start is None:
            start = posterior.params.sample_prior(nwalkers, rng)
        sampler.random_state = np.random.RandomState(rng.integers(2**32)).get_state()
        sampler.run_mcmc(start, nsteps, progress=False)

    # Keep each walker's chain together, one walker after another, so that the
    # chains can be told apart (for ArviZ's R-hat, say).
    points = _by_walker(sampler.get_chain(discard=discard, thin=thin))
    log_post = _by_walker(sampler.get_log_prob(discard=discard, thin=thin))
    log_prior = posterior.log_prior(points)
    tau = emcee.autocorr.integrated_time(
        sampler.get_chain(discard=discard), quiet=True
    ).max()
    blobs = (
        None
        if dtype is None
        else _by_walker(sampler.get_blobs(discard=discard, thin=thin))
    )
    n = len(points)
    return Samples(
        names=posterior.names,
        points=points,
        weights=np.full(n, 1 / n),
        log_likelihood=log_post - log_prior,
        log_prior=log_prior,
        derived=_unpack(posterior, blobs, n),
        n_effective=n * thin / tau,
        n_chains=nwalkers,
        sampler="emcee",
    )


def run_nautilus(
    posterior: Posterior,
    *,
    n_live: int = 2000,
    n_eff: float = 10_000,
    f_live: float = 0.01,
    seed: int | None = None,
    checkpoint: str | os.PathLike | None = None,
    discard_exploration: bool = True,
    n_like_max: float = np.inf,
    timeout: float = np.inf,
    **kwargs: object,
) -> Samples:
    """Sample ``posterior`` with nautilus, evaluating each batch of points at once.

    nautilus is a nested sampler: besides the samples, it estimates the evidence. It
    needs at least two sampled parameters.

    Parameters
    ----------
    posterior
        The posterior to sample.
    n_live, n_eff, f_live
        The number of live points, and the stopping criteria: the number of
        effective samples, and the largest fraction of the evidence that may remain
        in the live set. See :class:`nautilus.Sampler`.
    seed
        Seeds nautilus.
    checkpoint
        An HDF5 file (``.h5`` or ``.hdf5``) to write the run to as it goes. If it
        already holds a run, the run resumes from it. Each write replaces the file in
        one step, so stopping the run at any moment, even by killing it, leaves a
        usable checkpoint.
    discard_exploration
        Whether to discard the points drawn while exploring, which nautilus needs for
        unbiased samples and evidence.
    n_like_max, timeout
        Stop after this many likelihood evaluations, or this many seconds. The
        result then has ``finished=False``, and calling again with the same
        ``checkpoint`` continues the run.
    kwargs
        Passed to :class:`nautilus.Sampler`, such as ``n_batch``.
    """
    from ._nautilus import AtomicSampler

    dtype = _derived_dtype(posterior)

    def likelihood(theta: np.ndarray) -> np.ndarray | tuple:
        """The log-likelihood of each point, with its derived outputs."""
        lnl, blobs = posterior.log_likelihood(theta)
        if dtype is None:
            return lnl
        return (
            lnl,
            *(
                _numeric(blobs[s, o], dtype[f"{s}.{o}"].shape)
                for s, o in posterior.derived
            ),
        )

    sampler = AtomicSampler(
        posterior.prior_transform,
        likelihood,
        n_dim=posterior.ndim,
        n_live=n_live,
        vectorized=True,
        pass_dict=False,
        seed=seed,
        blobs_dtype=dtype,
        filepath=None if checkpoint is None else str(checkpoint),
        resume=True,
        **kwargs,
    )
    finished = sampler.run(
        f_live=f_live,
        n_eff=n_eff,
        discard_exploration=discard_exploration,
        n_like_max=n_like_max,
        timeout=timeout,
    )
    if dtype is None:
        points, log_w, log_l = sampler.posterior()
        blobs = None
    else:
        points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)
    return Samples(
        names=posterior.names,
        points=points,
        weights=np.exp(log_w - special.logsumexp(log_w)),
        log_likelihood=log_l,
        log_prior=posterior.log_prior(points),
        derived=_unpack(posterior, blobs, len(points)),
        n_effective=sampler.n_eff,
        log_evidence=sampler.log_z,
        finished=bool(finished),
        sampler="nautilus",
    )
