"""The posterior: parameters, stages and likelihoods behind one sampler-facing object.

A :class:`Posterior` joins a :class:`~yabf.params.ParamSpace` to a graph of stages. It
gives samplers what they need (``ndim``, ``names``, ``prior_transform``,
``log_prior`` and ``log_likelihood``), and gives people the whole forward model in one
call:

.. code-block:: python

    posterior = Posterior(params=[...], stages=[...], derived=[("sim", "xHI")])
    out = posterior.forward(theta)  # every output of every stage
    lnl, blobs = posterior.log_likelihood(theta)
    lp, blobs = posterior.log_posterior(theta)

The log-likelihood is the sum of every stage output whose type is
:class:`LogLikelihood`, and the blobs are the outputs listed in ``derived``. The
simplest way to add one is a :class:`LikelihoodTerm`, which scores a model output
against data with a :mod:`~yabf.likelihoods` likelihood. ``posterior.simulate(theta,
rng)`` then draws mock data from the same likelihoods. To analyse a mock, build the
posterior without data, simulate, and add the mock with :meth:`Posterior.with_data`:

.. code-block:: python

    model = Posterior(params, stages, likelihoods=[LikelihoodTerm("lk", lik, Spectrum)])
    posterior = model.with_data(model.simulate(truth, rng))

These methods take one point, ``theta`` of shape ``(ndim,)``, or a batch of points,
of shape ``(n, ndim)``. A batch is evaluated with :meth:`~yabf.pipeline.Pipeline.batch`:
stages that declare ``batched = True`` see the whole batch at once, and the others
are called once per point. Both give the same results.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import attrs
import numpy as np
from numpy.typing import ArrayLike

from .likelihoods import Likelihood, ParametrisedLikelihood
from .params import DerivedParam, Param, ParamSpace
from .pipeline import Output, Pipeline
from .stages import One, ParamSpec, Stage


class LogLikelihood(float):
    """A log-likelihood term, as a stage output.

    A stage that provides an output of this type contributes it to the
    posterior's log-likelihood. Return the value as ``LogLikelihood(x)``.
    """


def _as_param_space(
    params: ParamSpace | Iterable[Param | DerivedParam],
) -> ParamSpace:
    """Return ``params`` as a :class:`~yabf.params.ParamSpace`."""
    return params if isinstance(params, ParamSpace) else ParamSpace(params)


def _as_outputs(outputs: Iterable[Sequence[str]]) -> tuple[Output, ...]:
    """Return ``(stage, output)`` pairs as a tuple of tuples."""
    return tuple((stage, output) for stage, output in outputs)


def _spread(values: np.ndarray | None, inside: np.ndarray) -> np.ndarray:
    """Place the values at the points ``inside`` among all the points of a batch.

    The other points get NaN if the values are numeric, and None otherwise.
    """
    shape = (len(inside), *(() if values is None else values.shape[1:]))
    if values is not None and values.dtype.kind in "fc":
        out = np.full(shape, np.nan, dtype=values.dtype)
    else:
        out = np.full(shape, None, dtype=object)
    if values is not None:
        out[inside] = values
    return out


def _as_model(model: One | type) -> One:
    """Return the requirement for the scored model; a type is the output "model"."""
    if isinstance(model, One):
        return model
    if isinstance(model, type):
        return One(model, name="model")
    raise TypeError(f"The model to score must be a type or One(...), got {model!r}.")


@attrs.frozen(eq=False)
class LikelihoodTerm:
    """A stage that scores one model output against data with a likelihood.

    It requires the model output to score, and provides the log-likelihood of ``data``
    given it, as ``"lnl"``. Instrument effects, such as a gain or a window function,
    are stages of the graph like any other: a term scores their output by asking for
    it by name. Several terms can score different outputs of one model run, each
    computed once.

    The data may be left out, to simulate mock data first: without data, the
    log-likelihood is NaN. Add the data with :meth:`Posterior.with_data`.

    Parameters
    ----------
    name
        The stage name.
    likelihood
        The likelihood, from :mod:`yabf.likelihoods` or any with the same interface.
        If it has parameters of its own (``parameters`` and ``with_params``), the term
        takes them and passes them on.
    model
        The model output to score: a type, for the output of that type named
        ``"model"``, or a :class:`~yabf.stages.One`, such as ``One(Spectrum,
        name="sky")``.
    data
        The data, as values or an object with ``values``, or None if there are no
        data yet.
    """

    name: str
    likelihood: Likelihood
    model: One = attrs.field(converter=_as_model)
    data: object = attrs.field(default=None, kw_only=True)

    @property
    def parameters(self) -> Mapping[str, ParamSpec]:
        """The parameters of the likelihood, if it has any."""
        if isinstance(self.likelihood, ParametrisedLikelihood):
            return self.likelihood.parameters
        return {}

    @property
    def requires(self) -> dict[str, One]:
        """The model output that the term scores."""
        return {"model": self.model}

    @property
    def provides(self) -> dict[str, type]:
        """The log-likelihood."""
        return {"lnl": LogLikelihood}

    def likelihood_at(self, params: Mapping[str, object]) -> Likelihood:
        """Return the likelihood at the values of its own parameters in ``params``."""
        if isinstance(self.likelihood, ParametrisedLikelihood):
            own = {k: params[k] for k in self.likelihood.parameters}
            return self.likelihood.with_params(own)
        return self.likelihood

    def compute(self, params: Mapping[str, object], model: object) -> dict:
        """Score the data against the model; NaN if there are no data."""
        if self.data is None:
            return {"lnl": LogLikelihood(np.nan)}
        return {
            "lnl": LogLikelihood(self.likelihood_at(params).loglike(model, self.data))
        }

    def parameter_values(self, params: Mapping[str, object]) -> dict[str, object]:
        """Return the values of the term's parameters: from ``params``, or defaults."""
        return {k: params.get(k, spec.default) for k, spec in self.parameters.items()}


@attrs.frozen(eq=False)
class Posterior:
    """The posterior of a set of parameters, given stages that compute a likelihood.

    Parameters
    ----------
    params
        The parameters, as a :class:`~yabf.params.ParamSpace` or a sequence of
        :class:`~yabf.params.Param` and :class:`~yabf.params.DerivedParam`.
    stages
        The stages of the forward model. They are resolved, and checked against
        ``params``, when the posterior is built (see :func:`~yabf.stages.resolve`).
    likelihoods
        :class:`LikelihoodTerm` stages, which score model outputs against data. They
        join the graph with ``stages``; together, at least one stage must provide a
        :class:`LogLikelihood` output.
    derived
        Outputs to return as blobs with the log-likelihood, each named by its
        ``(stage, output)`` pair.
    cache_size
        The number of stage results kept across single-point evaluations; see
        :class:`~yabf.pipeline.Pipeline`.
    """

    params: ParamSpace = attrs.field(converter=_as_param_space)
    stages: tuple[Stage, ...] = attrs.field(converter=tuple)
    likelihoods: tuple[LikelihoodTerm, ...] = attrs.field(
        default=(), converter=tuple, kw_only=True
    )
    derived: tuple[Output, ...] = attrs.field(
        default=(), converter=_as_outputs, kw_only=True
    )
    cache_size: int = attrs.field(default=128, kw_only=True)
    pipeline: Pipeline = attrs.field(init=False, repr=False)

    @pipeline.default
    def _pipeline_default(self) -> Pipeline:
        """Resolve the stages against the parameters."""
        return Pipeline.build(
            (*self.stages, *self.likelihoods), self.params, cache_size=self.cache_size
        )

    def __attrs_post_init__(self) -> None:
        """Check that some stage provides a log-likelihood, and that blobs exist."""
        if not self.loglike_outputs:
            raise ValueError(
                "No stage provides a LogLikelihood output, so the posterior has no "
                "likelihood."
            )
        unknown = [d for d in self.derived if d not in self.pipeline.outputs]
        if unknown:
            raise ValueError(f"The derived outputs {unknown} are not stage outputs.")

    @property
    def loglike_outputs(self) -> tuple[Output, ...]:
        """The outputs summed into the log-likelihood, as ``(stage, output)`` pairs."""
        return tuple(
            (s.name, k)
            for s in self.pipeline.graph.stages
            for k, kind in s.provides.items()
            if issubclass(kind, LogLikelihood)
        )

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the sampled parameters, in order."""
        return self.params.names

    @property
    def ndim(self) -> int:
        """The number of sampled parameters."""
        return self.params.ndim

    def prior_transform(self, u: ArrayLike) -> np.ndarray:
        """Map points of the unit hypercube to the sampled parameters."""
        return self.params.prior_transform(u)

    def log_prior(self, theta: ArrayLike) -> np.ndarray:
        """The log prior density of the sampled parameters (``-inf`` outside it)."""
        return self.params.log_prior(theta)

    def forward(
        self, theta: ArrayLike, want: Iterable[Sequence[str]] | None = None
    ) -> dict[Output, object]:
        """Compute the forward model at ``theta``: every stage output, by default.

        ``want`` restricts the result to some ``(stage, output)`` outputs. For a batch
        of points, each output is a tuple with one value per point.
        """
        theta = self._check_theta(theta)
        want = self.pipeline.outputs if want is None else _as_outputs(want)
        model = self.params.to_model(theta)
        if theta.ndim == 1:
            return self.pipeline(model, want)
        return self.pipeline.batch(model, len(theta), want)

    def log_likelihood(
        self, theta: ArrayLike
    ) -> tuple[float | np.ndarray, dict[Output, object]]:
        """Return the log-likelihood at ``theta``, and the derived outputs as blobs.

        For a batch of points, the log-likelihood is an array with one value per
        point, and each blob an array stacking its values over the points.
        """
        theta = self._check_theta(theta)
        missing = [t.name for t in self.likelihoods if t.data is None]
        if missing:
            raise ValueError(
                f"The likelihood terms {missing} have no data. Add it with "
                "with_data, for example from simulate."
            )
        out = self.forward(theta, self.loglike_outputs + self.derived)
        if theta.ndim == 1:
            return (
                float(sum(out[k] for k in self.loglike_outputs)),
                {d: out[d] for d in self.derived},
            )
        terms = [np.asarray(out[k], dtype=float) for k in self.loglike_outputs]
        lnl = np.sum(terms, axis=0)
        return lnl, {d: np.asarray(out[d]) for d in self.derived}

    def log_posterior(
        self, theta: ArrayLike
    ) -> tuple[float | np.ndarray, dict[Output, object]]:
        """Return the log posterior at ``theta``, up to a constant, and the blobs.

        The log posterior is the log prior plus the log-likelihood. Outside the prior
        it is ``-inf``, and the likelihood is not computed there: the blobs of such a
        point are None, or NaN for numeric blobs of a batch. For a batch of points,
        the log posterior and each blob have one value per point, as for
        :meth:`log_likelihood`.
        """
        theta = self._check_theta(theta)
        lp = np.array(self.log_prior(theta), dtype=float)
        inside = lp > -np.inf
        if theta.ndim == 1:
            if not inside:
                return -np.inf, dict.fromkeys(self.derived)
            lnl, blobs = self.log_likelihood(theta)
            return float(lp) + lnl, blobs
        blobs = {}
        if inside.any():
            lnl, blobs = self.log_likelihood(theta[inside])
            lp[inside] += lnl
        return lp, {d: _spread(blobs.get(d), inside) for d in self.derived}

    def with_data(self, data: Mapping[str, object]) -> Posterior:
        """Return this posterior with new data for its likelihood terms, by name.

        ``data`` maps the names of terms in ``likelihoods`` to their data, as
        :meth:`simulate` returns them; terms left out keep their data.
        """
        unknown = sorted(set(data) - {t.name for t in self.likelihoods})
        if unknown:
            raise ValueError(f"The posterior has no likelihood terms named {unknown}.")
        terms = [
            attrs.evolve(t, data=data[t.name]) if t.name in data else t
            for t in self.likelihoods
        ]
        return attrs.evolve(self, likelihoods=terms)

    def simulate(self, theta: ArrayLike, rng: np.random.Generator) -> dict[str, object]:
        """Draw mock data at one point ``theta``, for each likelihood term, by name.

        Each term's likelihood draws data from the output it scores, so a mock is
        distributed as the likelihood assumes. Randomness inside the forward model,
        such as a simulator's seed, is not redrawn: it is set by the parameters (see
        :attr:`random_inputs`).
        """
        theta = self._check_theta(theta)
        if theta.ndim != 1:
            raise ValueError("simulate draws mock data at one point at a time.")
        scored = self.scored
        models = self.forward(theta, scored.values())
        params = self.params.to_model(theta)
        return {
            t.name: t.likelihood_at(t.parameter_values(params)).simulate(
                models[scored[t.name]], rng
            )
            for t in self.likelihoods
        }

    @property
    def scored(self) -> dict[str, Output]:
        """The output each likelihood term scores, as a ``(stage, output)`` pair."""
        return {
            link.stage: (link.provider, link.output)
            for link in self.pipeline.graph.links
            if link.stage in {t.name for t in self.likelihoods}
        }

    @property
    def random_inputs(self) -> dict[str, str]:
        """How each random input of a stage is set, by name.

        Each is ``"sampled"``, ``"fixed"``, ``"derived"`` or ``"default"``. An input
        is random when a stage's :class:`~yabf.stages.ParamSpec` says so, as for a
        simulator's seed. Leaving it at its default or fixing it conditions the
        posterior on one realisation; sampling it treats the realisation as noise.
        """
        by_name = {p.name: p for p in self.params.params}
        out = {}
        for stage in self.pipeline.graph.stages:
            for k, spec in stage.parameters.items():
                if spec.random:
                    p = by_name.get(k)
                    out[k] = (
                        "default"
                        if p is None
                        else "derived"
                        if isinstance(p, DerivedParam)
                        else "sampled"
                        if p.is_sampled
                        else "fixed"
                    )
        return out

    def _check_theta(self, theta: ArrayLike) -> np.ndarray:
        """Return ``theta`` as an array of shape ``(ndim,)`` or ``(n, ndim)``."""
        theta = np.asarray(theta, dtype=float)
        if theta.ndim not in (1, 2) or theta.shape[-1] != self.ndim:
            raise ValueError(
                f"theta must have shape ({self.ndim},) for one point or (n, "
                f"{self.ndim}) for a batch, got {theta.shape}."
            )
        return theta
