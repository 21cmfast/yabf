"""The posterior: parameters, stages and likelihoods behind one sampler-facing object.

A :class:`Posterior` joins a :class:`~yabf.params.ParamSpace` to a graph of stages. It
gives samplers what they need (``ndim``, ``names``, ``prior_transform``,
``log_prior`` and ``log_likelihood``), and gives people the whole forward model in one
call:

.. code-block:: python

    posterior = Posterior(params=[...], stages=[...], derived=[("sim", "xHI")])
    out = posterior.forward(theta)  # every output of every stage
    lnl, blobs = posterior.log_likelihood(theta)

The log-likelihood is the sum of every stage output whose type is
:class:`LogLikelihood`, and the blobs are the outputs listed in ``derived``. The
simplest way to add one is a :class:`LikelihoodTerm`, which scores a model output
against data with a :mod:`~yabf.likelihoods` likelihood, after an optional chain of
instrument stages of its own; ``posterior.simulate(theta, rng)`` then draws mock data
from the same likelihoods.

Both methods take one point, ``theta`` of shape ``(ndim,)``, or a batch of points,
of shape ``(n, ndim)``. A batch is evaluated with :meth:`~yabf.pipeline.Pipeline.batch`:
stages that declare ``batched = True`` see the whole batch at once, and the others
are called once per point. Both give the same results.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property

import attrs
import numpy as np
from numpy.typing import ArrayLike

from .likelihoods import Likelihood, ParametrisedLikelihood
from .params import DerivedParam, Param, ParamSpace
from .pipeline import Output, Pipeline
from .stages import One, ParamSpec, Stage, call_stage


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


def _as_model(model: One | type) -> One:
    """Return the requirement for the scored model; a type is the output "model"."""
    if isinstance(model, One):
        return model
    if isinstance(model, type):
        return One(model, name="model")
    raise TypeError(f"The model to score must be a type or One(...), got {model!r}.")


def _forward_stage(inst: LikelihoodTerm, att: attrs.Attribute, value: tuple) -> None:
    """Check that each forward stage maps one input to one output."""
    for stage in value:
        if len(stage.requires) != 1 or len(stage.provides) != 1:
            raise ValueError(
                f"Forward stage {stage.name!r} of {inst.name!r} must take one input "
                "and provide one output, the model before and after it."
            )


@attrs.frozen(eq=False)
class LikelihoodTerm:
    """A stage that scores one model output against data with a likelihood.

    It takes the model output to score, passes it through the ``forward`` stages in
    order (instrument effects, such as a window function, applied for this data alone),
    and provides the log-likelihood of ``data`` given the result as ``"lnl"``. The
    model as scored is also provided, as ``"scored"``.

    Because the forward stages run inside the term, several terms can apply their own
    instrument effects to one shared model output without competing to provide it.

    Parameters
    ----------
    name
        The stage name.
    likelihood
        The likelihood, from :mod:`yabf.likelihoods` or any with the same interface.
        If it has parameters of its own (``parameters`` and ``with_params``), the term
        takes them and passes them on.
    data
        The data, as values or an object with ``values``.
    model
        The model output to score: a type, for the output of that type named
        ``"model"``, or a :class:`~yabf.stages.One`, such as ``One(Spectrum,
        name="sky")``.
    forward
        Stages applied to the model before scoring, each taking one input and
        providing one output of the same type.
    """

    name: str
    likelihood: Likelihood
    data: object
    model: One = attrs.field(converter=_as_model)
    forward: tuple[Stage, ...] = attrs.field(
        factory=tuple, converter=tuple, validator=_forward_stage, kw_only=True
    )

    def __attrs_post_init__(self) -> None:
        """Check that stages sharing a parameter agree on what it accepts."""
        _ = self.parameters

    @cached_property
    def parameters(self) -> Mapping[str, ParamSpec]:
        """The parameters of the forward stages and of the likelihood."""
        sources = [*self.forward]
        if isinstance(self.likelihood, ParametrisedLikelihood):
            sources.append(self.likelihood)
        out: dict[str, ParamSpec] = {}
        for source in sources:
            for k, spec in source.parameters.items():
                if k in out and out[k] != spec:
                    raise ValueError(
                        f"{self.name!r} takes {k!r} twice, with different specs: "
                        f"{out[k]} and {spec}."
                    )
                out[k] = spec
        return out

    @property
    def requires(self) -> dict[str, One]:
        """The model output that the term scores."""
        return {"model": self.model}

    @property
    def provides(self) -> dict[str, type]:
        """The log-likelihood, and the model as scored."""
        return {"lnl": LogLikelihood, "scored": self.model.type}

    def forward_model(self, params: Mapping[str, object], model: object) -> object:
        """Pass ``model`` through the forward stages, in order."""
        for stage in self.forward:
            (inp,) = stage.requires
            (out,) = stage.provides
            model = call_stage(stage, params, {inp: model})[out]
        return model

    def likelihood_at(self, params: Mapping[str, object]) -> Likelihood:
        """Return the likelihood at the values of its own parameters in ``params``."""
        if isinstance(self.likelihood, ParametrisedLikelihood):
            own = {k: params[k] for k in self.likelihood.parameters}
            return self.likelihood.with_params(own)
        return self.likelihood

    def compute(self, params: Mapping[str, object], model: object) -> dict:
        """Forward the model, and score the data against it."""
        model = self.forward_model(params, model)
        lnl = self.likelihood_at(params).loglike(model, self.data)
        return {"lnl": LogLikelihood(lnl), "scored": model}

    def check(self, params: Mapping[str, object]) -> None:
        """Run the checks of the forward stages that define one."""
        for stage in self.forward:
            if hasattr(stage, "check"):
                stage.check({k: params[k] for k in stage.parameters if k in params})

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
        out = self.forward(theta, self.loglike_outputs + self.derived)
        if theta.ndim == 1:
            return (
                float(sum(out[k] for k in self.loglike_outputs)),
                {d: out[d] for d in self.derived},
            )
        terms = [np.asarray(out[k], dtype=float) for k in self.loglike_outputs]
        lnl = np.sum(terms, axis=0)
        return lnl, {d: np.asarray(out[d]) for d in self.derived}

    def simulate(self, theta: ArrayLike, rng: np.random.Generator) -> dict[str, object]:
        """Draw mock data at one point ``theta``, for each likelihood term, by name.

        Each term's likelihood draws data from the model it would score, so a mock is
        distributed as the likelihood assumes. Randomness inside the forward model,
        such as a simulator's seed, is not redrawn: it is set by the parameters (see
        :attr:`random_inputs`).
        """
        theta = self._check_theta(theta)
        if theta.ndim != 1:
            raise ValueError("simulate draws mock data at one point at a time.")
        models = self.forward(theta, [(t.name, "scored") for t in self.likelihoods])
        params = self.params.to_model(theta)
        return {
            t.name: t.likelihood_at(t.parameter_values(params)).simulate(
                models[t.name, "scored"], rng
            )
            for t in self.likelihoods
        }

    @property
    def random_inputs(self) -> dict[str, str]:
        """How each random input of a stage is set, by name.

        Each is ``"sampled"``, ``"fixed"``, ``"tied"`` or ``"default"``. An input is
        random when a stage's :class:`~yabf.stages.ParamSpec` says so, as for a
        simulator's seed. Leaving it at its default or fixing it conditions the
        posterior on one realisation; sampling it treats the realisation as noise.
        """
        by_feeds = {p.feeds: p for p in self.params.params}
        out = {}
        for stage in self.pipeline.graph.stages:
            for k, spec in stage.parameters.items():
                if spec.random:
                    p = by_feeds.get(k)
                    out[k] = (
                        "default"
                        if p is None
                        else "sampled"
                        if p.is_sampled
                        else "fixed"
                        if p.is_fixed
                        else "tied"
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
