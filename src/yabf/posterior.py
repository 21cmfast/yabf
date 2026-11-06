"""The posterior: parameters, stages and likelihoods behind one sampler-facing object.

A :class:`Posterior` joins a :class:`~yabf.params.ParamSpace` to a graph of stages. It
gives samplers what they need (``ndim``, ``names``, ``prior_transform``,
``log_prior`` and ``log_likelihood``), and gives people the whole forward model in one
call:

.. code-block:: python

    posterior = Posterior(params=[...], stages=[...], derived=[("sim", "xHI")])
    out = posterior.forward(theta)  # every output of every stage
    lnl, blobs = posterior.log_likelihood(theta)

The log-likelihood is the sum of every stage output whose slot type is
:class:`LogLikelihood`, and the blobs are the outputs listed in ``derived``.

Both methods take one point, ``theta`` of shape ``(ndim,)``, or a batch of points,
of shape ``(n, ndim)``. A batch is evaluated with :meth:`~yabf.pipeline.Pipeline.batch`:
stages that declare ``batched = True`` see the whole batch at once, and the others
are called once per point. Both give the same results.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import attrs
import numpy as np
from numpy.typing import ArrayLike

from .params import Param, ParamSpace
from .pipeline import Output, Pipeline
from .stages import Stage, as_slot


class LogLikelihood(float):
    """A log-likelihood term, as a stage output.

    A stage that provides a slot of this type contributes that output to the
    posterior's log-likelihood. Return the value as ``LogLikelihood(x)``.
    """


def _as_param_space(params: ParamSpace | Iterable[Param]) -> ParamSpace:
    """Return ``params`` as a :class:`~yabf.params.ParamSpace`."""
    return params if isinstance(params, ParamSpace) else ParamSpace(params)


def _as_outputs(outputs: Iterable[Sequence[str]]) -> tuple[Output, ...]:
    """Return ``(stage, slot)`` pairs as a tuple of tuples."""
    return tuple((stage, slot) for stage, slot in outputs)


@attrs.frozen(eq=False)
class Posterior:
    """The posterior of a set of parameters, given stages that compute a likelihood.

    Parameters
    ----------
    params
        The parameters, as a :class:`~yabf.params.ParamSpace` or a sequence of
        :class:`~yabf.params.Param`.
    stages
        The stages of the forward model. At least one must provide a
        :class:`LogLikelihood` slot. They are resolved, and checked against
        ``params``, when the posterior is built (see :func:`~yabf.stages.resolve`).
    derived
        Outputs to return as blobs with the log-likelihood, each named by its
        ``(stage, slot)`` pair.
    cache_size
        The number of stage results kept across single-point evaluations; see
        :class:`~yabf.pipeline.Pipeline`.
    """

    params: ParamSpace = attrs.field(converter=_as_param_space)
    stages: tuple[Stage, ...] = attrs.field(converter=tuple)
    derived: tuple[Output, ...] = attrs.field(
        default=(), converter=_as_outputs, kw_only=True
    )
    cache_size: int = attrs.field(default=128, kw_only=True)
    pipeline: Pipeline = attrs.field(init=False, repr=False)

    @pipeline.default
    def _pipeline_default(self) -> Pipeline:
        """Resolve the stages against the parameters."""
        return Pipeline.build(self.stages, self.params, cache_size=self.cache_size)

    def __attrs_post_init__(self) -> None:
        """Check that some stage provides a log-likelihood, and that blobs exist."""
        if not self.likelihoods:
            raise ValueError(
                "No stage provides a LogLikelihood slot, so the posterior has no "
                "likelihood."
            )
        unknown = [d for d in self.derived if d not in self.pipeline.outputs]
        if unknown:
            raise ValueError(f"The derived outputs {unknown} are not stage outputs.")

    @property
    def likelihoods(self) -> tuple[Output, ...]:
        """The outputs summed into the log-likelihood, as ``(stage, slot)`` pairs."""
        return tuple(
            (s.name, k)
            for s in self.pipeline.graph.stages
            for k, slot in s.provides.items()
            if issubclass(as_slot(slot).type, LogLikelihood)
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

        ``want`` restricts the result to some ``(stage, slot)`` outputs. For a batch
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
        out = self.forward(theta, self.likelihoods + self.derived)
        if theta.ndim == 1:
            return (
                float(sum(out[k] for k in self.likelihoods)),
                {d: out[d] for d in self.derived},
            )
        terms = [np.asarray(out[k], dtype=float) for k in self.likelihoods]
        lnl = np.sum(terms, axis=0)
        return lnl, {d: np.asarray(out[d]) for d in self.derived}

    def _check_theta(self, theta: ArrayLike) -> np.ndarray:
        """Return ``theta`` as an array of shape ``(ndim,)`` or ``(n, ndim)``."""
        theta = np.asarray(theta, dtype=float)
        if theta.ndim not in (1, 2) or theta.shape[-1] != self.ndim:
            raise ValueError(
                f"theta must have shape ({self.ndim},) for one point or (n, "
                f"{self.ndim}) for a batch, got {theta.shape}."
            )
        return theta
