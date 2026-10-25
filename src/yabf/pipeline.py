"""Evaluating a graph of stages.

A :class:`Pipeline` computes the outputs of a resolved stage graph for given values of
the model parameters. Within one evaluation it:

- computes only the stages that the requested outputs depend on, each of them once;
- passes outputs from stage to stage by reference, never copying them;
- drops each output as soon as the last stage that needs it has run, unless that
  output was requested, so that large products such as simulated fields do not
  outlive their use.

Across evaluations, it keeps the outputs of stages that set ``cacheable = True``, in a
bounded least-recently-used cache. A stage's cache key is made of the parameter values
it consumes and the keys of the stages that feed it, so it changes whenever anything
upstream of the stage changes. A cache hit skips the stage and everything upstream of
it that nothing else needs: if only the parameters of stages downstream of a cached
observable change, the simulation is not run again. Caching is reserved for small
outputs such as observables; a simulator's products are never cached.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Iterable, Mapping
from typing import Any

import attrs
import numpy as np

from .params import ParamSpace
from .stages import Stage, StageGraph, call_stage, resolve

Output = tuple[str, str]


def _freeze(value):
    """Return a hashable stand-in for a parameter value, for use in cache keys."""
    if isinstance(value, np.ndarray):
        return (value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    return value


@attrs.frozen
class Pipeline:
    """Computes the outputs of a stage graph, sharing and releasing them as it goes.

    Build one from stages with :meth:`build`, then call it with the values of the
    model parameters and the outputs wanted, each named by its ``(stage, slot)``
    pair. Stages must not change between evaluations, and must not modify their
    inputs.

    Parameters
    ----------
    graph
        The resolved stage graph.
    cache_size
        The number of stage results kept across evaluations, over all cacheable
        stages. Zero turns the cache off.
    """

    graph: StageGraph
    cache_size: int = attrs.field(
        default=128, kw_only=True, validator=attrs.validators.ge(0)
    )
    _cache: OrderedDict = attrs.field(
        factory=OrderedDict, init=False, repr=False, eq=False
    )

    @classmethod
    def build(
        cls,
        stages: Iterable[Stage],
        params: ParamSpace | None = None,
        *,
        cache_size: int = 128,
    ) -> Pipeline:
        """Resolve ``stages`` (see :func:`~yabf.stages.resolve`) into a pipeline."""
        return cls(resolve(stages, params), cache_size=cache_size)

    @property
    def outputs(self) -> tuple[Output, ...]:
        """Every output of the pipeline, as ``(stage, slot)`` pairs."""
        return tuple((s.name, slot) for s in self.graph.stages for slot in s.provides)

    @property
    def final_outputs(self) -> tuple[Output, ...]:
        """The outputs that no stage consumes: those computed by default."""
        consumed = {(link.provider, link.provided) for link in self.graph.links}
        return tuple(o for o in self.outputs if o not in consumed)

    def clear_cache(self) -> None:
        """Forget every result kept across evaluations."""
        self._cache.clear()

    def __call__(
        self, params: Mapping[str, Any], want: Iterable[Output] | None = None
    ) -> dict[Output, Any]:
        """Compute the outputs in ``want`` (default: the final outputs) at ``params``.

        ``params`` maps model parameter names to values for one evaluation; each stage
        receives those it consumes.
        """
        want = self._check_want(want)
        wanted = set(want)
        by_name = {s.name: s for s in self.graph.stages}
        keys: dict[str, tuple] = {}

        def key(name: str) -> tuple:
            if name not in keys:
                consumed = sorted(
                    (k, _freeze(v))
                    for k, v in params.items()
                    if k in by_name[name].consumes
                )
                upstream = sorted(
                    (slot, key(provider))
                    for slot, (provider, _) in self.graph.inputs_of(name).items()
                )
                keys[name] = (name, tuple(consumed), tuple(upstream))
            return keys[name]

        # Find the stages to run, walking up from the wanted outputs and stopping at
        # cached results.
        run, hits = set(), {}

        def need(name: str) -> None:
            if name in run or name in hits:
                return
            if self._cacheable(by_name[name]) and key(name) in self._cache:
                self._cache.move_to_end(key(name))
                hits[name] = self._cache[key(name)]
                return
            run.add(name)
            for provider, _ in self.graph.inputs_of(name).values():
                need(provider)

        for name, _ in want:
            need(name)

        remaining = Counter(
            src for name in run for src in self.graph.inputs_of(name).values()
        )
        store = {
            (name, slot): value
            for name, outs in hits.items()
            for slot, value in outs.items()
            if (name, slot) in wanted or remaining[name, slot]
        }
        for stage in self.graph.stages:
            if stage.name in run:
                self._run(stage, params, store, remaining, wanted, key)
        return {w: store[w] for w in want}

    def _cacheable(self, stage: Stage) -> bool:
        return self.cache_size > 0 and getattr(stage, "cacheable", False)

    def _run(self, stage, params, store, remaining, wanted, key) -> None:
        """Compute one stage, then release the inputs that nothing else needs."""
        links = self.graph.inputs_of(stage.name)
        out = call_stage(
            stage, params, {slot: store[src] for slot, src in links.items()}
        )
        if self._cacheable(stage):
            self._cache[key(stage.name)] = out
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        for slot, value in out.items():
            if (stage.name, slot) in wanted or remaining[stage.name, slot]:
                store[stage.name, slot] = value
        for src in links.values():
            remaining[src] -= 1
            if not remaining[src] and src not in wanted:
                del store[src]

    def _check_want(self, want) -> tuple[Output, ...]:
        if want is None:
            return self.final_outputs
        want = tuple(tuple(w) for w in want)
        known = set(self.outputs)
        unknown = [w for w in want if w not in known]
        if unknown:
            raise ValueError(
                f"The pipeline has no outputs {unknown}; it has {sorted(known)}."
            )
        return want
