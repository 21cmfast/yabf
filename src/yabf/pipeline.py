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
it takes and the keys of the stages that feed it, so it changes whenever anything
upstream of the stage changes. A cache hit skips the stage and everything upstream of
it that nothing else needs: if only the parameters of stages downstream of a cached
observable change, the simulation is not run again. Caching is reserved for small
outputs such as observables; a simulator's products are never cached.
"""

from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Any

import attrs
import numpy as np
from numpy.typing import ArrayLike

from .params import ParamSpace
from .stages import Stage, StageGraph, call_stage, call_stage_batch, resolve

Output = tuple[str, str]


def _freeze(value: object) -> Hashable:
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
    model parameters and the outputs wanted, each named by its ``(stage, output)``
    pair. Stages must not change between evaluations, and must not modify their
    inputs. A pipeline is not thread-safe.

    Calling the pipeline evaluates one point. :meth:`batch` evaluates many: stages that
    declare ``batched = True`` are called once for the whole batch, and the others
    once per point.

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
        """Every output of the pipeline, as ``(stage, output)`` pairs."""
        return tuple((s.name, out) for s in self.graph.stages for out in s.provides)

    @property
    def final_outputs(self) -> tuple[Output, ...]:
        """The outputs that no stage takes as input: those computed by default."""
        consumed = {(link.provider, link.output) for link in self.graph.links}
        return tuple(o for o in self.outputs if o not in consumed)

    def clear_cache(self) -> None:
        """Forget every result kept across evaluations."""
        self._cache.clear()

    def __call__(
        self, params: Mapping[str, Any], want: Iterable[Output] | None = None
    ) -> dict[Output, Any]:
        """Compute the outputs in ``want`` (default: the final outputs) at ``params``.

        ``params`` maps model parameter names to values for one evaluation; each stage
        receives those it takes.
        """
        want = self._check_want(want)
        wanted = set(want)
        by_name = {s.name: s for s in self.graph.stages}
        keys: dict[str, tuple] = {}

        def key(name: str) -> tuple:
            """Stage ``name``'s cache key: its consumed values and its inputs' keys."""
            if name not in keys:
                consumed = sorted(
                    (k, _freeze(v))
                    for k, v in params.items()
                    if k in by_name[name].parameters
                )
                upstream = sorted(
                    (k, tuple(key(provider) for provider, _ in srcs))
                    for k, srcs in self.graph.inputs_of(name).items()
                )
                keys[name] = (name, tuple(consumed), tuple(upstream))
            return keys[name]

        # Find the stages to run, walking up from the wanted outputs and stopping at
        # cached results.
        run, hits = set(), {}

        def need(name: str) -> None:
            """Mark stage ``name``, and what it needs, to run unless it is cached."""
            if name in run or name in hits:
                return
            cached = (
                self._cache.get(key(name)) if self._cacheable(by_name[name]) else None
            )
            if cached is not None:
                self._cache.move_to_end(key(name))
                hits[name] = cached
                return
            run.add(name)
            for srcs in self.graph.inputs_of(name).values():
                for provider, _ in srcs:
                    need(provider)

        for name, _ in want:
            need(name)

        remaining = Counter(
            src
            for name in run
            for srcs in self.graph.inputs_of(name).values()
            for src in srcs
        )
        store = {
            (name, out): value
            for name, outs in hits.items()
            for out, value in outs.items()
            if (name, out) in wanted or remaining[name, out]
        }
        for stage in self.graph.stages:
            if stage.name in run:
                self._run(stage, params, store, remaining, wanted, key)
        return {w: store[w] for w in want}

    def _cacheable(self, stage: Stage) -> bool:
        """Whether results of ``stage`` are kept across evaluations."""
        return self.cache_size > 0 and getattr(stage, "cacheable", False)

    def _run(
        self,
        stage: Stage,
        params: Mapping[str, Any],
        store: dict[Output, object],
        remaining: Counter[Output],
        wanted: set[Output],
        key: Callable[[str], tuple],
    ) -> None:
        """Compute one stage, then release the inputs that nothing else needs."""
        inputs = self.graph.gather(stage.name, store)
        if getattr(stage, "batched", False):
            batch = {
                k: np.expand_dims(params[k], 0) for k in stage.parameters if k in params
            }
            outs = call_stage_batch(
                stage, batch, {k: (v,) for k, v in inputs.items()}, 1
            )
            out = {k: v[0] for k, v in outs.items()}
        else:
            out = call_stage(stage, params, inputs)
        del inputs
        if self._cacheable(stage):
            self._cache[key(stage.name)] = out
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        self._keep(stage, out, store, remaining, wanted)

    def _keep(
        self,
        stage: Stage,
        out: Mapping[str, object],
        store: dict[Output, object],
        remaining: Counter[Output],
        wanted: set[Output],
    ) -> None:
        """Store the outputs of ``stage`` that are needed, and release spent inputs."""
        for k, value in out.items():
            if (stage.name, k) in wanted or remaining[stage.name, k]:
                store[stage.name, k] = value
        for srcs in self.graph.inputs_of(stage.name).values():
            for src in srcs:
                remaining[src] -= 1
                if not remaining[src] and src not in wanted:
                    del store[src]

    def batch(
        self,
        params: Mapping[str, ArrayLike],
        n: int,
        want: Iterable[Output] | None = None,
    ) -> dict[Output, tuple[object, ...]]:
        """Compute the outputs in ``want`` (default: the final outputs) at ``n`` points.

        ``params`` maps model parameter names to arrays whose leading axis, of length
        ``n``, runs over the points. Each output is returned as a tuple of ``n`` values.

        A stage that declares ``batched = True`` is called once with the whole batch;
        the others are called once per point. Consecutive unbatched stages run point by
        point, so that each point's intermediate products are released before the next
        point starts. The cache is neither read nor filled.
        """
        want = self._check_want(want)
        wanted = set(want)
        params = {k: np.asarray(v) for k, v in params.items()}
        wrong = sorted(k for k, v in params.items() if v.shape[:1] != (n,))
        if wrong:
            raise ValueError(
                f"The parameters {wrong} need a leading batch axis of length {n}."
            )

        run: set[str] = set()
        todo = list({name for name, _ in want})
        while todo:
            name = todo.pop()
            if name not in run:
                run.add(name)
                todo.extend(
                    p for srcs in self.graph.inputs_of(name).values() for p, _ in srcs
                )
        stages = [s for s in self.graph.stages if s.name in run]
        consumers = Counter(
            src
            for name in run
            for srcs in self.graph.inputs_of(name).values()
            for src in srcs
        )
        stores: list[dict[Output, object]] = [{} for _ in range(n)]
        remaining = [consumers.copy() for _ in range(n)]

        # Split the stages, in order, into batched stages and runs of unbatched ones.
        segments: list[list[Stage]] = []
        for stage in stages:
            if (
                getattr(stage, "batched", False)
                or not segments
                or getattr(segments[-1][0], "batched", False)
            ):
                segments.append([stage])
            else:
                segments[-1].append(stage)

        for segment in segments:
            if getattr(segment[0], "batched", False):
                stage = segment[0]
                per_point = [self.graph.gather(stage.name, st) for st in stores]
                inputs = {
                    k: tuple(p[k] for p in per_point)
                    for k in self.graph.inputs_of(stage.name)
                }
                del per_point
                outs = call_stage_batch(stage, params, inputs, n)
                del inputs
                for i in range(n):
                    out = {k: v[i] for k, v in outs.items()}
                    self._keep(stage, out, stores[i], remaining[i], wanted)
                del outs
                continue
            for i in range(n):
                point = {k: v[i] for k, v in params.items()}
                for stage in segment:
                    out = call_stage(
                        stage, point, self.graph.gather(stage.name, stores[i])
                    )
                    self._keep(stage, out, stores[i], remaining[i], wanted)
                    del out
        return {w: tuple(st[w] for st in stores) for w in want}

    def _check_want(self, want: Iterable[Output] | None) -> tuple[Output, ...]:
        """Return the requested outputs as a tuple, checking the pipeline has them."""
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
