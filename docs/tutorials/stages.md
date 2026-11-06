---
file_format: mystnb
kernelspec:
  name: python3
---

# Stages and the pipeline

A model in yabf is a graph of *stages*. Each stage is one pure computation: it
declares the model parameters it consumes, the typed inputs it requires and the typed
outputs it provides, and its `compute` method turns the first two into the third.
Stages never name each other. yabf connects each input to the output of matching type,
so stages can be swapped, added and reused without changing the others.

In this tutorial we build a model of the sky-averaged radio spectrum between 50 and
100 MHz: a bright foreground, a faint absorption trough from the 21-cm line, and an
instrument that measures their sum.

```{code-cell} python
from collections.abc import Mapping
from typing import ClassVar

import attrs
import matplotlib.pyplot as plt
import numpy as np

from yabf.params import Param, ParamSpace
from yabf.pipeline import Pipeline
from yabf.stages import All, GraphError, ParamSpec, Slot, resolve

NU = np.linspace(50.0, 100.0, 101)  # frequencies, in MHz


@attrs.frozen
class Spectrum:
    """A brightness temperature in K, at the frequencies NU."""

    values: np.ndarray
```

## A first stage

The foreground is a power law in frequency. Its stage consumes two parameters, each
described by a {class}`~yabf.stages.ParamSpec`: a default value, the values the stage
accepts (its *domain*) and what the parameter means. It requires no inputs, and
provides one {class}`~yabf.stages.Slot`, a `Spectrum` tagged `"component"`:

```{code-cell} python
@attrs.frozen
class Foreground:
    """A power-law foreground."""

    name: str = "foreground"
    consumes: ClassVar = {
        "fg_amp": ParamSpec(1000.0, domain=(0, np.inf), description="At 75 MHz, in K."),
        "fg_index": ParamSpec(-2.5, description="The spectral index."),
    }
    requires: ClassVar = {}
    provides: ClassVar = {"spectrum": Slot(Spectrum, "component")}

    def compute(self, params: Mapping[str, float]) -> dict[str, Spectrum]:
        """The foreground spectrum."""
        return {
            "spectrum": Spectrum(params["fg_amp"] * (NU / 75) ** params["fg_index"])
        }
```

Any object with these attributes is a stage; frozen [attrs](https://www.attrs.org)
classes make it easy to give stages settings of their own, and keep them immutable.
A stage can be called on its own:

```{code-cell} python
Foreground().compute({"fg_amp": 1000.0, "fg_index": -2.5})["spectrum"].values[:3]
```

## More stages

The 21-cm absorption trough is a Gaussian in frequency. In a real analysis this
stage would run a simulation, which is slow, so it records each time it runs, and
declares `cacheable = True`: the pipeline may then reuse its output when its
parameters have not changed.

```{code-cell} python
@attrs.frozen
class Trough:
    """A Gaussian absorption trough, standing in for a slow simulation."""

    name: str = "signal"
    runs: list = attrs.field(factory=list, eq=False)
    cacheable: ClassVar = True
    consumes: ClassVar = {
        "depth": ParamSpec(0.5, domain=(0, np.inf), description="In K."),
        "centre": ParamSpec(78.0, description="In MHz."),
        "width": ParamSpec(5.0, domain=(0, np.inf), description="In MHz."),
    }
    requires: ClassVar = {}
    provides: ClassVar = {"spectrum": Slot(Spectrum, "component")}

    def compute(self, params: Mapping[str, float]) -> dict[str, Spectrum]:
        """The trough, recording the run."""
        self.runs.append(dict(params))
        x = (NU - params["centre"]) / params["width"]
        return {"spectrum": Spectrum(-params["depth"] * np.exp(-0.5 * x**2))}
```

The sky is the sum of every component. Rather than naming the foreground and signal
stages, it requires {class}`~yabf.stages.All` the spectra tagged `"component"`, so a
third component would be added without changing it:

```{code-cell} python
@attrs.frozen
class Sky:
    """The sum of every component of the sky."""

    name: str = "sky"
    consumes: ClassVar = {}
    requires: ClassVar = {"components": All(Spectrum, "component")}
    provides: ClassVar = {"sky": Spectrum}

    def compute(
        self, params: Mapping[str, float], components: tuple[Spectrum, ...]
    ) -> dict[str, Spectrum]:
        """The sum of the components."""
        return {"sky": Spectrum(sum(c.values for c in components))}
```

Finally, the instrument multiplies the sky by a gain that is known only roughly. Its
input is the untagged `Spectrum` that the sky provides:

```{code-cell} python
@attrs.frozen
class Gain:
    """An instrument with an uncertain gain."""

    name: str = "gain"
    consumes: ClassVar = {"gain": ParamSpec(1.0, domain=(0, np.inf))}
    requires: ClassVar = {"sky": Spectrum}
    provides: ClassVar = {"observed": Slot(Spectrum, "observed")}

    def compute(
        self, params: Mapping[str, float], sky: Spectrum
    ) -> dict[str, Spectrum]:
        """The sky, as the instrument measures it."""
        return {"observed": Spectrum(params["gain"] * sky.values)}
```

## Resolving the graph

{func}`~yabf.stages.resolve` links each input to the output that satisfies it, and
checks the parameters against what each stage accepts. Stages may be given in any
order:

```{code-cell} python
signal = Trough()
stages = [Gain(), Sky(), Foreground(), signal]
space = ParamSpace(
    [
        Param("fg_amp", prior=(500.0, 1500.0)),
        Param("fg_index", prior=(-3.0, -2.0)),
        Param("depth", prior=(0.0, 2.0)),
        Param("centre", prior=(60.0, 90.0)),
        Param("width", prior=(1.0, 15.0)),
        Param("gain", prior=(0.95, 1.05)),
    ]
)
graph = resolve(stages, space)
for link in graph.links:
    print(f"{link.stage}.{link.slot}  <-  {link.provider}.{link.provided}")
```

```{code-cell} python
[s.name for s in graph.stages]
```

The stages are put in the order they must run. Problems are found before anything is
computed, and all of them are reported at once. Here a prior lets the width of the
trough become negative, which the stage does not accept, and without the sky stage
nothing provides the instrument's input:

```{code-cell} python
broken = ParamSpace([Param("width", prior=(-1.0, 10.0))])
try:
    resolve([Foreground(), Trough(), Gain()], broken)
except GraphError as e:
    print(e)
```

## Running the pipeline

A {class}`~yabf.pipeline.Pipeline` computes the outputs at given values of the model
parameters. Parameters not given take their defaults. By default it returns the final
outputs, but any output can be asked for by its `(stage, slot)` pair:

```{code-cell} python
pipeline = Pipeline.build(stages, space)
values = space.to_model(space.fiducial)
out = pipeline(
    values,
    want=[("foreground", "spectrum"), ("signal", "spectrum"), ("gain", "observed")],
)

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
top.plot(NU, out["gain", "observed"].values, label="observed")
top.set_ylabel("T [K]")
top.legend()
bottom.plot(
    NU,
    out["gain", "observed"].values - out["foreground", "spectrum"].values,
    label="observed - foreground",
)
bottom.plot(NU, out["signal", "spectrum"].values, "--", label="signal")
bottom.set_xlabel(r"$\nu$ [MHz]")
bottom.set_ylabel("T [K]")
bottom.legend();
```

The observed spectrum is dominated by the foreground: the trough is less than a
thousandth of it. With a gain of one, subtracting the foreground leaves the signal.

### Caching

The pipeline has kept the output of the cacheable trough stage. Changing only the
gain, which is downstream of it, does not run it again; changing its own parameters
does:

```{code-cell} python
signal.runs.clear()
pipeline({**values, "gain": 1.01})
pipeline({**values, "gain": 0.99})
print(len(signal.runs), "runs after changing the gain")
pipeline({**values, "width": 6.0})
print(len(signal.runs), "run after changing the width")
```

Only stages with small outputs, such as summary statistics, should be cacheable; the
cache never holds a simulator's large products.

### Batches of points

Samplers often propose many points at once. {meth}`~yabf.pipeline.Pipeline.batch`
evaluates a batch, with each parameter given as an array over the points. A stage
that can compute the whole batch at once declares `batched = True`, receives arrays,
and returns one output per point; the others are called once per point.

```{code-cell} python
@attrs.frozen
class BatchedForeground(Foreground):
    """The power-law foreground, computed for a whole batch at once."""

    batched: ClassVar = True

    def compute(self, params: Mapping[str, np.ndarray]) -> dict[str, list[Spectrum]]:
        """The foreground spectrum at every point of the batch."""
        amp = np.asarray(params["fg_amp"])[:, None]
        index = np.asarray(params["fg_index"])[:, None]
        return {"spectrum": [Spectrum(t) for t in amp * (NU / 75) ** index]}


rng = np.random.default_rng(1)
theta = space.sample_prior(500, rng)
batch = space.to_model(theta)

slow = Pipeline.build(stages, space).batch(batch, len(theta))
fast = Pipeline.build([Gain(), Sky(), BatchedForeground(), Trough()], space).batch(
    batch, len(theta)
)
np.allclose(
    [s.values for s in slow["gain", "observed"]],
    [s.values for s in fast["gain", "observed"]],
)
```

Both give the same result, one value per point. Stages that are called point by point
release each point's outputs before the next point starts, so a batch never holds
more than one simulation in memory at a time.

The next tutorial turns this model into a posterior.
