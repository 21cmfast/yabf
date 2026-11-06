---
file_format: mystnb
kernelspec:
  name: python3
---

# Stages and the pipeline

A model in yabf is a graph of *stages*. Each stage is one pure computation: it
declares the model parameters it takes, the inputs it requires and the outputs it
provides, and its `compute` method turns the first two into the third. Every output
has a name and a type, and an input asks for an output by its name and type, never by
the stage that makes it. So stages can be swapped, added and reused without changing
the others.

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
from yabf.stages import All, GraphError, One, ParamSpec, resolve

NU = np.linspace(50.0, 100.0, 101)  # frequencies, in MHz


@attrs.frozen
class Spectrum:
    """A brightness temperature in K, at the frequencies NU."""

    values: np.ndarray
```

## A first stage

The foreground is a power law in frequency. Its stage takes two `parameters`, each
described by a {class}`~yabf.stages.ParamSpec`: a default value, the values the stage
accepts (its *domain*) and what the parameter means. It requires no inputs, and
provides one output: a `Spectrum` named `"component"`, since it is one component of
the sky.

```{code-cell} python
@attrs.frozen
class Foreground:
    """A power-law foreground."""

    name: str = "foreground"
    parameters: ClassVar = {
        "fg_amp": ParamSpec(1000.0, domain=(0, np.inf), description="At 75 MHz, in K."),
        "fg_index": ParamSpec(-2.5, description="The spectral index."),
    }
    requires: ClassVar = {}
    provides: ClassVar = {"component": Spectrum}

    def compute(self, params: Mapping[str, float]) -> dict[str, Spectrum]:
        """The foreground spectrum."""
        return {
            "component": Spectrum(params["fg_amp"] * (NU / 75) ** params["fg_index"])
        }
```

Any object with these attributes is a stage; frozen [attrs](https://www.attrs.org)
classes make it easy to give stages settings of their own, and keep them immutable.
`compute` receives the parameters' values in its first argument, and returns a
dictionary of the outputs, by name. A stage can be called on its own:

```{code-cell} python
Foreground().compute({"fg_amp": 1000.0, "fg_index": -2.5})["component"].values[:3]
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
    parameters: ClassVar = {
        "depth": ParamSpec(0.5, domain=(0, np.inf), description="In K."),
        "centre": ParamSpec(78.0, description="In MHz."),
        "width": ParamSpec(5.0, domain=(0, np.inf), description="In MHz."),
    }
    requires: ClassVar = {}
    provides: ClassVar = {"component": Spectrum}

    def compute(self, params: Mapping[str, float]) -> dict[str, Spectrum]:
        """The trough, recording the run."""
        self.runs.append(dict(params))
        x = (NU - params["centre"]) / params["width"]
        return {"component": Spectrum(-params["depth"] * np.exp(-0.5 * x**2))}
```

The sky is the sum of every component. Rather than naming the foreground and signal
stages, it requires {class}`~yabf.stages.All` the spectra named `"component"`, so a
third component would be added without changing it. It receives them, as a tuple, in
its argument `components`, and provides a `Spectrum` named `"sky"`:

```{code-cell} python
@attrs.frozen
class Sum:
    """The sum of every component of the sky."""

    name: str = "sum"
    parameters: ClassVar = {}
    requires: ClassVar = {"components": All(Spectrum, name="component")}
    provides: ClassVar = {"sky": Spectrum}

    def compute(
        self, params: Mapping[str, float], components: tuple[Spectrum, ...]
    ) -> dict[str, Spectrum]:
        """The sum of the components."""
        return {"sky": Spectrum(sum(c.values for c in components))}
```

Finally, the instrument multiplies the sky by a gain that is known only roughly. It
requires the `Spectrum` named `"sky"`, which it receives in its argument `sky`, and
provides a `Spectrum` named `"observed"`:

```{code-cell} python
@attrs.frozen
class Gain:
    """An instrument with an uncertain gain."""

    name: str = "gain"
    parameters: ClassVar = {"gain": ParamSpec(1.0, domain=(0, np.inf))}
    requires: ClassVar = {"sky": Spectrum}
    provides: ClassVar = {"observed": Spectrum}

    def compute(
        self, params: Mapping[str, float], sky: Spectrum
    ) -> dict[str, Spectrum]:
        """The sky, as the instrument measures it."""
        return {"observed": Spectrum(params["gain"] * sky.values)}
```

## Resolving the graph

{func}`~yabf.stages.resolve` links each input to the outputs that match it, and checks
the parameters against what each stage accepts. Stages may be given in any order;
printing the graph shows them in the order they run, with what each takes, where it
comes from, and what it provides:

```{code-cell} python
signal = Trough()
stages = [Gain(), Sum(), Foreground(), signal]
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
print(graph)
```

An output is written `stage.name`: `signal.component` is the output named
`"component"` of the stage named `"signal"`.

Problems are found before anything is computed, and all of them are reported at once.
Here a prior lets the width of the trough become negative, which the stage does not
accept, and without the sum nothing provides the instrument's input:

```{code-cell} python
broken = ParamSpace([Param("width", prior=(-1.0, 10.0))])
try:
    resolve([Foreground(), Trough(), Gain()], broken)
except GraphError as e:
    print(e)
```

### Asking for inputs

A bare type asks for the output named like the input, as `Gain` does. Two more
requirements cover the other cases. {class}`~yabf.stages.One` asks for the only output
of a type, whatever its name, or for an output of another name; {class}`~yabf.stages.All`
collects every output that matches, of a type and, optionally, some names:

| `requires = {...}` | receives |
|---|---|
| `{"sky": Spectrum}` | the `Spectrum` named `"sky"` |
| `{"measured": One(Spectrum, name="observed")}` | the `Spectrum` named `"observed"`, as `measured` |
| `{"spectrum": One(Spectrum)}` | the only `Spectrum`, whatever its name |
| `{"components": All(Spectrum, name="component")}` | every `Spectrum` named `"component"` |
| `{"spectra": All(Spectrum)}` | every `Spectrum` |

`One` is an error unless exactly one output matches. In this graph there are several
spectra, so a stage asking for "the" spectrum is told which ones it could mean:

```{code-cell} python
@attrs.frozen
class Peak:
    """The frequency at which a spectrum is brightest."""

    name: str = "peak"
    parameters: ClassVar = {}
    requires: ClassVar = {"spectrum": One(Spectrum)}
    provides: ClassVar = {"peak": float}

    def compute(
        self, params: Mapping[str, float], spectrum: Spectrum
    ) -> dict[str, float]:
        """The frequency of the maximum."""
        return {"peak": float(NU[np.argmax(spectrum.values)])}


try:
    resolve([*stages, Peak()])
except GraphError as e:
    print(e)
```

## Running the pipeline

A {class}`~yabf.pipeline.Pipeline` computes the outputs at given values of the model
parameters. Parameters not given take their defaults. By default it returns the final
outputs, but any output can be asked for by its `(stage, name)` pair:

```{code-cell} python
pipeline = Pipeline.build(stages, space)
values = space.to_model(space.fiducial)
out = pipeline(
    values,
    want=[("foreground", "component"), ("signal", "component"), ("gain", "observed")],
)

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
top.plot(NU, out["gain", "observed"].values, label="observed")
top.set_ylabel("T [K]")
top.legend()
bottom.plot(
    NU,
    out["gain", "observed"].values - out["foreground", "component"].values,
    label="observed - foreground",
)
bottom.plot(NU, out["signal", "component"].values, "--", label="signal")
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
        return {"component": [Spectrum(t) for t in amp * (NU / 75) ** index]}


rng = np.random.default_rng(1)
theta = space.sample_prior(500, rng)
batch = space.to_model(theta)

slow = Pipeline.build(stages, space).batch(batch, len(theta))
fast = Pipeline.build([Gain(), Sum(), BatchedForeground(), Trough()], space).batch(
    batch, len(theta)
)
assert np.allclose(
    [s.values for s in slow["gain", "observed"]],
    [s.values for s in fast["gain", "observed"]],
)
```

Both give the same result, one value per point. Stages that are called point by point
release each point's outputs before the next point starts, so a batch never holds
more than one simulation in memory at a time.

The next tutorial turns this model into a posterior.
