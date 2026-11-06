---
file_format: mystnb
kernelspec:
  name: python3
---

# Stages and the pipeline

In this tutorial we build a model of the sky-averaged radio spectrum between 50 and
100 MHz: a bright foreground, a faint absorption trough from the 21-cm line, and an
instrument that measures their sum. Along the way, we see how `yabf` splits a model
into pieces, connects them, and runs them efficiently.

A model in `yabf` is a graph of *stages*. Each stage is one pure computation, which
declares three things: the model parameters it takes, the inputs it requires from
other stages, and the outputs it provides. Its `compute` method then turns the first
two into the third. Every output has a name and a type, and an input asks for an
output by its name and type, never by the stage that makes it. This means that stages
can be swapped, added and reused without changing the others.

We start with a few imports, the grid of frequencies, and a small class to hold a
spectrum. Stages can pass any Python object between them; giving each kind of value
its own type lets `yabf` check that each input receives what it expects.

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

Our first stage computes the foreground, a power law in frequency. To write it, we
declare what it needs and what it makes:

- its `parameters`: the amplitude and the spectral index of the power law. Each is
  described by a {class}`~yabf.stages.ParamSpec`, which gives a default value, the
  values the stage accepts (its *domain*) and what the parameter means;
- its `requires`: nothing, since it is computed from its parameters alone;
- its `provides`: one output, a `Spectrum` named `"component"`, since it is one
  component of the sky.

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

Any object with these attributes is a stage. We use frozen
[attrs](https://www.attrs.org) classes, which make it easy to give stages settings of
their own, and keep them immutable. `compute` receives the values of the parameters in
its first argument, and returns a dictionary of the outputs, by name.

To check that the stage does what we expect, we can call it on its own, without the
rest of the model:

```{code-cell} python
Foreground().compute({"fg_amp": 1000.0, "fg_index": -2.5})["component"].values[:3]
```

## More stages

Next, we need the 21-cm absorption trough, which we model as a Gaussian in frequency.
In a real analysis this stage would run a simulation, which is slow. So that we can
see later when it runs, it records each run in a list. It also declares
`cacheable = True`, which allows the pipeline to reuse its output when its parameters
have not changed. Like the foreground, it provides a `Spectrum` named `"component"`.

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

We now want to add the components up into the sky spectrum. We could write a stage
that requires the foreground and the trough by name, but then adding a third component
would mean changing it. Instead, the stage requires {class}`~yabf.stages.All` the
spectra named `"component"`, however many there are. It receives them, as a tuple, in
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

Finally, we model the instrument, which multiplies the sky by a gain that is known
only roughly. Its stage takes the gain as a parameter, requires the `Spectrum` named
`"sky"`, which it receives in its argument `sky`, and provides a `Spectrum` named
`"observed"`:

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

We now have four stages, and need to connect them into a model. This is what
{func}`~yabf.stages.resolve` does: it links each input to the outputs that match it,
works out an order in which the stages can run, and checks the parameters against
what each stage accepts. We can give it the stages in any order, along with the
parameter space we want to use (built as in the previous tutorial). Printing the
resulting graph shows the stages in the order they run, with what each takes, where
it comes from, and what it provides:

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

Mistakes in a model are best found before anything slow is computed. So `resolve`
checks the whole graph first, and reports every problem at once. To see this, we make
two mistakes on purpose: a prior that lets the width of the trough become negative,
which the stage does not accept, and a model without the sum, so that nothing provides
the instrument's input:

```{code-cell} python
broken = ParamSpace([Param("width", prior=(-1.0, 10.0))])
try:
    resolve([Foreground(), Trough(), Gain()], broken)
except GraphError as e:
    print(e)
```

### Asking for inputs

So far, each input has asked for its output in one of two ways: by its name, as `Gain`
asks for `"sky"`, or by collecting every output of a name, as the sum does. A bare type
asks for the output named like the input. Two more requirements cover the other
cases. {class}`~yabf.stages.One` asks for the only output
of a type, whatever its name, or for an output of another name; {class}`~yabf.stages.All`
collects every output that matches, of a type and, optionally, some names:

| `requires = {...}` | receives |
|---|---|
| `{"sky": Spectrum}` | the `Spectrum` named `"sky"` |
| `{"measured": One(Spectrum, name="observed")}` | the `Spectrum` named `"observed"`, as `measured` |
| `{"spectrum": One(Spectrum)}` | the only `Spectrum`, whatever its name |
| `{"components": All(Spectrum, name="component")}` | every `Spectrum` named `"component"` |
| `{"spectra": All(Spectrum)}` | every `Spectrum` |

`One(Spectrum)` is convenient when a stage needs an input of some kind, but does not
care what it is called. It is an error unless exactly one output matches. In our graph
there are several spectra, so a stage asking for "the" spectrum is told which ones it
could mean:

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

With the graph resolved, we can compute the model. A {class}`~yabf.pipeline.Pipeline`
runs the stages at given values of the model parameters; any parameter not given takes
its default. By default it returns the final outputs, those that no other stage uses,
but we can ask for any output by its `(stage, name)` pair. Here we compute the model at
the fiducial point, and ask for the two components and the observed spectrum, so that
we can plot them:

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
thousandth of it. With a gain of one, subtracting the foreground leaves the signal,
which is exactly the difficulty a real measurement faces.

### Caching

Samplers evaluate the model many times, often changing only some parameters. When a
stage is slow, we would like to avoid running it again when its own inputs have not
changed. This is what `cacheable = True` asks for: the pipeline keeps the outputs of
the trough stage, and reuses them as long as its parameters and inputs are the same.

To see this, we clear the record of the trough's runs, and evaluate the model twice
with different gains, then once with a different width. The gain only affects the
instrument, downstream of the trough, so the trough does not run again until its own
width changes:

```{code-cell} python
signal.runs.clear()
pipeline({**values, "gain": 1.01})
pipeline({**values, "gain": 0.99})
print(len(signal.runs), "runs after changing the gain")
pipeline({**values, "width": 6.0})
print(len(signal.runs), "run after changing the width")
```

Only stages with small outputs, such as summary statistics, should be cacheable: the
cache never holds a simulator's large products.

### Batches of points

Samplers also often propose many points at once, and some stages, such as emulators,
are much faster when they compute many points together. To take advantage of this,
{meth}`~yabf.pipeline.Pipeline.batch` evaluates a whole batch, with each parameter
given as an array over the points. A stage that can compute the whole batch at once
declares `batched = True`: it then receives arrays, and returns one output per point.
Stages that do not are simply called once per point.

To try this, we write a batched version of the foreground, and compare the results of
the two versions on 500 points drawn from the prior:

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
