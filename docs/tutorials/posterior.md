---
file_format: mystnb
kernelspec:
  name: python3
---

# Posteriors, likelihoods and mock data

In this tutorial we put the pieces of the previous two together into an inference.
We draw a mock measurement of the global 21-cm signal, recover the trough from it with
an MCMC sampler, and check that the posterior covers the true values. The object that
ties everything together is the {class}`~yabf.posterior.Posterior`: it joins a
parameter space to a graph of stages, and scores the model's outputs against data.

```{code-cell} python
from collections.abc import Mapping
from typing import ClassVar

import attrs
import emcee
import getdist
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots

from yabf.likelihoods import Gaussian
from yabf.params import Param
from yabf.posterior import LikelihoodTerm, Posterior
from yabf.stages import REQUIRED, All, One, ParamSpec

rng = np.random.default_rng(2024)
getdist.chains.print_load_details = False
```

We reuse the stages of the previous tutorial: a power-law foreground and a Gaussian
absorption trough, summed into the sky spectrum, and an instrument gain. Their code is
the same as there; click below to show it.

```{code-cell} python
:tags: [hide-input]

NU = np.linspace(50.0, 100.0, 101)  # frequencies, in MHz


@attrs.frozen
class Spectrum:
    """A brightness temperature in K, at the frequencies NU."""

    values: np.ndarray


@attrs.frozen
class Foreground:
    """A power-law foreground."""

    name: str = "foreground"
    parameters: ClassVar = {
        "fg_amp": ParamSpec(1000.0, domain=(0, np.inf)),
        "fg_index": ParamSpec(-2.5),
    }
    requires: ClassVar = {}
    provides: ClassVar = {"component": Spectrum}

    def compute(self, params: Mapping[str, float]) -> dict[str, Spectrum]:
        """The foreground spectrum."""
        return {
            "component": Spectrum(params["fg_amp"] * (NU / 75) ** params["fg_index"])
        }


@attrs.frozen
class Trough:
    """A Gaussian absorption trough, counting its runs."""

    name: str = "signal"
    runs: list = attrs.field(factory=list, eq=False)
    parameters: ClassVar = {
        "depth": ParamSpec(0.5, domain=(0, np.inf)),
        "centre": ParamSpec(78.0),
        "width": ParamSpec(5.0, domain=(0, np.inf)),
    }
    requires: ClassVar = {}
    provides: ClassVar = {"component": Spectrum}

    def compute(self, params: Mapping[str, float]) -> dict[str, Spectrum]:
        """The trough, recording the run."""
        self.runs.append(1)
        x = (NU - params["centre"]) / params["width"]
        return {"component": Spectrum(-params["depth"] * np.exp(-0.5 * x**2))}


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

## Likelihood terms

To compare the model with data, we need a likelihood: a model of the noise in the
measurement. In `yabf`, a likelihood provides two things, both describing one
distribution of the data given the model: `loglike(model, data)`, the log probability
density of the data, and `simulate(model, rng)`, a draw of data from that same
distribution. Because both come from one place, mock data and inference cannot
disagree about the noise.

{mod}`yabf.likelihoods` has generic likelihoods for the common cases: Gaussian,
Student's t, Poisson, upper limits, asymmetric errors and tabulated distributions. Our
radiometer has 50 mK of independent Gaussian noise in each channel, so we use a
Gaussian with that standard deviation:

```{code-cell} python
SIGMA = 0.05  # K
radiometer = Gaussian.from_sigma(np.full(NU.size, SIGMA))
```

Next, we need to say which model output the likelihood should score, and against
which data. A {class}`~yabf.posterior.LikelihoodTerm` does this: it is a stage that
takes one model output, here the `Spectrum` named `"sky"`, and scores the data against
it with the likelihood. Together with the parameters and the stages, the terms make a
posterior. Since we will build several posteriors below, we write a small function
that makes one from a measured spectrum:

```{code-cell} python
params = [
    Param("fg_amp", prior=(500.0, 1500.0), latex=r"A_{\rm fg}"),
    Param("fg_index", prior=(-3.0, -2.0), latex=r"\beta"),
    Param("depth", prior=(0.0, 2.0), latex="A"),
    Param("centre", prior=(60.0, 90.0), latex=r"\nu_0"),
    Param("width", prior=(1.0, 15.0), latex="w"),
]
stages = [Foreground(), Trough(), Sum()]
SKY = One(Spectrum, name="sky")


def posterior(data: np.ndarray) -> Posterior:
    """The posterior of our parameters, given a measured sky spectrum."""
    term = LikelihoodTerm("radiometer", radiometer, data, SKY)
    return Posterior(params, stages, likelihoods=[term])
```

## Mock data

We do not have a real measurement, so we make a mock one, at known true values of the
parameters. `posterior.simulate` does this: at a given point, it draws data from each
term's likelihood, given the model that term would score. It does not look at the
data, so we can build a posterior on a placeholder to make the mock, and then build
the posterior we will sample on the mock itself:

```{code-cell} python
truth = np.array([1000.0, -2.5, 0.5, 78.0, 5.0])
mock = posterior(np.zeros(NU.size)).simulate(truth, rng)["radiometer"]
post = posterior(mock)
post.names
```

Let us look at the mock. The trough is invisible under the foreground, but stands out
once the true foreground is subtracted:

```{code-cell} python
out = post.forward(truth)
foreground = out["foreground", "component"].values

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, figsize=(6, 5))
top.plot(NU, mock, ".", ms=3)
top.set_ylabel("T [K]")
bottom.errorbar(NU, mock - foreground, SIGMA, fmt=".", ms=3, label="mock - foreground")
bottom.plot(NU, out["signal", "component"].values, label="true signal")
bottom.set_xlabel(r"$\nu$ [MHz]")
bottom.set_ylabel("T [K]")
bottom.legend();
```

## The log-likelihood

With data and a posterior, we can score any point of the parameter space.
`log_likelihood` returns the log-likelihood, along with any outputs listed as
`derived` when the posterior was made (here none), as blobs. At the true values:

```{code-cell} python
lnl, blobs = post.log_likelihood(truth)
lnl
```

```{code-cell} python
draws = post.params.sample_prior(4, rng)
post.log_likelihood(draws)[0]
```

It also takes a batch of points, one per row. Points drawn from the prior fit the
data very badly, which is why we need a sampler to find the region of good fits.

## Sampling the posterior

We now explore the posterior with a sampler. A posterior gives a sampler what it
needs: `ndim`, `names`, `log_prior`, `prior_transform` and `log_likelihood`. Here we
use [emcee](https://emcee.readthedocs.io), an MCMC sampler. It needs one function,
the log posterior, which we write from the log prior and the log-likelihood. emcee can
call it on all its walkers at once, which `yabf` evaluates as a batch:

```{code-cell} python
def log_posterior(theta: np.ndarray) -> np.ndarray:
    """The log posterior of a batch of points; -inf outside the prior."""
    out = post.log_prior(theta)
    inside = np.isfinite(out)
    if inside.any():
        out[inside] += post.log_likelihood(theta[inside])[0]
    return out


nwalkers = 32
start = post.params.gaussian_ball(nwalkers, rng, center=truth)
sampler = emcee.EnsembleSampler(nwalkers, post.ndim, log_posterior, vectorize=True)
sampler.run_mcmc(start, 4000, progress=False)
print("autocorrelation times:", sampler.get_autocorr_time(quiet=True).round())
chain = sampler.get_chain(discard=1000, thin=25, flat=True)
```

Starting the walkers near the truth makes them converge quickly. With real data the
truth is unknown: start them across the prior, for example with
`post.params.sample_prior`, and run until the chains have converged.

To see the result, we plot the samples as a corner plot. The posterior covers the true
values, marked by the dotted lines:

```{code-cell} python
samples = MCSamples(
    samples=chain,
    names=post.names,
    labels=[p.latex for p in post.params.sampled],
    label="posterior",
)
g = plots.get_subplot_plotter()
g.triangle_plot(samples, filled=True, markers=dict(zip(post.names, truth)))
```

```{code-cell} python
for name, mean, std, true in zip(post.names, chain.mean(0), chain.std(0), truth):
    print(f"{name:>8}: {mean:.6g} ± {std:.2g}   (truth {true:g})")
```

The mean of each parameter is within about one standard deviation of the truth. The
foreground, being so bright, is measured far more precisely than the trough.

Finally, to see what the posterior means for the signal itself, we run the model at
100 points drawn from it, and plot the trough at each. Together they show the range
of signals the data allow:

```{code-cell} python
fig, ax = plt.subplots(figsize=(6, 3.5))
for theta in chain[rng.choice(len(chain), 100)]:
    ax.plot(NU, post.forward(theta)["signal", "component"].values, color="C0", alpha=0.1)
ax.plot(NU, out["signal", "component"].values, "k--", label="truth")
ax.set_xlabel(r"$\nu$ [MHz]")
ax.set_ylabel("T [K]")
ax.legend();
```

## Several data sets

Real analyses often combine several measurements of the same thing, each made with its
own instrument. Here we add a second radiometer, twice as noisy, whose gain is
uncertain. Both measure the same sky, so the sky should be computed once, and only the
instrument effect should differ between them.

A likelihood term can apply instrument effects of its own before scoring the model, in
its `forward` stages. So the two measurements are two terms scoring one shared sky
spectrum, the second through its own gain stage. Each term provides the model it
scored as its output `"scored"`:

```{code-cell} python
second = Gaussian.from_sigma(np.full(NU.size, 2 * SIGMA))
signal = Trough()


def two_instruments(data: np.ndarray, second_data: np.ndarray) -> Posterior:
    """The posterior given the spectra measured by both instruments."""
    return Posterior(
        [*params, Param("gain", prior=(0.98, 1.02))],
        [Foreground(), signal, Sum()],
        likelihoods=[
            LikelihoodTerm("radiometer", radiometer, data, SKY),
            LikelihoodTerm("second", second, second_data, SKY, forward=[Gain()]),
        ],
    )


theta = np.append(truth, 1.01)
placeholder = np.zeros(NU.size)
mocks = two_instruments(placeholder, placeholder).simulate(theta, rng)
two = two_instruments(mocks["radiometer"], mocks["second"])

signal.runs.clear()
out = two.forward(theta)
print(len(signal.runs), "run of the signal")
out["second", "scored"].values[:3] / out["radiometer", "scored"].values[:3]
```

The log-likelihood is the sum of the terms:

```{code-cell} python
two.loglike_outputs, two.log_likelihood(theta)[0]
```

## Likelihoods with parameters of their own

The noise level of a measurement is not always known, and then it has to be fitted
along with the model. To allow this, a likelihood can have parameters of its own. It
declares them in `parameters`, like a stage, and its `with_params` method returns the
likelihood at given values of them. Here we write a Gaussian likelihood whose noise
level, the same in every channel, is an unknown parameter `noise`, in K:

```{code-cell} python
@attrs.frozen
class UnknownNoise:
    """Independent Gaussian noise of unknown amplitude."""

    size: int
    parameters: ClassVar = {"noise": ParamSpec(REQUIRED, domain=(0, np.inf))}

    def with_params(self, params: Mapping[str, float]) -> Gaussian:
        """The likelihood at the noise level ``params["noise"]``."""
        return Gaussian.from_sigma(np.full(self.size, params["noise"]))

    def loglike(self, model: Spectrum, data: np.ndarray) -> float:
        """Refuse: the likelihood needs a noise level."""
        raise TypeError("Give the noise level with `with_params`.")

    def simulate(self, model: Spectrum, rng: np.random.Generator) -> np.ndarray:
        """Refuse: the likelihood needs a noise level."""
        raise TypeError("Give the noise level with `with_params`.")


unknown = Posterior(
    [*params, Param("noise", prior=(0.01, 0.2), latex=r"\sigma")],
    stages,
    likelihoods=[LikelihoodTerm("radiometer", UnknownNoise(NU.size), mock, SKY)],
)
unknown.names
```

The likelihood term passes the noise level to the likelihood, and the posterior
checks its prior against the domain like any stage parameter. At the true signal and
foreground, the likelihood of the noise level peaks at the root-mean-square of the
mock's residuals, as it should for Gaussian noise. From mock to mock, that scatters
about the 50 mK we simulated with a standard deviation of σ/√(2n) = 3.5 mK, for
n = 101 channels:

```{code-cell} python
grid = np.linspace(0.03, 0.08, 101)
points = np.column_stack([np.tile(truth, (grid.size, 1)), grid])
lnl, _ = unknown.log_likelihood(points)

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(grid * 1e3, np.exp(lnl - lnl.max()))
ax.axvline(SIGMA * 1e3, color="k", ls="--", label="simulated")
ax.legend()
ax.set_xlabel("noise [mK]")
ax.set_ylabel("likelihood (relative)")

residual = mock - post.forward(truth)["sum", "sky"].values
print(f"peak: {grid[np.argmax(lnl)] * 1e3:.1f} mK")
print(f"RMS of the residuals: {np.sqrt(np.mean(residual**2)) * 1e3:.1f} mK")
```

## Random inputs

Finally, some models are random themselves. A simulator that draws random numbers,
such as initial conditions, makes its output depend on a seed. How the seed is set
changes what the posterior means, so `yabf` asks you to be explicit about it. A stage
marks such an input with `ParamSpec(..., random=True)`, and the posterior reports how
each one is set. Here a foreground with a random ripple stands in for such a
simulator:

```{code-cell} python
@attrs.frozen
class NoisyForeground(Foreground):
    """A foreground with a random ripple, as a stand-in for a random simulation."""

    parameters: ClassVar = {**Foreground.parameters, "seed": ParamSpec(0, random=True)}

    def compute(self, params: Mapping[str, float]) -> dict[str, Spectrum]:
        """The foreground, with a ripple drawn from the seed."""
        ripple = np.random.default_rng(int(params["seed"])).normal(0, 0.01, NU.size)
        return {
            "component": Spectrum(super().compute(params)["component"].values + ripple)
        }


noisy = Posterior(
    params,
    [NoisyForeground(), Trough(), Sum()],
    likelihoods=[LikelihoodTerm("radiometer", radiometer, mock, SKY)],
)
noisy.random_inputs
```

The seed is left at its `"default"`, so this posterior is conditioned on one
realisation of the ripple. Fixing the seed (`"fixed"`) does the same with a chosen
realisation, and sampling it (`"sampled"`) treats the ripple as noise to marginalise
over. Mock data never redraw it: the seed is a parameter like any other, and
randomness that should be part of the noise belongs in the likelihood.
