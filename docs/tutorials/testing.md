---
file_format: mystnb
kernelspec:
  name: python3
---

# Testing samplers and stages

Before trusting a sampler, or a stage you have written, it is worth checking it on a
problem whose answer you already know. {mod}`yabf.testing` provides such problems:
posteriors that are known exactly, and a check that a stage keeps the stage protocol.
This tutorial shows how to use them.

```{code-cell} python
from collections.abc import Mapping
from typing import ClassVar

import attrs
import emcee
import getdist
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots

from yabf import testing
from yabf.stages import ParamSpec

rng = np.random.default_rng(1)
getdist.chains.print_load_details = False
```

## Posteriors with a known answer

{func}`~yabf.testing.correlated_gaussian` builds a posterior, with parameters, a prior
and a likelihood like any other, whose posterior distribution is a correlated
multivariate normal. It returns an {class}`~yabf.testing.AnalyticPosterior`: the
`posterior` itself, its exact solution `truth`, and its `log_evidence`. The truth is a
{class}`~yabf.testing.GaussianMixture`, which knows its mean, covariance, density and
marginal distributions, and can draw exact samples:

```{code-cell} python
gaussian = testing.correlated_gaussian(ndim=4, rho=0.8)
post = gaussian.posterior
post.names, gaussian.truth.mean
```

To check a sampler, we run it on the posterior and compare its samples with the truth.
Here we use emcee, which samples `post.log_posterior`:

```{code-cell} python
nwalkers = 16
start = post.params.gaussian_ball(nwalkers, rng, center=gaussian.truth.mean)
sampler = emcee.EnsembleSampler(
    nwalkers, post.ndim, lambda theta: post.log_posterior(theta)[0], vectorize=True
)
sampler.run_mcmc(start, 6000, progress=False)
chain = sampler.get_chain(discard=1000, thin=10, flat=True)
```

The samples should agree with exact draws from the truth, up to Monte Carlo error. A
corner plot shows the agreement at a glance:

```{code-cell} python
exact = gaussian.truth.sample(len(chain), rng)
g = plots.get_subplot_plotter()
g.triangle_plot(
    [
        MCSamples(samples=chain, names=post.names, label="emcee"),
        MCSamples(samples=exact, names=post.names, label="exact"),
    ],
    filled=True,
)
```

A test needs a number rather than a plot. The mean of a chain of `n_eff` effective
samples has a standard error of `σ / √n_eff`, where `σ` is the posterior standard
deviation, and emcee estimates `n_eff` from the autocorrelation time. So the error of
each mean, in units of its standard error, should be of order one:

```{code-cell} python
steps = sampler.get_chain(discard=1000, flat=True)
n_eff = len(steps) / sampler.get_autocorr_time(discard=1000).max()
sd = np.sqrt(np.diag(gaussian.truth.covariance))
(steps.mean(axis=0) - gaussian.truth.mean) / (sd / np.sqrt(n_eff))
```

{func}`~yabf.testing.bimodal` is a harder case: two well-separated modes, with weights
0.3 and 0.7. A sampler that claims to handle several modes should find both, with the
right weights. A nested sampler can also be checked against the evidence, which is
known exactly:

```{code-cell} python
two_modes = testing.bimodal()
x = np.linspace(-10, 10, 400)
density = np.gradient(two_modes.truth.marginal_cdf(x, 0), x)

fig, ax = plt.subplots(figsize=(5, 3))
ax.hist(two_modes.truth.sample(20_000, rng)[:, 0], bins=80, density=True, alpha=0.5)
ax.plot(x, density)
ax.set_xlabel("$x_0$")
ax.set_ylabel("posterior density")
print(f"log evidence: {two_modes.log_evidence:.4f}")
```

## A toy simulator

Real models run a slow simulator, which makes a large, often random, product, and then
summarise it. {func}`~yabf.testing.toy_simulation` builds a posterior with the same
shape:
- {class}`~yabf.testing.ToySimulator` makes a seeded "lightcone", a smooth signal at
  each frequency plus noise across pixels;
- {class}`~yabf.testing.GlobalSignal` averages the lightcone over pixels;
- a likelihood term scores that average against a mock measurement.

The noise is made to average to zero across the pixels, so the average is linear in
the parameters and does not depend on the seed. The posterior is therefore known
exactly, while the pipeline does everything a real one does:

```{code-cell} python
toy = testing.toy_simulation(n_theta=3, n_pix=500)
simulator, average = toy.posterior.stages
theta = toy.truth.mean

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, seed in zip(axes[:2], [1, 2], strict=True):
    lightcone = simulator.compute({"theta": theta, "seed": seed})["lightcone"]
    ax.imshow(lightcone, aspect="auto", origin="lower")
    ax.set_title(f"seed {seed}")
    ax.set_xlabel("pixel")
    axes[2].plot(average.compute({}, lightcone)["global_signal"], label=f"seed {seed}")
axes[0].set_ylabel("frequency channel")
axes[2].set_title("average over pixels")
axes[2].legend();
```

The two lightcones differ, and their averages are the same. A batched version of the
simulator, `toy_simulation(batched=True)`, computes many points in one call, as an
emulator would. It gives the same posterior, which is how a batched sampler adapter is
checked against an unbatched one.

## Checking a stage

{func}`~yabf.testing.assert_stage_contract` checks that a stage keeps the stage
protocol at given values. It checks that the stage:
- returns exactly the outputs it provides, of their types;
- gives the same outputs when called again;
- does not change its parameters or inputs;
- if batched, gives at each point of a batch what that point alone gives.

Here is a stage that subtracts a constant from a spectrum, written with a subtle bug:
it subtracts in place, changing its input. Other stages that take the same spectrum
would then see the changed values:

```{code-cell} python
@attrs.frozen
class Offset:
    """Subtract a constant offset from a spectrum."""

    name: str = "offset"
    parameters: ClassVar = {"offset": ParamSpec(0.0)}
    requires: ClassVar = {"spectrum": np.ndarray}
    provides: ClassVar = {"corrected": np.ndarray}

    def compute(
        self, params: Mapping[str, float], spectrum: np.ndarray
    ) -> dict[str, np.ndarray]:
        """The spectrum minus the offset."""
        spectrum -= params["offset"]
        return {"corrected": spectrum}


try:
    testing.assert_stage_contract(
        Offset(), {"offset": 1.0}, {"spectrum": np.linspace(0, 1, 5)}
    )
except AssertionError as e:
    print(e)
```

Writing `spectrum - params["offset"]` instead returns a new array, and the check
passes. Running the check on each stage of a package, at a few values, catches such
bugs before they reach a posterior.
