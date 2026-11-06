---
file_format: mystnb
kernelspec:
  name: python3
---

# Parameters and priors

An inference has two kinds of parameters. The *sampled* parameters are what a sampler
explores, each with a prior. The *model* parameters are what the model computes with.
Often they are the same, but not always: it is easier to sample an amplitude
spanning decades in its logarithm, some model parameters are held fixed, and some are
computed from others.

yabf describes each sampled, fixed or computed parameter with a {class}`~yabf.params.Param`,
and collects them in a {class}`~yabf.params.ParamSpace`, which gives a sampler
everything it needs.

```{code-cell} python
import getdist
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots

from yabf import priors
from yabf.params import Param, ParamSpace, param_vector

rng = np.random.default_rng(42)
getdist.chains.print_load_details = False
```

## Sampled parameters and their priors

A sampled parameter needs a name and a prior. A `(low, high)` tuple is a uniform
prior:

```{code-cell} python
depth = Param("depth", prior=(0.0, 2.0), latex="A")
depth.fiducial
```

The *fiducial* value, where a sampler can start, defaults to the median of the prior.

A prior is anything with vectorised `logpdf` and `ppf` (the inverse of the cumulative
distribution) methods, so every frozen {mod}`scipy.stats` distribution works. The
{mod}`yabf.priors` module has short names for the common ones:

```{code-cell} python
examples = {
    "uniform(1, 15)": priors.uniform(1, 15),
    "log_uniform(1, 15)": priors.log_uniform(1, 15),
    "normal(8, 3)": priors.normal(8, 3),
    "truncated_normal(8, 3, low=5)": priors.truncated_normal(8, 3, low=5),
}

x = np.linspace(0, 16, 400)
fig, ax = plt.subplots(figsize=(6, 3.5))
for label, prior in examples.items():
    ax.plot(x, np.exp(prior.logpdf(x)), label=label)
ax.set_xlabel("x")
ax.set_ylabel("prior density")
ax.legend();
```

## Sampling in a different space

The foreground of the sky spectrum is a power law whose amplitude we only know to an
order of magnitude. Sampling its logarithm uniformly is the natural choice, but the
model needs the amplitude itself. A parameter can *feed* a model parameter of
another name, through a named transform:

```{code-cell} python
log_amp = Param(
    "log10_fg_amp",
    prior=(2.0, 4.0),
    feeds="fg_amp",
    transform="pow10",
    latex=r"\log_{10} A_{\rm fg}",
)
log_amp.to_model(3.0)
```

The built-in transforms are `identity`, `pow10`, `log10`, `exp` and `log`;
{func}`~yabf.params.register_transform` adds more. Transforms are referred to by name
so that a parameter space can be written to a configuration file and read back.

## Fixed and tied parameters

A fixed parameter always takes the same value; the sampler never sees it:

```{code-cell} python
index = Param("fg_index", fixed=-2.5)
```

A *tied* parameter is computed from other model parameters. 21cmFAST, for instance,
can describe star formation in minihaloes with parameters that default to those of
atomic-cooling haloes:

```{code-cell} python
f_star10 = Param("F_STAR10", prior=(-3.0, -0.5), latex=r"\log_{10} f_{*,10}")
alpha_star = Param("ALPHA_STAR", prior=priors.normal(0.5, 0.1), latex=r"\alpha_*")
f_star7 = Param("F_STAR7_MINI", tied={"F_STAR10": 1, "ALPHA_STAR": -3})
alpha_mini = Param("ALPHA_STAR_MINI", tied="ALPHA_STAR")
```

A dictionary ties a parameter to a linear combination of others, here
`F_STAR7_MINI = F_STAR10 - 3 ALPHA_STAR`, and a name ties it to that parameter's value.
Other relations can be added with {func}`~yabf.params.register_relation`.

## Vector parameters

Some models take a vector, such as the coefficients of a polynomial.
{func}`~yabf.params.param_vector` makes one parameter per element, which the sampler
sees as scalars, and the model receives as one array. Elements can be fixed:

```{code-cell} python
beta = param_vector(
    "beta", 3, prior=priors.normal(0, 1), fixed={2: 0.0}, latex=r"\beta"
)
[p.name for p in beta]
```

## Joint priors

Parameters can also share a prior. A correlated Gaussian prior on the centre and width
of the absorption trough, as might come from an earlier measurement, is a
{func}`~yabf.priors.multivariate_normal`, and each parameter takes its
{class}`~yabf.priors.Marginal`:

```{code-cell} python
measured = priors.multivariate_normal(mean=[78.0, 5.0], cov=[[4.0, 1.5], [1.5, 1.0]])
trough = [
    Param("centre", prior=priors.Marginal(measured, 0), latex=r"\nu_0"),
    Param("width", prior=priors.Marginal(measured, 1), latex="w"),
]
```

The parameter space draws them together, from the joint prior. A whole vector can
take a joint prior too, with `param_vector(..., prior=measured)`.

## The parameter space

A {class}`~yabf.params.ParamSpace` collects the parameters. It checks that they are
consistent (that ties refer to parameters it contains, without cycles, and that each
vector is complete) and orders the sampled ones:

```{code-cell} python
space = ParamSpace(
    [log_amp, index, f_star10, alpha_star, f_star7, alpha_mini, *beta, *trough]
)
space.names
```

It provides what samplers need, for one point or many at once: the map from the
unit hypercube that nested samplers use, the log prior density, draws from the prior
and a small ball of starting points around the fiducial values for MCMC walkers.

```{code-cell} python
u = rng.uniform(size=(3, space.ndim))
theta = space.prior_transform(u)
space.log_prior(theta)
```

`to_model` turns sampled values into the model parameters: it applies transforms,
inserts fixed values, assembles vectors and computes tied parameters.

```{code-cell} python
model = space.to_model(theta[0])
model
```

Draws from the prior show all of this at once. The amplitude is uniform in its
logarithm, the elements of `beta` are independent, and the centre and width of the
trough are correlated as their joint prior says:

```{code-cell} python
draws = space.sample_prior(20_000, rng)
samples = MCSamples(
    samples=draws,
    names=space.names,
    labels=[p.latex for p in space.sampled],
    label="prior",
)
g = plots.get_subplot_plotter()
g.triangle_plot(samples, ["log10_fg_amp", "beta_0", "centre", "width"], filled=True)
```

And the tied parameter follows its relation at every draw:

```{code-cell} python
values = space.to_model(draws[:2000])
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.scatter(values["F_STAR10"] - 3 * values["ALPHA_STAR"], values["F_STAR7_MINI"], s=2)
ax.set_xlabel(r"$\log_{10} f_{*,10} - 3\alpha_*$")
ax.set_ylabel(r"$\log_{10} f_{*,7}^{\rm mini}$");
```
