---
file_format: mystnb
kernelspec:
  name: python3
---

# Parameters and priors

Every model in `yabf` declares the parameters it needs in order to compute. In these
tutorials, for example, the foreground of the sky spectrum needs an amplitude and a
spectral index, and the 21-cm absorption trough needs a depth, a centre and a width.
We call these the *model parameters*.

When you set up an inference, you decide where each model parameter gets its value.
Most are explored by a sampler, each within a prior. But the parameters the sampler
explores need not be exactly the parameters the model asks for: you can provide any
set of parameters, as long as they are ultimately turned into the ones the model
requests.

So user-defined parameters need not always be sampled from a prior directly. They can
be sampled in one space and transformed to another (for example, sampled uniformly in
log-space and transformed to linear space). They can be fixed to a specific value
rather than sampled, or they can be computed deterministically from other parameters.

In `yabf`, you define each input parameter as a {class}`~yabf.params.Param`. The
collection of all parameters is defined as a {class}`~yabf.params.ParamSpace`, which
supplies the extra methods that tie everything together. This tutorial walks through
each kind of parameter in turn, and then shows how the parameter space combines them.

```{code-cell} python
import getdist
import matplotlib.pyplot as plt
import numpy as np
from getdist import MCSamples, plots

from yabf import priors
from yabf.params import (
    DerivedParam,
    Param,
    ParamSpace,
    param_vector,
    register_relation,
)

rng = np.random.default_rng(42)
getdist.chains.print_load_details = False
```

## Sampled parameters and their priors

The most common kind of parameter is one that the sampler explores directly, within a
prior. To define one, give it a name, which must match the name of the model
parameter it provides, and a prior. The simplest prior is uniform between two values,
which you can give as a `(low, high)` tuple. Here is the depth of the absorption
trough, between 0 and 2 K:

```{code-cell} python
depth = Param("depth", prior=(0.0, 2.0), latex="A")
depth.fiducial
```

The *fiducial* value defaults to the median of the prior, and influences where a
sampler initializes its samples. It can also be used as a simple default parameter
set when exploring the posterior. The `latex` label is used when plotting.

Uniform priors are not the only option. A prior is anything with vectorised `logpdf`
and `ppf` (the inverse of the cumulative distribution) methods, so every frozen
{mod}`scipy.stats` distribution works automatically. The {mod}`yabf.priors` module has
short names for the common ones. Here they are, side by side:

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

## Sampling in a different space: derived parameters

The foreground of the sky spectrum is a power law whose amplitude we only know to an
order of magnitude. Instead of sampling the amplitude directly, it is more natural to
sample its logarithm. To do this in `yabf`, define two parameters. The first is the
logarithm, a `Param` with a prior, which the sampler explores. The second is the
amplitude itself, a {class}`~yabf.params.DerivedParam`, which `yabf` computes from the
first. It takes the name of the derived parameter, which here matches the model
parameter `fg_amp`, the function computing it, and the names of its inputs:

```{code-cell} python
log_amp = Param("log10_fg_amp", prior=(2.0, 4.0), latex=r"\log_{10} A_{\rm fg}")
amp = DerivedParam("fg_amp", "pow10", "log10_fg_amp", latex=r"A_{\rm fg}")
amp({"log10_fg_amp": 3.0})
```

The sampler explores `log10_fg_amp`, uniformly between 2 and 4, and the model receives
`fg_amp`: a sampled value of 3 becomes an amplitude of 1000.

The function can be any vectorised Python function, or the name of a registered one.
The functions `identity`, `pow10`, `log10`, `exp`, `log` and `stack` are registered
already. Referring to a function by its name means it can be used in a static
configuration file. You can register your own with the
{func}`~yabf.params.register_relation` decorator, as the next sections show.

## Fixed parameters

Not every model parameter needs to be explored. Sometimes you want to hold one at a
known value: the spectral index of the foreground, say, if it is already well
measured. To fix a parameter to a specific value, instead of sampling it, set the
`fixed=` option:

```{code-cell} python
index = Param("fg_index", fixed=-2.5)
```

The sampler never sees a fixed parameter, but the model still receives its value.

## Parameters derived from several others

Some parameters are not free at all, but follow from the values of others. 21cmFAST,
for instance, describes star formation in minihaloes with parameters that, by default,
follow those of atomic-cooling haloes. To express this, derive them from the
parameters they follow. A derived parameter can have several inputs: write a function
that takes them, in order, and returns the derived value:

```{code-cell} python
f_star10 = Param("F_STAR10", prior=(-3.0, -0.5), latex=r"\log_{10} f_{*,10}")
alpha_star = Param("ALPHA_STAR", prior=priors.normal(0.5, 0.1), latex=r"\alpha_*")


@register_relation
def f_star7_mini(f_star10, alpha_star):
    """Return log10 of the minihalo star formation normalisation."""
    return f_star10 - 3 * alpha_star


f_star7 = DerivedParam("F_STAR7_MINI", f_star7_mini, ["F_STAR10", "ALPHA_STAR"])
alpha_mini = DerivedParam("ALPHA_STAR_MINI", "identity", "ALPHA_STAR")
```

Here `F_STAR7_MINI = F_STAR10 - 3 ALPHA_STAR`, and `ALPHA_STAR_MINI = ALPHA_STAR`.
Registering `f_star7_mini` is optional: it lets a configuration file refer to the
function by its name. When the arguments of the function have the same names as its
inputs, you can leave the inputs out, and `yabf` takes them from the function. A
derived parameter can also be derived from other derived parameters.

## Vector parameters

Some models take a whole vector of parameters, such as the coefficients of a
polynomial. Rather than writing a `Param` for each element, use
{func}`~yabf.params.param_vector`. It makes one parameter per element, which the
sampler sees as separate scalars, and one derived parameter for the vector, which
stacks them into the array the model receives. As for single parameters, elements can
be fixed. Here is a vector of three coefficients, with the last one held at zero:

```{code-cell} python
beta = param_vector(
    "beta", 3, prior=priors.normal(0, 1), fixed={2: 0.0}, latex=r"\beta"
)
[p.name for p in beta]
```

## Joint priors

Sometimes parameters are not independent before we see the data. An earlier
measurement may have constrained the centre and width of the absorption trough
together, with correlated errors. To express this, define their joint prior, here a
correlated Gaussian with {func}`~yabf.priors.multivariate_normal`, and give each
parameter its {class}`~yabf.priors.Marginal`:

```{code-cell} python
measured = priors.multivariate_normal(mean=[78.0, 5.0], cov=[[4.0, 1.5], [1.5, 1.0]])
trough = [
    Param("centre", prior=priors.Marginal(measured, 0), latex=r"\nu_0"),
    Param("width", prior=priors.Marginal(measured, 1), latex="w"),
]
```

The marginal is not what each parameter is drawn from. It describes the parameter on
its own, which is what `yabf` needs for everything that concerns one parameter: its
fiducial value (the median of the marginal), the width of a ball of starting points,
and checking its range against what the model accepts. When it draws samples, or
computes the prior density, the parameter space recognises that the parameters share
a joint prior, and treats them together. It maps the unit hypercube onto the joint
distribution as `mean + L z`, with `L` the lower-triangular Cholesky factor of the
covariance and `z` standard normal: the first parameter from its marginal, the second
from its distribution conditional on the first, and so on. The correlation is kept, as
the plot at the end of this tutorial shows.

A whole vector can take a joint prior too, with `param_vector(..., prior=measured)`.

## The parameter space

Once each parameter is defined, collect them all in a
{class}`~yabf.params.ParamSpace`. This is the object that the rest of `yabf`, and the
samplers, work with. When you build it, it checks that the parameters are consistent:
that derived parameters are computed from parameters it contains, without cycles. It
also puts the sampled parameters in a fixed order:

```{code-cell} python
space = ParamSpace(
    [log_amp, amp, index, f_star10, alpha_star, f_star7, alpha_mini, *beta, *trough]
)
space.names
```

Of the thirteen parameters, seven are sampled: the fixed and derived ones are not, and
neither is the fixed element of `beta`.

The parameter space provides what samplers need, for one point or many at once: the
map from the unit hypercube that nested samplers use, the log prior density, draws
from the prior, and a small ball of starting points around the fiducial values for
MCMC walkers. For example, mapping three points of the unit hypercube to the
parameters, and evaluating their prior density:

```{code-cell} python
u = rng.uniform(size=(3, space.ndim))
theta = space.prior_transform(u)
space.log_prior(theta)
```

The last step is to turn the sampled values into the values the model receives, which
is what `to_model` does. It inserts the fixed values and computes the derived
parameters, each after those it depends on. It returns every parameter by name, and
each part of the model takes the ones it needs:

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

A derived parameter has no prior of its own: its prior follows from the priors of the
parameters it is derived from. This is worth checking before you sample. The amplitude
is uniform in its logarithm, so its density is proportional to $1/A$, and most of the
prior is at small amplitudes:

```{code-cell} python
values = space.to_model(draws)
edges = np.logspace(2, 4, 41)
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.hist(values["fg_amp"], bins=edges, density=True, alpha=0.5, label="prior draws")
ax.plot(edges, 1 / (edges * np.log(100)), label=r"$1 / (A \ln 100)$")
ax.set_xscale("log")
ax.set_xlabel(r"$A_{\rm fg}$")
ax.set_ylabel("prior density")
ax.legend();
```

And the minihalo parameter follows its relation at every draw:

```{code-cell} python
fig, ax = plt.subplots(figsize=(4.5, 3.5))
ax.scatter(
    values["F_STAR10"][:2000] - 3 * values["ALPHA_STAR"][:2000],
    values["F_STAR7_MINI"][:2000],
    s=2,
)
ax.set_xlabel(r"$\log_{10} f_{*,10} - 3\alpha_*$")
ax.set_ylabel(r"$\log_{10} f_{*,7}^{\rm mini}$");
```

The next tutorial builds the model that these parameters feed.
