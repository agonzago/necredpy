# necredpy

Piecewise-linear DSGE/QPM solver with endogenous credibility and production networks.

## Install

```bash
pip install -e .
```

## What's inside

- **`necredpy.pontus`** -- Piecewise-linear solver (Rendahl 2017): structured doubling, backward recursion, endogenous switching
- **`necredpy.jax_model`** -- JAX-differentiable inversion filter for Bayesian estimation (NumPyro NUTS)
- **`necredpy.models.credibility_nk`** -- 3-equation NK model with credibility regimes
- **`necredpy.utils.dynare_parser`** -- Parses Dynare `.mod` files into A,B,C matrices with JAX support
- **`necredpy.stability`** -- Eigenvalue stability checks

## Quick example

```python
from necredpy import baseline_theta, build_model, solve_terminal

theta = baseline_theta()
M1, M2 = build_model(theta)
F, Q = solve_terminal(*M1[:3])
```

## Estimation

```python
from necredpy.jax_model import compile_jax_model, inversion_filter_partial

model = compile_jax_model(open("model.mod").read())
ll, shocks, cred, state = inversion_filter_partial(model, obs, params,
                                                     obs_indices, shock_indices)
```
