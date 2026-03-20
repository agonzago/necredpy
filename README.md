# necredpy

Piecewise-linear DSGE/QPM solver with endogenous credibility and production networks.

**necredpy** solves and estimates models where central bank credibility evolves endogenously: when inflation breaches a tolerance band, credibility depletes and Phillips curves become more backward-looking, amplifying the shock. The library supports multi-sector production networks with input-output linkages, so the sacrifice ratio depends on both the credibility state *and* the shock's origin in the network.

## Install

```bash
pip install -e .
```

**Dependencies:** numpy, pandas, scipy, matplotlib, sympy, jax, numpyro, arviz.

## Quick start

```python
from necredpy import Model

# Load a model from a Dynare .mod file
m = Model("dynare/five_sector_network.mod")

# Impulse response (linear, no credibility switching)
irf = m.irf("eps_pi1", size=3.0, T=60)
irf["pi_agg"].plot()

# Impulse response with endogenous credibility
irf = m.irf("eps_pi1", size=3.0, T=60, credibility=True)
irf[["pi_agg", "credibility"]].plot(subplots=True)

# Simulate with custom shock sequences
data = m.simulate({"eps_pi1": [3.0, 1.0, 0.5], "eps_m": [0, 0, 0.2]}, T=60)

# Bayesian estimation (requires priors in the .mod file)
m = Model("dynare/credibility_nk_regimes.mod")
results = m.estimate(data, obs_vars=["y", "pi", "ii"])
print(results.summary)
```

All results are returned as pandas DataFrames. When `credibility=True`, the DataFrame includes `regime` and `credibility` columns.

## API

### `Model(mod_file, param_overrides=None)`

Load and compile a Dynare `.mod` file. The expensive symbolic parsing happens once at construction.

**Attributes:**
- `m.var_names` -- list of variable names
- `m.shock_names` -- list of shock names
- `m.params` -- current parameter values (defaults + overrides)
- `m.n_vars`, `m.n_shocks` -- dimensions

### `m.irf(shock_name, size=1.0, T=60, credibility=False, cred_init=None)`

Compute impulse response to a single shock at t=0.

| Parameter | Description |
|-----------|-------------|
| `shock_name` | Shock name matching a `varexo` in the `.mod` file (e.g., `"eps_pi1"`) |
| `size` | Shock size at t=0 |
| `T` | Horizon in quarters |
| `credibility` | If `True`, solve with endogenous credibility switching (PL solver) |
| `cred_init` | Initial credibility level (default from `.mod` file, usually 1.0) |

Returns a `pandas.DataFrame` with variable names as columns. When `credibility=True`, includes `regime` (0=M1, 1=M2) and `credibility` columns.

### `m.simulate(shocks, T=None, credibility=False, cred_init=None)`

Simulate with custom shock sequences.

| Parameter | Description |
|-----------|-------------|
| `shocks` | `dict` mapping shock names to arrays, `DataFrame`, or `ndarray (T, n_shocks)` |
| `T` | Horizon (inferred from shocks if not given) |
| `credibility` | Enable credibility switching |
| `cred_init` | Initial credibility level |

### `m.estimate(data, obs_vars, shock_vars=None, **kwargs)`

Bayesian estimation via NumPyro NUTS. Requires a `priors;` block in the `.mod` file.

| Parameter | Description |
|-----------|-------------|
| `data` | `DataFrame` or `ndarray (T, n_obs)` of observed data |
| `obs_vars` | List of observed variable names |
| `shock_vars` | List of equation names where shocks enter (defaults to `obs_vars`) |
| `tau` | Sigmoid smoothing parameter (default 0.2) |
| `num_warmup` | NUTS warmup samples (default 500) |
| `num_samples` | Posterior samples per chain (default 500) |
| `num_chains` | Number of MCMC chains (default 2) |
| `target_accept_prob` | NUTS target acceptance (default 0.8) |
| `max_tree_depth` | NUTS max tree depth (default 10) |

Returns an `EstimationResult` with:
- `.samples` -- dict of posterior samples
- `.summary` -- DataFrame with mean, std, quantiles
- `.mcmc` -- full NumPyro MCMC object
- `.credibility` -- credibility path at posterior mean
- `.log_lik` -- log-likelihood at posterior mean

## How it works

The model has two regimes:
- **M1 (high credibility):** Phillips curves are forward-looking (weight `omega_H`)
- **M2 (low credibility):** Phillips curves are backward-looking (weight `omega_L`)

Credibility is a stock variable that evolves each period:
```
miss_t = 1{|pi_t| > epsilon_bar}
cred_t = cred_{t-1} + delta_up * (1 - cred_{t-1}) * (1 - miss_t)
                     - delta_down * cred_{t-1} * miss_t
```
When `cred_t < cred_threshold`, the economy switches to M2. Credibility depletes fast (`delta_down = 0.70`) and rebuilds slowly (`delta_up = 0.05`).

The **PL solver** iterates:
1. Solve the terminal regime: `CF^2 + BF + A = 0` (structured doubling)
2. Backward recursion: compute time-varying policy functions `F_t, E_t, Q_t`
3. Forward simulation: `u_t = E_t + F_t u_{t-1} + Q_t epsilon_t`
4. Update the regime sequence from the simulated path; repeat until consistent

For **estimation**, a sigmoid approximation (`tau` parameter) smooths the credibility threshold, making the likelihood differentiable for HMC/NUTS.

## Dynare `.mod` files

Models are defined in Dynare-compatible `.mod` files in the `dynare/` directory. The parser reads variable declarations, parameter calibrations, model equations, and (optionally) prior specifications. It supports arbitrary leads and lags directly in equations -- no auxiliary variables needed.

| File | Description |
|------|-------------|
| `credibility_nk.mod` | 3-equation NK model (3 variables, 3 shocks) |
| `credibility_nk_regimes.mod` | Same model with regime specification and priors for estimation |
| `five_sector_network.mod` | 5-sector QPM with I-O linkages (27 variables, 12 shocks) |
| `five_sector_est.mod` | 5-sector model configured for estimation (8 observables, 8 shocks) |
| `two_sector_*.mod` | 2-sector variants |

### Priors in `.mod` files

Priors for Bayesian estimation are specified directly in the `.mod` file:

```
priors;
  kappa,  normal,    0.3,  0.15, 0.01, 1.0;   // name, dist, mean, std, low, high
  rho_i,  beta_dist, 0.7,  0.1;                // mean, std -> shape params
  sigma,  inv_gamma, 0.5,  0.3;                // mean, std -> shape params
end;
```

The parser converts (mean, std) to proper shape parameters for beta/gamma/inv_gamma distributions and builds NumPyro sampling functions automatically.

## Examples

All example scripts are in `examples/`. Figures are saved to `figures/`.

```bash
python examples/exp1_propagation.py       # Network IRFs (linear)
python examples/exp2_credibility_irfs.py  # Linear vs PL by sector
python examples/exp3_summary.py           # Sacrifice ratios + M2 duration
python examples/exp4_fragility.py         # Initial credibility effect
python examples/estimate_nk.py            # Bayesian estimation (3-eq model)
python examples/estimate_five_sector.py   # Bayesian estimation (5-sector)
```

## Tests

```bash
pytest
```
