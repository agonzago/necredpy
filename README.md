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

# BanRep Gaussian credibility (user-defined signal + accumulation)
m_banrep = Model("dynare/five_sector_banrep_cred.mod")
irf_banrep = m_banrep.irf("eps_pi1", size=5.0, T=60, credibility=True)
irf_banrep[["pi_agg", "credibility"]].plot(subplots=True)

# Compare Isard vs BanRep credibility dynamics
irf_isard = m.irf("eps_pi1", size=5.0, T=60, credibility=True)
print("Isard cred min:", irf_isard["credibility"].min())
print("BanRep cred min:", irf_banrep["credibility"].min())

# Bayesian estimation (requires priors in the .mod file)
m_est = Model("dynare/credibility_nk_regimes.mod")
results = m_est.estimate(data, obs_vars=["y", "pi", "ii"])
print(results.summary)

# Estimation with BanRep credibility
m_banrep_est = Model("dynare/credibility_nk_banrep.mod")
results = m_banrep_est.estimate(data, obs_vars=["y", "pi", "ii"])
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

### Credibility dynamics

Credibility is a stock variable that evolves each period. The library supports two ways to specify how it evolves:

**1. Built-in Isard specification** (via `regimes;` block):
```
miss_t = 1{|pi_t| > epsilon_bar}
cred_t = cred_{t-1} + delta_up * (1 - cred_{t-1}) * (1 - miss_t)
                     - delta_down * cred_{t-1} * miss_t
```

**2. User-defined specification** (via `credibility;` block):
```
credibility;
  monitor:   pi_agg;
  threshold: 0.5;
  cred_init: 1.0;
  signal = exp(-omega_1 - omega_2*(pi(-1) - pi_star)^2) - omega_3;
  accumulation = psi*cred + (1-psi)*s;
end;
```

The `credibility;` block lets you write arbitrary signal and accumulation functions as mathematical expressions. They are parsed by sympy and compiled to both numpy (for the PL solver) and JAX (for Bayesian estimation). Any parameter declared in the `parameters` block can appear in these expressions and can be estimated via the `priors;` block.

Available symbols in `signal`: `pi` (current monitored variable), `pi(-1)` (lagged), `pi_star` (target), and any declared parameter. Available symbols in `accumulation`: `cred` (previous stock), `s` (current signal), `miss` (band indicator), and any declared parameter. Standard math functions are supported: `exp`, `log`, `sqrt`, `abs`.

When both blocks are present, `credibility;` takes precedence for switching dynamics while `regimes;` still provides the M1/M2 parameter overrides.

When `cred_t < threshold`, the economy switches to M2.

### Solver

The **PL solver** iterates:
1. Solve the terminal regime: `CF^2 + BF + A = 0` (structured doubling)
2. Backward recursion: compute time-varying policy functions `F_t, E_t, Q_t`
3. Forward simulation: `u_t = E_t + F_t u_{t-1} + Q_t epsilon_t`
4. Update the regime sequence from the simulated path; repeat until consistent

For **estimation**, a sigmoid approximation (`tau` parameter) smooths the credibility threshold, making the likelihood differentiable for HMC/NUTS. When using a `credibility;` block, the signal and accumulation functions are compiled to JAX-traced callables, so they are fully differentiable without additional smoothing.

## Dynare `.mod` files

Models are defined in Dynare-compatible `.mod` files in the `dynare/` directory. The parser reads variable declarations, parameter calibrations, model equations, and (optionally) prior specifications. It supports arbitrary leads and lags directly in equations -- no auxiliary variables needed.

| File | Description |
|------|-------------|
| `credibility_nk.mod` | 3-equation NK model (3 variables, 3 shocks) |
| `credibility_nk_regimes.mod` | Same model with Isard credibility and priors for estimation |
| `credibility_nk_banrep.mod` | Same model with BanRep Gaussian credibility (`credibility;` block) |
| `five_sector_network.mod` | 5-sector QPM with I-O linkages (27 variables, 12 shocks) |
| `five_sector_banrep_cred.mod` | 5-sector model with BanRep Gaussian credibility |
| `five_sector_est.mod` | 5-sector model configured for estimation (8 observables, 8 shocks) |
| `two_sector_*.mod` | 2-sector variants |

### Credibility specification in `.mod` files

Credibility dynamics can be specified in two ways. The `regimes;` block uses the built-in Isard indicator-function specification:

```
regimes;
  M1: omega = omega_H;
  M2: omega = omega_L;
  switch: shadow_cred;
  monitor: pi_agg;
  band: epsilon_bar;
  delta_up: delta_up;
  delta_down: delta_down;
  cred_init: 1.0;
end;
```

The `credibility;` block allows arbitrary user-defined signal and accumulation functions:

```
credibility;
  monitor:   pi_agg;
  threshold: 0.5;
  cred_init: 1.0;
  signal = exp(-omega_sig_ub - omega_sig1*(pi(-1) - pi_star)^2) - omega_sig3;
  accumulation = psi_cred*cred + (1-psi_cred)*s;
end;
```

Both blocks can coexist in the same file. The `credibility;` block takes precedence for switching dynamics, while `regimes;` provides the M1/M2 parameter overrides.

### Priors in `.mod` files

Priors for Bayesian estimation are specified directly in the `.mod` file:

```
priors;
  kappa,  normal,    0.3,  0.15, 0.01, 1.0;   // name, dist, mean, std, low, high
  rho_i,  beta_dist, 0.7,  0.1;                // mean, std -> shape params
  sigma,  inv_gamma, 0.5,  0.3;                // mean, std -> shape params
end;
```

The parser converts (mean, std) to proper shape parameters for beta/gamma/inv_gamma distributions and builds NumPyro sampling functions automatically. Credibility parameters used in the `credibility;` block (e.g., `psi_cred`, `omega_sig1`) can be included in the priors for joint estimation.

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
