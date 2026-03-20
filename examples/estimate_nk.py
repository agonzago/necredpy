"""
Bayesian estimation of the 3-equation New Keynesian model.

This example shows the full estimation workflow:
  1. Load a model that has a priors; block in the .mod file.
  2. Generate synthetic observed data by simulating the model with
     known shocks, including a credibility-eroding supply shock episode
     at t=25-27 and a smaller one at t=60.
  3. Call m.estimate() to recover the parameters via NumPyro NUTS (HMC).
     The estimator uses a sigmoid-smoothed inversion filter internally
     to compute the likelihood.
  4. Plot posterior distributions and the estimated credibility path.

The priors are read directly from the .mod file (priors; block).
Parameters not listed in the priors block are held fixed at their
calibrated values.

This is the small model (3 variables, 3 shocks, 3 observables) and
runs in a few minutes on CPU.

Output:
  figures/simple_estimate_nk.png       -- posterior histograms
  figures/simple_estimate_nk_trace.png -- trace plots (mixing diagnostics)
  figures/simple_estimate_nk_cred.png  -- credibility path
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from necredpy import Model

# ---- Step 1: Load model ----
m = Model("dynare/credibility_nk_regimes.mod")
print("Model: %d vars, %d shocks" % (m.n_vars, m.n_shocks))
print("Variables:", m.var_names)

# ---- Step 2: Generate synthetic data ----
# We need observed data as a DataFrame. Here we simulate from the model
# itself using known shocks, then treat the output as "observed."
T = 100
np.random.seed(42)

shocks = {
    'eps_d': np.random.normal(0, 0.15, T),
    'eps_s': np.random.normal(0, 0.15, T),
    'eps_m': np.random.normal(0, 0.05, T),
}
# Add credibility-eroding supply shock episode
shocks['eps_s'][25] += 2.5
shocks['eps_s'][26] += 1.5
shocks['eps_s'][27] += 0.8
shocks['eps_s'][60] += 1.8

data = m.simulate(shocks, T=T)
print("Simulated data: pi range [%.2f, %.2f]" % (
    data['pi'].min(), data['pi'].max()))

# ---- Step 3: Estimate ----
print("\nRunning estimation (this takes a few minutes)...")
results = m.estimate(
    data=data,
    obs_vars=['y', 'pi', 'ii'],
    shock_vars=['y', 'pi', 'ii'],
    num_warmup=500,
    num_samples=500,
    num_chains=2,
)

print("\nPosterior summary:")
print(results.summary)
print("\nLog-likelihood at posterior mean: %.2f" % results.log_lik)

# ---- Step 4: Plot ----
os.makedirs('figures', exist_ok=True)

# Posterior histograms
n_est = len(results.estimated_names)
fig, axes = plt.subplots(1, n_est, figsize=(5 * n_est, 4))
if n_est == 1:
    axes = [axes]

for i, name in enumerate(results.estimated_names):
    ax = axes[i]
    s = np.array(results.samples[name])
    ax.hist(s, bins=40, density=True, alpha=0.7, color='steelblue',
            edgecolor='white')
    true_val = m.params.get(name)
    if true_val is not None:
        ax.axvline(float(true_val), color='red', lw=2, ls='--',
                   label='True = %.3f' % float(true_val))
    ax.axvline(np.mean(s), color='black', lw=1.5,
               label='Mean = %.3f' % np.mean(s))
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/simple_estimate_nk.png', dpi=150, bbox_inches='tight')
print("Saved: figures/simple_estimate_nk.png")

# Trace plots (chain mixing diagnostics)
import arviz as az
idata = az.from_numpyro(results.mcmc)

fig_t, axes_t = plt.subplots(n_est, 2, figsize=(14, 2.5 * n_est))
if n_est == 1:
    axes_t = axes_t.reshape(1, 2)

posterior = idata.posterior
for i, name in enumerate(results.estimated_names):
    n_chains = posterior.sizes.get('chain', 1)

    # Density (left column)
    ax = axes_t[i, 0]
    for chain in range(n_chains):
        vals = np.array(posterior[name].sel(chain=chain))
        ax.hist(vals, bins=30, density=True, alpha=0.5, label='Chain %d' % chain)
    true_val = m.params.get(name)
    if true_val is not None:
        ax.axvline(float(true_val), color='red', lw=1.5, ls='--')
    ax.set_title(name)
    ax.set_ylabel('Density')
    if i == 0:
        ax.legend(fontsize=7)

    # Trace (right column)
    ax = axes_t[i, 1]
    for chain in range(n_chains):
        vals = np.array(posterior[name].sel(chain=chain))
        ax.plot(vals, alpha=0.6, lw=0.5)
    ax.set_title(name + ' (trace)')
    ax.set_ylabel(name)

axes_t[-1, 0].set_xlabel('Value')
axes_t[-1, 1].set_xlabel('Draw')
plt.tight_layout()
plt.savefig('figures/simple_estimate_nk_trace.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/simple_estimate_nk_trace.png")

# Credibility path
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
t_ax = np.arange(T)

ax1.plot(t_ax, data['pi'].values, 'k-', lw=1.5)
ax1.axhline(0, color='gray', lw=0.5, ls=':')
ax1.fill_between(t_ax, data['pi'].values, 0, alpha=0.15, color='steelblue')
ax1.set_title('Observed Inflation')
ax1.set_ylabel('pi')
ax1.grid(True, alpha=0.3)

ax2.plot(t_ax, results.credibility, '-', color='#d62728', lw=2.5)
ax2.axhline(0.5, color='gray', lw=0.8, ls='--')
ax2.fill_between(t_ax, results.credibility, 0.5,
                 where=results.credibility < 0.5, alpha=0.2, color='red')
ax2.set_ylim(-0.05, 1.1)
ax2.set_title('Estimated Credibility Path (Posterior Mean)')
ax2.set_ylabel('credibility')
ax2.set_xlabel('Quarter')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/simple_estimate_nk_cred.png', dpi=150, bbox_inches='tight')
print("Saved: figures/simple_estimate_nk_cred.png")
