"""
Bayesian estimation of the 5-sector network model.

Same workflow as simple_estimate_nk.py but applied to the larger
5-sector model with input-output linkages (27 variables, 8 shocks,
8 observables).

The key difference from the 3-equation model is the use of a
partial-observation inversion filter: we only observe 8 of the 27
variables (output gap, nominal rate, aggregate and sectoral inflation).
The filter solves an 8x8 subsystem to recover the 8 structural shocks,
then reconstructs the full 27-dimensional state.

Synthetic data includes two credibility-eroding episodes:
  - Energy cost-push at t=25-27 (large, persistent)
  - Food cost-push at t=60 (moderate)

This model is slower to estimate on CPU due to its size.

Output:
  figures/simple_estimate_5s.png       -- posterior histograms
  figures/simple_estimate_5s_trace.png -- trace plots (mixing diagnostics)
  figures/simple_estimate_5s_cred.png  -- credibility path
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from necredpy import Model

# ---- Step 1: Load model ----
m = Model("dynare/five_sector_est.mod")
print("Model: %d vars, %d shocks" % (m.n_vars, m.n_shocks))

# ---- Step 2: Generate synthetic data ----
T = 120
np.random.seed(42)

shocks = {}
shock_sigmas = {
    'eps_y': 0.5, 'eps_m': 0.25, 'eps_agg': 0.10,
    'eps_pi1': 1.0, 'eps_pi2': 1.0, 'eps_pi3': 0.5,
    'eps_pi4': 0.5, 'eps_pi5': 0.3,
}
for s in m.shock_names:
    sigma = shock_sigmas.get(s, 0.5)
    shocks[s] = np.random.normal(0, sigma, T)

# Credibility-eroding episodes
shocks['eps_pi1'][25] += 3.5   # energy
shocks['eps_pi1'][26] += 2.0
shocks['eps_pi1'][27] += 1.0
shocks['eps_pi2'][60] += 2.5   # food

data = m.simulate(shocks, T=T)

obs_vars = ['y', 'i_nom', 'pi_agg', 'pi1', 'pi2', 'pi3', 'pi4', 'pi5']
shock_eq_vars = ['y', 'i_nom', 'pi_agg', 'cp1', 'cp2', 'cp3', 'cp4', 'cp5']

print("pi_agg range: [%.2f, %.2f]" % (
    data['pi_agg'].min(), data['pi_agg'].max()))

# ---- Step 3: Estimate ----
print("\nRunning estimation (5-sector model, this will take a while)...")
results = m.estimate(
    data=data,
    obs_vars=obs_vars,
    shock_vars=shock_eq_vars,
    num_warmup=300,
    num_samples=300,
    num_chains=2,
    max_tree_depth=7,
)

print("\nPosterior summary:")
print(results.summary)

# ---- Step 4: Plot ----
os.makedirs('figures', exist_ok=True)

# Posterior histograms
n_est = len(results.estimated_names)
n_cols = min(4, n_est)
n_rows = (n_est + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
axes = np.array(axes).ravel()

for i, name in enumerate(results.estimated_names):
    ax = axes[i]
    s = np.array(results.samples[name])
    ax.hist(s, bins=30, density=True, alpha=0.7, color='steelblue',
            edgecolor='white')
    true_val = m.params.get(name)
    if true_val is not None:
        ax.axvline(float(true_val), color='red', lw=2, ls='--',
                   label='True=%.3f' % float(true_val))
    ax.axvline(np.mean(s), color='black', lw=1.5,
               label='Mean=%.3f' % np.mean(s))
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

for i in range(n_est, len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig('figures/simple_estimate_5s.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/simple_estimate_5s.png")

# Trace plots (chain mixing diagnostics)
import arviz as az
idata = az.from_numpyro(results.mcmc)

fig_t, axes_t = plt.subplots(n_est, 2, figsize=(14, 2.2 * n_est))
if n_est == 1:
    axes_t = axes_t.reshape(1, 2)

posterior = idata.posterior
for i, name in enumerate(results.estimated_names):
    n_chains = posterior.sizes.get('chain', 1)

    ax = axes_t[i, 0]
    for chain in range(n_chains):
        vals = np.array(posterior[name].sel(chain=chain))
        ax.hist(vals, bins=25, density=True, alpha=0.5)
    true_val = m.params.get(name)
    if true_val is not None:
        ax.axvline(float(true_val), color='red', lw=1.5, ls='--')
    ax.set_title(name, fontsize=9)

    ax = axes_t[i, 1]
    for chain in range(n_chains):
        vals = np.array(posterior[name].sel(chain=chain))
        ax.plot(vals, alpha=0.5, lw=0.4)
    ax.set_title(name + ' (trace)', fontsize=9)

plt.tight_layout()
plt.savefig('figures/simple_estimate_5s_trace.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/simple_estimate_5s_trace.png")

# Credibility path
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
t_ax = np.arange(T)

ax1.plot(t_ax, data['pi_agg'].values, 'k-', lw=1.5)
ax1.axhline(0, color='gray', lw=0.5, ls=':')
ax1.fill_between(t_ax, data['pi_agg'].values, 0, alpha=0.15, color='steelblue')
ax1.set_title('Aggregate Inflation')
ax1.set_ylabel('pi_agg')
ax1.grid(True, alpha=0.3)

ax2.plot(t_ax, results.credibility, '-', color='#d62728', lw=2.5)
ax2.axhline(0.5, color='gray', lw=0.8, ls='--')
ax2.fill_between(t_ax, results.credibility, 0.5,
                 where=results.credibility < 0.5, alpha=0.2, color='red')
ax2.set_ylim(-0.05, 1.1)
ax2.set_title('Estimated Credibility (Posterior Mean)')
ax2.set_ylabel('credibility')
ax2.set_xlabel('Quarter')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/simple_estimate_5s_cred.png', dpi=150, bbox_inches='tight')
print("Saved: figures/simple_estimate_5s_cred.png")
