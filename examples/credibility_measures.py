"""
Off-the-model credibility stock measures.

Computes reference credibility measures from observed data (inflation,
expectations surveys, target).  These are standalone statistics to be
compared later against the model-consistent credibility path obtained
from Bayesian estimation.

Measures:
  1. Bomfim-Rudebusch Kalman filter (λ_t via NumPyro MCMC)
  2. Gaussian signal regression + AR(1) stock
  3. Cecchetti et al. (2002) index
  4. Expectations gap

Output:
  figures/credibility_measures.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from necredpy.credibility_stock import (
    bomfim_rudebusch,
    gaussian_signal,
    credibility_stock_ar1,
    fit_signal_regression,
    cecchetti_index,
    expectations_gap,
)

# ---- Data ----
# Replace with actual data (expectations survey, inflation, target)
# Here we use synthetic data for demonstration
np.random.seed(42)
T = 80
q = 4

pi_target = np.full(T, 3.0)

# Inflation with a surge episode
pi_full = np.full(T + q, 3.0)
for t in range(1, T + q):
    shock = 0.0
    if 40 + q <= t <= 45 + q:
        shock = [3.0, 4.0, 3.5, 2.5, 1.5, 0.8][t - 40 - q]
    pi_full[t] = 0.7 * pi_full[t - 1] + 0.3 * 3.0 + shock + \
                 np.random.normal(0, 0.3)
pi_obs = pi_full[q:]
pi_tilde = np.array([np.mean(pi_full[t:t + q]) for t in range(T)])

# Synthetic expectations (replace with survey data)
cred_dgp = np.ones(T)
for t in range(1, T):
    s = np.exp(-0.5 * 0.06 * (pi_obs[t - 1] - pi_target[t - 1])**2)
    cred_dgp[t] = 0.85 * cred_dgp[t - 1] + 0.15 * s
cred_dgp = np.clip(cred_dgp, 0.0, 1.0)
pi_e = cred_dgp * pi_target + (1 - cred_dgp) * pi_tilde + \
       np.random.normal(0, 0.2, T)

# ---- Compute measures ----

# 1. Bomfim-Rudebusch
print("--- Bomfim-Rudebusch (NumPyro MCMC) ---")
mcmc = bomfim_rudebusch(pi_e, pi_target, pi_full, q=q,
                        num_warmup=300, num_samples=500, num_chains=2,
                        seed=1)
samples = mcmc.get_samples()
lambda_br = np.array(samples['lambda_t'].mean(axis=0))
print("  psi0=%.3f  psi1=%.3f  λ range=[%.3f, %.3f]" % (
    float(samples['psi0'].mean()), float(samples['psi1'].mean()),
    lambda_br.min(), lambda_br.max()))

# 2. Gaussian signal + AR(1) stock
print("\n--- Gaussian signal + AR(1) stock ---")
result = fit_signal_regression(lambda_br[1:], pi_obs[:-1], pi_target[:-1])
print("  ω1=%.3f  ω2=%.3f  ω3=%.3f" % (
    float(result['omega1']), float(result['omega2']),
    float(result['omega3'])))
psi_est = float(samples['psi1'].mean())
signal_full = gaussian_signal(
    pi_obs, pi_target, result['omega1'], result['omega2'], result['omega3'])
c_stock = credibility_stock_ar1(signal_full, psi_est)

# 3. Cecchetti index
ci = cecchetti_index(pi_e, pi_target, band_width=1.0)

# 4. Expectations gap
eg = expectations_gap(pi_e, pi_target, normalize=True)

# ---- Plot ----
os.makedirs("figures", exist_ok=True)
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

ax = axes[0]
ax.plot(pi_obs, label='Inflation', color='tab:red')
ax.plot(pi_e, label='Expectations', color='tab:blue', alpha=0.7)
ax.axhline(3.0, color='black', ls='--', lw=0.8, label='Target')
ax.fill_between(range(T), 2.0, 4.0, alpha=0.08, color='gray',
                label='Tolerance band')
ax.set_ylabel('Percent')
ax.legend(loc='upper left', fontsize=9)
ax.set_title('Data')

ax = axes[1]
ax.plot(lambda_br, label='Bomfim-Rudebusch (KF)', color='tab:blue')
ax.plot(np.array(c_stock), label='Signal + AR(1) stock', color='tab:orange')
ax.plot(np.array(ci), label='Cecchetti index', color='tab:green', alpha=0.7)
ax.plot(np.array(eg), label='Expectations gap', color='tab:purple', alpha=0.7)
ax.set_ylabel('Credibility [0,1]')
ax.set_xlabel('Quarter')
ax.legend(loc='lower left', fontsize=9)
ax.set_title('Off-the-model credibility measures')
ax.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.savefig("figures/credibility_measures.png", dpi=150)
print("\nFigure saved: figures/credibility_measures.png")
