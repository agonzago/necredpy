"""
Bayesian estimation of the 5-sector network model with credibility.

STRATEGY
--------
1. Parse five_sector_est.mod (27 vars, 8 shocks, 8 observables).
2. Generate synthetic data with known parameters + credibility episode.
3. Estimate parameters listed in the priors; block using NUTS.
4. Plot posteriors, traces, credibility path, and recovered shocks.

Observables (8): y, i_nom, pi_agg, pi1, pi2, pi3, pi4, pi5
Shocks (8): eps_y, eps_m, eps_agg, eps_pi1..eps_pi5

The partial-observation inversion filter solves the 8x8 subsystem
from observed variables, then reconstructs the full 27-dim state.

Produces:
  figures/fig_5s_posterior.png
  figures/fig_5s_trace.png
  figures/fig_5s_credibility.png

Usage: .venv/bin/python scripts/estimate_five_sector.py
"""
import os
import time

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpyro
from numpyro.infer import MCMC, NUTS

from necredpy.jax_model import (compile_jax_model, inversion_filter_partial,
                                 solve_terminal_jax)
from necredpy.utils.dynare_parser import extract_priors, build_numpyro_prior_fn

numpyro.set_host_device_count(2)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MOD_FILE = os.path.join(os.path.dirname(__file__), '..',
                         'dynare', 'five_sector_est.mod')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')

# Observed variable names and their indices in the state vector
OBS_VAR_NAMES = ['y', 'i_nom', 'pi_agg', 'pi1', 'pi2', 'pi3', 'pi4', 'pi5']

# Equations where shocks enter (maps to shock ordering in varexo)
SHOCK_EQ_NAMES = ['y', 'i_nom', 'pi_agg', 'cp1', 'cp2', 'cp3', 'cp4', 'cp5']

# Simulation length
T_SIM = 120


# ---------------------------------------------------------------------------
# Load and compile model
# ---------------------------------------------------------------------------

def load_model():
    with open(MOD_FILE) as f:
        mod_string = f.read()

    print("Compiling 5-sector model from .mod file...")
    t0 = time.time()
    model = compile_jax_model(mod_string)
    print("  Compiled in %.1fs (%d vars, %d shocks)" % (
        time.time() - t0, model['n_vars'], model['n_shocks']))

    var_names = model['var_names']
    obs_indices = [var_names.index(v) for v in OBS_VAR_NAMES]
    shock_indices = [var_names.index(v) for v in SHOCK_EQ_NAMES]

    priors = model['priors']
    print("  Priors: %d parameters to estimate" % len(priors))
    for p in priors:
        print("    %-14s ~ %-12s mean=%-6s std=%-6s" % (
            p['name'], p['dist'], p['mean'], p['std']))

    return model, priors, obs_indices, shock_indices


# ---------------------------------------------------------------------------
# True parameters
# ---------------------------------------------------------------------------

def get_true_params(model):
    """Build true parameter dict from .mod defaults + shock std devs."""
    p = dict(model['param_defaults'])
    p['tau'] = 0.2
    # Shock std devs (must match sigma_<shock_name> convention)
    shock_sigmas = {
        'eps_y': 0.5, 'eps_m': 0.25, 'eps_agg': 0.10,
        'eps_pi1': 1.0, 'eps_pi2': 1.0, 'eps_pi3': 0.5,
        'eps_pi4': 0.5, 'eps_pi5': 0.3,
    }
    for s in model['shock_names']:
        p['sigma_' + s] = shock_sigmas.get(s, 0.5)
    return p


# ---------------------------------------------------------------------------
# Generate synthetic data (iterative smooth solve)
# ---------------------------------------------------------------------------

def generate_data(model, true_params, obs_indices, shock_indices, T):
    """Generate synthetic data with a credibility episode.

    Uses iterative solve: guess path -> compute omega -> backward recursion
    -> forward simulate -> repeat.
    """
    n = model['n_vars']
    n_shocks = model['n_shocks']
    var_names = model['var_names']
    build_ABC = model['build_ABC']

    omega_H = true_params['omega_H']
    omega_L = true_params['omega_L']
    tau = true_params['tau']
    epsilon_bar = true_params['epsilon_bar']
    delta_up = true_params['delta_up']
    delta_down = true_params['delta_down']

    # Determine omega param and monitor index
    omega_param = 'omega'
    regime_spec = model['regime_spec']
    if regime_spec:
        m1_params = regime_spec.get('M1_params', {})
        if m1_params:
            omega_param = list(m1_params.keys())[0]
    monitor_var = regime_spec['monitor'] if regime_spec else 'pi_agg'
    monitor_idx = var_names.index(monitor_var)

    # Generate shocks
    np.random.seed(42)
    eps_active = np.zeros((T, n_shocks))
    shock_names = model['shock_names']
    for i, s in enumerate(shock_names):
        sigma = true_params['sigma_' + s]
        eps_active[:, i] = np.random.normal(0, sigma, T)

    # Credibility-eroding episodes
    # Big energy cost-push at t=25-27
    eps_active[25, 3] += 3.5   # eps_pi1 (energy)
    eps_active[26, 3] += 2.0
    eps_active[27, 3] += 1.0
    # Food shock at t=60
    eps_active[60, 4] += 2.5   # eps_pi2 (food)

    # Build full shock vector
    shk_idx = np.array(shock_indices)
    eps_full_np = np.zeros((T, n))
    for i in range(n_shocks):
        eps_full_np[:, shk_idx[i]] = eps_active[:, i]

    # Terminal solution at M1
    params_M1 = dict(true_params)
    params_M1[omega_param] = omega_H
    A_M1, B_M1, C_M1 = build_ABC(params_M1)
    F_terminal = solve_terminal_jax(A_M1, B_M1, C_M1)
    Q_M1 = -jnp.linalg.inv(B_M1 + C_M1 @ F_terminal)

    # Initial forward sim at constant M1
    u_path = np.zeros((T, n))
    u_prev = np.zeros(n)
    for t in range(T):
        u_path[t] = np.array(F_terminal @ u_prev + Q_M1 @ eps_full_np[t])
        u_prev = u_path[t]

    # Iterate with sigmoid credibility
    from numpy.linalg import solve as np_solve, inv as np_inv

    for outer in range(80):
        # Compute cred/omega path
        pi_monitor = u_path[:, monitor_idx]
        cred = 1.0
        omega_arr = np.zeros(T)
        for t in range(T):
            miss = 1.0 / (1.0 + np.exp(
                -(np.abs(pi_monitor[t]) - epsilon_bar) / tau))
            cred = cred + delta_up * (1.0 - cred) * (1.0 - miss) \
                        - delta_down * cred * miss
            cred = max(0.0, min(1.0, cred))
            omega_arr[t] = omega_L + (omega_H - omega_L) * cred

        # Per-period matrices, backward recursion, forward sim
        F_path = np.zeros((T, n, n))
        E_path = np.zeros((T, n))
        Q_path = np.zeros((T, n, n))
        F_next = np.array(F_terminal)
        E_next = np.zeros(n)

        for t in range(T - 1, -1, -1):
            p_t = dict(true_params)
            p_t[omega_param] = omega_arr[t]
            A_t, B_t, C_t = build_ABC(p_t)
            A_t, B_t, C_t = np.array(A_t), np.array(B_t), np.array(C_t)
            M_t = B_t + C_t @ F_next
            F_path[t] = -np_solve(M_t, A_t)
            E_path[t] = -np_solve(M_t, C_t @ E_next)
            Q_path[t] = -np_inv(M_t)
            F_next = F_path[t]
            E_next = E_path[t]

        u_new = np.zeros((T, n))
        u_prev = np.zeros(n)
        for t in range(T):
            u_new[t] = E_path[t] + F_path[t] @ u_prev + Q_path[t] @ eps_full_np[t]
            u_prev = u_new[t]

        diff = np.max(np.abs(u_new - u_path))
        u_path = u_new
        if diff < 1e-10:
            print("  Data generation converged in %d iterations" % (outer + 1))
            break

    obs_partial = jnp.array(u_path[:, obs_indices])
    return obs_partial, eps_active, u_path


# ---------------------------------------------------------------------------
# NumPyro model builder
# ---------------------------------------------------------------------------

def build_nk_model(model, priors, true_params, obs_indices, shock_indices):
    """Build NumPyro model: sample priors, run inversion filter."""
    sample_priors_fn, estimated_names = build_numpyro_prior_fn(priors)

    def nk_model(obs_partial):
        sampled = sample_priors_fn()
        params = {k: v for k, v in true_params.items()}
        params.update(sampled)

        ll, _, _, _ = inversion_filter_partial(
            model, obs_partial, params,
            obs_indices=obs_indices,
            shock_indices=shock_indices)

        numpyro.factor('log_lik', ll)

    return nk_model, estimated_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("5-Sector Network Model: Bayesian Estimation")
    print("=" * 65)

    # Step 1: Load model
    model, priors, obs_indices, shock_indices = load_model()
    true_params = get_true_params(model)

    # Step 2: Generate data
    print("\n1. Generating synthetic data (T=%d)..." % T_SIM)
    obs_partial, eps_true, u_true = generate_data(
        model, true_params, obs_indices, shock_indices, T_SIM)
    print("   pi_agg range: [%.2f, %.2f]" % (
        float(jnp.min(obs_partial[:, 2])), float(jnp.max(obs_partial[:, 2]))))

    # Step 3: Verify inversion at true params
    print("\n2. Verifying inversion filter at true parameters...")
    ll_true, eps_rec, cred_true, u_rec = inversion_filter_partial(
        model, obs_partial, true_params,
        obs_indices=obs_indices, shock_indices=shock_indices)
    err = float(jnp.max(jnp.abs(eps_rec - jnp.array(eps_true))))
    print("   Log-lik at true params: %.2f" % float(ll_true))
    print("   Max shock recovery error: %.2e" % err)
    print("   Min credibility: %.3f" % float(jnp.min(cred_true)))

    # Step 4: Build NumPyro model and run NUTS
    nk_model, estimated_names = build_nk_model(
        model, priors, true_params, obs_indices, shock_indices)

    n_warmup = 300
    n_samples = 300
    n_chains = 2
    print("\n3. Running NUTS (%d chains, %d warmup, %d samples)..." % (
        n_chains, n_warmup, n_samples))
    print("   Estimated: %s" % estimated_names)
    print("   This will take a while on CPU (27-var model)...")

    kernel = NUTS(nk_model, target_accept_prob=0.8, max_tree_depth=7)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=n_chains, progress_bar=True)

    rng_key = random.PRNGKey(123)
    t0 = time.time()
    mcmc.run(rng_key, obs_partial=obs_partial)
    elapsed = time.time() - t0
    print("   MCMC completed in %.0fs (%.1f min)" % (elapsed, elapsed / 60))

    # Step 5: Summary
    print("\n4. Posterior summary:")
    mcmc.print_summary()
    samples = mcmc.get_samples()

    # Step 6: Plots
    os.makedirs(FIG_DIR, exist_ok=True)

    # --- Trace plot ---
    print("\n5. Generating plots...")
    import arviz as az
    idata = az.from_numpyro(mcmc)

    n_est = len(estimated_names)
    fig_t, axes_t = plt.subplots(n_est, 2, figsize=(14, 2.2 * n_est))
    if n_est == 1:
        axes_t = axes_t.reshape(1, 2)

    posterior = idata.posterior
    for i, name in enumerate(estimated_names):
        ax = axes_t[i, 0]
        for chain in range(posterior.sizes.get('chain',
                           posterior[name].sizes.get('chain', 1))):
            vals = np.array(posterior[name].sel(chain=chain))
            ax.hist(vals, bins=25, density=True, alpha=0.5)
        true_val = true_params.get(name)
        if true_val is not None:
            ax.axvline(float(true_val), color='red', lw=1.5, ls='--')
        ax.set_title(name, fontsize=9)

        ax = axes_t[i, 1]
        for chain in range(posterior.sizes.get('chain',
                           posterior[name].sizes.get('chain', 1))):
            vals = np.array(posterior[name].sel(chain=chain))
            ax.plot(vals, alpha=0.5, lw=0.4)
        ax.set_title(name + ' (trace)', fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_5s_trace.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: %s" % out)

    # --- Posterior histograms ---
    n_cols = min(4, n_est)
    n_rows = (n_est + n_cols - 1) // n_cols
    fig_p, axes_p = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes_p = np.array(axes_p).ravel()

    for i, name in enumerate(estimated_names):
        ax = axes_p[i]
        s = np.array(samples[name])
        ax.hist(s, bins=30, density=True, alpha=0.7, color='steelblue',
                edgecolor='white')
        true_val = true_params.get(name)
        if true_val is not None:
            ax.axvline(float(true_val), color='red', lw=2, ls='--',
                       label='True=%.3f' % float(true_val))
        ax.axvline(np.mean(s), color='black', lw=1.5,
                   label='Mean=%.3f' % np.mean(s))
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for i in range(n_est, len(axes_p)):
        axes_p[i].set_visible(False)

    fig_p.suptitle('5-Sector Model: Posterior Distributions (T=%d)' % T_SIM,
                   fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_5s_posterior.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: %s" % out)

    # --- Credibility path at posterior mean ---
    post_mean_params = dict(true_params)
    for name in estimated_names:
        if name in samples:
            post_mean_params[name] = float(jnp.mean(samples[name]))

    _, _, cred_post, _ = inversion_filter_partial(
        model, obs_partial, post_mean_params,
        obs_indices=obs_indices, shock_indices=shock_indices)

    t_ax = np.arange(T_SIM)
    sector_names = ['Energy', 'Food', 'Transport', 'Goods', 'Services']
    sector_colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

    fig_c, axes_c = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 0: Sectoral inflations
    ax = axes_c[0]
    for j in range(5):
        ax.plot(t_ax, np.array(obs_partial[:, 3 + j]),
                color=sector_colors[j], lw=1.2, label=sector_names[j])
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_title('Sectoral Inflation')
    ax.set_ylabel('pi_j')
    ax.legend(fontsize=8, ncol=5, loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 1: Aggregate inflation
    ax = axes_c[1]
    ax.plot(t_ax, np.array(obs_partial[:, 2]), 'k-', lw=1.5)
    ax.axhline(true_params['epsilon_bar'], color='red', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(-true_params['epsilon_bar'], color='red', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.fill_between(t_ax, np.array(obs_partial[:, 2]), 0,
                    alpha=0.15, color='steelblue')
    ax.set_title('Aggregate Inflation')
    ax.set_ylabel('pi_agg')
    ax.grid(True, alpha=0.3)

    # Panel 2: Credibility
    ax = axes_c[2]
    ax.plot(t_ax, np.array(cred_true), '-', color='gray', lw=1.5,
             label='True params')
    ax.plot(t_ax, np.array(cred_post), '-', color='#d62728', lw=2.5,
             label='Posterior mean')
    ax.axhline(0.5, color='gray', lw=0.8, ls='--')
    ax.fill_between(t_ax, np.array(cred_post), 0.5,
                    where=np.array(cred_post) < 0.5,
                    alpha=0.2, color='red', label='Low cred zone')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Credibility Capital')
    ax.set_ylabel('cred_t')
    ax.set_xlabel('Quarter')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig_c.suptitle('5-Sector Network + Credibility (T=%d)' % T_SIM,
                   fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig_5s_credibility.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print("   Saved: %s" % out)

    print("\nDone.")


if __name__ == '__main__':
    main()
