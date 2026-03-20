"""
Bayesian estimation of the NK model with endogenous credibility via NumPyro.

STRATEGY
--------
1. Parse the .mod file to get model structure, calibration, AND priors.
2. Generate synthetic data from the model with known parameters.
3. Estimate parameters listed in the priors; block using NUTS (HMC).
4. Compare posterior to true values.
5. Plot credibility path from posterior mean.

The priors are specified in the .mod file following Dynare-inspired syntax:
    priors;
      kappa, normal, 0.3, 0.15, 0.01, 1.0;
      sigma_d, inv_gamma, 3.0, 1.0;
      ...
    end;

Parameters NOT listed in the priors block are held fixed at their
calibrated values from the .mod file.

Produces: figures/fig_posterior.png, figures/fig_credibility_estimated.png

Usage: .venv/bin/python scripts/estimate_numpyro.py
"""
import os

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpyro
from numpyro.infer import MCMC, NUTS

from necredpy.jax_model import inversion_filter_jax
from necredpy.utils.dynare_parser import extract_declarations, extract_priors, build_numpyro_prior_fn

numpyro.set_host_device_count(2)


# ---------------------------------------------------------------------------
# Load .mod file
# ---------------------------------------------------------------------------

MOD_FILE = os.path.join(os.path.dirname(__file__), '..', 'dynare',
                         'credibility_nk_regimes.mod')


def load_mod():
    """Read .mod file, extract calibration and priors."""
    with open(MOD_FILE) as f:
        mod_string = f.read()

    _, _, _, param_defaults = extract_declarations(mod_string)
    priors = extract_priors(mod_string)

    print("Calibrated parameters from .mod file:")
    for k, v in sorted(param_defaults.items()):
        print("  %-18s = %s" % (k, v))

    print("\nPriors from .mod file:")
    for p in priors:
        print("  %-10s ~ %-12s mean=%-6s std=%-6s shape=%s" % (
            p['name'], p['dist'], p['mean'], p['std'],
            tuple(round(x, 3) for x in p['shape'])))

    return param_defaults, priors


# ---------------------------------------------------------------------------
# True parameters (for synthetic data -- uses .mod calibration + shock vols)
# ---------------------------------------------------------------------------

def get_true_params(param_defaults):
    """Build full parameter dict for the JAX inversion filter.

    The inversion filter needs: beta, sigma, kappa, rho_i, phi_pi, phi_y,
    omega_H, omega_L, delta_up, delta_down, epsilon_bar, tau,
    sigma_d, sigma_s, sigma_m.

    tau and sigma_* are not in the .mod file, so we add them here.
    """
    p = dict(param_defaults)
    # Sigmoid smoothing parameter (not in .mod -- estimation choice)
    p['tau'] = 0.2
    # Shock standard deviations (match Dynare shocks block)
    p.setdefault('sigma_d', 0.5)
    p.setdefault('sigma_s', 0.5)
    p.setdefault('sigma_m', 0.25)
    return p


# ---------------------------------------------------------------------------
# Generate synthetic data
# ---------------------------------------------------------------------------

def generate_data(rng_key, true_params, T=100):
    """Generate synthetic observed data from the true model."""
    from necredpy.models.credibility_nk import build_matrices
    from necredpy.pontus import solve_terminal, backward_recursion, simulate_forward

    n = 4
    tau = true_params['tau']
    omega_H = true_params['omega_H']
    omega_L = true_params['omega_L']

    key1, key2, key3 = random.split(rng_key, 3)
    eps_d = float(true_params['sigma_d']) * np.array(random.normal(key1, (T,)))
    eps_s = float(true_params['sigma_s']) * np.array(random.normal(key2, (T,)))
    eps_m = float(true_params['sigma_m']) * np.array(random.normal(key3, (T,)))

    # Credibility-eroding episodes
    eps_s[25] += 2.5
    eps_s[26] += 1.5
    eps_s[27] += 0.8
    eps_s[60] += 1.8

    epsilon = np.zeros((T, n))
    epsilon[:, 0] = eps_d
    epsilon[:, 1] = eps_s
    epsilon[:, 2] = eps_m

    # Iterative smooth solve
    theta = {k: float(v) for k, v in true_params.items()}
    A1, B1, C1, D1 = build_matrices(theta, omega_H)
    F_terminal, _ = solve_terminal(A1, B1, C1)

    regime_lin = np.zeros(T, dtype=int)
    F_p, E_p, Q_p = backward_recursion(
        regime_lin, F_terminal, (A1, B1, C1, D1), (A1, B1, C1, D1))
    u_path = simulate_forward(F_p, E_p, Q_p, epsilon)

    from numpy.linalg import solve as np_solve, inv as np_inv

    for outer in range(100):
        pi_path = u_path[:, 1]
        cred = 1.0
        omega_arr = np.zeros(T)
        for t in range(T):
            miss_t = 1.0 / (1.0 + np.exp(
                -(np.abs(pi_path[t]) - theta['epsilon_bar']) / tau))
            cred = cred + theta['delta_up'] * (1.0 - cred) * (1.0 - miss_t) \
                        - theta['delta_down'] * cred * miss_t
            cred = max(0.0, min(1.0, cred))
            omega_arr[t] = omega_L + (omega_H - omega_L) * cred

        F_path = np.zeros((T, n, n))
        E_path = np.zeros((T, n))
        Q_path = np.zeros((T, n, n))
        F_next = F_terminal.copy()
        E_next = np.zeros(n)

        for t in range(T - 1, -1, -1):
            A_t, B_t, C_t, D_t = build_matrices(theta, omega_arr[t])
            M_t = B_t + C_t @ F_next
            F_path[t] = -np_solve(M_t, A_t)
            E_path[t] = -np_solve(M_t, C_t @ E_next + D_t)
            Q_path[t] = -np_inv(M_t)
            F_next = F_path[t]
            E_next = E_path[t]

        u_new = simulate_forward(F_path, E_path, Q_path, epsilon)
        if np.max(np.abs(u_new - u_path)) < 1e-10:
            u_path = u_new
            break
        u_path = u_new

    return (jnp.array(u_path[:, 0]),
            jnp.array(u_path[:, 1]),
            jnp.array(u_path[:, 2]),
            epsilon)


# ---------------------------------------------------------------------------
# NumPyro model builder (reads priors from .mod file)
# ---------------------------------------------------------------------------

def build_nk_model(priors, true_params):
    """Build a NumPyro model function from parsed priors.

    Parameters listed in priors are sampled; all others are fixed at
    true_params values.

    Parameters
    ----------
    priors : list of dict
        From extract_priors().
    true_params : dict
        Full parameter dictionary (fixed values for non-estimated params).

    Returns
    -------
    nk_model : callable(obs_y, obs_pi, obs_i)
    estimated_names : list of str
    """
    sample_priors_fn, estimated_names = build_numpyro_prior_fn(priors)

    def nk_model(obs_y, obs_pi, obs_i):
        # Sample estimated parameters from priors
        sampled = sample_priors_fn()

        # Build full parameter dict: start from fixed, override with sampled
        params = {k: v for k, v in true_params.items()}
        params.update(sampled)

        # Log-likelihood via inversion filter
        log_lik, _, _ = inversion_filter_jax(obs_y, obs_pi, obs_i, params)
        numpyro.factor('log_lik', log_lik)

    return nk_model, estimated_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("NumPyro Estimation: NK Model with Credibility")
    print("  Priors read from: %s" % os.path.basename(MOD_FILE))
    print("=" * 60)

    # Step 1: Load .mod file
    param_defaults, priors = load_mod()
    true_params = get_true_params(param_defaults)

    # Step 2: Generate data
    print("\n1. Generating synthetic data (T=100)...")
    rng_key = random.PRNGKey(42)
    T = 100
    obs_y, obs_pi, obs_i, eps_true = generate_data(rng_key, true_params, T)
    print("   Data generated. pi range: [%.2f, %.2f]" % (
        float(jnp.min(obs_pi)), float(jnp.max(obs_pi))))

    # Step 3: Verify inversion at true params
    print("\n2. Verifying inversion filter at true parameters...")
    ll_true, eps_rec, cred_rec = inversion_filter_jax(
        obs_y, obs_pi, obs_i, true_params)
    err = float(jnp.max(jnp.abs(eps_rec - jnp.array(eps_true[:, :3]))))
    print("   Log-lik at true params: %.2f" % float(ll_true))
    print("   Max shock recovery error: %.2e" % err)

    # Step 4: Build model from parsed priors and run NUTS
    nk_model, estimated_names = build_nk_model(priors, true_params)
    print("\n3. Estimated parameters: %s" % estimated_names)
    print("   Running NUTS (2 chains, 500 warmup, 500 samples)...")

    kernel = NUTS(nk_model, target_accept_prob=0.8, max_tree_depth=8)
    mcmc = MCMC(kernel, num_warmup=500, num_samples=500, num_chains=2,
                progress_bar=True)

    rng_key = random.PRNGKey(123)
    mcmc.run(rng_key, obs_y=obs_y, obs_pi=obs_pi, obs_i=obs_i)

    # Step 5: Summary
    print("\n4. Posterior summary:")
    mcmc.print_summary()
    samples = mcmc.get_samples()

    # Step 5b: ArviZ trace plots
    print("\n5. Generating ArviZ trace plot...")
    import arviz as az
    idata = az.from_numpyro(mcmc)

    # ArviZ v1.0 changed the API; use plot_trace with backend kwarg
    try:
        fig_trace = az.plot_trace(idata, backend="matplotlib")
    except (TypeError, ValueError):
        # Fallback: manual trace plot if ArviZ API is incompatible
        fig_trace = None

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)

    if fig_trace is not None:
        trace_out = os.path.join(out_dir, 'fig_trace.png')
        plt.savefig(trace_out, dpi=150, bbox_inches='tight')
        plt.close()
        print("   Saved: %s" % trace_out)
    else:
        # Manual trace plot as fallback
        n_est = len(estimated_names)
        fig_t, axes_t = plt.subplots(n_est, 2, figsize=(14, 2.5 * n_est))
        if n_est == 1:
            axes_t = axes_t.reshape(1, 2)

        posterior = idata.posterior
        for i, name in enumerate(estimated_names):
            # Density (left)
            ax = axes_t[i, 0]
            for chain in range(posterior.sizes.get('chain', posterior[name].sizes.get('chain', 1))):
                vals = np.array(posterior[name].sel(chain=chain))
                ax.hist(vals, bins=30, density=True, alpha=0.5,
                        label='Chain %d' % chain)
            ax.set_title(name)
            ax.set_ylabel('Density')
            if i == 0:
                ax.legend(fontsize=7)

            # Trace (right)
            ax = axes_t[i, 1]
            for chain in range(posterior.sizes.get('chain', posterior[name].sizes.get('chain', 1))):
                vals = np.array(posterior[name].sel(chain=chain))
                ax.plot(vals, alpha=0.6, lw=0.5)
            ax.set_title(name + ' (trace)')
            ax.set_ylabel(name)

        axes_t[-1, 0].set_xlabel('Value')
        axes_t[-1, 1].set_xlabel('Draw')
        plt.tight_layout()
        trace_out = os.path.join(out_dir, 'fig_trace.png')
        plt.savefig(trace_out, dpi=150, bbox_inches='tight')
        plt.close()
        print("   Saved: %s (manual fallback)" % trace_out)

    # Step 6: Plot posteriors
    print("\n6. Generating posterior plot...")
    n_est = len(estimated_names)
    n_cols = min(3, n_est)
    n_rows = (n_est + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_est == 1:
        axes = [axes]
    else:
        axes = np.array(axes).ravel()

    for i, name in enumerate(estimated_names):
        ax = axes[i]
        s = np.array(samples[name])
        ax.hist(s, bins=40, density=True, alpha=0.7, color='steelblue',
                edgecolor='white')
        if name in true_params:
            true_val = float(true_params[name])
            ax.axvline(true_val, color='red', lw=2, ls='--',
                       label='True = %.3f' % true_val)
        ax.axvline(np.mean(s), color='black', lw=1.5,
                   label='Mean = %.3f' % np.mean(s))
        ax.set_title(name, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_est, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Posterior Distributions (T=%d, tau=%.2f)' % (
        T, true_params['tau']), fontsize=14, y=1.01)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'fig_posterior.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print("   Saved: %s" % out)

    # Step 7: Credibility plot from posterior mean
    print("\n6. Computing credibility path at posterior mean...")
    post_mean_params = dict(true_params)
    for name in estimated_names:
        if name in samples:
            post_mean_params[name] = float(jnp.mean(samples[name]))

    _, _, cred_post = inversion_filter_jax(obs_y, obs_pi, obs_i, post_mean_params)

    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    t_ax = np.arange(T)

    ax1.plot(t_ax, np.array(obs_pi), 'k-', lw=1.5)
    ax1.axhline(true_params['epsilon_bar'], color='red', lw=0.8, ls='--', alpha=0.5)
    ax1.axhline(-true_params['epsilon_bar'], color='red', lw=0.8, ls='--', alpha=0.5)
    ax1.axhline(0, color='gray', lw=0.5, ls=':')
    ax1.fill_between(t_ax, np.array(obs_pi), 0, alpha=0.15, color='steelblue')
    ax1.set_title('Observed Inflation')
    ax1.set_ylabel('pi_t')
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_ax, np.array(cred_rec), '-', color='gray', lw=1.5,
             label='True params')
    ax2.plot(t_ax, np.array(cred_post), '-', color='#d62728', lw=2.5,
             label='Posterior mean')
    ax2.axhline(0.5, color='gray', lw=0.8, ls='--')
    ax2.fill_between(t_ax, np.array(cred_post), 0.5,
                     where=np.array(cred_post) < 0.5,
                     alpha=0.2, color='red')
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_title('Credibility Capital Over Time')
    ax2.set_ylabel('cred_t')
    ax2.set_xlabel('Quarter')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig2.suptitle('Estimated Credibility Path (Posterior Mean)', fontsize=13, y=1.01)
    plt.tight_layout()
    out2 = os.path.join(out_dir, 'fig_credibility_estimated.png')
    plt.savefig(out2, dpi=150, bbox_inches='tight')
    print("   Saved: %s" % out2)

    print("\nDone.")


if __name__ == '__main__':
    main()
