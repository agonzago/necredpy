"""
Inversion filter demo: simulate data, recover shocks, plot credibility.

PURPOSE
-------
1. Simulate the model with known shocks (including a credibility-eroding
   episode) to generate synthetic "observed" data.
2. Apply the inversion filter to recover the structural shocks from the
   observed data, given the model parameters.
3. Plot the recovered credibility path over time.

The inversion filter works period-by-period:
  Given observables (y_t, pi_t, i_t) and the state u_{t-1}:
    1. Determine cred_t from the credibility law of motion
       (using smooth sigmoid or sharp threshold on pi_t)
    2. Determine omega_t = omega_L + (omega_H - omega_L)*cred_t
    3. Build matrices A_t, B_t, C_t for this period
    4. Knowing u_t (from observables) and u_{t-1}, invert to get epsilon_t:
         epsilon_t = Q_t^{-1} (u_t - E_t - F_t u_{t-1})

For the inversion to work we need:
  - As many shocks as observables (3 shocks, 3 observables)
  - The policy functions F_t, E_t at each t, which depend on future
    expectations and thus on the regime/omega path going forward.

APPROACH: We solve the full model first (given observed data, iterate to
find the consistent omega path and policy functions), then back out shocks.

Produces: figures/fig_inversion_filter.png

Usage: .venv/bin/python scripts/inversion_filter_demo.py
"""
import os

import numpy as np
from numpy.linalg import solve, inv, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from necredpy.models.credibility_nk import build_matrices, baseline_theta
from necredpy.pontus import solve_terminal, simulate_forward


# ---------------------------------------------------------------------------
# Helpers (reused from compare script)
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def smooth_miss(pi, epsilon_bar, tau):
    """Smooth miss indicator: sigmoid((|pi| - epsilon_bar) / tau)."""
    return sigmoid((np.abs(pi) - epsilon_bar) / tau)


def compute_cred_path(pi_path, theta, tau=None):
    """Compute credibility path from inflation path.

    Parameters
    ----------
    pi_path : ndarray (T,)
    theta : dict
    tau : float or None
        If None, use sharp threshold. If float, use sigmoid smoothing.

    Returns
    -------
    cred_path : ndarray (T,)
    omega_path : ndarray (T,)
    """
    T = len(pi_path)
    epsilon_bar = theta['epsilon_bar']
    delta_up = theta['delta_up']
    delta_down = theta['delta_down']
    omega_H = theta['omega_H']
    omega_L = theta['omega_L']

    cred_path = np.zeros(T)
    cred = 1.0

    for t in range(T):
        if tau is None:
            miss_t = 1.0 if abs(pi_path[t]) > epsilon_bar else 0.0
        else:
            miss_t = smooth_miss(pi_path[t], epsilon_bar, tau)

        cred = cred + delta_up * (1.0 - cred) * (1.0 - miss_t) \
                     - delta_down * cred * miss_t
        cred = max(0.0, min(1.0, cred))
        cred_path[t] = cred

    omega_path = omega_L + (omega_H - omega_L) * cred_path
    return cred_path, omega_path


def backward_recursion_timevarying(matrices_list, F_terminal):
    """Backward recursion with per-period matrices."""
    T = len(matrices_list)
    n = F_terminal.shape[0]
    F_path = np.zeros((T, n, n))
    E_path = np.zeros((T, n))
    Q_path = np.zeros((T, n, n))

    F_next = F_terminal.copy()
    E_next = np.zeros(n)

    for t in range(T - 1, -1, -1):
        A_t, B_t, C_t, D_t = matrices_list[t]
        M_t = B_t + C_t @ F_next
        F_t = -solve(M_t, A_t)
        E_t = -solve(M_t, C_t @ E_next + D_t)
        Q_t = -inv(M_t)
        F_path[t] = F_t
        E_path[t] = E_t
        Q_path[t] = Q_t
        F_next = F_t
        E_next = E_t

    return F_path, E_path, Q_path


# ---------------------------------------------------------------------------
# Forward solver with smooth credibility (for generating data)
# ---------------------------------------------------------------------------

def solve_smooth_model(theta, epsilon, tau, T, max_outer=100, tol=1e-10):
    """Solve model with sigmoid-smoothed credibility. Returns path + cred."""
    omega_H = theta['omega_H']
    pi_index = 1

    A1, B1, C1, D1 = build_matrices(theta, omega_H)
    n = A1.shape[0]
    F_terminal, _ = solve_terminal(A1, B1, C1)

    # Initial guess: linear M1
    from necredpy.pontus import backward_recursion as br_two
    regime_lin = np.zeros(T, dtype=int)
    F_p, E_p, Q_p = br_two(regime_lin, F_terminal,
                            (A1, B1, C1, D1), (A1, B1, C1, D1))
    u_path = simulate_forward(F_p, E_p, Q_p, epsilon)

    for outer in range(max_outer):
        cred_path, omega_path = compute_cred_path(
            u_path[:, pi_index], theta, tau=tau)

        matrices_list = [build_matrices(theta, omega_path[t]) for t in range(T)]
        F_path, E_path, Q_path = backward_recursion_timevarying(
            matrices_list, F_terminal)
        u_new = simulate_forward(F_path, E_path, Q_path, epsilon)

        diff = np.max(np.abs(u_new - u_path))
        u_path = u_new
        if diff < tol:
            return u_path, cred_path, omega_path, F_path, E_path, Q_path, True
    return u_path, cred_path, omega_path, F_path, E_path, Q_path, False


# ---------------------------------------------------------------------------
# Inversion filter
# ---------------------------------------------------------------------------

def inversion_filter(obs_y, obs_pi, obs_i, theta, tau=0.2,
                     max_outer=100, tol=1e-10):
    """Recover structural shocks from observed data using model inversion.

    Given observed (y_t, pi_t, i_t) for t=0..T-1, recovers the structural
    shocks (eps_d, eps_s, eps_m) that rationalize the data under the model.

    The filter is iterative because the policy functions {F_t, E_t} depend
    on the omega path, which depends on the inflation path, which is observed.
    So we:
      1. Compute cred/omega path from observed pi (this is deterministic)
      2. Build per-period matrices
      3. Solve for policy functions via backward recursion
      4. Invert: eps_t = Q_t^{-1} * (u_t - E_t - F_t * u_{t-1})

    Steps 1-4 are done once (no iteration needed) because the omega path
    is fully determined by the observed inflation path.

    Parameters
    ----------
    obs_y, obs_pi, obs_i : ndarray (T,)
        Observed output gap, inflation, interest rate (deviations from SS).
    theta : dict
    tau : float
        Sigmoid smoothing parameter.

    Returns
    -------
    eps_recovered : ndarray (T, 3) -- [eps_d, eps_s, eps_m]
    cred_path : ndarray (T,)
    omega_path : ndarray (T,)
    """
    T = len(obs_y)
    n = 4  # state dim (y, pi, i, pi_lag)
    pi_index = 1

    # Step 1: Compute cred/omega from observed inflation
    cred_path, omega_path = compute_cred_path(obs_pi, theta, tau=tau)

    # Step 2: Build per-period matrices
    omega_H = theta['omega_H']
    A1, B1, C1, D1 = build_matrices(theta, omega_H)
    F_terminal, _ = solve_terminal(A1, B1, C1)

    matrices_list = [build_matrices(theta, omega_path[t]) for t in range(T)]

    # Step 3: Backward recursion for policy functions
    F_path, E_path, Q_path = backward_recursion_timevarying(
        matrices_list, F_terminal)

    # Step 4: Reconstruct full state vector and invert
    # u_t = [y_t, pi_t, i_t, pi_lag_t]
    # pi_lag_t = pi_{t-1} (observed)
    u_path = np.zeros((T, n))
    u_path[:, 0] = obs_y
    u_path[:, 1] = obs_pi
    u_path[:, 2] = obs_i
    u_path[0, 3] = 0.0  # pi_lag at t=0 = steady state
    for t in range(1, T):
        u_path[t, 3] = obs_pi[t - 1]

    # Invert: eps_full_t = Q_t^{-1} (u_t - E_t - F_t u_{t-1})
    eps_full = np.zeros((T, n))
    u_prev = np.zeros(n)
    for t in range(T):
        residual = u_path[t] - E_path[t] - F_path[t] @ u_prev
        eps_full[t] = solve(Q_path[t], residual)
        u_prev = u_path[t]

    # Extract the 3 structural shocks (eps_d, eps_s, eps_m)
    # The 4th "shock" (identity equation) should be ~0
    eps_recovered = eps_full[:, :3]

    return eps_recovered, cred_path, omega_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    theta = baseline_theta()
    T = 80
    n = 4
    pi_index = 1
    tau = 0.2  # smoothing parameter for sigmoid

    np.random.seed(42)

    # -----------------------------------------------------------------------
    # Step 1: Generate synthetic data with known shocks
    # -----------------------------------------------------------------------
    # Design a scenario with a credibility episode:
    #   - Normal period (t=0..19): small random shocks
    #   - Supply shock episode (t=20..22): large cost-push shocks
    #   - Recovery (t=23..79): small random shocks, credibility rebuilds

    eps_true = np.zeros((T, n))

    # Small random shocks throughout
    eps_true[:, 0] = np.random.normal(0, theta['sigma_d'] * 0.3, T)  # demand
    eps_true[:, 1] = np.random.normal(0, theta['sigma_s'] * 0.3, T)  # supply
    eps_true[:, 2] = np.random.normal(0, theta['sigma_m'] * 0.2, T)  # monetary

    # Large supply shock episode at t=20-22
    eps_true[20, 1] += 2.5  # big cost-push
    eps_true[21, 1] += 1.5
    eps_true[22, 1] += 0.8

    # A second smaller episode at t=55
    eps_true[55, 1] += 1.8

    print("Solving model with known shocks (tau=%.2f)..." % tau)
    u_true, cred_true, omega_true, F_p, E_p, Q_p, conv = solve_smooth_model(
        theta, eps_true, tau, T)
    print("  Converged: %s" % conv)
    print("  Min credibility: %.3f at t=%d" % (
        np.min(cred_true), np.argmin(cred_true)))

    obs_y = u_true[:, 0]
    obs_pi = u_true[:, 1]
    obs_i = u_true[:, 2]

    # -----------------------------------------------------------------------
    # Step 2: Recover shocks via inversion filter
    # -----------------------------------------------------------------------
    print("\nRunning inversion filter...")
    eps_recovered, cred_recovered, omega_recovered = inversion_filter(
        obs_y, obs_pi, obs_i, theta, tau=tau)

    # Check recovery accuracy
    max_err = np.max(np.abs(eps_recovered - eps_true[:, :3]))
    print("  Max shock recovery error: %.2e" % max_err)

    # -----------------------------------------------------------------------
    # Step 3: Plot
    # -----------------------------------------------------------------------
    t_ax = np.arange(T)
    fig, axes = plt.subplots(4, 2, figsize=(14, 14))

    # --- Panel (0,0): Observed inflation ---
    ax = axes[0, 0]
    ax.plot(t_ax, obs_pi, 'k-', lw=1.5)
    ax.axhline(theta['epsilon_bar'], color='red', lw=0.8, ls='--', alpha=0.5,
               label='epsilon_bar')
    ax.axhline(-theta['epsilon_bar'], color='red', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.fill_between(t_ax, obs_pi, 0, alpha=0.15, color='steelblue')
    ax.set_title('Observed Inflation')
    ax.set_ylabel('pi_t')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (0,1): Observed output gap ---
    ax = axes[0, 1]
    ax.plot(t_ax, obs_y, 'k-', lw=1.5)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.fill_between(t_ax, obs_y, 0, alpha=0.15, color='steelblue')
    ax.set_title('Observed Output Gap')
    ax.set_ylabel('y_t')
    ax.grid(True, alpha=0.3)

    # --- Panel (1,0): Credibility over time (the dream plot!) ---
    ax = axes[1, 0]
    ax.plot(t_ax, cred_recovered, '-', color='#d62728', lw=2.5, label='Credibility')
    ax.axhline(theta['cred_threshold'], color='gray', lw=0.8, ls='--',
               label='Threshold (%.1f)' % theta['cred_threshold'])
    ax.fill_between(t_ax, cred_recovered, theta['cred_threshold'],
                    where=cred_recovered < theta['cred_threshold'],
                    alpha=0.2, color='red', label='Low credibility zone')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Credibility Capital Over Time')
    ax.set_ylabel('cred_t')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    # --- Panel (1,1): omega over time ---
    ax = axes[1, 1]
    ax.plot(t_ax, omega_recovered, '-', color='#9467bd', lw=2)
    ax.axhline(theta['omega_H'], color='gray', lw=0.5, ls=':',
               label='omega_H=%.2f' % theta['omega_H'])
    ax.axhline(theta['omega_L'], color='gray', lw=0.5, ls=':',
               label='omega_L=%.2f' % theta['omega_L'])
    ax.set_title('Forward-looking weight (omega)')
    ax.set_ylabel('omega_t')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (2,0): True vs recovered demand shocks ---
    ax = axes[2, 0]
    ax.stem(t_ax, eps_true[:, 0], linefmt='gray', markerfmt='o',
            basefmt='gray', label='True eps_d')
    ax.plot(t_ax, eps_recovered[:, 0], 'rx', ms=4, label='Recovered eps_d')
    ax.set_title('Demand Shocks: True vs Recovered')
    ax.set_ylabel('eps_d')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (2,1): True vs recovered supply shocks ---
    ax = axes[2, 1]
    ax.stem(t_ax, eps_true[:, 1], linefmt='gray', markerfmt='o',
            basefmt='gray', label='True eps_s')
    ax.plot(t_ax, eps_recovered[:, 1], 'rx', ms=4, label='Recovered eps_s')
    ax.set_title('Supply Shocks: True vs Recovered')
    ax.set_ylabel('eps_s')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (3,0): True vs recovered monetary shocks ---
    ax = axes[3, 0]
    ax.stem(t_ax, eps_true[:, 2], linefmt='gray', markerfmt='o',
            basefmt='gray', label='True eps_m')
    ax.plot(t_ax, eps_recovered[:, 2], 'rx', ms=4, label='Recovered eps_m')
    ax.set_title('Monetary Shocks: True vs Recovered')
    ax.set_ylabel('eps_m')
    ax.set_xlabel('Quarter')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel (3,1): Recovery error ---
    ax = axes[3, 1]
    err = eps_recovered - eps_true[:, :3]
    for j, (name, color) in enumerate(
            [('eps_d', '#1f77b4'), ('eps_s', '#d62728'), ('eps_m', '#2ca02c')]):
        ax.plot(t_ax, err[:, j], '-', color=color, lw=1, label=name, alpha=0.7)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_title('Shock Recovery Error (recovered - true)')
    ax.set_ylabel('Error')
    ax.set_xlabel('Quarter')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Inversion Filter: Shock Recovery + Credibility Path (tau=%.2f)'
                 % tau, fontsize=13, y=1.01)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'fig_inversion_filter.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print("\nSaved: %s" % out)


if __name__ == '__main__':
    main()
