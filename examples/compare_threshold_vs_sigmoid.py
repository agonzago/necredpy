"""
Compare sharp-threshold vs sigmoid-smoothed credibility mechanism.

PURPOSE
-------
The sharp threshold miss_t = 1{|pi_t| > epsilon_bar} creates discontinuities
in the likelihood as a function of parameters, which breaks HMC/NUTS.
The sigmoid approximation miss_t = sigmoid((|pi_t| - epsilon_bar) / tau)
smooths this, making the likelihood differentiable.

This script compares the two approaches:
  1. PL solver with sharp threshold (existing Pontus code)
  2. Iterative solver with sigmoid-smoothed credibility (continuous omega_t)

For several values of tau, we show that as tau -> 0 the smooth solution
converges to the sharp-threshold PL solution.

Produces: figures/fig_threshold_vs_sigmoid.png

Usage: .venv/bin/python scripts/compare_threshold_vs_sigmoid.py
"""
import os

import numpy as np
from numpy.linalg import solve, inv, norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from necredpy.models.credibility_nk import (
    build_matrices, baseline_theta, make_switching_fn_cred, build_matrices_with_cred
)
from necredpy.pontus import solve_terminal, backward_recursion, simulate_forward, solve_endogenous


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def smooth_miss(pi, epsilon_bar, tau):
    """Smooth approximation of miss_t = 1{|pi_t| > epsilon_bar}.

    miss_t = sigmoid((|pi_t| - epsilon_bar) / tau)

    As tau -> 0, this converges to the sharp indicator.
    """
    return sigmoid((np.abs(pi) - epsilon_bar) / tau)


# ---------------------------------------------------------------------------
# Smooth credibility solver (iterative, time-varying omega)
# ---------------------------------------------------------------------------

def backward_recursion_timevarying(matrices_list, F_terminal):
    """Backward recursion with per-period matrices.

    Parameters
    ----------
    matrices_list : list of (A_t, B_t, C_t, D_t) tuples, length T
    F_terminal : ndarray (n, n)

    Returns
    -------
    F_path, E_path, Q_path : ndarray (T, n, n), (T, n), (T, n, n)
    """
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


def solve_smooth(theta, epsilon, tau, T_max, max_outer=80, tol=1e-8,
                 pi_index=1):
    """Solve the model with sigmoid-smoothed credibility.

    Instead of discrete M1/M2, omega_t varies continuously:
      miss_t = sigmoid((|pi_t| - epsilon_bar) / tau)
      cred_t = cred_{t-1} + delta_up*(1-cred_{t-1})*(1-miss_t)
                           - delta_down*cred_{t-1}*miss_t
      omega_t = omega_L + (omega_H - omega_L) * cred_t

    Uses iterative scheme: guess path -> compute omega_t -> solve
    linear system with time-varying coefficients -> update path.

    Parameters
    ----------
    theta : dict
    epsilon : ndarray (T_max, n_shocks)
    tau : float
        Smoothing parameter. Smaller = closer to sharp threshold.
    T_max : int
    max_outer : int
    tol : float
        Convergence tolerance on max change in path.
    pi_index : int

    Returns
    -------
    u_path : ndarray (T_max, n)
    cred_path : ndarray (T_max,)
    omega_path : ndarray (T_max,)
    converged : bool
    iters : int
    """
    epsilon_bar = theta['epsilon_bar']
    delta_up = theta['delta_up']
    delta_down = theta['delta_down']
    omega_H = theta['omega_H']
    omega_L = theta['omega_L']

    # Use 4-variable model (no cred state -- cred is tracked externally)
    # Terminal regime is M1 (high credibility)
    A1, B1, C1, D1 = build_matrices(theta, omega_H)
    n = A1.shape[0]
    F_terminal, _ = solve_terminal(A1, B1, C1)

    # Initial guess: solve linear M1 model (all high credibility)
    regime_seq = np.zeros(T_max, dtype=int)
    matrices_M1 = (A1, B1, C1, D1)
    F_path, E_path, Q_path = backward_recursion(
        regime_seq, F_terminal, matrices_M1, matrices_M1)
    u_path = simulate_forward(F_path, E_path, Q_path, epsilon)

    for outer in range(max_outer):
        # Compute smooth credibility path from current inflation path
        pi_path = u_path[:, pi_index]
        cred_path = np.zeros(T_max)
        cred = 1.0  # start at full credibility

        for t in range(T_max):
            miss_t = smooth_miss(pi_path[t], epsilon_bar, tau)
            cred = cred + delta_up * (1.0 - cred) * (1.0 - miss_t) \
                       - delta_down * cred * miss_t
            cred = max(0.0, min(1.0, cred))
            cred_path[t] = cred

        # Compute omega_t = omega_L + (omega_H - omega_L) * cred_t
        omega_path = omega_L + (omega_H - omega_L) * cred_path

        # Build per-period matrices
        matrices_list = []
        for t in range(T_max):
            A_t, B_t, C_t, D_t = build_matrices(theta, omega_path[t])
            matrices_list.append((A_t, B_t, C_t, D_t))

        # Backward recursion with time-varying matrices
        F_path, E_path, Q_path = backward_recursion_timevarying(
            matrices_list, F_terminal)

        # Forward simulate
        u_path_new = simulate_forward(F_path, E_path, Q_path, epsilon)

        # Check convergence
        diff = np.max(np.abs(u_path_new - u_path))
        u_path = u_path_new

        if diff < tol:
            return u_path, cred_path, omega_path, True, outer + 1

    return u_path, cred_path, omega_path, False, max_outer


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def main():
    theta = baseline_theta()
    T = 60
    n = 4  # state vector size (y, pi, i, pi_lag)
    pi_index = 1

    # Shock: cost-push, size 3.0 (big enough to breach epsilon_bar=2.0)
    shock_size = 3.0
    epsilon = np.zeros((T, n))
    epsilon[0, 1] = shock_size  # eps_s at t=0 (inflationary cost-push)

    # -----------------------------------------------------------------------
    # 1. Sharp threshold (PL solver)
    # -----------------------------------------------------------------------
    matrices_M1 = build_matrices(theta, theta['omega_H'])
    matrices_M2 = build_matrices(theta, theta['omega_L'])

    sw_fn = make_switching_fn_cred(
        epsilon_bar=theta['epsilon_bar'],
        cred_threshold=theta['cred_threshold'],
        delta_up=theta['delta_up'],
        delta_down=theta['delta_down'],
        pi_index=pi_index,
        cred_init=1.0,
    )

    u_pl, regime_pl, _, _, _, conv_pl, iters_pl = solve_endogenous(
        matrices_M1, matrices_M2, sw_fn, epsilon, T, max_outer=80)
    cred_pl = sw_fn.cred_path

    print("PL (sharp threshold): converged=%s, iters=%d" % (conv_pl, iters_pl))
    print("  M2 periods: %d" % np.sum(regime_pl == 1))

    # -----------------------------------------------------------------------
    # 2. Linear (no credibility switching, always M1)
    # -----------------------------------------------------------------------
    F_term, _ = solve_terminal(*matrices_M1[:3])
    regime_lin = np.zeros(T, dtype=int)
    F_p, E_p, Q_p = backward_recursion(regime_lin, F_term, matrices_M1, matrices_M1)
    u_lin = simulate_forward(F_p, E_p, Q_p, epsilon)

    # -----------------------------------------------------------------------
    # 3. Sigmoid-smoothed for different tau values
    # -----------------------------------------------------------------------
    tau_values = [1.0, 0.5, 0.2, 0.1, 0.05]
    smooth_results = {}

    for tau in tau_values:
        u_sm, cred_sm, omega_sm, conv_sm, iters_sm = solve_smooth(
            theta, epsilon, tau, T, max_outer=150, tol=1e-10)
        smooth_results[tau] = {
            'u': u_sm, 'cred': cred_sm, 'omega': omega_sm,
            'converged': conv_sm, 'iters': iters_sm,
        }
        print("Sigmoid tau=%.2f: converged=%s, iters=%d" % (tau, conv_sm, iters_sm))

    # -----------------------------------------------------------------------
    # 4. Plot comparison
    # -----------------------------------------------------------------------
    t_ax = np.arange(T)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Color map for tau values (darker = smaller tau = sharper)
    from matplotlib.cm import Blues
    tau_colors = {tau: Blues(0.3 + 0.6 * i / (len(tau_values) - 1))
                  for i, tau in enumerate(tau_values)}

    # --- Panel (0,0): Inflation ---
    ax = axes[0, 0]
    ax.plot(t_ax, u_lin[:, pi_index], '--', color='gray', lw=1, label='Linear (no cred)')
    ax.plot(t_ax, u_pl[:, pi_index], 'k-', lw=2.5, label='PL (sharp)')
    for tau in tau_values:
        r = smooth_results[tau]
        ax.plot(t_ax, r['u'][:, pi_index], '-', color=tau_colors[tau], lw=1.5,
                label='tau=%.2f' % tau)
    ax.axhline(theta['epsilon_bar'], color='red', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(-theta['epsilon_bar'], color='red', lw=0.8, ls='--', alpha=0.5)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_title('Inflation (pi)')
    ax.set_ylabel('Deviation from target')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Panel (0,1): Output gap ---
    ax = axes[0, 1]
    ax.plot(t_ax, u_lin[:, 0], '--', color='gray', lw=1, label='Linear')
    ax.plot(t_ax, u_pl[:, 0], 'k-', lw=2.5, label='PL (sharp)')
    for tau in tau_values:
        r = smooth_results[tau]
        ax.plot(t_ax, r['u'][:, 0], '-', color=tau_colors[tau], lw=1.5,
                label='tau=%.2f' % tau)
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.set_title('Output gap (y)')
    ax.set_ylabel('Deviation')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    # --- Panel (1,0): Credibility path ---
    ax = axes[1, 0]
    ax.plot(t_ax, cred_pl, 'k-', lw=2.5, label='PL (sharp)')
    for tau in tau_values:
        r = smooth_results[tau]
        ax.plot(t_ax, r['cred'], '-', color=tau_colors[tau], lw=1.5,
                label='tau=%.2f' % tau)
    ax.axhline(theta['cred_threshold'], color='red', lw=0.8, ls='--', alpha=0.5,
               label='cred threshold')
    ax.set_ylim(-0.05, 1.1)
    ax.set_title('Credibility capital')
    ax.set_ylabel('cred_t')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel (1,1): omega path ---
    ax = axes[1, 1]
    # For PL: omega is omega_H in M1, omega_L in M2
    omega_pl = np.where(regime_pl == 0, theta['omega_H'], theta['omega_L'])
    ax.step(t_ax, omega_pl, 'k-', lw=2.5, where='mid', label='PL (sharp)')
    for tau in tau_values:
        r = smooth_results[tau]
        ax.plot(t_ax, r['omega'], '-', color=tau_colors[tau], lw=1.5,
                label='tau=%.2f' % tau)
    ax.axhline(theta['omega_H'], color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.axhline(theta['omega_L'], color='gray', lw=0.5, ls=':', alpha=0.5)
    ax.set_title('Forward-looking weight (omega)')
    ax.set_ylabel('omega_t')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel (2,0): miss_t indicator for different tau ---
    ax = axes[2, 0]
    pi_grid = np.linspace(-5, 5, 500)
    ax.plot(pi_grid, (np.abs(pi_grid) > theta['epsilon_bar']).astype(float),
            'k-', lw=2.5, label='Sharp')
    for tau in tau_values:
        ax.plot(pi_grid, smooth_miss(pi_grid, theta['epsilon_bar'], tau),
                '-', color=tau_colors[tau], lw=1.5, label='tau=%.2f' % tau)
    ax.axvline(theta['epsilon_bar'], color='red', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(-theta['epsilon_bar'], color='red', lw=0.5, ls='--', alpha=0.5)
    ax.set_title('miss(pi) function')
    ax.set_xlabel('pi (deviation)')
    ax.set_ylabel('miss probability')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel (2,1): Max abs difference vs tau ---
    ax = axes[2, 1]
    diffs_pi = []
    diffs_y = []
    diffs_cred = []
    for tau in tau_values:
        r = smooth_results[tau]
        diffs_pi.append(np.max(np.abs(r['u'][:, pi_index] - u_pl[:, pi_index])))
        diffs_y.append(np.max(np.abs(r['u'][:, 0] - u_pl[:, 0])))
        diffs_cred.append(np.max(np.abs(r['cred'] - cred_pl)))
    ax.semilogy(tau_values, diffs_pi, 'o-', color='#d62728', lw=2, label='max|pi_smooth - pi_PL|')
    ax.semilogy(tau_values, diffs_y, 's-', color='#1f77b4', lw=2, label='max|y_smooth - y_PL|')
    ax.semilogy(tau_values, diffs_cred, '^-', color='#2ca02c', lw=2, label='max|cred_smooth - cred_PL|')
    ax.set_title('Convergence: smooth -> sharp as tau -> 0')
    ax.set_xlabel('tau (smoothing parameter)')
    ax.set_ylabel('Max absolute difference')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    fig.suptitle(
        'Sharp Threshold vs Sigmoid Smoothing (cost-push = %.1f, eps_bar = %.1f)'
        % (shock_size, theta['epsilon_bar']),
        fontsize=13, y=1.01)
    plt.tight_layout()

    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'fig_threshold_vs_sigmoid.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print("\nSaved: %s" % out)


if __name__ == '__main__':
    main()
