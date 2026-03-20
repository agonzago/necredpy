"""
Stability check for the piecewise-linear model.

The solution is u_t = E_t + F_t u_{t-1} + Q_t epsilon_t.
The system is stable iff all eigenvalues of F lie strictly inside the
unit circle: max|lambda(F)| < 1.

For the smooth-credibility model, we need stability for ALL omega in
[omega_L, omega_H], not just the two extreme regimes. We check this
by evaluating the spectral radius of F(omega) on a grid.

Usage:
    from necredpy.stability import check_stability, stability_region
"""

import numpy as np
from numpy.linalg import eigvals
from necredpy.models.credibility_nk import build_matrices
from necredpy.pontus import solve_terminal_pontus


def spectral_radius(F):
    """Max absolute eigenvalue of F."""
    return np.max(np.abs(eigvals(F)))


def check_stability(theta, omega, verbose=True):
    """Check if the terminal solution F(omega) is stable.

    Parameters
    ----------
    theta : dict
    omega : float
        Credibility weight.
    verbose : bool

    Returns
    -------
    stable : bool
    rho : float
        Spectral radius of F.
    eigenvalues : ndarray
    """
    A, B, C, D = build_matrices(theta, omega)
    F, Q, converged, iters = solve_terminal_pontus(A, B, C, max_iter=2000)

    if not converged:
        if verbose:
            print("  WARNING: terminal solver did not converge for omega=%.3f" % omega)
        return False, np.inf, np.array([])

    eigs = eigvals(F)
    rho = np.max(np.abs(eigs))
    stable = rho < 1.0

    if verbose:
        print("  omega=%.3f: rho(F)=%.6f  stable=%s  eigs=%s" % (
            omega, rho, stable, np.round(eigs, 4)))

    return stable, rho, eigs


def stability_region(theta, n_grid=50, verbose=True):
    """Check stability across omega in [omega_L, omega_H] and beyond.

    Parameters
    ----------
    theta : dict
    n_grid : int
        Number of grid points.
    verbose : bool

    Returns
    -------
    omega_grid : ndarray
    rho_grid : ndarray
        Spectral radius at each grid point.
    all_stable : bool
    """
    # Check a range slightly wider than [omega_L, omega_H]
    omega_lo = max(0.01, theta['omega_L'] - 0.1)
    omega_hi = min(0.99, theta['omega_H'] + 0.1)
    omega_grid = np.linspace(omega_lo, omega_hi, n_grid)
    rho_grid = np.zeros(n_grid)

    if verbose:
        print("Stability check: omega in [%.2f, %.2f]" % (omega_lo, omega_hi))

    for i, omega in enumerate(omega_grid):
        _, rho, _ = check_stability(theta, omega, verbose=False)
        rho_grid[i] = rho

    all_stable = np.all(rho_grid < 1.0)

    if verbose:
        print("  max spectral radius: %.6f at omega=%.3f" % (
            np.max(rho_grid), omega_grid[np.argmax(rho_grid)]))
        print("  ALL STABLE: %s" % all_stable)

        # Highlight the two regime endpoints
        for label, omega in [('omega_L', theta['omega_L']),
                             ('omega_H', theta['omega_H'])]:
            s, r, e = check_stability(theta, omega, verbose=False)
            print("  %s=%.2f: rho=%.6f stable=%s" % (label, omega, r, s))

    return omega_grid, rho_grid, all_stable


if __name__ == '__main__':
    from necredpy.models.credibility_nk import baseline_theta
    theta = baseline_theta()
    stability_region(theta)
