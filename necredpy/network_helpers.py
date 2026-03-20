"""
Shared helpers for the network + credibility experiments.

PIPELINE OVERVIEW
=================

The model is defined in a single file:
    dynare/five_sector_network.mod

This .mod file contains:
    - Variable declarations (var, varexo)
    - Parameter declarations and values (parameters block)
    - Model equations (model block)
    - Credibility regime specification (regimes block)

To change the model (add sectors, change I-O matrix, change parameters),
edit the .mod file directly. The parser reads it automatically.

The pipeline is:
    1. dynare_parser.py reads the .mod file and computes symbolic Jacobians
       (A, B, C matrices) via sympy. This is the expensive step.
    2. The regimes block defines two regimes (M1 = high credibility,
       M2 = low credibility) that differ through the parameter omega.
    3. pontus.py solves the model:
       - Terminal regime: F_{k+1} = -(B + C*F_k)^{-1} * A
       - Backward recursion: time-varying F_t, E_t, Q_t
       - Forward simulation: u_t = E_t + F_t * u_{t-1} + Q_t * eps_t
       - Endogenous switching: iterate until regime sequence converges.

WHAT EACH FUNCTION DOES
=======================

load_model()       : Parse the .mod file, return regime matrices and info.
                     The info dict contains variable names, shock names,
                     parameter values, and the credibility specification.

solve_linear()     : Solve the model in a single regime (no credibility
                     switching). Uses the default parameter values from
                     the .mod file.

solve_pl()         : Solve with piecewise-linear credibility switching.
                     The regime sequence is determined endogenously by
                     the credibility law of motion.

find_threshold()   : Find the minimum shock size that triggers credibility
                     loss (M2 activation) using bisection.

sacrifice_ratio()  : Compute cum|output gap| / peak|inflation|.

HOW TO CHANGE SECTORS
=====================

SECTOR_NAMES and SECTOR_COLORS below are display labels only. They must
match the sectors defined in the .mod file (pi1..pi5, mc1..mc5, etc.).
To change the number of sectors, edit the .mod file and update these lists.
"""

import os
import copy
import numpy as np

from necredpy.pontus import solve_terminal, solve_endogenous, simulate_forward
from necredpy.utils.dynare_parser import (parse_two_regime_model, get_model_matrices,
                                           build_switching_fn)


# Display names and colors for the 5 sectors defined in the .mod file.
# These must match the sector ordering in five_sector_network.mod:
#   j=1: Energy, j=2: Food, j=3: Transport, j=4: Goods, j=5: Services
SECTOR_NAMES = ['Energy', 'Food', 'Transport', 'Goods', 'Services']
SECTOR_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

# Path to the .mod file. Change this to use a different model.
MOD_PATH = os.path.join(os.path.dirname(__file__), '..', 'dynare',
                        'five_sector_network.mod')


def load_model(param_overrides=None):
    """Parse the .mod file and return two-regime matrices + model info.

    Parameters
    ----------
    param_overrides : dict or None
        Override parameter values from the .mod file (e.g.,
        {'kappa1': 0.30, 's1': 0.15}). Applied to both regimes.

    Returns
    -------
    M1 : tuple (A1, B1, C1, D1)
        System matrices for the high-credibility regime.
    M2 : tuple (A2, B2, C2, D2)
        System matrices for the low-credibility regime.
    info : dict
        Contains: 'var_names' (list of variable names in order),
        'shock_names' (list of shock names), 'D_shock' (shock selection
        matrix, n_eq x n_shocks), 'regime_spec' (credibility parameters),
        'params_M1', 'params_M2' (full parameter dicts per regime).
    mod_string : str
        Raw .mod file content (needed for re-parsing in solve_pl).
    """
    with open(MOD_PATH, 'r') as f:
        mod_string = f.read()
    M1, M2, _, info = parse_two_regime_model(
        mod_string, param_overrides=param_overrides)
    return M1, M2, info, mod_string


def solve_linear(mod_string, overrides, shock_name, shock_size, T=60):
    """Solve the model in a single regime (no credibility switching).

    Uses the default omega value from the .mod file. This is the
    linear rational expectations solution: u_t = F * u_{t-1} + Q * eps_t.

    Parameters
    ----------
    mod_string : str
        Raw .mod file content (from load_model).
    overrides : dict
        Parameter overrides (e.g., {} for defaults).
    shock_name : str
        Name of the shock (e.g., 'eps_pi1' for Energy cost-push).
        Must match a varexo name in the .mod file.
    shock_size : float
        Size of the shock (positive = inflationary for cost-push shocks).
    T : int
        Simulation horizon in quarters.

    Returns
    -------
    u : ndarray (T, n)
        Simulated path. Columns correspond to info['var_names'].
    """
    A, B, C, D, info = get_model_matrices(mod_string, overrides or {})
    vn = info['var_names']
    sn = info['shock_names']
    n = len(vn)

    # Solve for the time-invariant policy function F
    F, Q = solve_terminal(A, B, C)

    # Build the shock vector. D_shock maps structural shocks (eps_pi1, etc.)
    # into equation-space. The negative sign accounts for the Pontus
    # convention Q = -(B+CF)^{-1}.
    si = sn.index(shock_name)
    eps = np.zeros((T, n))
    eps[0, :] = -info['D_shock'][:, si] * shock_size

    # Simulate with constant policy functions (single regime)
    u = simulate_forward(np.tile(F, (T, 1, 1)), np.zeros((T, n)),
                         np.tile(Q, (T, 1, 1)), eps)
    return u


def solve_pl(M1, M2, mod_string, shock_name, shock_size, info,
             T=60, param_overrides=None, cred_init=None):
    """Solve with piecewise-linear credibility switching.

    The regime sequence is determined endogenously: the solver iterates
    between (backward recursion + forward simulation) and (credibility
    update) until the regime sequence converges.

    Parameters
    ----------
    M1, M2 : tuple (A, B, C, D)
        System matrices for high/low credibility regimes (from load_model).
    mod_string : str
        Raw .mod file content (needed to construct a fresh switching
        function for each solve, since the switching function carries
        internal state).
    shock_name : str
        Name of the shock (e.g., 'eps_pi1').
    shock_size : float
        Size of the shock.
    info : dict
        Model info from load_model.
    T : int
        Simulation horizon.
    param_overrides : dict or None
        Parameter overrides (applied when constructing the switching fn).
    cred_init : float or None
        Override initial credibility level (default: 1.0 from .mod file).
        Use cred_init=0.30 to start from impaired credibility.

    Returns
    -------
    u : ndarray (T, n)
        Simulated path.
    reg : ndarray (T,)
        Regime sequence (0 = M1, 1 = M2).
    cred : ndarray (T,)
        Credibility path.
    conv : bool
        Whether the endogenous switching converged.
    """
    if cred_init is not None:
        # Build a switching function with a custom initial credibility
        spec = copy.deepcopy(info['regime_spec'])
        spec['cred_init'] = cred_init
        sw = build_switching_fn(spec, info['var_names'])
    else:
        # Parse fresh switching function (each solve needs its own,
        # because the switching function carries internal cred state)
        _, _, sw, _ = parse_two_regime_model(
            mod_string, param_overrides=param_overrides)

    vn = info['var_names']
    sn = info['shock_names']
    n = len(vn)

    # Build shock vector (same convention as solve_linear)
    si = sn.index(shock_name)
    eps = np.zeros((T, n))
    eps[0, :] = -info['D_shock'][:, si] * shock_size

    # Solve with endogenous switching
    u, reg, _, _, _, conv, iters = solve_endogenous(M1, M2, sw, eps, T)
    cred = sw.cred_path
    return u, reg, cred, conv


def sacrifice_ratio(y, pi):
    """Sacrifice ratio: cumulative |output gap| / peak |inflation|."""
    cum_y = np.sum(np.abs(y))
    peak_pi = np.max(np.abs(pi))
    return cum_y / peak_pi if peak_pi > 1e-12 else 0.0


def find_threshold(M1, M2, mod_string, info, shock_name, T=60,
                   cred_init=None):
    """Find the minimum shock size that triggers credibility loss.

    Uses bisection: for each candidate shock size, solve the PL model
    and check whether any period is in M2. Converges to the threshold
    in ~30 iterations (precision ~10^{-9}).

    Parameters
    ----------
    M1, M2 : regime matrices
    mod_string : str
    info : dict
    shock_name : str
        Which shock to test (e.g., 'eps_pi1').
    T : int
        Simulation horizon.
    cred_init : float or None
        Initial credibility (default: 1.0).

    Returns
    -------
    threshold : float
        Minimum shock size that activates M2.
    """
    vn = info['var_names']
    sn = info['shock_names']
    n = len(vn)
    D_shock = info['D_shock']
    si = sn.index(shock_name)

    lo, hi = 0.1, 80.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        if cred_init is not None:
            spec = copy.deepcopy(info['regime_spec'])
            spec['cred_init'] = cred_init
            sw = build_switching_fn(spec, vn)
        else:
            _, _, sw, _ = parse_two_regime_model(mod_string)
        eps = np.zeros((T, n))
        eps[0, :] = -D_shock[:, si] * mid
        try:
            u, reg, _, _, _, conv, _ = solve_endogenous(M1, M2, sw, eps, T)
            if np.any(reg == 1):
                hi = mid
            else:
                lo = mid
        except Exception:
            hi = mid
    return (lo + hi) / 2.0


def fig_path(filename):
    """Return full path to a figure in the figures/ directory."""
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def print_io_table(info):
    """Print the I-O matrix and CPI weights to the console."""
    params = info['params_M1']
    print("CPI weights: ", end="")
    for j in range(5):
        print("%s=%.2f  " % (SECTOR_NAMES[j], params['s%d' % (j+1)]), end="")
    print()
    print("\nI-O matrix (omega_jk, row=buyer, col=seller):")
    print("%12s" % "", end="")
    for j in range(5):
        print("%10s" % SECTOR_NAMES[j], end="")
    print()
    for j in range(5):
        print("%12s" % SECTOR_NAMES[j], end="")
        for k in range(5):
            if j == k:
                print("%10s" % "--", end="")
            else:
                key = 'omega_%d%d' % (j + 1, k + 1)
                print("%10.2f" % params.get(key, 0.0), end="")
        print()
