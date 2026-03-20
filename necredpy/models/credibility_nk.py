"""
Three-equation New Keynesian model with credibility-weighted Phillips curve.

State vector: u_t = [y_t, pi_t, i_t, pi_{t-1}]'
All variables in deviation from steady state (pi*, 0, r_n + pi*, pi*).

Equations (see paper_plan.md Section 2):

  IS:       y_t = E[y_{t+1}] - sigma*(i_t - E[pi_{t+1}]) + eps_d
  PC:       pi_t = omega*beta*E[pi_{t+1}] + (1-omega)*pi_{t-1} + kappa*y_t + eps_s
  Taylor:   i_t = rho_i*i_{t-1} + (1-rho_i)*(phi_pi*pi_t + phi_y*y_t) + eps_m
  Identity: pi_{t-1}^{today} = pi_t^{yesterday}

Written as A u_{t-1} + B u_t + C E[u_{t+1}] + D = 0.

Shock ordering: epsilon = [eps_d, eps_s, eps_m]'
The shock enters through Q = -(B + CF)^{-1}, so we do not need a separate
shock-selection matrix. The shocks map one-to-one with equations 1, 2, 3
(the identity has no shock).

Regimes differ only through omega in the Phillips curve:
  M1 (high credibility): omega = omega_H
  M2 (low credibility):  omega = omega_L
"""

import numpy as np


def build_matrices(theta, omega):
    """Build A, B, C, D matrices for one regime.

    Parameters
    ----------
    theta : dict with keys:
        beta, sigma, kappa, rho_i, phi_pi, phi_y
    omega : float
        Credibility weight for this regime.

    Returns
    -------
    A, B, C, D : ndarray (4, 4), (4, 4), (4, 4), (4,)

    Matrix derivation
    -----------------
    Rewrite each equation as: row_A * u_{t-1} + row_B * u_t + row_C * u_{t+1} = 0

    Variables indexed as: 0=y, 1=pi, 2=i, 3=pi_lag

    Eq 1 (IS): y - E[y+1] + sigma*i - sigma*E[pi+1] = 0
      (we move eps_d to the RHS; it enters through Q, not D)
      A: [0, 0, 0, 0]
      B: [1, 0, sigma, 0]
      C: [-1, -sigma, 0, 0]

    Eq 2 (PC): -kappa*y + pi - omega*beta*E[pi+1] - (1-omega)*pi_lag = 0
      A: [0, 0, 0, 0]
      B: [-kappa, 1, 0, -(1-omega)]
      C: [0, -omega*beta, 0, 0]

    Eq 3 (Taylor): -(1-rho_i)*phi_y*y - (1-rho_i)*phi_pi*pi + i - rho_i*i_{t-1} = 0
      A: [0, 0, -rho_i, 0]
      B: [-(1-rho_i)*phi_y, -(1-rho_i)*phi_pi, 1, 0]
      C: [0, 0, 0, 0]

    Eq 4 (Identity): pi_lag_t - pi_{t-1} = 0
      A: [0, -1, 0, 0]
      B: [0, 0, 0, 1]
      C: [0, 0, 0, 0]
    """
    beta = theta['beta']
    sigma = theta['sigma']
    kappa = theta['kappa']
    rho_i = theta['rho_i']
    phi_pi = theta['phi_pi']
    phi_y = theta['phi_y']

    A = np.array([
        [0.0,  0.0,     0.0,  0.0],
        [0.0,  0.0,     0.0,  0.0],
        [0.0,  0.0,  -rho_i,  0.0],
        [0.0, -1.0,     0.0,  0.0],
    ])

    B = np.array([
        [1.0,                    0.0,     sigma,           0.0],
        [-kappa,                 1.0,     0.0,   -(1.0 - omega)],
        [-(1-rho_i)*phi_y, -(1-rho_i)*phi_pi, 1.0,       0.0],
        [0.0,                    0.0,     0.0,             1.0],
    ])

    C = np.array([
        [-1.0,        -sigma,  0.0,  0.0],
        [0.0,   -omega*beta,   0.0,  0.0],
        [0.0,           0.0,   0.0,  0.0],
        [0.0,           0.0,   0.0,  0.0],
    ])

    D = np.zeros(4)

    return A, B, C, D


def build_model(theta):
    """Build matrices for both regimes.

    Parameters
    ----------
    theta : dict with keys:
        beta, sigma, kappa, rho_i, phi_pi, phi_y, omega_H, omega_L

    Returns
    -------
    matrices_M1 : tuple (A1, B1, C1, D1) -- high credibility
    matrices_M2 : tuple (A2, B2, C2, D2) -- low credibility
    """
    matrices_M1 = build_matrices(theta, theta['omega_H'])
    matrices_M2 = build_matrices(theta, theta['omega_L'])
    return matrices_M1, matrices_M2


def baseline_theta():
    """Baseline calibration.

    omega_H and omega_L set so that the backward-looking share
    (1 - omega) is meaningfully positive in BOTH regimes:
      M1: 1 - 0.65 = 0.35  (moderate persistence)
      M2: 1 - 0.35 = 0.65  (high persistence)
    This follows the QPM/FPAS tradition (Isard-Laxton-Eliasson, 2001)
    where credibility shifts the RELATIVE weight, not a binary switch.

    Credibility capital parameters (Laxton-style):
      delta_up: rate at which cred rebuilds when on target (slow)
      delta_down: rate at which cred depletes when off target (fast)
      cred_threshold: cred level above which M1 restores
    """
    return {
        'beta': 0.99,
        'sigma': 1.0,
        'kappa': 0.3,
        'rho_i': 0.7,
        'phi_pi': 1.5,
        'phi_y': 0.5,
        'omega_H': 0.65,
        'omega_L': 0.35,
        'epsilon_bar': 2.0,
        'k_restore': 4,
        'delta_up': 0.05,
        'delta_down': 0.7,
        'cred_threshold': 0.5,
        'sigma_d': 0.5,
        'sigma_s': 0.5,
        'sigma_m': 0.25,
    }


def make_switching_fn(epsilon_bar, k_restore, pi_index=1):
    """Create a switching function for the credibility model.

    M2 activates when |pi_t| > epsilon_bar.
    M1 restores when |pi_t| <= epsilon_bar for k_restore consecutive periods.

    Parameters
    ----------
    epsilon_bar : float
        Inflation tolerance (in deviation from target).
    k_restore : int
        Consecutive periods within band needed to restore M1.
    pi_index : int
        Index of inflation in the state vector. Default 1.

    Returns
    -------
    switching_fn : callable(u_path) -> regime_seq
    """
    def switching_fn(u_path):
        T = u_path.shape[0]
        regime_seq = np.zeros(T, dtype=int)

        in_M2 = False
        consecutive_in_band = 0

        for t in range(T):
            pi_t = u_path[t, pi_index]

            if not in_M2:
                # Currently M1
                if abs(pi_t) > epsilon_bar:
                    in_M2 = True
                    consecutive_in_band = 0
                    regime_seq[t] = 1
                else:
                    regime_seq[t] = 0
            else:
                # Currently M2
                if abs(pi_t) <= epsilon_bar:
                    consecutive_in_band += 1
                    if consecutive_in_band >= k_restore:
                        in_M2 = False
                        regime_seq[t] = 0
                    else:
                        regime_seq[t] = 1
                else:
                    consecutive_in_band = 0
                    regime_seq[t] = 1

        return regime_seq

    return switching_fn


# ---------------------------------------------------------------------------
# Credibility capital stock model (5-variable)
# ---------------------------------------------------------------------------
# State vector: u_t = [y_t, pi_t, i_t, pi_{t-1}, cred_t]'
#
# The first 4 equations are identical to the 4-variable model.
# Equation 5 (cred law of motion) differs by regime:
#   M1: cred_t = (1-delta_up)*cred_{t-1} + delta_up      (drift toward 1)
#   M2: cred_t = (1-delta_down)*cred_{t-1}                (drift toward 0)
#
# All deviations are from the M1 steady state (cred*=1):
#   M1: cred_hat_t = (1-delta_up)*cred_hat_{t-1},  D=0
#   M2: cred_hat_t = (1-delta_down)*cred_hat_{t-1} - delta_down,  D[4]=-delta_down
#
# The cred equation is decoupled from the first 4 equations: omega is fixed
# within each regime (omega_H in M1, omega_L in M2), so cred does not feed
# back into IS/PC/Taylor. But cred controls the switching criterion.
# ---------------------------------------------------------------------------


def build_matrices_with_cred(theta, omega, delta, is_M2=False):
    """Build 5x5 A, B, C, D matrices with credibility capital stock.

    Parameters
    ----------
    theta : dict
        Model parameters.
    omega : float
        Credibility weight for this regime (omega_H or omega_L).
    delta : float
        Cred adjustment rate (delta_up for M1, delta_down for M2).
    is_M2 : bool
        If True, sets D[4] = -delta_down (constant from shared SS reference).

    Returns
    -------
    A, B, C, D : ndarray (5,5), (5,5), (5,5), (5,)

    Matrix derivation for row 4 (cred equation):
      M1: cred_hat_t - (1-delta_up)*cred_hat_{t-1} = 0
        A[4,4] = -(1-delta_up),  B[4,4] = 1,  C[4,:] = 0,  D[4] = 0
      M2: cred_hat_t - (1-delta_down)*cred_hat_{t-1} + delta_down = 0
        A[4,4] = -(1-delta_down),  B[4,4] = 1,  C[4,:] = 0,  D[4] = +delta_down

    Derivation of D[4] for M2:
      Actual: cred_t = (1-delta_down)*cred_{t-1}
      Deviation from cred*=1: cred_hat = cred - 1
      (cred_hat_t+1) = (1-delta_down)*(cred_hat_{t-1}+1)
      cred_hat_t = (1-delta_down)*cred_hat_{t-1} - delta_down
      Rearranged: -(1-delta_down)*cred_hat_{t-1} + cred_hat_t + delta_down = 0
      So D[4] = +delta_down.
    """
    beta = theta['beta']
    sigma = theta['sigma']
    kappa = theta['kappa']
    rho_i = theta['rho_i']
    phi_pi = theta['phi_pi']
    phi_y = theta['phi_y']

    A = np.zeros((5, 5))
    A[2, 2] = -rho_i
    A[3, 1] = -1.0
    A[4, 4] = -(1.0 - delta)

    B = np.zeros((5, 5))
    B[0, 0] = 1.0
    B[0, 2] = sigma
    B[1, 0] = -kappa
    B[1, 1] = 1.0
    B[1, 3] = -(1.0 - omega)
    B[2, 0] = -(1.0 - rho_i) * phi_y
    B[2, 1] = -(1.0 - rho_i) * phi_pi
    B[2, 2] = 1.0
    B[3, 3] = 1.0
    B[4, 4] = 1.0

    C = np.zeros((5, 5))
    C[0, 0] = -1.0
    C[0, 1] = -sigma
    C[1, 1] = -omega * beta

    D = np.zeros(5)
    if is_M2:
        D[4] = theta['delta_down']

    return A, B, C, D


def build_model_with_cred(theta):
    """Build 5x5 matrices for both regimes with credibility capital.

    Parameters
    ----------
    theta : dict with keys:
        beta, sigma, kappa, rho_i, phi_pi, phi_y,
        omega_H, omega_L, delta_up, delta_down

    Returns
    -------
    matrices_M1 : tuple (A1, B1, C1, D1) -- high credibility, 5x5
    matrices_M2 : tuple (A2, B2, C2, D2) -- low credibility, 5x5
    """
    matrices_M1 = build_matrices_with_cred(
        theta, theta['omega_H'], theta['delta_up'], is_M2=False)
    matrices_M2 = build_matrices_with_cred(
        theta, theta['omega_L'], theta['delta_down'], is_M2=True)
    return matrices_M1, matrices_M2


def make_switching_fn_cred(epsilon_bar, cred_threshold, delta_up, delta_down,
                           pi_index=1, cred_init=1.0):
    """Create a switching function with shadow credibility capital stock.

    Tracks a shadow credibility variable using the nonlinear law of motion
    (same as the PF model), independent of the regime matrices:

      miss_t = 1 if |pi_t| > epsilon_bar else 0
      cred_t = cred_{t-1} + delta_up*(1 - cred_{t-1})*(1 - miss_t)
                           - delta_down*cred_{t-1}*miss_t

    On target (miss=0): cred drifts toward 1 at rate delta_up (slow)
    Off target (miss=1): cred drifts toward 0 at rate delta_down (fast)

    Regime assignment:
      M2 (omega_L) when cred_t < cred_threshold
      M1 (omega_H) when cred_t >= cred_threshold

    This produces shock-size-dependent sacrifice ratios because larger
    shocks push inflation out of band for more periods, depleting cred
    further and requiring more time to rebuild past the threshold.

    The shadow cred path is stored as switching_fn.cred_path after each
    call, for plotting.

    Parameters
    ----------
    epsilon_bar : float
        Inflation tolerance (deviation from target).
    cred_threshold : float
        Credibility level below which M2 activates (e.g. 0.5).
    delta_up : float
        Rate at which cred rebuilds when on target (slow, e.g. 0.1).
    delta_down : float
        Rate at which cred depletes when off target (fast, e.g. 0.3).
    pi_index : int
        Index of inflation in the state vector. Default 1.
    cred_init : float
        Initial credibility level. Default 1.0 (full credibility).

    Returns
    -------
    switching_fn : callable(u_path) -> regime_seq
        Also stores switching_fn.cred_path after each call.
    """
    def switching_fn(u_path):
        T = u_path.shape[0]
        regime_seq = np.zeros(T, dtype=int)
        cred_path = np.zeros(T)

        cred = cred_init

        for t in range(T):
            pi_t = u_path[t, pi_index]

            # Update cred using nonlinear law of motion
            if abs(pi_t) > epsilon_bar:
                # Off target: deplete
                cred = cred - delta_down * cred
            else:
                # On target: rebuild
                cred = cred + delta_up * (1.0 - cred)

            # Clamp to [0, 1] for safety
            cred = max(0.0, min(1.0, cred))
            cred_path[t] = cred

            # Assign regime based on cred level
            if cred < cred_threshold:
                regime_seq[t] = 1  # M2
            else:
                regime_seq[t] = 0  # M1

        # Store cred path for plotting
        switching_fn.cred_path = cred_path
        return regime_seq

    switching_fn.cred_path = None
    return switching_fn
