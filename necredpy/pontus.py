"""
Piecewise-linear solver for two-regime DSGE models.

Implements the Pontus time iteration method (Rendahl, 2017) with:
  - Simple iteration and structured doubling for the terminal regime
  - Backward recursion for the temporary regime
  - Forward simulation
  - Endogenous switching loop

All notation follows research_agenda.md Sections 2.1-2.5 and CLAUDE.md.

System: A u_{t-1} + B u_t + C E_t[u_{t+1}] + D = 0
Solution: u_t = E_const + F u_{t-1} + Q epsilon_t
"""

import numpy as np
from numpy.linalg import solve, inv, norm


# ---------------------------------------------------------------------------
# Terminal regime solvers
# ---------------------------------------------------------------------------

def solve_terminal_pontus(A, B, C, tol=1e-13, max_iter=1000):
    """Solve CF^2 + BF + A = 0 by simple Pontus iteration.

    F_{k+1} = -(B + C F_k)^{-1} A

    Converges linearly. Used as reference / fallback.
    See research_agenda.md Section 2.2.

    Parameters
    ----------
    A, B, C : ndarray (n, n)
        Jacobian matrices of the terminal regime.
    tol : float
        Convergence tolerance on the residual ||A + BF + CF^2||.
    max_iter : int
        Maximum iterations.

    Returns
    -------
    F : ndarray (n, n)
        Policy function matrix.
    Q : ndarray (n, n)
        Impact matrix Q = -(B + CF)^{-1}.
    converged : bool
    iters : int
    """
    n = A.shape[0]
    F = np.zeros((n, n))
    for k in range(max_iter):
        F_new = -solve(B + C @ F, A)
        residual = norm(A + B @ F_new + C @ F_new @ F_new)
        F = F_new
        if residual < tol:
            Q = -inv(B + C @ F)
            return F, Q, True, k + 1
    Q = -inv(B + C @ F)
    return F, Q, False, max_iter


def solve_terminal_doubling(A, B, C, tol=1e-13, max_iter=100):
    """Solve CF^2 + BF + A = 0 by structured doubling algorithm.

    Quadratic convergence (~15-25 iterations).
    See research_agenda.md Section 9.3 and CLAUDE.md.

    Cyclic reduction (Anderson, 1978). Starting: A0=A, B0=B, C0=C, B_hat=B.
    Iterate:
        G_k = inv(B_k)
        A_{k+1} = -A_k G_k A_k
        C_{k+1} = -C_k G_k C_k
        B_{k+1} = B_k - A_k G_k C_k - C_k G_k A_k
        B_hat  -= C_k G_k A_k
    Converge when ||A_k|| < tol.
    Recover F = -solve(B_hat, A_original).

    Parameters
    ----------
    A, B, C : ndarray (n, n)
    tol : float
    max_iter : int

    Returns
    -------
    F, Q, converged, iters
    """
    A0 = A.copy()
    Ak = A.copy()
    Bk = B.copy()
    Ck = C.copy()
    # B_hat accumulates only the forward elimination (C_k Gk A_k).
    # At convergence: B_hat u_t + A_0 u_{t-1} = 0
    # so F = -B_hat^{-1} A_0.
    B_hat = B.copy()
    for k in range(max_iter):
        Gk = inv(Bk)
        Ak_new = -Ak @ Gk @ Ak
        Ck_new = -Ck @ Gk @ Ck
        Bk_new = Bk - Ak @ Gk @ Ck - Ck @ Gk @ Ak
        B_hat = B_hat - Ck @ Gk @ Ak
        Ak = Ak_new
        Bk = Bk_new
        Ck = Ck_new
        if norm(Ak) < tol:
            F = -solve(B_hat, A0)
            Q = -inv(B + C @ F)
            return F, Q, True, k + 1
    F = -solve(B_hat, A0)
    Q = -inv(B + C @ F)
    return F, Q, False, max_iter


def solve_terminal(A, B, C, tol=1e-13, max_iter=100):
    """Solve terminal regime. Uses structured doubling.

    Returns
    -------
    F : ndarray (n, n) -- policy function
    Q : ndarray (n, n) -- impact matrix
    """
    F, Q, converged, iters = solve_terminal_doubling(A, B, C, tol, max_iter)
    if not converged:
        raise RuntimeError(
            "Structured doubling did not converge in %d iterations" % max_iter
        )
    return F, Q


# ---------------------------------------------------------------------------
# Backward recursion
# ---------------------------------------------------------------------------

def backward_recursion(regime_seq, F_terminal, matrices_M1, matrices_M2):
    """Backward recursion for time-varying policy functions.

    Given a regime sequence and the terminal regime solution, compute
    {F_t, E_t, Q_t} by iterating backward from T.

    See research_agenda.md Section 2.4.

    For t = T-1, ..., 0:
        A_t, B_t, C_t, D_t = matrices(regime_seq[t])
        M_t = B_t + C_t @ F_{t+1}
        F_t = -solve(M_t, A_t)
        E_t = -solve(M_t, C_t @ E_{t+1} + D_t)
        Q_t = -inv(M_t)

    Parameters
    ----------
    regime_seq : array of int, length T
        0 = M1 (terminal), 1 = M2 (temporary).
    F_terminal : ndarray (n, n)
        Policy function of the terminal regime.
    matrices_M1 : tuple (A1, B1, C1, D1)
    matrices_M2 : tuple (A2, B2, C2, D2)

    Returns
    -------
    F_path : ndarray (T, n, n)
    E_path : ndarray (T, n)
    Q_path : ndarray (T, n, n)
    """
    T = len(regime_seq)
    A1, B1, C1, D1 = matrices_M1
    A2, B2, C2, D2 = matrices_M2
    n = A1.shape[0]

    F_path = np.zeros((T, n, n))
    E_path = np.zeros((T, n))
    Q_path = np.zeros((T, n, n))

    # Terminal condition
    F_next = F_terminal.copy()
    E_next = np.zeros(n)

    for t in range(T - 1, -1, -1):
        if regime_seq[t] == 0:
            A_t, B_t, C_t, D_t = A1, B1, C1, D1
        else:
            A_t, B_t, C_t, D_t = A2, B2, C2, D2

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
# Forward simulation
# ---------------------------------------------------------------------------

def simulate_forward(F_path, E_path, Q_path, epsilon, u0=None):
    """Forward simulation given time-varying policy functions and shocks.

    u_t = E_t + F_t u_{t-1} + Q_t epsilon_t

    See research_agenda.md Section 2.4.

    Parameters
    ----------
    F_path : ndarray (T, n, n)
    E_path : ndarray (T, n)
    Q_path : ndarray (T, n, n)
    epsilon : ndarray (T, n_shocks)
        Structural shock sequence. Can have fewer columns than n if Q
        maps a smaller shock vector into the state space.
    u0 : ndarray (n,) or None
        Initial state. Defaults to zero (steady state).

    Returns
    -------
    u_path : ndarray (T, n)
    """
    T = F_path.shape[0]
    n = F_path.shape[1]
    u_path = np.zeros((T, n))

    if u0 is None:
        u_prev = np.zeros(n)
    else:
        u_prev = u0.copy()

    for t in range(T):
        u_t = E_path[t] + F_path[t] @ u_prev + Q_path[t] @ epsilon[t]
        u_path[t] = u_t
        u_prev = u_t

    return u_path


# ---------------------------------------------------------------------------
# Endogenous switching loop
# ---------------------------------------------------------------------------

def solve_endogenous(matrices_M1, matrices_M2, switching_fn, epsilon,
                     T_max, max_outer=50, u0=None):
    """Solve with endogenous regime switching.

    Iterates between:
      1. Backward recursion given regime sequence
      2. Forward simulation
      3. Update regime sequence from simulated path

    See research_agenda.md Section 2.5.

    Parameters
    ----------
    matrices_M1 : tuple (A1, B1, C1, D1)
    matrices_M2 : tuple (A2, B2, C2, D2)
    switching_fn : callable(u_path) -> array of int
        Takes the full simulated path (T, n) and returns regime
        sequence (T,) with 0=M1, 1=M2.
    epsilon : ndarray (T_max, n_shocks)
    T_max : int
    max_outer : int
    u0 : ndarray (n,) or None

    Returns
    -------
    u_path : ndarray (T_max, n)
    regime_seq : ndarray (T_max,) of int
    F_path, E_path, Q_path : policy function paths
    converged : bool
    outer_iters : int
    """
    A1 = matrices_M1[0]
    n = A1.shape[0]

    # Solve terminal regime (always M1)
    F_terminal, _ = solve_terminal(matrices_M1[0], matrices_M1[1],
                                   matrices_M1[2])

    # Initial guess: all M1
    regime_seq = np.zeros(T_max, dtype=int)

    for outer in range(max_outer):
        F_path, E_path, Q_path = backward_recursion(
            regime_seq, F_terminal, matrices_M1, matrices_M2
        )
        u_path = simulate_forward(F_path, E_path, Q_path, epsilon, u0)
        regime_seq_new = switching_fn(u_path)

        if np.array_equal(regime_seq_new, regime_seq):
            return u_path, regime_seq, F_path, E_path, Q_path, True, outer + 1

        regime_seq = regime_seq_new

    return u_path, regime_seq, F_path, E_path, Q_path, False, max_outer
