"""
JAX-compatible model-agnostic inversion filter with sigmoid-smoothed credibility.

This module is MODEL-AGNOSTIC: it reads matrices from the sympy parser
(via jax_lambdify) rather than hardcoding any specific model.

Any .mod file with a regimes; block can be estimated by:
  1. parse_mod() + jax_lambdify() -> JAX-traceable matrix functions
  2. compile_jax_model() -> bundles everything for the inversion filter
  3. inversion_filter() -> recovers shocks and computes log-likelihood

Key JAX patterns:
  - jnp arrays instead of np arrays
  - jax.lax.scan instead of Python for-loops
  - jax.lax.fori_loop for the terminal solver iteration
  - No in-place mutation (all pure functions)
"""

import jax
import jax.numpy as jnp
from jax import lax


# ---------------------------------------------------------------------------
# Terminal regime solver (simple iteration, fixed number of steps)
# ---------------------------------------------------------------------------

def solve_terminal_jax(A, B, C, n_iter=500):
    """Solve CF^2 + BF + A = 0 by simple Pontus iteration.

    F_{k+1} = -(B + C F_k)^{-1} A

    Uses a fixed number of iterations (no early stopping) so the
    computation graph is static and JIT-compilable.

    Parameters
    ----------
    A, B, C : jnp.ndarray (n, n)
    n_iter : int

    Returns
    -------
    F : jnp.ndarray (n, n)
    """
    n = A.shape[0]
    F = jnp.zeros((n, n))

    def body_fn(k, F):
        return -jnp.linalg.solve(B + C @ F, A)

    F = lax.fori_loop(0, n_iter, body_fn, F)
    return F


# ---------------------------------------------------------------------------
# Sigmoid credibility
# ---------------------------------------------------------------------------

def smooth_miss(pi, epsilon_bar, tau):
    """miss_t = sigmoid((|pi_t| - epsilon_bar) / tau)."""
    return jax.nn.sigmoid((jnp.abs(pi) - epsilon_bar) / tau)


def credibility_step(cred, pi_t, epsilon_bar, tau, delta_up, delta_down):
    """One step of the credibility law of motion.

    cred_t = cred_{t-1} + delta_up*(1-cred_{t-1})*(1-miss_t)
                         - delta_down*cred_{t-1}*miss_t
    """
    miss_t = smooth_miss(pi_t, epsilon_bar, tau)
    cred_new = cred + delta_up * (1.0 - cred) * (1.0 - miss_t) \
                    - delta_down * cred * miss_t
    return jnp.clip(cred_new, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Model-agnostic compiled model
# ---------------------------------------------------------------------------

def compile_jax_model(mod_string, verbose=False):
    """Parse a .mod file and return a JAX-ready model bundle.

    This is the expensive step (sympy parsing + lambdification).
    Call once at startup, then use the returned bundle for estimation.

    Parameters
    ----------
    mod_string : str
        Full .mod file content.
    verbose : bool

    Returns
    -------
    model : dict with keys:
        'build_ABC'    : callable(params_dict) -> (A, B, C) as jnp arrays
        'var_names'    : list of str
        'shock_names'  : list of str
        'param_names'  : list of str
        'param_defaults': dict
        'n_vars'       : int
        'n_shocks'     : int
        'regime_spec'  : dict or None (from regimes; block)
        'priors'       : list of dict (from priors; block)
    """
    from necredpy.utils.dynare_parser import (parse_mod, jax_lambdify,
                                               extract_regimes, extract_priors)

    parsed = parse_mod(mod_string, verbose=verbose)
    jax_funcs = jax_lambdify(parsed)

    param_names = jax_funcs['param_names']
    var_names = jax_funcs['var_names']
    shock_names = jax_funcs['shock_names']
    n_vars = len(var_names)
    n_shocks = len(shock_names)

    # Build a convenience function: params_dict -> (A, B, C)
    func_A = jax_funcs['func_A']
    func_B = jax_funcs['func_B']
    func_C = jax_funcs['func_C']

    def build_ABC(params_dict):
        """Build A, B, C matrices from a parameter dictionary.

        Uses JAX-lambdified functions from the parser. Fully differentiable.
        """
        args = [params_dict[p] for p in param_names]
        A = func_A(*args)
        B = func_B(*args)
        C = func_C(*args)
        return A, B, C

    regime_spec = extract_regimes(mod_string)
    priors = extract_priors(mod_string)

    return {
        'build_ABC': build_ABC,
        'var_names': var_names,
        'shock_names': shock_names,
        'param_names': param_names,
        'param_defaults': jax_funcs['param_defaults'],
        'n_vars': n_vars,
        'n_shocks': n_shocks,
        'regime_spec': regime_spec,
        'priors': priors,
    }


# ---------------------------------------------------------------------------
# Model-agnostic inversion filter
# ---------------------------------------------------------------------------

def inversion_filter(model, obs, params, monitor_index=None):
    """Recover structural shocks and compute log-likelihood.

    Model-agnostic: works with any .mod file compiled by compile_jax_model().

    Parameters
    ----------
    model : dict
        Output of compile_jax_model().
    obs : jnp.ndarray (T, n_vars)
        Observed state vector (all variables, deviations from SS).
        Variables not observed should be reconstructed from identities
        before calling this function.
    params : dict
        Full parameter dictionary (estimated + fixed).
        Must include: omega_H, omega_L, delta_up, delta_down,
        epsilon_bar, tau, and all shock std devs named 'sigma_<shock>'.
        Plus the omega_param name (default 'omega') which credibility
        scales.
    monitor_index : int or None
        Index of the variable to monitor for credibility (e.g., pi_agg).
        If None, uses the regime_spec['monitor'] from the .mod file.

    Returns
    -------
    log_lik : scalar
    eps_recovered : jnp.ndarray (T, n_shocks)
    cred_path : jnp.ndarray (T,)
    """
    build_ABC = model['build_ABC']
    n = model['n_vars']
    n_shocks = model['n_shocks']
    param_names = model['param_names']
    var_names = model['var_names']
    shock_names = model['shock_names']

    T = obs.shape[0]

    omega_H = params['omega_H']
    omega_L = params['omega_L']
    delta_up = params['delta_up']
    delta_down = params['delta_down']
    epsilon_bar = params['epsilon_bar']
    tau = params['tau']

    # Determine monitor variable index
    if monitor_index is None:
        regime_spec = model['regime_spec']
        if regime_spec and 'monitor' in regime_spec:
            monitor_var = regime_spec['monitor']
            monitor_index = var_names.index(monitor_var)
        else:
            raise ValueError("No monitor_index given and no regimes block.")

    # Determine which parameter is omega (credibility-scaled)
    omega_param = 'omega'
    if model['regime_spec']:
        m1_params = model['regime_spec'].get('M1_params', {})
        if m1_params:
            omega_param = list(m1_params.keys())[0]

    # ----- Step 1: Credibility path from observed monitor variable -----
    obs_monitor = obs[:, monitor_index]

    def cred_scan_fn(cred, pi_t):
        cred_new = credibility_step(cred, pi_t, epsilon_bar, tau,
                                     delta_up, delta_down)
        return cred_new, cred_new

    _, cred_path = lax.scan(cred_scan_fn, 1.0, obs_monitor)
    omega_path = omega_L + (omega_H - omega_L) * cred_path

    # ----- Step 2: Terminal solution (M1 = high credibility) -----
    params_M1 = dict(params)
    params_M1[omega_param] = omega_H
    A_term, B_term, C_term = build_ABC(params_M1)
    F_terminal = solve_terminal_jax(A_term, B_term, C_term)

    # ----- Step 3: Build per-period matrices -----
    def build_for_omega(omega_t):
        p = dict(params)
        p[omega_param] = omega_t
        return build_ABC(p)

    A_all, B_all, C_all = jax.vmap(build_for_omega)(omega_path)

    # ----- Step 4: Backward recursion (reversed scan) -----
    D_zero = jnp.zeros(n)

    def backward_step(carry, inputs):
        F_next, E_next = carry
        A_t, B_t, C_t = inputs
        M_t = B_t + C_t @ F_next
        F_t = -jnp.linalg.solve(M_t, A_t)
        E_t = -jnp.linalg.solve(M_t, C_t @ E_next + D_zero)
        Q_t = -jnp.linalg.inv(M_t)
        return (F_t, E_t), (F_t, E_t, Q_t)

    init_carry = (F_terminal, jnp.zeros(n))
    _, (F_rev, E_rev, Q_rev) = lax.scan(
        backward_step, init_carry,
        (A_all[::-1], B_all[::-1], C_all[::-1]))

    F_path = F_rev[::-1]
    E_path = E_rev[::-1]
    Q_path = Q_rev[::-1]

    # ----- Step 5: Invert for shocks -----
    def invert_step(u_prev, inputs):
        u_t, F_t, E_t, Q_t = inputs
        residual = u_t - E_t - F_t @ u_prev
        eps_t = jnp.linalg.solve(Q_t, residual)
        return u_t, eps_t

    _, eps_full = lax.scan(
        invert_step, jnp.zeros(n),
        (obs, F_path, E_path, Q_path))

    # eps_full is (T, n_vars). Extract the n_shocks actual shocks.
    eps_recovered = eps_full[:, :n_shocks]

    # ----- Step 6: Gaussian log-likelihood -----
    sigma_vec = jnp.array([params['sigma_' + s] for s in shock_names])
    log_lik = -0.5 * T * jnp.sum(jnp.log(2 * jnp.pi * sigma_vec**2)) \
              - 0.5 * jnp.sum((eps_recovered / sigma_vec)**2)

    return log_lik, eps_recovered, cred_path


# ---------------------------------------------------------------------------
# Partial-observation inversion filter (for large models)
# ---------------------------------------------------------------------------

def inversion_filter_partial(model, obs_partial, params,
                              obs_indices, shock_indices,
                              monitor_index=None):
    """Recover shocks from PARTIAL observations of the state vector.

    For models where n_vars > n_shocks (e.g., 27 vars, 8 shocks, 8 observed).
    Solves the n_shocks x n_shocks subsystem from observed variables, then
    reconstructs the full state for the next period.

    The key equation at each t:
      u_t = E_t + F_t u_{t-1} + Q_t eps_t

    Selecting observed rows:
      u_t[obs] = E_t[obs] + F_t[obs,:] u_{t-1} + Q_t[obs, shock] eps_active

    This is n_obs x n_shocks (must be square). Solve for eps_active.
    Then reconstruct full state: u_t = E_t + F_t u_{t-1} + Q_t eps_full

    Parameters
    ----------
    model : dict
        Output of compile_jax_model().
    obs_partial : jnp.ndarray (T, n_obs)
        Observed variables only (in order matching obs_indices).
    params : dict
        Full parameter dictionary.
    obs_indices : array-like of int, length n_obs
        Indices into the full state vector for observed variables.
    shock_indices : array-like of int, length n_shocks
        Indices into the full state vector for equations WITH shocks.
        These are the EQUATION indices (rows) where shocks enter.
        Must satisfy: n_obs == n_shocks == len(obs_indices) == len(shock_indices).
    monitor_index : int or None
        Index into obs_partial (NOT full state) for credibility monitor.
        E.g., if pi_agg is the 3rd observed variable, monitor_index=2.

    Returns
    -------
    log_lik : scalar
    eps_recovered : jnp.ndarray (T, n_shocks)
    cred_path : jnp.ndarray (T,)
    u_full : jnp.ndarray (T, n_vars) -- reconstructed full state
    """
    build_ABC = model['build_ABC']
    n = model['n_vars']
    n_shocks = model['n_shocks']
    var_names = model['var_names']
    shock_names = model['shock_names']

    obs_idx = jnp.array(obs_indices)
    shk_idx = jnp.array(shock_indices)
    n_obs = len(obs_indices)
    T = obs_partial.shape[0]

    omega_H = params['omega_H']
    omega_L = params['omega_L']
    delta_up = params['delta_up']
    delta_down = params['delta_down']
    epsilon_bar = params['epsilon_bar']
    tau = params['tau']

    # Monitor from partial obs
    if monitor_index is None:
        regime_spec = model['regime_spec']
        if regime_spec and 'monitor' in regime_spec:
            monitor_var = regime_spec['monitor']
            full_idx = var_names.index(monitor_var)
            # Find it in obs_indices
            monitor_index = list(obs_indices).index(full_idx)

    obs_monitor = obs_partial[:, monitor_index]

    # Omega parameter name
    omega_param = 'omega'
    if model['regime_spec']:
        m1_params = model['regime_spec'].get('M1_params', {})
        if m1_params:
            omega_param = list(m1_params.keys())[0]

    # ----- Step 1: Credibility path -----
    def cred_scan_fn(cred, pi_t):
        cred_new = credibility_step(cred, pi_t, epsilon_bar, tau,
                                     delta_up, delta_down)
        return cred_new, cred_new

    _, cred_path = lax.scan(cred_scan_fn, 1.0, obs_monitor)
    omega_path = omega_L + (omega_H - omega_L) * cred_path

    # ----- Step 2: Terminal solution -----
    params_M1 = dict(params)
    params_M1[omega_param] = omega_H
    A_term, B_term, C_term = build_ABC(params_M1)
    F_terminal = solve_terminal_jax(A_term, B_term, C_term)

    # ----- Step 3: Per-period matrices -----
    def build_for_omega(omega_t):
        p = dict(params)
        p[omega_param] = omega_t
        return build_ABC(p)

    A_all, B_all, C_all = jax.vmap(build_for_omega)(omega_path)

    # ----- Step 4: Backward recursion -----
    D_zero = jnp.zeros(n)

    def backward_step(carry, inputs):
        F_next, E_next = carry
        A_t, B_t, C_t = inputs
        M_t = B_t + C_t @ F_next
        F_t = -jnp.linalg.solve(M_t, A_t)
        E_t = -jnp.linalg.solve(M_t, C_t @ E_next + D_zero)
        Q_t = -jnp.linalg.inv(M_t)
        return (F_t, E_t), (F_t, E_t, Q_t)

    init_carry = (F_terminal, jnp.zeros(n))
    _, (F_rev, E_rev, Q_rev) = lax.scan(
        backward_step, init_carry,
        (A_all[::-1], B_all[::-1], C_all[::-1]))

    F_path = F_rev[::-1]
    E_path = E_rev[::-1]
    Q_path = Q_rev[::-1]

    # ----- Step 5: Partial-observation inversion -----
    # At each t:
    #   obs_t = E_t[obs_idx] + F_t[obs_idx,:] @ u_{t-1} + Q_t[obs_idx, shk_idx] @ eps_active
    #   => eps_active = Q_sub^{-1} (obs_t - E_t[obs_idx] - F_t[obs_idx,:] @ u_{t-1})
    #   => u_t = E_t + F_t @ u_{t-1} + Q_t @ eps_full
    #      where eps_full has eps_active at shk_idx positions, 0 elsewhere

    def partial_invert_step(u_prev, inputs):
        obs_t, F_t, E_t, Q_t = inputs

        # Subsystem for observed variables
        E_obs = E_t[obs_idx]
        F_obs = F_t[obs_idx, :]
        Q_sub = Q_t[obs_idx][:, shk_idx]  # (n_obs, n_shocks)

        residual_obs = obs_t - E_obs - F_obs @ u_prev
        eps_active = jnp.linalg.solve(Q_sub, residual_obs)

        # Reconstruct full shock vector (zeros except at shock positions)
        eps_full = jnp.zeros(n).at[shk_idx].set(eps_active)

        # Reconstruct full state
        u_t = E_t + F_t @ u_prev + Q_t @ eps_full

        return u_t, (eps_active, u_t)

    _, (eps_all, u_full) = lax.scan(
        partial_invert_step, jnp.zeros(n),
        (obs_partial, F_path, E_path, Q_path))

    # ----- Step 6: Log-likelihood -----
    sigma_vec = jnp.array([params['sigma_' + s] for s in shock_names])
    log_lik = -0.5 * T * jnp.sum(jnp.log(2 * jnp.pi * sigma_vec**2)) \
              - 0.5 * jnp.sum((eps_all / sigma_vec)**2)

    return log_lik, eps_all, cred_path, u_full


# ---------------------------------------------------------------------------
# Legacy 3-equation convenience wrapper (backward-compatible)
# ---------------------------------------------------------------------------

def build_matrices_jax(beta, sigma, kappa, rho_i, phi_pi, phi_y, omega):
    """Build A, B, C for the 3-equation NK model (legacy, hardcoded).

    DEPRECATED: Use compile_jax_model() + build_ABC() instead.
    Kept for backward compatibility with existing scripts.
    """
    A = jnp.array([
        [0.0,  0.0,     0.0,  0.0],
        [0.0,  0.0,     0.0,  0.0],
        [0.0,  0.0,  -rho_i,  0.0],
        [0.0, -1.0,     0.0,  0.0],
    ])
    B = jnp.array([
        [1.0,                       0.0,  sigma,              0.0],
        [-kappa,                    1.0,    0.0, -(1.0 - omega)],
        [-(1-rho_i)*phi_y, -(1-rho_i)*phi_pi, 1.0,        0.0],
        [0.0,                       0.0,    0.0,            1.0],
    ])
    C = jnp.array([
        [-1.0,         -sigma,  0.0,  0.0],
        [ 0.0,  -omega * beta,  0.0,  0.0],
        [ 0.0,            0.0,  0.0,  0.0],
        [ 0.0,            0.0,  0.0,  0.0],
    ])
    return A, B, C


def inversion_filter_jax(obs_y, obs_pi, obs_i, params):
    """Legacy 3-equation inversion filter (backward-compatible).

    DEPRECATED: Use compile_jax_model() + inversion_filter() instead.
    Kept for backward compatibility with existing scripts.
    """
    T = obs_y.shape[0]
    n = 4

    beta = params['beta']
    sigma = params['sigma']
    kappa = params['kappa']
    rho_i = params['rho_i']
    phi_pi = params['phi_pi']
    phi_y = params['phi_y']
    omega_H = params['omega_H']
    omega_L = params['omega_L']
    delta_up = params['delta_up']
    delta_down = params['delta_down']
    epsilon_bar = params['epsilon_bar']
    tau = params['tau']
    sigma_d = params['sigma_d']
    sigma_s = params['sigma_s']
    sigma_m = params['sigma_m']

    def cred_scan_fn(cred, pi_t):
        cred_new = credibility_step(cred, pi_t, epsilon_bar, tau,
                                     delta_up, delta_down)
        return cred_new, cred_new

    _, cred_path = lax.scan(cred_scan_fn, 1.0, obs_pi)
    omega_path = omega_L + (omega_H - omega_L) * cred_path

    A_term, B_term, C_term = build_matrices_jax(
        beta, sigma, kappa, rho_i, phi_pi, phi_y, omega_H)
    F_terminal = solve_terminal_jax(A_term, B_term, C_term)

    def build_for_omega(omega_t):
        return build_matrices_jax(beta, sigma, kappa, rho_i, phi_pi, phi_y,
                                   omega_t)

    A_all, B_all, C_all = jax.vmap(build_for_omega)(omega_path)

    D_zero = jnp.zeros(n)

    def backward_step(carry, inputs):
        F_next, E_next = carry
        A_t, B_t, C_t = inputs
        M_t = B_t + C_t @ F_next
        F_t = -jnp.linalg.solve(M_t, A_t)
        E_t = -jnp.linalg.solve(M_t, C_t @ E_next + D_zero)
        Q_t = -jnp.linalg.inv(M_t)
        return (F_t, E_t), (F_t, E_t, Q_t)

    init_carry = (F_terminal, jnp.zeros(n))
    _, (F_rev, E_rev, Q_rev) = lax.scan(
        backward_step, init_carry,
        (A_all[::-1], B_all[::-1], C_all[::-1]))

    F_path = F_rev[::-1]
    E_path = E_rev[::-1]
    Q_path = Q_rev[::-1]

    pi_lag = jnp.concatenate([jnp.array([0.0]), obs_pi[:-1]])
    u_obs = jnp.stack([obs_y, obs_pi, obs_i, pi_lag], axis=1)

    def invert_step(u_prev, inputs):
        u_t, F_t, E_t, Q_t = inputs
        residual = u_t - E_t - F_t @ u_prev
        eps_t = jnp.linalg.solve(Q_t, residual)
        return u_t, eps_t

    _, eps_full = lax.scan(
        invert_step, jnp.zeros(n),
        (u_obs, F_path, E_path, Q_path))

    eps_recovered = eps_full[:, :3]

    sigma_vec = jnp.array([sigma_d, sigma_s, sigma_m])
    log_lik = -0.5 * T * jnp.sum(jnp.log(2 * jnp.pi * sigma_vec**2)) \
              - 0.5 * jnp.sum((eps_recovered / sigma_vec)**2)

    return log_lik, eps_recovered, cred_path
