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

def solve_terminal_jax(A, B, C, n_iter=25, tol=1e-12):
    """Solve CF^2 + BF + A = 0 by structured doubling (cyclic reduction).

    Quadratic convergence: each iteration roughly doubles the number of
    correct bits. ~15 iterations is enough to reach machine precision in
    float64 for well-conditioned linear DSGEs. Compare to simple Pontus
    iteration F_{k+1} = -(B + CF_k)^{-1} A which converges linearly and
    needs many more steps when the spectral radius of F is close to 1.

    Algorithm (Anderson 1978; see also numpy reference at
    necredpy/pontus.py:solve_terminal_doubling):

        A0 = A;  Ak = A;  Bk = B;  Ck = C;  B_hat = B
        for k = 1, 2, ...:
            Gk      = inv(Bk)
            Ak_new  = -Ak Gk Ak
            Ck_new  = -Ck Gk Ck
            Bk_new  = Bk - Ak Gk Ck - Ck Gk Ak
            B_hat  -= Ck Gk Ak
            stop when ||Ak_new|| < tol
        F = -solve(B_hat, A0)

    Implementation details:
      - Uses lax.scan with a frozen-update mask, so the iteration is
        differentiable through reverse-mode autodiff (lax.while_loop is
        not). Once ||Ak|| < tol, all updates are masked off.
      - Solves Bk Gk = I via lu_factor + lu_solve once, then reuses the
        factorization (3 solves with 1 factor).
      - The returned F is exact when iteration converges. When the
        model is non-stationary the iteration does NOT converge --
        ||Ak|| stays large, the residual A + BF + CF^2 also stays
        large, and the caller can detect non-stationarity by checking
        the QME residual rather than the eigenvalues of the (junk)
        returned F.

    Parameters
    ----------
    A, B, C : jnp.ndarray (n, n)
    n_iter : int
        Maximum doubling iterations (default 25, generous upper bound).
    tol : float
        Convergence tolerance on max|Ak|. Once met, all updates freeze.

    Returns
    -------
    F : jnp.ndarray (n, n)
    """
    n = A.shape[0]
    eye = jnp.eye(n)
    A0 = A

    def step(carry, _):
        Ak, Bk, Ck, B_hat, frozen = carry
        # Gk = inv(Bk) via LU (one factor, three reuses)
        lu, piv = jax.scipy.linalg.lu_factor(Bk)
        Gk = jax.scipy.linalg.lu_solve((lu, piv), eye)

        Ak_new = -Ak @ Gk @ Ak
        Ck_new = -Ck @ Gk @ Ck
        Bk_new = Bk - Ak @ Gk @ Ck - Ck @ Gk @ Ak
        B_hat_new = B_hat - Ck @ Gk @ Ak

        # Convergence: max|Ak_new| below tol
        new_frozen = frozen | (jnp.max(jnp.abs(Ak_new)) < tol)

        # Freeze updates once converged
        Ak_out = jnp.where(new_frozen, Ak, Ak_new)
        Bk_out = jnp.where(new_frozen, Bk, Bk_new)
        Ck_out = jnp.where(new_frozen, Ck, Ck_new)
        Bhat_out = jnp.where(new_frozen, B_hat, B_hat_new)
        return (Ak_out, Bk_out, Ck_out, Bhat_out, new_frozen), None

    init = (A, B, C, B, jnp.array(False))
    (Ak_final, Bk_final, Ck_final, B_hat_final, _), _ = lax.scan(
        step, init, None, length=n_iter)

    # Recover F from the converged forward elimination:
    #   B_hat u_t + A0 u_{t-1} = 0  =>  F = -B_hat^{-1} A0
    F = -jnp.linalg.solve(B_hat_final, A0)
    return F


def _build_cred_scan_fn(model, params):
    """Build the credibility scan function for lax.scan.

    Returns (cred_scan_fn, init_carry) where:
        cred_scan_fn(carry, monitor_t) -> (new_carry, (cred_t, omega_t))

    The scan output is a (cred, omega) tuple for ALL grammar variants,
    so callers can always do:
        _, (cred_path, omega_path) = lax.scan(fn, init, obs_monitor)

    Uses the new credibility grammar (credibility; ... end; block with
    var/input/output declarations). Raises ValueError if no credibility
    block is found.
    """
    cred_new = model.get('credibility_new')

    # ----- Path 1: New grammar (credibility; ... end;) -----
    if cred_new is not None:
        credibility_fn = cred_new['fn']
        state_vars = cred_new['state_vars']
        input_vars = cred_new['input_vars']
        output_vars = cred_new['output_vars']
        used_params = cred_new['used_params']
        lagged_input_vars = cred_new.get('lagged_input_vars', [])

        # Pre-extract credibility parameters (closed over, not re-read)
        cred_params = {p: params[p] for p in used_params}

        # Initial state: full credibility + lagged inputs at SS (0.0)
        init_state = {v: jnp.array(1.0) if v == 'cred_state'
                      else jnp.array(0.0) for v in state_vars}
        for v in lagged_input_vars:
            init_state[v] = jnp.array(0.0)

        def cred_scan_fn(carry, monitor_t):
            prev_state = carry
            inputs = {input_vars[0]: monitor_t}
            outputs, new_state = credibility_fn(inputs, prev_state,
                                                cred_params)
            omega_t = outputs[output_vars[0]]
            cred_t = new_state.get('cred_state', jnp.array(1.0))
            # new_state already includes current inputs for lagged-input
            # support (propagated by credibility_fn)
            return new_state, (cred_t, omega_t)

        return cred_scan_fn, init_state

    raise ValueError(
        "No credibility block found. The .mod file must have a "
        "credibility; ... end; block with var/input/output declarations."
    )


# ---------------------------------------------------------------------------
# Model-agnostic compiled model
# ---------------------------------------------------------------------------

def compile_jax_model(mod_string, verbose=False):
    """Parse a .mod file and return a JAX-ready model bundle.

    This is the expensive step (sympy parsing + lambdification).
    Call once at startup, then use the returned bundle for estimation.

    Uses the new credibility grammar (credibility; ... end; block with
    var/input/output declarations). Credibility outputs (e.g. omega_pc)
    become time-varying coefficients, not estimated parameters.

    Parameters
    ----------
    mod_string : str
        Full .mod file content.
    verbose : bool

    Returns
    -------
    model : dict with keys:
        'build_ABC'         : callable(params_dict, *coeff_values) -> (A, B, C)
        'var_names'         : list of str
        'shock_names'       : list of str
        'param_names'       : list of str
        'param_defaults'    : dict
        'n_vars'            : int
        'n_shocks'          : int
        'coefficient_names' : list of str (e.g. ['omega_pc'])
        'regime_spec'       : dict or None (from regimes; block)
        'priors'            : list of dict (from priors; block)
        'credibility_new'   : dict or None (compiled credibility block)
    """
    from necredpy.utils.dynare_parser import (parse_credibility_mod,
                                               jax_lambdify,
                                               extract_priors)

    # Unified parsing: handles both new and legacy grammar transparently.
    # For new grammar with model(pwl/nn), credibility outputs (e.g.
    # omega_pc) are passed as coefficient_names to parse_mod, keeping
    # them out of param_names so NUTS never sees them.
    pcm = parse_credibility_mod(mod_string, verbose=verbose)
    model_parsed = pcm['model']
    cred_compiled = pcm['credibility']  # New grammar compiled, or None

    jax_funcs = jax_lambdify(model_parsed)

    param_names = jax_funcs['param_names']
    coeff_names = jax_funcs.get('coefficient_names', [])
    var_names = jax_funcs['var_names']
    shock_names = jax_funcs['shock_names']
    n_vars = len(var_names)
    n_shocks = len(shock_names)

    # JAX-lambdified matrix functions.  When coefficient_names is
    # non-empty, these take (*param_values, *coeff_values).
    func_A = jax_funcs['func_A']
    func_B = jax_funcs['func_B']
    func_C = jax_funcs['func_C']
    func_D = jax_funcs.get('func_D')
    func_D_const = jax_funcs.get('func_D_const')

    def build_ABC(params_dict, *coeff_values):
        """Build A, B, C matrices from parameters and optional coefficients.

        For legacy models (no coefficient_names): call as build_ABC(params).
        For new-grammar models: call as build_ABC(params, omega_pc_value).
        Fully differentiable through JAX.
        """
        args = [params_dict[p] for p in param_names] + list(coeff_values)
        A = func_A(*args)
        B = func_B(*args)
        C = func_C(*args)
        return A, B, C

    def build_D(params_dict, *coeff_values):
        """Shock selection matrix D of shape (n_eq, n_shocks)."""
        args = [params_dict[p] for p in param_names] + list(coeff_values)
        return func_D(*args)

    def build_D_const(params_dict, *coeff_values):
        """Constant intercept vector of shape (n_eq,)."""
        if func_D_const is None:
            return jnp.zeros(n_vars)
        args = [params_dict[p] for p in param_names] + list(coeff_values)
        return func_D_const(*args)

    regime_spec = None  # Legacy regimes block removed
    priors = extract_priors(mod_string)

    return {
        'build_ABC': build_ABC,
        'build_D': build_D,
        'build_D_const': build_D_const,
        'var_names': var_names,
        'shock_names': shock_names,
        'param_names': param_names,
        'param_defaults': jax_funcs['param_defaults'],
        'n_vars': n_vars,
        'n_shocks': n_shocks,
        'coefficient_names': coeff_names,
        'regime_spec': regime_spec,
        'priors': priors,
        'credibility_new': cred_compiled,
        'aux_resolution': jax_funcs.get('aux_resolution', {}),
        'monitor_resolution': jax_funcs.get('monitor_resolution', {}),
        'model_options': jax_funcs.get('model_options', {}),
    }


# ---------------------------------------------------------------------------
# Model-agnostic inversion filter
# ---------------------------------------------------------------------------

def inversion_filter(model, obs, params, monitor_index=None):
    """Recover structural shocks and compute log-likelihood.

    Model-agnostic: works with any .mod file compiled by compile_jax_model().
    Supports both the legacy credibility grammar (omega as parameter) and
    the new grammar (omega_pc as coefficient via coefficient_names).

    Parameters
    ----------
    model : dict
        Output of compile_jax_model().
    obs : jnp.ndarray (T, n_vars)
        Observed state vector (all variables, deviations from SS).
    params : dict
        Full parameter dictionary (estimated + fixed).
        For legacy: must include omega_H, omega_L, delta_up, etc.
        For new grammar: must include all credibility parameters
        declared in the .mod file (e.g. omega_high, omega_low, etc.).
    monitor_index : int or None
        Index of the variable to monitor for credibility.
        If None, auto-detected from the credibility block.

    Returns
    -------
    log_lik : scalar
    eps_recovered : jnp.ndarray (T, n_shocks)
    cred_path : jnp.ndarray (T,)
    """
    build_ABC = model['build_ABC']
    n = model['n_vars']
    n_shocks = model['n_shocks']
    var_names = model['var_names']
    shock_names = model['shock_names']
    coeff_names = model.get('coefficient_names', [])
    cred_new = model.get('credibility_new')

    T = obs.shape[0]

    # Determine monitor variable index from credibility block
    if monitor_index is None:
        if cred_new is not None:
            monitor_var = cred_new['input_vars'][0]
            monitor_index = var_names.index(monitor_var)
        else:
            raise ValueError(
                "No monitor_index given and no credibility block.")

    # ----- Step 1: Credibility path from observed monitor variable -----
    obs_monitor = obs[:, monitor_index]

    cred_scan_fn, init_carry = _build_cred_scan_fn(model, params)
    _, (cred_path, omega_path) = lax.scan(
        cred_scan_fn, init_carry, obs_monitor)

    # ----- Step 2: Terminal solution (full credibility) -----
    if coeff_names:
        # New grammar: evaluate credibility at SS (full cred) to get
        # omega_terminal, then pass it as a coefficient to build_ABC.
        ss_inputs = {v: jnp.array(0.0) for v in cred_new['input_vars']}
        ss_state = {v: jnp.array(1.0) if v == 'cred_state'
                    else jnp.array(0.0) for v in cred_new['state_vars']}
        # Lagged inputs must also be in prev_state (the credibility fn
        # reads them from prev_state for the (-1) lag).
        for v in cred_new.get('lagged_input_vars', []):
            ss_state[v] = jnp.array(0.0)
        cred_params = {p: params[p] for p in cred_new['used_params']}
        out_ss, _ = cred_new['fn'](ss_inputs, ss_state, cred_params)
        omega_terminal = out_ss[cred_new['output_vars'][0]]
        A_term, B_term, C_term = build_ABC(params, omega_terminal)

    F_terminal = solve_terminal_jax(A_term, B_term, C_term)

    # Stationarity check: (a) SDA converged, (b) max|eig(F)| < 1.
    qme_residual = (A_term + B_term @ F_terminal
                    + C_term @ F_terminal @ F_terminal)
    res_norm = jnp.max(jnp.abs(qme_residual))
    res_tol = 1e-6 if F_terminal.dtype == jnp.float64 else 1e-3
    converged = res_norm < res_tol

    eigs_M1 = jnp.linalg.eigvals(F_terminal)
    rho_M1 = jnp.max(jnp.abs(eigs_M1))
    spec_ok = rho_M1 < 1.0

    stable_M1 = converged & spec_ok

    # ----- Step 3: Build per-period matrices -----
    def build_for_omega(omega_t):
        return build_ABC(params, omega_t)

    A_all, B_all, C_all = jax.vmap(build_for_omega)(omega_path)

    # ----- Step 4: Backward recursion (reversed scan) -----
    D_zero = jnp.zeros(n)

    def backward_step(carry, inputs):
        F_next, E_next = carry
        A_t, B_t, C_t = inputs
        M_t = B_t + C_t @ F_next
        # Factor M_t once, reuse for F, E, Q (3 solves with 1 factorization).
        # ~3x faster than three independent jnp.linalg.solve calls.
        n_local = M_t.shape[0]
        lu, piv = jax.scipy.linalg.lu_factor(M_t)
        F_t = -jax.scipy.linalg.lu_solve((lu, piv), A_t)
        E_t = -jax.scipy.linalg.lu_solve((lu, piv), C_t @ E_next + D_zero)
        Q_t = -jax.scipy.linalg.lu_solve((lu, piv), jnp.eye(n_local))
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

    # Reject non-stationary M1 and any non-finite log-likelihood (catches
    # singular per-period M_t in the recursion). NumPyro treats -inf as
    # a rejected proposal and skips this point cleanly.
    log_lik = jnp.where(stable_M1 & jnp.isfinite(log_lik),
                        log_lik, -jnp.inf)

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
    coeff_names = model.get('coefficient_names', [])
    cred_new = model.get('credibility_new')

    # Convert to Python lists for static operations (monitor detection),
    # then to jnp arrays for the traced scan/inversion.
    obs_indices_py = [int(x) for x in obs_indices]
    shk_indices_py = [int(x) for x in shock_indices]
    obs_idx = jnp.array(obs_indices_py)
    shk_idx = jnp.array(shk_indices_py)
    n_obs = len(obs_indices_py)
    T = obs_partial.shape[0]

    # Determine monitor variable and compute its observed path.
    # Two cases:
    #   Case 1: monitor IS one of the observed variables -> read directly.
    #   Case 2: monitor is a PURE IDENTITY of observed variables (possibly
    #           at lags) -> use monitor_resolution map to reconstruct.
    if monitor_index is None:
        # Identify the monitor variable name from the credibility block
        if cred_new is not None:
            monitor_var = cred_new['input_vars'][0]
        else:
            raise ValueError(
                "No monitor_index given and no credibility block.")

        full_idx = var_names.index(monitor_var)
        if full_idx in obs_indices_py:
            monitor_index = obs_indices_py.index(full_idx)
            obs_monitor = obs_partial[:, monitor_index]
        else:
            monitor_resolution = model.get('monitor_resolution', {})
            if monitor_var not in monitor_resolution:
                raise ValueError(
                    f"Credibility monitor '{monitor_var}' is neither "
                    f"observed nor a resolved identity of observed "
                    f"variables. Define '{monitor_var}' via a pure "
                    f"linear identity equation (no shock, no leads) "
                    f"whose RHS contains only observed variables "
                    f"(possibly at lags)."
                )
            terms = monitor_resolution[monitor_var]
            obs_monitor = jnp.zeros(T)
            for src_var, lag, coeff in terms:
                src_full = var_names.index(src_var)
                if src_full not in obs_indices_py:
                    raise ValueError(
                        f"Monitor '{monitor_var}' resolves to "
                        f"'{src_var}' which is not observed."
                    )
                src_col = obs_indices_py.index(src_full)
                series = obs_partial[:, src_col]
                if lag == 0:
                    shifted = series
                elif lag < 0:
                    k = -lag
                    pad = jnp.full((k,), series[0])
                    shifted = jnp.concatenate([pad, series[:-k]])
                else:
                    k = lag
                    pad = jnp.full((k,), series[-1])
                    shifted = jnp.concatenate([series[k:], pad])
                obs_monitor = obs_monitor + coeff * shifted
    else:
        obs_monitor = obs_partial[:, monitor_index]

    # ----- Step 1: Credibility path -----
    cred_scan_fn, init_carry = _build_cred_scan_fn(model, params)
    _, (cred_path, omega_path) = lax.scan(
        cred_scan_fn, init_carry, obs_monitor)

    # ----- Step 2: Terminal solution (full credibility) -----
    if coeff_names:
        # New grammar: evaluate credibility at SS to get omega_terminal
        ss_inputs = {v: jnp.array(0.0) for v in cred_new['input_vars']}
        ss_state = {v: jnp.array(1.0) if v == 'cred_state'
                    else jnp.array(0.0) for v in cred_new['state_vars']}
        # Lagged inputs must also be in prev_state (the credibility fn
        # reads them from prev_state for the (-1) lag).
        for v in cred_new.get('lagged_input_vars', []):
            ss_state[v] = jnp.array(0.0)
        cred_params = {p: params[p] for p in cred_new['used_params']}
        out_ss, _ = cred_new['fn'](ss_inputs, ss_state, cred_params)
        omega_terminal = out_ss[cred_new['output_vars'][0]]
        A_term, B_term, C_term = build_ABC(params, omega_terminal)

    F_terminal = solve_terminal_jax(A_term, B_term, C_term)

    # Stationarity check: (a) SDA converged (residual ~ 0),
    # (b) max|eig(F)| < 1.
    qme_residual = (A_term + B_term @ F_terminal
                    + C_term @ F_terminal @ F_terminal)
    res_norm = jnp.max(jnp.abs(qme_residual))
    res_tol = 1e-6 if F_terminal.dtype == jnp.float64 else 1e-3
    converged = res_norm < res_tol

    eigs_M1 = jnp.linalg.eigvals(F_terminal)
    rho_M1 = jnp.max(jnp.abs(eigs_M1))
    spec_ok = rho_M1 < 1.0

    stable_M1 = converged & spec_ok

    # ----- Step 3: Per-period matrices -----
    def build_for_omega(omega_t):
        return build_ABC(params, omega_t)

    A_all, B_all, C_all = jax.vmap(build_for_omega)(omega_path)

    # ----- Step 4: Backward recursion -----
    D_zero = jnp.zeros(n)

    def backward_step(carry, inputs):
        F_next, E_next = carry
        A_t, B_t, C_t = inputs
        M_t = B_t + C_t @ F_next
        # Factor M_t once, reuse for F, E, Q (3 solves with 1 factorization).
        # ~3x faster than three independent jnp.linalg.solve calls.
        n_local = M_t.shape[0]
        lu, piv = jax.scipy.linalg.lu_factor(M_t)
        F_t = -jax.scipy.linalg.lu_solve((lu, piv), A_t)
        E_t = -jax.scipy.linalg.lu_solve((lu, piv), C_t @ E_next + D_zero)
        Q_t = -jax.scipy.linalg.lu_solve((lu, piv), jnp.eye(n_local))
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

    # Reject non-stationary M1 (terminal-regime spectral radius >= 1) and
    # any non-finite log-likelihood (catches singular per-period M_t in
    # the backward recursion, regardless of regime). NumPyro treats -inf
    # as a rejected proposal, so the sampler skips this point cleanly.
    log_lik = jnp.where(stable_M1 & jnp.isfinite(log_lik),
                        log_lik, -jnp.inf)

    return log_lik, eps_all, cred_path, u_full, F_path, E_path, Q_path
