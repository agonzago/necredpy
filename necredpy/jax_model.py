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


# ---------------------------------------------------------------------------
# Sigmoid credibility
# ---------------------------------------------------------------------------

def smooth_miss(pi, epsilon_bar, tau):
    """miss_t = sigmoid((pi_t^2 - epsilon_bar^2) / tau)."""
    return jax.nn.sigmoid((pi**2 - epsilon_bar**2) / tau)


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
# Pluggable credibility: JAX-compiled signal and accumulation functions
# ---------------------------------------------------------------------------

def compile_credibility_fn_jax(cred_spec, param_names):
    """Compile credibility signal/accumulation expressions for JAX.

    Same as compile_credibility_fn() in dynare_parser.py but targets JAX
    so the functions are differentiable and JIT-compilable.

    Parameters
    ----------
    cred_spec : dict
        Output of extract_credibility().
    param_names : list of str
        All parameter names from the .mod file.

    Returns
    -------
    compiled : dict with keys:
        'signal_fn_jax'       : callable or None
        'signal_params'       : list of str (param names used in signal)
        'accumulation_fn_jax' : callable or None
        'accumulation_params' : list of str (param names used in accumulation)
        'monitor', 'threshold', 'cred_init', 'signal_lag' : from cred_spec
    """
    import sympy

    compiled = {
        'monitor': cred_spec['monitor'],
        'threshold': cred_spec['threshold'],
        'cred_init': cred_spec['cred_init'],
        'signal_lag': cred_spec.get('signal_lag', 0),
    }

    jax_mod = [{'ImmutableDenseMatrix': jnp.array}, 'jax']

    # ---- Signal function ----
    sig_expr_str = cred_spec.get('signal_expr')

    if sig_expr_str is None:
        compiled['signal_fn_jax'] = None
        compiled['signal_params'] = []
    else:
        expr_str = sig_expr_str.replace('pi(-1)', 'pi_lag')
        expr_str = expr_str.replace('^', '**')

        sym_pi = sympy.Symbol('pi')
        sym_pi_lag = sympy.Symbol('pi_lag')
        sym_pi_star = sympy.Symbol('pi_star')

        param_syms = {p: sympy.Symbol(p) for p in param_names}
        local_dict = {'pi': sym_pi, 'pi_lag': sym_pi_lag,
                      'pi_star': sym_pi_star}
        local_dict.update(param_syms)
        local_dict['exp'] = sympy.exp
        local_dict['log'] = sympy.log
        local_dict['abs'] = sympy.Abs
        local_dict['sqrt'] = sympy.sqrt

        sym_expr = sympy.sympify(expr_str, locals=local_dict)

        all_sym_names = {str(s) for s in sym_expr.free_symbols}
        used_params = [p for p in param_names if p in all_sym_names]
        compiled['signal_params'] = used_params

        arg_list = [sym_pi, sym_pi_lag, sym_pi_star]
        arg_list += [param_syms[p] for p in used_params]

        compiled['signal_fn_jax'] = sympy.lambdify(
            arg_list, sym_expr, modules=jax_mod)

    # ---- Accumulation function ----
    acc_expr_str = cred_spec.get('accumulation_expr')

    if acc_expr_str is None:
        compiled['accumulation_fn_jax'] = None
        compiled['accumulation_params'] = []
    else:
        expr_str = acc_expr_str.replace('^', '**')

        sym_cred = sympy.Symbol('cred')
        sym_s = sympy.Symbol('s')
        sym_miss = sympy.Symbol('miss')

        param_syms = {p: sympy.Symbol(p) for p in param_names}
        local_dict = {'cred': sym_cred, 's': sym_s, 'miss': sym_miss}
        local_dict.update(param_syms)
        local_dict['exp'] = sympy.exp
        local_dict['log'] = sympy.log
        local_dict['abs'] = sympy.Abs
        local_dict['sqrt'] = sympy.sqrt
        local_dict['max'] = sympy.Max
        local_dict['min'] = sympy.Min

        sym_expr = sympy.sympify(expr_str, locals=local_dict)

        all_sym_names = {str(s) for s in sym_expr.free_symbols}
        used_params = [p for p in param_names if p in all_sym_names]
        compiled['accumulation_params'] = used_params

        arg_list = [sym_cred, sym_s, sym_miss]
        arg_list += [param_syms[p] for p in used_params]

        compiled['accumulation_fn_jax'] = sympy.lambdify(
            arg_list, sym_expr, modules=jax_mod)

    return compiled


def _build_cred_scan_fn(model, params):
    """Build the credibility scan function for lax.scan.

    Returns (cred_scan_fn, init_carry) where:
        cred_scan_fn(carry, pi_t) -> (new_carry, cred_value)
        carry = (cred, pi_lag)

    Uses pluggable credibility if available, otherwise falls back to
    the hardcoded Isard specification.
    """
    cred_jax = model.get('credibility_jax')

    if cred_jax is not None and (
            cred_jax['signal_fn_jax'] is not None or
            cred_jax['accumulation_fn_jax'] is not None):
        # Pluggable credibility
        signal_fn = cred_jax['signal_fn_jax']
        acc_fn = cred_jax['accumulation_fn_jax']
        signal_lag = cred_jax['signal_lag']
        signal_params = cred_jax['signal_params']
        acc_params = cred_jax['accumulation_params']
        cred_init = cred_jax['cred_init']

        # Pre-extract parameter values for the signal/accumulation
        sig_param_vals = [params[p] for p in signal_params]
        acc_param_vals = [params[p] for p in acc_params]

        epsilon_bar = params.get('epsilon_bar', 2.0)
        tau = params.get('tau', 0.2)

        def cred_scan_fn(carry, pi_t):
            cred, pi_lag = carry

            # Smooth miss indicator (squared deviation: everywhere differentiable)
            miss = jax.nn.sigmoid((pi_t**2 - epsilon_bar**2) / tau)

            # Signal
            if signal_fn is not None:
                pi_input = pi_lag if signal_lag == 1 else pi_t
                sig = signal_fn(pi_t, pi_input, 0.0, *sig_param_vals)
            else:
                sig = 0.0

            # Accumulation
            if acc_fn is not None:
                cred_new = acc_fn(cred, sig, miss, *acc_param_vals)
            else:
                # Isard fallback
                delta_up = params['delta_up']
                delta_down = params['delta_down']
                cred_new = (cred + delta_up * (1.0 - cred) * (1.0 - miss)
                            - delta_down * cred * miss)

            cred_new = jnp.clip(cred_new, 0.0, 1.0)
            return (cred_new, pi_t), cred_new

        init_carry = (cred_init, 0.0)
        return cred_scan_fn, init_carry

    else:
        # Hardcoded Isard specification
        delta_up = params['delta_up']
        delta_down = params['delta_down']
        epsilon_bar = params['epsilon_bar']
        tau = params['tau']

        def cred_scan_fn(carry, pi_t):
            cred, pi_lag = carry
            cred_new = credibility_step(cred, pi_t, epsilon_bar, tau,
                                         delta_up, delta_down)
            return (cred_new, pi_t), cred_new

        init_carry = (1.0, 0.0)
        return cred_scan_fn, init_carry


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
        'credibility_jax' : dict or None (JAX-compiled credibility functions)
    """
    from necredpy.utils.dynare_parser import (parse_mod, jax_lambdify,
                                               extract_regimes, extract_priors,
                                               extract_credibility)

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

    # Compile pluggable credibility for JAX if present
    cred_spec = extract_credibility(mod_string)
    credibility_jax = None
    if cred_spec is not None:
        credibility_jax = compile_credibility_fn_jax(cred_spec, param_names)

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
        'credibility_jax': credibility_jax,
        'aux_resolution': jax_funcs.get('aux_resolution', {}),
        'monitor_resolution': jax_funcs.get('monitor_resolution', {}),
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

    # Determine monitor variable index
    if monitor_index is None:
        cred_jax = model.get('credibility_jax')
        if cred_jax and 'monitor' in cred_jax:
            monitor_index = var_names.index(cred_jax['monitor'])
        elif model['regime_spec'] and 'monitor' in model['regime_spec']:
            monitor_index = var_names.index(model['regime_spec']['monitor'])
        else:
            raise ValueError("No monitor_index given and no regimes/credibility block.")

    # Determine which parameter is omega (credibility-scaled)
    omega_param = 'omega'
    if model['regime_spec']:
        m1_params = model['regime_spec'].get('M1_params', {})
        if m1_params:
            omega_param = list(m1_params.keys())[0]

    # ----- Step 1: Credibility path from observed monitor variable -----
    obs_monitor = obs[:, monitor_index]

    cred_scan_fn, init_carry = _build_cred_scan_fn(model, params)
    _, cred_path = lax.scan(cred_scan_fn, init_carry, obs_monitor)
    omega_path = omega_L + (omega_H - omega_L) * cred_path

    # ----- Step 2: Terminal solution (M1 = high credibility) -----
    params_M1 = dict(params)
    params_M1[omega_param] = omega_H
    A_term, B_term, C_term = build_ABC(params_M1)
    F_terminal = solve_terminal_jax(A_term, B_term, C_term)

    # Stationarity check at M1: see the matching block in
    # inversion_filter_partial for the full rationale. Both conditions
    # must hold: (a) SDA converged (residual ~ 0), and
    # (b) max|eig(F)| < 1.
    qme_residual = A_term + B_term @ F_terminal + C_term @ F_terminal @ F_terminal
    res_norm = jnp.max(jnp.abs(qme_residual))
    res_tol = 1e-6 if F_terminal.dtype == jnp.float64 else 1e-3
    converged = res_norm < res_tol

    eigs_M1 = jnp.linalg.eigvals(F_terminal)
    rho_M1 = jnp.max(jnp.abs(eigs_M1))
    spec_ok = rho_M1 < 1.0

    stable_M1 = converged & spec_ok

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

    obs_idx = jnp.array(obs_indices)
    shk_idx = jnp.array(shock_indices)
    n_obs = len(obs_indices)
    T = obs_partial.shape[0]

    omega_H = params['omega_H']
    omega_L = params['omega_L']

    # Determine monitor variable and compute its observed path.
    # Two cases:
    #   Case 1: monitor IS one of the observed variables  ->  read directly.
    #   Case 2: monitor is a PURE IDENTITY of observed variables (possibly at
    #           lags)  ->  use the parser-built `monitor_resolution` map which
    #           stores entries like
    #               {var_name: [(obs_var, lag, coeff), ...]}
    #           and compute monitor[t] = sum_j coeff_j * obs[t + lag_j, src_j].
    if monitor_index is None:
        cred_jax = model.get('credibility_jax')
        if cred_jax and 'monitor' in cred_jax:
            monitor_var = cred_jax['monitor']
        elif model['regime_spec'] and 'monitor' in model['regime_spec']:
            monitor_var = model['regime_spec']['monitor']
        else:
            raise ValueError("No monitor_index given and no regimes/credibility block.")
        full_idx = var_names.index(monitor_var)
        obs_idx_list = list(obs_indices)
        if full_idx in obs_idx_list:
            monitor_index = obs_idx_list.index(full_idx)
            obs_monitor = obs_partial[:, monitor_index]
        else:
            monitor_resolution = model.get('monitor_resolution', {})
            if monitor_var not in monitor_resolution:
                raise ValueError(
                    f"Credibility monitor '{monitor_var}' is neither observed "
                    f"nor a resolved identity of observed variables. Define "
                    f"'{monitor_var}' via a pure linear identity equation "
                    f"(no shock, no leads) whose RHS contains only observed "
                    f"variables (possibly at lags)."
                )
            terms = monitor_resolution[monitor_var]
            obs_monitor = jnp.zeros(T)
            for src_var, lag, coeff in terms:
                src_full = var_names.index(src_var)
                if src_full not in obs_idx_list:
                    raise ValueError(
                        f"Monitor '{monitor_var}' resolves to '{src_var}' "
                        f"which is not observed."
                    )
                src_col = obs_idx_list.index(src_full)
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

    # Omega parameter name
    omega_param = 'omega'
    if model['regime_spec']:
        m1_params = model['regime_spec'].get('M1_params', {})
        if m1_params:
            omega_param = list(m1_params.keys())[0]

    # ----- Step 1: Credibility path -----
    cred_scan_fn, init_carry = _build_cred_scan_fn(model, params)
    _, cred_path = lax.scan(cred_scan_fn, init_carry, obs_monitor)
    omega_path = omega_L + (omega_H - omega_L) * cred_path

    # ----- Step 2: Terminal solution -----
    params_M1 = dict(params)
    params_M1[omega_param] = omega_H
    A_term, B_term, C_term = build_ABC(params_M1)
    F_terminal = solve_terminal_jax(A_term, B_term, C_term)

    # Stationarity check at the M1 (terminal) regime. Two conditions
    # must hold for the PWL solution to be well-posed (Rendahl 2017):
    #
    #   (a) The structured doubling iteration CONVERGED to a true fixed
    #       point, i.e. the residual of the quadratic matrix equation
    #           A + B F + C F^2 = 0
    #       is numerically zero. When the model is non-stationary SDA
    #       does NOT converge -- it returns whatever F it had at the
    #       last iteration, whose eigenvalues are arbitrary numerical
    #       artifacts and CANNOT be used as the stability diagnostic.
    #       The residual norm is the reliable signal.
    #
    #   (b) The converged F is stationary: max|eig(F)| < 1. M1 is the
    #       terminal regime by assumption (the system reverts to it as
    #       credibility recovers), so this is the necessary condition
    #       for the backward recursion to be globally well-posed. M2
    #       does NOT need its own stationary fixed point -- it only
    #       enters as per-period A_t, B_t, C_t in the recursion.
    #
    # The residual tolerance scales with the working precision: float64
    # easily reaches 1e-10 even for borderline rho ~ 0.99, so 1e-6 is
    # comfortable. float32 only reaches ~1e-5 in the same case, so the
    # tolerance is relaxed proportionally. ALWAYS run estimation in
    # float64 (jax.config.update('jax_enable_x64', True)).
    qme_residual = A_term + B_term @ F_terminal + C_term @ F_terminal @ F_terminal
    res_norm = jnp.max(jnp.abs(qme_residual))
    res_tol = 1e-6 if F_terminal.dtype == jnp.float64 else 1e-3
    converged = res_norm < res_tol

    eigs_M1 = jnp.linalg.eigvals(F_terminal)
    rho_M1 = jnp.max(jnp.abs(eigs_M1))
    spec_ok = rho_M1 < 1.0

    stable_M1 = converged & spec_ok

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
