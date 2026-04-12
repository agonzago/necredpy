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
        cred_scan_fn(carry, monitor_t) -> (new_carry, (cred_t, omega_t))

    The scan output is a (cred, omega) tuple for ALL grammar variants,
    so callers can always do:
        _, (cred_path, omega_path) = lax.scan(fn, init, obs_monitor)

    Three paths (checked in order):
      1. New grammar (credibility_new): self-contained credibility block
         compiled from credibility; ... end; with JAX-traceable callable.
      2. Legacy pluggable (credibility_jax): signal/accumulation expressions.
      3. Hardcoded Isard: fallback with delta_up/delta_down.
    """
    cred_new = model.get('credibility_new')

    # ----- Path 1: New grammar (credibility; ... end;) -----
    if cred_new is not None:
        credibility_fn = cred_new['fn']
        state_vars = cred_new['state_vars']
        input_vars = cred_new['input_vars']
        output_vars = cred_new['output_vars']
        used_params = cred_new['used_params']

        # Pre-extract credibility parameters (closed over, not re-read)
        cred_params = {p: params[p] for p in used_params}

        # Initial state: full credibility
        init_state = {v: jnp.array(1.0) if v == 'cred_state'
                      else jnp.array(0.0) for v in state_vars}

        def cred_scan_fn(carry, monitor_t):
            prev_state = carry
            inputs = {input_vars[0]: monitor_t}
            outputs, new_state = credibility_fn(inputs, prev_state,
                                                cred_params)
            omega_t = outputs[output_vars[0]]
            cred_t = new_state.get('cred_state', jnp.array(1.0))
            return new_state, (cred_t, omega_t)

        return cred_scan_fn, init_state

    # ----- Path 2: Legacy pluggable credibility -----
    cred_jax = model.get('credibility_jax')

    if cred_jax is not None and (
            cred_jax['signal_fn_jax'] is not None or
            cred_jax['accumulation_fn_jax'] is not None):
        signal_fn = cred_jax['signal_fn_jax']
        acc_fn = cred_jax['accumulation_fn_jax']
        signal_lag = cred_jax['signal_lag']
        signal_params = cred_jax['signal_params']
        acc_params = cred_jax['accumulation_params']
        cred_init = cred_jax['cred_init']

        sig_param_vals = [params[p] for p in signal_params]
        acc_param_vals = [params[p] for p in acc_params]

        epsilon_bar = params.get('epsilon_bar', 2.0)
        tau = params.get('tau', 0.2)
        omega_H = params['omega_H']
        omega_L = params['omega_L']

        def cred_scan_fn(carry, pi_t):
            cred, pi_lag = carry

            miss = jax.nn.sigmoid((pi_t**2 - epsilon_bar**2) / tau)

            if signal_fn is not None:
                pi_input = pi_lag if signal_lag == 1 else pi_t
                sig = signal_fn(pi_t, pi_input, 0.0, *sig_param_vals)
            else:
                sig = 0.0

            if acc_fn is not None:
                cred_new = acc_fn(cred, sig, miss, *acc_param_vals)
            else:
                delta_up = params['delta_up']
                delta_down = params['delta_down']
                cred_new = (cred + delta_up * (1.0 - cred) * (1.0 - miss)
                            - delta_down * cred * miss)

            cred_new = jnp.clip(cred_new, 0.0, 1.0)
            omega_t = omega_L + (omega_H - omega_L) * cred_new
            return (cred_new, pi_t), (cred_new, omega_t)

        init_carry = (cred_init, 0.0)
        return cred_scan_fn, init_carry

    else:
        # ----- Path 3: Hardcoded Isard -----
        delta_up = params['delta_up']
        delta_down = params['delta_down']
        epsilon_bar = params['epsilon_bar']
        tau = params['tau']
        omega_H = params['omega_H']
        omega_L = params['omega_L']

        def cred_scan_fn(carry, pi_t):
            cred, pi_lag = carry
            cred_new = credibility_step(cred, pi_t, epsilon_bar, tau,
                                         delta_up, delta_down)
            omega_t = omega_L + (omega_H - omega_L) * cred_new
            return (cred_new, pi_t), (cred_new, omega_t)

        init_carry = (1.0, 0.0)
        return cred_scan_fn, init_carry


# ---------------------------------------------------------------------------
# Model-agnostic compiled model
# ---------------------------------------------------------------------------

def compile_jax_model(mod_string, verbose=False):
    """Parse a .mod file and return a JAX-ready model bundle.

    This is the expensive step (sympy parsing + lambdification).
    Call once at startup, then use the returned bundle for estimation.

    Supports both the legacy credibility grammar (monitor:/signal=/
    accumulation=) and the new credibility; ... end; block with explicit
    var/input/output declarations.  Detection is automatic: if the .mod
    file has a new-grammar credibility block AND model(pwl) or model(nn),
    credibility outputs become time-varying coefficients (not estimated
    parameters).  Otherwise the legacy path is used unchanged.

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
        'coefficient_names' : list of str (e.g. ['omega_pc']; empty for legacy)
        'regime_spec'       : dict or None (from regimes; block)
        'priors'            : list of dict (from priors; block)
        'credibility_jax'   : dict or None (legacy JAX-compiled credibility)
        'credibility_new'   : dict or None (new grammar compiled credibility)
    """
    from necredpy.utils.dynare_parser import (parse_credibility_mod,
                                               jax_lambdify,
                                               extract_regimes, extract_priors,
                                               extract_credibility)

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

    regime_spec = extract_regimes(mod_string)
    priors = extract_priors(mod_string)

    # Legacy credibility (only when new grammar was NOT found)
    credibility_jax = None
    if cred_compiled is None:
        cred_spec = extract_credibility(mod_string)
        if cred_spec is not None:
            credibility_jax = compile_credibility_fn_jax(cred_spec, param_names)

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
        'credibility_jax': credibility_jax,
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

    # Determine monitor variable index
    if monitor_index is None:
        if cred_new is not None:
            monitor_var = cred_new['input_vars'][0]
            monitor_index = var_names.index(monitor_var)
        else:
            cred_jax = model.get('credibility_jax')
            if cred_jax and 'monitor' in cred_jax:
                monitor_index = var_names.index(cred_jax['monitor'])
            elif model['regime_spec'] and 'monitor' in model['regime_spec']:
                monitor_index = var_names.index(
                    model['regime_spec']['monitor'])
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
        cred_params = {p: params[p] for p in cred_new['used_params']}
        out_ss, _ = cred_new['fn'](ss_inputs, ss_state, cred_params)
        omega_terminal = out_ss[cred_new['output_vars'][0]]
        A_term, B_term, C_term = build_ABC(params, omega_terminal)
    else:
        # Legacy: stuff omega into params dict
        omega_H = params['omega_H']
        omega_param = 'omega'
        if model['regime_spec']:
            m1_params = model['regime_spec'].get('M1_params', {})
            if m1_params:
                omega_param = list(m1_params.keys())[0]
        params_M1 = dict(params)
        params_M1[omega_param] = omega_H
        A_term, B_term, C_term = build_ABC(params_M1)

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
    if coeff_names:
        # New grammar: omega_pc is a coefficient, not a parameter
        def build_for_omega(omega_t):
            return build_ABC(params, omega_t)
    else:
        # Legacy: stuff omega into params dict
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
            cred_jax = model.get('credibility_jax')
            if cred_jax and 'monitor' in cred_jax:
                monitor_var = cred_jax['monitor']
            elif model['regime_spec'] and 'monitor' in model['regime_spec']:
                monitor_var = model['regime_spec']['monitor']
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
        cred_params = {p: params[p] for p in cred_new['used_params']}
        out_ss, _ = cred_new['fn'](ss_inputs, ss_state, cred_params)
        omega_terminal = out_ss[cred_new['output_vars'][0]]
        A_term, B_term, C_term = build_ABC(params, omega_terminal)
    else:
        # Legacy: stuff omega into params dict
        omega_H = params['omega_H']
        omega_param = 'omega'
        if model['regime_spec']:
            m1_params = model['regime_spec'].get('M1_params', {})
            if m1_params:
                omega_param = list(m1_params.keys())[0]
        params_M1 = dict(params)
        params_M1[omega_param] = omega_H
        A_term, B_term, C_term = build_ABC(params_M1)

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
    if coeff_names:
        def build_for_omega(omega_t):
            return build_ABC(params, omega_t)
    else:
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
