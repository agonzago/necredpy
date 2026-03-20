"""
Lightweight Dynare .mod file parser for the Pontus piecewise-linear solver.

Parses a standard Dynare .mod file, extracts structural equations, computes
symbolic Jacobians via sympy, and returns lambdified matrix functions that
can be evaluated at different parameter values.

Adapted from qpm_toolbox/parser.py (Gonzalez, 2024).

Convention:
  Parser computes A_lead (coeff of y(t+1)), B_t (coeff of y(t)), C_lag (coeff of y(t-1))
  Pontus solver expects A u_{t-1} + B u_t + C E[u_{t+1}] + D = 0
  So: pontus_A = C_lag, pontus_B = B_t, pontus_C = A_lead

Dependencies: re, sympy, numpy (no JAX)
"""

import re
import sympy
import numpy as np
from collections import OrderedDict


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _create_timed_symbol(base_name, time_shift):
    """Create a sympy symbol for a variable at a given time shift."""
    if time_shift == -1:
        return sympy.symbols(f"{base_name}_m1")
    elif time_shift == 0:
        return sympy.symbols(f"{base_name}_t")
    elif time_shift == 1:
        return sympy.symbols(f"{base_name}_p1")
    else:
        raise ValueError(f"Unexpected time shift {time_shift} for {base_name}")


def extract_declarations(mod_string):
    """Parse var, varexo, parameters blocks from a .mod file.

    Returns
    -------
    var_names : list of str
    shock_names : list of str
    param_names : list of str
    param_assignments : dict {name: float}
    """
    # Strip comments
    processed = re.sub(r'/\*.*?\*/', '', mod_string, flags=re.DOTALL)
    lines = processed.split('\n')
    cleaned = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed = " \n ".join(cleaned)

    # Only search before model block
    model_marker = re.search(r'\bmodel\b', processed, re.IGNORECASE)
    content = processed[:model_marker.start()] if model_marker else processed

    declarations = {'var': [], 'varexo': [], 'parameters': []}
    block_matches = re.finditer(
        r'(?i)\b(var|varexo|parameters)\b(.*?)(?=\b(?:var|varexo|parameters|model)\b|$)',
        content, re.DOTALL | re.IGNORECASE
    )

    for match in block_matches:
        keyword = match.group(1).lower()
        block_content = match.group(2).strip()
        # Take content up to first semicolon
        semi = re.search(r';', block_content)
        if semi:
            block_content = block_content[:semi.start()].strip()
        block_content = block_content.replace('\n', ' ')
        names = re.split(r'[,\s]+', block_content)
        names = [n for n in names if n and re.fullmatch(r'[a-zA-Z_]\w*', n)]
        keywords = {'var', 'varexo', 'parameters', 'model', 'end'}
        names = [n for n in names if n not in keywords]
        declarations[keyword].extend(names)

    # Deduplicate preserving order
    declarations = {k: list(dict.fromkeys(v)) for k, v in declarations.items()}

    # Extract parameter assignments (name = value;)
    # Two passes: first collect all assignments as strings, then resolve
    # in order (handles expressions like alpha_VA1 = 1 - omega_12).
    param_assignments = {}
    param_names_set = set(declarations.get('parameters', []))
    raw_assignments = []
    for match in re.finditer(r'\b([a-zA-Z_]\w*)\b\s*=\s*([^;]+);', content):
        name = match.group(1)
        value_str = match.group(2).strip()
        if name in param_names_set:
            raw_assignments.append((name, value_str))

    for name, value_str in raw_assignments:
        try:
            param_assignments[name] = float(value_str)
        except ValueError:
            # Try evaluating as an expression using already-resolved params
            try:
                param_assignments[name] = float(
                    eval(value_str, {"__builtins__": {}}, param_assignments))
            except Exception:
                pass

    return (declarations['var'], declarations['varexo'],
            declarations['parameters'], param_assignments)


def extract_model_equations(mod_string):
    """Extract equations from the model; ... end; block.

    Each equation 'LHS = RHS' is converted to '(LHS) - (RHS)' so the
    zero-residual form F(...) = 0 is used for Jacobian computation.

    Returns
    -------
    equations : list of str
    """
    processed = re.sub(r'/\*.*?\*/', '', mod_string, flags=re.DOTALL)
    lines = processed.split('\n')
    cleaned = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed = " ".join(cleaned)

    # Handle model; and model(linear); variants
    model_match = re.search(
        r'(?i)\bmodel\b\s*(?:\([^)]*\))?\s*;(.*?)\bend\b\s*;',
        processed, re.DOTALL
    )
    if not model_match:
        raise ValueError("Could not find 'model; ... end;' block.")

    model_content = model_match.group(1)
    raw_equations = [eq.strip() for eq in model_content.split(';') if eq.strip()]

    equations = []
    for line in raw_equations:
        if '=' in line:
            parts = line.split('=', 1)
            lhs, rhs = parts[0].strip(), parts[1].strip()
            equations.append(f"({lhs}) - ({rhs})")
        else:
            equations.append(line)

    return equations


# ---------------------------------------------------------------------------
# Symbolic Jacobian computation
# ---------------------------------------------------------------------------

def parse_mod(mod_string, verbose=False):
    """Parse a .mod file and return lambdified matrix functions.

    Parameters
    ----------
    mod_string : str
        Full content of the .mod file.
    verbose : bool
        Print progress info.

    Returns
    -------
    result : dict with keys:
        'func_A', 'func_B', 'func_C', 'func_D' : callables
            Each takes (*param_values) and returns a numpy array.
            A = coeff of y(t-1) [pontus convention]
            B = coeff of y(t)
            C = coeff of y(t+1)
            D = coeff of shocks (with minus sign for Q calculation)
        'var_names' : list of str
        'shock_names' : list of str
        'param_names' : list of str
        'param_defaults' : dict {name: float}
    """
    # --- Step 1: Parse declarations ---
    var_names, shock_names, param_names, param_defaults = extract_declarations(mod_string)
    if verbose:
        print(f"Variables ({len(var_names)}): {var_names}")
        print(f"Shocks ({len(shock_names)}): {shock_names}")
        print(f"Parameters ({len(param_names)}): {param_names}")

    # --- Step 2: Parse equations ---
    raw_equations = extract_model_equations(mod_string)
    num_vars = len(var_names)
    num_eq = len(raw_equations)
    num_shocks = len(shock_names)

    if num_vars != num_eq:
        raise ValueError(
            f"Model not square: {num_vars} variables vs {num_eq} equations."
        )
    if verbose:
        print(f"Equations ({num_eq}):")
        for i, eq in enumerate(raw_equations):
            print(f"  [{i}] {eq}")

    # --- Step 3: Create symbolic representations ---
    param_syms = {p: sympy.symbols(p) for p in param_names}
    shock_syms = {s: sympy.symbols(s) for s in shock_names}

    var_syms = {}
    all_syms_for_parsing = set(param_syms.values()) | set(shock_syms.values())
    for var in var_names:
        sym_m1 = _create_timed_symbol(var, -1)
        sym_t = _create_timed_symbol(var, 0)
        sym_p1 = _create_timed_symbol(var, 1)
        var_syms[var] = {'m1': sym_m1, 't': sym_t, 'p1': sym_p1}
        all_syms_for_parsing.update([sym_m1, sym_t, sym_p1])

    local_dict = {str(s): s for s in all_syms_for_parsing}
    local_dict.update({'log': sympy.log, 'exp': sympy.exp,
                       'sqrt': sympy.sqrt, 'abs': sympy.Abs})

    from sympy.parsing.sympy_parser import (
        parse_expr, standard_transformations,
        implicit_multiplication_application, rationalize
    )
    transformations = (standard_transformations +
                       (implicit_multiplication_application, rationalize))

    # Regex for var(time_shift) patterns
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')

    sym_equations = []
    for i, eq_str in enumerate(raw_equations):
        eq_sym = eq_str

        # Replace var(+1), var(-1) etc. with timed symbols
        def replace_var_time(match):
            base_name, time_str = match.groups()
            time_shift = int(time_str)
            if base_name in shock_syms:
                if time_shift == 0:
                    return str(shock_syms[base_name])
                raise ValueError(f"Shock {base_name}({time_shift}) invalid.")
            elif base_name in var_syms:
                if time_shift == -1:
                    return str(var_syms[base_name]['m1'])
                if time_shift == 0:
                    return str(var_syms[base_name]['t'])
                if time_shift == 1:
                    return str(var_syms[base_name]['p1'])
                raise ValueError(
                    f"Time shift {time_shift} for {base_name}: "
                    "only -1, 0, +1 supported (no auxiliary vars)."
                )
            elif base_name in local_dict:
                return match.group(0)
            else:
                raise ValueError(f"Unknown symbol {base_name}({time_shift})")

        eq_sym = var_time_regex.sub(replace_var_time, eq_sym)

        # Replace bare variable names (implicitly time t)
        all_names = sorted(
            list(var_syms.keys()) + param_names + shock_names,
            key=len, reverse=True
        )
        for name in all_names:
            pattern = r'\b' + re.escape(name) + r'\b'
            if name in var_syms:
                replacement = str(var_syms[name]['t'])
            elif name in param_syms:
                replacement = str(param_syms[name])
            elif name in shock_syms:
                replacement = str(shock_syms[name])
            else:
                continue
            eq_sym = re.sub(pattern, replacement, eq_sym)

        sym_eq = parse_expr(eq_sym, local_dict=local_dict,
                            transformations=transformations)
        sym_equations.append(sym_eq)

    if verbose:
        print("Symbolic equations:")
        for i, eq in enumerate(sym_equations):
            print(f"  [{i}] {eq}")

    # --- Step 4: Compute Jacobians ---
    # Convention for pontus solver: A u_{t-1} + B u_t + C E[u_{t+1}] = 0
    # A = dF/dy_{t-1},  B = dF/dy_t,  C = dF/dy_{t+1}
    # D = -dF/de_t (minus sign for shock impact)

    var_m1_syms = [var_syms[v]['m1'] for v in var_names]
    var_t_syms = [var_syms[v]['t'] for v in var_names]
    var_p1_syms = [var_syms[v]['p1'] for v in var_names]
    shock_t_syms = [shock_syms[s] for s in shock_names]

    sympy_A = sympy.zeros(num_eq, num_vars)  # coeff of y(t-1)
    sympy_B = sympy.zeros(num_eq, num_vars)  # coeff of y(t)
    sympy_C = sympy.zeros(num_eq, num_vars)  # coeff of y(t+1)
    sympy_D = sympy.zeros(num_eq, num_shocks)  # -coeff of e(t)

    for i, eq in enumerate(sym_equations):
        for j, s in enumerate(var_m1_syms):
            sympy_A[i, j] = sympy.diff(eq, s)
        for j, s in enumerate(var_t_syms):
            sympy_B[i, j] = sympy.diff(eq, s)
        for j, s in enumerate(var_p1_syms):
            sympy_C[i, j] = sympy.diff(eq, s)
        for k, s in enumerate(shock_t_syms):
            sympy_D[i, k] = -sympy.diff(eq, s)

    if verbose:
        print("Symbolic A (lag):\n", sympy_A)
        print("Symbolic B (contemp):\n", sympy_B)
        print("Symbolic C (lead):\n", sympy_C)
        print("Symbolic D (shocks):\n", sympy_D)

    # --- Step 5: Extract constant terms ---
    # D_const[i] = F_i evaluated at all variables = 0 and all shocks = 0
    # For models linearized around SS this is zero; for regime-dependent SS
    # (e.g., cred capital in M2) it captures the constant offset.
    all_var_and_shock_syms = (var_m1_syms + var_t_syms + var_p1_syms
                              + shock_t_syms)
    zero_subs = {s: 0 for s in all_var_and_shock_syms}
    sympy_D_const = sympy.zeros(num_eq, 1)
    for i, eq in enumerate(sym_equations):
        sympy_D_const[i] = eq.subs(zero_subs)

    # --- Step 6: Lambdify ---
    param_sym_list = [param_syms[p] for p in param_names]

    func_A = sympy.lambdify(param_sym_list, sympy_A, modules='numpy')
    func_B = sympy.lambdify(param_sym_list, sympy_B, modules='numpy')
    func_C = sympy.lambdify(param_sym_list, sympy_C, modules='numpy')
    func_D = sympy.lambdify(param_sym_list, sympy_D, modules='numpy')
    func_D_const = sympy.lambdify(param_sym_list, sympy_D_const, modules='numpy')

    result = {
        'func_A': func_A,
        'func_B': func_B,
        'func_C': func_C,
        'func_D': func_D,
        'func_D_const': func_D_const,
        'var_names': var_names,
        'shock_names': shock_names,
        'param_names': param_names,
        'param_defaults': param_defaults,
        # Store symbolic matrices for JAX re-lambdification
        '_sympy_A': sympy_A,
        '_sympy_B': sympy_B,
        '_sympy_C': sympy_C,
        '_sympy_D': sympy_D,
        '_sympy_D_const': sympy_D_const,
        '_param_sym_list': param_sym_list,
    }

    return result


def jax_lambdify(parsed):
    """Re-lambdify parsed model matrices for JAX (differentiable).

    Takes the output of parse_mod() and returns JAX-compatible matrix
    functions. These can be traced by JAX's autodiff for HMC/NUTS.

    Parameters
    ----------
    parsed : dict
        Output of parse_mod() (must contain '_sympy_*' keys).

    Returns
    -------
    jax_funcs : dict with keys 'func_A', 'func_B', 'func_C', 'func_D',
        'func_D_const', 'param_names', 'var_names', 'shock_names',
        'param_defaults'.
        Each func takes (*param_values) and returns a jnp.ndarray.
    """
    import jax.numpy as jnp

    jax_mod = [{'ImmutableDenseMatrix': jnp.array}, 'jax']
    psl = parsed['_param_sym_list']

    return {
        'func_A': sympy.lambdify(psl, parsed['_sympy_A'], modules=jax_mod),
        'func_B': sympy.lambdify(psl, parsed['_sympy_B'], modules=jax_mod),
        'func_C': sympy.lambdify(psl, parsed['_sympy_C'], modules=jax_mod),
        'func_D': sympy.lambdify(psl, parsed['_sympy_D'], modules=jax_mod),
        'func_D_const': sympy.lambdify(psl, parsed['_sympy_D_const'],
                                        modules=jax_mod),
        'param_names': parsed['param_names'],
        'var_names': parsed['var_names'],
        'shock_names': parsed['shock_names'],
        'param_defaults': parsed['param_defaults'],
    }


# ---------------------------------------------------------------------------
# Convenience: parse + evaluate -> numpy matrices
# ---------------------------------------------------------------------------

def get_model_matrices(mod_string, params_dict, verbose=False):
    """Parse a .mod file and evaluate matrices at given parameter values.

    Parameters
    ----------
    mod_string : str
        Full .mod file content.
    params_dict : dict
        Parameter values. Missing keys use defaults from the .mod file.
    verbose : bool

    Returns
    -------
    A, B, C, D : ndarray
        In pontus convention: A u_{t-1} + B u_t + C E[u_{t+1}] + D = 0
        D is the constant vector (zeros for models linearized around SS).
    info : dict
        Variable names, shock names, parameter names.
    """
    parsed = parse_mod(mod_string, verbose=verbose)

    # Merge defaults with user-supplied params
    params = dict(parsed['param_defaults'])
    params.update(params_dict)

    # Build argument list in the order the lambdified functions expect
    args = [params[p] for p in parsed['param_names']]

    A = np.array(parsed['func_A'](*args), dtype=float)
    B = np.array(parsed['func_B'](*args), dtype=float)
    C = np.array(parsed['func_C'](*args), dtype=float)
    D_shock = np.array(parsed['func_D'](*args), dtype=float)

    # D constant vector: nonzero when regime has different effective SS
    # Guard: sympy.lambdify may return scalar 0 instead of an array when
    # all constant terms vanish. Ensure shape is always (n,).
    n = len(parsed['var_names'])
    D_const_raw = parsed['func_D_const'](*args)
    D_const = np.zeros(n)
    raw_flat = np.atleast_1d(np.array(D_const_raw, dtype=float)).flatten()
    if raw_flat.size == n:
        D_const[:] = raw_flat
    elif raw_flat.size == 1:
        D_const[:] = raw_flat[0]

    info = {
        'var_names': parsed['var_names'],
        'shock_names': parsed['shock_names'],
        'param_names': parsed['param_names'],
        'D_shock': D_shock,  # shock selection matrix for Q calculation
    }

    return A, B, C, D_const, info


def get_two_regime_matrices(mod_string, params_base, omega_H, omega_L,
                            omega_param='omega', verbose=False):
    """Parse .mod file and return matrices for both credibility regimes.

    Parameters
    ----------
    mod_string : str
    params_base : dict
        Base parameter values (without omega).
    omega_H, omega_L : float
        Credibility weights for M1 and M2.
    omega_param : str
        Name of the omega parameter in the .mod file.
    verbose : bool

    Returns
    -------
    matrices_M1 : tuple (A1, B1, C1, D1)
    matrices_M2 : tuple (A2, B2, C2, D2)
    info : dict
    """
    params_M1 = dict(params_base)
    params_M1[omega_param] = omega_H
    A1, B1, C1, D1, info = get_model_matrices(mod_string, params_M1, verbose=verbose)

    params_M2 = dict(params_base)
    params_M2[omega_param] = omega_L
    A2, B2, C2, D2, _ = get_model_matrices(mod_string, params_M2)

    return (A1, B1, C1, D1), (A2, B2, C2, D2), info


# ---------------------------------------------------------------------------
# Regimes block parser
# ---------------------------------------------------------------------------

def extract_regimes(mod_string):
    """Parse the regimes; ... end; block from a .mod file.

    Syntax:
        regimes;
          M1: omega = 0.65;
          M2: omega = 0.35;

          switch: shadow_cred;       % or k_restore
          monitor: pi;               % variable to watch
          band: 2.0;                 % |monitor| > band triggers miss
          cred_threshold: 0.5;       % below this -> M2 (shadow_cred only)
          delta_up: 0.05;            % rebuild rate  (shadow_cred only)
          delta_down: 0.70;          % depletion rate (shadow_cred only)
          cred_init: 1.0;            % initial credibility (shadow_cred only)
          k_restore: 4;              % consecutive periods in band (k_restore only)
        end;

    Returns
    -------
    regime_spec : dict with keys:
        'M1_params' : dict {param_name: value_expr}
        'M2_params' : dict {param_name: value_expr}
        'switch_type' : str ('shadow_cred' or 'k_restore')
        'monitor' : str (variable name)
        'band' : float (threshold for |monitor|)
        + switching-type-specific keys (cred_threshold, delta_up, etc.)
    Returns None if no regimes block found.
    """
    # Strip block comments
    processed = re.sub(r'/\*.*?\*/', '', mod_string, flags=re.DOTALL)
    lines = processed.split('\n')
    cleaned = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed = " ".join(cleaned)

    # Find regimes; ... end; block
    regime_match = re.search(
        r'(?i)\bregimes\b\s*;(.*?)\bend\b\s*;',
        processed, re.DOTALL
    )
    if not regime_match:
        return None

    block = regime_match.group(1)

    # Parse M1: and M2: parameter overrides
    regime_spec = {'M1_params': {}, 'M2_params': {}}

    for regime_key in ['M1', 'M2']:
        pattern = r'\b' + regime_key + r'\s*:(.*?);'
        match = re.search(pattern, block, re.IGNORECASE)
        if match:
            assignments_str = match.group(1).strip()
            # Parse comma-separated param = value pairs
            for pair in re.split(r',', assignments_str):
                pair = pair.strip()
                if not pair:
                    continue
                m = re.match(r'([a-zA-Z_]\w*)\s*=\s*(.+)', pair)
                if m:
                    pname = m.group(1).strip()
                    pval_str = m.group(2).strip()
                    try:
                        regime_spec[regime_key + '_params'][pname] = float(pval_str)
                    except ValueError:
                        # Store as string (could be another param name)
                        regime_spec[regime_key + '_params'][pname] = pval_str

    # Parse switching specification key: value; pairs
    switch_keys = ['switch', 'monitor', 'band', 'cred_threshold',
                   'delta_up', 'delta_down', 'cred_init', 'k_restore']
    for key in switch_keys:
        pattern = r'\b' + key + r'\s*:\s*([^;]+);'
        match = re.search(pattern, block, re.IGNORECASE)
        if match:
            val_str = match.group(1).strip()
            if key == 'switch':
                regime_spec['switch_type'] = val_str
            elif key == 'monitor':
                regime_spec['monitor'] = val_str
            else:
                try:
                    regime_spec[key] = float(val_str)
                except ValueError:
                    regime_spec[key] = val_str

    return regime_spec


# ---------------------------------------------------------------------------
# Priors block parser
# ---------------------------------------------------------------------------

# Supported distributions and their NumPyro mappings.
#
# Following Dynare convention: p1=mean, p2=std for most distributions.
# The parser converts (mean, std) to shape parameters internally.
#
#   normal,      p1=mean, p2=std [, p3=low, p4=high]  -> TruncatedNormal or Normal
#   beta_dist,   p1=mean, p2=std                       -> Beta(a, b)
#   gamma_dist,  p1=mean, p2=std                       -> Gamma(a, b)
#   inv_gamma,   p1=mean, p2=std                       -> InverseGamma(a, b)
#   uniform,     p1=low,  p2=high                      -> Uniform(low, high)
#   half_normal, p1=scale                               -> HalfNormal(scale)
#
# Beta(a,b) conversion from (mean, std):
#   mean = a/(a+b),  var = ab/((a+b)^2(a+b+1))
#   Let nu = a+b = mean*(1-mean)/var - 1
#   a = mean*nu,  b = (1-mean)*nu
#   Requires 0 < mean < 1 and std < sqrt(mean*(1-mean))
#
# Gamma(a,b) conversion from (mean, std):
#   mean = a/b,  var = a/b^2
#   b = mean/var,  a = mean*b
#
# InverseGamma(a,b) conversion from (mean, std):
#   mean = b/(a-1) for a>1,  var = b^2/((a-1)^2(a-2)) for a>2
#   a = (mean/std)^2 + 2,  b = mean*(a-1)
#
# The parser returns a list of dicts that can be consumed by NumPyro or
# any other sampler without importing NumPyro at parse time.

KNOWN_DISTRIBUTIONS = {
    'normal', 'beta_dist', 'gamma_dist', 'inv_gamma',
    'uniform', 'half_normal',
    # Dynare aliases
    'normal_pdf', 'beta_pdf', 'gamma_pdf', 'inv_gamma_pdf', 'uniform_pdf',
}

# Map Dynare _pdf names to our canonical names
_DIST_ALIASES = {
    'normal_pdf': 'normal',
    'beta_pdf': 'beta_dist',
    'gamma_pdf': 'gamma_dist',
    'inv_gamma_pdf': 'inv_gamma',
    'uniform_pdf': 'uniform',
}


def _beta_mean_std_to_ab(mean, std):
    """Convert Beta(mean, std) to Beta(a, b) shape parameters.

    Dynare convention: user specifies mean and std of the Beta distribution.
    We convert to shape parameters (a, b) for NumPyro.

    Requires: 0 < mean < 1, std < sqrt(mean*(1-mean)).
    """
    var = std ** 2
    max_var = mean * (1.0 - mean)
    if var >= max_var:
        raise ValueError(
            "Beta prior: std=%.4f too large for mean=%.4f. "
            "Need std < %.4f = sqrt(mean*(1-mean))."
            % (std, mean, max_var ** 0.5))
    nu = mean * (1.0 - mean) / var - 1.0
    a = mean * nu
    b = (1.0 - mean) * nu
    return a, b


def _gamma_mean_std_to_ab(mean, std):
    """Convert Gamma(mean, std) to Gamma(concentration, rate) parameters.

    NumPyro Gamma(a, b): mean = a/b, var = a/b^2.
    """
    var = std ** 2
    rate = mean / var
    concentration = mean * rate
    return concentration, rate


def _inv_gamma_mean_std_to_ab(mean, std):
    """Convert InverseGamma(mean, std) to InverseGamma(a, b) parameters.

    InverseGamma(a, b): mean = b/(a-1) for a>1, var = b^2/((a-1)^2(a-2)) for a>2.
    Solving: a = (mean/std)^2 + 2,  b = mean*(a-1).
    """
    var = std ** 2
    a = (mean / std) ** 2 + 2.0
    b = mean * (a - 1.0)
    return a, b


def extract_priors(mod_string):
    """Parse the priors; ... end; block from a .mod file.

    Syntax (Dynare-inspired, all use mean/std):
        priors;
          kappa,    normal,      0.3,  0.15, 0.01, 1.0;   // mean, std, low, high
          phi_pi,   normal,      1.5,  0.3,  1.01, 3.0;
          sigma_d,  inv_gamma,   0.5,  0.3;                // mean, std
          sigma_s,  half_normal, 1.0;                       // scale
          rho_i,    beta_dist,   0.7,  0.1;                // mean, std
          alpha,    uniform,     0.0,  1.0;                // low, high
        end;

    Also accepts Dynare's 'estimated_params; ... end;' as a synonym.

    For beta_dist, gamma_dist, inv_gamma: p1=mean, p2=std (converted to
    shape parameters internally). This prevents U-shaped beta priors from
    accidental low shape parameters.

    Parameters
    ----------
    mod_string : str
        Full .mod file content.

    Returns
    -------
    priors : list of dict, each with keys:
        'name'  : str (parameter name)
        'dist'  : str (canonical distribution name)
        'mean'  : float (prior mean, for reporting)
        'std'   : float (prior std, for reporting)
        'shape' : tuple (shape parameters for NumPyro, distribution-specific)
        'p3'    : float or None (lower bound for truncation)
        'p4'    : float or None (upper bound for truncation)
    Returns empty list if no priors block found.
    """
    processed = re.sub(r'/\*.*?\*/', '', mod_string, flags=re.DOTALL)
    lines = processed.split('\n')
    cleaned = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed = " ".join(cleaned)

    # Try both 'priors;' and 'estimated_params;'
    block_match = re.search(
        r'(?i)\b(?:priors|estimated_params)\b\s*;(.*?)\bend\b\s*;',
        processed, re.DOTALL
    )
    if not block_match:
        return []

    block = block_match.group(1)
    raw_entries = [e.strip() for e in block.split(';') if e.strip()]

    priors = []
    for entry in raw_entries:
        parts = [p.strip() for p in entry.split(',')]
        if len(parts) < 2:
            continue

        name = parts[0]
        dist_name = parts[1].lower()

        # Resolve aliases
        dist_name = _DIST_ALIASES.get(dist_name, dist_name)

        if dist_name not in KNOWN_DISTRIBUTIONS:
            raise ValueError(
                "Unknown prior distribution '%s' for parameter '%s'. "
                "Known distributions: %s"
                % (dist_name, name, sorted(KNOWN_DISTRIBUTIONS)))

        p1 = float(parts[2]) if len(parts) > 2 else None
        p2 = float(parts[3]) if len(parts) > 3 else None
        p3 = float(parts[4]) if len(parts) > 4 else None
        p4 = float(parts[5]) if len(parts) > 5 else None

        # Convert (mean, std) to shape parameters for relevant distributions
        mean_val = p1
        std_val = p2
        if dist_name == 'beta_dist' and p1 is not None and p2 is not None:
            a, b = _beta_mean_std_to_ab(p1, p2)
            shape = (a, b)
        elif dist_name == 'gamma_dist' and p1 is not None and p2 is not None:
            a, b = _gamma_mean_std_to_ab(p1, p2)
            shape = (a, b)
        elif dist_name == 'inv_gamma' and p1 is not None and p2 is not None:
            a, b = _inv_gamma_mean_std_to_ab(p1, p2)
            shape = (a, b)
        elif dist_name == 'normal':
            shape = (p1, p2)  # mean, std directly
        elif dist_name == 'uniform':
            shape = (p1, p2)  # low, high
            mean_val = (p1 + p2) / 2 if p1 is not None and p2 is not None else None
            std_val = None
        elif dist_name == 'half_normal':
            shape = (p1,)  # scale
            mean_val = p1
            std_val = None
        else:
            shape = (p1, p2)

        priors.append({
            'name': name,
            'dist': dist_name,
            'mean': mean_val,
            'std': std_val,
            'shape': shape,
            'p3': p3,
            'p4': p4,
        })

    return priors


def priors_to_numpyro(priors):
    """Convert parsed priors to NumPyro sample statements (as source code).

    Uses the pre-computed shape parameters from extract_priors(), which
    already converted (mean, std) -> (a, b) for beta/gamma/inv_gamma.

    Parameters
    ----------
    priors : list of dict
        Output of extract_priors().

    Returns
    -------
    numpyro_code : str
        Python source code for NumPyro prior sampling.
    """
    lines = []
    for pr in priors:
        name = pr['name']
        d = pr['dist']
        shape = pr['shape']
        p3, p4 = pr['p3'], pr['p4']

        if d == 'normal':
            if p3 is not None and p4 is not None:
                lines.append(
                    "%s = numpyro.sample('%s', dist.TruncatedNormal("
                    "loc=%s, scale=%s, low=%s, high=%s))"
                    % (name, name, shape[0], shape[1], p3, p4))
            else:
                lines.append(
                    "%s = numpyro.sample('%s', dist.Normal(%s, %s))"
                    % (name, name, shape[0], shape[1]))
        elif d == 'beta_dist':
            lines.append(
                "# Beta(mean=%.3f, std=%.3f) -> Beta(a=%.3f, b=%.3f)\n"
                "%s = numpyro.sample('%s', dist.Beta(%.6f, %.6f))"
                % (pr['mean'], pr['std'], shape[0], shape[1],
                   name, name, shape[0], shape[1]))
        elif d == 'gamma_dist':
            lines.append(
                "# Gamma(mean=%.3f, std=%.3f) -> Gamma(a=%.3f, b=%.3f)\n"
                "%s = numpyro.sample('%s', dist.Gamma(%.6f, %.6f))"
                % (pr['mean'], pr['std'], shape[0], shape[1],
                   name, name, shape[0], shape[1]))
        elif d == 'inv_gamma':
            lines.append(
                "# InvGamma(mean=%.3f, std=%.3f) -> InvGamma(a=%.3f, b=%.3f)\n"
                "%s = numpyro.sample('%s', dist.InverseGamma(%.6f, %.6f))"
                % (pr['mean'], pr['std'], shape[0], shape[1],
                   name, name, shape[0], shape[1]))
        elif d == 'uniform':
            lines.append(
                "%s = numpyro.sample('%s', dist.Uniform(%s, %s))"
                % (name, name, shape[0], shape[1]))
        elif d == 'half_normal':
            lines.append(
                "%s = numpyro.sample('%s', dist.HalfNormal(%s))"
                % (name, name, shape[0]))

    return "\n".join(lines)


def build_numpyro_prior_fn(priors):
    """Build a callable that samples all priors and returns a dict.

    Parameters
    ----------
    priors : list of dict
        Output of extract_priors().

    Returns
    -------
    sample_priors : callable() -> dict {name: sampled_value}
        When called inside a NumPyro model context, samples all priors
        and returns them as a dictionary.
    estimated_names : list of str
        Names of the estimated parameters (in order).
    """
    import numpyro
    import numpyro.distributions as ndist

    def sample_priors():
        sampled = {}
        for pr in priors:
            name = pr['name']
            d = pr['dist']
            shape = pr['shape']
            p3, p4 = pr['p3'], pr['p4']

            if d == 'normal':
                if p3 is not None and p4 is not None:
                    sampled[name] = numpyro.sample(
                        name, ndist.TruncatedNormal(
                            loc=shape[0], scale=shape[1], low=p3, high=p4))
                else:
                    sampled[name] = numpyro.sample(
                        name, ndist.Normal(shape[0], shape[1]))
            elif d == 'beta_dist':
                sampled[name] = numpyro.sample(
                    name, ndist.Beta(shape[0], shape[1]))
            elif d == 'gamma_dist':
                sampled[name] = numpyro.sample(
                    name, ndist.Gamma(shape[0], shape[1]))
            elif d == 'inv_gamma':
                sampled[name] = numpyro.sample(
                    name, ndist.InverseGamma(shape[0], shape[1]))
            elif d == 'uniform':
                sampled[name] = numpyro.sample(
                    name, ndist.Uniform(shape[0], shape[1]))
            elif d == 'half_normal':
                sampled[name] = numpyro.sample(
                    name, ndist.HalfNormal(shape[0]))
        return sampled

    estimated_names = [pr['name'] for pr in priors]
    return sample_priors, estimated_names


def _resolve_param_value(val, param_defaults):
    """Resolve a regime parameter value that may reference another parameter."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and val in param_defaults:
        return param_defaults[val]
    try:
        return float(val)
    except (ValueError, TypeError):
        raise ValueError(
            "Cannot resolve regime parameter value '%s'. "
            "Must be a number or a declared parameter name." % val
        )


def build_switching_fn(regime_spec, var_names):
    """Build a switching function from a parsed regimes specification.

    Parameters
    ----------
    regime_spec : dict
        Output of extract_regimes().
    var_names : list of str
        Variable names from the model (to find monitor index).

    Returns
    -------
    switching_fn : callable(u_path) -> regime_seq
        Compatible with pontus.solve_endogenous().
    """
    monitor_var = regime_spec['monitor']
    if monitor_var not in var_names:
        raise ValueError(
            "Monitor variable '%s' not found in model variables: %s"
            % (monitor_var, var_names)
        )
    pi_index = var_names.index(monitor_var)
    band = regime_spec['band']
    switch_type = regime_spec.get('switch_type')
    if not switch_type:
        raise ValueError("Missing 'switch' type in regimes block.")

    if switch_type == 'shadow_cred':
        # Validate required keys
        for req in ['delta_up', 'delta_down']:
            if req not in regime_spec:
                raise ValueError(
                    "shadow_cred switching requires '%s' in regimes block." % req)
        cred_threshold = regime_spec.get('cred_threshold', 0.5)
        delta_up = regime_spec['delta_up']
        delta_down = regime_spec['delta_down']
        cred_init = regime_spec.get('cred_init', 1.0)

        def switching_fn(u_path):
            """Shadow credibility switching: cred depletes when |monitor| > band,
            rebuilds otherwise. M2 when cred < cred_threshold."""
            T = u_path.shape[0]
            regime_seq = np.zeros(T, dtype=int)
            cred_path = np.zeros(T)
            cred = cred_init
            for t in range(T):
                if abs(u_path[t, pi_index]) > band:
                    cred = cred - delta_down * cred
                else:
                    cred = cred + delta_up * (1.0 - cred)
                cred = max(0.0, min(1.0, cred))
                cred_path[t] = cred
                regime_seq[t] = 1 if cred < cred_threshold else 0
            switching_fn.cred_path = cred_path
            return regime_seq

        switching_fn.cred_path = None
        return switching_fn

    elif switch_type == 'k_restore':
        if 'k_restore' not in regime_spec:
            raise ValueError(
                "k_restore switching requires 'k_restore' in regimes block.")
        k_restore = int(regime_spec['k_restore'])

        def switching_fn(u_path):
            """k-restore switching: M2 when |monitor| > band, M1 restored after
            k_restore consecutive periods within band."""
            T = u_path.shape[0]
            regime_seq = np.zeros(T, dtype=int)
            in_M2 = False
            consecutive_in_band = 0
            for t in range(T):
                if not in_M2:
                    if abs(u_path[t, pi_index]) > band:
                        in_M2 = True
                        consecutive_in_band = 0
                        regime_seq[t] = 1
                    else:
                        regime_seq[t] = 0
                else:
                    if abs(u_path[t, pi_index]) <= band:
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

    else:
        raise ValueError(
            "Unknown switch type '%s'. Use 'shadow_cred' or 'k_restore'."
            % switch_type
        )


# ---------------------------------------------------------------------------
# High-level: compile once / evaluate many (estimation-ready architecture)
# ---------------------------------------------------------------------------

def compile_two_regime_model(mod_string, verbose=False):
    """Parse a .mod file with a regimes block (expensive sympy step, done once).

    For estimation: call this once, then call evaluate_two_regime_model()
    repeatedly with different param_overrides per MCMC draw.

    Parameters
    ----------
    mod_string : str
        Full .mod file content with regimes; ... end; block.
    verbose : bool

    Returns
    -------
    compiled : dict with keys:
        'parsed' : dict from parse_mod() (lambdified functions)
        'regime_spec_raw' : dict from extract_regimes() (unresolved strings)
    """
    regime_spec_raw = extract_regimes(mod_string)
    if regime_spec_raw is None:
        raise ValueError("No 'regimes; ... end;' block found in .mod file.")

    parsed = parse_mod(mod_string, verbose=verbose)

    return {
        'parsed': parsed,
        'regime_spec_raw': regime_spec_raw,
    }


def evaluate_two_regime_model(compiled, param_overrides=None):
    """Evaluate a compiled two-regime model at specific parameter values.

    Cheap numpy-only step: no sympy. Call repeatedly for estimation.

    Parameters
    ----------
    compiled : dict
        Output of compile_two_regime_model().
    param_overrides : dict or None
        Parameter overrides applied to BOTH regimes. These override the
        defaults from the .mod file and flow through to regime parameter
        resolution and switching function construction.

    Returns
    -------
    matrices_M1 : tuple (A1, B1, C1, D1)
    matrices_M2 : tuple (A2, B2, C2, D2)
    switching_fn : callable(u_path) -> regime_seq
    info : dict with keys:
        'var_names', 'shock_names', 'param_names', 'D_shock',
        'regime_spec', 'params_M1', 'params_M2'
    """
    import copy
    parsed = compiled['parsed']
    # Deep copy to avoid mutating the raw spec (strings -> floats)
    regime_spec = copy.deepcopy(compiled['regime_spec_raw'])

    # Build base params from defaults + overrides
    base_params = dict(parsed['param_defaults'])
    if param_overrides:
        base_params.update(param_overrides)

    # Resolve regime-specific parameter overrides
    params_M1 = dict(base_params)
    for pname, pval in regime_spec['M1_params'].items():
        if pname not in parsed['param_names']:
            raise ValueError(
                "Regime M1 overrides parameter '%s' which is not declared "
                "in the parameters block." % pname)
        params_M1[pname] = _resolve_param_value(pval, base_params)

    params_M2 = dict(base_params)
    for pname, pval in regime_spec['M2_params'].items():
        if pname not in parsed['param_names']:
            raise ValueError(
                "Regime M2 overrides parameter '%s' which is not declared "
                "in the parameters block." % pname)
        params_M2[pname] = _resolve_param_value(pval, base_params)

    # Evaluate matrices (including regime-dependent D constant vector)
    n = len(parsed['var_names'])

    def _eval_matrices(params):
        args = [params[p] for p in parsed['param_names']]
        A = np.array(parsed['func_A'](*args), dtype=float)
        B = np.array(parsed['func_B'](*args), dtype=float)
        C = np.array(parsed['func_C'](*args), dtype=float)
        D_shock = np.array(parsed['func_D'](*args), dtype=float)
        # Guard: lambdify may return scalar 0 when all constants vanish
        D_const = np.zeros(n)
        raw_flat = np.atleast_1d(
            np.array(parsed['func_D_const'](*args), dtype=float)).flatten()
        if raw_flat.size == n:
            D_const[:] = raw_flat
        elif raw_flat.size == 1:
            D_const[:] = raw_flat[0]
        return A, B, C, D_const, D_shock

    A1, B1, C1, D1, D_shock = _eval_matrices(params_M1)
    A2, B2, C2, D2, _ = _eval_matrices(params_M2)

    matrices_M1 = (A1, B1, C1, D1)
    matrices_M2 = (A2, B2, C2, D2)

    # Resolve ALL switching parameters against model parameter names.
    # This allows the regimes block to reference declared parameters
    # (e.g., band: epsilon_bar;) so they are estimable via param_overrides.
    switch_numeric_keys = ['band', 'cred_threshold', 'delta_up', 'delta_down',
                           'cred_init', 'k_restore']
    for key in switch_numeric_keys:
        if key in regime_spec:
            regime_spec[key] = _resolve_param_value(
                regime_spec[key], base_params)

    # Build switching function
    switching_fn = build_switching_fn(regime_spec, parsed['var_names'])

    info = {
        'var_names': parsed['var_names'],
        'shock_names': parsed['shock_names'],
        'param_names': parsed['param_names'],
        'D_shock': D_shock,
        'regime_spec': regime_spec,
        'params_M1': params_M1,
        'params_M2': params_M2,
    }

    return matrices_M1, matrices_M2, switching_fn, info


def parse_two_regime_model(mod_string, param_overrides=None, verbose=False):
    """Parse a .mod file with a regimes block and return everything needed
    for the Pontus endogenous switching solver.

    Convenience wrapper: compile + evaluate in one call.
    For estimation loops, use compile_two_regime_model() +
    evaluate_two_regime_model() separately.

    Parameters
    ----------
    mod_string : str
        Full .mod file content.
    param_overrides : dict or None
        Additional parameter overrides applied to BOTH regimes.
    verbose : bool

    Returns
    -------
    matrices_M1 : tuple (A1, B1, C1, D1)
    matrices_M2 : tuple (A2, B2, C2, D2)
    switching_fn : callable(u_path) -> regime_seq
    info : dict with keys:
        'var_names', 'shock_names', 'param_names', 'D_shock',
        'regime_spec', 'params_M1', 'params_M2'
    """
    compiled = compile_two_regime_model(mod_string, verbose=verbose)
    return evaluate_two_regime_model(compiled, param_overrides=param_overrides)
