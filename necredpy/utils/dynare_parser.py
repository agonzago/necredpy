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

    # Strip credibility; ... end; blocks so their internal var/parameters
    # keywords are not captured as global declarations.
    processed = re.sub(
        r'(?i)\bcredibility\b\s*;.*?\bend\b\s*;', '', processed,
        flags=re.DOTALL)

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
    """Extract equations and options from the model; ... end; block.

    Each equation 'LHS = RHS' is converted to '(LHS) - (RHS)' so the
    zero-residual form F(...) = 0 is used for Jacobian computation.

    Recognised model-block options (Dynare-style, comma-separated inside
    the parentheses after `model`):

      linear         : informational only -- the parser already builds
                       linear matrices via symbolic differentiation, so
                       this flag is accepted for compatibility but does
                       not change behaviour.
      nocredibility  : declares that the model does NOT have a
                       credibility/regimes block. Downstream code that
                       would normally require one (compile_two_regime_model,
                       nn_solver.update_credibility, ...) will treat the
                       model as fully linear with cred==1, omega==1.

    Returns
    -------
    equations : list of str
    options : dict
        Parsed model-block options. Currently:
            {'nocredibility': bool, 'linear': bool}
    """
    processed = re.sub(r'/\*.*?\*/', '', mod_string, flags=re.DOTALL)
    lines = processed.split('\n')
    cleaned = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed = " ".join(cleaned)

    # Capture the options-list (if any) AND the body together so we can
    # parse the flags inside the parens.
    model_match = re.search(
        r'(?i)\bmodel\b\s*(?:\(([^)]*)\))?\s*;(.*?)\bend\b\s*;',
        processed, re.DOTALL
    )
    if not model_match:
        raise ValueError("Could not find 'model; ... end;' block.")

    options_str = (model_match.group(1) or "").strip()
    model_content = model_match.group(2)

    # Parse comma-separated options inside model(...).
    #   model(linear)         -- target: linear (default)
    #   model(pwl)            -- target: piecewise-linear with credibility
    #   model(nn)             -- target: NN solver with credibility merged
    #   model(nocredibility)  -- flag: no credibility block
    #   model(linear, nocredibility)  -- target + flag
    #
    # The solution target is stored in options['target'].
    # Default target is 'linear' if none specified.
    options = {'linear': False, 'nocredibility': False, 'target': 'linear'}
    target_keywords = {'linear', 'pwl', 'nn'}
    if options_str:
        for tok in options_str.split(','):
            tok = tok.strip().lower()
            if not tok:
                continue
            if tok in target_keywords:
                options['target'] = tok
            if tok in options and tok != 'target':
                options[tok] = True
            # Unknown options are silently ignored for forward-compat.

    raw_equations = [eq.strip() for eq in model_content.split(';') if eq.strip()]

    equations = []
    for line in raw_equations:
        if '=' in line:
            parts = line.split('=', 1)
            lhs, rhs = parts[0].strip(), parts[1].strip()
            equations.append(f"({lhs}) - ({rhs})")
        else:
            equations.append(line)

    return equations, options


# ---------------------------------------------------------------------------
# Symbolic Jacobian computation
# ---------------------------------------------------------------------------

def _expand_aux_variables(raw_equations, var_names, shock_names, verbose=False):
    """Expand lags/leads of order > 1 by adding auxiliary variables.

    See docstring elsewhere. In addition to expanding, returns an
    `aux_resolution` map: {aux_name: (base_var, total_lag)} so that any
    auxiliary can be resolved back to its source variable at the right lag.
    For example, aux_pi_cpi_lag_m2 -> ('pi_cpi', -2).

    Returns
    -------
    expanded_equations : list of str
    expanded_var_names : list of str
    aux_vars_added : list of str
    aux_resolution : dict {aux_name: (base_name, lag)}
    """
    var_time_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*([+-]?\d+)\s*\)')

    # Work on copies
    expanded_equations = list(raw_equations)
    expanded_var_names = list(var_names)
    aux_vars_added = []           # track order of insertion
    aux_definitions = {}          # aux_name -> definition equation string
    aux_resolution = {}           # aux_name -> (base_name, total_lag)
    endogenous_set = set(var_names)
    shock_set = set(shock_names)

    def _ensure_aux_chain(base_name, shift):
        """Create (if needed) the chain of aux variables for base_name(shift).

        For shift = -k (k >= 2), builds the chain:
            aux_<base>_lag_m1 = base(-1)        (represents base(-1) at time t)
            aux_<base>_lag_m2 = aux_<base>_lag_m1(-1)  (represents base(-2) at time t)
            ...
            aux_<base>_lag_m{k-1} = aux_<base>_lag_m{k-2}(-1)
        and returns the name of the tail aux variable (here `aux_<base>_lag_m{k-1}`).
        The caller then substitutes `base(-k)` with `aux_<base>_lag_m{k-1}(-1)`.

        Analogously for leads (shift = +k).
        """
        k = abs(shift)
        if k < 2:
            return None
        is_lag = shift < 0
        tag = 'lag_m' if is_lag else 'lead_p'
        chain_step = '(-1)' if is_lag else '(+1)'
        sign = -1 if is_lag else +1

        for j in range(1, k):  # j = 1 .. k-1
            aux_name = f"aux_{base_name}_{tag}{j}"
            if aux_name in aux_definitions:
                continue
            prev = base_name if j == 1 else f"aux_{base_name}_{tag}{j-1}"
            def_eq = f"{aux_name} - {prev}{chain_step}"
            aux_definitions[aux_name] = def_eq
            aux_vars_added.append(aux_name)
            # aux_<base>_lag_m{j} at time t represents base(-j)
            aux_resolution[aux_name] = (base_name, sign * j)
            if aux_name not in endogenous_set:
                expanded_var_names.append(aux_name)
                endogenous_set.add(aux_name)
            expanded_equations.append(def_eq)

        return f"aux_{base_name}_{tag}{k-1}"

    # Walk each original equation and substitute every x(-k)/x(+k) with
    # k >= 2 by the corresponding CHAIN TAIL aux variable referenced at
    # (-1) or (+1). For k == 1 references, leave them alone (the parser
    # natively handles single-step shifts via the A/C matrices).
    #
    # Substitution: x(-3) -> aux_x_lag_m2(-1)
    # Aux chain:    aux_x_lag_m1 = x(-1)
    #               aux_x_lag_m2 = aux_x_lag_m1(-1)     (= x(-2))
    #
    # This matches the qpm_toolbox convention: substitutions stay as
    # (-1) references so existing A/B/C Jacobians work unchanged.
    n_original = len(raw_equations)
    for i in range(n_original):
        eq = expanded_equations[i]
        matches = list(var_time_regex.finditer(eq))
        modified = eq
        for match in reversed(matches):
            base_name = match.group(1)
            shift = int(match.group(2))
            if base_name in shock_set:
                continue
            if base_name not in endogenous_set:
                continue
            if abs(shift) <= 1:
                continue
            aux_name = _ensure_aux_chain(base_name, shift)
            chain_step = '(-1)' if shift < 0 else '(+1)'
            replacement = f"{aux_name}{chain_step}"
            start, end = match.span()
            modified = modified[:start] + replacement + modified[end:]
        expanded_equations[i] = modified

    return expanded_equations, expanded_var_names, aux_vars_added, aux_resolution


def parse_mod(mod_string, verbose=False, coefficient_names=None):
    """Parse a .mod file and return lambdified matrix functions.

    Parameters
    ----------
    mod_string : str
        Full content of the .mod file.
    verbose : bool
        Print progress info.
    coefficient_names : list of str or None
        Names of time-varying coefficients that appear in the model
        equations but are NOT estimated parameters and NOT endogenous
        variables. Typical example: 'omega_pc' -- the credibility
        output that enters the Phillips curve as a coefficient.

        These are kept SEPARATE from param_names so that NUTS never
        sees them. The lambdified matrix functions take them as extra
        arguments AFTER the parameter list:

            func_A(*param_values, *coeff_values) -> numpy array

        When coefficient_names is None (the default), the lambdified
        functions take only parameters and any free symbol in the
        equations that is not a variable, shock, or parameter will
        cause a runtime NameError. This is the correct behavior for
        model(linear), where the user must declare everything.

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

    # --- Step 2: Parse equations and model-block options ---
    raw_equations, model_options = extract_model_equations(mod_string)

    # --- Step 2b: Expand lags/leads > 1 via auxiliary variables ---
    # For lags: x(-k) becomes a chain aux_x_lag_m{j} = prev(-1) for j=1..k-1,
    # and the original term is replaced by aux_x_lag_m{k-1}(-1).
    # For leads: x(+k) similarly with aux_x_lead_p{j}.
    # Auxiliary variable definitions are appended to the equation list so
    # the system remains square.
    raw_equations, var_names, aux_vars_added, aux_resolution = _expand_aux_variables(
        raw_equations, var_names, shock_names, verbose=verbose)
    if verbose and aux_vars_added:
        print(f"Added {len(aux_vars_added)} auxiliary variables for lags/leads > 1:")
        for aux in aux_vars_added:
            base, lag = aux_resolution[aux]
            print(f"  {aux}  represents  {base}({lag:+d})")

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

    # Coefficient symbols: time-varying coefficients (e.g. omega_pc from
    # credibility block) that are NOT parameters and NOT variables.
    # They appear in A/B/C as symbolic coefficients and are passed as
    # separate arguments to the lambdified functions.
    coeff_names = coefficient_names or []
    coeff_syms = {c: sympy.symbols(c) for c in coeff_names}

    var_syms = {}
    all_syms_for_parsing = (set(param_syms.values())
                            | set(shock_syms.values())
                            | set(coeff_syms.values()))
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
            list(var_syms.keys()) + param_names + shock_names + coeff_names,
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
            elif name in coeff_syms:
                replacement = str(coeff_syms[name])
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

    # --- Step 4a: Reorder equation rows to match var_names ---
    # The .mod file can declare variables in one order and write equations
    # in a different order. Downstream filter code assumes ROW i of the
    # Jacobian matrices corresponds to the equation that OWNS var_names[i]
    # (i.e., the equation in which var_names[i] appears on the LHS of "=").
    # We identify equation owners from the raw equation strings and build
    # a permutation that places each owner-equation at the row of its
    # owned variable.
    _lhs_re = re.compile(r'^\s*\(\s*([A-Za-z_]\w*)\s*\)\s*-')
    _aux_re = re.compile(r'^\s*([A-Za-z_]\w*)\s*-')
    eq_owner_name = [None] * num_eq
    for i, eq_str in enumerate(raw_equations):
        m = _lhs_re.match(eq_str)
        if m:
            eq_owner_name[i] = m.group(1)
        else:
            m2 = _aux_re.match(eq_str)
            if m2:
                eq_owner_name[i] = m2.group(1)

    # Build permutation: for each variable (in var_names order), find the
    # equation row that owns it. This gives new_row_index -> old_row_index.
    var_to_old_eq = {}
    for old_i, owner in enumerate(eq_owner_name):
        if owner in var_to_old_eq:
            # Ambiguous: two equations claim the same owner. Keep the first.
            continue
        var_to_old_eq[owner] = old_i

    missing = [v for v in var_names if v not in var_to_old_eq]
    if missing:
        if verbose:
            print(f"Warning: could not identify owner equation for "
                  f"{len(missing)} variables: {missing[:5]}... "
                  f"Keeping original equation order.")
        # Fall back: no reordering. Rows stay as written.
        permutation = list(range(num_eq))
    else:
        permutation = [var_to_old_eq[v] for v in var_names]

    # Apply permutation to sympy matrices (which are indexed by eq row).
    # This happens ONCE at parse time on symbolic matrices. After
    # lambdification, build_ABC returns pre-permuted matrices at every
    # call, so the filter never re-permutes at runtime.
    #
    # Skip the work if the permutation is already the identity (a
    # well-written .mod file with eq order matching var_names order).
    if permutation != list(range(num_eq)):
        sympy_A = sympy.Matrix([sympy_A.row(old_i) for old_i in permutation])
        sympy_B = sympy.Matrix([sympy_B.row(old_i) for old_i in permutation])
        sympy_C = sympy.Matrix([sympy_C.row(old_i) for old_i in permutation])
        sympy_D = sympy.Matrix([sympy_D.row(old_i) for old_i in permutation])
        sym_equations = [sym_equations[old_i] for old_i in permutation]
        raw_equations = [raw_equations[old_i] for old_i in permutation]
        if verbose:
            print(f"Reordered {num_eq} equations to match var_names order")

    # --- Step 4b: Build the (var, lag)-dependency graph and monitor_resolution ---
    #
    # The A/B/C/D matrices encode a dependency graph:
    #   nodes     : (variable, relative_lag) pairs
    #   directed edges : equation that defines `owner` at lag 0 contributes
    #                    edges  (owner, 0) -> (other_var, lag, coeff)
    # The "defining equation" of each variable is identified via equation
    # OWNERSHIP, which we read from the raw equation strings (LHS of "=" for
    # user equations, first token before "-" for appended aux definitions).
    #
    # Only equations with NO shocks, NO leads, and NO lags in any of their
    # terms can be resolved purely from data. BUT: we can still resolve a
    # variable whose defining equation contains lags, as long as those lags
    # are of variables that can themselves be resolved (recursively).
    #
    # The resolution is a DFS walk of the graph. We stop at "sink" nodes:
    # source variables (non-identity, non-aux) whose value is presumed to
    # be read from observations. We collapse the walk into a flat list of
    # (source_var, lag, coeff) tuples.
    import networkx as nx

    # Identify equation owners: equation row index -> variable it "defines".
    # User equations have been transformed to the form "(LHS) - (RHS)" by
    # extract_model_equations(). Aux definition equations (appended by
    # _expand_aux_variables) have the form "aux_name - prev_term".
    user_eq_re = re.compile(r'^\s*\(\s*([A-Za-z_]\w*)\s*\)\s*-')
    aux_def_re = re.compile(r'^\s*([A-Za-z_]\w*)\s*-')

    eq_owner = {}  # eq_row_index -> var_name
    for i, eq_str in enumerate(raw_equations):
        m = user_eq_re.match(eq_str)
        if m:
            candidate = m.group(1)
        else:
            m2 = aux_def_re.match(eq_str)
            candidate = m2.group(1) if m2 else None
        if candidate and candidate in var_names:
            eq_owner[i] = candidate

    # Build a directed graph: owner -> (dep_var, lag, weight).
    # We skip equations with SHOCKS (can't be resolved from data) and
    # equations with LEADS (forward-looking, not a pure definition).
    # Equations with lags ARE allowed — we'll resolve the lagged vars
    # recursively. Aux definitions (which have exactly one lag term)
    # are handled specially via the `aux_resolution` map to avoid
    # circular graph walks.
    dep_graph = nx.MultiDiGraph()

    for eq_i, owner in eq_owner.items():
        # Skip if this equation has a shock: the owner is not resolvable
        # from data alone. Parameter-dependent coefficients are OK.
        if any(sympy_D[eq_i, k] != 0 for k in range(num_shocks)):
            continue
        # Skip if the equation has any forward expectation.
        if any(sympy_C[eq_i, k] != 0 for k in range(num_vars)):
            continue
        # Skip aux definitions -- those are handled by aux_resolution.
        if owner in aux_resolution:
            continue
        j = var_names.index(owner)
        b_jj = sympy_B[eq_i, j]
        if b_jj == 0:
            continue
        # Collect dependencies: contemporaneous (lag 0) from B, lagged
        # (lag -1) from A. Skip self-reference at lag 0.
        for k in range(num_vars):
            if k == j:
                continue
            coeff = sympy_B[eq_i, k]
            if coeff == 0:
                continue
            try:
                w = float(-coeff / b_jj)
            except (TypeError, ValueError):
                w = None  # parameter-dependent; unresolvable
            dep_graph.add_edge(owner, var_names[k], lag=0, weight=w)
        for k in range(num_vars):
            coeff = sympy_A[eq_i, k]
            if coeff == 0:
                continue
            try:
                w = float(-coeff / b_jj)
            except (TypeError, ValueError):
                w = None
            dep_graph.add_edge(owner, var_names[k], lag=-1, weight=w)

    if verbose:
        print(f"Dependency graph: {dep_graph.number_of_nodes()} nodes, "
              f"{dep_graph.number_of_edges()} edges")

    # Resolution: walk the graph depth-first from a query variable,
    # accumulating lag offsets and coefficient products. Aux variables
    # are substituted via aux_resolution. A variable is "fully resolvable"
    # only if EVERY edge encountered in its walk has a numeric weight
    # and every terminal node is a source (non-identity, non-aux). If the
    # walk hits a parameter-dependent edge or exceeds depth, the whole
    # variable is declared unresolvable (no entry in monitor_resolution).
    class _Unresolvable(Exception):
        pass

    def _resolve_graph(var_name, lag_acc, coeff_acc, out_terms, depth=0):
        if depth > 30:
            raise _Unresolvable(f"depth > 30 at {var_name}")
        if var_name in aux_resolution:
            base, aux_lag = aux_resolution[var_name]
            _resolve_graph(base, lag_acc + aux_lag, coeff_acc,
                           out_terms, depth + 1)
            return
        if var_name not in dep_graph or dep_graph.out_degree(var_name) == 0:
            # Source variable -- emit the accumulated term.
            out_terms.append((var_name, lag_acc, coeff_acc))
            return
        for _, neighbor, data in dep_graph.out_edges(var_name, data=True):
            w = data.get('weight')
            if w is None:
                raise _Unresolvable(f"param edge at {var_name}->{neighbor}")
            edge_lag = data.get('lag', 0)
            _resolve_graph(neighbor, lag_acc + edge_lag, coeff_acc * w,
                           out_terms, depth + 1)

    # Build monitor_resolution map for every variable that has a defining
    # equation (i.e., is a node in dep_graph with outgoing edges).
    monitor_resolution = {}
    for var in dep_graph.nodes():
        if dep_graph.out_degree(var) == 0:
            continue
        raw_terms = []
        try:
            _resolve_graph(var, 0, 1.0, raw_terms)
        except _Unresolvable:
            continue  # silently skip; not all variables are data-computable
        collapsed = {}
        for src, lag, c in raw_terms:
            collapsed[(src, lag)] = collapsed.get((src, lag), 0.0) + c
        entries = [(src, lag, c) for (src, lag), c in collapsed.items()
                   if abs(c) > 1e-14]
        monitor_resolution[var] = entries

    if verbose and monitor_resolution:
        print(f"monitor_resolution ({len(monitor_resolution)} variables "
              "resolved to observables):")
        for v, entries in monitor_resolution.items():
            desc = " + ".join(f"{c:.4g}*{src}({lag:+d})" for src, lag, c in entries)
            print(f"  {v} = {desc}")

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
    coeff_sym_list = [coeff_syms[c] for c in coeff_names]
    # Lambdified functions take: func_A(*param_values, *coeff_values)
    full_sym_list = param_sym_list + coeff_sym_list

    func_A = sympy.lambdify(full_sym_list, sympy_A, modules='numpy')
    func_B = sympy.lambdify(full_sym_list, sympy_B, modules='numpy')
    func_C = sympy.lambdify(full_sym_list, sympy_C, modules='numpy')
    func_D = sympy.lambdify(full_sym_list, sympy_D, modules='numpy')
    func_D_const = sympy.lambdify(full_sym_list, sympy_D_const, modules='numpy')

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
        'coefficient_names': coeff_names,
        # Model-block options parsed from `model(...)`. Currently:
        #   nocredibility -- declares that the model has no credibility
        #                    block; downstream code (compile_two_regime_model,
        #                    nn_solver) treats it as fully linear.
        'model_options': model_options,
        # Aux and identity resolution (used by inversion filter to look up
        # monitor variables that are not directly observed).
        'aux_resolution': aux_resolution,
        'monitor_resolution': monitor_resolution,
        # Store symbolic matrices for JAX re-lambdification
        '_sympy_A': sympy_A,
        '_sympy_B': sympy_B,
        '_sympy_C': sympy_C,
        '_sympy_D': sympy_D,
        '_sympy_D_const': sympy_D_const,
        '_param_sym_list': param_sym_list,
        '_coeff_sym_list': coeff_sym_list,
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
    csl = parsed.get('_coeff_sym_list', [])
    full_sym_list = psl + csl

    return {
        'func_A': sympy.lambdify(full_sym_list, parsed['_sympy_A'],
                                  modules=jax_mod),
        'func_B': sympy.lambdify(full_sym_list, parsed['_sympy_B'],
                                  modules=jax_mod),
        'func_C': sympy.lambdify(full_sym_list, parsed['_sympy_C'],
                                  modules=jax_mod),
        'func_D': sympy.lambdify(full_sym_list, parsed['_sympy_D'],
                                  modules=jax_mod),
        'func_D_const': sympy.lambdify(full_sym_list,
                                        parsed['_sympy_D_const'],
                                        modules=jax_mod),
        'param_names': parsed['param_names'],
        'coefficient_names': parsed.get('coefficient_names', []),
        'var_names': parsed['var_names'],
        'shock_names': parsed['shock_names'],
        'param_defaults': parsed['param_defaults'],
        'aux_resolution': parsed.get('aux_resolution', {}),
        'monitor_resolution': parsed.get('monitor_resolution', {}),
        'model_options': parsed.get('model_options', {}),
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
# Credibility block -- NEW GRAMMAR
# ---------------------------------------------------------------------------
#
# The `credibility; ... end;` block is a self-contained sub-model with
# its own variable and parameter namespaces. At parse time the local
# variables and parameters are merged into the global lists used by the
# estimation/solver, but inside the block they are written as regular
# Dynare-style equations with real names. No magic symbols (no `pi`
# meaning the monitor, no `s` meaning the signal output, no `miss`).
#
# Grammar:
#
#   credibility;
#       var <name1> <name2> ...;             // local variables
#       parameters <name1> <name2> ...;      // local parameters
#       input  <model_var1> ...;             // variables read from model;
#       output <local_var1> ...;             // variables exported to model;
#
#       <lhs1> = <rhs1>;                     // equations in declaration order
#       <lhs2> = <rhs2>;
#       ...
#   end;
#
# Rules enforced by the parser:
#
#   * Every name on an equation LHS must be declared in `var`.
#   * Every declared var must have exactly one equation.
#   * Every `input` name must exist in the model's `var` list (checked
#     later, at the top-level parse stage).
#   * Every `output` name must be declared in the block's `var`.
#   * A local var may appear with (-1) lag in its own defining equation
#     or in later equations -- this marks it as a "credibility state"
#     that the PWL compile path lifts into the outer regime loop.
#
# Returns a dict:
#
#   {
#     'local_vars'  : list[str]     -- declared variables
#     'local_params': list[str]     -- declared parameters
#     'inputs'      : list[str]     -- names read from the model block
#     'outputs'     : list[str]     -- names exported to the model block
#     'equations'   : list[dict]    -- [{'lhs': str, 'rhs': str}, ...]
#                                       in declaration order
#   }
#
# Returns None if no credibility block is found in the .mod string.


def parse_credibility_block(mod_string):
    """Parse a `credibility; ... end;` block written in the NEW grammar.

    See the module-level comment above the function definition for the
    grammar rules and the return schema. Returns None if the block is
    absent from the .mod file.

    Raises
    ------
    ValueError
        If a declaration keyword is mis-used, if an LHS is not declared,
        if a declared variable has no equation, or if `output` refers to
        a name that was not declared in `var`.
    """
    # ---------- Strip comments ----------
    processed = re.sub(r'/\*.*?\*/', '', mod_string, flags=re.DOTALL)
    lines = processed.split('\n')
    cleaned = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed = " \n ".join(cleaned)

    # ---------- Locate the credibility; ... end; block ----------
    match = re.search(
        r'(?i)\bcredibility\b\s*;(.*?)\bend\b\s*;',
        processed, re.DOTALL
    )
    if not match:
        return None

    block = match.group(1)

    # Quick guard against the LEGACY magic-symbol form (monitor:,
    # signal = ..., accumulation = ...). If we see those tokens, hand
    # back None here so the caller can fall through to the legacy path.
    legacy_markers = (
        r'\bmonitor\s*:',
        r'\bsignal\s*=',
        r'\baccumulation\s*=',
    )
    for pat in legacy_markers:
        if re.search(pat, block, re.IGNORECASE):
            return None

    # ---------- Split the block into semicolon-terminated statements ----------
    # A statement is everything up to the next top-level `;`. The block
    # has no parentheses that span semicolons so a naive split is safe.
    raw_stmts = [s.strip() for s in block.split(';')]
    raw_stmts = [s for s in raw_stmts if s]  # drop empty tails

    local_vars = []
    inputs = []
    outputs = []
    equations = []

    # Keywords that start a declaration; the first word of the statement
    # tells us what kind of statement it is.
    # NOTE: `parameters` is NOT a credibility-block keyword. All parameters
    # are declared globally (in the .mod file's `parameters` block) and
    # simply referenced by name in credibility equations, same as in the
    # model; block. This keeps one flat parameter namespace.
    declaration_heads = {
        'var':    local_vars,
        'input':  inputs,
        'output': outputs,
    }

    name_regex = re.compile(r'[a-zA-Z_]\w*')

    for stmt in raw_stmts:
        # Is it a declaration? (starts with one of the keyword heads,
        # checked case-insensitively)
        first_word_match = re.match(r'([a-zA-Z_]\w*)\b', stmt)
        if first_word_match is None:
            raise ValueError(
                f"Malformed credibility statement (no leading word): {stmt!r}")
        head = first_word_match.group(1).lower()

        if head in declaration_heads:
            # Collect names from the body after the keyword
            body = stmt[first_word_match.end():]
            names = name_regex.findall(body)
            # Filter reserved tokens just in case
            names = [n for n in names if n.lower() not in declaration_heads]
            declaration_heads[head].extend(names)
            continue

        # Otherwise, it must be an equation of the form LHS = RHS.
        if '=' not in stmt:
            raise ValueError(
                f"Credibility statement is neither a declaration nor an "
                f"equation: {stmt!r}")
        lhs, rhs = stmt.split('=', 1)
        equations.append({'lhs': lhs.strip(), 'rhs': rhs.strip()})

    # ---------- Validate ----------
    # (1) Every equation LHS is a simple declared var name (no lags,
    # no arithmetic). `cred_state = cred_state(-1) + ...` is fine --
    # the (-1) is on the RHS, not the LHS.
    declared_var_set = set(local_vars)
    lhs_names = []
    for eq in equations:
        lhs_match = re.match(r'\s*([a-zA-Z_]\w*)\s*$', eq['lhs'])
        if lhs_match is None:
            raise ValueError(
                f"Credibility equation LHS must be a bare variable name, "
                f"got: {eq['lhs']!r}")
        name = lhs_match.group(1)
        if name not in declared_var_set:
            raise ValueError(
                f"Credibility equation LHS {name!r} is not declared in "
                f"the block's `var` list. Declared: {local_vars}")
        lhs_names.append(name)

    # (2) Every declared var has exactly one equation.
    if sorted(lhs_names) != sorted(local_vars):
        missing = set(local_vars) - set(lhs_names)
        extra = set(lhs_names) - set(local_vars)
        parts = []
        if missing:
            parts.append(f"declared but no equation: {sorted(missing)}")
        if extra:
            parts.append(f"equation without declaration: {sorted(extra)}")
        raise ValueError(
            "Credibility block not square. " + "; ".join(parts))

    # (3) Every output name is declared in var.
    for o in outputs:
        if o not in declared_var_set:
            raise ValueError(
                f"Credibility output {o!r} is not declared in the "
                f"block's `var` list. Declared: {local_vars}")

    # (4) No duplicate declarations.
    for label, lst in [
        ("var", local_vars),
        ("input", inputs),
        ("output", outputs),
    ]:
        if len(set(lst)) != len(lst):
            raise ValueError(
                f"Duplicate names in credibility `{label}` declaration: {lst}")

    return {
        'local_vars' : local_vars,
        'inputs'     : inputs,
        'outputs'    : outputs,
        'equations'  : equations,
    }


def compile_credibility_block(spec, param_names=None, verbose=False):
    """Compile the output of parse_credibility_block into a straight-line
    JAX-traceable callable.

    The returned function has signature

        credibility_fn(inputs, prev_state, params) -> (outputs, new_state)

    where `inputs`, `prev_state`, `params`, `outputs`, `new_state` are all
    dicts keyed by variable / parameter name. The function evaluates each
    equation of the credibility block in declaration order, storing each
    LHS value as a local that subsequent equations can reference. It uses
    jax.numpy internally (via sympy.lambdify with a jnp module map) so
    the whole thing can be jit'd and differentiated through.

    Classification:
      * A local variable that appears with a (-1) lag anywhere in the
        block is a STATE variable. Its (-1) reference in an equation is
        resolved against the `prev_state` dict, and its current-period
        value is written to `new_state`.
      * A local variable that never appears with a (-1) lag is an
        ALGEBRAIC variable. It is computed from earlier locals / inputs /
        params and is only used later in the block (or as an output).

    Parameters are NOT declared inside the credibility block. They live in
    the global `parameters` block and are simply referenced by name in the
    equations. The `param_names` argument is the global parameter list;
    any RHS symbol not in local_vars, inputs, or known functions is looked
    up there.

    Compile-time validation:
      * Each equation's RHS may only reference locals that were defined
        strictly earlier in declaration order. Forward or circular local
        references are rejected.

    Parameters
    ----------
    spec : dict
        Output of `parse_credibility_block`.
    param_names : list of str or None
        Global parameter names (from the .mod file's `parameters` block).
        If None, any RHS symbol not in local_vars or inputs is assumed to
        be a parameter (auto-discovered).
    verbose : bool
        Print the classification and equation trace.

    Returns
    -------
    compiled : dict with keys:
        'fn'               : callable(inputs, prev_state, params) -> (out, new_state)
        'state_vars'       : list[str] -- locals with (-1) lag
        'algebraic_vars'   : list[str] -- locals without (-1) lag
        'input_vars'       : list[str]
        'output_vars'      : list[str]
        'used_params'      : list[str] -- global params actually referenced
        'equations_debug'  : list of dicts (for introspection / tests)
    """
    import jax.numpy as jnp

    local_vars   = spec['local_vars']
    input_vars   = spec['inputs']
    output_vars  = spec['outputs']
    equations    = spec['equations']

    # If no explicit param list given, we'll auto-discover params from
    # RHS symbols that aren't local vars, inputs, or known functions.
    param_names_set = set(param_names) if param_names else None

    # ---- Identify state vars (locals that appear with (-1) lag) ----
    lag_regex = re.compile(r'\b([a-zA-Z_]\w*)\s*\(\s*-1\s*\)')
    state_set = set()
    for eq in equations:
        for m in lag_regex.finditer(eq['rhs']):
            name = m.group(1)
            if name in local_vars:
                state_set.add(name)
    state_vars     = [v for v in local_vars if v in state_set]
    algebraic_vars = [v for v in local_vars if v not in state_set]

    # ---- Build sympy symbols ----
    # Inputs:      <name>           (comes from inputs dict at runtime)
    # Prev state:  <name>_m1        (comes from prev_state dict at runtime)
    # Locals:      <name>           (comes from the local scratch dict,
    #                                 populated as earlier equations run)
    # Parameters are handled dynamically: any RHS symbol not matching a
    # local var, input, prev-state, or known function is treated as a
    # parameter and looked up in the global params dict at runtime.
    sym_inputs = {v: sympy.Symbol(v)           for v in input_vars}
    sym_prev   = {v: sympy.Symbol(f"{v}_m1")   for v in state_vars}
    sym_local  = {v: sympy.Symbol(v)           for v in local_vars}

    # Name-collision guard: an input and a local must never share a name
    collisions = set(input_vars) & set(local_vars)
    if collisions:
        raise ValueError(
            f"Credibility block: name(s) appear in both input and var: "
            f"{sorted(collisions)}")

    # Collect all parameter symbols discovered across equations
    all_used_params = set()

    # ---- jnp module map for lambdify ----
    # `sigmoid` is exposed as a first-class function so .mod files can
    # write, e.g., `sigmoid(cred_sharpness*(cred_dev - cred_band))`
    # directly. This maps to jax.nn.sigmoid which is numerically stable
    # for large arguments (unlike 1/(1+exp(-x)) which overflows and
    # produces NaN gradients at high sharpness, e.g. BE 1304).
    try:
        import jax.nn as jnn
        _sigmoid_impl = jnn.sigmoid
    except ImportError:
        # Fall back to the naive form if jax.nn isn't available
        _sigmoid_impl = lambda x: 1.0 / (1.0 + jnp.exp(-x))

    jnp_module = {
        'exp':     jnp.exp,
        'log':     jnp.log,
        'sqrt':    jnp.sqrt,
        'sin':     jnp.sin,
        'cos':     jnp.cos,
        'tan':     jnp.tan,
        'Abs':     jnp.abs,
        'Max':     jnp.maximum,
        'Min':     jnp.minimum,
        'sigmoid': _sigmoid_impl,
    }

    # ---- Compile each equation ----
    compiled_eqs = []
    local_index = {v: i for i, v in enumerate(local_vars)}
    for eq_idx, eq in enumerate(equations):
        lhs = eq['lhs']
        # Substitute `name(-1)` -> `name_m1` in the RHS string, but ONLY
        # for locals that are state vars (others shouldn't have (-1) lags;
        # if they do, that is an error below).
        def _sub_lag(m):
            name = m.group(1)
            if name in state_set:
                return f"{name}_m1"
            # Lagged references to NON-state locals or to inputs are
            # not supported inside the credibility block (inputs have no
            # lag; locals are computed within one period). Flag them.
            raise ValueError(
                f"Credibility equation {eq_idx} ({lhs}): cannot take "
                f"lag (-1) of {name!r}. Only declared local variables "
                f"that themselves carry a (-1) lag somewhere in the "
                f"block are allowed as states.")
        rhs_str = lag_regex.sub(_sub_lag, eq['rhs'])

        # Build the local_dict for sympify. We include ALL known symbols
        # (inputs, prev-state, locals) plus math functions. Anything left
        # over as a free symbol after sympify is treated as a parameter
        # (if in param_names_set) or flagged as unknown.
        local_dict = {}
        local_dict.update(sym_inputs)
        local_dict.update({f"{v}_m1": sym_prev[v] for v in state_vars})
        local_dict.update(sym_local)
        # Math functions
        local_dict['exp']     = sympy.exp
        local_dict['log']     = sympy.log
        local_dict['sqrt']    = sympy.sqrt
        local_dict['Abs']     = sympy.Abs
        local_dict['Max']     = sympy.Max
        local_dict['Min']     = sympy.Min
        local_dict['sigmoid'] = sympy.Function('sigmoid')

        try:
            rhs_expr = sympy.sympify(rhs_str, locals=local_dict)
        except Exception as exc:
            raise ValueError(
                f"Credibility equation {eq_idx} ({lhs}): "
                f"could not sympify RHS {rhs_str!r}: {exc}")

        # Identify which symbols the expression uses.
        used_syms = rhs_expr.free_symbols
        used_inputs = [v for v in input_vars if sym_inputs[v] in used_syms]
        used_prev   = [v for v in state_vars if sym_prev[v]   in used_syms]
        used_locals = [v for v in local_vars if sym_local[v]  in used_syms]

        # Auto-discover parameters: any free symbol not in locals/inputs/prev
        known_syms = (set(sym_inputs.values())
                      | set(sym_prev.values())
                      | set(sym_local.values()))
        unknown_syms = used_syms - known_syms
        used_params = []
        for sym in sorted(unknown_syms, key=str):
            name = str(sym)
            if param_names_set is not None and name not in param_names_set:
                raise ValueError(
                    f"Credibility equation {eq_idx} ({lhs}): symbol "
                    f"{name!r} is not declared as a variable, input, or "
                    f"global parameter.")
            used_params.append(name)
            all_used_params.add(name)

        # Forward/self-reference guard: a local used on this RHS must be
        # defined strictly earlier in declaration order.
        for u in used_locals:
            if local_index[u] >= eq_idx:
                raise ValueError(
                    f"Credibility equation {eq_idx} ({lhs}): RHS "
                    f"references local var {u!r} which is defined at or "
                    f"after this equation. Forward/self-reference of "
                    f"current-period locals is not allowed.")

        # Build the argument list for lambdify (order matters).
        # For params, the symbols were auto-created by sympify so we
        # look them up by name from the expression's free symbols.
        param_sym_map = {str(s): s for s in unknown_syms}
        arg_list = (
            [sym_inputs[v] for v in used_inputs]
            + [sym_prev[v]   for v in used_prev]
            + [param_sym_map[v] for v in used_params]
            + [sym_local[v]  for v in used_locals]
        )
        fn = sympy.lambdify(arg_list, rhs_expr,
                            modules=[jnp_module, 'numpy'])

        compiled_eqs.append({
            'lhs':         lhs,
            'fn':          fn,
            'used_inputs': used_inputs,
            'used_prev':   used_prev,
            'used_params': used_params,
            'used_locals': used_locals,
            'rhs_expr':    rhs_expr,
        })

        if verbose:
            deps = []
            if used_inputs: deps.append(f"inputs={used_inputs}")
            if used_prev:   deps.append(f"prev_state={used_prev}")
            if used_params: deps.append(f"params={used_params}")
            if used_locals: deps.append(f"locals={used_locals}")
            print(f"  [{eq_idx}] {lhs} = {rhs_expr}    ({', '.join(deps)})")

    # ---- Build the straight-line callable ----
    output_var_set = set(output_vars)
    state_var_set  = set(state_vars)

    def credibility_fn(inputs, prev_state, params):
        """Evaluate the credibility block once.

        Parameters
        ----------
        inputs : dict[str, jnp.ndarray or float]
            Model variable values at current period t, keyed by name.
            Must contain at least every name in input_vars.
        prev_state : dict[str, jnp.ndarray or float]
            Credibility-state values at period t-1, keyed by name.
            Must contain at least every name in state_vars.
        params : dict[str, float]
            Credibility parameter values, keyed by name. Must contain
            at least every name in local_params.

        Returns
        -------
        outputs : dict[str, jnp.ndarray or float]
            Credibility outputs at period t (e.g., {'omega_pc': ...}).
        new_state : dict[str, jnp.ndarray or float]
            Credibility state at period t (e.g., {'cred_state': ...}).
        """
        locals_dict = {}
        for ceq in compiled_eqs:
            args = (
                [inputs[v]     for v in ceq['used_inputs']]
                + [prev_state[v] for v in ceq['used_prev']]
                + [params[v]     for v in ceq['used_params']]
                + [locals_dict[v] for v in ceq['used_locals']]
            )
            locals_dict[ceq['lhs']] = ceq['fn'](*args)

        outputs   = {v: locals_dict[v] for v in output_vars}
        new_state = {v: locals_dict[v] for v in state_vars}
        return outputs, new_state

    used_params_list = sorted(all_used_params)

    if verbose:
        print(f"compile_credibility_block:")
        print(f"  state_vars     ({len(state_vars)}):     {state_vars}")
        print(f"  algebraic_vars ({len(algebraic_vars)}): {algebraic_vars}")
        print(f"  used_params    ({len(used_params_list)}):    {used_params_list}")

    return {
        'fn':              credibility_fn,
        'state_vars':      state_vars,
        'algebraic_vars':  algebraic_vars,
        'input_vars':      input_vars,
        'output_vars':     output_vars,
        'used_params':     used_params_list,
        'equations_debug': compiled_eqs,
    }


# ---------------------------------------------------------------------------
# Pluggable credibility functions (LEGACY magic-symbol grammar)
# ---------------------------------------------------------------------------

def extract_credibility(mod_string):
    """Parse the credibility; ... end; block from a .mod file.

    Returns
    -------
    cred_spec : dict or None
        Keys: 'monitor', 'threshold', 'cred_init',
              'signal_expr', 'accumulation_expr',
              'signal_lag' (int, 0 or 1),
        Returns None if no credibility; block found.
    """
    # Strip block comments
    processed = re.sub(r'/\*.*?\*/', '', mod_string, flags=re.DOTALL)
    lines = processed.split('\n')
    cleaned = [re.sub(r'(//|%).*$', '', line).strip() for line in lines]
    processed = " ".join(cleaned)

    # Find credibility; ... end; block
    match = re.search(
        r'(?i)\bcredibility\b\s*;(.*?)\bend\b\s*;',
        processed, re.DOTALL
    )
    if not match:
        return None

    block = match.group(1)

    cred_spec = {}

    # Parse key: value; pairs
    for key in ['monitor', 'threshold', 'cred_init']:
        pattern = r'\b' + key + r'\s*:\s*([^;]+);'
        m = re.search(pattern, block, re.IGNORECASE)
        if m:
            val_str = m.group(1).strip()
            if key == 'monitor':
                cred_spec[key] = val_str
            else:
                try:
                    cred_spec[key] = float(val_str)
                except ValueError:
                    cred_spec[key] = val_str

    # Defaults
    cred_spec.setdefault('threshold', 0.5)
    cred_spec.setdefault('cred_init', 1.0)

    # Parse signal = <expr>;
    sig_match = re.search(r'\bsignal\s*=\s*([^;]+);', block, re.IGNORECASE)
    if sig_match:
        cred_spec['signal_expr'] = sig_match.group(1).strip()
    else:
        cred_spec['signal_expr'] = None

    # Parse accumulation = <expr>;
    acc_match = re.search(r'\baccumulation\s*=\s*([^;]+);', block, re.IGNORECASE)
    if acc_match:
        cred_spec['accumulation_expr'] = acc_match.group(1).strip()
    else:
        cred_spec['accumulation_expr'] = None

    # Detect lag usage: pi(-1) in signal expression
    if cred_spec.get('signal_expr'):
        cred_spec['signal_lag'] = 1 if 'pi(-1)' in cred_spec['signal_expr'] else 0
    else:
        cred_spec['signal_lag'] = 0

    return cred_spec


def compile_credibility_fn(cred_spec, param_names, param_defaults):
    """Compile signal and accumulation expressions into callables.

    Parameters
    ----------
    cred_spec : dict
        Output of extract_credibility().
    param_names : list of str
        All parameter names from the .mod file.
    param_defaults : dict
        Default parameter values.

    Returns
    -------
    compiled : dict with keys:
        'signal_fn'       : callable(pi, pi_lag, params_dict) -> float
        'accumulation_fn' : callable(cred, signal, miss, params_dict) -> float
        'monitor'         : str
        'threshold'       : float
        'cred_init'       : float
        'signal_lag'      : int (0 or 1)
    """
    compiled = {
        'monitor': cred_spec['monitor'],
        'threshold': cred_spec['threshold'],
        'cred_init': cred_spec['cred_init'],
        'signal_lag': cred_spec.get('signal_lag', 0),
    }

    # ---- Signal function ----
    sig_expr_str = cred_spec.get('signal_expr')

    if sig_expr_str is None:
        compiled['signal_fn'] = None
    else:
        # Replace pi(-1) with a symbol pi_lag, and plain pi with pi_curr
        expr_str = sig_expr_str.replace('pi(-1)', 'pi_lag')

        # Replace ^ with ** for sympy
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

        sig_numpy = sympy.lambdify(arg_list, sym_expr, modules='numpy')

        # Use default arg to capture current value (avoid closure late-binding)
        def signal_fn(pi_val, pi_lag_val, params_dict, pi_star=0.0,
                      _fn=sig_numpy, _params=list(used_params)):
            args = [pi_val, pi_lag_val, pi_star]
            args += [params_dict[p] for p in _params]
            return float(_fn(*args))

        compiled['signal_fn'] = signal_fn

    # ---- Accumulation function ----
    acc_expr_str = cred_spec.get('accumulation_expr')

    if acc_expr_str is None:
        compiled['accumulation_fn'] = None
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

        acc_numpy = sympy.lambdify(arg_list, sym_expr, modules='numpy')

        # Use default arg to capture current value (avoid closure late-binding)
        def accumulation_fn(cred_val, signal_val, miss_val, params_dict,
                            _fn=acc_numpy, _params=list(used_params)):
            args = [cred_val, signal_val, miss_val]
            args += [params_dict[p] for p in _params]
            return float(_fn(*args))

        compiled['accumulation_fn'] = accumulation_fn

    return compiled


def build_credibility_switching_fn(compiled, var_names, param_values,
                                   band=None):
    """Build a switching function from compiled credibility spec.

    Returns a callable with the same signature as build_switching_fn():
        switching_fn(u_path) -> regime_seq (ndarray of 0/1)
        switching_fn.cred_path available after call

    Parameters
    ----------
    compiled : dict
        Output of compile_credibility_fn().
    var_names : list of str
        Variable names from the model.
    param_values : dict
        Current parameter values (for evaluating signal/accumulation).
    band : float or None
        Tolerance band for the miss indicator.
    """
    monitor_var = compiled['monitor']
    pi_index = var_names.index(monitor_var)
    cred_threshold = compiled['threshold']
    cred_init = compiled['cred_init']
    signal_lag = compiled['signal_lag']
    signal_fn = compiled['signal_fn']
    accumulation_fn = compiled['accumulation_fn']

    use_isard_signal = (signal_fn is None)
    use_isard_accumulation = (accumulation_fn is None)

    if use_isard_accumulation:
        delta_up = param_values.get('delta_up', 0.05)
        delta_down = param_values.get('delta_down', 0.70)

    if band is None:
        band = param_values.get('epsilon_bar',
               param_values.get('band', 2.0))

    def switching_fn(u_path):
        T = u_path.shape[0]
        regime_seq = np.zeros(T, dtype=int)
        cred_path = np.zeros(T)
        signal_path = np.zeros(T)

        cred = cred_init

        for t in range(T):
            pi_t = u_path[t, pi_index]
            pi_lag = u_path[t - 1, pi_index] if t > 0 else 0.0

            miss = 1.0 if abs(pi_t) > band else 0.0

            if use_isard_signal:
                sig = 0.0
            else:
                if signal_lag == 1:
                    sig = signal_fn(pi_t, pi_lag, param_values)
                else:
                    sig = signal_fn(pi_t, pi_t, param_values)
            signal_path[t] = sig

            if use_isard_accumulation:
                if miss > 0.5:
                    cred = cred - delta_down * cred
                else:
                    cred = cred + delta_up * (1.0 - cred)
            else:
                cred = accumulation_fn(cred, sig, miss, param_values)

            cred = max(0.0, min(1.0, cred))
            cred_path[t] = cred

            regime_seq[t] = 1 if cred < cred_threshold else 0

        switching_fn.cred_path = cred_path
        switching_fn.signal_path = signal_path
        return regime_seq

    switching_fn.cred_path = None
    switching_fn.signal_path = None
    return switching_fn


# ---------------------------------------------------------------------------
# High-level: compile once / evaluate many (estimation-ready architecture)
# ---------------------------------------------------------------------------

def compile_two_regime_model(mod_string, verbose=False):
    """Parse a .mod file with a regimes and/or credibility block.

    For estimation: call this once, then call evaluate_two_regime_model()
    repeatedly with different param_overrides per MCMC draw.

    Parameters
    ----------
    mod_string : str
        Full .mod file content with regimes; and/or credibility; block.
    verbose : bool

    Returns
    -------
    compiled : dict with keys:
        'parsed' : dict from parse_mod() (lambdified functions)
        'regime_spec_raw' : dict from extract_regimes() (may be None)
        'credibility_spec' : dict from extract_credibility() (may be None)
        'credibility_compiled' : dict from compile_credibility_fn() (if applicable)
    """
    regime_spec_raw = extract_regimes(mod_string)
    cred_spec = extract_credibility(mod_string)

    parsed = parse_mod(mod_string, verbose=verbose)
    nocred = parsed.get('model_options', {}).get('nocredibility', False)

    if regime_spec_raw is None and cred_spec is None and not nocred:
        raise ValueError(
            "No 'regimes; ... end;' or 'credibility; ... end;' block found "
            "in .mod file. If this model is intentionally linear with no "
            "credibility regime, declare it with 'model(nocredibility);'.")

    result = {
        'parsed': parsed,
        'regime_spec_raw': regime_spec_raw,
        'credibility_spec': cred_spec,
        'nocredibility': nocred,
    }

    # Pre-compile credibility expressions (expensive sympy step)
    if cred_spec is not None:
        result['credibility_compiled'] = compile_credibility_fn(
            cred_spec,
            parsed['param_names'],
            parsed['param_defaults'],
        )
    else:
        result['credibility_compiled'] = None

    return result


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
    regime_spec = copy.deepcopy(compiled.get('regime_spec_raw'))

    # Build base params from defaults + overrides
    base_params = dict(parsed['param_defaults'])
    if param_overrides:
        base_params.update(param_overrides)

    nocred = compiled.get('nocredibility', False)

    if regime_spec is None:
        if nocred:
            # nocredibility model: single-regime, no parameter overrides.
            # M1 == M2 == the base parameter set; the switching function
            # will return all-zeros (always M1).
            regime_spec = {'M1_params': {}, 'M2_params': {}}
        else:
            # Legacy fallback: derive omega_H/omega_L from defaults.
            regime_spec = {
                'M1_params': {'omega': base_params.get('omega_H', 0.65)},
                'M2_params': {'omega': base_params.get('omega_L', 0.35)},
            }

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
    switch_numeric_keys = ['band', 'cred_threshold', 'delta_up', 'delta_down',
                           'cred_init', 'k_restore']
    for key in switch_numeric_keys:
        if key in regime_spec:
            regime_spec[key] = _resolve_param_value(
                regime_spec[key], base_params)

    # Check if we have a pluggable credibility specification
    cred_compiled = compiled.get('credibility_compiled')
    if cred_compiled is not None:
        # Use the pluggable credibility system
        band = None
        if regime_spec and 'band' in regime_spec:
            band = _resolve_param_value(regime_spec['band'], base_params)
        switching_fn = build_credibility_switching_fn(
            cred_compiled, parsed['var_names'], base_params, band=band)
    elif nocred:
        # nocredibility model: trivial switching function. M1 == M2,
        # so the regime sequence is irrelevant -- always return M1.
        def switching_fn(u_path):
            return np.zeros(u_path.shape[0], dtype=int)
        switching_fn.cred_path = None
        switching_fn.signal_path = None
    else:
        # Fall back to old hardcoded switching
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

    if cred_compiled is not None:
        info['credibility_compiled'] = cred_compiled

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


# ---------------------------------------------------------------------------
# NEW top-level dispatcher: parse_credibility_mod + compile_mod
# ---------------------------------------------------------------------------
#
# This replaces compile_two_regime_model / parse_two_regime_model for new-
# grammar .mod files. It reads model(target) to decide the compilation
# path, parses both the model; and credibility; blocks, and dispatches to
# the appropriate compile_* function.
#
# Usage:
#
#     result = compile_mod(open("open_soe_cred.mod").read())
#
# The result dict depends on the target:
#   linear -> same shape as parse_mod output (func_A/B/C/D, var_names, ...)
#   pwl    -> model matrices + credibility callable + regime info
#   nn     -> full nonlinear residual function with cred_state in state vec

def parse_credibility_mod(mod_string, verbose=False):
    """Parse a .mod file that may contain a credibility; ... end; block.

    This is the unified entry point that reads both blocks, compiles the
    credibility block (if present) into a JAX-traceable callable, and
    extracts the solution target from model(target).

    Parameters
    ----------
    mod_string : str
        Full .mod file content.
    verbose : bool
        Print progress info from sub-parsers.

    Returns
    -------
    parsed : dict with keys:
        'target'       : str ('linear', 'pwl', 'nn')
        'model'        : dict -- output of parse_mod (func_A/B/C/D, var_names, ...)
        'credibility'  : dict or None -- output of compile_credibility_block
        'cred_spec'    : dict or None -- output of parse_credibility_block (raw spec)
        'model_options': dict -- options from model(...) directive
        'mod_string'   : str  -- original .mod source
    """
    # ---- Extract target from model(target) ----
    _, model_options = extract_model_equations(mod_string)
    target = model_options.get('target', 'linear')

    # ---- Parse the credibility block (new grammar) first ----
    # We need to know the output variable names BEFORE calling parse_mod,
    # so we can pass them as coefficient_names. This ensures the sympy
    # matrices treat credibility outputs (e.g. omega_pc) as symbolic
    # coefficients, NOT as estimated parameters.
    cred_spec = parse_credibility_block(mod_string)

    # For model(pwl) and model(nn), credibility outputs are time-varying
    # coefficients in the model equations. For model(linear), the user
    # must declare them as parameters themselves.
    coeff_names = None
    if cred_spec is not None and target in ('pwl', 'nn'):
        coeff_names = cred_spec['outputs']

    # ---- Parse the model block (existing, proven at 4.3e-14 vs Dynare) ----
    # extract_declarations now strips the credibility; block internally,
    # so parse_mod is safe to call on .mod files with credibility blocks.
    model_parsed = parse_mod(mod_string, verbose=verbose,
                             coefficient_names=coeff_names)

    # If new grammar not found, try legacy path for backward compat
    if cred_spec is None:
        legacy_spec = extract_credibility(mod_string)
        if legacy_spec is not None and verbose:
            print("Note: using LEGACY credibility grammar (monitor:/signal=/accumulation=).")
            print("      Consider migrating to the new credibility; ... end; format.")

    # ---- Compile the credibility block if present ----
    # Pass the global parameter list so the compiler can validate that
    # every symbol in credibility equations is either a declared var,
    # input, or global parameter. Undeclared symbols → ValueError.
    cred_compiled = None
    if cred_spec is not None:
        global_param_names = model_parsed.get('param_names', [])
        cred_compiled = compile_credibility_block(
            cred_spec, param_names=global_param_names, verbose=verbose)

    # ---- Validate target vs credibility presence ----
    if target in ('pwl', 'nn') and cred_compiled is None and cred_spec is None:
        raise ValueError(
            f"model({target}) requires a credibility; ... end; block, "
            f"but none was found in the .mod file.")

    return {
        'target':        target,
        'model':         model_parsed,
        'credibility':   cred_compiled,
        'cred_spec':     cred_spec,
        'model_options': model_options,
        'mod_string':    mod_string,
    }


def compile_mod(mod_string, target=None, verbose=False):
    """Top-level entry: parse and compile a .mod file for the requested target.

    If `target` is None, the target is read from the model(target) directive
    in the .mod file. If `target` is given, it overrides whatever the file says.

    Parameters
    ----------
    mod_string : str
        Full .mod file content.
    target : str or None
        Override for the solution target ('linear', 'pwl', 'nn').
    verbose : bool
        Print progress info.

    Returns
    -------
    result : dict
        Structure depends on the resolved target:

        target='linear':
            Same as parse_mod output: func_A, func_B, func_C, func_D,
            var_names, shock_names, param_names, param_defaults, plus
            'credibility_fn' (the compiled credibility callable, or None).

        target='pwl':
            (Not yet implemented -- raises NotImplementedError)

        target='nn':
            (Not yet implemented -- raises NotImplementedError)
    """
    parsed = parse_credibility_mod(mod_string, verbose=verbose)
    resolved = target or parsed['target']

    if resolved == 'linear':
        return _compile_linear(parsed, verbose=verbose)
    elif resolved == 'pwl':
        return _compile_pwl(parsed, verbose=verbose)
    elif resolved == 'nn':
        return _compile_nn(parsed, verbose=verbose)
    else:
        raise ValueError(f"Unknown target: {resolved!r}. "
                         f"Valid targets: linear, pwl, nn.")


def _compile_linear(parsed, verbose=False):
    """Compile for target=linear.

    Simply returns the parse_mod output. The credibility block (if present)
    is IGNORED -- extract_declarations already strips it, so the model;
    equations are parsed as-is. If those equations reference a credibility
    output variable (like omega_pc) that isn't declared as a parameter,
    the parser will naturally fail at evaluation time. That's the expected
    behavior: the linear case uses a separate .mod file without a
    credibility block, or the user declares omega_pc as a fixed parameter.
    """
    result = dict(parsed['model'])
    result['target'] = 'linear'
    if verbose:
        print(f"_compile_linear: returning parse_mod output "
              f"({len(result['var_names'])} vars, "
              f"{len(result['shock_names'])} shocks)")
    return result


def _compile_pwl(parsed, verbose=False):
    """Compile for target=pwl (piecewise-linear with continuous credibility).

    Returns a compiled model where:
      - The A/B/C/D matrix functions take (*param_values, omega_pc) as
        arguments, with omega_pc as a separate coefficient (NOT an
        estimated parameter).
      - A build_ABC(params_dict, omega_pc) convenience function evaluates
        the matrices at given parameters and credibility forward weight.
      - The credibility callable advances cred_state one period given
        inputs (e.g. pi_cpi_yoy) and previous state.
      - solve_irf() runs the full Pontus outer loop with continuous
        omega path convergence.

    The PWL solver treats credibility as a continuous state variable:
    omega_pc = omega_low + (omega_high - omega_low) * cred_state, where
    cred_state evolves according to the credibility law of motion. At
    each period, the model is linearized at the current omega_pc value.
    The outer loop iterates until the omega path converges.

    Parameters
    ----------
    parsed : dict
        Output of parse_credibility_mod().
    verbose : bool

    Returns
    -------
    compiled : dict with keys:
        'target'           : 'pwl'
        'func_A/B/C/D/D_const' : callables taking (*param_values, omega_pc)
        'build_ABC'        : callable(params_dict, omega_pc) -> (A, B, C)
        'build_ABCD'       : callable(params_dict, omega_pc) -> (A, B, C, D, D_const)
        'var_names'        : list[str]
        'shock_names'      : list[str]
        'param_names'      : list[str] (estimated parameters only)
        'param_defaults'   : dict
        'coefficient_names': list[str] (e.g. ['omega_pc'])
        'credibility_fn'   : callable(inputs, prev_state, params) -> (out, new_state)
        'credibility_info' : dict with state_vars, input_vars, output_vars, used_params
        'solve_irf'        : callable(shock_name, size, T, params_dict, ...) -> result
    """
    model = parsed['model']
    cred = parsed['credibility']
    cred_spec = parsed['cred_spec']

    if cred is None:
        raise ValueError(
            "model(pwl) requires a credibility block, but none was compiled. "
            "Check that the .mod file has a credibility; ... end; block.")

    coeff_names = model.get('coefficient_names', [])
    if not coeff_names:
        raise ValueError(
            "model(pwl): no coefficient_names found in parsed model. "
            "The credibility output variables should appear as coefficients.")

    # ---- Matrix evaluation helpers ----
    func_A = model['func_A']
    func_B = model['func_B']
    func_C = model['func_C']
    func_D = model['func_D']
    func_D_const = model['func_D_const']
    param_names = model['param_names']
    n = len(model['var_names'])

    def _param_args(params_dict):
        """Extract ordered parameter values from a dict."""
        return [params_dict[p] for p in param_names]

    def build_ABC(params_dict, omega_pc):
        """Build A, B, C matrices at given parameters and omega_pc.

        Parameters
        ----------
        params_dict : dict
            Parameter values (estimated params only, e.g. beta, kappa, ...).
        omega_pc : float
            Current-period credibility forward weight.

        Returns
        -------
        A, B, C : ndarray (n, n)
        """
        args = _param_args(params_dict) + [omega_pc]
        A = np.array(func_A(*args), dtype=float)
        B = np.array(func_B(*args), dtype=float)
        C = np.array(func_C(*args), dtype=float)
        return A, B, C

    def build_ABCD(params_dict, omega_pc):
        """Build A, B, C, D, D_const at given parameters and omega_pc."""
        args = _param_args(params_dict) + [omega_pc]
        A = np.array(func_A(*args), dtype=float)
        B = np.array(func_B(*args), dtype=float)
        C = np.array(func_C(*args), dtype=float)
        D = np.array(func_D(*args), dtype=float)
        raw_const = np.atleast_1d(
            np.array(func_D_const(*args), dtype=float)).flatten()
        D_const = np.zeros(n)
        if raw_const.size == n:
            D_const[:] = raw_const
        elif raw_const.size == 1:
            D_const[:] = raw_const[0]
        return A, B, C, D, D_const

    # ---- Credibility stepping function (numpy-compatible) ----
    cred_fn = cred['fn']
    cred_state_vars = cred['state_vars']
    cred_input_vars = cred['input_vars']
    cred_output_vars = cred['output_vars']
    cred_used_params = cred['used_params']

    # Map input variables to column indices in u_path
    var_names = model['var_names']
    input_indices = {}
    for v in cred_input_vars:
        if v in var_names:
            input_indices[v] = var_names.index(v)
        else:
            raise ValueError(
                f"Credibility input '{v}' not in model var_names. "
                f"Available: {var_names}")

    def compute_omega_path(u_path, params_dict, cred_init=1.0):
        """Compute the continuous omega path from a simulated u_path.

        Runs the credibility law of motion forward through the path,
        extracting input variables (e.g. pi_cpi_yoy) from u_path at
        each period.

        Parameters
        ----------
        u_path : ndarray (T, n)
            Simulated model path (deviations from SS).
        params_dict : dict
            Full parameter values (must include credibility params).
        cred_init : float
            Initial credibility state (default 1.0 = full credibility).

        Returns
        -------
        omega_path : ndarray (T,)
            Time-varying forward weight.
        cred_path : ndarray (T,)
            Credibility state path.
        """
        T = u_path.shape[0]
        cred_path = np.zeros(T)
        omega_path = np.zeros(T)

        # Extract credibility parameters
        cred_params = {p: params_dict[p] for p in cred_used_params}

        # Initialize state
        prev_state = {}
        for v in cred_state_vars:
            prev_state[v] = np.array(
                cred_init if v == 'cred_state' else 0.0)

        for t in range(T):
            # Extract inputs from model path
            inputs = {v: np.array(float(u_path[t, input_indices[v]]))
                      for v in cred_input_vars}

            # Run one credibility step
            outputs, new_state = cred_fn(inputs, prev_state, cred_params)

            # Record
            cred_path[t] = float(new_state.get('cred_state', cred_init))
            omega_path[t] = float(outputs.get(cred_output_vars[0],
                                               params_dict.get('omega_high', 1.0)))

            prev_state = new_state

        return omega_path, cred_path

    def solve_irf(shock_name, size=1.0, T=60, params_dict=None,
                  cred_init=1.0, max_outer=50, tol=1e-8):
        """Compute an impulse response with continuous credibility dynamics.

        Uses the Pontus outer loop: backward recurse with per-period
        matrices A(omega_t), B(omega_t), C(omega_t), forward simulate,
        update the omega path from the credibility dynamics, repeat
        until the omega path converges.

        Parameters
        ----------
        shock_name : str
            Name of the shock (must be in shock_names).
        size : float
            Shock size (in units of the shock, not std devs).
        T : int
            IRF horizon.
        params_dict : dict or None
            Parameter values. If None, uses param_defaults.
        cred_init : float
            Initial credibility state.
        max_outer : int
            Maximum outer iterations.
        tol : float
            Convergence tolerance on max|omega_new - omega_old|.

        Returns
        -------
        result : dict with keys:
            'u_path'     : ndarray (T, n) -- IRF path
            'omega_path' : ndarray (T,) -- converged omega path
            'cred_path'  : ndarray (T,) -- converged credibility path
            'converged'  : bool
            'outer_iters': int
            'F_path'     : ndarray (T, n, n) -- policy functions
        """
        from necredpy.pontus import (solve_terminal,
                                     backward_recursion_continuous,
                                     simulate_forward_with_shocks)

        if params_dict is None:
            params_dict = dict(model['param_defaults'])

        shock_names = model['shock_names']
        if shock_name not in shock_names:
            raise ValueError(
                f"Unknown shock '{shock_name}'. "
                f"Available: {shock_names}")
        shock_idx = shock_names.index(shock_name)
        n_shocks = len(shock_names)

        # Build shock sequence: unit impulse at t=0
        epsilon = np.zeros((T, n_shocks))
        epsilon[0, shock_idx] = size

        # Terminal regime: full credibility (omega = omega_high)
        omega_high = params_dict.get('omega_high', 1.0)
        A_H, B_H, C_H = build_ABC(params_dict, omega_high)
        F_terminal, _ = solve_terminal(A_H, B_H, C_H)

        # Initial guess: constant omega = omega_high (full credibility)
        omega_path = np.full(T, omega_high)

        for outer in range(max_outer):
            # Build per-period matrices
            A_all = np.zeros((T, n, n))
            B_all = np.zeros((T, n, n))
            C_all = np.zeros((T, n, n))
            D_all = np.zeros((T, n, n_shocks))
            D_const_all = np.zeros((T, n))

            for t in range(T):
                A_t, B_t, C_t, D_t, Dc_t = build_ABCD(
                    params_dict, omega_path[t])
                A_all[t] = A_t
                B_all[t] = B_t
                C_all[t] = C_t
                D_all[t] = D_t
                D_const_all[t] = Dc_t

            # Backward recursion with per-period matrices
            F_path, E_path, Q_path = backward_recursion_continuous(
                A_all, B_all, C_all, D_const_all, F_terminal)

            # Forward simulate
            # Impact: Q_t maps into state space, D_t maps shocks
            u_path = simulate_forward_with_shocks(
                F_path, E_path, Q_path, D_all, epsilon)

            # Compute new omega path from credibility dynamics
            omega_new, cred_path = compute_omega_path(
                u_path, params_dict, cred_init=cred_init)

            # Check convergence
            max_diff = np.max(np.abs(omega_new - omega_path))
            if verbose:
                print(f"  PWL outer {outer + 1}: "
                      f"max|d_omega| = {max_diff:.2e}, "
                      f"cred range = [{cred_path.min():.4f}, "
                      f"{cred_path.max():.4f}]")

            if max_diff < tol:
                return {
                    'u_path': u_path,
                    'omega_path': omega_new,
                    'cred_path': cred_path,
                    'converged': True,
                    'outer_iters': outer + 1,
                    'F_path': F_path,
                }

            omega_path = omega_new

        return {
            'u_path': u_path,
            'omega_path': omega_path,
            'cred_path': cred_path,
            'converged': False,
            'outer_iters': max_outer,
            'F_path': F_path,
        }

    # ---- Assemble result ----
    if verbose:
        print(f"_compile_pwl: {len(var_names)} vars, "
              f"{len(model['shock_names'])} shocks, "
              f"coefficients = {coeff_names}")
        print(f"  credibility inputs:  {cred_input_vars}")
        print(f"  credibility outputs: {cred_output_vars}")
        print(f"  credibility states:  {cred_state_vars}")
        print(f"  credibility params:  {cred_used_params}")

    return {
        'target': 'pwl',
        'func_A': func_A,
        'func_B': func_B,
        'func_C': func_C,
        'func_D': func_D,
        'func_D_const': func_D_const,
        'build_ABC': build_ABC,
        'build_ABCD': build_ABCD,
        'var_names': var_names,
        'shock_names': model['shock_names'],
        'param_names': param_names,
        'param_defaults': model['param_defaults'],
        'coefficient_names': coeff_names,
        'credibility_fn': cred_fn,
        'credibility_info': {
            'state_vars': cred_state_vars,
            'input_vars': cred_input_vars,
            'output_vars': cred_output_vars,
            'used_params': cred_used_params,
        },
        'compute_omega_path': compute_omega_path,
        'solve_irf': solve_irf,
        'aux_resolution': model.get('aux_resolution', {}),
        'monitor_resolution': model.get('monitor_resolution', {}),
    }


def _compile_nn(parsed, verbose=False):
    """Compile for target=nn (full nonlinear residual with credibility merged).

    Not yet implemented.
    """
    raise NotImplementedError(
        "compile_nn is not yet implemented. Use target='linear' for now.")
