"""Model-agnostic NN-PEA helpers for parser-driven Maliar solvers.

This module exposes the small set of primitives needed to wire a neural
network into a model produced by `compile_jax_model`. It does NOT know
anything about the credibility paper specifically -- everything is
discovered from the parser output.

Main entry points
-----------------
discover_forward_vars(model, params=None)
    Identify which variables appear at lead +1 in any equation.
    Auxiliary lead variables (created by the parser for `x(+k)` with
    k >= 2) are included automatically. The result is the canonical
    list of variables the NN must approximate as expectations.

split_state(model, params=None)
    Partition the full state into (predetermined, forward-looking).

resolve_state(u_lag, E_fwd_next, eps_t, model, params, omega_t=None)
    Solve `B u_t = -A u_{t-1} - C u_{t+1} - D eps_t` for u_t in one
    linear solve. The NN-supplied expectation `E_fwd_next` is dropped
    into the forward slots of u_{t+1}. Predetermined slots have zero
    coefficient in C by construction so they do not appear.

equation_residuals(u_lag, u_t, u_next, eps_t, model, params, omega_t=None)
    Compute `A u_lag + B u_t + C u_next + D eps_t`. At a correctly
    resolved step this is zero (modulo float roundoff). Useful as a
    sanity check during NN training.

update_credibility(u_lag, cred_lag, model, params)
    Advance the credibility stock one step using the parser-compiled
    credibility law of motion (BE 1304 or whatever the .mod file
    declares). Returns (cred_t, omega_t) where omega_t is the
    credibility-scaled forward weight.

simulate_one_step(u_lag, cred_lag, eps_t, net_apply, model, params)
    Convenience wrapper. Computes (cred_t, omega_t) from the lagged
    monitor, applies the NN to the lagged state to get
    E_t[u_{t+1}[fwd]], then resolves u_t. Returns (u_t, cred_t).

Design notes
------------
- Everything is JAX-traceable. discover_forward_vars and split_state are
  called once at setup time and produce static int arrays that get baked
  into JIT'd training loops.
- The omega override pattern (passing omega_t as a separate argument
  instead of mutating the params dict) works under JIT because the dict
  mutation happens at tracing time, not execution time.
- The full square system B u_t = RHS is solved in one shot. This is
  equivalent to the branch script's manual resolve_fwd / transition
  split, but model-agnostic: any .mod file works.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Forward-variable discovery
# ---------------------------------------------------------------------------

def _default_params(model, params=None):
    """Build a complete params dict, falling back to model defaults."""
    if params is not None:
        p = dict(params)
    else:
        p = dict(model["param_defaults"])
    p.setdefault("tau", 0.2)
    for s in model["shock_names"]:
        p.setdefault("sigma_" + s, 0.5)
    if "omega_H" in p and "omega" not in p:
        p["omega"] = float(p["omega_H"])
    return p


def discover_forward_vars(model, params=None, tol=1e-10):
    """Identify forward-looking variables from the parser's C matrix.

    A variable v_j is forward-looking iff column-j of C is non-zero in
    at least one equation. By construction, auxiliary lead variables
    (created by the parser for k >= 2 leads via aux chains) are also
    in C and so are picked up automatically. This means a model with
    `x(+3)` produces an NN with three outputs covering the 1-step,
    2-step, and 3-step expectations of x.

    Parameters
    ----------
    model : dict
        Output of compile_jax_model().
    params : dict or None
        Parameter values to evaluate C at. If None, the model defaults
        are used. The result is robust to params because the sparsity
        pattern of C does not depend on parameter values for any model
        the parser produces (parameters only scale existing entries).
    tol : float
        Numerical tolerance for "non-zero" classification.

    Returns
    -------
    fwd_idx : jnp.ndarray of int32, shape (n_fwd,)
        Indices into model['var_names'] for forward-looking variables.
        Returned as a JAX array so it can be used directly inside JIT.
    fwd_names : list of str
        Variable names corresponding to fwd_idx.
    """
    p = _default_params(model, params)
    coeff_names = model.get("coefficient_names", [])
    if coeff_names:
        # New grammar: pass omega at full credibility for sparsity check
        omega_high = float(p.get("omega_high", 1.0))
        _, _, C = model["build_ABC"](p, omega_high)
    else:
        _, _, C = model["build_ABC"](p)
    C_np = jnp.asarray(C)
    n = C_np.shape[1]
    raw = []
    for j in range(n):
        if bool(jnp.any(jnp.abs(C_np[:, j]) > tol)):
            raw.append(j)
    fwd_idx = jnp.array(raw, dtype=jnp.int32)
    fwd_names = [model["var_names"][j] for j in raw]
    return fwd_idx, fwd_names


def split_state(model, params=None, tol=1e-10):
    """Partition var_names into (predetermined, forward-looking).

    Returns
    -------
    pred_idx, fwd_idx : jnp.ndarray of int32
    pred_names, fwd_names : list of str
    """
    fwd_idx, fwd_names = discover_forward_vars(model, params, tol=tol)
    n = model["n_vars"]
    fwd_set = set(int(j) for j in fwd_idx)
    pred_raw = [j for j in range(n) if j not in fwd_set]
    pred_idx = jnp.array(pred_raw, dtype=jnp.int32)
    pred_names = [model["var_names"][j] for j in pred_raw]
    return pred_idx, fwd_idx, pred_names, fwd_names


# ---------------------------------------------------------------------------
# Linear-solve forward step
# ---------------------------------------------------------------------------

def resolve_state(u_lag, E_fwd_next, eps_t, model, params,
                  fwd_idx=None, omega_t=None):
    """Solve the model equations at time t in one linear solve.

    System (Pontus form):
        A u_{t-1} + B u_t + C u_{t+1} + D eps_t = 0
        =>  B u_t = -(A u_{t-1} + C u_{t+1} + D eps_t)

    The NN supplies u_{t+1} only for the forward-looking slots; the
    predetermined slots are set to zero in u_{t+1} but their columns
    in C are zero by construction, so they do not contribute to the
    right-hand side.

    Parameters
    ----------
    u_lag : (n,) jnp array
        Full state vector at t-1.
    E_fwd_next : (n_fwd,) jnp array
        NN output: E_t[u_{t+1}[fwd_idx]].
    eps_t : (n_shocks,) jnp array
        Structural shock realisation at t.
    model : dict from compile_jax_model
    params : dict
        Parameter dict (must contain every name in model['param_names']).
    fwd_idx : jnp.ndarray or None
        Forward-variable indices. If None, computed from
        discover_forward_vars (slow path; pass it pre-computed inside
        a training loop).
    omega_t : float or None
        If given, override params['omega'] with this value before
        building A, B, C, D. Used to flow time-varying credibility
        through the matrices without rebuilding params from scratch
        every step.

    Returns
    -------
    u_t : (n,) jnp array
        Full state vector at t.
    """
    if fwd_idx is None:
        fwd_idx, _ = discover_forward_vars(model, params)

    coeff_names = model.get("coefficient_names", [])
    if coeff_names and omega_t is not None:
        # New grammar: omega_pc is a coefficient, not a parameter
        A, B, C = model["build_ABC"](params, omega_t)
        D = model["build_D"](params, omega_t)
    elif omega_t is not None:
        # Legacy: stuff omega into params dict
        p = dict(params)
        p["omega"] = omega_t
        A, B, C = model["build_ABC"](p)
        D = model["build_D"](p)
    else:
        A, B, C = model["build_ABC"](params)
        D = model["build_D"](params)

    n = u_lag.shape[0]
    u_next = jnp.zeros(n).at[fwd_idx].set(E_fwd_next)

    # Note the sign convention: the parser stores D as -(coeff of e_t),
    # so the model equation is A u_lag + B u_t + C u_next - D eps_t = 0,
    # i.e., B u_t = -A u_lag - C u_next + D eps_t.
    rhs = -A @ u_lag - C @ u_next + D @ eps_t
    u_t = jnp.linalg.solve(B, rhs)
    return u_t


def equation_residuals(u_lag, u_t, u_next, eps_t, model, params,
                       omega_t=None):
    """Evaluate A u_lag + B u_t + C u_next + D eps_t.

    Returns the equation-by-equation residual vector. At a correctly
    resolved (u_lag, u_t, u_next, eps_t) this is zero up to float
    roundoff. Useful as a sanity check during training.
    """
    coeff_names = model.get("coefficient_names", [])
    if coeff_names and omega_t is not None:
        A, B, C = model["build_ABC"](params, omega_t)
        D = model["build_D"](params, omega_t)
    elif omega_t is not None:
        p = dict(params)
        p["omega"] = omega_t
        A, B, C = model["build_ABC"](p)
        D = model["build_D"](p)
    else:
        A, B, C = model["build_ABC"](params)
        D = model["build_D"](params)
    # Sign convention: parser stores D as -(coeff of e_t), so the
    # equation residual is A u_lag + B u_t + C u_next - D eps_t.
    return A @ u_lag + B @ u_t + C @ u_next - D @ eps_t


# ---------------------------------------------------------------------------
# Credibility law of motion (reuses parser-compiled scan)
# ---------------------------------------------------------------------------

def update_credibility(monitor_lag, cred_lag, model, params):
    """Advance the credibility stock one step.

    Calls the parser-compiled credibility scan (the same one used by
    the inversion filter) on a single time step. The signal in the
    .mod file may use the lagged monitor (BE 1304 default) or the
    contemporaneous one; the parser tracks this via signal_lag.

    Models declared with `model(nocredibility);` short-circuit this
    function: cred is fixed at 1.0 and omega at params['omega'] (or
    1.0 if absent). The lagged monitor and cred_lag are ignored.

    Parameters
    ----------
    monitor_lag : scalar
        Value of the credibility monitor at the relevant lag (typically
        u_{t-1}[monitor_index]).
    cred_lag : scalar
        Credibility stock at t-1.
    model : dict from compile_jax_model
    params : dict

    Returns
    -------
    cred_t : scalar
    omega_t : scalar
        Credibility-scaled forward weight,
        omega_t = omega_L + (omega_H - omega_L) * cred_t.
    """
    cred_new = model.get("credibility_new")
    cred_jax = model.get("credibility_jax")

    if cred_new is None and cred_jax is None:
        # No credibility regime in the model. Treat as fully linear:
        # cred is constant 1, omega is the constant value in params
        # (or 1.0 if the model has no omega parameter at all).
        omega_const = float(params.get("omega",
                            params.get("omega_high", 1.0)))
        return jnp.array(1.0), jnp.array(omega_const)

    from necredpy.jax_model import _build_cred_scan_fn

    cred_scan_fn, init_template = _build_cred_scan_fn(model, params)

    if cred_new is not None:
        # New grammar: carry is a dict {'cred_state': ...}
        carry = {v: (cred_lag if v == 'cred_state' else jnp.array(0.0))
                 for v in cred_new['state_vars']}
        new_carry, (cred_t, omega_t) = cred_scan_fn(carry, monitor_lag)
    else:
        # Legacy: carry is (cred, pi_lag) tuple
        carry = (cred_lag, monitor_lag)
        new_carry, (cred_t, omega_t) = cred_scan_fn(carry, monitor_lag)

    return cred_t, omega_t


# ---------------------------------------------------------------------------
# One-step simulation (for training and validation)
# ---------------------------------------------------------------------------

def simulate_one_step(u_lag, cred_lag, eps_t, net_apply, model, params,
                      fwd_idx, monitor_index, n_inner=1):
    """Advance the model one step under the NN policy.

    Algorithm with fixed-point refinement:
      1. Compute cred_t, omega_t from the lagged monitor and cred_lag.
      2. Stale forecast: E0 = net(u_lag) (conditioned on the lagged
         state, NOT u_t). Resolve u_t_guess.
      3. Refinement: re-evaluate NN at the resolved guess and re-solve.
         Repeat n_inner times.

    Why the refinement matters: the equation at time t needs E_t[u_{t+1}]
    in the C-slot. The NN approximates E[next | current state]. The
    "current state" should be u_t (post-shock), but to compute u_t we
    need the C-slot value -- chicken-and-egg. The refinement breaks
    this by using a stale guess on the first iteration and improving
    it. Empirically, n_inner=1 (one refinement) reduces the bias from
    O(0.1) per panel down to machine precision when the NN is at the
    linear solution.

    Parameters
    ----------
    u_lag : (n,) jnp array
    cred_lag : scalar
    eps_t : (n_shocks,) jnp array
    net_apply : callable
        Pure function net_apply(u) -> (n_fwd,) array.
    model : dict from compile_jax_model
    params : dict
    fwd_idx : (n_fwd,) jnp array
    monitor_index : int
    n_inner : int
        Number of inner refinement iterations. 0 = pure stale (fastest,
        biased). 1 = one refinement (good balance). 2+ = diminishing
        returns. Default 1.

    Returns
    -------
    u_t : (n,) jnp array
    cred_t : scalar
    """
    monitor_lag = u_lag[monitor_index]
    cred_t, omega_t = update_credibility(monitor_lag, cred_lag, model, params)

    # First pass: stale NN at u_lag
    E_fwd_next = net_apply(u_lag)
    u_t = resolve_state(u_lag, E_fwd_next, eps_t, model, params,
                         fwd_idx=fwd_idx, omega_t=omega_t)

    # Refinements: re-evaluate NN at the (improving) current state
    for _ in range(n_inner):
        E_fwd_next = net_apply(u_t)
        u_t = resolve_state(u_lag, E_fwd_next, eps_t, model, params,
                             fwd_idx=fwd_idx, omega_t=omega_t)

    return u_t, cred_t
