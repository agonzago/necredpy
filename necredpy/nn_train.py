"""Model-agnostic Maliar online training of an NN-PEA solution.

Pairs with `necredpy.nn_solver` -- this module supplies the training
loop, the architecture, and the loss; `nn_solver` supplies the
parser-driven primitives that turn a parsed model into the system
of equations the NN has to satisfy.

Method: Maliar, Maliar & Winant (2021, JME), online deep learning with
the product-of-residuals (two-draw) loss.

  At each training step at calendar time t:
    1. Read the current state x_t (from the simulation carry).
    2. Compute the NN's forecast of next-period forward variables:
           E_t  =  net(x_t)        in R^{n_fwd}
    3. Draw two independent shock vectors eps_a, eps_b ~ N(0, diag(stds)).
    4. Update credibility: cred_{t+1}, omega_{t+1} from x_t[monitor].
    5. Resolve x_{t+1} under each path:
           x_{t+1}^a  =  resolve_state(x_t, E_t, eps_a, omega_{t+1})
           x_{t+1}^b  =  resolve_state(x_t, E_t, eps_b, omega_{t+1})
       (Same E_t is used in both -- it is the prediction the NN makes
        before seeing the shock realisation. Using the same E_t is what
        keeps the two paths "conditional on x_t" so the two-draw trick
        works.)
    6. Form residuals
           R_a = x_{t+1}^a[fwd] - E_t
           R_b = x_{t+1}^b[fwd] - E_t
    7. Loss = mean(R_a * R_b).
       Because eps_a and eps_b are independent given x_t,
           E[R_a R_b | x_t] = E[R_a | x_t] * E[R_b | x_t]
                           = (E[x_{t+1}[fwd] | x_t] - E_t)^2
                           = bias of NN squared.
       So this is an unbiased estimator of the squared bias.
    8. Gradient step on net.
    9. Advance the simulation: carry = (x_{t+1}^a, cred_{t+1}, ...)

Architecture: Equinox MLP with 2 hidden layers x 64 tanh by default.
The whole training run lives inside one lax.scan, JIT-compiled with
eqx.filter_jit.

Usage
-----
    from necredpy.nn_train import ExpectNet, train

    net_init = ExpectNet(key=jrandom.PRNGKey(0),
                         n_in=model['n_vars'],
                         n_out=len(fwd_idx))
    net_trained, history = train(
        model, params,
        net_init,
        n_steps=20000,
        seed=42,
    )
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom

from necredpy.nn_solver import (
    discover_forward_vars,
    resolve_state,
    update_credibility,
)


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

try:
    import equinox as eqx
except ImportError as exc:
    raise ImportError(
        "necredpy.nn_train requires equinox. Install with: pip install equinox optax"
    ) from exc

try:
    import optax
except ImportError as exc:
    raise ImportError(
        "necredpy.nn_train requires optax. Install with: pip install optax"
    ) from exc


class ExpectNet(eqx.Module):
    """MLP that maps the full state vector to forward-variable expectations.

    Default: 2 hidden layers x 64 units, tanh activations, linear output.
    Same architecture as the branch script (Brazil + credibility paper),
    but with the input/output dimensions discovered from the parser
    rather than hardcoded.
    """
    layers: list

    def __init__(self, key, n_in, n_out, hidden=64):
        keys = jrandom.split(key, 3)
        self.layers = [
            eqx.nn.Linear(n_in, hidden, key=keys[0]),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            eqx.nn.Linear(hidden, n_out, key=keys[2]),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jnp.tanh(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Loss and one training step
# ---------------------------------------------------------------------------

def maliar_loss(net, u_t, eps_a, eps_b, cred_t, model, params,
                fwd_idx, monitor_index, omega_next, n_inner=1):
    """Two-draw product-of-residuals loss for one training observation.

    With n_inner inner refinement iterations: each refinement re-evaluates
    NN at the resolved next-period state and re-solves. n_inner=0 is the
    pure stale-NN approach; n_inner=1 reduces the staleness bias by ~10x
    in the linear case (verified empirically).

    Parameters
    ----------
    net : ExpectNet
    u_t : (n,) jnp array
    eps_a, eps_b : (n_shocks,) jnp arrays
    cred_t : scalar
    model, params : as in nn_solver
    fwd_idx : (n_fwd,) jnp int array
    monitor_index : int
    omega_next : scalar
    n_inner : int
        Number of inner refinement iterations per path. Default 1.

    Returns
    -------
    loss : scalar
    u_next_a : (n,) jnp array
        State at t+1 under the eps_a path (used to advance the carry).
    """
    # NN prediction of E_t[u_{t+1}[fwd]] -- this is the value the NN is
    # trained to make equal to the conditional expectation at u_t.
    E_t = net(u_t)

    # First-pass resolve under each shock with the stale NN.
    u_next_a = resolve_state(u_t, E_t, eps_a, model, params,
                              fwd_idx=fwd_idx, omega_t=omega_next)
    u_next_b = resolve_state(u_t, E_t, eps_b, model, params,
                              fwd_idx=fwd_idx, omega_t=omega_next)

    # Inner refinements: re-evaluate NN at the resolved states and
    # re-solve. This is the same fix as simulate_one_step but applied
    # during training.
    for _ in range(n_inner):
        E_a_refined = net(u_next_a)
        E_b_refined = net(u_next_b)
        u_next_a = resolve_state(u_t, E_a_refined, eps_a, model, params,
                                  fwd_idx=fwd_idx, omega_t=omega_next)
        u_next_b = resolve_state(u_t, E_b_refined, eps_b, model, params,
                                  fwd_idx=fwd_idx, omega_t=omega_next)

    # Forward-variable realisations after refinement
    actual_a = u_next_a[fwd_idx]
    actual_b = u_next_b[fwd_idx]

    # Two-draw residuals: NN(u_t) compared to two independent
    # realisations of the forward variables at t+1.
    R_a = actual_a - E_t
    R_b = actual_b - E_t

    loss = jnp.mean(R_a * R_b)
    return loss, u_next_a


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, params, net_init,
          n_steps=20000,
          learning_rate=1e-3,
          shock_stds=None,
          monitor_index=None,
          seed=0,
          report_every=1000,
          warm_start_state=None,
          reset_every=100,
          n_inner=1):
    """Train an ExpectNet via online Maliar with the two-draw loss.

    Parameters
    ----------
    model : dict from compile_jax_model
    params : dict
        Parameter values (must include every name in model['param_names']).
    net_init : eqx.Module
        The initial NN. Use ExpectNet(key, n_in=model['n_vars'],
        n_out=len(fwd_idx)) for the default architecture.
    n_steps : int
        Number of online training steps.
    learning_rate : float
        Adam learning rate.
    shock_stds : (n_shocks,) array or None
        Per-shock standard deviations. If None, uses params['sigma_<name>']
        for each shock in model['shock_names'].
    monitor_index : int or None
        Index of the credibility monitor variable in u. If None, derived
        from model['credibility_jax']['monitor'] via var_names.
    seed : int
    report_every : int
        Print loss every N steps. Set to 0 to disable.
    warm_start_state : (n,) array or None
        Initial state of the simulation. Defaults to the steady state
        (zeros).
    reset_every : int
        Reset the simulation state to (zero, cred=1) every N steps.
        Without resets, the cumulative random walk drives the
        simulation away from the steady state and the NN learns the
        drift's neighbourhood instead of the steady-state neighbourhood
        (verified empirically: 20k steps without resets gave 47pp IRF
        errors). reset_every=100 keeps the simulation centred while
        still letting it explore +/- 5pp around zero between resets.
        Set to 0 to disable resets (Maliar pure online).

    Returns
    -------
    net : eqx.Module
        Trained NN.
    history : dict with keys 'loss' (1-d array of losses per step),
              'final_state' (the last simulation state).
    """
    n = model["n_vars"]
    n_shocks = model["n_shocks"]
    var_names = model["var_names"]

    # Discover forward variables once. Bake into the JIT loop.
    fwd_idx, fwd_names = discover_forward_vars(model, params)
    n_fwd = int(fwd_idx.shape[0])
    if int(net_init(jnp.zeros(n)).shape[0]) != n_fwd:
        raise ValueError(
            "ExpectNet output dim (" + str(net_init(jnp.zeros(n)).shape[0])
            + ") does not match number of forward variables ("
            + str(n_fwd) + ")"
        )

    # Default monitor index
    if monitor_index is None:
        cred_jax = model.get("credibility_jax")
        if cred_jax and "monitor" in cred_jax:
            monitor_index = var_names.index(cred_jax["monitor"])
        else:
            raise ValueError(
                "monitor_index not given and no credibility block in model"
            )

    # Default shock stds
    if shock_stds is None:
        shock_stds = jnp.array([
            float(params["sigma_" + s]) for s in model["shock_names"]
        ])
    else:
        shock_stds = jnp.asarray(shock_stds, dtype=jnp.float64)

    # Default warm start: steady state
    if warm_start_state is None:
        warm_start_state = jnp.zeros(n)
    else:
        warm_start_state = jnp.asarray(warm_start_state, dtype=jnp.float64)

    # Optimiser
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(net_init, eqx.is_inexact_array))

    # ----- One training step (single Maliar update + advance) -----
    def step_fn(carry, key):
        u_t, cred_t, net, opt_state = carry

        # Update credibility from u_t's monitor (uses the same scan as the
        # inversion filter -- consistent with the PWL solver).
        cred_next, omega_next = update_credibility(
            u_t[monitor_index], cred_t, model, params)

        # Two independent shock draws
        k_a, k_b = jrandom.split(key)
        eps_a = jrandom.normal(k_a, (n_shocks,)) * shock_stds
        eps_b = jrandom.normal(k_b, (n_shocks,)) * shock_stds

        # Loss + gradient (filter_grad handles eqx.Module pytrees)
        def loss_only(net_arg):
            loss, u_next_a = maliar_loss(
                net_arg, u_t, eps_a, eps_b, cred_t, model, params,
                fwd_idx, monitor_index, omega_next, n_inner=n_inner)
            return loss, u_next_a

        (loss_val, u_next_a), grads = eqx.filter_value_and_grad(
            loss_only, has_aux=True)(net)

        updates, opt_state = optimizer.update(grads, opt_state, net)
        net = eqx.apply_updates(net, updates)

        new_carry = (u_next_a, cred_next, net, opt_state)
        return new_carry, loss_val

    init_carry = (warm_start_state, jnp.array(1.0), net_init, opt_state)

    # Run the training loop. Use lax.scan if we want it JIT-compiled in
    # one go; for moderate n_steps a Python loop with periodic reporting
    # is more user-friendly.
    if report_every and report_every > 0:
        # Python loop with progress reporting -- slower but easier to
        # diagnose. JIT only the per-step body.
        step_jit = eqx.filter_jit(step_fn)
        carry = init_carry
        losses = []
        key = jrandom.PRNGKey(seed)
        for i in range(n_steps):
            key, subkey = jrandom.split(key)
            carry, loss_val = step_jit(carry, subkey)
            losses.append(float(loss_val))

            # Periodic reset: bring simulation back near steady state.
            if reset_every > 0 and (i + 1) % reset_every == 0:
                u_t, cred_t, net_, opt_state_ = carry
                carry = (warm_start_state, jnp.array(1.0), net_, opt_state_)

            if (i + 1) % report_every == 0:
                ema = sum(losses[-report_every:]) / report_every
                print("  step " + str(i + 1).rjust(6) + " / "
                      + str(n_steps) + "  loss(EMA " + str(report_every)
                      + ") = " + ("%.4e" % ema))
        losses = jnp.array(losses)
    else:
        # Pure scan -- one big JIT compile, fastest for many steps.
        # NOTE: pure scan path does NOT support periodic resets cleanly
        # (would need a static reset_every). For production runs use
        # report_every > 0.
        keys = jrandom.split(jrandom.PRNGKey(seed), n_steps)
        carry, losses = jax.lax.scan(step_fn, init_carry, keys)

    u_final, cred_final, net_final, _ = carry
    history = {
        "loss": losses,
        "final_state": u_final,
        "final_cred": cred_final,
        "fwd_idx": fwd_idx,
        "fwd_names": fwd_names,
        "monitor_index": monitor_index,
        "shock_stds": shock_stds,
    }
    return net_final, history
