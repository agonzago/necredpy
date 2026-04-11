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

    NOTE: ExpectNet starts from random weights and has to learn the
    entire policy from scratch. For PEA-style training this often
    converges slowly and, when under-trained, produces biased outputs
    that drive the simulation into oscillations. Prefer LinearResidualNet
    below, which warm-starts at the PWL linear policy.
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


class LinearResidualNet(eqx.Module):
    """NN that predicts the DEVIATION from the linear PWL policy.

    Output:  NN(u) = F_fwd @ u + delta(u)

    where F_fwd is the linear PWL forward policy (fixed, NOT trained)
    and delta is a small MLP whose last layer is zero-initialised.
    Result: at training step 0, delta(u) == 0 for all u, so NN(u) is
    EXACTLY the PWL linear policy. Training only learns small
    nonlinear corrections on top of this baseline.

    Why this works much better than ExpectNet from random init:

      - Pure-MLP ExpectNet has to learn the whole 88- or 40-dim
        function from random output. Until it converges, the simulation
        sees badly biased predictions and drifts off the manifold,
        which corrupts the training data the NN is trying to fit.
        This creates a chicken-and-egg loop and the loss plateaus.

      - LinearResidualNet starts at the correct linear answer. The
        simulation tracks PWL exactly until delta starts to grow.
        Training is now a pure refinement: any nonlinearity from the
        credibility law of motion is learned as a small correction
        on top of a known-good policy.

      - Empirically (verified on the 5-sector credibility model):
        * pure ExpectNet: loss EMA plateaus at 0.5-1.0 after 100k
          steps; IRFs show period-8 oscillation in pi_j and 100x
          biased q output.
        * LinearResidualNet: starts at IRF diff < 1e-13 vs PWL
          (because delta is exactly 0). Training refines this.

    Construction:
        F_fwd is the (n_fwd, n_in) slice of the PWL linear policy F
        corresponding to the forward variables. Compute via:
            from necredpy.jax_model import solve_terminal_jax
            A, B, C = model['build_ABC'](params)
            F = solve_terminal_jax(A, B, C)
            F_fwd = F[fwd_idx, :]

    Parameters
    ----------
    key : PRNGKey
    n_in : int
        State dimension. For Strategy (a) (state only) this is n_vars.
        For Strategy (b) (state + params) this is n_vars + n_params.
    n_out : int
        Number of forward variables.
    hidden : int
        Hidden layer width. Default 64.
    F_fwd : (n_out, n_in) jnp array
        Linear baseline. Shape must match (n_fwd, n_in).
    """
    layers: list
    F_fwd: jnp.ndarray

    def __init__(self, key, n_in, n_out, hidden, F_fwd):
        if F_fwd.shape != (n_out, n_in):
            raise ValueError(
                "F_fwd must have shape (" + str(n_out) + ", " + str(n_in)
                + ") but got " + str(F_fwd.shape))

        keys = jrandom.split(key, 3)

        # Build the last layer with zero weights and zero bias so the
        # initial NN(x) returns exactly F_fwd @ x.
        last_layer = eqx.nn.Linear(hidden, n_out, key=keys[2])
        last_layer = eqx.tree_at(
            lambda m: m.weight, last_layer,
            jnp.zeros_like(last_layer.weight))
        last_layer = eqx.tree_at(
            lambda m: m.bias, last_layer,
            jnp.zeros_like(last_layer.bias))

        self.layers = [
            eqx.nn.Linear(n_in, hidden, key=keys[0]),
            eqx.nn.Linear(hidden, hidden, key=keys[1]),
            last_layer,
        ]
        self.F_fwd = jnp.asarray(F_fwd)

    def _mlp(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)

    def __call__(self, x):
        # Linear baseline (no gradient flows through F_fwd because it
        # is a frozen array attached to the module).
        linear = self.F_fwd @ x

        # Centered correction: delta(x) = MLP(x) - MLP(0). Anchors the
        # NN at zero by construction so that NN(0) = F_fwd @ 0 = 0 for
        # ALL training time, not just at init. This fixes the
        # steady-state asymptote drift seen in earlier runs (NN(zero)
        # max abs ~0.05 -> ~0). Costs one extra forward pass per call.
        correction = self._mlp(x) - self._mlp(jnp.zeros_like(x))

        return linear + correction


# ---------------------------------------------------------------------------
# Loss and one training step
# ---------------------------------------------------------------------------

def maliar_loss(net, u_t, eps_a, eps_b, cred_t, model, params,
                fwd_idx, monitor_index, omega_next, n_inner=20):
    """Two-draw product-of-residuals loss for one training observation.

    With n_inner inner refinement iterations: each refinement re-evaluates
    NN at the resolved next-period state and re-solves. n_inner=0 is the
    pure stale-NN approach; n_inner=20 is enough to converge the inner
    fixed-point on models with max|eig(F)| up to ~0.9 (verified on
    open_no_cred where the required count was proven by n_inner sweep
    1->50 moving the PWL-vs-NN diff from 4.1e-1 to 4.4e-16). Earlier
    default was n_inner=1 which gave BIASED gradient estimates on
    near-unit-root models; the bias grew as 0.9^(-n_inner) and dominated
    Maliar training even before variance kicked in.

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
          reset_every=500,
          n_inner=20,
          shock_scale=5.0,
          batch_size=256,
          final_lr_ratio=0.01,
          ema_decay=0.999):
    """Train an ExpectNet via online Maliar with the two-draw loss.

    This is the "fixed up" Maliar loop that implements the five
    stochastic-approximation conditions missing from the earlier
    online batch=1 version (see
    /home/andres/.claude/plans/eventual-wobbling-reef.md Part A for
    derivation):

      1. Batched two-draw via vmap (batch_size>=256) -- variance O(1/K)
         per step instead of O(1).
      2. Cosine learning-rate decay from `learning_rate` to
         `final_lr_ratio * learning_rate` -- Robbins-Monro condition for
         SGD to converge to a fixed point of an unbiased-but-noisy
         gradient rather than random-walk around one.
      3. Polyak/EMA averaging on the weights -- final network is a
         running exponential average, not the last Adam iterate.
         Filters the random walk.
      4. Default n_inner bumped 1 -> 20 in maliar_loss -- the earlier
         n_inner=1 gave BIASED residuals on models with max|eig(F)|~0.9
         (proven on open_no_cred: n_inner 1 -> 50 moved PWL match from
         4e-1 to 4e-16). Without this, training chases a biased
         gradient.
      5. Longer simulation chains and larger shock scale
         (reset_every=500, shock_scale=5.0) -- so the simulation
         actually visits credibility-stressed states where the signal
         function is curved and the NN has something nonlinear to
         learn. Earlier defaults kept the simulation pinned within
         the high-credibility linear neighbourhood.

    Parameters
    ----------
    model : dict from compile_jax_model
    params : dict
        Parameter values (must include every name in model['param_names']).
    net_init : eqx.Module
        The initial NN. Use ExpectNet or LinearResidualNet.
    n_steps : int
        Number of online training steps. At batch_size=256 and lr decay
        from 1e-3 to 1e-5, n_steps=20000 is usually enough for the
        nonlinear correction to settle. Use 50000+ for production.
    learning_rate : float
        Initial Adam learning rate. The schedule decays this to
        `final_lr_ratio * learning_rate` over n_steps via cosine decay.
        Default 1e-3 is the peak; schedule ends at 1e-5 by default.
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
        errors). reset_every=500 (new default) lets the simulation
        explore +/- 10pp around zero before pulling back, which is
        necessary for visiting credibility-stressed states. Set to 0
        to disable resets (Maliar pure online).
    n_inner : int
        Inner fixed-point iterations inside maliar_loss. See maliar_loss
        docstring. Default 20 (was 1 in the previous version).
    shock_scale : float
        Multiplier on shock_stds during training. Default 5.0 (was 3.0
        in the previous version). With shock_scale=5 and
        reset_every=500, the simulation reaches pi_cpi_yoy of O(3-5pp)
        which is where the credibility signal function is most curved.
    batch_size : int
        Number of two-draw pairs per gradient step. Per-step variance
        of R_a*R_b scales as 1/batch_size. Default 256 gives a ~16x
        reduction in noise vs the earlier batch=1.
    final_lr_ratio : float
        Cosine decay endpoint as a fraction of `learning_rate`. Default
        0.01 means the schedule ends at lr/100.
    ema_decay : float
        Polyak EMA decay for the returned network weights. Higher =
        longer averaging window. Default 0.999 averages over the last
        ~1000 training steps, filtering Adam's random walk around the
        optimum.

    Returns
    -------
    net : eqx.Module
        Trained NN (EMA-averaged weights, not the last iterate).
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
        elif cred_jax is None:
            # No credibility block. update_credibility will short-circuit
            # to (cred=1, omega=const), so the value of monitor_lag is
            # never read. Use 0 as a benign sentinel.
            monitor_index = 0
        else:
            raise ValueError(
                "monitor_index not given and credibility block has no "
                "'monitor' field"
            )

    # Default shock stds
    if shock_stds is None:
        shock_stds = jnp.array([
            float(params["sigma_" + s]) for s in model["shock_names"]
        ])
    else:
        shock_stds = jnp.asarray(shock_stds, dtype=jnp.float64)

    # Scale shocks during training so the simulation visits
    # credibility-stressed states. See `shock_scale` in the docstring.
    shock_stds = shock_stds * float(shock_scale)

    # Default warm start: steady state
    if warm_start_state is None:
        warm_start_state = jnp.zeros(n)
    else:
        warm_start_state = jnp.asarray(warm_start_state, dtype=jnp.float64)

    # ----- Optimiser with cosine-decay schedule -----
    # Robbins-Monro: lr decays from `learning_rate` to
    # `final_lr_ratio * learning_rate`. Adam with a schedule converges
    # to a fixed point of an unbiased-but-noisy gradient (rather than
    # random-walking around one as it does with constant lr).
    if final_lr_ratio <= 0 or final_lr_ratio >= 1:
        # Fallback: constant lr if final_lr_ratio is out of range
        lr_schedule = learning_rate
    else:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=max(1, n_steps),
            alpha=final_lr_ratio,
        )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(eqx.filter(net_init, eqx.is_inexact_array))

    # ----- Freeze the linear baseline (LinearResidualNet.F_fwd) -----
    # LinearResidualNet declares F_fwd as a plain jnp.ndarray leaf,
    # which means Adam treats it as a trainable parameter. At the
    # linear optimum (delta=0) the gradient wrt F_fwd is zero in
    # EXPECTATION but has non-zero VARIANCE, so Adam random-walks
    # F_fwd during training and corrupts the linear baseline.
    # Verified empirically on open_no_cred: after 5000 steps at
    # lr=1e-3->1e-5, F_fwd drifted by 0.03 in max-abs norm -- big
    # enough to dominate the final NN-vs-PWL gap.
    #
    # Fix: capture F_fwd_init (if the net has a .F_fwd attribute) and
    # overwrite F_fwd after every Adam update. Equivalent to a
    # stop-gradient on the linear baseline but keeps the rest of the
    # pytree structure intact.
    has_F_fwd = hasattr(net_init, "F_fwd")
    if has_F_fwd:
        F_fwd_init = net_init.F_fwd

    def _freeze_F_fwd(net_any):
        if has_F_fwd:
            return eqx.tree_at(lambda m: m.F_fwd, net_any, F_fwd_init)
        return net_any

    # ----- Polyak / EMA of trainable parameters -----
    # We split net_init into inexact-array leaves (EMA-tracked) and
    # the static skeleton (untouched). Averaging only the inexact
    # leaves also makes the EMA op a no-op on non-float fields.
    ema_params, net_static = eqx.partition(net_init, eqx.is_inexact_array)

    def _ema_update(ema, current_params):
        return jax.tree_util.tree_map(
            lambda e, c: ema_decay * e + (1.0 - ema_decay) * c,
            ema, current_params,
        )

    # ----- One training step: batched Maliar loss + Adam + EMA -----
    def step_fn(carry, key):
        u_t, cred_t, net, opt_state, ema_params = carry

        # Update credibility from u_t's monitor (same scan as the
        # inversion filter -- consistent with the PWL solver).
        cred_next, omega_next = update_credibility(
            u_t[monitor_index], cred_t, model, params)

        # Batched two-draw shocks: draw (2 * batch_size * n_shocks)
        # standard normals and split into (batch_size, n_shocks)
        # pairs. Each pair is one independent two-draw observation.
        k_a, k_b = jrandom.split(key)
        eps_a_batch = (
            jrandom.normal(k_a, (batch_size, n_shocks)) * shock_stds)
        eps_b_batch = (
            jrandom.normal(k_b, (batch_size, n_shocks)) * shock_stds)

        def loss_batched(net_arg):
            # vmap maliar_loss over the batch axis. All pairs share
            # the same current state u_t and same stale NN forecast
            # net_arg(u_t), so the two-draw independence is preserved
            # per pair (eps_a_i, eps_b_i) and the average over pairs
            # is an unbiased estimate of ||bias||^2.
            def one_pair(ea, eb):
                loss_i, u_next_i = maliar_loss(
                    net_arg, u_t, ea, eb, cred_t, model, params,
                    fwd_idx, monitor_index, omega_next, n_inner=n_inner)
                return loss_i, u_next_i

            losses_batch, u_next_batch = jax.vmap(one_pair)(
                eps_a_batch, eps_b_batch)
            # Advance the simulation carry using the first pair's
            # u_next_a -- deterministic rather than averaged so the
            # state follows a legitimate trajectory.
            return jnp.mean(losses_batch), u_next_batch[0]

        (loss_val, u_next_a), grads = eqx.filter_value_and_grad(
            loss_batched, has_aux=True)(net)

        updates, opt_state = optimizer.update(grads, opt_state, net)
        net = eqx.apply_updates(net, updates)

        # Freeze the linear baseline: reset F_fwd back to its init
        # value after every Adam update, so F_fwd stays at the exact
        # PWL linear policy throughout training. Only the MLP residual
        # (delta) is trained. See discussion above _freeze_F_fwd.
        net = _freeze_F_fwd(net)

        # EMA update on trainable leaves only.
        current_params, _ = eqx.partition(net, eqx.is_inexact_array)
        ema_params = _ema_update(ema_params, current_params)

        new_carry = (u_next_a, cred_next, net, opt_state, ema_params)
        return new_carry, loss_val

    init_carry = (warm_start_state, jnp.array(1.0), net_init, opt_state,
                  ema_params)

    # Run the training loop. Python loop with progress reporting when
    # report_every > 0; pure lax.scan otherwise.
    if report_every and report_every > 0:
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
                u_t, cred_t, net_, opt_state_, ema_params_ = carry
                carry = (warm_start_state, jnp.array(1.0), net_,
                         opt_state_, ema_params_)

            if (i + 1) % report_every == 0:
                ema = sum(losses[-report_every:]) / report_every
                print("  step " + str(i + 1).rjust(6) + " / "
                      + str(n_steps) + "  loss(EMA " + str(report_every)
                      + ") = " + ("%.4e" % ema))
        losses = jnp.array(losses)
    else:
        # Pure scan -- one big JIT compile, fastest for many steps.
        # Note: pure scan does not support periodic resets.
        keys = jrandom.split(jrandom.PRNGKey(seed), n_steps)
        carry, losses = jax.lax.scan(step_fn, init_carry, keys)

    u_final, cred_final, net_final, _, ema_params_final = carry

    # Reconstruct the EMA-averaged network from the averaged
    # trainable leaves + the static skeleton. This is what we return
    # as the trained net -- NOT the last Adam iterate.
    net_ema_final = eqx.combine(ema_params_final, net_static)

    history = {
        "loss": losses,
        "final_state": u_final,
        "final_cred": cred_final,
        "fwd_idx": fwd_idx,
        "fwd_names": fwd_names,
        "monitor_index": monitor_index,
        "shock_stds": shock_stds,
    }
    return net_ema_final, history
