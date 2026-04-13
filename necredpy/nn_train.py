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
    simulate_one_step_full,
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


class FullSystemNet(eqx.Module):
    """NN for the full nonlinear system with credibility endogenous.

    Solves the FULL feedback loop: inflation -> credibility erosion ->
    more backward-looking expectations -> more persistent inflation.
    This is the correct nonlinear solution; the PWL operator-splitting
    approach is the approximation (see config/nn_solver_plan.md).

    Input:  (u_t, cred_t) -- n_vars + 1 dimensions
    Output: (E_t[u_{t+1}[fwd]], E_t[cred_{t+1}]) -- n_fwd + 1 dimensions

    Architecture (regime-aware LinearResidualNet):

        F_blend(cred) = cred * F_high + (1 - cred) * F_low
        linear        = [F_blend @ u,  cred]
        correction    = MLP(u, cred) - MLP(0, 1)      (centered at SS)
        output        = linear + correction

    At steady state (u=0, cred=1): F_blend = F_high, output = (0, 1).
    The MLP correction is zero by construction.

    Why this works:
    1. The sharp regime transition (high-cred vs low-cred policy) is
       handled by F_blend, not the MLP. The MLP only learns the small
       bilinear cross-term and the nonlinear credibility dynamics.
    2. F_high and F_low are frozen (not trained). They come from
       solve_terminal at omega_H and omega_L respectively.
    3. The cred output teaches the NN the credibility law of motion.
       The gradient from the cred prediction error flows back through
       the MLP, linking inflation dynamics to credibility.

    Parameters
    ----------
    key : PRNGKey
    n_vars : int
        Number of model variables (e.g. 42).
    n_fwd : int
        Number of forward-looking variables (e.g. 9).
    hidden : int
        Hidden layer width.
    F_high : (n_fwd, n_vars) array
        Linear policy rows at omega_H (full credibility).
    F_low : (n_fwd, n_vars) array
        Linear policy rows at omega_L (zero credibility).
    """
    layers: list
    F_high: jnp.ndarray   # (n_fwd, n_vars) -- frozen
    F_low: jnp.ndarray    # (n_fwd, n_vars) -- frozen

    def __init__(self, key, n_vars, n_fwd, hidden, F_high, F_low):
        if F_high.shape != (n_fwd, n_vars):
            raise ValueError(
                "F_high must have shape (" + str(n_fwd) + ", " + str(n_vars)
                + ") but got " + str(F_high.shape))
        if F_low.shape != (n_fwd, n_vars):
            raise ValueError(
                "F_low must have shape (" + str(n_fwd) + ", " + str(n_vars)
                + ") but got " + str(F_low.shape))

        n_in = n_vars + 1    # u + cred
        n_out = n_fwd + 1    # fwd_vars + cred_next

        keys = jrandom.split(key, 3)

        # Zero-init last layer so initial correction is exactly zero
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
        self.F_high = jnp.asarray(F_high)
        self.F_low = jnp.asarray(F_low)

    def _mlp(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)

    def __call__(self, u, cred):
        """Predict E_t[u_{t+1}[fwd], cred_{t+1}] given (u_t, cred_t).

        Parameters
        ----------
        u : (n_vars,) array -- model state in deviation space
        cred : scalar -- credibility stock in [0, 1]

        Returns
        -------
        out : (n_fwd + 1,) array
            First n_fwd entries: E_t[u_{t+1}[fwd]]
            Last entry: E_t[cred_{t+1}]
        """
        # Linear baseline: interpolate policy between regimes
        F_blend = cred * self.F_high + (1.0 - cred) * self.F_low
        linear_fwd = F_blend @ u

        # Full linear baseline: [fwd prediction, cred persistence]
        linear = jnp.concatenate([linear_fwd, jnp.atleast_1d(cred)])

        # Nonlinear correction centered at steady state (u=0, cred=1)
        x = jnp.concatenate([u, jnp.atleast_1d(cred)])
        x_ss = jnp.concatenate([jnp.zeros_like(u), jnp.ones(1)])
        correction = self._mlp(x) - self._mlp(x_ss)

        return linear + correction


class PolicyNet(eqx.Module):
    """NN that predicts the POLICY FUNCTION u_t = g(u_{t-1}, cred_t, eps_t).

    Unlike FullSystemNet (which predicts expectations and requires an inner
    fixed-point iteration to resolve u_t), PolicyNet directly outputs u_t.
    The Euler equation residual is then a direct function of the NN output
    at two consecutive time steps -- NO inner fixed-point needed.

    This follows Pascal (QuantEcon notebook) rather than Maliar's PEA:
    the NN maps state -> decision, and the loss checks the Euler equation.

    Architecture:

        F_blend = cred * F_high + (1 - cred) * F_low
        Q_blend = cred * Q_high + (1 - cred) * Q_low
        linear  = F_blend @ u_{t-1} + Q_blend @ eps_t
        correction = MLP(u_{t-1}, cred, eps_t) - MLP(0, 1, 0)
        u_t     = linear + correction

    At steady state (u=0, cred=1, eps=0): u_t = 0 exactly.

    F_high, F_low: terminal policy matrices at omega_H, omega_L (full model).
    Q_high, Q_low: impact matrices Q = (B + CF)^{-1} D at omega_H, omega_L.

    Parameters
    ----------
    key : PRNGKey
    n_vars : int
    n_shocks : int
    hidden : int
    F_high, F_low : (n_vars, n_vars) arrays
    Q_high, Q_low : (n_vars, n_shocks) arrays
    """
    layers: list
    F_high: jnp.ndarray   # (n_vars, n_vars) -- frozen
    F_low: jnp.ndarray    # (n_vars, n_vars) -- frozen
    Q_high: jnp.ndarray   # (n_vars, n_shocks) -- frozen
    Q_low: jnp.ndarray    # (n_vars, n_shocks) -- frozen

    def __init__(self, key, n_vars, n_shocks, hidden, F_high, F_low, Q_high, Q_low):
        n_in = n_vars + 1 + n_shocks   # u_lag + cred + eps
        n_out = n_vars

        keys = jrandom.split(key, 3)

        # Zero-init last layer: initial correction is zero
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
        self.F_high = jnp.asarray(F_high)
        self.F_low = jnp.asarray(F_low)
        self.Q_high = jnp.asarray(Q_high)
        self.Q_low = jnp.asarray(Q_low)

    def _mlp(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = jnp.tanh(layer(h))
        return self.layers[-1](h)

    def __call__(self, u_lag, cred, eps):
        """Predict u_t given (u_{t-1}, cred_t, eps_t).

        Parameters
        ----------
        u_lag : (n_vars,) array
        cred : scalar -- credibility at time t (precomputed from law of motion)
        eps : (n_shocks,) array -- structural shock at time t

        Returns
        -------
        u_t : (n_vars,) array
        """
        F_blend = cred * self.F_high + (1.0 - cred) * self.F_low
        Q_blend = cred * self.Q_high + (1.0 - cred) * self.Q_low
        linear = F_blend @ u_lag + Q_blend @ eps

        x = jnp.concatenate([u_lag, jnp.atleast_1d(cred), eps])
        x_ss = jnp.concatenate([jnp.zeros_like(u_lag), jnp.ones(1),
                                jnp.zeros_like(eps)])
        correction = self._mlp(x) - self._mlp(x_ss)

        return linear + correction


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


def maliar_loss_full(net, u_t, eps_a, eps_b, cred_t, model, params,
                     fwd_idx, monitor_index, n_fwd, n_inner=20):
    """Two-draw loss for the full-system NN (credibility endogenous).

    Same principle as maliar_loss, but the NN takes (u, cred) as input
    and predicts (fwd_vars, cred_next). The cred target is deterministic
    (depends on u_t and cred_t through the credibility law of motion),
    so its contribution to the two-draw product is the exact squared
    error -- no variance to average out.

    Parameters
    ----------
    net : FullSystemNet
        Callable net(u, cred) -> (n_fwd+1,) array.
    u_t : (n,) array -- current state
    eps_a, eps_b : (n_shocks,) arrays -- two independent shock draws
    cred_t : scalar -- current credibility stock
    model, params : as in nn_solver
    fwd_idx : (n_fwd,) int array
    monitor_index : int
    n_fwd : int
    n_inner : int

    Returns
    -------
    loss : scalar
    u_next_a : (n,) array -- for advancing the simulation carry
    cred_next : scalar -- for advancing the credibility carry
    """
    # NN prediction of E_t[u_{t+1}[fwd], cred_{t+1}]
    out_t = net(u_t, cred_t)
    E_fwd = out_t[:n_fwd]
    E_cred = out_t[n_fwd]

    # Next-period credibility from the law of motion (deterministic)
    monitor_t = u_t[monitor_index]
    cred_next, omega_next = update_credibility(
        monitor_t, cred_t, model, params)

    # First-pass resolve under each shock
    u_next_a = resolve_state(u_t, E_fwd, eps_a, model, params,
                              fwd_idx=fwd_idx, omega_t=omega_next)
    u_next_b = resolve_state(u_t, E_fwd, eps_b, model, params,
                              fwd_idx=fwd_idx, omega_t=omega_next)

    # Inner refinements: re-evaluate NN at the resolved states using
    # the realized cred_next (fixed for all inner iterations).
    for _ in range(n_inner):
        out_a = net(u_next_a, cred_next)
        out_b = net(u_next_b, cred_next)
        u_next_a = resolve_state(u_t, out_a[:n_fwd], eps_a, model, params,
                                  fwd_idx=fwd_idx, omega_t=omega_next)
        u_next_b = resolve_state(u_t, out_b[:n_fwd], eps_b, model, params,
                                  fwd_idx=fwd_idx, omega_t=omega_next)

    # Targets: realized forward vars + deterministic cred
    actual_a = jnp.concatenate([u_next_a[fwd_idx], jnp.atleast_1d(cred_next)])
    actual_b = jnp.concatenate([u_next_b[fwd_idx], jnp.atleast_1d(cred_next)])

    # Two-draw residuals against the full prediction
    E_full = jnp.concatenate([E_fwd, jnp.atleast_1d(E_cred)])
    R_a = actual_a - E_full
    R_b = actual_b - E_full

    loss = jnp.mean(R_a * R_b)
    return loss, u_next_a, cred_next


def euler_residual_loss(net, u_lag, cred_lag, eps_t, eps_a, eps_b,
                        model, params, monitor_index):
    """Two-draw Euler-residual loss for PolicyNet (no inner fixed point).

    The NN directly predicts u_t = net(u_lag, cred_t, eps_t). The Euler
    equation residual is then computed from two independent next-period
    shock draws, giving an unbiased estimator of ||E[residual]||^2.

    Flow (3 NN forward passes, zero inner iterations):
      1. cred_t = credibility_law_of_motion(cred_lag, monitor(u_lag))
      2. u_t = net(u_lag, cred_t, eps_t)
      3. cred_{t+1} = credibility_law_of_motion(cred_t, monitor(u_t))
      4. u_{t+1}^a = net(u_t, cred_{t+1}, eps_a)
      5. u_{t+1}^b = net(u_t, cred_{t+1}, eps_b)
      6. R_a = A u_lag + B u_t + C u_{t+1}^a - D eps_t
      7. R_b = A u_lag + B u_t + C u_{t+1}^b - D eps_t
      8. Loss = mean(R_a * R_b)

    For predetermined equations (C_j = 0), R_a_j = R_b_j, so the
    product is the exact squared residual. For forward-looking equations
    (C_j != 0), the two-draw trick gives an unbiased estimator.

    Parameters
    ----------
    net : PolicyNet
        Callable net(u_lag, cred, eps) -> u_t.
    u_lag : (n,) array
    cred_lag : scalar
    eps_t : (n_shocks,) array -- current-period shock
    eps_a, eps_b : (n_shocks,) arrays -- two independent next-period draws
    model, params : as in nn_solver
    monitor_index : int

    Returns
    -------
    loss : scalar
    u_t : (n,) array -- for advancing the simulation carry
    cred_t : scalar
    """
    # Step 1: current-period credibility
    monitor_lag = u_lag[monitor_index]
    cred_t, omega_t = update_credibility(monitor_lag, cred_lag, model, params)

    # Step 2: NN predicts u_t directly
    u_t = net(u_lag, cred_t, eps_t)

    # Step 3: next-period credibility
    monitor_t = u_t[monitor_index]
    cred_next, omega_next = update_credibility(monitor_t, cred_t, model, params)

    # Steps 4-5: next-period states under two independent shocks
    u_next_a = net(u_t, cred_next, eps_a)
    u_next_b = net(u_t, cred_next, eps_b)

    # Steps 6-7: Euler residuals at the CURRENT period's matrices
    coeff_names = model.get("coefficient_names", [])
    if coeff_names:
        A, B, C = model["build_ABC"](params, omega_t)
        D = model["build_D"](params, omega_t)
    else:
        A, B, C = model["build_ABC"](params)
        D = model["build_D"](params)

    # Sign convention: A u_lag + B u_t + C u_next - D eps_t = 0
    R_a = A @ u_lag + B @ u_t + C @ u_next_a - D @ eps_t
    R_b = A @ u_lag + B @ u_t + C @ u_next_b - D @ eps_t

    # Step 8: two-draw product (unbiased estimator of squared residual)
    loss = jnp.mean(R_a * R_b)
    return loss, u_t, cred_t


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


# ---------------------------------------------------------------------------
# Full-system training loop (credibility endogenous)
# ---------------------------------------------------------------------------

def train_full(model, params, net_init,
               n_steps=20000,
               learning_rate=1e-3,
               shock_stds=None,
               monitor_index=None,
               seed=0,
               report_every=1000,
               reset_every=500,
               n_inner=20,
               shock_scale=5.0,
               batch_size=16,
               final_lr_ratio=0.01,
               ema_decay=0.999):
    """Train a FullSystemNet via online Maliar with credibility endogenous.

    Same Maliar two-draw method as train(), but uses maliar_loss_full
    which passes (u, cred) to the NN and targets both forward vars and
    credibility. The NN learns the full nonlinear feedback loop.

    Parameters
    ----------
    model : dict from compile_jax_model
    params : dict
    net_init : FullSystemNet
        Must have signature net(u, cred) -> (n_fwd+1,).
    n_steps, learning_rate, shock_stds, monitor_index, seed,
    report_every, reset_every, n_inner, shock_scale, batch_size,
    final_lr_ratio, ema_decay : same as train().

    Returns
    -------
    net : FullSystemNet (EMA-averaged)
    history : dict with 'loss', 'final_state', 'final_cred', etc.
    """
    n = model["n_vars"]
    n_shocks = model["n_shocks"]
    var_names = model["var_names"]

    fwd_idx, fwd_names = discover_forward_vars(model, params)
    n_fwd = int(fwd_idx.shape[0])

    # Verify net output dimension: n_fwd + 1 (fwd vars + cred)
    test_out = net_init(jnp.zeros(n), jnp.array(1.0))
    if int(test_out.shape[0]) != n_fwd + 1:
        raise ValueError(
            "FullSystemNet output dim (" + str(test_out.shape[0])
            + ") must be n_fwd + 1 = " + str(n_fwd + 1))

    # Default monitor index
    if monitor_index is None:
        cred_new = model.get("credibility_new")
        cred_jax = model.get("credibility_jax")
        if cred_new and cred_new.get("input_vars"):
            monitor_index = var_names.index(cred_new["input_vars"][0])
        elif cred_jax and "monitor" in cred_jax:
            monitor_index = var_names.index(cred_jax["monitor"])
        else:
            monitor_index = 0

    # Default shock stds
    if shock_stds is None:
        shock_stds = jnp.array([
            float(params["sigma_" + s]) for s in model["shock_names"]
        ])
    else:
        shock_stds = jnp.asarray(shock_stds, dtype=jnp.float64)
    shock_stds = shock_stds * float(shock_scale)

    # Optimizer with cosine decay
    if final_lr_ratio <= 0 or final_lr_ratio >= 1:
        lr_schedule = learning_rate
    else:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=max(1, n_steps),
            alpha=final_lr_ratio,
        )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(eqx.filter(net_init, eqx.is_inexact_array))

    # Freeze F_high and F_low (linear baselines, not trained)
    F_high_init = net_init.F_high
    F_low_init = net_init.F_low

    def _freeze_baselines(net_any):
        net_any = eqx.tree_at(lambda m: m.F_high, net_any, F_high_init)
        net_any = eqx.tree_at(lambda m: m.F_low, net_any, F_low_init)
        return net_any

    # EMA tracking
    ema_params, net_static = eqx.partition(net_init, eqx.is_inexact_array)

    def _ema_update(ema, current):
        return jax.tree_util.tree_map(
            lambda e, c: ema_decay * e + (1.0 - ema_decay) * c,
            ema, current)

    # One training step
    def step_fn(carry, key):
        u_t, cred_t, net, opt_state, ema_params = carry

        # Batched two-draw shocks
        k_a, k_b = jrandom.split(key)
        eps_a_batch = jrandom.normal(k_a, (batch_size, n_shocks)) * shock_stds
        eps_b_batch = jrandom.normal(k_b, (batch_size, n_shocks)) * shock_stds

        def loss_batched(net_arg):
            def one_pair(ea, eb):
                loss_i, u_next_i, cred_next_i = maliar_loss_full(
                    net_arg, u_t, ea, eb, cred_t, model, params,
                    fwd_idx, monitor_index, n_fwd, n_inner=n_inner)
                return loss_i, (u_next_i, cred_next_i)

            losses_batch, (u_next_batch, cred_next_batch) = jax.vmap(
                one_pair)(eps_a_batch, eps_b_batch)
            return jnp.mean(losses_batch), (u_next_batch[0],
                                             cred_next_batch[0])

        (loss_val, (u_next, cred_next)), grads = eqx.filter_value_and_grad(
            loss_batched, has_aux=True)(net)

        updates, opt_state = optimizer.update(grads, opt_state, net)
        net = eqx.apply_updates(net, updates)
        net = _freeze_baselines(net)

        current_params, _ = eqx.partition(net, eqx.is_inexact_array)
        ema_params = _ema_update(ema_params, current_params)

        new_carry = (u_next, cred_next, net, opt_state, ema_params)
        return new_carry, loss_val

    warm_start = jnp.zeros(n)
    init_carry = (warm_start, jnp.array(1.0), net_init, opt_state, ema_params)

    # Python loop with reporting
    if report_every and report_every > 0:
        step_jit = eqx.filter_jit(step_fn)
        carry = init_carry
        losses = []
        key = jrandom.PRNGKey(seed)
        for i in range(n_steps):
            key, subkey = jrandom.split(key)
            carry, loss_val = step_jit(carry, subkey)
            losses.append(float(loss_val))

            if reset_every > 0 and (i + 1) % reset_every == 0:
                u_t, cred_t, net_, opt_state_, ema_params_ = carry
                carry = (warm_start, jnp.array(1.0), net_,
                         opt_state_, ema_params_)

            if (i + 1) % report_every == 0:
                ema = sum(losses[-report_every:]) / report_every
                print("  step " + str(i + 1).rjust(6) + " / "
                      + str(n_steps) + "  loss(EMA " + str(report_every)
                      + ") = " + ("%.4e" % ema))
        losses = jnp.array(losses)
    else:
        keys = jrandom.split(jrandom.PRNGKey(seed), n_steps)
        carry, losses = jax.lax.scan(step_fn, init_carry, keys)

    u_final, cred_final, net_final, _, ema_params_final = carry
    net_ema_final = eqx.combine(ema_params_final, net_static)

    history = {
        "loss": losses,
        "final_state": u_final,
        "final_cred": cred_final,
        "fwd_idx": fwd_idx,
        "fwd_names": fwd_names,
        "n_fwd": n_fwd,
        "monitor_index": monitor_index,
        "shock_stds": shock_stds,
    }
    return net_ema_final, history


# ---------------------------------------------------------------------------
# Policy-function training loop (no inner fixed point)
# ---------------------------------------------------------------------------

def train_policy(model, params, net_init,
                 n_steps=20000,
                 learning_rate=1e-3,
                 shock_stds=None,
                 monitor_index=None,
                 seed=0,
                 report_every=1000,
                 reset_every=500,
                 shock_scale=5.0,
                 batch_size=64,
                 final_lr_ratio=0.01,
                 ema_decay=0.999):
    """Train a PolicyNet via Euler-residual loss (Pascal/two-draw, no fixed point).

    Unlike train() and train_full(), this uses euler_residual_loss which
    evaluates the NN at three consecutive states (u_lag -> u_t -> u_{t+1})
    without any inner fixed-point iteration. The Euler equation residual
    is checked directly.

    Because there is no n_inner loop, each training step is O(3) NN
    forward passes instead of O(n_inner) -- dramatically faster per step,
    and the gradient is UNBIASED regardless of max|eig(F)|.

    Parameters
    ----------
    model : dict from compile_jax_model
    params : dict
    net_init : PolicyNet
        Must have signature net(u_lag, cred, eps) -> u_t.
    n_steps, learning_rate, shock_stds, monitor_index, seed,
    report_every, reset_every, shock_scale, batch_size,
    final_lr_ratio, ema_decay : same semantics as train().

    Returns
    -------
    net : PolicyNet (EMA-averaged)
    history : dict with 'loss', 'final_state', 'final_cred', etc.
    """
    n = model["n_vars"]
    n_shocks = model["n_shocks"]
    var_names = model["var_names"]

    # Verify net output dimension
    test_out = net_init(jnp.zeros(n), jnp.array(1.0), jnp.zeros(n_shocks))
    if int(test_out.shape[0]) != n:
        raise ValueError(
            "PolicyNet output dim (" + str(test_out.shape[0])
            + ") must equal n_vars = " + str(n))

    # Default monitor index
    if monitor_index is None:
        cred_new = model.get("credibility_new")
        cred_jax = model.get("credibility_jax")
        if cred_new and cred_new.get("input_vars"):
            monitor_index = var_names.index(cred_new["input_vars"][0])
        elif cred_jax and "monitor" in cred_jax:
            monitor_index = var_names.index(cred_jax["monitor"])
        else:
            monitor_index = 0

    # Default shock stds
    if shock_stds is None:
        shock_stds = jnp.array([
            float(params["sigma_" + s]) for s in model["shock_names"]
        ])
    else:
        shock_stds = jnp.asarray(shock_stds, dtype=jnp.float64)
    shock_stds = shock_stds * float(shock_scale)

    # Optimizer with cosine decay
    if final_lr_ratio <= 0 or final_lr_ratio >= 1:
        lr_schedule = learning_rate
    else:
        lr_schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=max(1, n_steps),
            alpha=final_lr_ratio,
        )
    optimizer = optax.adam(lr_schedule)
    opt_state = optimizer.init(eqx.filter(net_init, eqx.is_inexact_array))

    # Freeze linear baselines (F_high, F_low, Q_high, Q_low)
    F_high_init = net_init.F_high
    F_low_init = net_init.F_low
    Q_high_init = net_init.Q_high
    Q_low_init = net_init.Q_low

    def _freeze_baselines(net_any):
        net_any = eqx.tree_at(lambda m: m.F_high, net_any, F_high_init)
        net_any = eqx.tree_at(lambda m: m.F_low, net_any, F_low_init)
        net_any = eqx.tree_at(lambda m: m.Q_high, net_any, Q_high_init)
        net_any = eqx.tree_at(lambda m: m.Q_low, net_any, Q_low_init)
        return net_any

    # EMA tracking
    ema_params, net_static = eqx.partition(net_init, eqx.is_inexact_array)

    def _ema_update(ema, current):
        return jax.tree_util.tree_map(
            lambda e, c: ema_decay * e + (1.0 - ema_decay) * c,
            ema, current)

    # One training step
    def step_fn(carry, key):
        u_lag, cred_lag, net, opt_state, ema_params = carry

        # Three independent shock draws: current period + two next-period
        k_t, k_a, k_b = jrandom.split(key, 3)
        eps_t_batch = jrandom.normal(k_t, (batch_size, n_shocks)) * shock_stds
        eps_a_batch = jrandom.normal(k_a, (batch_size, n_shocks)) * shock_stds
        eps_b_batch = jrandom.normal(k_b, (batch_size, n_shocks)) * shock_stds

        def loss_batched(net_arg):
            def one_sample(eps_t, eps_a, eps_b):
                loss_i, u_t_i, cred_t_i = euler_residual_loss(
                    net_arg, u_lag, cred_lag, eps_t, eps_a, eps_b,
                    model, params, monitor_index)
                return loss_i, (u_t_i, cred_t_i)

            losses, (u_ts, cred_ts) = jax.vmap(one_sample)(
                eps_t_batch, eps_a_batch, eps_b_batch)
            # Advance carry using first sample's u_t
            return jnp.mean(losses), (u_ts[0], cred_ts[0])

        (loss_val, (u_next, cred_next)), grads = eqx.filter_value_and_grad(
            loss_batched, has_aux=True)(net)

        updates, opt_state = optimizer.update(grads, opt_state, net)
        net = eqx.apply_updates(net, updates)
        net = _freeze_baselines(net)

        current_params, _ = eqx.partition(net, eqx.is_inexact_array)
        ema_params = _ema_update(ema_params, current_params)

        new_carry = (u_next, cred_next, net, opt_state, ema_params)
        return new_carry, loss_val

    warm_start = jnp.zeros(n)
    init_carry = (warm_start, jnp.array(1.0), net_init, opt_state, ema_params)

    # Python loop with reporting
    if report_every and report_every > 0:
        step_jit = eqx.filter_jit(step_fn)
        carry = init_carry
        losses = []
        key = jrandom.PRNGKey(seed)
        for i in range(n_steps):
            key, subkey = jrandom.split(key)
            carry, loss_val = step_jit(carry, subkey)
            losses.append(float(loss_val))

            if reset_every > 0 and (i + 1) % reset_every == 0:
                u_t, cred_t, net_, opt_state_, ema_params_ = carry
                carry = (warm_start, jnp.array(1.0), net_,
                         opt_state_, ema_params_)

            if (i + 1) % report_every == 0:
                ema = sum(losses[-report_every:]) / report_every
                print("  step " + str(i + 1).rjust(6) + " / "
                      + str(n_steps) + "  loss(EMA " + str(report_every)
                      + ") = " + ("%.4e" % ema))
        losses = jnp.array(losses)
    else:
        keys = jrandom.split(jrandom.PRNGKey(seed), n_steps)
        carry, losses = jax.lax.scan(step_fn, init_carry, keys)

    u_final, cred_final, net_final, _, ema_params_final = carry
    net_ema_final = eqx.combine(ema_params_final, net_static)

    history = {
        "loss": losses,
        "final_state": u_final,
        "final_cred": cred_final,
        "monitor_index": monitor_index,
        "shock_stds": shock_stds,
    }
    return net_ema_final, history
