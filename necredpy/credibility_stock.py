"""
Off-the-model reference measures for the stock of credibility.

These are standalone statistics for **comparison** against the model-consistent
credibility path produced by necredpy's solver.  They do not feed into the
DSGE estimation pipeline.

Measures implemented
--------------------
1. Bomfim-Rudebusch Kalman filter  (BE 1304, Section 3)
2. Gaussian signal regression + AR(1) stock  (BE 1304, Section 4)
3. Cecchetti et al. (2002) index
4. Expectations gap

References
----------
- Bomfim & Rudebusch (2000), JMCB 32(4):707-21
- Grajales-Olarte et al. (2025), Borradores de Economía 1304
- Cecchetti et al. (2002), Review-Federal Reserve Bank of Saint Louis
"""

import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


# ============================================================================
# 1.  General-purpose Kalman filter & smoother  (scalar or multivariate)
# ============================================================================

@partial(jax.jit, static_argnames=())
def kalman_filter(y, Z, T, H, Q, a0, P0):
    """Kalman filter for a linear-Gaussian state-space model.

    State:       a_t = T a_{t-1} + η_t,   η_t ~ N(0, Q)
    Observation: y_t = Z_t a_t + ε_t,     ε_t ~ N(0, H)

    Parameters
    ----------
    y : (T,) or (T, p)
        Observations.
    Z : (T, p, m) or (p, m)
        Observation loading.  If 3-d, time-varying; if 2-d, constant.
    T : (m, m)
        State transition matrix.
    H : (p, p) or scalar
        Observation noise covariance.
    Q : (m, m) or scalar
        State noise covariance.
    a0 : (m,)
        Initial state mean.
    P0 : (m, m)
        Initial state covariance.

    Returns
    -------
    filtered_a : (T, m)   filtered state means
    filtered_P : (T, m, m) filtered state covariances
    log_lik    : scalar    total log-likelihood
    """
    # Normalise shapes
    y = jnp.atleast_2d(y)                   # (T, p)
    if y.ndim == 1:
        y = y[:, None]
    n_t, p = y.shape
    m = a0.shape[0]

    T_mat = jnp.atleast_2d(T)               # (m, m)
    Q_mat = jnp.atleast_2d(Q) if jnp.ndim(Q) > 0 else Q * jnp.eye(m)
    H_mat = jnp.atleast_2d(H) if jnp.ndim(H) > 0 else H * jnp.eye(p)

    # If Z is constant, broadcast to (T, p, m)
    Z_arr = jnp.array(Z)
    if Z_arr.ndim == 2:
        Z_arr = jnp.broadcast_to(Z_arr[None], (n_t, p, m))

    def _step(carry, t):
        a, P, ll = carry
        # Predict
        a_pred = T_mat @ a
        P_pred = T_mat @ P @ T_mat.T + Q_mat
        # Update
        Z_t = Z_arr[t]
        v = y[t] - Z_t @ a_pred                 # innovation
        F = Z_t @ P_pred @ Z_t.T + H_mat        # innovation variance
        F_inv = jnp.linalg.inv(F)
        K = P_pred @ Z_t.T @ F_inv              # Kalman gain
        a_new = a_pred + K @ v
        P_new = P_pred - K @ Z_t @ P_pred
        # Log-likelihood contribution
        sign, logdet = jnp.linalg.slogdet(F)
        ll_t = -0.5 * (p * jnp.log(2 * jnp.pi) + logdet + v @ F_inv @ v)
        return (a_new, P_new, ll + ll_t), (a_new, P_new, a_pred, P_pred)

    init = (a0, P0, 0.0)
    (_, _, log_lik), (filtered_a, filtered_P, predicted_a, predicted_P) = (
        lax.scan(_step, init, jnp.arange(n_t))
    )
    return filtered_a, filtered_P, predicted_a, predicted_P, log_lik


def kalman_smoother(y, Z, T, H, Q, a0, P0):
    """Rauch-Tung-Striebel smoother.

    Returns
    -------
    smoothed_a : (T, m)
    smoothed_P : (T, m, m)
    log_lik    : scalar  (from the filter pass)
    """
    filtered_a, filtered_P, predicted_a, predicted_P, log_lik = (
        kalman_filter(y, Z, T, H, Q, a0, P0)
    )
    m = a0.shape[0]
    T_mat = jnp.atleast_2d(T)
    n_t = filtered_a.shape[0]

    def _smooth_step(carry, t):
        a_s, P_s = carry
        a_f = filtered_a[t]
        P_f = filtered_P[t]
        P_p = predicted_P[t + 1]  # predicted at t+1
        # Smoother gain
        L = P_f @ T_mat.T @ jnp.linalg.inv(P_p)
        a_s_new = a_f + L @ (a_s - predicted_a[t + 1])
        P_s_new = P_f + L @ (P_s - P_p) @ L.T
        return (a_s_new, P_s_new), (a_s_new, P_s_new)

    # Backward pass: start from last filtered state
    init = (filtered_a[-1], filtered_P[-1])
    # Indices from T-2 down to 0
    indices = jnp.arange(n_t - 2, -1, -1)
    (_, _), (smooth_rev_a, smooth_rev_P) = lax.scan(
        _smooth_step, init, indices
    )
    # Reverse and prepend last filtered (which equals smoothed at T)
    smoothed_a = jnp.concatenate(
        [smooth_rev_a[::-1], filtered_a[-1:]], axis=0
    )
    smoothed_P = jnp.concatenate(
        [smooth_rev_P[::-1], filtered_P[-1:]], axis=0
    )
    return smoothed_a, smoothed_P, log_lik


# ============================================================================
# 2.  Bomfim-Rudebusch credibility measure
# ============================================================================

def _build_pi_tilde(pi_past, q=4):
    """Backward-looking inflation component: rolling mean of past q quarters.

    Parameters
    ----------
    pi_past : (T+q,) array
        Full inflation history (needs q pre-sample observations).
    q : int
        Number of lags to average.

    Returns
    -------
    pi_tilde : (T,)
    """
    # Convolution-based rolling mean
    kernel = jnp.ones(q) / q
    full_conv = jnp.convolve(pi_past, kernel, mode='valid')
    # full_conv[t] = mean(pi_past[t], ..., pi_past[t+q-1])
    # We want pi_tilde_t = mean(pi_{t-1}, ..., pi_{t-q})
    # If pi_past includes pre-sample, just take the right slice
    return full_conv


def bomfim_rudebusch_numpyro(pi_e, pi_target, pi_tilde):
    """NumPyro model for Bomfim-Rudebusch state-space credibility.

    Observation:  y_t = λ_t * d_t + ε_t^y
    State:        λ_t = ψ_0 + ψ_1 * λ_{t-1} + ε_t^λ

    where  y_t = π^e_t - π̃_t   and   d_t = π̄_t - π̃_t
    so that  π^e_t = λ_t * π̄_t + (1 - λ_t) * π̃_t + ε^y_t

    Parameters (arrays, not sampled — passed as observed data)
    ----------
    pi_e      : (T,)  inflation expectations (survey mean)
    pi_target : (T,)  inflation target
    pi_tilde  : (T,)  backward-looking inflation component
    """
    import numpyro
    import numpyro.distributions as dist

    n_t = pi_e.shape[0]

    # Transformed observations
    y = pi_e - pi_tilde                     # (T,)
    d = pi_target - pi_tilde                # (T,)  -- time-varying loading

    # --- Priors ---
    psi0 = numpyro.sample("psi0", dist.Normal(0.1, 0.1))
    psi1 = numpyro.sample("psi1", dist.Beta(5.0, 1.0))   # persistent
    sigma_lambda = numpyro.sample("sigma_lambda", dist.HalfNormal(0.1))
    gamma = numpyro.sample("gamma", dist.HalfNormal(1.0))

    # --- State-space matrices (scalar state) ---
    T_mat = jnp.array([[psi1]])
    Q_mat = jnp.array([[sigma_lambda**2]])
    H_mat = jnp.array([[gamma * sigma_lambda**2]])

    # Time-varying Z: Z_t = [[d_t]]
    Z_tv = d[:, None, None]                 # (T, 1, 1)

    # Initial state: stationary distribution
    a0 = jnp.array([psi0 / (1.0 - psi1 + 1e-8)])
    P0 = jnp.array([[sigma_lambda**2 / (1.0 - psi1**2 + 1e-8)]])

    # Shift for intercept: a_t = psi0 + psi1 * a_{t-1} + η
    # Rewrite as: a_t - μ = psi1 * (a_{t-1} - μ) + η  where μ = psi0/(1-psi1)
    # Or handle intercept by augmenting state.
    # Simpler: run filter on de-meaned state, add back.
    mu = psi0 / (1.0 - psi1 + 1e-8)

    # Adjust observation: y_t = d_t * (a_t + mu) + noise  =>  y_t - d_t*mu = d_t * a_t + noise
    y_adj = y - d * mu

    # Run Kalman filter (a_t is zero-mean AR(1) with coeff psi1)
    a0_dm = jnp.array([0.0])
    _, _, _, _, ll = kalman_filter(
        y_adj[:, None], Z_tv, T_mat, H_mat, Q_mat, a0_dm, P0
    )

    numpyro.factor("log_lik", ll)

    # Deterministic: run smoother to get λ path
    smoothed_a, _, _ = kalman_smoother(
        y_adj[:, None], Z_tv, T_mat, H_mat, Q_mat, a0_dm, P0
    )
    lambda_path = jnp.clip(smoothed_a[:, 0] + mu, 0.0, 1.0)
    numpyro.deterministic("lambda_t", lambda_path)


def bomfim_rudebusch(pi_e, pi_target, pi_past, q=4,
                     num_warmup=500, num_samples=1000, num_chains=2,
                     seed=0, progress_bar=True):
    """Estimate Bomfim-Rudebusch credibility via NumPyro MCMC.

    Parameters
    ----------
    pi_e      : (T,) array  — inflation expectations (survey mean)
    pi_target : (T,) array  — inflation target
    pi_past   : (T+q,) array — inflation history (needs q extra pre-sample obs)
    q         : int          — lags for backward-looking component (default 4)
    num_warmup, num_samples, num_chains : MCMC settings
    seed      : int
    progress_bar : bool

    Returns
    -------
    mcmc : numpyro.infer.MCMC object
        Access samples via mcmc.get_samples().
        Key variables: 'psi0', 'psi1', 'sigma_lambda', 'gamma', 'lambda_t'.
    """
    import numpyro
    from numpyro.infer import MCMC, NUTS
    import jax.random as random

    if num_chains > 1:
        numpyro.set_host_device_count(num_chains)

    pi_e = jnp.array(pi_e)
    pi_target = jnp.array(pi_target)
    pi_past = jnp.array(pi_past)

    pi_tilde = _build_pi_tilde(pi_past, q=q)
    # Align: pi_tilde has length len(pi_past)-q+1, take last T
    T = pi_e.shape[0]
    pi_tilde = pi_tilde[-T:]

    kernel = NUTS(bomfim_rudebusch_numpyro, max_tree_depth=10)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, progress_bar=progress_bar)
    mcmc.run(
        random.PRNGKey(seed),
        pi_e=pi_e,
        pi_target=pi_target,
        pi_tilde=pi_tilde,
    )
    return mcmc


# ============================================================================
# 3.  Gaussian signal regression + AR(1) credibility stock
# ============================================================================

def gaussian_signal(pi, pi_star, omega1, omega2, omega3):
    """Credibility signal (BE 1304, eq. 8).

    s_t = exp(-ω₁ - ω₂(π_{t-1} - π*_{t-1})²) - ω₃

    Parameters
    ----------
    pi      : (T,)  inflation (lagged: π_{t-1})
    pi_star : (T,)  inflation target (lagged: π*_{t-1})
    omega1, omega2, omega3 : scalar parameters

    Returns
    -------
    s : (T,) signal values
    """
    dev = pi - pi_star
    return jnp.exp(-omega1 - omega2 * dev**2) - omega3


def credibility_stock_ar1(signal, psi, c0=None):
    """AR(1) credibility stock (BE 1304, eq. 6).

    c_t = ψ c_{t-1} + (1 - ψ) s_t

    Parameters
    ----------
    signal : (T,)
    psi    : scalar, persistence
    c0     : scalar, initial stock (default: signal[0])

    Returns
    -------
    c : (T,) credibility stock path
    """
    if c0 is None:
        c0 = signal[0]

    def _step(c, s_t):
        c_new = psi * c + (1.0 - psi) * s_t
        c_new = jnp.clip(c_new, 0.0, 1.0)
        return c_new, c_new

    _, c_path = lax.scan(_step, c0, signal)
    return c_path


def fit_signal_regression(lambda_t, pi_lag, pi_star_lag):
    """Estimate Gaussian signal parameters by nonlinear least squares.

    Fits:  λ_t ≈ exp(-ω₁ - ω₂(π_{t-1} - π*_{t-1})²) - ω₃

    Uses JAX-based L-BFGS minimization of sum of squared residuals.

    Parameters
    ----------
    lambda_t   : (T,)  credibility measure (e.g. from Bomfim-Rudebusch)
    pi_lag     : (T,)  lagged inflation
    pi_star_lag: (T,)  lagged inflation target

    Returns
    -------
    dict with keys 'omega1', 'omega2', 'omega3', 'signal', 'residuals'
    """
    from jax.scipy.optimize import minimize

    dev_sq = (pi_lag - pi_star_lag)**2

    def loss(params):
        omega1, omega2, omega3 = params
        pred = jnp.exp(-omega1 - omega2 * dev_sq) - omega3
        return jnp.sum((lambda_t - pred)**2)

    # Initial guess (from BE 1304 Table 2)
    x0 = jnp.array([jnp.log(1.0 / (1.0 - 0.43)), 0.06, 0.43])
    result = minimize(loss, x0, method='BFGS')

    omega1, omega2, omega3 = result.x
    signal = jnp.exp(-omega1 - omega2 * dev_sq) - omega3
    return {
        'omega1': omega1,
        'omega2': omega2,
        'omega3': omega3,
        'signal': signal,
        'residuals': lambda_t - signal,
        'converged': result.success,
    }


# ============================================================================
# 4.  Ad-hoc credibility indices
# ============================================================================

def cecchetti_index(pi_e, pi_target, band_width):
    """Cecchetti et al. (2002) credibility index.

    CI_t = 1                                        if |π^e_t - π*_t| ≤ band
    CI_t = 1 - (|π^e_t - π*_t| - band) / band      if band < |...| ≤ 2*band
    CI_t = 0                                        if |π^e_t - π*_t| > 2*band

    Parameters
    ----------
    pi_e       : (T,) inflation expectations
    pi_target  : (T,) inflation target
    band_width : scalar, half-width of tolerance band (e.g. 1.0 for ±1pp)

    Returns
    -------
    CI : (T,) credibility index in [0, 1]
    """
    dev = jnp.abs(pi_e - pi_target)
    ci = 1.0 - jnp.maximum(dev - band_width, 0.0) / band_width
    return jnp.clip(ci, 0.0, 1.0)


def expectations_gap(pi_e, pi_target, normalize=True):
    """Expectations gap credibility measure.

    gap_t = |π^e_t - π*_t|

    If normalize=True, returns  1 - gap_t / max(gap_t)  ∈ [0, 1]
    (higher = more credible).

    Parameters
    ----------
    pi_e      : (T,) inflation expectations
    pi_target : (T,) inflation target
    normalize : bool

    Returns
    -------
    measure : (T,)
    """
    gap = jnp.abs(pi_e - pi_target)
    if normalize:
        max_gap = jnp.max(gap)
        return 1.0 - gap / (max_gap + 1e-10)
    return gap
