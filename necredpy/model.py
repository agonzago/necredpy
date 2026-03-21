"""
High-level Model interface for necredpy.

Usage:
    from necredpy import Model

    m = Model("dynare/five_sector_network.mod")
    irf = m.irf("eps_pi1", size=3.0)
    irf["pi_agg"].plot()

    m = Model("dynare/credibility_nk_regimes.mod")
    results = m.estimate(data, obs_vars=["y", "pi", "i"])
"""

import copy
import os

import numpy as np
import pandas as pd

from necredpy.pontus import (
    solve_terminal, backward_recursion, simulate_forward, solve_endogenous,
)
from necredpy.utils.dynare_parser import (
    parse_two_regime_model, get_model_matrices, build_switching_fn,
    compile_two_regime_model, evaluate_two_regime_model,
    extract_priors, build_numpyro_prior_fn,
    build_credibility_switching_fn,
)


class Model:
    """High-level interface for DSGE/QPM models with endogenous credibility.

    Parameters
    ----------
    mod_file : str
        Path to a Dynare .mod file.
    param_overrides : dict or None
        Override parameter values from the .mod file.
    """

    def __init__(self, mod_file, param_overrides=None):
        with open(mod_file, 'r') as f:
            self._mod_string = f.read()
        self._mod_file = os.path.abspath(mod_file)
        self._param_overrides = param_overrides or {}

        # Compile once (expensive sympy step)
        self._compiled = compile_two_regime_model(self._mod_string)
        self._regime_spec = self._compiled.get('regime_spec_raw')
        parsed = self._compiled['parsed']

        self.var_names = parsed['var_names']
        self.shock_names = parsed['shock_names']
        self.param_names = parsed['param_names']
        self.param_defaults = dict(parsed['param_defaults'])
        self.n_vars = len(self.var_names)
        self.n_shocks = len(self.shock_names)

        # Evaluate at default params to cache regime matrices
        self._evaluate()

    def _evaluate(self, param_overrides=None):
        """Evaluate regime matrices at current (or given) parameter values."""
        overrides = dict(self._param_overrides)
        if param_overrides:
            overrides.update(param_overrides)
        M1, M2, sw, info = evaluate_two_regime_model(
            self._compiled, param_overrides=overrides)
        self._M1 = M1
        self._M2 = M2
        self._info = info
        return info

    @property
    def params(self):
        """Current parameter values (defaults + overrides)."""
        p = dict(self.param_defaults)
        p.update(self._param_overrides)
        return p

    # ------------------------------------------------------------------
    # IRF
    # ------------------------------------------------------------------

    def irf(self, shock_name, size=1.0, T=60, credibility=False,
            cred_init=None, param_overrides=None):
        """Compute impulse response function.

        Parameters
        ----------
        shock_name : str
            Name of the shock (e.g., 'eps_pi1'). Must match a varexo in
            the .mod file.
        size : float
            Shock size at t=0.
        T : int
            Simulation horizon (quarters).
        credibility : bool
            If False, solve single-regime linear model.
            If True, solve with endogenous credibility switching.
        cred_init : float or None
            Initial credibility level (default from .mod file, usually 1.0).
            Only used when credibility=True.
        param_overrides : dict or None
            Temporary parameter overrides for this IRF only.

        Returns
        -------
        pandas.DataFrame
            Columns: variable names + 'regime' and 'credibility' (when
            credibility=True). Index: quarters 0..T-1.
        """
        info = self._evaluate(param_overrides)
        vn = info['var_names']
        sn = info['shock_names']
        n = len(vn)

        # Build shock vector
        si = sn.index(shock_name)
        eps = np.zeros((T, n))
        eps[0, :] = -info['D_shock'][:, si] * size

        if not credibility:
            # Single-regime linear solve
            A, B, C, D = self._M1
            F, Q = solve_terminal(A, B, C)
            u = simulate_forward(
                np.tile(F, (T, 1, 1)),
                np.zeros((T, n)),
                np.tile(Q, (T, 1, 1)),
                eps,
            )
            df = pd.DataFrame(u, columns=vn)
            return df

        # Piecewise-linear with credibility switching
        cred_compiled = info.get('credibility_compiled')
        if cred_compiled is not None:
            if cred_init is not None:
                cred_compiled = copy.deepcopy(cred_compiled)
                cred_compiled['cred_init'] = cred_init
            regime_spec = info.get('regime_spec', {})
            band = regime_spec.get('band') if regime_spec else None
            sw = build_credibility_switching_fn(
                cred_compiled, vn, info['params_M1'], band=band)
        else:
            regime_spec = copy.deepcopy(info['regime_spec'])
            if cred_init is not None:
                regime_spec['cred_init'] = cred_init
            sw = build_switching_fn(regime_spec, vn)

        u, reg, _, _, _, conv, iters = solve_endogenous(
            self._M1, self._M2, sw, eps, T)

        df = pd.DataFrame(u, columns=vn)
        df['regime'] = reg
        if hasattr(sw, 'cred_path') and sw.cred_path is not None:
            df['credibility'] = sw.cred_path
        return df

    # ------------------------------------------------------------------
    # Simulate (general: custom shock sequences)
    # ------------------------------------------------------------------

    def simulate(self, shocks, T=None, credibility=False, cred_init=None,
                 param_overrides=None):
        """Simulate the model with a custom shock sequence.

        Parameters
        ----------
        shocks : dict or pandas.DataFrame or ndarray
            Shock sequence. Can be:
            - dict mapping shock names to 1-D arrays:
              {'eps_pi1': [3.0, 0, 0, ...], 'eps_m': [0, 0.5, ...]}
            - DataFrame with shock names as columns
            - ndarray of shape (T, n_shocks) in varexo order
        T : int or None
            Simulation horizon. Inferred from shocks if None.
        credibility : bool
            If True, solve with endogenous credibility switching.
        cred_init : float or None
            Initial credibility level.
        param_overrides : dict or None
            Temporary parameter overrides.

        Returns
        -------
        pandas.DataFrame
        """
        info = self._evaluate(param_overrides)
        vn = info['var_names']
        sn = info['shock_names']
        n = len(vn)

        # Parse shocks into (T, n_vars) array
        if isinstance(shocks, pd.DataFrame):
            if T is None:
                T = len(shocks)
            eps_active = np.zeros((T, len(sn)))
            for i, s in enumerate(sn):
                if s in shocks.columns:
                    vals = shocks[s].values
                    eps_active[:len(vals), i] = vals[:T]
        elif isinstance(shocks, dict):
            lengths = [len(v) for v in shocks.values()]
            if T is None:
                T = max(lengths) if lengths else 60
            eps_active = np.zeros((T, len(sn)))
            for i, s in enumerate(sn):
                if s in shocks:
                    vals = np.asarray(shocks[s])
                    eps_active[:len(vals), i] = vals[:T]
        else:
            # ndarray (T, n_shocks)
            shocks = np.asarray(shocks)
            if T is None:
                T = shocks.shape[0]
            eps_active = np.zeros((T, len(sn)))
            eps_active[:shocks.shape[0], :shocks.shape[1]] = shocks[:T]

        # Map active shocks into full state-space shock vector
        D_shock = info['D_shock']
        eps = np.zeros((T, n))
        for t in range(T):
            eps[t, :] = -D_shock @ eps_active[t]

        if not credibility:
            A, B, C, D = self._M1
            F, Q = solve_terminal(A, B, C)
            u = simulate_forward(
                np.tile(F, (T, 1, 1)),
                np.zeros((T, n)),
                np.tile(Q, (T, 1, 1)),
                eps,
            )
            return pd.DataFrame(u, columns=vn)

        cred_compiled = info.get('credibility_compiled')
        if cred_compiled is not None:
            if cred_init is not None:
                cred_compiled = copy.deepcopy(cred_compiled)
                cred_compiled['cred_init'] = cred_init
            regime_spec = info.get('regime_spec', {})
            band = regime_spec.get('band') if regime_spec else None
            sw = build_credibility_switching_fn(
                cred_compiled, vn, info['params_M1'], band=band)
        else:
            regime_spec = copy.deepcopy(info['regime_spec'])
            if cred_init is not None:
                regime_spec['cred_init'] = cred_init
            sw = build_switching_fn(regime_spec, vn)

        u, reg, _, _, _, conv, iters = solve_endogenous(
            self._M1, self._M2, sw, eps, T)

        df = pd.DataFrame(u, columns=vn)
        df['regime'] = reg
        if hasattr(sw, 'cred_path') and sw.cred_path is not None:
            df['credibility'] = sw.cred_path
        return df

    # ------------------------------------------------------------------
    # Estimate
    # ------------------------------------------------------------------

    def estimate(self, data, obs_vars, shock_vars=None, tau=0.2,
                 num_warmup=500, num_samples=500, num_chains=2,
                 target_accept_prob=0.8, max_tree_depth=10,
                 seed=0, progress_bar=True):
        """Bayesian estimation via NumPyro NUTS.

        Parameters
        ----------
        data : pandas.DataFrame or ndarray
            Observed data. If DataFrame, columns should match obs_vars.
            If ndarray, shape (T, n_obs) in obs_vars order.
        obs_vars : list of str
            Names of observed variables (must be in var_names).
        shock_vars : list of str or None
            Names of equations where shocks enter. If None, defaults to
            obs_vars (assumes one shock per observable).
        tau : float
            Sigmoid smoothing parameter for credibility (smaller = sharper).
        num_warmup : int
            Number of NUTS warmup samples.
        num_samples : int
            Number of posterior samples per chain.
        num_chains : int
            Number of MCMC chains.
        target_accept_prob : float
            NUTS target acceptance probability.
        max_tree_depth : int
            NUTS maximum tree depth.
        seed : int
            Random seed for MCMC.
        progress_bar : bool
            Show NumPyro progress bar.

        Returns
        -------
        EstimationResult with attributes:
            .samples : dict of posterior samples
            .summary : pandas.DataFrame of posterior summary
            .mcmc : numpyro.infer.MCMC object
            .credibility : ndarray, credibility path at posterior mean
            .log_lik : float, log-likelihood at posterior mean
        """
        import jax
        import jax.numpy as jnp
        import jax.random as random
        import numpyro
        from numpyro.infer import MCMC, NUTS

        from necredpy.jax_model import (
            compile_jax_model, inversion_filter_partial, solve_terminal_jax,
        )

        # Compile JAX model
        jax_model = compile_jax_model(self._mod_string)
        priors = extract_priors(self._mod_string)
        if not priors:
            raise ValueError("No priors block found in .mod file. "
                             "Add a priors; ... end; block to estimate.")

        # Build parameter dict
        true_params = dict(jax_model['param_defaults'])
        true_params.update(self._param_overrides)
        true_params['tau'] = tau
        # Default shock std devs if not provided
        for s in jax_model['shock_names']:
            true_params.setdefault('sigma_' + s, 0.5)

        # Observation and shock indices
        var_names = jax_model['var_names']
        obs_indices = [var_names.index(v) for v in obs_vars]
        if shock_vars is None:
            shock_vars = obs_vars
        shock_indices = [var_names.index(v) for v in shock_vars]

        # Convert data to jnp array
        if isinstance(data, pd.DataFrame):
            obs_partial = jnp.array(data[obs_vars].values)
        else:
            obs_partial = jnp.array(np.asarray(data))

        # Build NumPyro model
        sample_priors_fn, estimated_names = build_numpyro_prior_fn(priors)

        def nk_model(obs_partial):
            sampled = sample_priors_fn()
            params = {k: v for k, v in true_params.items()}
            params.update(sampled)
            ll, _, _, _ = inversion_filter_partial(
                jax_model, obs_partial, params,
                obs_indices=obs_indices,
                shock_indices=shock_indices)
            numpyro.factor('log_lik', ll)

        # Run MCMC
        kernel = NUTS(nk_model,
                      target_accept_prob=target_accept_prob,
                      max_tree_depth=max_tree_depth)
        mcmc = MCMC(kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    progress_bar=progress_bar)

        rng_key = random.PRNGKey(seed)
        mcmc.run(rng_key, obs_partial=obs_partial)

        samples = mcmc.get_samples()

        # Posterior mean credibility path
        post_mean_params = dict(true_params)
        for name in estimated_names:
            if name in samples:
                post_mean_params[name] = float(jnp.mean(samples[name]))

        ll_post, _, cred_post, _ = inversion_filter_partial(
            jax_model, obs_partial, post_mean_params,
            obs_indices=obs_indices,
            shock_indices=shock_indices)

        # Build summary DataFrame
        summary_data = {}
        for name in estimated_names:
            s = np.array(samples[name])
            summary_data[name] = {
                'mean': np.mean(s),
                'std': np.std(s),
                'q05': np.quantile(s, 0.05),
                'q50': np.quantile(s, 0.50),
                'q95': np.quantile(s, 0.95),
            }
        summary_df = pd.DataFrame(summary_data).T

        return EstimationResult(
            samples=samples,
            summary=summary_df,
            mcmc=mcmc,
            estimated_names=estimated_names,
            credibility=np.array(cred_post),
            log_lik=float(ll_post),
            obs_partial=np.array(obs_partial),
        )


class EstimationResult:
    """Container for estimation results.

    Attributes
    ----------
    samples : dict
        Posterior samples keyed by parameter name.
    summary : pandas.DataFrame
        Posterior summary (mean, std, quantiles).
    mcmc : numpyro.infer.MCMC
        Full MCMC object (for diagnostics, trace plots, etc.).
    estimated_names : list of str
        Names of estimated parameters.
    credibility : ndarray
        Credibility path at posterior mean.
    log_lik : float
        Log-likelihood at posterior mean.
    obs_partial : ndarray
        Observed data used for estimation.
    """

    def __init__(self, samples, summary, mcmc, estimated_names,
                 credibility, log_lik, obs_partial):
        self.samples = samples
        self.summary = summary
        self.mcmc = mcmc
        self.estimated_names = estimated_names
        self.credibility = credibility
        self.log_lik = log_lik
        self.obs_partial = obs_partial

    def __repr__(self):
        return (
            "EstimationResult(%d parameters, log_lik=%.2f)\n%s"
            % (len(self.estimated_names), self.log_lik, self.summary)
        )
