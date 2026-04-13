"""
High-level Model interface for necredpy.

Usage:
    from necredpy import Model

    m = Model("dynare/network_soe_cred.mod")
    results = m.estimate(data, obs_vars=[...], shock_vars=[...])
"""

import os

import numpy as np
import pandas as pd

from necredpy.utils.dynare_parser import (
    extract_priors, build_numpyro_prior_fn,
)


class Model:
    """High-level interface for DSGE/QPM models with endogenous credibility.

    Uses compile_jax_model (new credibility grammar with var/input/output
    declarations) as the single compilation path.

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

        # Single compilation path: compile_jax_model handles the new
        # credibility grammar (var/input/output) and model(pwl/nn).
        from necredpy.jax_model import compile_jax_model
        self._jax_model = compile_jax_model(self._mod_string)

        self.var_names = self._jax_model['var_names']
        self.shock_names = self._jax_model['shock_names']
        self.param_names = self._jax_model['param_names']
        self.param_defaults = dict(self._jax_model['param_defaults'])
        self.n_vars = self._jax_model['n_vars']
        self.n_shocks = self._jax_model['n_shocks']

    @property
    def params(self):
        """Current parameter values (defaults + overrides)."""
        p = dict(self.param_defaults)
        p.update(self._param_overrides)
        return p

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

        if num_chains > 1:
            numpyro.set_host_device_count(num_chains)

        from necredpy.jax_model import inversion_filter_partial

        jax_model = self._jax_model
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
            ll, _, _, _, _, _, _ = inversion_filter_partial(
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

        ll_post, _, cred_post, _, _, _, _ = inversion_filter_partial(
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
