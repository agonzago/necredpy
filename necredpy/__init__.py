"""
necredpy -- Production networks with endogenous credibility.

A toolkit for solving and estimating piecewise-linear DSGE/QPM models
with regime-dependent credibility and production network linkages.

Core modules:
    necredpy.pontus          -- Piecewise-linear solver (Rendahl 2017)
    necredpy.jax_model       -- JAX inversion filter for Bayesian estimation
    necredpy.stability       -- Eigenvalue stability checks
    necredpy.models          -- Model builders (credibility_nk)
    necredpy.utils           -- Dynare .mod file parser
"""

__version__ = "0.1.0"

from necredpy.pontus import (
    solve_terminal,
    solve_terminal_pontus,
    solve_terminal_doubling,
    backward_recursion,
    simulate_forward,
    solve_endogenous,
)

from necredpy.models.credibility_nk import (
    build_matrices,
    build_model,
    baseline_theta,
    make_switching_fn_cred,
)

from necredpy.utils.dynare_parser import (
    parse_mod,
    parse_two_regime_model,
    extract_priors,
)

from necredpy.model import Model
