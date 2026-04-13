"""
necredpy -- Production networks with endogenous credibility.

A toolkit for solving and estimating piecewise-linear DSGE/QPM models
with endogenous credibility and production network linkages.

Core modules:
    necredpy.model           -- High-level Model interface (parse, estimate)
    necredpy.jax_model       -- JAX inversion filter for Bayesian estimation
    necredpy.pontus          -- Piecewise-linear solver (Rendahl 2017)
    necredpy.utils           -- Dynare .mod file parser (new grammar)
"""

__version__ = "0.2.0"

from necredpy.model import Model

from necredpy.utils.dynare_parser import (
    parse_mod,
    extract_priors,
)

from necredpy.jax_model import (
    compile_jax_model,
    inversion_filter,
    inversion_filter_partial,
    solve_terminal_jax,
)
