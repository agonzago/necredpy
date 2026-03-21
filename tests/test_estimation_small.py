"""Smoke test: run estimation on the 3-equation NK model to completion."""

import numpy as np
from necredpy import Model


def test_estimation_nk():
    m = Model("dynare/credibility_nk_regimes.mod")

    # Generate synthetic data from the model
    np.random.seed(42)
    T = 80

    # Simulate from the model with some shocks
    shocks = {
        'eps_d': np.random.randn(T) * 0.5,
        'eps_s': np.random.randn(T) * 0.5,
        'eps_m': np.random.randn(T) * 0.25,
    }
    sim = m.simulate(shocks, T=T, credibility=False)
    data = sim[['y', 'pi', 'ii']]
    print("Simulated data shape:", data.shape)
    print("Data summary:\n", data.describe())

    # Run estimation with minimal samples
    print("\nStarting estimation (small run)...")
    result = m.estimate(
        data,
        obs_vars=['y', 'pi', 'ii'],
        shock_vars=['y', 'pi', 'ii'],
        tau=0.2,
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        seed=0,
        progress_bar=True,
    )

    print("\nEstimation completed!")
    print("Log-likelihood at posterior mean: %.2f" % result.log_lik)
    print("\nPosterior summary:")
    print(result.summary)
    print("\nCredibility path shape:", result.credibility.shape)
    print("Credibility range: [%.3f, %.3f]" % (
        result.credibility.min(), result.credibility.max()))

    # Basic sanity checks
    assert result.log_lik != 0.0, "Log-likelihood should be nonzero"
    assert len(result.credibility) == T, "Credibility path length mismatch"
    assert result.summary.shape[0] == 7, "Should have 7 estimated params"

    print("\nPASS: estimation_nk")


if __name__ == '__main__':
    test_estimation_nk()
