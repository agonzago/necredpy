"""Tests for pluggable credibility functions."""

import numpy as np
from necredpy import Model


def test_backward_compat():
    """Old regimes; block still works (no credibility; block)."""
    m = Model("dynare/five_sector_network.mod")
    irf = m.irf("eps_pi1", size=5.0, T=60, credibility=True)
    assert irf is not None
    assert len(irf['pi_agg']) == 60
    assert 'credibility' in irf.columns
    print("PASS: backward_compat")


def test_banrep_gaussian():
    """BanRep Gaussian credibility; block produces different dynamics."""
    m = Model("dynare/five_sector_banrep_cred.mod")
    irf = m.irf("eps_pi1", size=5.0, T=60, credibility=True)
    assert irf is not None
    assert 'credibility' in irf.columns

    # Compare against Isard (old regimes; block)
    irf_isard = Model("dynare/five_sector_network.mod").irf(
        "eps_pi1", size=5.0, T=60, credibility=True)

    # They should NOT be identical
    diff = np.max(np.abs(irf['pi_agg'].values - irf_isard['pi_agg'].values))
    assert diff > 0.01, "BanRep and Isard should produce different IRFs (diff=%.6f)" % diff
    print("PASS: banrep_gaussian (max diff = %.4f)" % diff)


def test_banrep_cred_init_override():
    """Overriding cred_init works with the credibility; block."""
    m = Model("dynare/five_sector_banrep_cred.mod")
    irf_full = m.irf("eps_pi1", size=5.0, T=60, credibility=True, cred_init=1.0)
    irf_low = m.irf("eps_pi1", size=5.0, T=60, credibility=True, cred_init=0.3)

    # Starting from lower credibility should produce different paths
    diff = np.max(np.abs(irf_full['pi_agg'].values - irf_low['pi_agg'].values))
    assert diff > 0.01, "cred_init override should change dynamics (diff=%.6f)" % diff
    print("PASS: banrep_cred_init_override (max diff = %.4f)" % diff)


def test_explicit_isard():
    """Explicit Isard in credibility; block matches old regimes; version."""
    # Read the base mod file and append a credibility block with explicit Isard
    with open("dynare/five_sector_network.mod", 'r') as f:
        base_mod = f.read()

    mod_with_cred = base_mod + """
credibility;
  monitor: pi_agg;
  threshold: 0.5;
  cred_init: 1.0;
  accumulation = cred + delta_up*(1-cred)*(1-miss) - delta_down*cred*miss;
end;
"""

    # Write temp file
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mod', delete=False) as f:
        f.write(mod_with_cred)
        tmp_path = f.name

    try:
        m_cred = Model(tmp_path)
        irf_cred = m_cred.irf("eps_pi1", size=5.0, T=60, credibility=True)

        m_old = Model("dynare/five_sector_network.mod")
        irf_old = m_old.irf("eps_pi1", size=5.0, T=60, credibility=True)

        diff = np.max(np.abs(
            irf_cred['pi_agg'].values - irf_old['pi_agg'].values))
        assert diff < 1e-10, (
            "Explicit Isard credibility; block should match regimes; "
            "version exactly (diff=%.2e)" % diff)
        print("PASS: explicit_isard (max diff = %.2e)" % diff)
    finally:
        os.unlink(tmp_path)


if __name__ == '__main__':
    test_backward_compat()
    test_banrep_gaussian()
    test_banrep_cred_init_override()
    test_explicit_isard()
    print("\nALL TESTS PASSED")
