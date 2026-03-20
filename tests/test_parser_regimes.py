"""
Tests for the enhanced Dynare parser with regimes block.

Tests:
  1. extract_regimes() correctly parses the regimes block (unresolved strings)
  2. parse_two_regime_model() returns correct matrices matching hand-built
  3. Switching function from parser matches make_switching_fn_cred()
  4. Full Pontus solve via parsed model matches existing solve
  5. Two-sector QPM parses and solves with regimes
  6. k_restore switching type works
  7. No regimes block raises clear error
  8. compile_two_regime_model + evaluate matches one-shot parse
  9. param_overrides flow through to switching function (estimability)
 10. Missing required switching param raises ValueError
"""

import os

import numpy as np
from numpy.linalg import norm, eigvals

from necredpy.utils.dynare_parser import (
    extract_regimes, build_switching_fn, parse_two_regime_model,
    compile_two_regime_model, evaluate_two_regime_model
)
from necredpy.models.credibility_nk import (
    build_model, baseline_theta, make_switching_fn_cred
)
from necredpy.pontus import solve_endogenous, solve_terminal


def _read_mod(filename):
    """Read a .mod file from the dynare/ directory."""
    mod_path = os.path.join(os.path.dirname(__file__), '..', 'dynare', filename)
    with open(mod_path, 'r') as f:
        return f.read()


# ---------------------------------------------------------------
# Test 1: extract_regimes parses correctly (unresolved strings)
# ---------------------------------------------------------------

def test_extract_regimes():
    """Regimes block is parsed into correct dict with parameter references
    stored as strings (resolution happens later in evaluate)."""
    mod_string = _read_mod('credibility_nk_regimes.mod')
    spec = extract_regimes(mod_string)

    assert spec is not None, "regimes block not found"
    assert spec['switch_type'] == 'shadow_cred'
    assert spec['monitor'] == 'pi'

    # Parameter references stored as strings (not yet resolved)
    assert spec['band'] == 'epsilon_bar', (
        "Expected string 'epsilon_bar', got %s" % repr(spec['band']))
    assert spec['M1_params']['omega'] == 'omega_H', (
        "Expected string 'omega_H', got %s" % repr(spec['M1_params']['omega']))
    assert spec['M2_params']['omega'] == 'omega_L', (
        "Expected string 'omega_L', got %s" % repr(spec['M2_params']['omega']))
    assert spec['delta_up'] == 'delta_up'
    assert spec['delta_down'] == 'delta_down'
    assert spec['cred_threshold'] == 'cred_threshold'

    # Literal numeric value stays as float
    assert spec['cred_init'] == 1.0

    print("  switch_type: %s" % spec['switch_type'])
    print("  monitor: %s, band: %s" % (spec['monitor'], spec['band']))
    print("  M1_params: %s" % spec['M1_params'])
    print("  M2_params: %s" % spec['M2_params'])


# ---------------------------------------------------------------
# Test 2: Matrices match hand-built credibility_nk
# ---------------------------------------------------------------

def test_matrices_match_hand_built():
    """Matrices from parse_two_regime_model match hand-built build_model()."""
    mod_string = _read_mod('credibility_nk_regimes.mod')
    matrices_M1_p, matrices_M2_p, _, info = parse_two_regime_model(mod_string)

    theta = baseline_theta()
    matrices_M1_h, matrices_M2_h = build_model(theta)

    for label, parsed, hand in [
        ("M1", matrices_M1_p, matrices_M1_h),
        ("M2", matrices_M2_p, matrices_M2_h),
    ]:
        for i, name in enumerate(['A', 'B', 'C', 'D']):
            err = norm(parsed[i] - hand[i])
            print("  %s %s: ||parsed - hand|| = %.2e" % (label, name, err))
            assert err < 1e-12, (
                "%s %s mismatch: %.2e" % (label, name, err)
            )


# ---------------------------------------------------------------
# Test 3: Switching function matches existing implementation
# ---------------------------------------------------------------

def test_switching_fn_matches():
    """Switching fn from parser produces same regime sequence as
    make_switching_fn_cred from credibility_nk.py."""
    mod_string = _read_mod('credibility_nk_regimes.mod')
    _, _, switching_fn_parsed, _ = parse_two_regime_model(mod_string)

    theta = baseline_theta()
    switching_fn_hand = make_switching_fn_cred(
        epsilon_bar=theta['epsilon_bar'],
        cred_threshold=theta['cred_threshold'],
        delta_up=theta['delta_up'],
        delta_down=theta['delta_down']
    )

    # Create a synthetic path with known inflation values
    T = 40
    u_path = np.zeros((T, 4))
    # Large inflation at t=0,1 (above band), then decaying
    u_path[0, 1] = 3.0
    u_path[1, 1] = 1.5
    u_path[2, 1] = 0.8
    for t in range(3, T):
        u_path[t, 1] = u_path[t-1, 1] * 0.7

    regime_parsed = switching_fn_parsed(u_path)
    regime_hand = switching_fn_hand(u_path)

    assert np.array_equal(regime_parsed, regime_hand), (
        "Regime sequences differ:\n  parsed: %s\n  hand:   %s"
        % (regime_parsed[:20], regime_hand[:20])
    )

    # Cred paths should also match
    cred_parsed = switching_fn_parsed.cred_path
    cred_hand = switching_fn_hand.cred_path
    err = norm(cred_parsed - cred_hand)
    print("  ||cred_parsed - cred_hand|| = %.2e" % err)
    assert err < 1e-14, "Cred paths differ: %.2e" % err

    print("  Regime seq (first 15): %s" % regime_parsed[:15])
    print("  Cred path  (first 15): %s" % np.round(cred_parsed[:15], 3))


# ---------------------------------------------------------------
# Test 4: Full Pontus solve via parsed model
# ---------------------------------------------------------------

def test_full_pontus_solve():
    """Full endogenous switching solve using parsed model matches
    solve using hand-built matrices."""
    mod_string = _read_mod('credibility_nk_regimes.mod')
    matrices_M1_p, matrices_M2_p, sw_fn_p, info = parse_two_regime_model(mod_string)

    theta = baseline_theta()
    matrices_M1_h, matrices_M2_h = build_model(theta)
    sw_fn_h = make_switching_fn_cred(
        epsilon_bar=theta['epsilon_bar'],
        cred_threshold=theta['cred_threshold'],
        delta_up=theta['delta_up'],
        delta_down=theta['delta_down']
    )

    T = 60
    epsilon = np.zeros((T, 4))
    epsilon[0, 1] = -3.0  # cost-push shock

    u_parsed, reg_p, _, _, _, conv_p, iters_p = solve_endogenous(
        matrices_M1_p, matrices_M2_p, sw_fn_p, epsilon, T)
    u_hand, reg_h, _, _, _, conv_h, iters_h = solve_endogenous(
        matrices_M1_h, matrices_M2_h, sw_fn_h, epsilon, T)

    assert conv_p, "Parsed model did not converge"
    assert conv_h, "Hand model did not converge"

    err = norm(u_parsed - u_hand)
    print("  ||u_parsed - u_hand|| = %.2e" % err)
    print("  Parsed iters: %d, Hand iters: %d" % (iters_p, iters_h))
    print("  Regime match: %s" % np.array_equal(reg_p, reg_h))
    assert err < 1e-12, "Solution paths differ: %.2e" % err
    assert np.array_equal(reg_p, reg_h), "Regime sequences differ"


# ---------------------------------------------------------------
# Test 5: Two-sector QPM parses and solves
# ---------------------------------------------------------------

def test_two_sector_qpm():
    """Two-sector QPM with regimes parses and the Pontus solver converges."""
    mod_string = _read_mod('two_sector_qpm_regimes.mod')
    matrices_M1, matrices_M2, switching_fn, info = parse_two_regime_model(
        mod_string)

    n = len(info['var_names'])
    print("  Variables (%d): %s" % (n, info['var_names']))
    print("  Shocks: %s" % info['shock_names'])
    print("  Monitor: %s" % info['regime_spec']['monitor'])

    # Check dimensions
    A1, B1, C1, D1 = matrices_M1
    assert A1.shape == (n, n), "A1 shape: %s" % str(A1.shape)

    # Both regimes should solve
    for label, matrices in [("M1", matrices_M1), ("M2", matrices_M2)]:
        A, B, C, D = matrices
        F, Q = solve_terminal(A, B, C)
        eigs = eigvals(F)
        max_eig = max(abs(eigs))
        print("  %s: max |eig(F)| = %.6f" % (label, max_eig))
        assert max_eig < 1.0, "%s: eigenvalue outside unit circle" % label

    # Endogenous switching solve with upstream supply shock
    T = 60
    epsilon = np.zeros((T, n))
    # eps_a1 is the upstream supply shock
    shock_idx = info['shock_names'].index('eps_a1')
    epsilon[0, shock_idx] = -0.05  # TFP shock (inflationary)

    u_path, regime_seq, _, _, _, converged, iters = solve_endogenous(
        matrices_M1, matrices_M2, switching_fn, epsilon, T)

    print("  Converged: %s in %d iters" % (converged, iters))
    print("  pi_agg (first 10): %s" % np.round(
        u_path[:10, info['var_names'].index('pi_agg')], 4))
    assert converged, "Two-sector model did not converge"


# ---------------------------------------------------------------
# Test 6: k_restore switching type
# ---------------------------------------------------------------

def test_k_restore_switching():
    """k_restore switching type parses and works correctly."""
    mod_string = """
    var y pi ii pi_lag;
    varexo eps_d eps_s eps_m;
    parameters beta sigma kappa rho_i phi_pi phi_y omega;
    beta=0.99; sigma=1.0; kappa=0.3; rho_i=0.7;
    phi_pi=1.5; phi_y=0.5; omega=0.65;

    model;
      y = y(+1) - sigma*(ii - pi(+1)) + eps_d;
      pi = omega*beta*pi(+1) + (1-omega)*pi_lag + kappa*y + eps_s;
      ii = rho_i*ii(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + eps_m;
      pi_lag = pi(-1);
    end;

    regimes;
      M1: omega = 0.65;
      M2: omega = 0.35;
      switch: k_restore;
      monitor: pi;
      band: 2.0;
      k_restore: 4;
    end;
    """
    matrices_M1, matrices_M2, switching_fn, info = parse_two_regime_model(
        mod_string)

    assert info['regime_spec']['switch_type'] == 'k_restore'

    # Solve with a large shock
    T = 60
    epsilon = np.zeros((T, 4))
    epsilon[0, 1] = -3.0

    u_path, regime_seq, _, _, _, converged, iters = solve_endogenous(
        matrices_M1, matrices_M2, switching_fn, epsilon, T)

    print("  Converged: %s in %d iters" % (converged, iters))
    print("  Regime (first 20): %s" % regime_seq[:20])
    assert converged, "k_restore model did not converge"
    assert np.any(regime_seq == 1), "No M2 activation"


# ---------------------------------------------------------------
# Test 7: No regimes block raises clear error
# ---------------------------------------------------------------

def test_no_regimes_raises():
    """parse_two_regime_model raises ValueError if no regimes block."""
    mod_string = """
    var y pi ii pi_lag;
    varexo eps_d eps_s eps_m;
    parameters beta sigma kappa rho_i phi_pi phi_y omega;
    beta=0.99; sigma=1.0; kappa=0.3; rho_i=0.7;
    phi_pi=1.5; phi_y=0.5; omega=0.65;

    model;
      y = y(+1) - sigma*(ii - pi(+1)) + eps_d;
      pi = omega*beta*pi(+1) + (1-omega)*pi_lag + kappa*y + eps_s;
      ii = rho_i*ii(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + eps_m;
      pi_lag = pi(-1);
    end;
    """
    try:
        parse_two_regime_model(mod_string)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print("  Correctly raised: %s" % e)


# ---------------------------------------------------------------
# Test 8: compile + evaluate matches one-shot parse
# ---------------------------------------------------------------

def test_compile_evaluate_separation():
    """compile_two_regime_model + evaluate_two_regime_model gives the
    same result as the one-shot parse_two_regime_model."""
    mod_string = _read_mod('credibility_nk_regimes.mod')

    # One-shot
    M1_full, M2_full, sw_full, info_full = parse_two_regime_model(mod_string)

    # Compile + evaluate
    compiled = compile_two_regime_model(mod_string)
    M1_ce, M2_ce, sw_ce, info_ce = evaluate_two_regime_model(compiled)

    for i in range(4):
        err1 = norm(M1_full[i] - M1_ce[i])
        err2 = norm(M2_full[i] - M2_ce[i])
        assert err1 < 1e-14, "M1[%d] mismatch: %.2e" % (i, err1)
        assert err2 < 1e-14, "M2[%d] mismatch: %.2e" % (i, err2)

    # Switching fn should give same result on a test path
    T = 30
    u_path = np.zeros((T, 4))
    u_path[0, 1] = 3.0
    for t in range(1, T):
        u_path[t, 1] = u_path[t-1, 1] * 0.8
    reg_full = sw_full(u_path)
    reg_ce = sw_ce(u_path)
    assert np.array_equal(reg_full, reg_ce), "Regime sequences differ"
    print("  Compile+evaluate matches one-shot: OK")


# ---------------------------------------------------------------
# Test 9: param_overrides flow through (estimability)
# ---------------------------------------------------------------

def test_param_overrides_estimable():
    """param_overrides change switching behavior and matrices correctly.
    This verifies the estimation-ready architecture works."""
    mod_string = _read_mod('credibility_nk_regimes.mod')
    compiled = compile_two_regime_model(mod_string)

    # Tight band -> more M2 activation
    _, _, sw_tight, info_tight = evaluate_two_regime_model(
        compiled, param_overrides={"epsilon_bar": 0.5})
    # Loose band -> less M2 activation
    _, _, sw_loose, info_loose = evaluate_two_regime_model(
        compiled, param_overrides={"epsilon_bar": 5.0})

    # Verify resolved band values
    assert info_tight['regime_spec']['band'] == 0.5, (
        "Expected band=0.5, got %s" % info_tight['regime_spec']['band'])
    assert info_loose['regime_spec']['band'] == 5.0, (
        "Expected band=5.0, got %s" % info_loose['regime_spec']['band'])

    # Test with a path that has moderate inflation
    T = 40
    u_path = np.zeros((T, 4))
    u_path[0, 1] = 2.0
    for t in range(1, T):
        u_path[t, 1] = u_path[t-1, 1] * 0.8

    reg_tight = sw_tight(u_path)
    reg_loose = sw_loose(u_path)

    m2_tight = np.sum(reg_tight)
    m2_loose = np.sum(reg_loose)
    print("  M2 periods (band=0.5): %d" % m2_tight)
    print("  M2 periods (band=5.0): %d" % m2_loose)
    assert m2_tight > m2_loose, (
        "Tighter band should cause more M2 periods: %d vs %d"
        % (m2_tight, m2_loose))

    # Verify omega overrides also change with param_overrides
    M1_new, M2_new, _, info_new = evaluate_two_regime_model(
        compiled, param_overrides={"omega_H": 0.80, "omega_L": 0.20})
    assert info_new['params_M1']['omega'] == 0.80, (
        "omega not updated in M1: %s" % info_new['params_M1']['omega'])
    assert info_new['params_M2']['omega'] == 0.20, (
        "omega not updated in M2: %s" % info_new['params_M2']['omega'])
    print("  param_overrides flow correctly: OK")


# ---------------------------------------------------------------
# Test 10: Missing required switching param raises error
# ---------------------------------------------------------------

def test_missing_switch_param_raises():
    """Missing required switching parameter raises ValueError."""
    mod_string = """
    var y pi ii pi_lag;
    varexo eps_d eps_s eps_m;
    parameters beta sigma kappa rho_i phi_pi phi_y omega;
    beta=0.99; sigma=1.0; kappa=0.3; rho_i=0.7;
    phi_pi=1.5; phi_y=0.5; omega=0.65;

    model;
      y = y(+1) - sigma*(ii - pi(+1)) + eps_d;
      pi = omega*beta*pi(+1) + (1-omega)*pi_lag + kappa*y + eps_s;
      ii = rho_i*ii(-1) + (1-rho_i)*(phi_pi*pi + phi_y*y) + eps_m;
      pi_lag = pi(-1);
    end;

    regimes;
      M1: omega = 0.65;
      M2: omega = 0.35;
      switch: shadow_cred;
      monitor: pi;
      band: 2.0;
    end;
    """
    # Missing delta_up and delta_down
    try:
        parse_two_regime_model(mod_string)
        assert False, "Should have raised ValueError for missing delta_up"
    except ValueError as e:
        assert 'delta_up' in str(e), "Error should mention delta_up: %s" % e
        print("  Correctly raised: %s" % e)


if __name__ == '__main__':
    tests = [
        ("Extract regimes block (unresolved strings)", test_extract_regimes),
        ("Matrices match hand-built", test_matrices_match_hand_built),
        ("Switching fn matches existing", test_switching_fn_matches),
        ("Full Pontus solve via parser", test_full_pontus_solve),
        ("Two-sector QPM parses and solves", test_two_sector_qpm),
        ("k_restore switching type", test_k_restore_switching),
        ("No regimes block raises error", test_no_regimes_raises),
        ("Compile+evaluate matches one-shot", test_compile_evaluate_separation),
        ("param_overrides estimable", test_param_overrides_estimable),
        ("Missing switch param raises", test_missing_switch_param_raises),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        print("\nTest: %s" % name)
        try:
            fn()
            print("  PASSED")
            passed += 1
        except Exception as e:
            print("  FAILED: %s" % e)
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 50)
    print("%d passed, %d failed" % (passed, failed))
