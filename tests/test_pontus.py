"""
Tests for the Pontus piecewise-linear solver.

Tests:
  1. Doubling and Pontus iteration converge to the same F
  2. The converged F satisfies CF^2 + BF + A = 0
  3. Backward recursion + forward simulation reproduce terminal-regime IRF
  4. Credibility NK model solves under baseline calibration
  5. Eigenvalues of F are inside the unit circle (stability)
"""

import numpy as np
from numpy.linalg import eigvals, norm
from necredpy.pontus import (solve_terminal_pontus, solve_terminal_doubling,
                              solve_terminal, backward_recursion, simulate_forward,
                              solve_endogenous)
from necredpy.models.credibility_nk import (build_model, build_model_with_cred,
                                             baseline_theta, make_switching_fn,
                                             make_switching_fn_cred)


def test_doubling_vs_pontus():
    """Doubling and Pontus iteration must produce the same F."""
    theta = baseline_theta()
    matrices_M1, _ = build_model(theta)
    A, B, C, D = matrices_M1

    F_pontus, Q_pontus, conv_p, iters_p = solve_terminal_pontus(A, B, C)
    F_doubl, Q_doubl, conv_d, iters_d = solve_terminal_doubling(A, B, C)

    assert conv_p, "Pontus iteration did not converge"
    assert conv_d, "Doubling did not converge"

    err_F = norm(F_pontus - F_doubl)
    err_Q = norm(Q_pontus - Q_doubl)

    print("  Pontus iterations: %d,  Doubling iterations: %d" % (iters_p, iters_d))
    print("  ||F_pontus - F_doubling|| = %.2e" % err_F)
    print("  ||Q_pontus - Q_doubling|| = %.2e" % err_Q)

    assert err_F < 1e-10, "F matrices differ: %.2e" % err_F
    assert err_Q < 1e-10, "Q matrices differ: %.2e" % err_Q


def test_matrix_equation():
    """F must satisfy A + BF + CF^2 = 0."""
    theta = baseline_theta()
    matrices_M1, _ = build_model(theta)
    A, B, C, D = matrices_M1

    F, Q = solve_terminal(A, B, C)
    residual = norm(A + B @ F + C @ F @ F)
    print("  ||A + BF + CF^2|| = %.2e" % residual)
    assert residual < 1e-10, "Matrix equation residual too large: %.2e" % residual


def test_stability():
    """All eigenvalues of F must be inside the unit circle."""
    theta = baseline_theta()
    matrices_M1, matrices_M2 = build_model(theta)

    for name, matrices in [("M1", matrices_M1), ("M2", matrices_M2)]:
        A, B, C, D = matrices
        F, Q = solve_terminal(A, B, C)
        eigs = eigvals(F)
        max_eig = max(abs(eigs))
        print("  %s: max |eigenvalue(F)| = %.6f" % (name, max_eig))
        print("    eigenvalues:", np.round(eigs, 6))
        assert max_eig < 1.0, "%s: eigenvalue outside unit circle: %.6f" % (name, max_eig)


def test_irf_terminal_regime():
    """IRF in the terminal regime: backward recursion with all-M1
    must reproduce solve_terminal result."""
    theta = baseline_theta()
    matrices_M1, matrices_M2 = build_model(theta)
    A1, B1, C1, D1 = matrices_M1

    F_term, Q_term = solve_terminal(A1, B1, C1)

    T = 40
    regime_seq = np.zeros(T, dtype=int)  # all M1

    F_path, E_path, Q_path = backward_recursion(
        regime_seq, F_term, matrices_M1, matrices_M2
    )

    # All F_t should equal F_terminal
    for t in range(T):
        err = norm(F_path[t] - F_term)
        assert err < 1e-10, "F_path[%d] differs from F_terminal: %.2e" % (t, err)

    # Simulate a cost-push shock (eps_s = 1 at t=0)
    epsilon = np.zeros((T, 4))
    epsilon[0, 1] = 1.0  # cost-push shock

    u_path = simulate_forward(F_path, E_path, Q_path, epsilon)

    # Check: u_1 = F * u_0, u_2 = F * u_1, etc (no more shocks)
    for t in range(1, T):
        u_expected = F_term @ u_path[t - 1]
        err = norm(u_path[t] - u_expected)
        assert err < 1e-10, "Forward sim differs at t=%d: %.2e" % (t, err)

    print("  Cost-push shock IRF (first 8 periods):")
    print("    y:   ", np.round(u_path[:8, 0], 4))
    print("    pi:  ", np.round(u_path[:8, 1], 4))
    print("    i:   ", np.round(u_path[:8, 2], 4))


def test_two_regime_exogenous():
    """Exogenous switching: M2 for first 6 periods, then M1.
    Must produce different IRFs from all-M1."""
    theta = baseline_theta()
    matrices_M1, matrices_M2 = build_model(theta)
    A1, B1, C1, D1 = matrices_M1

    F_term, Q_term = solve_terminal(A1, B1, C1)

    T = 40
    # M2 for first 6 periods
    regime_exo = np.zeros(T, dtype=int)
    regime_exo[:6] = 1

    F_path_exo, E_path_exo, Q_path_exo = backward_recursion(
        regime_exo, F_term, matrices_M1, matrices_M2
    )

    # All-M1
    regime_m1 = np.zeros(T, dtype=int)
    F_path_m1, E_path_m1, Q_path_m1 = backward_recursion(
        regime_m1, F_term, matrices_M1, matrices_M2
    )

    # Same shock
    epsilon = np.zeros((T, 4))
    epsilon[0, 1] = 1.0

    u_exo = simulate_forward(F_path_exo, E_path_exo, Q_path_exo, epsilon)
    u_m1 = simulate_forward(F_path_m1, E_path_m1, Q_path_m1, epsilon)

    # They should differ
    diff = norm(u_exo - u_m1)
    print("  ||u_exogenous_switch - u_all_M1|| = %.4f" % diff)
    assert diff > 0.01, "Two-regime and single-regime IRFs are too similar"

    # After the switch period, u_exo should converge back to zero
    assert abs(u_exo[-1, 0]) < 0.01, "y did not converge: %.4f" % u_exo[-1, 0]
    assert abs(u_exo[-1, 1]) < 0.01, "pi did not converge: %.4f" % u_exo[-1, 1]


def test_endogenous_switching():
    """Endogenous switching loop with a large cost-push shock."""
    theta = baseline_theta()
    matrices_M1, matrices_M2 = build_model(theta)

    switching_fn = make_switching_fn(
        epsilon_bar=theta['epsilon_bar'],
        k_restore=theta['k_restore'],
        pi_index=1
    )

    T = 60
    epsilon = np.zeros((T, 4))
    # Large cost-push shock that should trigger credibility loss
    epsilon[0, 1] = 3.0

    u_path, regime_seq, F_path, E_path, Q_path, converged, iters = \
        solve_endogenous(matrices_M1, matrices_M2, switching_fn, epsilon, T)

    print("  Endogenous switching converged: %s in %d iterations" % (converged, iters))
    print("  Regime sequence (first 20): ", regime_seq[:20])
    print("  Inflation path (first 12):  ", np.round(u_path[:12, 1], 3))
    print("  Output path (first 12):     ", np.round(u_path[:12, 0], 3))

    # Check convergence
    assert converged, "Endogenous switching did not converge"

    # Check that M2 was activated (the shock is big enough)
    assert np.any(regime_seq == 1), "No credibility loss detected with large shock"

    # Check return to steady state
    assert abs(u_path[-1, 1]) < 0.1, "Inflation did not return to SS"


def test_cred_5x5_solves():
    """5x5 system with credibility stock solves for both regimes."""
    theta = baseline_theta()
    matrices_M1, matrices_M2 = build_model_with_cred(theta)

    for name, matrices in [("M1", matrices_M1), ("M2", matrices_M2)]:
        A, B, C, D = matrices
        assert A.shape == (5, 5), "%s: A shape %s" % (name, A.shape)
        F, Q, conv, iters = solve_terminal_doubling(A, B, C)
        assert conv, "%s: 5x5 doubling did not converge" % name
        residual = norm(A + B @ F + C @ F @ F)
        print("  %s: converged in %d iters, ||A+BF+CF^2|| = %.2e" % (
            name, iters, residual))
        assert residual < 1e-10, "%s: residual %.2e" % (name, residual)


def test_cred_eigenvalues():
    """5th eigenvalue of F equals the cred decay rate."""
    theta = baseline_theta()
    matrices_M1, matrices_M2 = build_model_with_cred(theta)

    for name, matrices, expected_eig in [
        ("M1", matrices_M1, 1.0 - theta['delta_up']),
        ("M2", matrices_M2, 1.0 - theta['delta_down']),
    ]:
        A, B, C, D = matrices
        F, Q = solve_terminal(A, B, C)
        eigs = eigvals(F)
        print("  %s eigenvalues: %s" % (name, np.round(eigs, 6)))

        # One eigenvalue should match the cred rate
        diffs = np.abs(eigs - expected_eig)
        min_diff = np.min(diffs)
        print("  %s: closest eigenvalue to %.3f: diff = %.2e" % (
            name, expected_eig, min_diff))
        assert min_diff < 1e-10, (
            "%s: no eigenvalue matches cred rate %.3f" % (name, expected_eig))

        # All eigenvalues inside unit circle
        max_eig = max(abs(eigs))
        assert max_eig < 1.0, "%s: eigenvalue outside unit circle" % name


def test_cred_decoupled():
    """Upper-left 4x4 block of F_5x5 matches F_4x4.

    Since cred does not feed back into IS/PC/Taylor/identity within a
    regime, the first 4 variables should have identical dynamics whether
    or not cred is in the state vector.
    """
    theta = baseline_theta()
    matrices_M1_4, matrices_M2_4 = build_model(theta)
    matrices_M1_5, matrices_M2_5 = build_model_with_cred(theta)

    for name, mat4, mat5 in [
        ("M1", matrices_M1_4, matrices_M1_5),
        ("M2", matrices_M2_4, matrices_M2_5),
    ]:
        A4, B4, C4, D4 = mat4
        F4, Q4 = solve_terminal(A4, B4, C4)

        A5, B5, C5, D5 = mat5
        F5, Q5 = solve_terminal(A5, B5, C5)

        err = norm(F5[:4, :4] - F4)
        print("  %s: ||F5[:4,:4] - F4|| = %.2e" % (name, err))
        assert err < 1e-10, "%s: 4x4 block differs: %.2e" % (name, err)


def test_cred_switching():
    """Endogenous switching with shadow credibility capital converges."""
    theta = baseline_theta()
    matrices_M1, matrices_M2 = build_model(theta)

    switching_fn = make_switching_fn_cred(
        epsilon_bar=theta['epsilon_bar'],
        cred_threshold=theta['cred_threshold'],
        delta_up=theta['delta_up'],
        delta_down=theta['delta_down']
    )

    T = 80
    epsilon = np.zeros((T, 4))
    epsilon[0, 1] = 3.0  # large cost-push shock

    u_path, regime_seq, F_path, E_path, Q_path, converged, iters = \
        solve_endogenous(matrices_M1, matrices_M2, switching_fn, epsilon, T)

    print("  Cred switching converged: %s in %d iterations" % (converged, iters))
    print("  Regime sequence (first 25):", regime_seq[:25])

    # Shadow cred path from switching function
    cred_actual = switching_fn.cred_path
    print("  Cred path (first 15):     ", np.round(cred_actual[:15], 3))
    print("  Inflation (first 15):     ", np.round(u_path[:15, 1], 3))

    assert converged, "Cred-based switching did not converge"
    assert np.any(regime_seq == 1), "No M2 activation with large shock"

    # Cred should fall below threshold and recover
    min_cred = np.min(cred_actual)
    final_cred = cred_actual[-1]
    print("  Min cred = %.3f, final cred = %.3f" % (min_cred, final_cred))
    assert min_cred < theta['cred_threshold'], (
        "Cred did not fall below threshold: %.3f" % min_cred)
    assert final_cred > 0.85, "Cred did not recover: %.3f" % final_cred


def test_cred_dynamics():
    """Verify shadow cred stays in [0,1] and shows correct dynamics."""
    theta = baseline_theta()
    matrices_M1, matrices_M2 = build_model(theta)

    switching_fn = make_switching_fn_cred(
        epsilon_bar=theta['epsilon_bar'],
        cred_threshold=theta['cred_threshold'],
        delta_up=theta['delta_up'],
        delta_down=theta['delta_down']
    )

    # Very large shock to get extended M2
    T = 100
    epsilon = np.zeros((T, 4))
    epsilon[0, 1] = 5.0

    u_path, regime_seq, _, _, _, converged, _ = solve_endogenous(
        matrices_M1, matrices_M2, switching_fn, epsilon, T)

    cred_actual = switching_fn.cred_path

    # Boundedness
    assert np.all(cred_actual >= -0.01), "Cred went below 0: %.4f" % np.min(cred_actual)
    assert np.all(cred_actual <= 1.01), "Cred went above 1: %.4f" % np.max(cred_actual)

    # M2 should be activated
    assert np.any(regime_seq == 1), "No M2 with large shock"

    # Return to steady state
    assert abs(u_path[-1, 1]) < 0.1, "Inflation did not return to SS"
    assert cred_actual[-1] > 0.85, "Cred did not recover to near 1"

    print("  Cred bounded in [%.4f, %.4f]" % (np.min(cred_actual), np.max(cred_actual)))
    print("  M2 periods: %d" % np.sum(regime_seq == 1))
    print("  PASSED: cred dynamics correct")


if __name__ == '__main__':
    tests = [
        ("Doubling vs Pontus iteration", test_doubling_vs_pontus),
        ("Matrix equation CF^2+BF+A=0", test_matrix_equation),
        ("Stability (eigenvalues inside unit circle)", test_stability),
        ("IRF in terminal regime", test_irf_terminal_regime),
        ("Two-regime exogenous switching", test_two_regime_exogenous),
        ("Endogenous switching", test_endogenous_switching),
        ("5x5 cred system solves", test_cred_5x5_solves),
        ("5x5 cred eigenvalues", test_cred_eigenvalues),
        ("5x5 cred decoupled from 4x4", test_cred_decoupled),
        ("Cred-based endogenous switching", test_cred_switching),
        ("Cred dynamics and boundedness", test_cred_dynamics),
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
