"""Regression test: lock in baseline_4sec.mod parser + solver outputs.

This test pins the parser-output dimensions, terminal-solve residual,
and dominant eigenvalue at the calibrated parameter values for the
canonical reference model. Any change to the parser or solver that
silently shifts these values will trip this test.

Run from any directory:
    pytest tests/test_baseline_4sec_regression.py -v

Baseline established 2026-05-20 on master (commit 58e0300):
    38 vars, 14 shocks
    QME residual ~ 3.33e-16
    max|eig(F)| = 0.950818 at omega = 0.9
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from necredpy.jax_model import compile_jax_model, solve_terminal_jax


# The model file lives in network_credibility, not in necredpy. We resolve
# it via a relative path that holds in the standard repo layout where
# both packages sit side-by-side under work/credibility/.
HERE = os.path.dirname(os.path.abspath(__file__))
MOD_FILE = os.path.normpath(os.path.join(
    HERE, "..", "..", "network_credibility",
    "dynare", "experiments", "baseline_4sec.mod"))


def _load_model():
    with open(MOD_FILE) as f:
        mod_string = f.read()
    return compile_jax_model(mod_string, verbose=False)


def test_baseline_4sec_dimensions():
    """Parser produces 38 declared/aux variables and 14 structural shocks."""
    model = _load_model()
    assert model["n_vars"] == 38, (
        f"n_vars regression: expected 38, got {model['n_vars']}")
    assert model["n_shocks"] == 14, (
        f"n_shocks regression: expected 14, got {model['n_shocks']}")


def test_baseline_4sec_terminal_solve():
    """Terminal solve at omega_H is stable, QME residual is machine-zero."""
    model = _load_model()
    p = dict(model["param_defaults"])
    omega_H = p["omega_H"]

    A, B, C = model["build_ABC"](p, jnp.array(omega_H))
    F = solve_terminal_jax(
        jnp.array(np.asarray(A)),
        jnp.array(np.asarray(B)),
        jnp.array(np.asarray(C)))
    F_np = np.asarray(F)

    assert np.all(np.isfinite(F_np)), "F has non-finite entries"

    residual = (np.asarray(A)
                + np.asarray(B) @ F_np
                + np.asarray(C) @ F_np @ F_np)
    res_norm = np.max(np.abs(residual))
    assert res_norm < 1e-12, (
        f"QME residual regression: expected < 1e-12, got {res_norm:.3e}")


def test_baseline_4sec_dominant_eigenvalue():
    """max|eig(F)| at omega_H matches the recorded baseline within tol."""
    model = _load_model()
    p = dict(model["param_defaults"])
    omega_H = p["omega_H"]

    A, B, C = model["build_ABC"](p, jnp.array(omega_H))
    F = solve_terminal_jax(
        jnp.array(np.asarray(A)),
        jnp.array(np.asarray(B)),
        jnp.array(np.asarray(C)))
    F_np = np.asarray(F)

    max_eig = np.max(np.abs(np.linalg.eigvals(F_np)))
    # Recorded baseline 2026-05-20: 0.950818
    assert abs(max_eig - 0.950818) < 1e-5, (
        f"max|eig(F)| regression: expected 0.950818, got {max_eig:.6f}")
    assert max_eig < 1.0, f"Eigenvalue not inside unit circle: {max_eig:.6f}"


# ---------------------------------------------------------------------------
# Coverage test for the lead-aux mechanism that the commitment-channel
# extension (commitment_channel_revision.md) relies on. Documents the
# existing behaviour discovered during scoping on 2026-05-20.
# ---------------------------------------------------------------------------

def test_lead_aux_expansion_through_credibility_block():
    """Multi-step leads compose with a model-side aux variable that is
    consumed as a credibility-block input. This is the structural pattern
    used by Extension A of the commitment-channel revision. If this test
    breaks, both the multi-horizon Taylor rule and the forward-path
    credibility signal lose their backing.
    """
    mod_text = """
    var y i pi_cpi m_t_lead_avg;
    varexo eps_y eps_i eps_pi;
    parameters rho_y rho_pi phi_pi psi_cred sigma_s eps_bar omega_L omega_H;

    rho_y = 0.9;  rho_pi = 0.5;  phi_pi = 1.5;
    psi_cred = 0.94; sigma_s = 1.0; eps_bar = 1.27;
    omega_L = 0.05; omega_H = 0.9;

    credibility;
        var s_t cred_state omega_pc;
        input m_t_lead_avg;
        output omega_pc;
        s_t = exp(-(max(abs(m_t_lead_avg) - eps_bar, 0)/sigma_s)^2);
        cred_state = psi_cred*cred_state(-1) + (1-psi_cred)*s_t;
        omega_pc = omega_L + (omega_H - omega_L)*cred_state;
    end;

    model(pwl);
    y = rho_y*y(-1) + eps_y;
    i = phi_pi*pi_cpi(+1) + eps_i;
    pi_cpi = omega_pc*rho_pi*pi_cpi(-1) + 0.1*y + eps_pi;
    m_t_lead_avg = (pi_cpi(+1) + pi_cpi(+2) + pi_cpi(+3) + pi_cpi(+4))/4;
    end;
    """

    model = compile_jax_model(mod_text, verbose=False)

    # Three aux variables added for the chain pi_cpi(+2..+4).
    expected_aux = ['aux_pi_cpi_lead_p1',
                    'aux_pi_cpi_lead_p2',
                    'aux_pi_cpi_lead_p3']
    for aux in expected_aux:
        assert aux in model['var_names'], (
            f"Aux variable {aux} missing from compiled model; "
            f"lead expansion regressed.")

    # Terminal solve must succeed and stay stable.
    p = dict(model["param_defaults"])
    A, B, C = model["build_ABC"](p, jnp.array(p["omega_H"]))
    F = solve_terminal_jax(jnp.array(np.asarray(A)),
                           jnp.array(np.asarray(B)),
                           jnp.array(np.asarray(C)))
    F_np = np.asarray(F)
    assert np.all(np.isfinite(F_np))
    res = np.asarray(A) + np.asarray(B) @ F_np + np.asarray(C) @ F_np @ F_np
    assert np.max(np.abs(res)) < 1e-12
    assert np.max(np.abs(np.linalg.eigvals(F_np))) < 0.9999
