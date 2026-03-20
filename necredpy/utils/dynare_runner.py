"""
Run Dynare via Octave subprocess for linear and perfect foresight solutions.

Provides:
  - run_stoch_simul(): Linear RE IRFs via Dynare's stoch_simul(order=1)
  - run_perfect_foresight(): Perfect foresight solution via Dynare

Both call Octave as a subprocess and load results from .mat files.
"""

import subprocess
import os
import re
import numpy as np
import scipy.io

DYNARE_PATH = os.path.expanduser('~/home/dynare/dynare7/lib/dynare/matlab/')
DYNARE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'dynare')
DYNARE_DIR = os.path.abspath(DYNARE_DIR)


def _run_octave(script, cwd=None, timeout=120):
    """Run an Octave script string via subprocess."""
    if cwd is None:
        cwd = DYNARE_DIR

    result = subprocess.run(
        ['octave', '--no-gui', '--no-window-system', '--eval', script],
        cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Octave failed (code {result.returncode}):\n"
            f"STDOUT: {result.stdout[-1000:]}\n"
            f"STDERR: {result.stderr[-1000:]}"
        )
    return result


def run_stoch_simul(shock_name='eps_s', shock_size=3.0, T=40,
                    mod_file='credibility_nk', param_overrides=None):
    """Run Dynare stoch_simul and simulate an IRF.

    Parameters
    ----------
    shock_name : str
        Name of the exogenous shock.
    shock_size : float
        Magnitude of the shock.
    T : int
        IRF horizon.
    mod_file : str
        Name of .mod file (without extension).
    param_overrides : dict or None
        Optional parameter overrides (e.g., {'omega': 0.5}).

    Returns
    -------
    irf : dict
        {var_name: ndarray(T,)} for each endogenous variable.
    """
    # If param overrides, write a temp .mod file with modified values
    actual_mod = mod_file
    tmp_mod_path = None
    if param_overrides:
        mod_path = os.path.join(DYNARE_DIR, f'{mod_file}.mod')
        with open(mod_path) as f:
            mod_content = f.read()
        for k, v in param_overrides.items():
            mod_content = _replace_param(mod_content, k, v)
        tmp_mod_path = os.path.join(DYNARE_DIR, '_tmp_ss.mod')
        with open(tmp_mod_path, 'w') as f:
            f.write(mod_content)
        actual_mod = '_tmp_ss'

    script = f"""
mod_file = '{actual_mod}';
shock_name = '{shock_name}';
shock_size = {shock_size};
T_irf = {T};
run('run_stoch_simul.m');
"""

    try:
        _run_octave(script)
    finally:
        if tmp_mod_path and os.path.exists(tmp_mod_path):
            os.remove(tmp_mod_path)

    # Load results
    mat = scipy.io.loadmat(
        os.path.join(DYNARE_DIR, 'output', 'stoch_simul_irf.mat')
    )
    irf_data = mat['irf_data'][0, 0]

    irf = {}
    for name in irf_data.dtype.names:
        irf[name] = irf_data[name][:, 0]

    return irf


def run_perfect_foresight(mod_file='credibility_pf', shock_name=None,
                          shock_size=None, T=None, steep=None,
                          param_overrides=None, histval=None):
    """Run Dynare perfect foresight solver.

    By default uses the shock/periods configuration in the .mod file.
    Optional overrides can modify the .mod file before running.

    Parameters
    ----------
    mod_file : str
        Name of .mod file (without extension).
    shock_name : str or None
        If provided, override shock in .mod file.
    shock_size : float or None
        If provided, override shock size.
    T : int or None
        If provided, override number of periods.
    steep : float or None
        If provided, override sigmoid steepness.
    param_overrides : dict or None
        Other parameter overrides.
    histval : dict or None
        Initial conditions: {var_name: value} for period 0 (Dynare convention).
        Adds a histval block to override the initial steady state.

    Returns
    -------
    pf : dict
        {var_name: ndarray(n_periods,)} for each endogenous variable.
    """
    mod_path = os.path.join(DYNARE_DIR, f'{mod_file}.mod')

    # Read original .mod file
    with open(mod_path) as f:
        mod_content = f.read()

    # Apply overrides by writing a temporary .mod file
    modified = mod_content
    if steep is not None:
        modified = _replace_param(modified, 'steep', steep)
    if param_overrides:
        for k, v in param_overrides.items():
            modified = _replace_param(modified, k, v)
    if shock_size is not None and shock_name is not None:
        modified = _replace_shock(modified, shock_name, shock_size)
    if T is not None:
        modified = modified.replace(
            'perfect_foresight_setup(periods=80)',
            f'perfect_foresight_setup(periods={T})'
        )
    if histval is not None:
        # Replace initval block values to set initial conditions.
        # In Dynare PF, initval sets the pre-sample (period 0) values.
        # We replace values in the initval block AND remove the first
        # steady; command so Dynare uses our custom values instead of
        # computing the steady state (which would overwrite them).
        iv_match = re.search(r'initval;(.*?)end;', modified, re.DOTALL)
        if iv_match:
            iv_body = iv_match.group(1)
            for var, val in histval.items():
                iv_body = re.sub(
                    rf'{var}\s*=\s*[^;]+;',
                    f'{var} = {val};',
                    iv_body
                )
            modified = modified[:iv_match.start(1)] + iv_body + modified[iv_match.end(1):]

            # Remove the first 'steady;' after initval (which would
            # overwrite our custom values with the computed SS).
            # The second steady; (after endval) is kept.
            iv_end_pos = iv_match.end()
            first_steady = re.search(r'\bsteady\s*;', modified[iv_end_pos:])
            if first_steady:
                s_start = iv_end_pos + first_steady.start()
                s_end = iv_end_pos + first_steady.end()
                modified = modified[:s_start] + modified[s_end:]

    # Write temporary .mod file
    tmp_mod = os.path.join(DYNARE_DIR, '_tmp_pf.mod')
    with open(tmp_mod, 'w') as f:
        f.write(modified)

    script = f"""
mod_file = '_tmp_pf';
run('run_pf.m');
"""

    try:
        _run_octave(script)
    finally:
        # Clean up temp file
        if os.path.exists(tmp_mod):
            os.remove(tmp_mod)

    # Load results
    mat = scipy.io.loadmat(
        os.path.join(DYNARE_DIR, 'output', 'perfect_foresight.mat')
    )
    pf_data = mat['pf_data'][0, 0]

    pf = {}
    for name in pf_data.dtype.names:
        pf[name] = pf_data[name][:, 0]

    return pf


def _replace_param(mod_content, param_name, value):
    """Replace a parameter assignment in .mod content."""
    pattern = rf'(\b{param_name}\s*=\s*)[^;]+(;)'
    replacement = rf'\g<1>{value}\2'
    return re.sub(pattern, replacement, mod_content)


def _replace_shock(mod_content, shock_name, shock_size):
    """Replace shock value in .mod content."""
    # Pattern: var shock_name; ... values X;
    pattern = rf'(var\s+{shock_name}\s*;.*?values\s+)[^;]+(;)'
    replacement = rf'\g<1>{shock_size}\2'
    return re.sub(pattern, replacement, mod_content, flags=re.DOTALL)


