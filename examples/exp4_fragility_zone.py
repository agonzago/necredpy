"""
Experiment 4: Credibility fragility -- how initial credibility affects outcomes.

PURPOSE
-------
Shows the "double amplification" effect. Compares shock responses starting
from full credibility (cred=1.0) vs impaired credibility (cred=0.30).

With impaired credibility, the Phillips curves have lower forward-looking
weights (omega_L * lambda_j instead of omega_H * lambda_j), making inflation
more persistent. The same shock generates up to 45% more cumulative inflation.

HOW IT WORKS
------------
1. load_model() parses the .mod file (same as other experiments).
2. solve_pl() is called with cred_init=1.0 and cred_init=0.30 for each
   sector. The cred_init parameter overrides the initial credibility level
   in the switching function.
3. When cred_init=0.30 < cred_threshold=0.50, the economy starts in M2
   (low credibility). Even without a shock, the Phillips curves are more
   backward-looking. A shock on top of this makes inflation worse.

TO MODIFY
---------
- To change calibration: edit dynare/five_sector_network.mod
- To change initial credibility levels: modify cred_levels below
- To change shock size: modify the shock variable below
- SECTOR_NAMES and SECTOR_COLORS are defined in src/network_helpers.py

Produces: figures/fig15_five_sector_fragility.png

Usage: .venv/bin/python scripts/exp4_fragility_zone.py
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from necredpy.network_helpers import (load_model, solve_pl, find_threshold, fig_path,
                                       SECTOR_NAMES, SECTOR_COLORS)

def main():
    M1, M2, info, mod = load_model()
    vn = info['var_names']
    pai = vn.index('pi_agg')
    yi = vn.index('y')
    epsilon_bar = info['regime_spec']['band']
    T = 60; t = np.arange(T)

    # Two starting points: full credibility vs impaired
    cred_levels = [1.0, 0.30]
    cred_labels = ['Full Credibility (cred=1.0)', 'Impaired (cred=0.30)']
    cred_styles = ['-', '--']  # solid = full, dashed = impaired

    shock = 5.0  # moderate shock that shows differences

    print("=" * 70)
    print("EXPERIMENT 4: Credibility Fragility (M1 vs M2 start)")
    print("=" * 70)
    print("Shock size = %.1f" % shock)
    print()

    # --- Part A: IRF comparison for all 5 sectors ---
    fig, axes = plt.subplots(5, 3, figsize=(16, 18))

    for j in range(5):
        sn = 'eps_pi%d' % (j+1)
        color = SECTOR_COLORS[j]
        print("  %s:" % SECTOR_NAMES[j])

        for ci, (c0, clabel, cstyle) in enumerate(
                zip(cred_levels, cred_labels, cred_styles)):

            # solve_pl with cred_init overrides the starting credibility.
            # When c0=0.30 < threshold=0.50, the economy starts in M2.
            u, reg, cred, _ = solve_pl(M1, M2, mod, sn, shock, info, T,
                                       cred_init=c0)
            m2 = np.sum(reg == 1)
            cum_pi = np.sum(np.abs(u[:T, pai]))
            cum_y = np.sum(np.abs(u[:T, yi]))
            cred_min = np.min(cred[:T]) if cred is not None else c0
            print("    %s: cum_pi=%.2f cum_y=%.2f M2=%d cred_min=%.3f" % (
                clabel, cum_pi, cum_y, m2, cred_min))

            alpha = 1.0 if ci == 0 else 0.7

            # Column 0: Aggregate inflation
            ax = axes[j, 0]
            ax.plot(t, u[:T, pai], cstyle, color=color, lw=2, alpha=alpha,
                    label=clabel if j == 0 else '')
            if ci == 1:  # shade M2 periods for the impaired case
                for ts in range(T):
                    if reg[ts] == 1:
                        ax.axvspan(ts-0.5, ts+0.5, alpha=0.04, color=color)

            # Column 1: Credibility path
            ax = axes[j, 1]
            if cred is not None:
                ax.plot(t, cred[:T], cstyle, color=color, lw=2, alpha=alpha,
                        label=clabel if j == 0 else '')

            # Column 2: Cumulative |output gap|
            ax = axes[j, 2]
            ax.plot(t, np.cumsum(np.abs(u[:T, yi])), cstyle, color=color,
                    lw=2, alpha=alpha, label=clabel if j == 0 else '')

        # Format columns
        ax = axes[j, 0]
        ax.axhline(epsilon_bar, color='gray', lw=0.8, ls='--', alpha=0.4)
        ax.axhline(-epsilon_bar, color='gray', lw=0.8, ls='--', alpha=0.4)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        if j == 0:
            ax.set_title('Aggregate Inflation', fontsize=10)
            ax.legend(fontsize=6, loc='upper right')
        ax.set_ylabel(SECTOR_NAMES[j], fontsize=9, fontweight='bold')
        if j == 4: ax.set_xlabel('Quarter')
        ax.grid(True, alpha=0.3)

        ax = axes[j, 1]
        ax.axhline(info['regime_spec']['cred_threshold'], color='gray',
                   lw=0.8, ls='--', alpha=0.4)
        ax.set_ylim(-0.05, 1.1)
        if j == 0:
            ax.set_title('Credibility', fontsize=10)
            ax.legend(fontsize=6, loc='lower right')
        if j == 4: ax.set_xlabel('Quarter')
        ax.grid(True, alpha=0.3)

        ax = axes[j, 2]
        if j == 0:
            ax.set_title('Cumulative |Output Gap|', fontsize=10)
        if j == 4: ax.set_xlabel('Quarter')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'Double Amplification: Network x Credibility (shock=%.1f)\n'
        'Solid = Full credibility, Dashed = Impaired credibility' % shock,
        fontsize=13, y=1.02)
    plt.tight_layout()
    out = fig_path('fig15_five_sector_fragility.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print("\nSaved: %s" % out)

    # --- Part B: Summary table ---
    print()
    print("=" * 70)
    print("SUMMARY: Cumulative inflation increase from impaired credibility")
    print("=" * 70)
    print("%-12s %10s %10s %10s" % ('Sector', 'cum_pi M1', 'cum_pi M2',
                                     'Increase'))
    for j in range(5):
        sn = 'eps_pi%d' % (j+1)
        u1, _, _, _ = solve_pl(M1, M2, mod, sn, shock, info, T, cred_init=1.0)
        u2, _, _, _ = solve_pl(M1, M2, mod, sn, shock, info, T, cred_init=0.3)
        c1 = np.sum(np.abs(u1[:T, pai]))
        c2 = np.sum(np.abs(u2[:T, pai]))
        pct = (c2/c1 - 1) * 100 if c1 > 0.01 else 0
        print("%-12s %10.2f %10.2f %9.0f%%" % (SECTOR_NAMES[j], c1, c2, pct))

if __name__ == '__main__':
    main()
