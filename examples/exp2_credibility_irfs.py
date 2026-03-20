"""
Experiment 2: Credibility thresholds and IRFs by shock origin.

PURPOSE
-------
For each sector, finds the minimum cost-push shock size that triggers
credibility loss (M2 activation). Then shows PL vs Linear IRFs at a
common shock size to visualize the credibility cost.

HOW IT WORKS
------------
1. load_model() parses the .mod file (same as Experiment 1).
2. find_threshold() uses bisection to find the minimum shock size that
   causes any period to enter M2 (credibility < cred_threshold).
3. solve_linear() and solve_pl() compute IRFs for the linear model
   (no switching) and the piecewise-linear model (with switching).
4. The figure shows 5 rows (one per sector) x 3 columns (pi_agg,
   credibility path, cumulative output gap). Red shading = M2 periods.

TO MODIFY
---------
- To change calibration: edit dynare/five_sector_network.mod
- To change the test shock: modify the 1.5x multiplier below
- SECTOR_NAMES and SECTOR_COLORS are defined in src/network_helpers.py

Produces: figures/fig13_five_sector_credibility.png

Usage: .venv/bin/python scripts/exp2_credibility_irfs.py
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from necredpy.network_helpers import (load_model, solve_linear, solve_pl,
                                       find_threshold, sacrifice_ratio, fig_path,
                                       SECTOR_NAMES, SECTOR_COLORS)

def main():
    # Step 1: Load model
    M1, M2, info, mod = load_model()
    vn = info['var_names']
    pai = vn.index('pi_agg')   # aggregate inflation index
    yi = vn.index('y')         # output gap index
    epsilon_bar = info['regime_spec']['band']  # inflation tolerance band
    T = 60; t = np.arange(T)

    # Step 2: Find credibility thresholds for each sector
    print("=" * 70)
    print("EXPERIMENT 2: Credibility Thresholds + IRFs")
    print("=" * 70)

    thresholds = []
    for j in range(5):
        # find_threshold bisects over shock sizes until it finds the
        # smallest eps_pi{j+1} that causes credibility to drop below
        # cred_threshold (default 0.5) at any point during the simulation
        th = find_threshold(M1, M2, mod, info, 'eps_pi%d' % (j+1))
        thresholds.append(th)
        print("  %s: threshold = %.2f" % (SECTOR_NAMES[j], th))

    # Step 3: Set a common shock size (1.5x the smallest threshold)
    # This ensures at least one sector triggers M2
    shock = min(thresholds) * 1.5
    print("\nTest shock = %.1f (1.5x smallest threshold)" % shock)

    # Step 4: Compute and plot IRFs for each sector
    fig, axes = plt.subplots(5, 3, figsize=(16, 18))

    for j in range(5):
        sn = 'eps_pi%d' % (j+1)

        # Linear IRF (no credibility switching)
        u_lin = solve_linear(mod, {}, sn, shock, T)

        # PL IRF (with endogenous credibility switching)
        # Returns: u (path), reg (regime sequence), cred (credibility path)
        u_pl, reg, cred, _ = solve_pl(M1, M2, mod, sn, shock, info, T)

        m2 = np.sum(reg == 1)  # number of quarters in M2
        sr_l = sacrifice_ratio(u_lin[:, yi], u_lin[:, pai])
        sr_p = sacrifice_ratio(u_pl[:, yi], u_pl[:, pai])
        print("  %s: M2=%d SR_lin=%.2f SR_pl=%.2f" % (
            SECTOR_NAMES[j], m2, sr_l, sr_p))

        color = SECTOR_COLORS[j]

        # Column 0: Aggregate inflation (Linear vs PL)
        ax = axes[j, 0]
        ax.plot(t, u_lin[:T, pai], '-', color='gray', lw=1, label='Linear')
        ax.plot(t, u_pl[:T, pai], '-', color=color, lw=2, label='PL')
        ax.axhline(epsilon_bar, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(-epsilon_bar, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax.axhline(0, color='gray', lw=0.5, ls=':')
        # Shade M2 periods in red
        for ts in range(T):
            if reg[ts] == 1:
                ax.axvspan(ts-0.5, ts+0.5, alpha=0.06, color=color)
        if j == 0: ax.set_title('Aggregate Inflation', fontsize=10)
        ax.set_ylabel(SECTOR_NAMES[j], fontsize=9, fontweight='bold')
        if j == 0: ax.legend(fontsize=7)
        if j == 4: ax.set_xlabel('Quarter')
        ax.grid(True, alpha=0.3)

        # Column 1: Credibility capital stock
        ax = axes[j, 1]
        if cred is not None:
            ax.plot(t, cred[:T], '-', color=color, lw=2)
        ax.axhline(info['regime_spec']['cred_threshold'], color='gray',
                   lw=0.8, ls='--', alpha=0.5)
        ax.set_ylim(-0.05, 1.1)
        for ts in range(T):
            if reg[ts] == 1:
                ax.axvspan(ts-0.5, ts+0.5, alpha=0.06, color=color)
        if j == 0: ax.set_title('Credibility', fontsize=10)
        if j == 4: ax.set_xlabel('Quarter')
        ax.grid(True, alpha=0.3)

        # Column 2: Cumulative |output gap| (Linear vs PL)
        ax = axes[j, 2]
        ax.plot(t, np.cumsum(np.abs(u_lin[:T, yi])), '-', color='gray', lw=1)
        ax.plot(t, np.cumsum(np.abs(u_pl[:T, yi])), '-', color=color, lw=2)
        for ts in range(T):
            if reg[ts] == 1:
                ax.axvspan(ts-0.5, ts+0.5, alpha=0.06, color=color)
        if j == 0: ax.set_title('Cumulative |Output Gap|', fontsize=10)
        if j == 4: ax.set_xlabel('Quarter')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        '5-Sector Network + Credibility: Cost-Push = %.1f by Origin' % shock,
        fontsize=14, y=1.01)
    plt.tight_layout()
    out = fig_path('fig13_five_sector_credibility.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print("\nSaved: %s" % out)

if __name__ == '__main__':
    main()
