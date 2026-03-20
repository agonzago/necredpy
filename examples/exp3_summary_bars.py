"""
Experiment 3: Summary bar charts -- thresholds, sacrifice ratios, M2 duration.

PURPOSE
-------
Summarizes the key credibility results in three bar charts:
  Panel 1: Credibility threshold by sector (min shock to trigger M2)
  Panel 2: Sacrifice ratio under Linear vs PL at a common shock size
  Panel 3: Duration of credibility loss (quarters in M2)

HOW IT WORKS
------------
Same pipeline as Experiments 1-2: load_model() reads the .mod file,
find_threshold() computes thresholds, solve_linear() and solve_pl()
compute IRFs and sacrifice ratios.

TO MODIFY
---------
- To change calibration: edit dynare/five_sector_network.mod
- SECTOR_NAMES and SECTOR_COLORS are defined in src/network_helpers.py

Produces: figures/fig14_five_sector_summary.png

Usage: .venv/bin/python scripts/exp3_summary_bars.py
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from necredpy.network_helpers import (load_model, solve_linear, solve_pl,
                                       find_threshold, sacrifice_ratio, fig_path,
                                       SECTOR_NAMES, SECTOR_COLORS)

def main():
    M1, M2, info, mod = load_model()
    vn = info['var_names']
    pai = vn.index('pi_agg')
    yi = vn.index('y')
    T = 60

    # Find credibility thresholds for each sector
    thresholds = [find_threshold(M1, M2, mod, info, 'eps_pi%d' % (j+1))
                  for j in range(5)]
    shock = min(thresholds) * 1.5  # common shock size

    print("=" * 70)
    print("EXPERIMENT 3: Summary Bar Charts (shock=%.1f)" % shock)
    print("=" * 70)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Panel 1: Credibility thresholds ---
    ax = axes[0]
    ax.bar(SECTOR_NAMES, thresholds, color=SECTOR_COLORS, alpha=0.7)
    ax.set_title('Credibility Threshold\n(min shock to trigger M2)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Shock size')
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 2: Sacrifice ratios (Linear vs PL) ---
    ax = axes[1]
    sr_lin = []; sr_pl = []
    for j in range(5):
        sn = 'eps_pi%d' % (j+1)
        u_l = solve_linear(mod, {}, sn, shock, T)
        u_p, _, _, _ = solve_pl(M1, M2, mod, sn, shock, info, T)
        sr_lin.append(sacrifice_ratio(u_l[:, yi], u_l[:, pai]))
        sr_pl.append(sacrifice_ratio(u_p[:, yi], u_p[:, pai]))
    x = np.arange(5); w = 0.35
    ax.bar(x - w/2, sr_lin, w, color='gray', alpha=0.5, label='Linear')
    ax.bar(x + w/2, sr_pl, w, color=SECTOR_COLORS, alpha=0.7, label='PL')
    ax.set_xticks(x); ax.set_xticklabels(SECTOR_NAMES, fontsize=9)
    ax.set_title('Sacrifice Ratio (shock=%.1f)' % shock,
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('SR = cum|y| / peak|pi_agg|')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 3: M2 duration ---
    ax = axes[2]
    m2_dur = []
    for j in range(5):
        _, reg, _, _ = solve_pl(M1, M2, mod, 'eps_pi%d' % (j+1),
                                shock, info, T)
        m2_dur.append(np.sum(reg == 1))
    ax.bar(SECTOR_NAMES, m2_dur, color=SECTOR_COLORS, alpha=0.7)
    ax.set_title('M2 Duration (shock=%.1f)' % shock,
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Quarters in M2')
    ax.grid(True, alpha=0.3, axis='y')

    # Print summary
    for j in range(5):
        print("  %s: threshold=%.2f SR_lin=%.2f SR_pl=%.2f M2=%d" % (
            SECTOR_NAMES[j], thresholds[j], sr_lin[j], sr_pl[j], m2_dur[j]))

    fig.suptitle('5-Sector Network: Shock-Origin-Dependent Credibility Cost',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    out = fig_path('fig14_five_sector_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print("\nSaved: %s" % out)

if __name__ == '__main__':
    main()
