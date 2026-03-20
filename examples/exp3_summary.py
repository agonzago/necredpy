"""
Summary: sacrifice ratios and credibility loss duration by sector.

This example summarizes the credibility cost across all five sectors
in two bar charts:
  - Sacrifice ratio (cumulative |output gap| / peak |inflation|) under
    the linear model vs the PL model. A higher PL sacrifice ratio means
    credibility loss forces the central bank to accept more output loss
    per unit of disinflation.
  - M2 duration: how many quarters each sector's shock keeps the
    economy in the low-credibility regime.

Steps:
  1. Load the 5-sector model.
  2. For each sector, compute linear and PL IRFs using m.irf().
  3. Compute sacrifice ratios and M2 duration from the results.
  4. Plot side-by-side bar charts.

Output: figures/simple_exp3.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from necredpy import Model

SECTOR_NAMES = ['Energy', 'Food', 'Transport', 'Goods', 'Services']
SECTOR_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

m = Model("dynare/five_sector_network.mod")

T = 60
shock = 5.0

sr_lin = []
sr_pl = []
m2_dur = []

for j in range(5):
    sn = 'eps_pi%d' % (j+1)

    irf_l = m.irf(sn, size=shock, T=T, credibility=False)
    irf_p = m.irf(sn, size=shock, T=T, credibility=True)

    peak_l = irf_l['pi_agg'].abs().max()
    peak_p = irf_p['pi_agg'].abs().max()
    sr_lin.append(irf_l['y'].abs().sum() / peak_l if peak_l > 1e-12 else 0)
    sr_pl.append(irf_p['y'].abs().sum() / peak_p if peak_p > 1e-12 else 0)
    m2_dur.append((irf_p['regime'] == 1).sum() if 'regime' in irf_p else 0)

    print("  %s: SR_lin=%.2f SR_pl=%.2f M2=%d" % (
        SECTOR_NAMES[j], sr_lin[-1], sr_pl[-1], m2_dur[-1]))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel 1: Sacrifice ratios
ax = axes[0]
x = np.arange(5)
w = 0.35
ax.bar(x - w/2, sr_lin, w, color='gray', alpha=0.5, label='Linear')
ax.bar(x + w/2, sr_pl, w, color=SECTOR_COLORS, alpha=0.7, label='PL')
ax.set_xticks(x)
ax.set_xticklabels(SECTOR_NAMES, fontsize=9)
ax.set_title('Sacrifice Ratio (shock=%.1f)' % shock, fontweight='bold')
ax.set_ylabel('SR = cum|y| / peak|pi_agg|')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# Panel 2: M2 duration
ax = axes[1]
ax.bar(SECTOR_NAMES, m2_dur, color=SECTOR_COLORS, alpha=0.7)
ax.set_title('M2 Duration (shock=%.1f)' % shock, fontweight='bold')
ax.set_ylabel('Quarters in M2')
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Credibility Cost by Shock Origin', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('figures/simple_exp3.png', dpi=150, bbox_inches='tight')
print("Saved: figures/simple_exp3.png")
