"""
Credibility fragility: how initial credibility affects outcomes.

This example shows the "double amplification" effect. The same shock
is applied twice: once starting from full credibility (cred=1.0) and
once starting from impaired credibility (cred=0.30).

When credibility is already low, the Phillips curve is more backward-
looking from the start, so inflation is harder to control. The same
shock can generate up to 45% more cumulative inflation. This is the
"fragility zone" -- an economy with damaged credibility is much more
vulnerable to supply shocks.

Steps:
  1. Load the 5-sector model.
  2. For each sector, compute PL IRFs at cred_init=1.0 and cred_init=0.30
     using m.irf(..., credibility=True, cred_init=...).
  3. Plot inflation and credibility paths side by side.
     Solid lines = full credibility start.
     Dashed lines = impaired credibility start.

Output: figures/simple_exp4.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from necredpy import Model

SECTOR_NAMES = ['Energy', 'Food', 'Transport', 'Goods', 'Services']
SECTOR_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

m = Model("dynare/five_sector_network.mod")

T = 60
shock = 5.0
cred_levels = [1.0, 0.30]
cred_labels = ['Full (cred=1.0)', 'Impaired (cred=0.30)']
cred_styles = ['-', '--']

fig, axes = plt.subplots(5, 2, figsize=(14, 18))

for j in range(5):
    sn = 'eps_pi%d' % (j+1)
    color = SECTOR_COLORS[j]
    print("  %s:" % SECTOR_NAMES[j])

    for ci, (c0, clabel, cstyle) in enumerate(
            zip(cred_levels, cred_labels, cred_styles)):

        irf = m.irf(sn, size=shock, T=T, credibility=True, cred_init=c0)
        cum_pi = irf['pi_agg'].abs().sum()
        cum_y = irf['y'].abs().sum()
        print("    %s: cum_pi=%.2f cum_y=%.2f" % (clabel, cum_pi, cum_y))

        alpha = 1.0 if ci == 0 else 0.7

        # Column 0: Aggregate inflation
        ax = axes[j, 0]
        ax.plot(irf['pi_agg'], cstyle, color=color, lw=2, alpha=alpha,
                label=clabel if j == 0 else '')

        # Column 1: Credibility path
        ax = axes[j, 1]
        if 'credibility' in irf:
            ax.plot(irf['credibility'], cstyle, color=color, lw=2, alpha=alpha,
                    label=clabel if j == 0 else '')

    # Format
    axes[j, 0].axhline(0, color='gray', lw=0.5, ls=':')
    axes[j, 0].set_ylabel(SECTOR_NAMES[j], fontsize=9, fontweight='bold')
    axes[j, 0].grid(True, alpha=0.3)
    if j == 0:
        axes[j, 0].set_title('Aggregate Inflation', fontsize=10)
        axes[j, 0].legend(fontsize=7)
    if j == 4:
        axes[j, 0].set_xlabel('Quarter')

    axes[j, 1].axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.4)
    axes[j, 1].set_ylim(-0.05, 1.1)
    axes[j, 1].grid(True, alpha=0.3)
    if j == 0:
        axes[j, 1].set_title('Credibility', fontsize=10)
        axes[j, 1].legend(fontsize=7)
    if j == 4:
        axes[j, 1].set_xlabel('Quarter')

fig.suptitle('Double Amplification: Network x Credibility (shock=%.1f)\n'
             'Solid = Full, Dashed = Impaired' % shock, fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('figures/simple_exp4.png', dpi=150, bbox_inches='tight')
print("Saved: figures/simple_exp4.png")
