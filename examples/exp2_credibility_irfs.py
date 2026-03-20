"""
Linear vs credibility-switching IRFs by shock origin.

This example compares what happens when you account for credibility
versus when you don't. For each sector, it computes two IRFs:
  - Linear: the standard rational expectations solution (no switching).
  - PL (piecewise-linear): the model with endogenous credibility. When
    the shock pushes inflation outside the tolerance band, credibility
    falls and the Phillips curve becomes more backward-looking, making
    inflation harder to bring down.

The difference between the two shows the "credibility cost" of a shock.
Sectors whose shocks trigger more M2 periods have higher sacrifice
ratios under PL than under the linear model.

Steps:
  1. Load the 5-sector model.
  2. For each sector, compute linear and PL IRFs using m.irf().
  3. Plot inflation paths side by side with the credibility path.
     Red-shaded periods = low credibility (M2 regime).

Output: figures/simple_exp2.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from necredpy import Model

SECTOR_NAMES = ['Energy', 'Food', 'Transport', 'Goods', 'Services']
SECTOR_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

m = Model("dynare/five_sector_network.mod")

T = 60
shock = 5.0  # large enough to trigger credibility loss in some sectors

fig, axes = plt.subplots(5, 2, figsize=(14, 18))

for j in range(5):
    sn = 'eps_pi%d' % (j+1)
    color = SECTOR_COLORS[j]

    # Linear IRF (no credibility switching)
    irf_lin = m.irf(sn, size=shock, T=T, credibility=False)

    # PL IRF (with credibility switching)
    irf_pl = m.irf(sn, size=shock, T=T, credibility=True)

    m2_periods = (irf_pl['regime'] == 1).sum() if 'regime' in irf_pl else 0

    # Sacrifice ratios
    sr_lin = irf_lin['y'].abs().sum() / irf_lin['pi_agg'].abs().max()
    sr_pl = irf_pl['y'].abs().sum() / irf_pl['pi_agg'].abs().max()

    print("  %s: M2=%d quarters, SR_lin=%.2f, SR_pl=%.2f" % (
        SECTOR_NAMES[j], m2_periods, sr_lin, sr_pl))

    # Column 0: Aggregate inflation (Linear vs PL)
    ax = axes[j, 0]
    ax.plot(irf_lin['pi_agg'], color='gray', lw=1, label='Linear')
    ax.plot(irf_pl['pi_agg'], color=color, lw=2, label='PL')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    if 'regime' in irf_pl:
        for t in range(T):
            if irf_pl['regime'].iloc[t] == 1:
                ax.axvspan(t-0.5, t+0.5, alpha=0.06, color=color)
    if j == 0:
        ax.set_title('Aggregate Inflation', fontsize=10)
        ax.legend(fontsize=7)
    ax.set_ylabel(SECTOR_NAMES[j], fontsize=9, fontweight='bold')
    if j == 4:
        ax.set_xlabel('Quarter')
    ax.grid(True, alpha=0.3)

    # Column 1: Credibility path
    ax = axes[j, 1]
    if 'credibility' in irf_pl:
        ax.plot(irf_pl['credibility'], color=color, lw=2)
        ax.axhline(0.5, color='gray', lw=0.8, ls='--', alpha=0.5)
        for t in range(T):
            if irf_pl['regime'].iloc[t] == 1:
                ax.axvspan(t-0.5, t+0.5, alpha=0.06, color=color)
    ax.set_ylim(-0.05, 1.1)
    if j == 0:
        ax.set_title('Credibility', fontsize=10)
    if j == 4:
        ax.set_xlabel('Quarter')
    ax.grid(True, alpha=0.3)

fig.suptitle('5-Sector Network + Credibility (shock=%.1f)' % shock,
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('figures/simple_exp2.png', dpi=150, bbox_inches='tight')
print("Saved: figures/simple_exp2.png")
