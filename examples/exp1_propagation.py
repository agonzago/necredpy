"""
Network propagation of cost-push shocks (linear model).

This example loads the 5-sector model and hits each sector with a unit
cost-push shock one at a time. It then plots how each shock propagates
through the input-output network to aggregate inflation.

Key result: shock persistence matters as much as CPI weight. Energy has
a small CPI weight (0.10) but high persistence (rho=0.85), so its
cumulative impact on aggregate inflation exceeds Food (CPI weight 0.15,
rho=0.30).

Steps:
  1. Load the 5-sector model from the .mod file.
  2. For each sector j, compute the IRF to a unit cost-push shock
     using m.irf(). No credibility switching (linear model).
  3. Plot the aggregate inflation response and cumulative impact.

Output: figures/simple_exp1.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from necredpy import Model

SECTOR_NAMES = ['Energy', 'Food', 'Transport', 'Goods', 'Services']
SECTOR_COLORS = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']

m = Model("dynare/five_sector_network.mod")

T = 60
shock_size = 1.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Aggregate inflation response to each sector's shock
ax = axes[0]
for j in range(5):
    irf = m.irf('eps_pi%d' % (j+1), size=shock_size, T=T)
    ax.plot(irf['pi_agg'], color=SECTOR_COLORS[j], lw=2, label=SECTOR_NAMES[j])
    print("  %s: peak=%.4f  cum=%.4f" % (
        SECTOR_NAMES[j], irf['pi_agg'].abs().max(), irf['pi_agg'].abs().sum()))

ax.set_title('Aggregate Inflation Response')
ax.set_xlabel('Quarter')
ax.set_ylabel('pi_agg')
ax.axhline(0, color='gray', lw=0.5, ls=':')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: Cumulative |pi_agg| per unit shock (bar chart)
ax = axes[1]
cum = [m.irf('eps_pi%d' % (j+1), size=shock_size, T=T)['pi_agg'].abs().sum()
       for j in range(5)]
ax.bar(SECTOR_NAMES, cum, color=SECTOR_COLORS, alpha=0.7)
ax.set_title('Cumulative |pi_agg| per Unit Shock')
ax.set_ylabel('cum |pi_agg|')
ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('5-Sector Network: Propagation of Cost-Push Shocks', fontsize=14)
plt.tight_layout()
plt.savefig('figures/simple_exp1.png', dpi=150, bbox_inches='tight')
print("Saved: figures/simple_exp1.png")
