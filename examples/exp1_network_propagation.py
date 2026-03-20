"""
Experiment 1: Network propagation of cost-push shocks (linear model).

PURPOSE
-------
Shows how a unit cost-push shock to each sector propagates through the
I-O network. No credibility switching in this experiment (single regime).

The key result is that shock persistence matters as much as CPI weight:
Energy (s=0.10, rho=0.85) has higher cumulative inflation impact than
Food (s=0.15, rho=0.30) despite a smaller CPI weight.

HOW IT WORKS
------------
1. load_model() reads dynare/five_sector_network.mod and returns the
   system matrices for both regimes (M1, M2) and model info.
2. solve_linear() solves the single-regime model (no credibility switching)
   for each shock and returns the IRF path.
3. The figure shows: aggregate inflation response, sectoral responses to
   Energy and Services shocks, output gap, cumulative inflation, and
   cross-sector spillovers.

TO MODIFY
---------
- To change calibration: edit dynare/five_sector_network.mod
- To change shock size: modify shock_size_unit below
- SECTOR_NAMES and SECTOR_COLORS are defined in src/network_helpers.py
  ( We can add this to the parser in the future)
  
Produces: figures/fig12_five_sector_propagation.png

Usage: .venv/bin/python scripts/exp1_network_propagation.py
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from necredpy.network_helpers import (load_model, solve_linear, fig_path,
                                       print_io_table, SECTOR_NAMES, SECTOR_COLORS)

def main():
    # Step 1: Load model from .mod file
    M1, M2, info, mod = load_model()

    # Extract variable indices from the model info
    vn = info['var_names']    # e.g., ['y', 'i_nom', 'pi_agg', 'pi1', ...]
    pai = vn.index('pi_agg')  # index of aggregate inflation
    yi = vn.index('y')        # index of output gap
    pi_idx = [vn.index('pi%d' % (j+1)) for j in range(5)]  # sectoral inflation indices

    T = 60              # simulation horizon (quarters)
    t = np.arange(T)    # time axis for plotting
    shock_size_unit = 1.0  # unit shock to each sector

    print("=" * 70)
    print("EXPERIMENT 1: Network Propagation (unit cost-push, linear)")
    print("=" * 70)
    print_io_table(info)
    print()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # --- Top-left: aggregate inflation response to each sector's shock ---
    ax = axes[0, 0]
    for j in range(5):
        # solve_linear() solves the single-regime model for shock eps_pi{j+1}
        u = solve_linear(mod, {}, 'eps_pi%d' % (j+1), shock_size_unit, T)
        # u is (T, n_vars); u[:, pai] is the aggregate inflation path
        ax.plot(t, u[:T, pai], '-', color=SECTOR_COLORS[j], lw=2,
                label=SECTOR_NAMES[j])
        print("  %s: peak=%.4f  cum=%.4f  rho_cp=%.2f" % (
            SECTOR_NAMES[j], np.max(np.abs(u[:, pai])),
            np.sum(np.abs(u[:T, pai])),
            info['params_M1']['rho_cp%d' % (j+1)]))
    ax.set_title('Aggregate Inflation Response', fontsize=11, fontweight='bold')
    ax.set_xlabel('Quarter'); ax.set_ylabel('pi_agg')
    ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Top-middle: all sectoral inflation responses to Energy shock ---
    ax = axes[0, 1]
    u = solve_linear(mod, {}, 'eps_pi1', shock_size_unit, T)
    for j in range(5):
        ax.plot(t, u[:T, pi_idx[j]], '-', color=SECTOR_COLORS[j], lw=2,
                label=SECTOR_NAMES[j])
    ax.set_title('Response to Energy Cost-Push', fontsize=11, fontweight='bold')
    ax.set_xlabel('Quarter'); ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Top-right: all sectoral inflation responses to Services shock ---
    ax = axes[0, 2]
    u = solve_linear(mod, {}, 'eps_pi5', shock_size_unit, T)
    for j in range(5):
        ax.plot(t, u[:T, pi_idx[j]], '-', color=SECTOR_COLORS[j], lw=2,
                label=SECTOR_NAMES[j])
    ax.set_title('Response to Services Cost-Push', fontsize=11, fontweight='bold')
    ax.set_xlabel('Quarter'); ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Bottom-left: output gap response by shock origin ---
    ax = axes[1, 0]
    for j in range(5):
        u = solve_linear(mod, {}, 'eps_pi%d' % (j+1), shock_size_unit, T)
        ax.plot(t, u[:T, yi], '-', color=SECTOR_COLORS[j], lw=2,
                label=SECTOR_NAMES[j])
    ax.set_title('Output Gap Response', fontsize=11, fontweight='bold')
    ax.set_xlabel('Quarter'); ax.axhline(0, color='gray', lw=0.5, ls=':')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # --- Bottom-middle: cumulative |pi_agg| per unit shock (bar chart) ---
    ax = axes[1, 1]
    cum = [np.sum(np.abs(
               solve_linear(mod, {}, 'eps_pi%d' % (j+1), shock_size_unit, T)[:T, pai]))
           for j in range(5)]
    ax.bar(SECTOR_NAMES, cum, color=SECTOR_COLORS, alpha=0.7)
    ax.set_title('Cumulative |pi_agg| per Unit Shock', fontsize=11,
                 fontweight='bold')
    ax.set_ylabel('cum |pi_agg|'); ax.grid(True, alpha=0.3, axis='y')

    # --- Bottom-right: cross-sector spillover (how much each shock
    #     affects OTHER sectors' inflation, not its own) ---
    ax = axes[1, 2]
    for j in range(5):
        u = solve_linear(mod, {}, 'eps_pi%d' % (j+1), shock_size_unit, T)
        spill = sum(np.sum(np.abs(u[:T, pi_idx[k]])) for k in range(5) if k != j)
        ax.bar(j, spill, color=SECTOR_COLORS[j], alpha=0.7)
    ax.set_xticks(range(5)); ax.set_xticklabels(SECTOR_NAMES, fontsize=9)
    ax.set_title('Spillover to Other Sectors', fontsize=11, fontweight='bold')
    ax.set_ylabel('cum |pi_other|'); ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('5-Sector Network: Propagation of Cost-Push Shocks',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = fig_path('fig12_five_sector_propagation.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print("\nSaved: %s" % out)

if __name__ == '__main__':
    main()
