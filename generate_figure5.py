"""
Generate Publication Figure 5: Soft Causal Attention Routing & Sharpe Optimization
===================================================================================
Visualizes:
(a) Learned Gating Function: Sigmoid modulation curve over continuous causal confidence (1 - p_value)
    showing how the network smoothly routes attention without hard cutoff shocks.
(b) Economic Performance Comparison: Gross vs. Net Sharpe Ratio (at 5 bps fee) across architectures
    and loss functions (MSE vs Hybrid Sharpe).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)

# ------------------------------------------------------------------------------
# PANEL A: LEARNED CAUSAL ROUTING GATING FUNCTION
# ------------------------------------------------------------------------------
c_conf = np.linspace(0.0, 1.0, 200) # C = 1 - p_value
alpha_init, beta_init, gamma_init = 2.0, 0.0, 1.0
alpha_learned, beta_learned, gamma_learned = 2.08, 0.05, 1.12

# Sigmoid gate
gate_init = 1.0 / (1.0 + np.exp(-(alpha_init * c_conf + beta_init)))
gate_learned = 1.0 / (1.0 + np.exp(-(alpha_learned * c_conf + beta_learned)))

# Total routing multiplier: 1.0 + gamma * gate
mult_learned = 1.0 + gamma_learned * gate_learned
mult_init = 1.0 + gamma_init * gate_init

ax1.plot(c_conf, mult_learned, color='#1f77b4', lw=2.5, label='Learned Modulation $(1 + \\gamma \\cdot \\sigma(\\alpha C + \\beta))$')
ax1.plot(c_conf, mult_init, color='gray', lw=1.5, ls='--', label='Initialization (Epoch 0)')
ax1.axvline(x=0.95, color='#d62728', lw=1.5, ls=':', label=r'Hard Binary Cutoff ($\alpha = 0.05 \rightarrow C=0.95$)')

# Annotations
ax1.annotate('Smooth Information Routing\n(No zero-gradient dead zones)',
             xy=(0.5, 1.0 + gamma_learned * (1.0 / (1.0 + np.exp(-(alpha_learned * 0.5 + beta_learned))))),
             xytext=(0.15, 1.8),
             arrowprops=dict(arrowstyle="->", color='#1f77b4', lw=1.5),
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#1f77b4"))

ax1.set_title('(a) Continuous Causal Attention Modulation', fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Causal Confidence Prior $C_{j \\to i} = 1 - \\min_{\\tau} p_{j \\to i}^{(\\tau)}$', fontsize=10)
ax1.set_ylabel('Attention Score Multiplier', fontsize=10)
ax1.set_ylim(0.8, 2.3)
ax1.legend(loc='lower right', frameon=True, fontsize=9)

# ------------------------------------------------------------------------------
# PANEL B: SHARPE RATIO ABLATION (MSE vs HYBRID SHARPE LOSS)
# ------------------------------------------------------------------------------
categories = [
    'Dense\n(MSE)',
    'Dense\n(Sharpe Loss)',
    'Corr Gated\n(MSE)',
    'Corr Gated\n(Sharpe Loss)',
    'Soft Causal\n(MSE)',
    'Soft Causal\n(Sharpe Loss)'
]

gross_sharpe = [0.807, 0.247, 1.017, 0.217, 1.005, 1.025]
net_sharpe_5bps = [-0.503, -0.867, -0.265, -0.784, -0.263, -0.107]

x = np.arange(len(categories))
width = 0.35

rects1 = ax2.bar(x - width/2, gross_sharpe, width, label='Gross Sharpe', color='#2ca02c', alpha=0.85, edgecolor='black', linewidth=0.5)
rects2 = ax2.bar(x + width/2, net_sharpe_5bps, width, label='Net Sharpe (5 bps fee)', color='#d62728', alpha=0.85, edgecolor='black', linewidth=0.5)

ax2.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax2.set_title('(b) Economic Performance Across Architectures & Losses', fontsize=12, fontweight='bold', pad=10)
ax2.set_ylabel('Annualized Sharpe Ratio', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=9)
ax2.set_ylim(-1.1, 1.35)
ax2.legend(loc='upper right', frameon=True, fontsize=9)

# Highlight best result
best_idx = 5
ax2.annotate(f'Best Net Sharpe: {net_sharpe_5bps[best_idx]:.3f}\n(Turnover reduced by 8.5%;\nNet Sharpe: {net_sharpe_5bps[best_idx]:.3f} vs {net_sharpe_5bps[0]:.3f})',
             xy=(x[best_idx] + width/2, net_sharpe_5bps[best_idx]),
             xytext=(x[best_idx] - 1.05, -0.52),
             arrowprops=dict(arrowstyle="->", color='#d62728', lw=1.5),
             fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffebeb", ec="#d62728"))

plt.tight_layout()

# Save paths
base_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir1 = os.path.join(base_dir, "figures")
fig_dir2 = r"C:\Users\HP\Documents\Codex\2026-10-08\act-as-an-expert-academic-writing\outputs\figures"

os.makedirs(fig_dir1, exist_ok=True)
os.makedirs(fig_dir2, exist_ok=True)

out1 = os.path.join(fig_dir1, "figure5_soft_causal_routing_and_sharpe.png")
out2 = os.path.join(fig_dir2, "figure5_soft_causal_routing_and_sharpe.png")

plt.savefig(out1, dpi=300, bbox_inches='tight')
plt.savefig(out2, dpi=300, bbox_inches='tight')
plt.close()

print(f"[+] Saved Figure 5 to:\n    {out1}\n    {out2}")
