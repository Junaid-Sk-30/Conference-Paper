# Causal Graph Regularization for Multi-Asset Return Forecasting: A Reproducibility-Focused Evaluation

[![Paper PDF](https://img.shields.io/badge/Paper-PDF-red.svg)](paper_manuscript.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)

**Authors:** Shaik Junaid$^1$, Sk. Valijan$^1$, and Dr. Nadiminti Nagamani$^2$  
$^1$Department of Data Science, VIT-AP University, Amaravati, Andhra Pradesh, India  
$^2$School of Advanced Science, VIT-AP University, Amaravati, Andhra Pradesh, India  
**Contact:** `shaik.junaid@vitap.ac.in`  

---

## Overview

This repository contains the official implementation, empirical data artifacts, rolling causal graphs, figures, and LaTeX manuscript for the paper:  
**"Causal Graph Regularization for Multi-Asset Return Forecasting: A Reproducibility-Focused Evaluation"**.

Daily equity-return forecasting is fundamentally constrained by low signal-to-noise ratios (SNR), time-varying cross-asset dependencies, and transaction frictions that can quickly eliminate gross statistical signals. In this paper, we evaluate graph-regularized Inverted Transformers across a 30-equity U.S. panel spanning 10 years (2016–2025, 2,513 daily observations, 108 dynamic rolling windows). 

We compare unconstrained dense attention, correlation masking, constraint-based causal graph masking (PCMCI+), continuous soft causal routing, and Radial Basis Function Kolmogorov-Arnold Network (RBF-KAN) prediction heads against classical econometric baselines: Constant Mean Predictor, Naive Persistence, and Vector Autoregression (VAR).

---

## Key Findings

1. **Point Forecasting Realities**: Across all evaluated neural architectures, out-of-sample $R^2$ remains strictly negative ($-2.35\%$ to $-4.54\%$), confirming that complex nonlinear models fail to beat the historical training mean in point-forecast accuracy.
2. **Gross vs. Net Performance**: While soft causal routing with a linear head produces the highest annualized gross Sharpe ratio ($+1.455 \pm 0.177$ vs $+0.725 \pm 0.504$ for dense attention), daily portfolio turnover ($1.32\text{--}1.44$) creates an institutional hurdle of $18\%\text{--}36\%$ annual cost drag under 5 to 10 bps execution fees, driving all net Sharpe ratios negative ($-0.321$ to $-2.993$).
3. **RBF-KAN Evaluation**: The RBF-KAN nonlinear prediction head underperforms the simpler linear head ($+0.191$ gross Sharpe, $-1.403$ net Sharpe at 5 bps), demonstrating that excessive output flexibility increases estimation variance in low-SNR environments.

---

## Repository Structure

```text
├── data_artifacts/                        # Reconciled empirical metrics and equity curves
│   ├── table1_reproducible_forecasting_metrics.csv
│   ├── table2_reproducible_backtest_metrics.csv
│   └── adjusted_prices_30stocks.csv
├── figures/                               # Publication figures (PNG, 300 DPI)
│   ├── figure1_network_density_timeline.png
│   ├── figure2_causal_graph_network.png
│   ├── figure3_ablation_performance_comparison.png
│   ├── figure4_cumulative_backtest_curves.png
│   ├── figure5_soft_causal_routing_and_sharpe.png
│   └── figure6_multi_regime_wfo_and_macro_topology.png
├── paper_manuscript.tex                   # Master LaTeX source code
├── paper_manuscript.pdf                   # Compiled 8-page proceedings paper
├── rolling_graphs_30.pkl                  # Cached rolling PCMCI+ causal graphs (30 stocks)
├── rolling_graphs_10_continuous.pkl       # Continuous p-value confidence matrices (10 stocks)
├── rolling_graphs_macro_fdr.pkl           # FDR-controlled macro-conditioned graphs
├── production_experiment_suite.py         # Main training and evaluation suite (3 seeds)
├── run_sota_framework_experiments.py      # Benchmark runner across architectures
├── test_turnover_penalty_and_tkan.py      # Turnover ablation and RBF-KAN evaluation
├── test_soft_causal_routing_mini_ablation.py # Soft routing temperature ablation
├── test_differentiable_sharpe_loss.py     # Hybrid loss optimization experiments
├── generate_figure5.py                    # Soft routing figure generator
└── generate_figure6_wfo_regimes.py        # Multi-regime walk-forward figure generator
```

---

## Reproducing Experiments

### 1. Environment Setup

```bash
git clone https://github.com/Junaid-Sk-30/Conference-Paper.git
cd Conference-Paper
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install torch numpy pandas scikit-learn tigramite matplotlib
```

### 2. Running the Production Benchmark

To run the unified 30-stock benchmark across the three fixed random seeds (42, 123, 999):

```bash
python production_experiment_suite.py
```

### 3. Multi-Regime Walk-Forward Evaluation

To evaluate cross-regime generalization across 2020 (COVID-19), 2022 (Fed Tightening), and 2024 (AI Expansion):

```bash
python generate_figure6_wfo_regimes.py
```

---

## Citation

If you find this work or codebase useful for your research, please cite:

```bibtex
@inproceedings{junaid2026causal,
  title={Causal Graph Regularization for Multi-Asset Return Forecasting: A Reproducibility-Focused Evaluation},
  author={Junaid, Shaik and Valijan, Sk. and Nagamani, Nadiminti},
  booktitle={Proceedings of the Conference on Machine Learning and Quantitative Finance},
  year={2026}
}
```
