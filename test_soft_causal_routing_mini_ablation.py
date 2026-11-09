"""
Soft Causal Attention Routing Mini-Ablation
============================================
Implements:
1. Continuous P-Value Extraction from Rolling PCMCI+:
   C[i, j] = 1.0 - min_tau(p_value[j -> i]) in [0, 1]
2. Parameterized Learned Gating Layer:
   Scores_routed = Scores * (1.0 + gamma * sigmoid(alpha * C + beta))
   Replaces discontinuous masked_fill with smooth, end-to-end differentiable routing.
3. Mini-Ablation on 10 liquid stocks (5 epochs, seed 42):
   - Logs epoch-by-epoch train loss, validation loss, and gradient norms.
   - Verifies smooth convergence without exploding or vanishing gradients.
   - Evaluates out-of-sample R²_OOS and Sharpe before/after fees.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

import tigramite
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

warnings.filterwarnings("ignore")

# 10 Multi-Sector Liquid Large-Caps for Mini-Ablation
SUBSET_10 = [
    "AAPL", "MSFT", "GOOGL", "NVDA",  # Tech
    "JPM", "BAC",                     # Fin
    "XOM",                            # Energy
    "JNJ",                            # Healthcare
    "PG",                             # Consumer
    "CAT"                             # Industrial
]

CACHE_FILE_10 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rolling_graphs_10_continuous.pkl")

# ==============================================================================
# 1. CONTINUOUS GRAPH EXTRACTION
# ==============================================================================
def get_continuous_rolling_graphs(returns, tickers, window_size=252, step_size=21, max_lag=2, alpha=0.05):
    if os.path.exists(CACHE_FILE_10):
        print(f"[+] Loading cached continuous graphs from {CACHE_FILE_10}...")
        with open(CACHE_FILE_10, "rb") as f:
            import pickle
            return pickle.load(f)
            
    n_nodes = len(tickers)
    parcorr = ParCorr(significance='analytic')
    graphs = {}
    n_obs = len(returns)
    windows = [(returns.index[s + window_size - 1], returns.iloc[s:s + window_size][tickers].values)
               for s in range(0, n_obs - window_size + 1, step_size)]
               
    print(f"[*] Extracting continuous p-values across {len(windows)} rolling windows for {n_nodes} stocks...")
    t0 = time.time()
    for idx, (end_date, win_data) in enumerate(windows):
        # 1. Tigramite PCMCI+
        df_tig = pp.DataFrame(win_data)
        pcmci = PCMCI(dataframe=df_tig, cond_ind_test=parcorr, verbosity=0)
        pcmci_res = pcmci.run_pcmciplus(tau_max=max_lag, pc_alpha=alpha)
        
        # p_matrix has shape (N_source, N_target, tau_max + 1)
        # Take min p-value across positive lags 1..max_lag
        p_mat = pcmci_res['p_matrix'][:, :, 1:]
        min_p = np.min(p_mat, axis=2) # (N_source, N_target)
        
        # Continuous Causal Confidence: 1.0 - p_value
        # Transpose so row i is target (query) and col j is source (key)
        cont_pcmci = (1.0 - min_p).T.astype(np.float32)
        np.fill_diagonal(cont_pcmci, 1.0) # Self-loop maximum trust
        
        # 2. Continuous Correlation: absolute Pearson correlation
        corr_mat = np.abs(np.corrcoef(win_data, rowvar=False)).astype(np.float32)
        np.fill_diagonal(corr_mat, 1.0)
        
        graphs[end_date] = {
            'pcmci_cont': cont_pcmci,
            'corr_cont': corr_mat
        }
        if (idx + 1) % 25 == 0 or idx + 1 == len(windows):
            print(f"    Window {idx+1}/{len(windows)} ({time.time()-t0:.1f}s)")
            
    with open(CACHE_FILE_10, "wb") as f:
        import pickle
        pickle.dump(graphs, f)
    print(f"[+] Saved continuous graph cache to {CACHE_FILE_10}")
    return graphs

# ==============================================================================
# 2. SOFT CAUSAL ROUTING iTRANSFORMER
# ==============================================================================
class SoftCausalInvertedTransformer(nn.Module):
    """
    Inverted Transformer with Learnable Soft Causal Routing.
    Instead of hard binary masks (-inf), attention scores are modulated
    by a parameterized transformation of continuous causal confidence:
    
    Scores_routed = Scores * (1.0 + gamma * sigmoid(alpha * C + beta))
    """
    def __init__(self, num_nodes, lookback=20, num_features=3, d_model=64, n_heads=4, d_ff=128, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.token_embed = nn.Linear(lookback * num_features, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Learnable Soft Causal Routing Parameters
        # alpha: scaling factor per head
        self.alpha = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 2.0)
        # beta: bias per head
        self.beta = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
        # gamma: learnable global trust scale
        self.gamma = nn.Parameter(torch.tensor(1.0))
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        self.pred_head = nn.Linear(d_model, 1)

    def forward(self, x, graph_weights=None):
        B, L, N, F = x.shape
        x_inv = x.permute(0, 2, 1, 3).reshape(B, N, L * F)
        tokens = self.token_embed(x_inv)
        
        norm_tokens = self.norm1(tokens)
        Q = self.q_proj(norm_tokens).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(norm_tokens).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(norm_tokens).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Raw Scaled Dot-Product Scores: (B, H, N, N)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Soft Causal Attention Routing
        if graph_weights is not None:
            if graph_weights.dim() == 2:
                graph_weights = graph_weights.unsqueeze(0).unsqueeze(1)
            elif graph_weights.dim() == 3:
                graph_weights = graph_weights.unsqueeze(1) # (B, 1, N, N)
                
            # Continuous learned gate in (0, 1)
            gate = torch.sigmoid(self.alpha * graph_weights + self.beta)
            # Differentiable multiplicative modulation
            scores = scores * (1.0 + torch.relu(self.gamma) * gate)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, N, self.d_model)
        tokens = tokens + self.dropout(self.out_proj(attn_out))
        tokens = tokens + self.ffn(self.norm2(tokens))
        return self.pred_head(tokens).squeeze(-1)

# ==============================================================================
# 3. MINI-ABLATION EXECUTION & GRADIENT AUDIT
# ==============================================================================
def run_mini_ablation():
    print("=" * 80)
    print("RUNNING SOFT CAUSAL ATTENTION ROUTING MINI-ABLATION (10 STOCKS)")
    print("=" * 80)
    
    # Load or download data
    local_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_artifacts", "adjusted_prices_30stocks.csv")
    if os.path.exists(local_csv):
        print(f"[+] Loading price data from local archive: {local_csv}")
        raw = pd.read_csv(local_csv, index_col="Date", parse_dates=True)[SUBSET_10].ffill().dropna()
    else:
        print("[*] Downloading 10-stock panel...")
        raw = yf.download(SUBSET_10, start="2016-01-01", end="2025-12-31", auto_adjust=True, progress=False)["Close"].ffill().dropna()
    tickers = [t for t in SUBSET_10 if t in raw.columns]
    returns = np.log(raw[tickers] / raw[tickers].shift(1)).dropna()
    print(f"[+] Loaded {len(tickers)} stocks over {len(returns)} trading days.")
    
    feature_dict = {}
    for t in tickers:
        r = returns[t]
        vol = r.rolling(20).std()
        mom = r.rolling(20).mean()
        feature_dict[t] = pd.DataFrame({"return": r, "volatility": vol, "momentum": mom}, index=returns.index)
    combined = pd.concat(feature_dict, axis=1).dropna()
    common_idx = combined.index[:-1]
    target = returns.loc[combined.index].shift(-1).loc[common_idx]
    features = combined.loc[common_idx]
    
    # Graphs
    graphs = get_continuous_rolling_graphs(returns, tickers)
    
    # Sequences with Purging and Embargo
    lookback = 20
    dates = target.index
    sorted_gdates = sorted(graphs.keys())
    
    X_list, Y_list, pcmci_list, corr_list = [], [], [], []
    for t_idx in range(lookback, len(dates)):
        curr_d = dates[t_idx]
        frame_feats = [[features[tkr].iloc[t_idx - lag].values for tkr in tickers] for lag in range(lookback, 0, -1)]
        X_list.append(np.array(frame_feats, dtype=np.float32))
        Y_list.append(target.iloc[t_idx].values.astype(np.float32))
        cand = [gd for gd in sorted_gdates if gd <= curr_d]
        best_gd = cand[-1] if cand else sorted_gdates[0]
        pcmci_list.append(graphs[best_gd]['pcmci_cont'])
        corr_list.append(graphs[best_gd]['corr_cont'])
        
    X_arr = np.array(X_list)
    Y_arr = np.array(Y_list)
    pcmci_arr = np.array(pcmci_list)
    corr_arr = np.array(corr_list)
    
    # Splits (70% train, 15% val, 15% test with purging & embargo)
    total_s = len(X_arr)
    raw_tr_end = int(total_s * 0.70)
    raw_va_end = int(total_s * 0.85)
    
    tr_sl = slice(0, raw_tr_end)
    va_sl = slice(raw_tr_end + lookback, raw_va_end)
    te_sl = slice(raw_va_end + lookback + 5, total_s)
    
    X_tr, Y_tr, M_pcmci_tr, M_corr_tr = X_arr[tr_sl], Y_arr[tr_sl], pcmci_arr[tr_sl], corr_arr[tr_sl]
    X_va, Y_va, M_pcmci_va, M_corr_va = X_arr[va_sl], Y_arr[va_sl], pcmci_arr[va_sl], corr_arr[va_sl]
    X_te, Y_te, M_pcmci_te, M_corr_te = X_arr[te_sl], Y_arr[te_sl], pcmci_arr[te_sl], corr_arr[te_sl]
    
    # Scalers
    scaler_x = StandardScaler()
    B_tr, L, N, F = X_tr.shape
    X_tr_s = scaler_x.fit_transform(X_tr.reshape(-1, F)).reshape(B_tr, L, N, F)
    X_va_s = scaler_x.transform(X_va.reshape(-1, F)).reshape(X_va.shape[0], L, N, F)
    X_te_s = scaler_x.transform(X_te.reshape(-1, F)).reshape(X_te.shape[0], L, N, F)
    
    scaler_y = StandardScaler()
    Y_tr_s = scaler_y.fit_transform(Y_tr)
    Y_va_s = scaler_y.transform(Y_va)
    y_tr_mean = np.mean(Y_tr, axis=0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Running 5-epoch mini-ablation on device: {device}")
    
    models_to_test = [
        ("Dense (Unconstrained)", None, None, None),
        ("Correlation Gated", M_corr_tr, M_corr_va, M_corr_te),
        ("Soft Causal Routed (PCMCI+)", M_pcmci_tr, M_pcmci_va, M_pcmci_te)
    ]
    
    results_summary = []
    
    for name, m_train, m_val, m_test in models_to_test:
        print(f"\n---> Training: {name}...")
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = SoftCausalInvertedTransformer(len(tickers), lookback=20, num_features=F).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        crit = nn.MSELoss()
        
        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
        Y_tr_t = torch.tensor(Y_tr_s, dtype=torch.float32)
        X_va_t = torch.tensor(X_va_s, dtype=torch.float32).to(device)
        Y_va_t = torch.tensor(Y_va_s, dtype=torch.float32).to(device)
        
        if m_train is not None:
            ds = TensorDataset(X_tr_t, torch.tensor(m_train, dtype=torch.float32), Y_tr_t)
            m_val_t = torch.tensor(m_val, dtype=torch.float32).to(device)
            m_test_t = torch.tensor(m_test, dtype=torch.float32).to(device)
        else:
            ds = TensorDataset(X_tr_t, torch.zeros(len(X_tr_t)), Y_tr_t)
            m_val_t = None
            m_test_t = None
            
        loader = DataLoader(ds, batch_size=32, shuffle=True)
        
        print(f"    {'Epoch':5s} | {'Train MSE':10s} | {'Val MSE':10s} | {'Grad Norm':10s} | {'alpha':8s} | {'gamma':8s}")
        print("    " + "-" * 65)
        
        for ep in range(1, 6):
            model.train()
            running_loss = 0.0
            total_grad_norm = 0.0
            n_batches = 0
            
            for batch in loader:
                bx = batch[0].to(device)
                bm = batch[1].to(device) if m_train is not None else None
                by = batch[2].to(device)
                
                optimizer.zero_grad()
                out = model(bx, graph_weights=bm)
                loss = crit(out, by)
                loss.backward()
                
                # Check gradient norm
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                total_grad_norm += grad_norm.item()
                n_batches += 1
                
                optimizer.step()
                running_loss += loss.item() * bx.size(0)
                
            tr_loss = running_loss / len(X_tr)
            avg_grad = total_grad_norm / n_batches
            
            model.eval()
            with torch.no_grad():
                val_out = model(X_va_t, graph_weights=m_val_t)
                val_loss = crit(val_out, Y_va_t).item()
                
            alpha_val = f"{model.alpha.mean().item():.3f}"
            gamma_val = f"{model.gamma.item():.3f}"
            print(f"    {ep:02d}    | {tr_loss:.6f}   | {val_loss:.6f} | {avg_grad:10.4f} | {alpha_val:8s} | {gamma_val:8s}")
            
        # Out-of-Sample Evaluation
        model.eval()
        with torch.no_grad():
            preds_s = model(torch.tensor(X_te_s).to(device), graph_weights=m_test_t).cpu().numpy()
        preds = scaler_y.inverse_transform(preds_s)
        
        # Metrics
        mae = np.mean(np.abs(Y_te - preds))
        mse_m = np.sum((Y_te - preds)**2)
        mse_b = np.sum((Y_te - y_tr_mean)**2)
        oos_r2 = (1.0 - (mse_m / mse_b)) * 100.0
        da = np.mean(np.sign(Y_te) == np.sign(preds)) * 100.0
        
        # 1-day Backtest
        k = max(1, len(tickers) // 5) # 2 long, 2 short
        weights = np.zeros_like(preds)
        for t in range(len(preds)):
            rk = np.argsort(preds[t])
            weights[t, rk[-k:]] = 1.0 / (2.0 * k)
            weights[t, rk[:k]] = -1.0 / (2.0 * k)
        d_weights = np.sum(np.abs(weights[1:] - weights[:-1]), axis=1)
        turnover = np.mean(d_weights)
        gross_ret = np.sum(weights[1:] * Y_te[1:], axis=1)
        net_ret_5bps = gross_ret - (5.0 / 10000.0) * d_weights
        
        sr_gross = np.sqrt(252) * (np.mean(gross_ret) / np.std(gross_ret, ddof=1))
        sr_net5 = np.sqrt(252) * (np.mean(net_ret_5bps) / np.std(net_ret_5bps, ddof=1))
        
        results_summary.append({
            "Architecture": name,
            "Val Loss (Ep 5)": f"{val_loss:.6f}",
            "R2_OOS (%)": f"{oos_r2:.2f}%",
            "MAE": f"{mae:.4f}",
            "Dir Acc (%)": f"{da:.2f}%",
            "Turnover": f"{turnover:.2f}",
            "Gross Sharpe": f"{sr_gross:.3f}",
            "Net Sharpe (5 bps)": f"{sr_net5:.3f}"
        })
        
    df_res = pd.DataFrame(results_summary).set_index("Architecture")
    print("\n" + "=" * 90)
    print("MINI-ABLATION RESULTS (5 EPOCHS, SEED 42):")
    print("=" * 90)
    print(df_res.to_string())
    df_res.to_csv("mini_ablation_soft_routing_results.csv")
    print("\n[+] Saved mini_ablation_soft_routing_results.csv")

if __name__ == "__main__":
    run_mini_ablation()
