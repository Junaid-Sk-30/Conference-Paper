"""
Differentiable Sharpe Ratio Loss Optimization & Mini-Ablation
=============================================================
Implements:
1. Differentiable Portfolio Layer:
   w_t = (y_hat_t - mean(y_hat_t)) / (sum(|y_hat_t - mean(y_hat_t)|) + eps)
   Guarantees dollar-neutrality (sum(w_t) = 0) and bounded leverage (sum(|w_t|) = 1).
2. Differentiable Sharpe Loss:
   R_p,t = sum_i(w_{t,i} * Y_{t,i})
   Loss_Sharpe = - mean(R_p) / (std(R_p) + eps)
3. Combined Hybrid Loss:
   Loss_Total = Loss_MSE + lambda_sr * Loss_Sharpe
4. Systematic comparison across:
   - Dense iTransformer (MSE vs Hybrid Sharpe)
   - Correlation Gated iTransformer (MSE vs Hybrid Sharpe)
   - Soft Causal Routed iTransformer (MSE vs Hybrid Sharpe)
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE_10 = os.path.join(BASE_DIR, "rolling_graphs_10_continuous.pkl")
LOCAL_PRICES_CSV = os.path.join(BASE_DIR, "data_artifacts", "adjusted_prices_30stocks.csv")

SUBSET_10 = [
    "AAPL", "MSFT", "GOOGL", "NVDA",  # Tech
    "JPM", "BAC",                     # Fin
    "XOM",                            # Energy
    "JNJ",                            # Healthcare
    "PG",                             # Consumer
    "CAT"                             # Industrial
]

# ==============================================================================
# 1. DIFFERENTIABLE SHARPE LOSS MODULE
# ==============================================================================
class DifferentiableSharpeLoss(nn.Module):
    def __init__(self, eps=1e-6, scale_annual=np.sqrt(252)):
        super().__init__()
        self.eps = eps
        self.scale_annual = scale_annual

    def forward(self, pred_returns, true_returns):
        """
        pred_returns: (B, N) predicted returns
        true_returns: (B, N) actual realized returns
        """
        # Cross-sectional dollar-neutral portfolio weights
        pred_centered = pred_returns - torch.mean(pred_returns, dim=-1, keepdim=True)
        l1_norm = torch.sum(torch.abs(pred_centered), dim=-1, keepdim=True) + self.eps
        weights = pred_centered / l1_norm # sum(weights) = 0, sum(|weights|) = 1
        
        # Realized portfolio return per day in batch
        port_ret = torch.sum(weights * true_returns, dim=-1) # (B,)
        
        mean_ret = torch.mean(port_ret)
        std_ret = torch.std(port_ret, unbiased=True) + self.eps
        
        sharpe = (mean_ret / std_ret) * self.scale_annual
        return -sharpe # Minimize negative Sharpe

class HybridSharpeLoss(nn.Module):
    def __init__(self, lambda_sr=0.1, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.sharpe_loss = DifferentiableSharpeLoss(eps=eps)
        self.lambda_sr = lambda_sr

    def forward(self, pred_returns, true_returns):
        l_mse = self.mse(pred_returns, true_returns)
        l_sr = self.sharpe_loss(pred_returns, true_returns)
        return l_mse + self.lambda_sr * l_sr, l_mse, l_sr

# ==============================================================================
# 2. MODEL DEFINITION
# ==============================================================================
class SoftCausalInvertedTransformer(nn.Module):
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
        self.alpha = nn.Parameter(torch.ones(1, n_heads, 1, 1) * 2.0)
        self.beta = nn.Parameter(torch.zeros(1, n_heads, 1, 1))
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
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if graph_weights is not None:
            if graph_weights.dim() == 2:
                graph_weights = graph_weights.unsqueeze(0).unsqueeze(1)
            elif graph_weights.dim() == 3:
                graph_weights = graph_weights.unsqueeze(1)
            gate = torch.sigmoid(self.alpha * graph_weights + self.beta)
            scores = scores * (1.0 + torch.relu(self.gamma) * gate)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V).transpose(1, 2).reshape(B, N, self.d_model)
        tokens = tokens + self.dropout(self.out_proj(attn_out))
        tokens = tokens + self.ffn(self.norm2(tokens))
        return self.pred_head(tokens).squeeze(-1)

# ==============================================================================
# 3. EXPERIMENT EXECUTION
# ==============================================================================
def run_sharpe_ablation():
    print("=" * 80)
    print("DIFFERENTIABLE SHARPE RATIO LOSS MINI-ABLATION")
    print("=" * 80)
    
    # 1. Load data
    print(f"[+] Loading prices from {LOCAL_PRICES_CSV}...")
    raw = pd.read_csv(LOCAL_PRICES_CSV, index_col="Date", parse_dates=True)[SUBSET_10].ffill().dropna()
    tickers = [t for t in SUBSET_10 if t in raw.columns]
    returns = np.log(raw[tickers] / raw[tickers].shift(1)).dropna()
    print(f"[+] Panel: {len(tickers)} assets, {len(returns)} trading days.")
    
    # Features
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
    
    # Load cached continuous graphs
    print(f"[+] Loading cached graphs from {CACHE_FILE_10}...")
    with open(CACHE_FILE_10, "rb") as f:
        graphs = pickle.load(f)
        
    lookback = 20
    dates = target.index
    sorted_gdates = sorted(graphs.keys())
    
    # Vectorized / optimized sequence creation
    feat_vals = {tkr: features[tkr].values for tkr in tickers}
    target_vals = target.values.astype(np.float32)
    
    X_list, Y_list, pcmci_list, corr_list = [], [], [], []
    for t_idx in range(lookback, len(dates)):
        curr_d = dates[t_idx]
        frame_feats = [[feat_vals[tkr][t_idx - lag] for tkr in tickers] for lag in range(lookback, 0, -1)]
        X_list.append(np.array(frame_feats, dtype=np.float32))
        Y_list.append(target_vals[t_idx])
        cand = [gd for gd in sorted_gdates if gd <= curr_d]
        best_gd = cand[-1] if cand else sorted_gdates[0]
        pcmci_list.append(graphs[best_gd]['pcmci_cont'])
        corr_list.append(graphs[best_gd]['corr_cont'])
        
    X_arr = np.array(X_list)
    Y_arr = np.array(Y_list)
    pcmci_arr = np.array(pcmci_list)
    corr_arr = np.array(corr_list)
    
    # Splits (Purging & Embargo)
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
    print(f"[+] Device: {device}")
    
    # Experiments matrix
    configs = [
        ("Dense + MSE Loss", None, None, None, "mse", 0.0),
        ("Dense + Hybrid Sharpe Loss", None, None, None, "hybrid", 0.05),
        ("Correlation Gated + MSE Loss", M_corr_tr, M_corr_va, M_corr_te, "mse", 0.0),
        ("Correlation Gated + Hybrid Sharpe Loss", M_corr_tr, M_corr_va, M_corr_te, "hybrid", 0.05),
        ("Soft Causal Routed + MSE Loss", M_pcmci_tr, M_pcmci_va, M_pcmci_te, "mse", 0.0),
        ("Soft Causal Routed + Hybrid Sharpe Loss", M_pcmci_tr, M_pcmci_va, M_pcmci_te, "hybrid", 0.05),
    ]
    
    results = []
    
    for name, m_train, m_val, m_test, loss_type, lambda_sr in configs:
        print(f"\n---> Running: {name}...")
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = SoftCausalInvertedTransformer(len(tickers), lookback=20, num_features=F).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        if loss_type == "mse":
            crit = nn.MSELoss()
        else:
            crit = HybridSharpeLoss(lambda_sr=lambda_sr)
            
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
        
        for ep in range(1, 6):
            model.train()
            running_loss = 0.0
            for batch in loader:
                bx = batch[0].to(device)
                bm = batch[1].to(device) if m_train is not None else None
                by = batch[2].to(device)
                
                optimizer.zero_grad()
                out = model(bx, graph_weights=bm)
                
                if loss_type == "mse":
                    loss = crit(out, by)
                else:
                    loss, l_mse, l_sr = crit(out, by)
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                running_loss += loss.item() * bx.size(0)
                
            model.eval()
            with torch.no_grad():
                val_out = model(X_va_t, graph_weights=m_val_t)
                val_mse = nn.MSELoss()(val_out, Y_va_t).item()
                
            print(f"    Epoch {ep:02d} | Train Loss: {running_loss/len(X_tr):.5f} | Val MSE: {val_mse:.5f}")
            
        # OOS Evaluation
        model.eval()
        with torch.no_grad():
            preds_s = model(torch.tensor(X_te_s).to(device), graph_weights=m_test_t).cpu().numpy()
        preds = scaler_y.inverse_transform(preds_s)
        
        mae = np.mean(np.abs(Y_te - preds))
        mse_m = np.sum((Y_te - preds)**2)
        mse_b = np.sum((Y_te - y_tr_mean)**2)
        oos_r2 = (1.0 - (mse_m / mse_b)) * 100.0
        da = np.mean(np.sign(Y_te) == np.sign(preds)) * 100.0
        
        # Long-short Backtest (top 20% long, bottom 20% short)
        k = max(1, len(tickers) // 5)
        weights = np.zeros_like(preds)
        for t in range(len(preds)):
            rk = np.argsort(preds[t])
            weights[t, rk[-k:]] = 1.0 / (2.0 * k)
            weights[t, rk[:k]] = -1.0 / (2.0 * k)
            
        d_weights = np.sum(np.abs(weights[1:] - weights[:-1]), axis=1)
        turnover = np.mean(d_weights)
        gross_ret = np.sum(weights[1:] * Y_te[1:], axis=1)
        net_ret_5bps = gross_ret - (5.0 / 10000.0) * d_weights
        
        sr_gross = np.sqrt(252) * (np.mean(gross_ret) / (np.std(gross_ret, ddof=1) + 1e-8))
        sr_net5 = np.sqrt(252) * (np.mean(net_ret_5bps) / (np.std(net_ret_5bps, ddof=1) + 1e-8))
        
        results.append({
            "Configuration": name,
            "Loss": loss_type.upper(),
            "R2_OOS (%)": f"{oos_r2:.2f}%",
            "MAE": f"{mae:.4f}",
            "Dir Acc (%)": f"{da:.2f}%",
            "Turnover": f"{turnover:.2f}",
            "Gross Sharpe": f"{sr_gross:.3f}",
            "Net Sharpe (5 bps)": f"{sr_net5:.3f}"
        })
        
    df_res = pd.DataFrame(results).set_index("Configuration")
    print("\n" + "=" * 90)
    print("DIFFERENTIABLE SHARPE MINI-ABLATION RESULTS (5 EPOCHS, SEED 42):")
    print("=" * 90)
    print(df_res.to_string())
    
    out_csv = os.path.join(BASE_DIR, "mini_ablation_sharpe_loss_results.csv")
    df_res.to_csv(out_csv)
    print(f"\n[+] Saved results to {out_csv}")

if __name__ == "__main__":
    run_sharpe_ablation()
